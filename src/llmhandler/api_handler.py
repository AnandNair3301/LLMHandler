import os
import json
import asyncio
import traceback
from datetime import datetime
from pathlib import Path
from typing import (
    Dict,
    List,
    Optional,
    Sequence,
    Type,
    TypeVar,
    Union,
    Generic,
)

from dotenv import load_dotenv

# Load .env (override any pre-existing environment variables with what's in .env)
load_dotenv(override=True)

import logfire
from aiolimiter import AsyncLimiter
from pydantic import BaseModel, Field

# Core Pydantic AI
from pydantic_ai import Agent
from pydantic_ai.models import Model
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.gemini import GeminiModel
from pydantic_ai.models.vertexai import VertexAIModel
from pydantic_ai.exceptions import UserError

# Import your shared data models from models.py
from .models import (
    BatchMetadata,
    BatchResult,
    SimpleResponse,
    MathResponse,
    PersonResponse,
    UnifiedResponse,
)

# Configure logfire (optional)
logfire.configure(send_to_logfire="if-token-present")

T = TypeVar("T", bound=BaseModel)


def _json_schema_instructions(response_type: Type[BaseModel]) -> str:
    """
    Generate a short system message instructing the model to reply with
    valid JSON that matches the given Pydantic model schema.
    """
    schema_str = response_type.model_json_schema()
    return (
        "Please respond exclusively in valid JSON that matches this schema:\n"
        f"{schema_str}\n\n"
        "Do not include extra keys or text outside the JSON."
    )


class UnifiedLLMHandler:
    """
    A unified handler for processing single or multiple prompts with typed responses,
    optional batch mode, and multiple LLM providers.

    The user MUST pass provider: in the model string (e.g. 'openai:gpt-4'),
    or we raise an error. This eliminates ambiguous model name guessing.
    """

    def __init__(
        self,
        requests_per_minute: Optional[int] = None,
        batch_output_dir: str = "batch_output",
        openai_api_key: Optional[str] = None,
        openrouter_api_key: Optional[str] = None,
        deepseek_api_key: Optional[str] = None,
        anthropic_api_key: Optional[str] = None,
        gemini_api_key: Optional[str] = None,
    ):
        """
        :param requests_per_minute: If specified, uses an AsyncLimiter to throttle requests.
        :param batch_output_dir: Directory for saving batch output JSONL.
        :param openai_api_key: OpenAI API key override (falls back to OPENAI_API_KEY if None).
        :param openrouter_api_key: OpenRouter API key override (falls back to OPENROUTER_API_KEY).
        :param deepseek_api_key: DeepSeek API key override (falls back to DEEPSEEK_API_KEY).
        :param anthropic_api_key: Anthropic API key override (falls back to ANTHROPIC_API_KEY).
        :param gemini_api_key: Gemini API key override (falls back to GEMINI_API_KEY).
        """
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.openrouter_api_key = openrouter_api_key or os.getenv("OPENROUTER_API_KEY")
        self.deepseek_api_key = deepseek_api_key or os.getenv("DEEPSEEK_API_KEY")
        self.anthropic_api_key = anthropic_api_key or os.getenv("ANTHROPIC_API_KEY")
        self.gemini_api_key = gemini_api_key or os.getenv("GEMINI_API_KEY")

        self.rate_limiter = (
            AsyncLimiter(requests_per_minute, 60) if requests_per_minute else None
        )
        self.batch_output_dir = Path(batch_output_dir)
        self.batch_output_dir.mkdir(parents=True, exist_ok=True)

    def _build_model_instance(self, model_str: str) -> Model:
        """
        If model_str is not prefixed with a recognized provider, raise an error.
        Otherwise, build and return the correct pydantic_ai model instance.
        """
        if ":" not in model_str:
            raise UserError(
                "Model string must start with a recognized prefix, "
                "e.g. 'openai:gpt-4', 'openrouter:some_model', "
                "'deepseek:deepseek-chat', 'anthropic:some_claude', etc."
            )

        provider, real_model_name = model_str.split(":", 1)
        provider = provider.strip().lower()
        real_model_name = real_model_name.strip()

        if provider == "openai":
            if not self.openai_api_key:
                raise UserError("No OpenAI API key set. Provide openai_api_key= or set OPENAI_API_KEY.")
            return OpenAIModel(real_model_name, api_key=self.openai_api_key)

        elif provider == "openrouter":
            if not self.openrouter_api_key:
                raise UserError("No OpenRouter API key set.")
            return OpenAIModel(
                real_model_name,
                base_url="https://openrouter.ai/api/v1",
                api_key=self.openrouter_api_key,
            )

        elif provider == "deepseek":
            if not self.deepseek_api_key:
                raise UserError("No DeepSeek API key set.")
            return OpenAIModel(
                real_model_name,
                base_url="https://api.deepseek.com",
                api_key=self.deepseek_api_key,
            )

        elif provider == "anthropic":
            if not self.anthropic_api_key:
                raise UserError("No Anthropic API key set.")
            return AnthropicModel(real_model_name, api_key=self.anthropic_api_key)

        elif provider == "gemini":
            if not self.gemini_api_key:
                raise UserError("No Gemini API key set.")
            return GeminiModel(real_model_name, api_key=self.gemini_api_key)

        elif provider == "vertexai":
            return VertexAIModel(real_model_name)

        else:
            raise UserError(
                f"Unrecognized provider prefix: {provider}. "
                f"Must be one of: openai, openrouter, deepseek, anthropic, gemini, vertexai."
            )

    async def process(
        self,
        prompts: Union[str, List[str]],
        model: str,
        response_type: Type[T],
        *,
        system_message: Union[str, Sequence[str]] = (),
        batch_size: int = 1000,
        batch_mode: bool = False,
        retries: int = 1,
    ) -> UnifiedResponse[Union[T, List[T], BatchResult]]:
        """
        Main entry point for processing user prompts with typed responses.

        :param prompts: The prompt or list of prompts.
        :param model: Must be "provider:model_name", e.g. "openai:gpt-4".
        :param response_type: A pydantic BaseModel for typed responses.
        :param system_message: Optional system message(s) to guide the model.
        :param batch_size: If multiple prompts are provided, process them in chunks.
        :param batch_mode: If True, uses the OpenAI batch API (only for openai: models).
        :param retries: Number of times to retry on certain exceptions.
        """
        with logfire.span("llm_processing"):
            logfire.log(
                "debug",
                "Starting process() with LLM prompts",
                attributes={
                    "model": model,
                    "batch_mode": batch_mode,
                    "prompt_count": len(prompts) if isinstance(prompts, list) else 1,
                },
            )

            # Keep track of the first prompt for error reporting
            original_prompt_for_error: Optional[str] = None
            if isinstance(prompts, str):
                original_prompt_for_error = prompts
            elif isinstance(prompts, list) and prompts:
                original_prompt_for_error = prompts[0]

            try:
                if prompts is None:
                    raise UserError("Prompts cannot be None.")
                if isinstance(prompts, str) and not prompts.strip():
                    raise UserError("Prompt cannot be an empty string.")
                if isinstance(prompts, list) and len(prompts) == 0:
                    raise UserError("Prompts list cannot be empty.")

                # 1) Auto-generate JSON schema instructions and append to system_message
                schema_instructions = _json_schema_instructions(response_type)
                if isinstance(system_message, str):
                    system_message = [system_message, schema_instructions]
                else:
                    system_message = list(system_message)
                    system_message.append(schema_instructions)

                model_instance = self._build_model_instance(model)
                agent = Agent(
                    model_instance,
                    result_type=response_type,
                    system_prompt=system_message,
                    retries=retries,
                )

                if batch_mode:
                    # Only openai:* models support the batch API
                    if not isinstance(model_instance, OpenAIModel):
                        raise UserError("Batch API mode is only supported for openai:* models.")
                    batch_result = await self._process_batch(agent, prompts, response_type)
                    return UnifiedResponse(success=True, data=batch_result)

                if isinstance(prompts, str):
                    data = await self._process_single(agent, prompts)
                    return UnifiedResponse(success=True, data=data)

                # Otherwise, prompts is a list
                data = await self._process_multiple(agent, prompts, batch_size)
                return UnifiedResponse(success=True, data=data)

            except UserError as exc:
                # Re-raise so tests with pytest.raises(UserError) succeed.
                logfire.log("error", "Caught user error", attributes={"error": str(exc)})
                raise
            except Exception as exc:
                # For any other exception, we return a UnifiedResponse with error
                full_trace = traceback.format_exc()
                error_msg = f"Unexpected error: {exc}\nFull Traceback:\n{full_trace}"
                with logfire.span("error_handling", error=str(exc), error_type="unexpected_error"):
                    logfire.log("error", "Caught unexpected error", attributes={"trace": error_msg})

                return UnifiedResponse(
                    success=False,
                    error=error_msg,
                    original_prompt=original_prompt_for_error,
                )

    async def _process_single(self, agent: Agent, prompt: str) -> T:
        """
        Process a single prompt with optional rate limiting.
        """
        with logfire.span("process_single"):
            logfire.log("debug", "Processing single prompt", attributes={"prompt": prompt})
            if self.rate_limiter:
                async with self.rate_limiter:
                    result = await agent.run(prompt)
            else:
                result = await agent.run(prompt)
            return result.data

    async def _process_multiple(
        self, agent: Agent, prompts: List[str], batch_size: int
    ) -> List[T]:
        """
        Process multiple prompts in chunks using asyncio.gather for concurrency.
        """
        with logfire.span("process_multiple"):
            logfire.log("debug", "Processing multiple prompts", attributes={"batch_size": batch_size})
            results: List[T] = []
            for i in range(0, len(prompts), batch_size):
                batch = prompts[i : i + batch_size]

                async def process_prompt(p: str) -> T:
                    if self.rate_limiter:
                        async with self.rate_limiter:
                            res = await agent.run(p)
                    else:
                        res = await agent.run(p)
                    return res.data

                batch_results = await asyncio.gather(*(process_prompt(p) for p in batch))
                results.extend(batch_results)
            return results

    async def _process_batch(
        self, agent: Agent, prompts: List[str], response_type: Type[T]
    ) -> BatchResult:
        """
        Specialized method for the OpenAI Batch API workflow.
        Writes JSONL, uploads to OpenAI, polls for completion,
        and returns a BatchResult with typed responses.
        """
        with logfire.span("process_batch"):
            logfire.log("debug", "Initiating batch processing", attributes={"num_prompts": len(prompts)})
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            batch_file = self.batch_output_dir / f"batch_{timestamp}.jsonl"

            with batch_file.open("w", encoding="utf-8") as f:
                for i, prompt in enumerate(prompts):
                    request_data = {
                        "custom_id": f"req_{i}",
                        "method": "POST",
                        "url": "/v1/chat/completions",
                        "body": {
                            "model": agent.model.model_name,
                            "messages": [{"role": "user", "content": prompt}],
                        },
                    }
                    f.write(json.dumps(request_data) + "\n")

            batch_upload = await agent.model.client.files.create(
                file=batch_file.open("rb"), purpose="batch"
            )
            batch = await agent.model.client.batches.create(
                input_file_id=batch_upload.id,
                endpoint="/v1/chat/completions",
                completion_window="24h",
            )

            metadata = BatchMetadata(
                batch_id=batch.id,
                input_file_id=batch_upload.id,
                status="in_progress",
                created_at=datetime.now(),
                last_updated=datetime.now(),
                num_requests=len(prompts),
            )

            while True:
                status = await agent.model.client.batches.retrieve(batch.id)
                metadata.status = status.status
                metadata.last_updated = datetime.now()

                if status.status == "completed":
                    logfire.log("debug", "Batch completed successfully", attributes={"batch_id": batch.id})
                    break
                elif status.status in ["failed", "canceled"]:
                    metadata.error = f"Batch failed with status: {status.status}"
                    logfire.log("error", "Batch ended in failure/canceled", attributes={"batch_id": batch.id, "status": status.status})
                    return BatchResult(metadata=metadata, results=[])

                await asyncio.sleep(10)

            output_file = self.batch_output_dir / f"batch_{batch.id}_results.jsonl"
            result_content = await agent.model.client.files.content(status.output_file_id)
            with output_file.open("wb") as out_f:
                out_f.write(result_content.content)
            metadata.output_file_path = str(output_file)

            results: List[Dict[str, Union[str, BaseModel]]] = []
            with output_file.open("r", encoding="utf-8") as f:
                for line, prompt_text in zip(f, prompts):
                    data = json.loads(line)
                    try:
                        content = data["response"]["body"]["choices"][0]["message"]["content"]
                        r = response_type.model_construct()
                        if "content" in response_type.model_fields:
                            setattr(r, "content", content)
                        if "confidence" in response_type.model_fields:
                            setattr(r, "confidence", 0.95)
                        results.append({"prompt": prompt_text, "response": r})
                    except Exception as e:
                        full_trace = traceback.format_exc()
                        error_msg = f"Unexpected error: {e}\nFull Traceback:\n{full_trace}"
                        results.append({"prompt": prompt_text, "error": error_msg})

            return BatchResult(metadata=metadata, results=results)
