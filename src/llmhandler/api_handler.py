import os
import json
import asyncio
import traceback
import inspect
from datetime import datetime
from pathlib import Path
from typing import (
    Any,
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
from ._internal_models import (
    BatchMetadata,
    BatchResult,
    SimpleResponse,
    MathResponse,
    PersonResponse,
    UnifiedResponse,
)


# Configure logfire (optional)
logfire.configure(send_to_logfire="if-token-present")

# T is now unconstrained so that if no response type is provided, we treat it as unstructured.
T = TypeVar("T")

def _build_schema_instructions(response_type: Type[BaseModel]) -> str:
    """
    Generate system-prompt instructions telling the model
    to respond exclusively in valid JSON that matches this schema.
    """
    schema_str = response_type.model_json_schema()
    return (
        "Please respond exclusively in valid JSON that matches this schema:\n"
        f"{schema_str}\n\n"
        "Do not include extraneous keys. Return ONLY valid JSON."
    )

class PromptResult(BaseModel):
    """
    Stores the outcome of processing one prompt.
      - prompt: the original input prompt
      - data: the LLM response (typed or raw string) if successful
      - error: an error string if something failed
    """
    prompt: str
    data: Optional[Union[str, BaseModel]] = None
    error: Optional[str] = None


class UnifiedLLMHandler:
    """
    A unified handler for processing prompts with typed responses (when a Pydantic model is provided)
    or unstructured responses (when no model is provided).
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
        Otherwise, return the appropriate pydantic_ai model instance.
        """
        if ":" not in model_str:
            raise UserError(
                "Model string must start with a recognized prefix, e.g. 'openai:gpt-4'."
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
                f"Unrecognized provider prefix: {provider}. Must be one of: openai, openrouter, deepseek, anthropic, gemini, vertexai."
            )

    async def process(
        self,
        prompts: Union[str, List[str]],
        model: str,
        response_type: Optional[Type[T]] = None,
        *,
        system_message: Union[str, Sequence[str]] = (),
        batch_size: int = 1000,
        batch_mode: bool = False,
        retries: int = 1,
    ) -> Union[
        UnifiedResponse[Union[T, List[T], BatchResult, List[PromptResult]]],
        str,
        List[str],
        List[PromptResult]
    ]:
        """
        Process a prompt (or prompts) and return either:
         - A UnifiedResponse (with typed output) if `response_type` is a Pydantic model,
         - Or raw text (str or list[str]) if no Pydantic model is provided,
         - Or a list of PromptResult if you have multiple prompts in partial-failure mode.

        If a Pydantic model is provided, JSON schema instructions are appended to the system prompt.
        Batch mode is allowed only when a typed model is used.
        """
        with logfire.span("llm_processing"):
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

                model_instance = self._build_model_instance(model)

                # Check if we have a Pydantic model for typed responses.
                is_typed = (
                    response_type is not None
                    and inspect.isclass(response_type)
                    and issubclass(response_type, BaseModel)
                )

                # If typed, inject schema instructions into the system prompt.
                if is_typed:
                    schema_instructions = _build_schema_instructions(response_type)
                    if isinstance(system_message, str):
                        system_message = [system_message, schema_instructions]
                    else:
                        system_message = list(system_message)
                        system_message.append(schema_instructions)
                    agent = Agent(
                        model_instance,
                        result_type=response_type,
                        system_prompt=system_message,
                        retries=retries,
                    )
                else:
                    # No typed output: do not inject schema instructions.
                    agent = Agent(
                        model_instance,
                        system_prompt=system_message,
                        retries=retries,
                    )

                # Batch mode is only valid for typed, openai-based usage
                if batch_mode:
                    if not isinstance(model_instance, OpenAIModel):
                        raise UserError("Batch API mode is only supported for openai:* models.")
                    if not is_typed:
                        raise UserError("Batch mode requires a Pydantic model for typed results.")
                    batch_result = await self._process_batch(agent, prompts, response_type)
                    return UnifiedResponse(success=True, data=batch_result)

                # SINGLE PROMPT
                if isinstance(prompts, str):
                    result = await self._process_single(agent, prompts)
                    if is_typed:
                        return UnifiedResponse(success=True, data=result)
                    else:
                        return result

                # MULTIPLE PROMPTS (partial-failure approach)
                else:
                    multi_results = await self._process_multiple(agent, prompts, batch_size)
                    if is_typed:
                        # Return them in a UnifiedResponse so user sees partial successes/failures
                        return UnifiedResponse(success=True, data=multi_results)
                    else:
                        # Return the list of PromptResult unwrapped
                        return multi_results

            except UserError as exc:
                raise
            except Exception as exc:
                full_trace = traceback.format_exc()
                error_msg = f"Unexpected error: {exc}\nFull Traceback:\n{full_trace}"
                return UnifiedResponse(
                    success=False,
                    error=error_msg,
                    original_prompt=original_prompt_for_error,
                )

    async def _process_single(self, agent: Agent, prompt: str) -> Any:
        with logfire.span("process_single"):
            if self.rate_limiter:
                async with self.rate_limiter:
                    run_result = await agent.run(prompt)
            else:
                run_result = await agent.run(prompt)
        return run_result.data

    async def _process_multiple(
        self, agent: Agent, prompts: List[str], batch_size: int
    ) -> List[PromptResult]:
        """
        Updated to allow partial success. Each prompt yields a PromptResult
        with either data or error, so one failure won't spoil the entire set.
        """
        results: List[PromptResult] = []
        with logfire.span("process_multiple"):
            for i in range(0, len(prompts), batch_size):
                batch = prompts[i : i + batch_size]

                async def process_prompt(p: str) -> Any:
                    # We do agent.run(p) and return the .data
                    if self.rate_limiter:
                        async with self.rate_limiter:
                            r = await agent.run(p)
                    else:
                        r = await agent.run(p)
                    return r.data

                # return_exceptions=True prevents "fail fast" if any prompt raises
                chunk_results = await asyncio.gather(
                    *(process_prompt(p) for p in batch),
                    return_exceptions=True
                )
                for prompt_text, item in zip(batch, chunk_results):
                    if isinstance(item, Exception):
                        # Store the error
                        results.append(PromptResult(prompt=prompt_text, error=str(item)))
                    else:
                        # Store the successful data
                        results.append(PromptResult(prompt=prompt_text, data=item))
        return results

    async def _process_batch(
        self,
        agent: Agent,
        prompts: List[str],
        response_type: Type[BaseModel],
    ) -> BatchResult:
        """
        The existing batch-mode approach for large-scale async calls
        using the OpenAI 'files' & 'batches' endpoints.
        """
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
                break
            elif status.status in ["failed", "canceled"]:
                metadata.error = f"Batch failed with status: {status.status}"
                return BatchResult(metadata=metadata, results=[])
            await asyncio.sleep(10)

        output_file = self.batch_output_dir / f"batch_{batch.id}_results.jsonl"
        result_content = await agent.model.client.files.content(status.output_file_id)
        with output_file.open("wb") as out_f:
            out_f.write(result_content.content)
        metadata.output_file_path = str(output_file)

        results: List[Dict[str, Any]] = []
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
                    error_msg = f"Unexpected error: {e}\nFull Trace:\n{traceback.format_exc()}"
                    results.append({"prompt": prompt_text, "error": error_msg})
        return BatchResult(metadata=metadata, results=results)
