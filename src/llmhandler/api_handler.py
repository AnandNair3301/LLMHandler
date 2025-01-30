# File: src/LLMHandler/api_handler.py
"""
Core module for handling LLM prompts and responses in LLMHandler.
"""

import os
import json
import asyncio
import traceback
from pathlib import Path
from typing import Any, List, Optional, Sequence, Type, Union

from datetime import datetime
from dotenv import load_dotenv
from loguru import logger
from aiolimiter import AsyncLimiter
from pydantic import BaseModel

# Load .env file if present, overriding existing environment variables
load_dotenv(override=True)

# pydantic_ai imports
from pydantic_ai import Agent
from pydantic_ai.models import Model
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.gemini import GeminiModel
from pydantic_ai.models.vertexai import VertexAIModel
from pydantic_ai.exceptions import UserError

# Local models
from .models import (
    BatchMetadata,
    BatchResult,
    UnifiedResponse,
    T,
)


class UnifiedLLMHandler:
    """
    A unified handler for processing single or multiple prompts with typed responses.

    This class supports:
      - Multiple providers via 'provider:model_name' notation (e.g., 'openai:gpt-4o-mini').
      - Batch mode (only for OpenAI models).
      - Optional rate limiting.

    Attributes:
        rate_limiter: An optional AsyncLimiter for request throttling.
        batch_output_dir: Directory for saving batch output JSONL files.
        openai_api_key: OpenAI API key (defaults to env var OPENAI_API_KEY).
        openrouter_api_key: OpenRouter API key (defaults to env var OPENROUTER_API_KEY).
        deepseek_api_key: DeepSeek API key (defaults to env var DEEPSEEK_API_KEY).
        anthropic_api_key: Anthropic API key (defaults to env var ANTHROPIC_API_KEY).
        gemini_api_key: Gemini API key (defaults to env var GEMINI_API_KEY).
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
    ) -> None:
        """
        Initializes a UnifiedLLMHandler instance.

        Args:
            requests_per_minute: Maximum number of requests per minute. If provided,
                requests are rate-limited.
            batch_output_dir: Directory for saving OpenAI batch output.
            openai_api_key: Override for the OpenAI API key.
            openrouter_api_key: Override for the OpenRouter API key.
            deepseek_api_key: Override for the DeepSeek API key.
            anthropic_api_key: Override for the Anthropic API key.
            gemini_api_key: Override for the Gemini API key.
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
        Parses the provider/model string and returns a pydantic_ai Model instance.

        Args:
            model_str: A string of the form 'provider:model_name', e.g. 'openai:gpt-4o-mini'.

        Returns:
            A Model instance configured with the correct provider and model name.

        Raises:
            UserError: If the model string is invalid or if required API keys are missing.
        """
        if ":" not in model_str:
            raise UserError(
                "Model string must have the form 'provider:model_name', "
                "e.g. 'openai:gpt-4o-mini', 'anthropic:claude-2', etc."
            )

        provider, real_model_name = model_str.split(":", 1)
        provider = provider.strip().lower()
        real_model_name = real_model_name.strip()

        if provider == "openai":
            if not self.openai_api_key:
                raise UserError("No OpenAI API key set.")
            return OpenAIModel(real_model_name, api_key=self.openai_api_key)

        if provider == "openrouter":
            if not self.openrouter_api_key:
                raise UserError("No OpenRouter API key set.")
            return OpenAIModel(
                real_model_name,
                base_url="https://openrouter.ai/api/v1",
                api_key=self.openrouter_api_key,
            )

        if provider == "deepseek":
            if not self.deepseek_api_key:
                raise UserError("No DeepSeek API key set.")
            return OpenAIModel(
                real_model_name,
                base_url="https://api.deepseek.com",
                api_key=self.deepseek_api_key,
            )

        if provider == "anthropic":
            if not self.anthropic_api_key:
                raise UserError("No Anthropic API key set.")
            return AnthropicModel(real_model_name, api_key=self.anthropic_api_key)

        if provider == "gemini":
            if not self.gemini_api_key:
                raise UserError("No Gemini API key set.")
            return GeminiModel(real_model_name, api_key=self.gemini_api_key)

        if provider == "vertexai":
            return VertexAIModel(real_model_name)

        raise UserError(
            f"Unrecognized provider prefix: {provider}. "
            "Must be one of: openai, openrouter, deepseek, anthropic, gemini, vertexai."
        )

    async def process(
        self,
        prompts: Union[str, List[str]],
        model: str,
        response_type: Type[BaseModel],
        *,
        system_message: Union[str, Sequence[str]] = (),
        batch_size: int = 1000,
        batch_mode: bool = False,
        retries: int = 1,
    ) -> UnifiedResponse[Any]:
        """
        Processes user prompts with a specified model, returning typed responses.

        Args:
            prompts: A single prompt or a list of prompts to process.
            model: 'provider:model_name' (e.g. 'openai:gpt-4o-mini').
            response_type: The Pydantic model class to parse responses into.
            system_message: Optional system message(s) to guide the model.
            batch_size: Number of prompts to process concurrently (for multiple prompts).
            batch_mode: If True, uses the OpenAI batch API (only for 'openai:' models).
            retries: How many times to retry if the model call fails.

        Returns:
            A UnifiedResponse containing either a single typed response,
            multiple typed responses, or a BatchResult for batch mode.

        Raises:
            UserError: If prompts are invalid or if batch_mode is used incorrectly.
            Exception: For any unexpected errors.
        """
        logger.debug(
            "Starting process() with model='{}', batch_mode={}, number_of_prompts={}",
            model,
            batch_mode,
            len(prompts) if isinstance(prompts, list) else 1
        )

        original_prompt_for_error: Optional[str] = None
        if isinstance(prompts, str):
            original_prompt_for_error = prompts
        elif isinstance(prompts, list) and prompts:
            original_prompt_for_error = prompts[0]

        try:
            if not prompts:
                raise UserError("Prompts cannot be empty or None.")

            model_instance = self._build_model_instance(model)
            agent = Agent(
                model_instance,
                result_type=response_type,
                system_prompt=system_message,
                retries=retries,
            )

            if batch_mode:
                # Only openai: supports batch API
                if not isinstance(model_instance, OpenAIModel):
                    raise UserError("Batch API mode is only supported for openai: models.")
                batch_result = await self._process_batch(agent, prompts, response_type)
                logger.debug("Batch mode processing completed successfully.")
                return UnifiedResponse(success=True, data=batch_result)

            if isinstance(prompts, str):
                single_res = await self._process_single(agent, prompts)
                logger.debug("Single prompt processed successfully.")
                return UnifiedResponse(success=True, data=single_res)

            # Otherwise, prompts is a list
            multi_res = await self._process_multiple(agent, prompts, batch_size)
            logger.debug("Multiple prompts processed successfully. results_count={}", len(multi_res))
            return UnifiedResponse(success=True, data=multi_res)

        except UserError as exc:
            full_trace = traceback.format_exc()
            error_msg = f"UserError: {exc}\nFull Traceback:\n{full_trace}"
            logger.error(error_msg)
            return UnifiedResponse(
                success=False,
                error=error_msg,
                original_prompt=original_prompt_for_error,
            )
        except Exception as exc:
            full_trace = traceback.format_exc()
            error_msg = f"Unexpected error: {exc}\nFull Traceback:\n{full_trace}"
            logger.exception(error_msg)
            return UnifiedResponse(
                success=False,
                error=error_msg,
                original_prompt=original_prompt_for_error,
            )

    async def _process_single(self, agent: Agent, prompt: str) -> BaseModel:
        """
        Processes a single prompt with optional rate limiting.

        Args:
            agent: The configured Agent instance.
            prompt: A single prompt to send to the model.

        Returns:
            The typed response from the model.
        """
        logger.debug("Processing single prompt: {}", prompt)
        if self.rate_limiter:
            async with self.rate_limiter:
                res = await agent.run(prompt)
        else:
            res = await agent.run(prompt)
        return res.data

    async def _process_multiple(
        self,
        agent: Agent,
        prompts: List[str],
        batch_size: int
    ) -> List[BaseModel]:
        """
        Processes multiple prompts concurrently, in batches of size `batch_size`.

        Args:
            agent: The configured Agent instance.
            prompts: A list of user prompts.
            batch_size: The number of prompts to process in each chunk.

        Returns:
            A list of typed responses, one per prompt.
        """
        logger.debug("Processing multiple prompts in batches of {}", batch_size)
        results: List[BaseModel] = []
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i : i + batch_size]

            async def process_prompt(p: str) -> BaseModel:
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
        self,
        agent: Agent,
        prompts: List[str],
        response_type: Type[BaseModel]
    ) -> BatchResult:
        """
        Executes the OpenAI batch API workflow for multiple prompts.

        Args:
            agent: The configured Agent instance (OpenAIModel).
            prompts: A list of user prompts to be processed via batch API.
            response_type: The Pydantic model to parse batch responses into.

        Returns:
            A BatchResult object containing batch metadata and responses.

        Raises:
            Exception: If the batch fails unexpectedly.
        """
        logger.debug("Initiating batch processing for {} prompts.", len(prompts))
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        batch_file = self.batch_output_dir / f"batch_{timestamp}.jsonl"

        # Write prompts to JSONL
        with batch_file.open("w", encoding="utf-8") as file_obj:
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
                file_obj.write(json.dumps(request_data) + "\n")

        # Upload and create the batch job
        logger.debug("Uploading batch file '{}' to OpenAI.", batch_file)
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

        # Poll until batch completes or fails
        while True:
            status = await agent.model.client.batches.retrieve(batch.id)
            metadata.status = status.status
            metadata.last_updated = datetime.now()

            if status.status == "completed":
                logger.debug("Batch '{}' completed successfully.", batch.id)
                break
            if status.status in ["failed", "canceled"]:
                logger.error("Batch failed or canceled with status: {}", status.status)
                metadata.error = f"Batch failed with status: {status.status}"
                return BatchResult(metadata=metadata, results=[])

            await asyncio.sleep(10)

        # Download results
        output_file = self.batch_output_dir / f"batch_{batch.id}_results.jsonl"
        result_content = await agent.model.client.files.content(status.output_file_id)
        with output_file.open("wb") as out_f:
            out_f.write(result_content.content)
        metadata.output_file_path = str(output_file)

        logger.debug("Reading batch results from '{}'.", output_file)
        results_list = []
        with output_file.open("r", encoding="utf-8") as res_f:
            for line, prompt_text in zip(res_f, prompts):
                data = json.loads(line)
                try:
                    content = data["response"]["body"]["choices"][0]["message"]["content"]
                    resp_obj = response_type.construct()
                    # Fill "content" if present in the response model
                    if hasattr(resp_obj, "content"):
                        setattr(resp_obj, "content", content)
                    # Fill "confidence" if present in the response model
                    if hasattr(resp_obj, "confidence"):
                        setattr(resp_obj, "confidence", 0.95)
                    results_list.append({"prompt": prompt_text, "response": resp_obj})
                except Exception as exc:
                    full_trace = traceback.format_exc()
                    error_msg = f"Unexpected error: {exc}\nFull Traceback:\n{full_trace}"
                    logger.exception(error_msg)
                    results_list.append({"prompt": prompt_text, "error": error_msg})

        logger.debug("Batch processing complete with {} results.", len(results_list))
        return BatchResult(metadata=metadata, results=results_list)
