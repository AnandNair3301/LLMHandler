# File: tests/test_api_handler.py
"""
Tests for the UnifiedLLMHandler class in LLMHandler.
"""

import pytest
import asyncio
from unittest.mock import patch, MagicMock
from typing import List

import pytest_asyncio

from LLMHandler.api_handler import UnifiedLLMHandler
from LLMHandler.models import SimpleResponse, MathResponse, BatchResult
from pydantic_ai.exceptions import UserError


@pytest.mark.asyncio
async def test_single_prompt_basic() -> None:
    """Test processing a single prompt with a mocked run()."""
    handler = UnifiedLLMHandler(openai_api_key="fake_key")

    # Patch Agent.run so we don't make real external calls
    with patch("LLMHandler.api_handler.Agent.run") as mock_run:
        mock_run.return_value.data = SimpleResponse(content="Hello from mock")
        result = await handler.process(
            prompts="Hello world",
            model="openai:gpt-4o-mini",
            response_type=SimpleResponse
        )

    assert result.success is True
    assert isinstance(result.data, SimpleResponse)
    assert result.data.content == "Hello from mock"


@pytest.mark.asyncio
async def test_multiple_prompts_basic() -> None:
    """Test processing multiple prompts with a mocked run()."""
    handler = UnifiedLLMHandler(openai_api_key="fake_key")

    fake_responses = [
        SimpleResponse(content="Resp1"),
        SimpleResponse(content="Resp2"),
        SimpleResponse(content="Resp3"),
    ]

    async def async_run_side_effect(prompt: str):
        # Attempt to parse last char as index
        idx = int(prompt[-1]) if prompt and prompt[-1].isdigit() else 0
        fake = fake_responses[idx] if idx < len(fake_responses) else SimpleResponse(content="N/A")
        mock = MagicMock()
        mock.data = fake
        return mock

    with patch("LLMHandler.api_handler.Agent.run", side_effect=async_run_side_effect):
        prompts = ["Prompt0", "Prompt1", "Prompt2"]
        result = await handler.process(
            prompts=prompts,
            model="openai:gpt-4o",
            response_type=SimpleResponse
        )

    assert result.success is True
    assert isinstance(result.data, list)
    assert len(result.data) == 3
    assert all(isinstance(r, SimpleResponse) for r in result.data)
    assert result.data[0].content == "Resp1"


@pytest.mark.asyncio
async def test_invalid_provider() -> None:
    """Ensure that an invalid model string raises a UserError."""
    handler = UnifiedLLMHandler()
    with pytest.raises(UserError):
        # Missing 'provider:' => should raise an error
        await handler.process(
            prompts="Test prompt",
            model="gpt-4o",
            response_type=SimpleResponse
        )


@pytest.mark.asyncio
async def test_empty_prompts() -> None:
    """Ensure empty prompts raise a UserError."""
    handler = UnifiedLLMHandler()
    with pytest.raises(UserError):
        await handler.process(
            prompts="",
            model="openai:gpt-4o-mini",
            response_type=SimpleResponse
        )


@pytest.mark.asyncio
async def test_batch_mode_non_openai() -> None:
    """Batch mode with a non-OpenAI provider should raise UserError."""
    handler = UnifiedLLMHandler(gemini_api_key="fake_gemini_key")
    with pytest.raises(UserError):
        await handler.process(
            prompts=["Prompt A", "Prompt B"],
            model="gemini:gemini-1.5-flash",
            response_type=SimpleResponse,
            batch_mode=True,
        )


@pytest.mark.asyncio
async def test_batch_mode_mocked() -> None:
    """Test batch mode success path with mocked file upload and results."""
    handler = UnifiedLLMHandler(openai_api_key="fake_key")

    # We'll mock out the entire batch workflow.
    with patch("LLMHandler.api_handler.Agent.model") as mock_model:
        # Create a fake client with the needed methods
        fake_client = MagicMock()
        fake_client.files.create.return_value.id = "fake_file_id"
        fake_client.batches.create.return_value.id = "fake_batch_id"
        fake_client.batches.retrieve.side_effect = [
            MagicMock(status="in_progress"),
            MagicMock(status="completed")
        ]
        fake_client.files.content.return_value.content = b'{"response":{"body":{"choices":[{"message":{"content":"Batch result A"}}]}}}'

        mock_model.client = fake_client

        result = await handler.process(
            prompts=["BatchPromptA"],
            model="openai:gpt-4o-mini",
            response_type=SimpleResponse,
            batch_mode=True
        )

    assert result.success is True
    assert isinstance(result.data, BatchResult)
    assert result.data.results[0]["response"].content == "Batch result A"
