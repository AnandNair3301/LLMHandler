"""
Tests for the UnifiedLLMHandler class in llmhandler.
"""

import pytest
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock
from typing import List

from llmhandler.api_handler import UnifiedLLMHandler
from llmhandler.models import SimpleResponse, MathResponse, BatchResult
from pydantic_ai.exceptions import UserError


@pytest.mark.asyncio
async def test_single_prompt_basic():
    """
    Test processing a single prompt with a mocked run().
    Ensures that we get a typed response back.
    """
    handler = UnifiedLLMHandler(openai_api_key="fake_openai_key")

    # Patch Agent.run so we don't make a real external call
    with patch("llmhandler.api_handler.Agent.run") as mock_run:
        mock_run.return_value.data = SimpleResponse(content="Hello from mock")
        result = await handler.process(
            prompts="Hello world",
            model="openai:gpt-4o",  # allowed
            response_type=SimpleResponse
        )
    assert result.success is True
    assert isinstance(result.data, SimpleResponse)
    assert result.data.content == "Hello from mock"


@pytest.mark.asyncio
async def test_multiple_prompts_basic():
    """
    Test processing multiple prompts with a mocked run().
    Ensures we get multiple typed responses in a list.
    """
    handler = UnifiedLLMHandler(openai_api_key="fake_openai_key")

    fake_responses = [
        SimpleResponse(content="Resp A"),
        SimpleResponse(content="Resp B"),
        SimpleResponse(content="Resp C"),
    ]

    async def async_run_side_effect(prompt: str):
        idx = 0
        if prompt.endswith("B"):
            idx = 1
        elif prompt.endswith("C"):
            idx = 2
        mock_obj = MagicMock()
        mock_obj.data = fake_responses[idx]
        return mock_obj

    with patch("llmhandler.api_handler.Agent.run", side_effect=async_run_side_effect):
        prompts = ["Prompt A", "Prompt B", "Prompt C"]
        result = await handler.process(
            prompts=prompts,
            model="openai:gpt-4o-mini",  # allowed
            response_type=SimpleResponse
        )

    assert result.success is True
    assert isinstance(result.data, list)
    assert len(result.data) == 3
    for resp, expected in zip(result.data, ["Resp A", "Resp B", "Resp C"]):
        assert isinstance(resp, SimpleResponse)
        assert resp.content == expected


@pytest.mark.asyncio
async def test_invalid_provider():
    """
    Ensure that an invalid (missing prefix) model string raises a UserError.
    """
    handler = UnifiedLLMHandler()
    with pytest.raises(UserError):
        await handler.process(
            prompts="Test prompt",
            model="gpt-4o-mini",  # missing "provider:" => should raise
            response_type=SimpleResponse
        )


@pytest.mark.asyncio
async def test_empty_prompts():
    """
    Ensure empty string prompt raises a UserError.
    """
    handler = UnifiedLLMHandler()
    with pytest.raises(UserError):
        await handler.process(
            prompts="",
            model="openai:gpt-4o",  # allowed
            response_type=SimpleResponse
        )


@pytest.mark.asyncio
async def test_batch_mode_non_openai():
    """
    Batch mode with a non-OpenAI provider should raise UserError.
    """
    handler = UnifiedLLMHandler(gemini_api_key="fake_gemini_key")
    with pytest.raises(UserError):
        await handler.process(
            prompts=["Prompt A", "Prompt B"],
            model="gemini:gemini-1.5-flash-8b",
            response_type=SimpleResponse,
            batch_mode=True
        )


@pytest.mark.asyncio
async def test_batch_mode_mocked():
    """
    Test batch mode success path with mocked file upload and results.
    This requires AsyncMock for methods that are awaited inside the code.
    """
    handler = UnifiedLLMHandler(openai_api_key="fake_openai_key")

    # We'll mock out the entire batch workflow on the underlying agent.model
    # so that no real network calls occur.
    with patch("llmhandler.api_handler.Agent") as mock_agent_cls:
        mock_agent_inst = MagicMock()
        
        # We must replace the calls that get awaited with AsyncMock
        mock_model_client = MagicMock()
        mock_model_client.files.create = AsyncMock()
        mock_model_client.batches.create = AsyncMock()
        mock_model_client.batches.retrieve = AsyncMock()
        mock_model_client.files.content = AsyncMock()

        # Setup the return values
        mock_model_client.files.create.return_value.id = "fake_file_id"
        mock_model_client.batches.create.return_value.id = "fake_batch_id"
        # In-progress once, then completed
        mock_model_client.batches.retrieve.side_effect = [
            MagicMock(status="in_progress", output_file_id="fake_out_file"),
            MagicMock(status="completed", output_file_id="fake_out_file")
        ]
        mock_model_client.files.content.return_value.content = (
            b'{"response":{"body":{"choices":[{"message":{"content":"Batch result A"}}]}}}\n'
            b'{"response":{"body":{"choices":[{"message":{"content":"Batch result B"}}]}}}\n'
        )

        # Let the mock Agent instance have the needed .model and .run
        mock_agent_inst.model.model_name = "gpt-4o"
        mock_agent_inst.model.client = mock_model_client
        mock_agent_cls.return_value = mock_agent_inst

        # Now run the code under test
        result = await handler.process(
            prompts=["BatchPromptA", "BatchPromptB"],
            model="openai:gpt-4o",
            response_type=SimpleResponse,
            batch_mode=True,
        )

    assert result.success is True, f"Expected success, got {result.error}"
    assert isinstance(result.data, BatchResult)
    assert len(result.data.results) == 2
    assert result.data.results[0]["response"].content == "Batch result A"
    assert result.data.results[1]["response"].content == "Batch result B"
