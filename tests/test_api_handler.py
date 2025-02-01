"""
Tests for the UnifiedLLMHandler class in llmhandler.
This file now covers:
  - Structured (typed) responses (single and multiple prompts)
  - Unstructured (free-text) responses when response_type is omitted or None
  - Provider-specific tests for Anthropic, DeepSeek, Gemini, and OpenRouter
  - Batch mode tests for structured responses
  - Partial-failure tests for multiple prompts: one prompt fails while the others succeed.
"""

import pytest
import asyncio
import inspect
from unittest.mock import patch, MagicMock, AsyncMock
from typing import List

from llmhandler.api_handler import UnifiedLLMHandler
from llmhandler._internal_models import (
    BatchMetadata,
    BatchResult,
    SimpleResponse,
    MathResponse,
    PersonResponse,
    UnifiedResponse,
)
from pydantic_ai.exceptions import UserError


# --- Structured tests ---

@pytest.mark.asyncio
async def test_single_prompt_structured():
    """Test a single prompt with a typed response."""
    handler = UnifiedLLMHandler(openai_api_key="fake_openai_key")
    with patch("llmhandler.api_handler.Agent.run") as mock_run:
        # Return a valid Pydantic object
        fake_result = MagicMock()
        fake_result.data = SimpleResponse(content="Hello structured", confidence=0.95)
        mock_run.return_value = fake_result
        result = await handler.process(
            prompts="Hello world structured",
            model="openai:gpt-4o",
            response_type=SimpleResponse
        )
    assert result.success is True
    assert isinstance(result.data, SimpleResponse)
    assert result.data.content == "Hello structured"


@pytest.mark.asyncio
async def test_multiple_prompts_structured():
    """Test multiple prompts with a typed response."""
    handler = UnifiedLLMHandler(openai_api_key="fake_openai_key")
    responses = [
        SimpleResponse(content="Resp A", confidence=0.9),
        SimpleResponse(content="Resp B", confidence=0.9),
    ]
    async def side_effect(prompt: str):
        fake = MagicMock()
        idx = 0 if "A" in prompt else 1
        fake.data = responses[idx]
        return fake
    with patch("llmhandler.api_handler.Agent.run", side_effect=side_effect):
        prompts = ["Prompt A", "Prompt B"]
        result = await handler.process(
            prompts=prompts,
            model="openai:gpt-4o-mini",
            response_type=SimpleResponse
        )
    assert result.success is True
    # When multiple prompts are used in typed mode, the data is wrapped in a UnifiedResponse.
    # In this case our design returns a list of per-prompt results.
    assert isinstance(result.data, list)
    assert len(result.data) == 2
    for resp, expected in zip(result.data, ["Resp A", "Resp B"]):
        # Each item should have a non-null data field.
        assert resp.data is not None
        assert resp.data.content == expected


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


# --- Unstructured tests (free-text) ---

@pytest.mark.asyncio
async def test_single_prompt_unstructured_explicit_none():
    """
    Test processing a single prompt with unstructured response
    when response_type is explicitly set to None.
    """
    handler = UnifiedLLMHandler(openai_api_key="fake_openai_key")
    with patch("llmhandler.api_handler.Agent.run") as mock_run:
        fake = MagicMock()
        fake.data = "Free text response"
        mock_run.return_value = fake
        result = await handler.process(
            prompts="What is your favorite color?",
            model="openai:gpt-4o-mini",
            response_type=None
        )
    assert isinstance(result, str)
    assert "Free text" in result


@pytest.mark.asyncio
async def test_single_prompt_unstructured_omitted():
    """
    Test processing a single prompt with unstructured response
    when response_type is omitted.
    """
    handler = UnifiedLLMHandler(openai_api_key="fake_openai_key")
    with patch("llmhandler.api_handler.Agent.run") as mock_run:
        fake = MagicMock()
        fake.data = "Raw free text output"
        mock_run.return_value = fake
        result = await handler.process(
            prompts="Tell me a fun fact.",
            model="openai:gpt-4o-mini"
            # response_type omitted → unstructured output
        )
    assert isinstance(result, str)
    assert "free text" in result.lower()


@pytest.mark.asyncio
async def test_multiple_prompts_unstructured():
    """
    Test processing multiple prompts with unstructured responses.
    """
    handler = UnifiedLLMHandler(openai_api_key="fake_openai_key")
    async def side_effect(prompt: str):
        fake = MagicMock()
        fake.data = f"Response for {prompt}"
        return fake
    with patch("llmhandler.api_handler.Agent.run", side_effect=side_effect):
        prompts = ["Prompt one", "Prompt two"]
        result = await handler.process(
            prompts=prompts,
            model="openai:gpt-4o-mini",
            response_type=None  # unstructured
        )
    # In unstructured mode for multiple prompts, our design returns a list of PromptResult objects.
    assert isinstance(result, list)
    assert all(hasattr(r, "prompt") for r in result)
    assert "Prompt one" in result[0].prompt


# --- Explicit non-Pydantic type usage ---

@pytest.mark.asyncio
async def test_single_prompt_with_str_as_response_type():
    """
    Test that if the user explicitly passes 'str' as the response_type,
    the handler returns unstructured free text.
    """
    handler = UnifiedLLMHandler(openai_api_key="fake_openai_key")
    with patch("llmhandler.api_handler.Agent.run") as mock_run:
        fake = MagicMock()
        fake.data = "Raw text response"
        mock_run.return_value = fake
        result = await handler.process(
            prompts="What is the meaning of life?",
            model="openai:gpt-4o-mini",
            response_type=str  # Explicitly passing str should be treated as unstructured
        )
    assert isinstance(result, str)
    assert "Raw text" in result


# --- Provider-specific structured tests ---

@pytest.mark.asyncio
async def test_structured_anthropic():
    """Test a structured response with Anthropic provider."""
    handler = UnifiedLLMHandler(anthropic_api_key="fake_anthropic_key")
    with patch("llmhandler.api_handler.Agent.run") as mock_run:
        fake = MagicMock()
        fake.data = SimpleResponse(content="Anthropic result", confidence=0.95)
        mock_run.return_value = fake
        result = await handler.process(
            prompts="Tell me something interesting.",
            model="anthropic:claude-3-5-sonnet-20241022",
            response_type=SimpleResponse
        )
    assert result.success is True
    assert result.data.content == "Anthropic result"


@pytest.mark.asyncio
async def test_structured_deepseek():
    """Test a structured response with DeepSeek provider."""
    handler = UnifiedLLMHandler(deepseek_api_key="fake_deepseek_key")
    with patch("llmhandler.api_handler.Agent.run") as mock_run:
        fake = MagicMock()
        fake.data = SimpleResponse(content="DeepSeek result", confidence=0.95)
        mock_run.return_value = fake
        result = await handler.process(
            prompts="Explain quantum physics simply.",
            model="deepseek:deepseek-chat",
            response_type=SimpleResponse
        )
    assert result.success is True
    assert result.data.content == "DeepSeek result"


@pytest.mark.asyncio
async def test_structured_gemini():
    """Test a structured response with Gemini provider."""
    handler = UnifiedLLMHandler(gemini_api_key="fake_gemini_key")
    with patch("llmhandler.api_handler.Agent.run") as mock_run:
        fake = MagicMock()
        fake.data = SimpleResponse(content="Gemini result", confidence=0.95)
        mock_run.return_value = fake
        result = await handler.process(
            prompts="Compose a haiku about nature.",
            model="gemini:gemini-1.5-flash",
            response_type=SimpleResponse
        )
    assert result.success is True
    assert result.data.content == "Gemini result"


@pytest.mark.asyncio
async def test_structured_openrouter():
    """Test a structured response with OpenRouter (routing to Anthropic)."""
    handler = UnifiedLLMHandler(openrouter_api_key="fake_openrouter_key")
    with patch("llmhandler.api_handler.Agent.run") as mock_run:
        fake = MagicMock()
        fake.data = SimpleResponse(content="OpenRouter result", confidence=0.9)
        mock_run.return_value = fake
        result = await handler.process(
            prompts="List uses for AI in gardening.",
            model="openrouter:anthropic/claude-3-5-haiku-20241022",
            response_type=SimpleResponse
        )
    assert result.success is True
    assert result.data.content == "OpenRouter result"


@pytest.mark.asyncio
async def test_batch_mode_structured():
    """
    Test batch mode with a typed response.
    This uses mocked network calls to simulate file upload and batch processing.
    """
    handler = UnifiedLLMHandler(openai_api_key="fake_openai_key")
    with patch("llmhandler.api_handler.Agent") as mock_agent_cls:
        mock_agent_inst = MagicMock()
        mock_model_client = MagicMock()
        mock_model_client.files.create = AsyncMock()
        mock_model_client.batches.create = AsyncMock()
        mock_model_client.batches.retrieve = AsyncMock()
        mock_model_client.files.content = AsyncMock()
        mock_model_client.files.create.return_value.id = "fake_file_id"
        mock_model_client.batches.create.return_value.id = "fake_batch_id"
        # Simulate the batch status: first "in_progress", then "completed"
        mock_model_client.batches.retrieve.side_effect = [
            MagicMock(status="in_progress", output_file_id="fake_out_file"),
            MagicMock(status="completed", output_file_id="fake_out_file")
        ]
        # Simulate file content (two lines, one per prompt)
        mock_model_client.files.content.return_value.content = (
            b'{"response":{"body":{"choices":[{"message":{"content":"Batch result A"}}]}}}\n'
            b'{"response":{"body":{"choices":[{"message":{"content":"Batch result B"}}]}}}\n'
        )
        mock_agent_inst.model.model_name = "gpt-4o"
        mock_agent_inst.model.client = mock_model_client
        mock_agent_cls.return_value = mock_agent_inst

        result = await handler.process(
            prompts=["BatchPromptA", "BatchPromptB"],
            model="openai:gpt-4o",
            response_type=SimpleResponse,
            batch_mode=True,
        )

    assert result.success is True
    assert isinstance(result.data, BatchResult)
    assert len(result.data.results) == 2
    assert result.data.results[0]["response"].content == "Batch result A"
    assert result.data.results[1]["response"].content == "Batch result B"


# --- NEW: Partial Failure in Multiple Prompts Test ---

@pytest.mark.asyncio
async def test_partial_failure_multiple_prompts():
    """
    Test that when processing multiple prompts in regular (non-batch) mode,
    a failing prompt (e.g. one exceeding the token limit) is captured as an error
    in its PromptResult, while other prompts succeed.
    """
    handler = UnifiedLLMHandler(openai_api_key="fake_openai_key")

    # Simulate Agent.run so that:
    # - For a prompt explicitly labeled "Bad prompt", raise an Exception
    #   (simulate a token-limit error).
    # - Otherwise, return a valid result.
    async def side_effect(prompt: str):
        if prompt == "Bad prompt":
            raise Exception("Token limit exceeded")
        else:
            fake = MagicMock()
            fake.data = SimpleResponse(content=f"Response for {prompt}", confidence=0.9)
            return fake

    # Patch Agent.run using AsyncMock so that the asynchronous side_effect is properly awaited.
    with patch("llmhandler.api_handler.Agent.run", new_callable=AsyncMock, side_effect=side_effect):
        prompts = ["Good prompt", "Bad prompt", "Another good prompt"]
        result = await handler.process(
            prompts=prompts,
            model="openai:gpt-4o-mini",
            response_type=SimpleResponse
        )

    # The result should be a UnifiedResponse wrapping a list of PromptResult objects.
    assert result.success is True
    multi_results = result.data
    assert isinstance(multi_results, list)
    assert len(multi_results) == 3

    for pr in multi_results:
        if pr.prompt == "Bad prompt":
            assert pr.error is not None
            assert "Token limit exceeded" in pr.error
            assert pr.data is None
        else:
            assert pr.data is not None
            assert pr.error is None
            assert pr.data.content == f"Response for {pr.prompt}"
