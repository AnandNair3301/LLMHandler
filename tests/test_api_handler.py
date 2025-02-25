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
import os
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
    assert isinstance(result.data, list)
    assert len(result.data) == 2
    for resp, expected in zip(result.data, ["Resp A", "Resp B"]):
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
            model="openai:gpt-4o",
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
            response_type=None
        )
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
            response_type=str
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
async def test_structured_google_gla():
    """Test a structured response with 'google-gla' provider (Gemini hobby API)."""
    handler = UnifiedLLMHandler(google_gla_api_key="fake_gemini_key")
    with patch("llmhandler.api_handler.Agent.run") as mock_run:
        fake = MagicMock()
        fake.data = SimpleResponse(content="Gemini result", confidence=0.95)
        mock_run.return_value = fake
        result = await handler.process(
            prompts="Compose a haiku about nature.",
            model="google-gla:gemini-1.5-flash",
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
        mock_model_client.batches.retrieve.side_effect = [
            MagicMock(status="in_progress", output_file_id="fake_out_file"),
            MagicMock(status="completed", output_file_id="fake_out_file")
        ]
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


@pytest.mark.asyncio
async def test_partial_failure_multiple_prompts():
    """
    Test that when processing multiple prompts in regular (non-batch) mode,
    a failing prompt is captured as an error in its PromptResult,
    while other prompts succeed.
    """
    handler = UnifiedLLMHandler(openai_api_key="fake_openai_key")

    async def side_effect(prompt: str):
        if prompt == "Bad prompt":
            raise Exception("Token limit exceeded")
        else:
            fake = MagicMock()
            fake.data = SimpleResponse(content=f"Response for {prompt}", confidence=0.9)
            return fake

    with patch("llmhandler.api_handler.Agent.run", new_callable=AsyncMock, side_effect=side_effect):
        prompts = ["Good prompt", "Bad prompt", "Another good prompt"]
        result = await handler.process(
            prompts=prompts,
            model="openai:gpt-4o-mini",
            response_type=SimpleResponse
        )

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


@pytest.mark.asyncio
async def test_default_model():
    """Test that the default model is used when no model is provided in the process call."""
    handler = UnifiedLLMHandler(
        openai_api_key="fake_openai_key",
        default_model="openai:gpt-4o-mini"
    )
    with patch("llmhandler.api_handler.UnifiedLLMHandler._build_model_instance") as mock_build:
        mock_model = MagicMock()
        mock_build.return_value = mock_model
        
        # We need to mock agent.run as well
        with patch("llmhandler.api_handler.Agent") as mock_agent_cls:
            mock_agent = MagicMock()
            mock_agent.run = AsyncMock()  # Use AsyncMock instead of MagicMock
            fake_result = MagicMock()
            fake_result.data = "Default model response"
            mock_agent.run.return_value = fake_result
            mock_agent_cls.return_value = mock_agent
            
            result = await handler.process(
                prompts="Hello world",
                # No model provided, should use default
            )
            
    # Check that _build_model_instance was called with the default model
    mock_build.assert_called_once_with("openai:gpt-4o-mini")
    assert isinstance(result, str)
    assert result == "Default model response"


@pytest.mark.asyncio
async def test_model_precedence():
    """Test that a specified model takes precedence over the default model."""
    handler = UnifiedLLMHandler(
        openai_api_key="fake_openai_key",
        default_model="openai:gpt-4o-mini"
    )
    
    # Track which model is being used for each call
    used_models = []
    
    def track_model(model_str):
        used_models.append(model_str)
        mock_model = MagicMock()
        return mock_model
    
    with patch("llmhandler.api_handler.UnifiedLLMHandler._build_model_instance", side_effect=track_model):
        # We need to mock agent.run to return appropriate responses
        with patch("llmhandler.api_handler.Agent") as mock_agent_cls:
            mock_agent = MagicMock()
            mock_agent.run = AsyncMock()  # Use AsyncMock instead of MagicMock
            fake_result = MagicMock()
            fake_result.data = "Some response"
            mock_agent.run.return_value = fake_result
            mock_agent_cls.return_value = mock_agent
            
            # First call with explicit model
            await handler.process(
                prompts="Hello world",
                model="openai:gpt-4o",  # Should override default
            )
            
            # Second call without model
            await handler.process(
                prompts="Hello again",
                # No model provided, should use default
            )
    
    # Check that the correct models were used in each call
    assert used_models[0] == "openai:gpt-4o"  # First call used explicit model
    assert used_models[1] == "openai:gpt-4o-mini"  # Second call used default model


@pytest.mark.asyncio
async def test_vertex_ai_environment_variables():
    """Test that Vertex AI can use environment variables for configuration."""
    with patch.dict(os.environ, {
        "GOOGLE_APPLICATION_CREDENTIALS": "/path/to/fake_creds.json",
        "GOOGLE_VERTEX_REGION": "us-west1",
        "GOOGLE_VERTEX_PROJECT_ID": "test-project"
    }):
        handler = UnifiedLLMHandler()
        
        with patch("llmhandler.api_handler.VertexAIModel") as mock_vertex_cls:
            mock_vertex_inst = MagicMock()
            mock_vertex_cls.return_value = mock_vertex_inst
            
            # Mock Agent to prevent actual API calls
            with patch("llmhandler.api_handler.Agent") as mock_agent_cls:
                mock_agent = MagicMock()
                mock_agent.run = AsyncMock()  # Use AsyncMock instead of MagicMock
                fake_result = MagicMock()
                fake_result.data = "Vertex response"
                mock_agent.run.return_value = fake_result
                mock_agent_cls.return_value = mock_agent
                
                # We just check that the model is built with correct params
                await handler.process(
                    prompts="Test Vertex AI with env vars",
                    model="google-vertex:gemini-1.5-flash",
                    response_type=SimpleResponse
                )
        
        # Check that VertexAIModel was created with correct params from env
        mock_vertex_cls.assert_called_once_with(
            "gemini-1.5-flash",
            service_account_file=None,  # Not directly passed, using env var
            region="us-west1",
            project_id="test-project",
        )