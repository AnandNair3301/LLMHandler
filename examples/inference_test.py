"""
Comprehensive examples covering all usage scenarios for UnifiedLLMHandler.
Each provider is invoked at least once, and both structured and unstructured usage are demonstrated.
Run with: pdm run python examples/inference_test.py
Make sure your .env is configured with valid API keys.
"""

import asyncio
import os
from dotenv import load_dotenv

load_dotenv(override=True)

from llmhandler.api_handler import UnifiedLLMHandler
from llmhandler._internal_models import SimpleResponse, PersonResponse, UnifiedResponse

async def main():
    handler = UnifiedLLMHandler()

    # --- 1. Structured Usage (Typed Output) ---
    print("=== Structured Usage (Typed) ===")
    structured = await handler.process(
        prompts="Generate a short marketing slogan for a coffee brand.",
        model="openai:gpt-4o-mini",
        response_type=SimpleResponse
    )
    print("OpenAI structured result:")
    print(structured)

    # --- 2. Unstructured Usage (response_type omitted) ---
    print("\n=== Unstructured Usage (response_type omitted) ===")
    unstructured = await handler.process(
        prompts="Tell me a fun fact about dolphins.",
        model="openai:gpt-4o-mini"
    )
    print("OpenAI unstructured result:")
    print(unstructured)

    # --- 3. Unstructured Usage (Explicit response_type=None) ---
    print("\n=== Unstructured Usage (Explicit response_type=None) ===")
    unstructured_none = await handler.process(
        prompts="What is the capital of Italy?",
        model="openai:gpt-4o-mini",
        response_type=None
    )
    print("OpenAI unstructured (None) result:")
    print(unstructured_none)

    # --- 4. Explicit non-Pydantic type usage (str) ---
    print("\n=== Explicit non-Pydantic (str) Usage ===")
    explicit_str = await handler.process(
        prompts="Tell me a joke.",
        model="openai:gpt-4o-mini",
        response_type=str
    )
    print("Explicit str type result:")
    print(explicit_str)

    # --- 5. Structured Usage: Multiple Prompts ---
    print("\n=== Multiple Prompts Structured Usage ===")
    multi_structured = await handler.process(
        prompts=[
            "Describe a 28-year-old engineer named Alice with 3 key skills.",
            "Describe a 45-year-old pastry chef named Bob with 2 key skills."
        ],
        model="openai:gpt-4o-mini",
        response_type=PersonResponse
    )
    print("Multiple structured result:")
    print(multi_structured)

    # --- 6. Partial Failure Example (Real API Call) ---
    print("\n=== Partial Failure Example (Real API Call) ===")
    # Two good prompts and one extremely long (bad) prompt.
    good_prompt = "Tell me a fun fact about penguins."
    bad_prompt = "word " * 2000001  # deliberately exceeding token limits
    another_good = "What are the benefits of regular exercise?"
    partial_prompts = [good_prompt, bad_prompt, another_good]

    partial_results = await handler.process(
        prompts=partial_prompts,
        model="openai:gpt-4o-mini",
        response_type=SimpleResponse
    )
    print("Partial Failure Real API Result:")
    if isinstance(partial_results, UnifiedResponse):
        results_list = partial_results.data
    else:
        results_list = partial_results
    for pr in results_list:
        display_prompt = pr.prompt if len(pr.prompt) < 60 else pr.prompt[:60] + "..."
        print(f"Prompt: {display_prompt}")
        if pr.error:
            print(f"  ERROR: {pr.error}")
        else:
            print(f"  Response: {pr.data}")
        print("-" * 40)

    # --- 7. Provider-specific Examples ---
    print("\n=== Provider-specific Examples ===")

    # Anthropic
    anthropic = await handler.process(
        prompts="Summarize the benefits of daily meditation.",
        model="anthropic:claude-3-5-sonnet-20241022",
        response_type=SimpleResponse
    )
    print("Anthropic structured result:")
    print(anthropic)

    # DeepSeek
    deepseek = await handler.process(
        prompts="Explain quantum entanglement simply.",
        model="deepseek:deepseek-chat",
        response_type=SimpleResponse
    )
    print("DeepSeek structured result:")
    print(deepseek)

    # Gemini via Generative Language API
    gemini = await handler.process(
        prompts="Compose a haiku about the sunrise.",
        model="google-gla:gemini-1.5-flash",
        response_type=SimpleResponse
    )
    print("Gemini structured result:")
    print(gemini)

    # OpenRouter (routing to Anthropic)
    openrouter = await handler.process(
        prompts="List three creative uses for AI in home gardening.",
        model="openrouter:anthropic/claude-3-5-haiku-20241022",
        response_type=SimpleResponse
    )
    print("OpenRouter structured result:")
    print(openrouter)
    
    # --- 8. Batch Mode (Structured) ---

    # --- 9. Batch Mode (Structured) ---
    # print("\n=== Batch Mode Structured Usage ===")
    # batch_structured = await handler.process(
    #     prompts=[
    #         "Write a short story about a dragon who loves sunsets.",
    #         "Explain the top 5 health benefits of daily jogging in bullet points."
    #     ],
    #     model="openai:gpt-4o-mini",
    #     response_type=SimpleResponse,
    #     batch_mode=True
    # )
    # print("Batch mode structured result:")
    # print(batch_structured)


if __name__ == "__main__":
    asyncio.run(main())
