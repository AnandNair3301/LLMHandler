import asyncio
import os
from dotenv import load_dotenv

# Load .env so we can pick up API keys for OpenAI, Anthropic, etc.
load_dotenv(override=True)

from llmhandler.api_handler import UnifiedLLMHandler
from llmhandler.models import SimpleResponse, PersonResponse


async def main():
    """
    Quick script to test real calls to various providers with the "latest" model choices.
    
    Make sure your .env includes:
      OPENAI_API_KEY=
      ANTHROPIC_API_KEY=
      GEMINI_API_KEY=
      DEEPSEEK_API_KEY=
      OPENROUTER_API_KEY=
    Then run: pdm run python examples/manual_inference_test.py
    """

    # Handler will pull API keys from environment if None
    handler = UnifiedLLMHandler()

    print("---- 1) Single Prompt (OpenAI: gpt-4o-mini) ----")
    openai_result = await handler.process(
        prompts="Give me a short marketing slogan for a coffee brand.",
        model="openai:gpt-4o-mini",
        response_type=SimpleResponse,
    )
    print("OpenAI single-prompt response:\n", openai_result, "\n")

    print("---- 2) Single Prompt (Anthropic: claude-3-5-sonnet-20241022) ----")
    anthropic_result = await handler.process(
        prompts="Summarize the benefits of daily meditation in a single paragraph.",
        model="anthropic:claude-3-5-sonnet-20241022",
        response_type=SimpleResponse,
    )
    print("Anthropic single-prompt response:\n", anthropic_result, "\n")

    print("---- 3) Single Prompt (DeepSeek: deepseek-chat) ----")
    deepseek_result = await handler.process(
        prompts="Explain the basics of quantum entanglement in layman's terms.",
        model="deepseek:deepseek-chat",
        response_type=SimpleResponse,
    )
    print("DeepSeek single-prompt response:\n", deepseek_result, "\n")

    print("---- 4) Single Prompt (Gemini: gemini-1.5-flash) ----")
    gemini_result = await handler.process(
        prompts="Write a haiku about the sunrise.",
        model="gemini:gemini-1.5-flash",
        response_type=SimpleResponse,
    )
    print("Gemini single-prompt response:\n", gemini_result, "\n")

    print("---- 5) Single Prompt (OpenRouter => Anthropic: claude-3-5-haiku-20241022) ----")
    openrouter_result = await handler.process(
        prompts="List three imaginative uses for AI in home gardening.",
        model="openrouter:anthropic/claude-3-5-haiku-20241022",
        response_type=SimpleResponse,
    )
    print("OpenRouter single-prompt response:\n", openrouter_result, "\n")

    print("---- 6) Multiple Prompts with PersonResponse (OpenAI: gpt-4o-mini) ----")
    person_prompts = [
        "Describe a 28-year-old engineer named Alice with 3 key skills.",
        "Describe a 45-year-old pastry chef named Bob with 2 key skills.",
    ]
    multi_person_result = await handler.process(
        prompts=person_prompts,
        model="openai:gpt-4o-mini",
        response_type=PersonResponse,
    )
    print("Multiple prompts (PersonResponse) response:\n", multi_person_result, "\n")

    print("---- 7) Batch Mode (OpenAI: gpt-4o-mini) ----")
    # Only openai:* models support batch mode
    batch_prompts = [
        "Write a short story about a dragon who loves sunsets.",
        "Explain the top 5 health benefits of daily jogging in bullet points.",
    ]
    batch_result = await handler.process(
        prompts=batch_prompts,
        model="openai:gpt-4o-mini",
        response_type=SimpleResponse,
        batch_mode=True
    )
    print("OpenAI batch mode result:\n", batch_result, "\n")

    print("Done! If no errors, your code + keys are working end-to-end.")


if __name__ == "__main__":
    asyncio.run(main())
