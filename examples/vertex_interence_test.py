"""
Example demonstrating Vertex AI integration with UnifiedLLMHandler.
This example uses environment variables for authentication and configuration.

Required environment variables:
- GOOGLE_APPLICATION_CREDENTIALS: Path to your service account JSON file
- GOOGLE_VERTEX_PROJECT_ID: Your Google Cloud project ID

Optional environment variables:
- GOOGLE_VERTEX_REGION: Vertex AI region (defaults to us-central1 if not set)

Run with: python examples/vertex_inference_example.py
"""

import asyncio
import os
from pydantic import BaseModel
from dotenv import load_dotenv

# Load environment variables from .env file if present
load_dotenv(override=True)

# Import the UnifiedLLMHandler
from llmhandler import UnifiedLLMHandler

# Check if required environment variables are set
required_vars = ["GOOGLE_APPLICATION_CREDENTIALS", "GOOGLE_VERTEX_PROJECT_ID"]
missing_vars = [var for var in required_vars if not os.getenv(var)]
if missing_vars:
    print(f"Error: Missing required environment variables: {', '.join(missing_vars)}")
    print("Please set these variables before running the example.")
    exit(1)

# Print current configuration
print(f"Using service account: {os.getenv('GOOGLE_APPLICATION_CREDENTIALS')}")
print(f"Project ID: {os.getenv('GOOGLE_VERTEX_PROJECT_ID')}")
print(f"Region: {os.getenv('GOOGLE_VERTEX_REGION', 'us-central1')} (default is us-central1)")

# Example Pydantic model for structured responses
class AnalysisResponse(BaseModel):
    summary: str
    key_points: list[str]
    sentiment: str


async def run_typed_inference():
    """Run a structured inference example with Vertex AI Gemini 2.0 Flash"""
    print("\n=== Running Structured Inference with Vertex AI ===")
    
    # Create handler with Vertex AI model as default
    handler = UnifiedLLMHandler(
        default_model="google-vertex:gemini-2.0-flash-001"
    )
    
    # Example prompt
    prompt = "Analyze the latest trends in artificial intelligence and provide a summary, key points, and overall sentiment."
    
    # Process with typed response
    result = await handler.process(
        prompts=prompt,
        response_type=AnalysisResponse
    )
    
    if result.success:
        print("\nStructured Response:")
        print(f"Summary: {result.data.summary}")
        print("\nKey Points:")
        for i, point in enumerate(result.data.key_points, 1):
            print(f"{i}. {point}")
        print(f"\nSentiment: {result.data.sentiment}")
    else:
        print(f"Error: {result.error}")


async def run_unstructured_inference():
    """Run an unstructured (raw text) inference example with Vertex AI Gemini 2.0 Flash"""
    print("\n=== Running Unstructured Inference with Vertex AI ===")
    
    # Create handler with Vertex AI model as default
    handler = UnifiedLLMHandler(
        default_model="google-vertex:gemini-2.0-flash-001"
    )
    
    # Example prompt
    prompt = "Write a short poem about technology and nature."
    
    # Process with unstructured response (no response_type)
    result = await handler.process(prompts=prompt)
    
    print("\nUnstructured Response:")
    print(result)


async def run_multiple_prompts():
    """Run multiple prompts in a single call"""
    print("\n=== Running Multiple Prompts with Vertex AI ===")
    
    # Create handler with Vertex AI model as default
    handler = UnifiedLLMHandler(
        default_model="google-vertex:gemini-2.0-flash-001"
    )
    
    # List of prompts
    prompts = [
        "What are three benefits of cloud computing?",
        "Explain quantum computing in simple terms."
    ]
    
    # Process multiple prompts
    results = await handler.process(prompts=prompts)
    
    print("\nMultiple Prompt Results:")
    for i, result in enumerate(results):
        print(f"\n--- Prompt {i+1} ---")
        print(f"Prompt: {result.prompt}")
        if result.error:
            print(f"Error: {result.error}")
        else:
            print(f"Response: {result.data}")


async def main():
    """Run all examples"""
    try:
        await run_typed_inference()
        await run_unstructured_inference()
        await run_multiple_prompts()
    except Exception as e:
        print(f"Error during execution: {e}")


if __name__ == "__main__":
    asyncio.run(main())