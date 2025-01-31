Below is an example of a comprehensive, clear README that you can use as your project’s README.md. You can adjust any sections as needed.

---

# LLMHandler

**Unified LLM Handler for Multi-Provider Interaction with Structured & Unstructured Outputs**

LLMHandler is a Python package that provides a unified interface for interacting with various large language model (LLM) providers (e.g. OpenAI, Anthropic, Gemini, DeepSeek, VertexAI, and OpenRouter). It leverages [Pydantic](https://pydantic-docs.helpmanual.io/) to enable strongly typed (structured) responses while still supporting free-form (unstructured) output.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Configuration](#configuration)
- [Quick Start / Usage Examples](#quick-start--usage-examples)
- [API Reference](#api-reference)
  - [UnifiedLLMHandler](#unifiedllmhandler)
  - [Response Models](#response-models)
- [Testing](#testing)
- [Development & Contribution](#development--contribution)
- [License](#license)
- [Contact](#contact)

---

## Overview

LLMHandler offers a consistent API to interact with multiple LLM providers through a single interface. Whether you need to enforce a specific JSON schema using Pydantic models or simply work with raw text responses, LLMHandler streamlines the process.

---

## Features

- **Multi-Provider Support:**  
  Interact with providers such as OpenAI, Anthropic, Gemini, DeepSeek, VertexAI, and OpenRouter by specifying a model string with a provider prefix (e.g. `openai:gpt-4o-mini`).

- **Structured vs. Unstructured Responses:**  
  Supply a Pydantic model (like `SimpleResponse`, `PersonResponse`, etc.) to automatically validate and parse responses or omit the model (or set to `None`) to receive raw text.

- **Batch Processing:**  
  Process multiple prompts at once in batch mode (supported for structured responses with OpenAI models), with output written to JSONL files.

- **Rate Limiting:**  
  Optional built-in rate limiting to control requests per minute.

- **Easy Configuration:**  
  API keys and other settings can be managed through a `.env` file.

---

## Installation

### Requirements

- Python **3.13** or later

### Installing with PDM

This project uses [PDM](https://pdm.fming.dev/) for dependency management and packaging.

```bash
pdm install
```

### Installing via Pip

Once published on PyPI, you can install with:

```bash
pip install llmhandler
```

---

## Configuration

Create a `.env` file in your project’s root (or modify the provided template) to store your API keys and other sensitive configuration values:

```ini
# .env
OPENROUTER_API_KEY=your_openrouter_api_key
DEEPSEEK_API_KEY=your_deepseek_api_key
OPENAI_API_KEY=your_openai_api_key
GEMINI_API_KEY=your_gemini_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key
```

LLMHandler automatically loads these values at runtime.

---

## Quick Start / Usage Examples

Below is an example demonstrating both structured (typed) and unstructured usage. See the [examples/inference_test.py](examples/inference_test.py) file for a more comprehensive guide.

```python
import asyncio
from llmhandler.api_handler import UnifiedLLMHandler
from llmhandler.models import SimpleResponse, PersonResponse

async def main():
    # Initialize the handler with your API key(s)
    handler = UnifiedLLMHandler(openai_api_key="your_openai_api_key")
    
    # --- 1. Structured Usage (Typed Output) ---
    structured = await handler.process(
        prompts="Generate a short marketing slogan for a coffee brand.",
        model="openai:gpt-4o-mini",
        response_type=SimpleResponse
    )
    print("Structured result:")
    print(structured)
    
    # --- 2. Unstructured Usage (Raw Text) ---
    unstructured = await handler.process(
        prompts="Tell me a fun fact about dolphins.",
        model="openai:gpt-4o-mini"
        # response_type omitted → returns free-form text
    )
    print("\nUnstructured result:")
    print(unstructured)
    
    # --- 3. Multiple Prompts with Structured Output ---
    multi_structured = await handler.process(
        prompts=[
            "Describe a 28-year-old engineer named Alice with 3 key skills.",
            "Describe a 45-year-old pastry chef named Bob with 2 key skills."
        ],
        model="openai:gpt-4o-mini",
        response_type=PersonResponse
    )
    print("\nMultiple structured results:")
    print(multi_structured)
    
    # --- 4. Batch Mode Example (Structured Responses) ---
    batch_structured = await handler.process(
        prompts=[
            "Write a short story about a dragon who loves sunsets.",
            "Explain the top 5 health benefits of daily jogging."
        ],
        model="openai:gpt-4o-mini",
        response_type=SimpleResponse,
        batch_mode=True
    )
    print("\nBatch mode structured result:")
    print(batch_structured)

if __name__ == "__main__":
    asyncio.run(main())
```

---

## API Reference

### UnifiedLLMHandler

The primary class that provides methods to process prompts and interact with the underlying LLM providers.

#### Constructor Parameters

- **`openai_api_key`**, **`openrouter_api_key`**, **`deepseek_api_key`**, **`anthropic_api_key`**, **`gemini_api_key`**:  
  Your API keys for the respective providers. If not provided, the handler will attempt to load them from the environment variables.

- **`requests_per_minute`** (optional):  
  An integer specifying a rate limit for outgoing requests.

- **`batch_output_dir`** (optional):  
  The directory where batch output files will be stored (default is `"batch_output"`).

#### Method: `process()`

Processes one or more prompts and returns either:

- A **`UnifiedResponse`** containing structured (typed) data if `response_type` is provided as a Pydantic model.
- A raw string or list of strings if `response_type` is omitted or set to `None`.

Key Parameters:
- **`prompts`**: A single prompt (string) or a list of prompt strings.
- **`model`**: A model identifier string with a provider prefix (e.g. `"openai:gpt-4o-mini"`). The prefix tells the handler which API to use.
- **`response_type`** (optional): A Pydantic model class that the response should conform to (e.g. `SimpleResponse`, `PersonResponse`). Omit or set to `None` for unstructured output.
- **`system_message`** (optional): Additional instructions to be included in the system prompt.
- **`batch_mode`** (optional): Set to `True` to process multiple prompts in batch mode (supported only for structured responses using OpenAI models).
- **`retries`** (optional): Number of retry attempts if the request fails.

### Response Models

The package includes several predefined models in the `llmhandler.models` module:

- **`SimpleResponse`**  
  A basic model containing:
  - `content` (str): The text response.
  - `confidence` (float, optional): A confidence score between 0 and 1.

- **`MathResponse`**  
  Designed for math problems:
  - `answer` (float, optional)
  - `reasoning` (str, optional)
  - `confidence` (float, optional)

- **`PersonResponse`**  
  For describing a person:
  - `name` (str)
  - `age` (int)
  - `occupation` (str)
  - `skills` (list of strings)

- **`UnifiedResponse`**  
  An envelope that wraps the result:
  - `success` (bool)
  - `data` (structured data, list, or batch result)
  - `error` (str, optional)
  - `original_prompt` (str, optional)

---

## Testing

A comprehensive test suite is provided in the `tests/` directory. To run the tests, simply execute:

```bash
pytest
```

---

## Development & Contribution

Contributions and improvements are very welcome!

### Getting Started

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/yourusername/LLMHandler.git
   cd LLMHandler
   ```

2. **Install Dependencies:**

   Using PDM:
   ```bash
   pdm install
   ```

3. **Run Tests:**

   ```bash
   pytest
   ```

4. **Make Your Changes & Submit a PR**

---

## License

This project is licensed under the [MIT License](LICENSE).

---

## Contact

For questions, feedback, or contributions, please reach out to:

**Bryan Nsoh**  
Email: [bryan.anye.5@gmail.com](mailto:bryan.anye.5@gmail.com)

---

Happy coding with LLMHandler!

---

*This README is intended to provide a clear overview and serve as a guide for both users and developers looking to deploy or contribute to LLMHandler as a Python package.*

