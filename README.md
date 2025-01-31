# LLMHandler

**Unified LLM Interface with Typed & Unstructured Responses**

LLMHandler is a Python package that provides a single, consistent interface to interact with multiple large language model (LLM) providers such as OpenAI, Anthropic, Gemini, DeepSeek, VertexAI, and OpenRouter. Whether you need validated, structured responses (using Pydantic models) or unstructured free-form text, LLMHandler makes it simple to integrate LLM capabilities into your projects.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Configuration](#configuration)
- [Quick Start](#quick-start)
- [API Reference](#api-reference)
- [Testing](#testing)
- [Development & Contribution](#development--contribution)
- [License](#license)
- [Contact](#contact)

---

## Overview

LLMHandler unifies access to various LLM providers by letting you specify a model using a provider prefix (e.g. `openai:gpt-4o-mini`). The package automatically appends JSON schema instructions when a Pydantic model is provided to validate and parse responses. Alternatively, you can request unstructured free-form text. Additional features include batch processing and optional rate limiting.

---

## Features

- **Multi-Provider Support:**  
  Easily switch between providers (OpenAI, Anthropic, Gemini, DeepSeek, VertexAI, OpenRouter) using a simple model identifier.
  
- **Structured & Unstructured Responses:**  
  Validate responses against Pydantic models (e.g. `SimpleResponse`, `PersonResponse`) or receive raw text.
  
- **Batch Processing:**  
  Process multiple prompts in batch mode with results written to JSONL files.
  
- **Rate Limiting:**  
  Optionally control the number of requests per minute.
  
- **Easy Configuration:**  
  Load API keys and other settings automatically from a `.env` file.

---

## Installation

### Requirements

- Python **3.9** or later

### Using PDM

Install dependencies with [PDM](https://pdm.fming.dev/):

```bash
pdm install
```

### Using Pip (once published to PyPI)

```bash
pip install llmhandler
```

---

## Configuration

Create a `.env` file in your project’s root to store your API keys:

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

## Quick Start

Below is a simple example demonstrating both structured and unstructured usage:

```python
import asyncio
from llmhandler.api_handler import UnifiedLLMHandler
from llmhandler._internal_models import SimpleResponse, PersonResponse

async def main():
    # Initialize the handler with your API key (or let it load from .env)
    handler = UnifiedLLMHandler(openai_api_key="your_openai_api_key")

    # Structured response (typed output)
    structured = await handler.process(
        prompts="Generate a catchy marketing slogan for a coffee brand.",
        model="openai:gpt-4o-mini",
        response_type=SimpleResponse
    )
    print("Structured result:", structured)

    # Unstructured response (raw text output)
    unstructured = await handler.process(
        prompts="Tell me a fun fact about dolphins.",
        model="openai:gpt-4o-mini"
    )
    print("Unstructured result:", unstructured)

    # Multiple prompts with structured responses
    multiple = await handler.process(
        prompts=[
            "Describe a 28-year-old engineer named Alice with 3 key skills.",
            "Describe a 45-year-old pastry chef named Bob with 2 key skills."
        ],
        model="openai:gpt-4o-mini",
        response_type=PersonResponse
    )
    print("Multiple structured results:", multiple)

asyncio.run(main())
```

For additional examples, see the [examples/inference_test.py](examples/inference_test.py) file.

---

## API Reference

### UnifiedLLMHandler

The primary class for processing prompts.

#### Constructor Parameters

- **API Keys:**  
  `openai_api_key`, `openrouter_api_key`, `deepseek_api_key`, `anthropic_api_key`, `gemini_api_key`  
  (These can be provided as arguments or loaded automatically from the `.env` file.)

- **`requests_per_minute` (optional):**  
  Set a rate limit for outgoing requests.

- **`batch_output_dir` (optional):**  
  Directory for saving batch output files (default: `"batch_output"`).

#### Method: `process()`

- **Parameters:**
  - `prompts`: A single prompt (string) or a list of prompt strings.
  - `model`: Model identifier with a provider prefix (e.g., `"openai:gpt-4o-mini"`).
  - `response_type` (optional): A Pydantic model class for structured responses (e.g. `SimpleResponse`, `PersonResponse`). Omit or set to `None` for unstructured text.
  - `system_message` (optional): Additional instructions for the system prompt.
  - `batch_mode` (optional): Set to `True` to process multiple prompts in batch mode (supported only for structured responses using OpenAI models).
  - `retries` (optional): Number of retry attempts for failed requests.

- **Returns:**  
  A `UnifiedResponse` object (when using a typed response) or a raw string (or list of strings) for unstructured output.

---

## Testing

A comprehensive test suite is included. To run tests, simply execute:

```bash
pytest
```

---

## Development & Contribution

Contributions are welcome! To set up the development environment:

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/yourusername/LLMHandler.git
   cd LLMHandler
   ```

2. **Install Dependencies:**

   ```bash
   pdm install
   ```

3. **Run Tests:**

   ```bash
   pytest
   ```

4. **Submit a Pull Request:**  
   Make improvements or bug fixes and submit a PR.

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
