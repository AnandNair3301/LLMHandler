# LLMHandler

**Unified LLM Interface with Typed & Unstructured Responses**

LLMHandler is a Python package that provides a single, consistent interface to interact with multiple large language model (LLM) providers. It supports both structured (Pydantic‑validated) and unstructured free‑form responses, along with advanced features like rate limiting and batch processing.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Configuration](#configuration)
- [Model Format](#model-format)
- [Supported Providers and Their Models](#supported-providers-and-their-models)
- [Usage Examples](#usage-examples)
  - [Structured Response (Single Prompt)](#structured-response-single-prompt)
  - [Unstructured Response (Single Prompt)](#unstructured-response-single-prompt)
  - [Multiple Prompts (Structured)](#multiple-prompts-structured)
  - [Batch Processing Example](#batch-processing-example)
- [Advanced Features](#advanced-features)
- [Testing](#testing)
- [Development & Contribution](#development--contribution)
- [License](#license)
- [Contact](#contact)

---

## Overview

LLMHandler unifies access to various LLM providers by letting you specify a model using a provider prefix (e.g. `openai:gpt-4o-mini`). The package automatically appends JSON schema instructions when a Pydantic model is provided to validate and parse responses. Alternatively, you can request unstructured free‑form text. Advanced features include batch processing and rate limiting.

---

## Features

- **Multi‑Provider Support:**  
  Switch easily between providers (OpenAI, Anthropic, Gemini, DeepSeek, Ollama, etc.) using a simple model identifier.
  
- **Structured & Unstructured Responses:**  
  Validate outputs using Pydantic models or receive raw text.
  
- **Batch Processing:**  
  Process multiple prompts together with results written to JSONL files.
  
- **Rate Limiting:**  
  Optionally control the number of requests per minute.
  
- **Easy Configuration:**  
  Automatically load API keys and settings from a `.env` file.

---

## Installation

### Requirements

- Python **3.9** or later

### Using PDM

```bash
pdm install
```

### Using Pip (when available)

```bash
pip install llmhandler
```

---

## Configuration

Create a `.env` file in your project’s root and add your API keys:

```ini
OPENAI_API_KEY=your_openai_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key
GEMINI_API_KEY=your_gemini_api_key
OLLAMA_API_KEY=your_ollama_api_key
DEEPSEEK_API_KEY=your_deepseek_api_key
```

LLMHandler automatically loads these values at runtime.

---

## Model Format

Every model is passed as a string in the form:

```
<provider>:<model_name>
```

- **Provider Prefix:** Identifies the integration class and loads the proper API key and settings.
- **Model Name:** Often validated via a type alias (e.g. `KnownModelName`) to select the specific LLM.

---

## Supported Providers and Their Models

| **Provider**  | **Prefix**           | **Supported Models** |
|---------------|----------------------|----------------------|
| **OpenAI**    | `openai:`            | **GPT‑4o Series:**<br>• `openai:gpt-4o`<br>• `openai:gpt-4o-2024-05-13`<br>• `openai:gpt-4o-2024-08-06`<br>• `openai:gpt-4o-2024-11-20`<br>• `openai:gpt-4o-audio-preview`<br>• `openai:gpt-4o-audio-preview-2024-10-01`<br>• `openai:gpt-4o-audio-preview-2024-12-17`<br>• `openai:gpt-4o-mini`<br>• `openai:gpt-4o-mini-2024-07-18`<br>• `openai:gpt-4o-mini-audio-preview`<br>• `openai:gpt-4o-mini-audio-preview-2024-12-17`<br><br>**o1 Series:**<br>• `openai:o1`<br>• `openai:o1-2024-12-17`<br>• `openai:o1-mini`<br>• `openai:o1-mini-2024-09-12`<br>• `openai:o1-preview`<br>• `openai:o1-preview-2024-09-12` |
| **Anthropic** | `anthropic:`         | • `anthropic:claude-3-5-haiku-latest`<br>• `anthropic:claude-3-5-sonnet-latest`<br>• `anthropic:claude-3-opus-latest` |
| **Gemini**    | `google-gla:`<br>(Generative Language API)<br>`google-vertex:`<br>(Vertex AI) | • `gemini-1.0-pro`<br>• `gemini-1.5-flash`<br>• `gemini-1.5-flash-8b`<br>• `gemini-1.5-pro`<br>• `gemini-2.0-flash-exp`<br>• `gemini-2.0-flash-thinking-exp-01-21`<br>• `gemini-exp-1206` |
| **Ollama**    | `ollama:`            | Accepts any valid Ollama model. Common examples:<br>• `ollama:llama3.2`<br>• `ollama:llama3.2-vision`<br>• `ollama:llama3.3-70b-specdec`<br>(See [ollama.com/library](https://ollama.com/library)) |
| **Deepseek**  | `deepseek:`          | • `deepseek:deepseek-chat` |

*Note: For LLaMA-based models, Ollama (and providers like Groq, if available) are the primary options.*

---

## Usage Examples

### Structured Response (Single Prompt)

```python
import asyncio
from llmhandler.api_handler import UnifiedLLMHandler
from llmhandler._internal_models import SimpleResponse

async def structured_example():
    handler = UnifiedLLMHandler()  # API keys auto-loaded from .env
    result = await handler.process(
        prompts="Generate a catchy marketing slogan for a coffee brand.",
        model="openai:gpt-4o-mini",
        response_type=SimpleResponse
    )
    print("Structured Response:", result.data)

asyncio.run(structured_example())
```

### Unstructured Response (Single Prompt)

```python
import asyncio
from llmhandler.api_handler import UnifiedLLMHandler

async def unstructured_example():
    handler = UnifiedLLMHandler()
    result = await handler.process(
        prompts="Tell me a fun fact about dolphins.",
        model="openai:gpt-4o-mini"
        # No response_type provided: returns raw text.
    )
    print("Unstructured Response:", result)

asyncio.run(unstructured_example())
```

### Multiple Prompts (Structured)

```python
import asyncio
from llmhandler.api_handler import UnifiedLLMHandler
from llmhandler._internal_models import SimpleResponse

async def multiple_prompts_example():
    handler = UnifiedLLMHandler()
    prompts = [
        "Generate a slogan for a coffee brand.",
        "Create a tagline for a tea company."
    ]
    result = await handler.process(
        prompts=prompts,
        model="openai:gpt-4o-mini",
        response_type=SimpleResponse
    )
    print("Multiple Structured Responses:", result.data)

asyncio.run(multiple_prompts_example())
```

### Batch Processing Example

```python
import asyncio
from llmhandler.api_handler import UnifiedLLMHandler
from llmhandler._internal_models import SimpleResponse

async def batch_example():
    # Set a rate limit to avoid overwhelming the API
    handler = UnifiedLLMHandler(requests_per_minute=60)
    prompts = [
        "Generate a slogan for a coffee brand.",
        "Create a tagline for a tea company.",
        "Write a catchphrase for a juice brand."
    ]
    # Use batch_mode=True to process multiple prompts together (structured responses only)
    batch_result = await handler.process(
        prompts=prompts,
        model="openai:gpt-4o-mini",
        response_type=SimpleResponse,
        batch_mode=True
    )
    print("Batch Processing Result:", batch_result.data)

asyncio.run(batch_example())
```

---

## Advanced Features

- **Batch Processing & Rate Limiting:**  
  Initialize the handler with `requests_per_minute` to throttle calls. When processing a list of prompts, set `batch_mode=True` to handle them as a batch (supported only for structured responses).

- **Structured vs. Unstructured Responses:**  
  - Supply a Pydantic model as `response_type` for validated, structured output.  
  - Omit or set `response_type=None` to receive raw, unstructured text.

- **Troubleshooting:**  
  Error messages (such as schema validation failures or misconfigured API keys) are clearly reported. Ensure your model strings follow the `<provider>:<model_name>` format exactly.

---

## Testing

A comprehensive test suite is included. To run tests, simply execute:

```bash
pytest
```

---

## Development & Contribution

Contributions are welcome! To set up your development environment:

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

4. **Submit a Pull Request** with your improvements or bug fixes.

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