# LLMHandler

**Unified LLM Interface with Typed & Unstructured Responses**

LLMHandler is a Python package (published on PyPI as **`llm_handler_validator`**) that provides a single, consistent interface to interact with multiple large language model (LLM) providers. It supports both **structured (Pydantic‑validated)** and **unstructured free‑form** responses, along with advanced features like **rate limiting**, **batch processing**, and **per‑prompt partial failure handling**.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Configuration](#configuration)
- [Model Format](#model-format)
- [Supported Providers and Their Models](#supported-providers-and-their-models)
- [UnifiedLLMHandler Constructor](#unifiedllmhandler-constructor)
- [Usage Examples](#usage-examples)
  - [Structured Response (Single Prompt)](#structured-response-single-prompt)
  - [Unstructured Response (Single Prompt)](#unstructured-response-single-prompt)
  - [Multiple Prompts (Structured)](#multiple-prompts-structured)
  - [Batch Processing Example](#batch-processing-example)
  - [Partial Failure Example](#partial-failure-example)
  - [Vertex AI Usage Example](#vertex-ai-usage-example)
- [Advanced Features](#advanced-features)
- [Testing](#testing)
- [Development & Contribution](#development--contribution)
- [License](#license)
- [Contact](#contact)

---

## Overview

LLMHandler unifies access to various LLM providers by letting you specify a model using a **provider prefix** (e.g. `openai:gpt-4o-mini`). When a Pydantic model is provided, LLMHandler automatically appends JSON schema instructions to ensure the response is valid JSON matching your schema. If no model is provided, raw free‑form text is returned.

Additional capabilities include:

- **Rate limiting**: Control requests/minute to avoid overloading APIs.
- **Batch processing**: Send multiple prompts at once (OpenAI only).
- **Partial failure handling**: If one prompt fails due to an API error or input that’s too large, that error is captured per prompt while other prompts still succeed.

---

## Features

1. **Multi-Provider Support**  
   Easily switch between **OpenAI**, **Anthropic**, **Gemini** (Generative Language API or Vertex AI), **DeepSeek**, etc. by just changing the prefix in the model string.

2. **Structured & Unstructured Responses**  
   - **Structured**: Provide a Pydantic model for validated, strongly typed data.  
   - **Unstructured**: Omit or pass `response_type=None` to get free-text strings.

3. **Batch Processing**  
   For OpenAI typed responses, you can process multiple prompts in a single job, with results auto-written to JSONL files.

4. **Rate Limiting**  
   Avoid hitting provider rate limits by specifying a requests-per-minute value.

5. **Partial Failure Handling**  
   When multiple prompts are processed, each prompt’s success or error is returned individually.

6. **Easy Configuration**  
   Reads keys from your local environment or `.env` file by default, or you can pass them directly when instantiating `UnifiedLLMHandler`.

---

## Installation

Because this package is published to PyPI as **`llm_handler_validator`**, you can install it via **pip**:

```bash
pip install llm_handler_validator
```

Or using **PDM** (which will add it to your `pyproject.toml` dependencies):

```bash
pdm add llm_handler_validator
```

If you are working from a cloned repository and have a local `pdm.lock` file, simply running:

```bash
pdm install
```

will install **this package** (from local source) and all its dependencies. PDM detects the project name (`llm_handler_validator`) from your local `pyproject.toml`.

---

## Configuration

Create a `.env` file in your project’s root (or set environment variables in some other way). For example:

```ini
OPENAI_API_KEY=your_openai_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key
GEMINI_API_KEY=your_google_gla_api_key
DEEPSEEK_API_KEY=your_deepseek_api_key
```

*(If you plan to use Vertex AI, you can rely on application default credentials **OR** specify your own service account file, region, and project ID—see examples below.)*

---

## Model Format

Every model reference is a string:

```
<provider>:<model_name>
```

- **Provider Prefix:** Tells LLMHandler which integration class to instantiate, e.g. `openai:`, `anthropic:`, `google-gla:`, `google-vertex:`, etc.
- **Model Name:** The actual model ID to use (e.g. `gpt-4o-mini`, `gemini-2.0-flash`).

---

## Supported Providers and Their Models

Below is a summary of recognized providers and example model names.

### OpenAI (`openai:`)

- **GPT-4o Series**  
  - `openai:gpt-4o`  
  - `openai:gpt-4o-mini`

- **o1 Series**  
  - `openai:o1`  
  - `openai:o3-mini` 

### Anthropic (`anthropic:`)

- e.g.,  
  - `anthropic:claude-3-5-haiku-latest`  
  - `anthropic:claude-3-5-sonnet-latest`  
  - `anthropic:claude-3-opus-latest`

### Gemini  
**Generative Language API** (`google-gla:`)  
**Vertex AI** (`google-vertex:`)

Valid Gemini model names:  
- `gemini-2.0-flash`  
- `gemini-2.0-flash-lite-preview-02-05`  
- `gemini-2.0-flash-thinking-exp-01-21`  


### DeepSeek (`deepseek:`)

- e.g.,  
  - `deepseek:deepseek-chat`

### Ollama (`ollama:`)

- e.g.,  
  - `ollama:llama3.2`  
  - `ollama:llama3.3-70b-specdec`

*(See [ollama.com/library](https://ollama.com/library) for more details.)*

---

## UnifiedLLMHandler Constructor

```python
class UnifiedLLMHandler(
    requests_per_minute: Optional[int] = None,
    batch_output_dir: str = "batch_output",
    openai_api_key: Optional[str] = None,
    openrouter_api_key: Optional[str] = None,
    deepseek_api_key: Optional[str] = None,
    anthropic_api_key: Optional[str] = None,
    google_gla_api_key: Optional[str] = None,
    google_vertex_service_account_file: Optional[str] = None,
    google_vertex_region: Optional[str] = None,
    google_vertex_project_id: Optional[str] = None,
)
```

1. **`requests_per_minute`**  
   - An integer specifying how many requests per minute to allow.  
   - If given, the class internally rate-limits outgoing requests to avoid surpassing provider limits.

2. **`batch_output_dir`**  
   - Directory path where JSONL results are written if you use **batch mode** (`batch_mode=True`) with OpenAI typed responses.  
   - Defaults to `"batch_output"`.

3. **`openai_api_key`**  
   - Explicitly pass your OpenAI API key if you do not wish to rely on the `OPENAI_API_KEY` environment variable.

4. **`openrouter_api_key`**  
   - API key for OpenRouter (which also wraps Anthropic, OpenAI, etc.).  
   - If not given, we look for `OPENROUTER_API_KEY` in your environment.

5. **`deepseek_api_key`**  
   - API key for DeepSeek. If not given, we look for `DEEPSEEK_API_KEY` in your environment.

6. **`anthropic_api_key`**  
   - API key for Anthropic. If not given, we look for `ANTHROPIC_API_KEY`.

7. **`google_gla_api_key`**  
   - API key used for **Google’s Generative Language API** (often referred to as the “Gemini hobby” API).  
   - If not given, we look for `GEMINI_API_KEY`.

8. **`google_vertex_service_account_file`**, **`google_vertex_region`**, **`google_vertex_project_id`**  
   - Optional credentials/configuration for **Vertex AI** usage. If you are running inside GCP, application default credentials may suffice.  
   - If you need local dev or a custom service account, pass a path to the JSON file, the region (e.g. `"us-central1"`), and your GCP project ID.

Any parameters left as `None` will attempt to load from environment variables (where relevant) or rely on defaults if the environment variables are missing.

---

## Usage Examples

### Structured Response (Single Prompt)

```python
import asyncio
from llmhandler.api_handler import UnifiedLLMHandler
from llmhandler._internal_models import SimpleResponse

async def structured_example():
    handler = UnifiedLLMHandler()
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
        # No response_type => returns raw text
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

*(Available for **OpenAI** typed responses only.)*

```python
import asyncio
from llmhandler.api_handler import UnifiedLLMHandler
from llmhandler._internal_models import SimpleResponse

async def batch_example():
    handler = UnifiedLLMHandler(requests_per_minute=60)
    prompts = [
        "Generate a slogan for a coffee brand.",
        "Create a tagline for a tea company.",
        "Write a catchphrase for a juice brand."
    ]
    batch_result = await handler.process(
        prompts=prompts,
        model="openai:gpt-4o-mini",
        response_type=SimpleResponse,
        batch_mode=True
    )
    print("Batch Processing Result:", batch_result.data)

asyncio.run(batch_example())
```

### Partial Failure Example

```python
import asyncio
from llmhandler.api_handler import UnifiedLLMHandler
from llmhandler._internal_models import SimpleResponse

async def partial_failure_example():
    handler = UnifiedLLMHandler()
    good_prompt = "Tell me a fun fact about penguins."
    # Construct a 'bad' prompt that far exceeds typical token limits:
    bad_prompt = "word " * 2000001
    another_good = "What are the benefits of regular exercise?"
    partial_prompts = [good_prompt, bad_prompt, another_good]

    result = await handler.process(
        prompts=partial_prompts,
        model="openai:gpt-4o-mini",
        response_type=SimpleResponse
    )
    print("Partial Failure Real API Result:")
    # result.data is a list of per-prompt results
    for pr in result.data:
        display_prompt = pr.prompt if len(pr.prompt) < 60 else pr.prompt[:60] + "..."
        if pr.error:
            print(f"  ERROR: {pr.error}")
        else:
            print(f"  Response: {pr.data}")

asyncio.run(partial_failure_example())
```

### Vertex AI Usage Example

If you need to run Gemini models via **Vertex AI** (as opposed to the simpler “hobby” API with `google-gla:`), you can pass your service account JSON and region like so:

```python
import asyncio
from llmhandler.api_handler import UnifiedLLMHandler
from llmhandler._internal_models import SimpleResponse

async def vertex_example():
    handler = UnifiedLLMHandler(
        google_vertex_service_account_file="path/to/service_account.json",
        google_vertex_region="us-central1",
        google_vertex_project_id="my-vertex-project",
    )
    result = await handler.process(
        prompts="Summarize advanced deep learning concepts.",
        model="google-vertex:gemini-2.0-flash",  # Vertex AI usage
        response_type=SimpleResponse
    )
    if result.success:
        print("Vertex AI Gemini response:", result.data)
    else:
        print("Vertex AI error:", result.error)

asyncio.run(vertex_example())
```

---

## Advanced Features

1. **Batch Processing & Rate Limiting**  
   - Initialize the handler with `requests_per_minute` to throttle requests.  
   - For typed usage with **OpenAI**, pass `batch_mode=True` to process multiple prompts in a single job. Results get written to JSONL in your `batch_output_dir`.

2. **Structured vs. Unstructured**  
   - Supply a Pydantic model as `response_type` to parse the response into typed fields.  
   - Omit or set `response_type=None` to get raw text.

3. **Partial Failure Handling**  
   - Submitting multiple prompts returns a list of results, one per prompt. If an API call fails for just one prompt, that prompt’s result includes an error, while others remain unaffected.

4. **Google Gemini**  
   - **`google-gla:`** prefix → “hobby” Generative Language API.  
   - **`google-vertex:`** prefix → Vertex AI, recommended for production.

---

## Testing

To run the test suite:

```bash
pdm run pytest
```

This will execute all tests (including integration and unit tests). Make sure you have a valid `.env` (or environment variables) set if you want to test real API calls.

---

## Development & Contribution

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/yourusername/LLMHandler.git
   cd LLMHandler
   ```

2. **Install Dependencies** (including dev dependencies and your local package):

   ```bash
   pdm install
   ```

   > This command reads `pyproject.toml` and `pdm.lock`, installing the project (named `llm_handler_validator`) locally plus any dev tools (e.g. `pytest`).

3. **Run Tests**:

   ```bash
   pdm run pytest
   ```

4. **Publish to PyPI** (assuming you have permission and a valid account):

   ```bash
   pdm version patch
   pdm build
   pdm publish
   ```

   This will upload the built **`llm_handler_validator`** package to PyPI.

5. **Submit a Pull Request** for any improvements or bug fixes.

---

## License

This project is licensed under the [MIT License](LICENSE).

---

## Contact

For questions, feedback, or contributions, please reach out to:

**Bryan Nsoh**  
Email: [bryan.anye.5@gmail.com](mailto:bryan.anye.5@gmail.com)

---

**Happy coding with LLMHandler!**