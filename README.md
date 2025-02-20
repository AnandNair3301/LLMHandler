Below is a **fully revised** README that addresses every nuanced question about **how the library returns responses**—particularly when multiple prompts are provided—and **clarifies that users are expected to define their own Pydantic models**. It explicitly shows how to handle the returned data **without** directly importing any “internal” classes.

---

# LLMHandler

**Unified LLM Interface with Typed & Unstructured Responses**

LLMHandler is a Python package (published on PyPI as **`llm_handler_validator`**) that provides a single, consistent interface to interact with multiple large language model (LLM) providers. **We do not ship “built-in” Pydantic models that you must reuse**—rather, **you define your own** according to the schema you desire, and **LLMHandler** enforces JSON output matching that schema. Alternatively, you can request **unstructured** responses.

Key features include **rate limiting**, **batch processing** for OpenAI typed responses, and **per-prompt partial failure handling**—where multiple prompts return a list of results, each indicating success or error independently.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Configuration](#configuration)
- [Model Format](#model-format)
- [Supported Providers and Their Models](#supported-providers-and-their-models)
- [UnifiedLLMHandler Constructor](#unifiedllmhandler-constructor)
- [Defining Your Own Models](#defining-your-own-models)
- [Usage Examples](#usage-examples)
  - [Structured Response (Single Prompt)](#structured-response-single-prompt)
  - [Unstructured Response (Single Prompt)](#unstructured-response-single-prompt)
  - [Multiple Prompts (Structured)](#multiple-prompts-structured)
  - [Batch Processing Example](#batch-processing-example)
  - [Partial Failure Example (Multi-Prompt)](#partial-failure-example-multi-prompt)
  - [Vertex AI Usage Example](#vertex-ai-usage-example)
- [How Responses Are Returned](#how-responses-are-returned)
- [Advanced Features](#advanced-features)
- [Testing](#testing)
- [Development & Contribution](#development--contribution)
- [License](#license)
- [Contact](#contact)

---

## Overview

LLMHandler unifies access to various LLM providers by letting you specify a model with a prefix (e.g., `openai:gpt-4o-mini`). **If you supply a Pydantic model**, LLMHandler instructs the LLM to return JSON conforming to that schema, then validates the response into your model. **If you omit a Pydantic model**, you get raw text.

Additional capabilities include:

- **Rate limiting**: Control requests/minute to avoid hitting provider limits.
- **Batch processing**: Bulk-handle multiple prompts for typed OpenAI calls.
- **Partial failure**: If one prompt fails (e.g. token limit), that prompt’s error is captured, and the other prompts still succeed.

---

## Features

1. **Multi-Provider Support**  
   Swap providers (OpenAI, Anthropic, Gemini, etc.) by changing the prefix in your model string.

2. **Structured & Unstructured**  
   - **Structured**: Provide a custom Pydantic model to parse the JSON response.  
   - **Unstructured**: Skip the model (or set `response_type=None`) to receive free-text.

3. **Batch Processing**  
   In typed mode for OpenAI, you can process multiple prompts in a single background job, with outputs stored to JSONL.

4. **Rate Limiting**  
   Just set `requests_per_minute=...`, and the library paces API calls accordingly.

5. **Per-Prompt Partial Failure**  
   If you pass multiple prompts, each prompt’s success or failure stands alone. You see which prompts failed and which succeeded.

6. **Easy Configuration**  
   Keys can be read from environment variables or `.env`, or you can pass them explicitly.

---

## Installation

Since the package is on PyPI as **`llm_handler_validator`**, you can install it via **pip**:

```bash
pip install llm_handler_validator
```

**Or** if you use **PDM**, add it to your project with:

```bash
pdm add llm_handler_validator
```

If you’ve cloned the repository and have a local `pdm.lock`, then:

```bash
pdm install
```

will install this package (as specified in `pyproject.toml`) plus dependencies.

---

## Configuration

Create a `.env` file in your project’s root, or set environment variables. For example:

```ini
OPENAI_API_KEY=your_openai_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key
GEMINI_API_KEY=your_google_gla_api_key
DEEPSEEK_API_KEY=your_deepseek_api_key
```

If you plan to use **Vertex AI** (`google-vertex:`), you can rely on application default credentials **or** specify a service account JSON, region, and project ID. (See the [Vertex AI Usage Example](#vertex-ai-usage-example).)

---

## Model Format

For each LLM request, you specify a **provider** and a **model name** in the format:

```
<provider>:<model_name>
```

Example:  
- `openai:gpt-4o-mini`  
- `anthropic:claude-3-5-haiku-latest`  
- `google-gla:gemini-2.0-flash-001`  
- `deepseek:deepseek-chat`

---

## Supported Providers and Their Models

### OpenAI (`openai:`)

- **GPT-4o Series**  
  - `openai:gpt-4o`  
  - `openai:gpt-4o-mini`

- **o1/o3 Series**  
  - `openai:o1`  
  - `openai:o3-mini`

### Anthropic (`anthropic:`)
- e.g. `anthropic:claude-3-5-haiku-latest`

### Gemini
- **Generative Language API** (`google-gla:`)  
- **Vertex AI** (`google-vertex:`)  
  - e.g. `google-vertex:gemini-2.0-flash-001`

### DeepSeek (`deepseek:`)
- e.g. `deepseek:deepseek-chat`

### Ollama (`ollama:`)
- e.g. `ollama:llama3.2`, etc.

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

**Parameters**:
- **`requests_per_minute`**: Rate limit for requests (integer).
- **`batch_output_dir`**: Where JSONL results go in **batch mode** (OpenAI typed only).
- **`openai_api_key`, `openrouter_api_key`, `deepseek_api_key`, `anthropic_api_key`, `google_gla_api_key`**: Optional overrides for environment-based API keys.
- **`google_vertex_service_account_file`, `google_vertex_region`, `google_vertex_project_id`**: Credentials/config for Vertex AI usage. If omitted, you may rely on GCP default credentials.

---

## Defining Your Own Models

The point of LLMHandler is for **you to define your own Pydantic models** that describe the shape of data you want from the LLM. For example:

```python
from pydantic import BaseModel, Field
from typing import Optional

class MyCustomResponse(BaseModel):
    title: str
    summary: str
    rating: Optional[float] = Field(None, ge=0, le=5)
```

Then, when you call `process(..., response_type=MyCustomResponse)`, LLMHandler instructs the LLM: _“Return exclusively valid JSON matching MyCustomResponse”_ and parses it.

---

## Usage Examples

### Structured Response (Single Prompt)

```python
import asyncio
from pydantic import BaseModel
from llmhandler.api_handler import UnifiedLLMHandler

# Define your own model
class AdSlogan(BaseModel):
    slogan: str

async def structured_example():
    handler = UnifiedLLMHandler()
    result = await handler.process(
        prompts="Generate a short, clever slogan for a coffee brand.",
        model="openai:gpt-4o-mini",
        response_type=AdSlogan
    )
    if result.success:
        # result.data is an AdSlogan instance
        print("Structured Response:", result.data.slogan)
    else:
        print("Error:", result.error)

asyncio.run(structured_example())
```

In this example, we define a custom model named `AdSlogan`. We do **not** import anything from `_internal_models`. The LLM is told to produce a JSON object with a `slogan` field.

### Unstructured Response (Single Prompt)

```python
import asyncio
from llmhandler.api_handler import UnifiedLLMHandler

async def unstructured_example():
    handler = UnifiedLLMHandler()
    # Without a Pydantic model, we get raw text
    result = await handler.process(
        prompts="Tell me a fun fact about dolphins.",
        model="openai:gpt-4o-mini"
    )
    if isinstance(result, str):
        print("Unstructured Response:", result)
    else:
        # Could also check for errors in a more advanced scenario
        print("Unexpected or error:", result)

asyncio.run(unstructured_example())
```

### Multiple Prompts (Structured)

```python
import asyncio
from pydantic import BaseModel
from llmhandler.api_handler import UnifiedLLMHandler

class ShortTagline(BaseModel):
    tagline: str

async def multiple_prompts_example():
    handler = UnifiedLLMHandler()
    prompts = [
        "Generate a slogan for a coffee brand.",
        "Create a tagline for a tea company."
    ]
    result = await handler.process(
        prompts=prompts,
        model="openai:gpt-4o-mini",
        response_type=ShortTagline
    )
    # With multiple prompts, data is a list.
    if result.success and isinstance(result.data, list):
        for i, prompt_result in enumerate(result.data):
            if prompt_result.error:
                print(f"Prompt {i} had error:", prompt_result.error)
            else:
                # This is our ShortTagline instance
                print(f"Prompt {i} => Tagline:", prompt_result.data.tagline)
    else:
        print("Error or unexpected data:", result)

asyncio.run(multiple_prompts_example())
```

### Batch Processing Example

*(Available for **OpenAI** typed responses only.)*

```python
import asyncio
from pydantic import BaseModel
from llmhandler.api_handler import UnifiedLLMHandler

class CatchPhrase(BaseModel):
    phrase: str

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
        response_type=CatchPhrase,
        batch_mode=True
    )
    # In batch mode, result.data is a BatchResult with metadata/results
    if batch_result.success:
        print("Batch metadata:", batch_result.data.metadata)
        print("Results array:", batch_result.data.results)
    else:
        print("Batch error:", batch_result.error)

asyncio.run(batch_example())
```

### Partial Failure Example (Multi-Prompt)

```python
import asyncio
from pydantic import BaseModel
from llmhandler.api_handler import UnifiedLLMHandler

class FunFact(BaseModel):
    fact: str

async def partial_failure_example():
    handler = UnifiedLLMHandler()
    good_prompt = "Tell me a fun fact about penguins."
    # Construct a 'bad' prompt that far exceeds typical token limits:
    bad_prompt = "word " * 2000001
    another_good = "What are the benefits of regular exercise?"
    prompts = [good_prompt, bad_prompt, another_good]

    result = await handler.process(
        prompts=prompts,
        model="openai:gpt-4o-mini",
        response_type=FunFact
    )
    # Here, result.data is likely a list. Each item has `.prompt`, `.data`, and `.error`.
    if result.success and isinstance(result.data, list):
        for i, prompt_result in enumerate(result.data):
            print(f"Prompt {i}: {prompt_result.prompt[:50]}...")
            if prompt_result.error:
                print("  ERROR:", prompt_result.error)
            else:
                # This is our FunFact instance
                print("  Fact:", prompt_result.data.fact)
    else:
        print("Overall error:", result.error)

asyncio.run(partial_failure_example())
```

### Vertex AI Usage Example

If you want to run Gemini models via **Vertex AI** (instead of the simpler “hobby” `google-gla:` API), you can pass your service account JSON, region, and project ID:

```python
import asyncio
from pydantic import BaseModel
from llmhandler.api_handler import UnifiedLLMHandler

class MLConcepts(BaseModel):
    summary: str

async def vertex_example():
    handler = UnifiedLLMHandler(
        google_vertex_service_account_file="path/to/service_account.json",
        google_vertex_region="us-central1",
        google_vertex_project_id="my-vertex-project",
    )
    result = await handler.process(
        prompts="Summarize advanced deep learning concepts.",
        model="google-vertex:gemini-2.0-flash-001",
        response_type=MLConcepts
    )
    if result.success:
        print("Vertex AI Gemini response:", result.data.summary)
    else:
        print("Vertex AI error:", result.error)

asyncio.run(vertex_example())
```

---

## How Responses Are Returned

When you call `handler.process(...)`, you get back one of the following:

1. **Single Prompt, Typed**:  
   - If the call succeeds overall, `result.success` is `True` and `result.data` is **one instance** of your Pydantic model (e.g. `MyCustomResponse`).
   - If the call fails (e.g. invalid API key), `result.success` is `False` and `result.error` explains why.

2. **Single Prompt, Unstructured**:  
   - Returns a **raw string** on success (the text from the LLM).
   - Or if an error occurs, you get a library-defined error object (with `success=False`, etc.). In practice, check if `isinstance(result, str)`.

3. **Multiple Prompts**:
   - `result.data` is **a list** (one entry per prompt).
   - Each entry has three fields:  
     - `.prompt`: The text you passed,  
     - `.data`: The typed object (if success) or the raw text (if untyped),  
     - `.error`: A string if that single prompt failed (otherwise `None`).
   - This allows partial failure. If some prompts exceed token limits, you get an error for those, but the rest succeed.

**In short**: You typically just check `result.success`. If `result.data` is a list, loop over each item. If `result.data` is a single object, handle it directly.

**Note**: We do *not* require you to import any internal class like `PromptResult`. You can simply treat each list item as an object with `.prompt`, `.error`, and `.data`.

---

## Advanced Features

1. **Batch Processing & Rate Limiting**  
   - Use `requests_per_minute=...` to throttle calls.  
   - For typed usage with **OpenAI**, pass `batch_mode=True` to run multiple prompts in a single job.

2. **Partial Failure**  
   - Each prompt in a multi-prompt list is independent; if one fails, it doesn’t kill the entire call.

3. **Google Gemini**  
   - `google-gla:` prefix => Generative Language API (basic/hobby).  
   - `google-vertex:` prefix => Vertex AI, recommended for production or advanced usage.

---

## Testing

You can run the test suite with:

```bash
pdm run pytest
```

If you have real API keys set in `.env`, some tests may attempt live calls unless mocked. Alternatively, you can rely on mocking in the library’s test suite.

---

## Development & Contribution

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/yourusername/LLMHandler.git
   cd LLMHandler
   ```

2. **Install Dependencies** (including dev dependencies and this local package):

   ```bash
   pdm install
   ```

   This reads `pyproject.toml` & `pdm.lock` to set up an environment with everything needed (including `pytest`).

3. **Run Tests**:

   ```bash
   pdm run pytest
   ```

4. **Publish to PyPI** *(if you have permission and credentials)*:

   ```bash
   pdm build
   pdm publish
   ```

   This uploads the wheel/SDist to PyPI under **`llm_handler_validator`**.

5. **Submit a Pull Request** if you have improvements or bug fixes.

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