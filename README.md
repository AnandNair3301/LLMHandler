# LLMHandler

[![PyPI version](https://badge.fury.io/py/llm-handler-validator.svg)](https://badge.fury.io/py/llm-handler-validator)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

**A unified interface for multiple LLM providers with typed and unstructured responses**

LLMHandler simplifies working with multiple LLM providers through a single, consistent interface. Define your own Pydantic models to get structured JSON responses, or get raw text when you need it.

## 🚀 Key Features

- **Multi-Provider Support**: Use OpenAI, Anthropic, Gemini, DeepSeek, and more with a simple provider prefix
- **Default Model**: Set once, use everywhere in your codebase
- **Structured & Unstructured Responses**: Get typed data with Pydantic or raw text
- **Batch Processing**: Process multiple prompts efficiently (OpenAI only)
- **Rate Limiting**: Avoid hitting provider rate limits
- **Partial Failure Handling**: One prompt failing doesn't break everything
- **Environment Variable Support**: Configure via `.env` or environment variables

## 📦 Installation

```bash
pip install llm_handler_validator
```

Or with PDM:

```bash
pdm add llm_handler_validator
```

## 🔑 Configuration

### API Keys

Create a `.env` file in your project root:

```ini
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key
GEMINI_API_KEY=your_google_gla_key
DEEPSEEK_API_KEY=your_deepseek_key
OPENROUTER_API_KEY=your_openrouter_key

# For Vertex AI
GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json
GOOGLE_VERTEX_PROJECT_ID=your-project-id
GOOGLE_VERTEX_REGION=us-central1
```

Or pass keys directly when creating the handler:

```python
handler = UnifiedLLMHandler(
    openai_api_key="your_key_here",
    anthropic_api_key="your_key_here"
)
```

## 🧠 Core Concepts

### 1. Model Format

Specify providers with a prefix: `provider:model_name`

```
openai:gpt-4o-mini
anthropic:claude-3-5-haiku-latest
google-gla:gemini-2.0-flash-001
google-vertex:gemini-2.0-flash-001
deepseek:deepseek-chat
```

### 2. Response Types

- **Structured**: Provide a Pydantic model to get validated JSON
- **Unstructured**: Omit the model to get raw text

### 3. Default Model

Set a default model when creating the handler to simplify code:

```python
handler = UnifiedLLMHandler(default_model="openai:gpt-4o-mini")
```

## 📋 Quick Reference

```python
from llmhandler import UnifiedLLMHandler
from pydantic import BaseModel

# Define your response model
class Summary(BaseModel):
    main_points: list[str]
    conclusion: str

# Create handler with default model
handler = UnifiedLLMHandler(
    default_model="openai:gpt-4o",
    requests_per_minute=60
)

# Get structured response
result = await handler.process(
    prompts="Summarize the benefits of exercise",
    response_type=Summary
)

# Get unstructured response (raw text)
text = await handler.process(
    prompts="Tell me a joke",
    model="anthropic:claude-3-5-haiku-latest"  # Overrides default
)
```

## 📊 Supported Providers

| Provider | Prefix | Example Models | Notes |
|----------|--------|----------------|-------|
| OpenAI | `openai:` | `gpt-4o`, `gpt-4o-mini` | Full support for all features |
| Anthropic | `anthropic:` | `claude-3-5-haiku-latest`, `claude-3-5-sonnet-20241022` | Structured and unstructured responses |
| Gemini (GLA) | `google-gla:` | `gemini-2.0-flash-001` | Uses Generative Language API (hobby) |
| Gemini (Vertex) | `google-vertex:` | `gemini-2.0-flash-001` | Enterprise Google Cloud integration |
| DeepSeek | `deepseek:` | `deepseek-chat` | Structured and unstructured responses |
| OpenRouter | `openrouter:` | `anthropic/claude-3-5-haiku-20241022` | Route to multiple providers |

## 🔍 Detailed Usage Examples

### 1. Structured Response (Single Prompt)

```python
import asyncio
from pydantic import BaseModel
from llmhandler import UnifiedLLMHandler

class Recipe(BaseModel):
    title: str
    ingredients: list[str]
    instructions: list[str]
    prep_time_minutes: int

async def get_recipe():
    handler = UnifiedLLMHandler()
    
    result = await handler.process(
        prompts="Create a simple pasta recipe",
        model="openai:gpt-4o-mini",
        response_type=Recipe
    )
    
    if result.success:
        recipe = result.data  # This is a Recipe instance
        print(f"Recipe: {recipe.title}")
        print("Ingredients:")
        for item in recipe.ingredients:
            print(f"- {item}")
        print(f"Prep time: {recipe.prep_time_minutes} minutes")
    else:
        print(f"Error: {result.error}")

asyncio.run(get_recipe())
```

### 2. Unstructured Response (Raw Text)

```python
import asyncio
from llmhandler import UnifiedLLMHandler

async def get_story():
    handler = UnifiedLLMHandler()
    
    # Method 1: Omit response_type
    story = await handler.process(
        prompts="Write a short story about a robot learning to paint",
        model="anthropic:claude-3-5-haiku-latest"
    )
    
    # Method 2: Explicitly set response_type=None
    story2 = await handler.process(
        prompts="Write a poem about autumn",
        model="openai:gpt-4o-mini",
        response_type=None
    )
    
    print(story)
    print("\n--- Second story ---\n")
    print(story2)

asyncio.run(get_story())
```

### 3. Multiple Prompts (Structured)

```python
import asyncio
from pydantic import BaseModel
from llmhandler import UnifiedLLMHandler

class MovieReview(BaseModel):
    title: str
    rating: float
    summary: str

async def get_movie_reviews():
    handler = UnifiedLLMHandler(default_model="openai:gpt-4o-mini")
    
    prompts = [
        "Write a review of The Matrix (1999)",
        "Write a review of Inception (2010)",
        "Write a review of Interstellar (2014)"
    ]
    
    result = await handler.process(
        prompts=prompts,
        response_type=MovieReview
    )
    
    if result.success:
        # result.data is a list of PromptResults
        for i, review_result in enumerate(result.data):
            if review_result.error:
                print(f"Error in review {i+1}: {review_result.error}")
            else:
                review = review_result.data  # This is a MovieReview instance
                print(f"\nReview {i+1}: {review.title}")
                print(f"Rating: {review.rating}/10")
                print(f"Summary: {review.summary}")
    else:
        print(f"Overall error: {result.error}")

asyncio.run(get_movie_reviews())
```

### 4. Using Default Model

```python
import asyncio
from pydantic import BaseModel
from llmhandler import UnifiedLLMHandler

class Answer(BaseModel):
    response: str
    confidence: float

async def use_default_model():
    # Set default model once
    handler = UnifiedLLMHandler(
        default_model="openai:gpt-4o-mini",
        requests_per_minute=60
    )
    
    # Use default model (no need to specify model in each call)
    result1 = await handler.process(
        prompts="What is machine learning?",
        response_type=Answer
    )
    
    # Override default model for specific calls
    result2 = await handler.process(
        prompts="Explain quantum computing",
        model="anthropic:claude-3-5-sonnet-20241022",
        response_type=Answer
    )
    
    print(f"Result 1: {result1.data.response}")
    print(f"Result 2: {result2.data.response}")

asyncio.run(use_default_model())
```

### 5. Batch Processing (OpenAI Only)

```python
import asyncio
from pydantic import BaseModel
from llmhandler import UnifiedLLMHandler

class CountryFact(BaseModel):
    country: str
    capital: str
    population: str
    fun_fact: str

async def batch_process():
    handler = UnifiedLLMHandler(
        default_model="openai:gpt-4o-mini",
        batch_output_dir="my_batch_results"
    )
    
    countries = [
        "Tell me about France",
        "Tell me about Japan",
        "Tell me about Brazil",
        "Tell me about Egypt",
        "Tell me about Australia"
    ]
    
    # Process in batch mode (only works with OpenAI models)
    result = await handler.process(
        prompts=countries,
        response_type=CountryFact,
        batch_mode=True
    )
    
    if result.success:
        print(f"Batch ID: {result.data.metadata.batch_id}")
        print(f"Total requests: {result.data.metadata.num_requests}")
        print(f"Output file: {result.data.metadata.output_file_path}")
        
        # Access individual results
        for i, item in enumerate(result.data.results):
            if "error" in item:
                print(f"Error in item {i}: {item['error']}")
            else:
                fact = item["response"]  # This is a CountryFact instance
                print(f"\nCountry: {fact.country}")
                print(f"Capital: {fact.capital}")
                print(f"Fun fact: {fact.fun_fact}")
    else:
        print(f"Batch error: {result.error}")

asyncio.run(batch_process())
```

### 6. Vertex AI Integration

```python
import asyncio
from pydantic import BaseModel
from llmhandler import UnifiedLLMHandler

class Analysis(BaseModel):
    summary: str
    key_points: list[str]
    conclusion: str

async def vertex_example():
    # Using environment variables for auth (recommended)
    # GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json
    # GOOGLE_VERTEX_PROJECT_ID=my-project-id
    # GOOGLE_VERTEX_REGION=us-central1
    
    handler = UnifiedLLMHandler(
        default_model="google-vertex:gemini-2.0-flash-001"
    )
    
    result = await handler.process(
        prompts="Analyze the impact of artificial intelligence on healthcare",
        response_type=Analysis
    )
    
    if result.success:
        analysis = result.data
        print(f"Summary: {analysis.summary}")
        print("\nKey Points:")
        for point in analysis.key_points:
            print(f"- {point}")
        print(f"\nConclusion: {analysis.conclusion}")
    else:
        print(f"Error: {result.error}")

    # You can also pass credentials explicitly
    explicit_handler = UnifiedLLMHandler(
        google_vertex_service_account_file="/path/to/service-account.json",
        google_vertex_project_id="my-project-id",
        google_vertex_region="us-central1"
    )
    
    # Use explicit_handler the same way...

asyncio.run(vertex_example())
```

## 🔄 Response Handling

The library returns different response types depending on your input:

### 1. Single Prompt + Pydantic Model

```python
result = await handler.process(
    prompts="Single prompt",
    model="openai:gpt-4o-mini",
    response_type=MyModel
)

# result is a UnifiedResponse:
# - result.success: True if successful
# - result.data: Instance of MyModel if successful
# - result.error: Error message if unsuccessful
```

### 2. Single Prompt + No Model (Unstructured)

```python
result = await handler.process(
    prompts="Single prompt",
    model="openai:gpt-4o-mini"
)

# result is just a string (the raw LLM output)
```

### 3. Multiple Prompts + Pydantic Model

```python
result = await handler.process(
    prompts=["Prompt 1", "Prompt 2", "Prompt 3"],
    model="openai:gpt-4o-mini",
    response_type=MyModel
)

# result is a UnifiedResponse:
# - result.success: True if the overall operation succeeded
# - result.data: List of PromptResult objects, each with:
#   - .prompt: The original prompt text
#   - .data: Instance of MyModel if this prompt succeeded
#   - .error: Error message if this specific prompt failed
```

### 4. Multiple Prompts + No Model (Unstructured)

```python
result = await handler.process(
    prompts=["Prompt 1", "Prompt 2", "Prompt 3"],
    model="openai:gpt-4o-mini"
)

# result is a list of PromptResult objects, each with:
# - .prompt: The original prompt text
# - .data: Raw string response if this prompt succeeded
# - .error: Error message if this specific prompt failed
```

### 5. Batch Mode

```python
result = await handler.process(
    prompts=["Prompt 1", "Prompt 2", "Prompt 3"],
    model="openai:gpt-4o-mini",
    response_type=MyModel,
    batch_mode=True
)

# result is a UnifiedResponse:
# - result.success: True if successful
# - result.data: BatchResult object with:
#   - .metadata: Information about the batch job
#   - .results: List of dictionaries, each with 'prompt' and 'response' or 'error'
```

## ⚙️ Advanced Usage

### Custom System Message

```python
result = await handler.process(
    prompts="Tell me about quantum physics",
    model="anthropic:claude-3-5-haiku-latest",
    system_message="You are a physics professor explaining concepts to undergraduate students."
)
```

### Rate Limiting

```python
handler = UnifiedLLMHandler(
    requests_per_minute=60  # Limit to 60 requests per minute
)
```

### Full Configuration

```python
handler = UnifiedLLMHandler(
    # Default model (optional)
    default_model="openai:gpt-4o-mini",
    
    # Rate limiting
    requests_per_minute=60,
    
    # Batch processing output directory
    batch_output_dir="my_batch_results",
    
    # API keys (overrides environment variables)
    openai_api_key="your_openai_key",
    anthropic_api_key="your_anthropic_key",
    google_gla_api_key="your_gemini_key",
    deepseek_api_key="your_deepseek_key",
    openrouter_api_key="your_openrouter_key",
    
    # Vertex AI configuration (overrides environment variables)
    google_vertex_service_account_file="/path/to/service-account.json",
    google_vertex_region="us-central1",
    google_vertex_project_id="my-project-id"
)
```

## 💡 Tips and Best Practices

1. **Use default_model** for cleaner code when using the same model throughout your application
2. **Define clear Pydantic models** to get structured, validated responses
3. **Handle partial failures** by checking each item's `.error` field in multi-prompt responses
4. **Set appropriate rate limits** to avoid hitting provider API limits
5. **Use environment variables** for sensitive API keys rather than hardcoding
6. **For Vertex AI**, rely on application default credentials when possible for simpler authentication
7. **Use batch mode** for large sets of prompts to process more efficiently (OpenAI only)

## 📝 FAQ

### Q: How do I handle errors?

For structured responses, always check `result.success` before accessing `result.data`. For multi-prompt calls, check each individual prompt result's `.error` field.

### Q: Can I mix providers in the same application?

Yes! Simply specify the appropriate provider prefix for each model (e.g., `openai:gpt-4o-mini`, `anthropic:claude-3-5-haiku-latest`).

### Q: How does the default model work with multiple providers?

The default model is completely provider-agnostic. You can set an OpenAI model as default, then occasionally use Anthropic or Gemini models for specific calls.

### Q: How do I authenticate with Vertex AI?

Option 1: Set `GOOGLE_APPLICATION_CREDENTIALS` environment variable pointing to your service account JSON
Option 2: Pass `google_vertex_service_account_file` explicitly when creating the handler

### Q: What's the difference between 'google-gla' and 'google-vertex'?

- `google-gla:`: Generative Language API (simpler, "hobby" API)
- `google-vertex:`: Enterprise Vertex AI service on Google Cloud (more features, scalability)

### Q: What if a large language model errors out in a batch request?

For multi-prompt requests, other prompts will still be processed. For batch mode, if critical errors occur, the `metadata.error` field will contain details.

## 🧪 Testing

Run tests with:

```bash
pdm run pytest
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📬 Contact

**Bryan Nsoh**  
Email: [bryan.anye.5@gmail.com](mailto:bryan.anye.5@gmail.com)