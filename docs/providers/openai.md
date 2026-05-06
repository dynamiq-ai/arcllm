# OpenAI Provider Documentation

This document covers the OpenAI provider integration in arcllm — API
references, supported model families, and implementation details.

For the canonical list of available models and current pricing, consult
[OpenAI's models page](https://platform.openai.com/docs/models). Model
metadata that arcllm uses for capability checks and cost calculation
lives in `tmp/model_manifests/openai.json` and is refreshed monthly via
the `model-drift.yml` workflow.

## Quick Links to Official OpenAI Documentation

| Resource | URL |
|----------|-----|
| **API Reference** | https://platform.openai.com/docs/api-reference |
| **Chat Completions** | https://platform.openai.com/docs/api-reference/chat/create |
| **Embeddings** | https://platform.openai.com/docs/api-reference/embeddings/create |
| **Models** | https://platform.openai.com/docs/models |
| **Pricing** | https://openai.com/pricing |
| **Error Codes** | https://platform.openai.com/docs/guides/error-codes |

## Supported Models

### GPT-5 Family (Current Flagship)
- **Latest models with enhanced reasoning, vision, PDF support**
- Docs: https://platform.openai.com/docs/models/gpt-5
- ⚠️ **Note**: GPT-5 models use `max_completion_tokens` (not `max_tokens`)
- ⚠️ **Note**: Some GPT-5 models have restricted parameters (no temperature control)

| Model ID | Context | Max Output | Features | Price (Input/Output per 1M) |
|----------|---------|------------|----------|----------------------------|
| `gpt-5.2` | 256K | 32,768 | Vision, PDF, Tools, JSON Schema | $5.00 / $15.00 |
| `gpt-5.2-pro` | 256K | 131,072 | Extended Reasoning, Vision, PDF | $20.00 / $80.00 |
| `gpt-5.1` | 200K | 32,768 | Vision, PDF, Tools, JSON Schema | $4.00 / $12.00 |
| `gpt-5.1-codex` | 200K | 32,768 | Coding Optimized | $4.00 / $12.00 |
| `gpt-5` | 200K | 32,768 | Vision, PDF, Tools, JSON Schema | $3.00 / $10.00 |
| `gpt-5-mini` | 128K | 16,384 | Vision, Tools, JSON Schema | $0.50 / $2.00 |
| `gpt-5-nano` | 64K | 8,192 | Restricted params, fast | $0.15 / $0.60 |
| `gpt-5-pro` | 256K | 131,072 | Extended Reasoning | $15.00 / $60.00 |

### GPT-4.1 Family
- **Mid-range models with good parameter flexibility**
- Use `max_completion_tokens` instead of `max_tokens`

| Model ID | Context | Max Output | Features | Price (Input/Output per 1M) |
|----------|---------|------------|----------|----------------------------|
| `gpt-4.1` | 128K | 32,768 | Vision, Tools, JSON Schema | $2.00 / $8.00 |
| `gpt-4.1-mini` | 128K | 16,384 | Vision, Tools, JSON Schema | $0.10 / $0.40 |
| `gpt-4.1-nano` | 64K | 8,192 | Vision, Tools, Restricted params | $0.05 / $0.20 |

### o1/o3 Reasoning Models
- **Designed for complex reasoning tasks**
- Docs: https://platform.openai.com/docs/models/o1
- Note: Use `max_completion_tokens` instead of `max_tokens`

| Model ID | Context | Max Output | Features | Price (Input/Output per 1M) |
|----------|---------|------------|----------|----------------------------|
| `o3` | 256K | 131,072 | Vision, PDF, Tools, JSON Schema | $20.00 / $80.00 |
| `o3-mini` | 128K | 65,536 | Vision, Tools, JSON Schema | $5.00 / $20.00 |
| `o1` | 200K | 100,000 | Vision, Tools, JSON Schema | $15.00 / $60.00 |
| `o1-mini` | 128K | 65,536 | Limited | $3.00 / $12.00 |

### GPT-4o Family (⚠️ DEPRECATED - Retiring Feb 16, 2026)
- **Migrate to GPT-5 series before February 16, 2026**
- Docs: https://platform.openai.com/docs/models/gpt-4o

| Model ID | Context | Max Output | Features | Price (Input/Output per 1M) |
|----------|---------|------------|----------|----------------------------|
| `gpt-4o` | 128K | 16,384 | Vision, Tools, JSON Schema | $2.50 / $10.00 |
| `gpt-4o-mini` | 128K | 16,384 | Vision, Tools, JSON Schema | $0.15 / $0.60 |

### Legacy Models (⛔ DEPRECATED - Not Recommended)

| Model ID | Status | Recommendation |
|----------|--------|----------------|
| `gpt-4-turbo` | Deprecated | Use `gpt-5.2` |
| `gpt-4` | Deprecated | Use `gpt-5.2` |
| `gpt-3.5-turbo` | Deprecated | Use `gpt-5.2-instant` |

### Embedding Models
- Docs: https://platform.openai.com/docs/models/embeddings

| Model ID | Dimensions | Max Input | Price per 1M tokens |
|----------|------------|-----------|---------------------|
| `text-embedding-3-small` | 1536 (default), 256-1536 | 8,191 | $0.02 |
| `text-embedding-3-large` | 3072 (default), 256-3072 | 8,191 | $0.13 |

## API Usage

### Basic Completion
```python
import arcllm

# For general use with full parameter support, use gpt-4o-mini (cheap, fast)
response = arcllm.completion(
    model="openai/gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"},
    ],
    max_tokens=100,
    temperature=0.7,
)

# For GPT-5 models, use max_completion_tokens (temperature may be restricted)
response = arcllm.completion(
    model="openai/gpt-5.2",
    messages=[{"role": "user", "content": "Hello!"}],
    max_completion_tokens=100,  # Note: max_completion_tokens, not max_tokens
)

print(response.choices[0].message.content)
```

### Streaming
```python
stream = arcllm.completion(
    model="openai/gpt-5.2-instant",
    messages=[{"role": "user", "content": "Count to 5"}],
    stream=True,
    stream_options={"include_usage": True},  # Get usage in final chunk
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

### Tool Calling
```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"}
                },
                "required": ["location"]
            }
        }
    }
]

response = arcllm.completion(
    model="openai/gpt-5.2-instant",
    messages=[{"role": "user", "content": "What's the weather in Paris?"}],
    tools=tools,
    tool_choice="auto",
)

if response.choices[0].message.tool_calls:
    for tc in response.choices[0].message.tool_calls:
        print(f"Function: {tc.function.name}")
        print(f"Args: {tc.function.arguments}")
```

### Structured Output (JSON Mode)
```python
response = arcllm.completion(
    model="openai/gpt-5.2-instant",
    messages=[{
        "role": "user",
        "content": 'Return JSON: {"name": "...", "age": ...}'
    }],
    response_format={"type": "json_object"},
)
```

### Structured Output (JSON Schema)
- **Supported on all GPT-5 models**
- Docs: https://platform.openai.com/docs/guides/structured-outputs

```python
response = arcllm.completion(
    model="openai/gpt-5.2",
    messages=[{"role": "user", "content": "Generate a person named Bob, age 25"}],
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "person",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "age": {"type": "integer"}
                },
                "required": ["name", "age"],
                "additionalProperties": False
            }
        }
    },
)
```

### Embeddings
```python
response = arcllm.embedding(
    model="openai/text-embedding-3-small",
    input=["Hello, world!", "How are you?"],
    dimensions=256,  # Optional: reduce dimensions
)

for item in response.data:
    print(f"Embedding {item.index}: {len(item.embedding)} dims")
```

## Request Parameters

### Chat Completion Parameters
| Parameter | Type | Description | Docs |
|-----------|------|-------------|------|
| `model` | string | Model ID | Required |
| `messages` | array | Message list | Required |
| `max_tokens` | int | Max output tokens | Optional |
| `temperature` | float | Sampling temperature (0-2) | Optional |
| `top_p` | float | Nucleus sampling | Optional |
| `stream` | bool | Enable streaming | Optional |
| `stream_options` | object | `{"include_usage": true}` | Optional |
| `tools` | array | Tool definitions | Optional |
| `tool_choice` | string/object | Tool selection strategy | Optional |
| `response_format` | object | JSON mode/schema | Optional |
| `seed` | int | Deterministic output | Optional |
| `stop` | string/array | Stop sequences | Optional |
| `presence_penalty` | float | -2.0 to 2.0 | Optional |
| `frequency_penalty` | float | -2.0 to 2.0 | Optional |
| `logprobs` | bool | Return log probabilities | Optional |
| `top_logprobs` | int | Number of logprobs (0-20) | Optional |
| `n` | int | Number of completions | Optional |
| `user` | string | End-user identifier | Optional |

### o1 Model-Specific Parameters
| Parameter | Type | Description |
|-----------|------|-------------|
| `max_completion_tokens` | int | Use instead of `max_tokens` |
| `reasoning_effort` | string | "low", "medium", "high" |

### Embedding Parameters
| Parameter | Type | Description |
|-----------|------|-------------|
| `model` | string | Model ID (required) |
| `input` | string/array | Text(s) to embed (required) |
| `dimensions` | int | Output dimensions (text-embedding-3 only) |
| `encoding_format` | string | "float" or "base64" |
| `user` | string | End-user identifier |

## Response Structure

### Chat Completion Response
```json
{
  "id": "chatcmpl-xxx",
  "object": "chat.completion",
  "created": 1234567890,
  "model": "gpt-4o-mini",
  "choices": [{
    "index": 0,
    "message": {
      "role": "assistant",
      "content": "Hello!",
      "tool_calls": null
    },
    "finish_reason": "stop",
    "logprobs": null
  }],
  "usage": {
    "prompt_tokens": 10,
    "completion_tokens": 5,
    "total_tokens": 15
  },
  "system_fingerprint": "fp_xxx"
}
```

### Tool Call Response
```json
{
  "choices": [{
    "message": {
      "role": "assistant",
      "content": null,
      "tool_calls": [{
        "id": "call_xxx",
        "type": "function",
        "function": {
          "name": "get_weather",
          "arguments": "{\"location\": \"Paris\"}"
        }
      }]
    },
    "finish_reason": "tool_calls"
  }]
}
```

### Streaming Response (SSE)
```
data: {"id":"chatcmpl-xxx","object":"chat.completion.chunk","choices":[{"delta":{"role":"assistant"}}]}

data: {"id":"chatcmpl-xxx","object":"chat.completion.chunk","choices":[{"delta":{"content":"Hello"}}]}

data: {"id":"chatcmpl-xxx","object":"chat.completion.chunk","choices":[{"delta":{},"finish_reason":"stop"}]}

data: {"id":"chatcmpl-xxx","usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}}

data: [DONE]
```

## Error Handling

| Status Code | Error Type | Description |
|-------------|------------|-------------|
| 401 | `AuthenticationError` | Invalid API key |
| 429 | `RateLimitError` | Rate limit exceeded |
| 400 | `InvalidRequestError` | Malformed request |
| 400 | `ContentFilterError` | Content policy violation |
| 404 | `UnsupportedModelError` | Model not found |
| 500+ | `ProviderAPIError` | Server error |

Docs: https://platform.openai.com/docs/guides/error-codes

## Implementation Files

| File | Purpose |
|------|---------|
| `arcllm/providers/openai_adapter.py` | Main adapter implementation |
| `arcllm/pricing/tables.py` | Pricing data (OPENAI_PRICING) |
| `arcllm/capabilities/tables.py` | Model capabilities (OPENAI_CAPABILITIES) |
| `tests/providers/test_openai.py` | Unit tests |
| `tests/integration/test_openai_integration.py` | Integration tests |
| `tests/test_openai_live.py` | Live API tests |

## How to Update

### When OpenAI Releases New Models:
1. Add model to `OPENAI_PRICING` in `arcllm/pricing/tables.py`
2. Add model to `OPENAI_CAPABILITIES` in `arcllm/capabilities/tables.py`
3. Update `PRICING_VERSION` and `CAPABILITIES_VERSION`
4. Run tests: `pytest tests/test_pricing.py tests/test_capabilities.py`

### When OpenAI Changes API:
1. Update `arcllm/providers/openai_adapter.py`
2. Update request/response handling as needed
3. Run live tests: `pytest tests/test_openai_live.py -v -s`

### When OpenAI Changes Pricing:
1. Check https://openai.com/pricing
2. Update prices in `OPENAI_PRICING`
3. Update `PRICING_VERSION`
4. Run: `pytest tests/test_pricing.py`

## Environment Variables

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | API key (required) |
| `OPENAI_ORGANIZATION` | Organization ID (optional) |
| `OPENAI_PROJECT` | Project ID (optional) |
| `OPENAI_API_BASE` | Custom API base URL (optional) |

## Rate Limits

Rate limits vary by model and tier. Check your current limits at:
https://platform.openai.com/account/limits

Common limits:
- Tier 1: 500 RPM, 10K TPM (gpt-4o-mini)
- Tier 2: 5K RPM, 100K TPM
- Tier 3: 10K RPM, 500K TPM

## Changelog Tracking

When OpenAI makes changes, they typically announce on:
- https://platform.openai.com/docs/changelog
- https://openai.com/blog

Monitor these for:
- New model releases
- Pricing changes
- API updates
- Deprecation notices
