# Anthropic Provider

## Overview

arcllm supports Claude models via the Anthropic Messages API.

## Official API Documentation References

> **For AI Coding Agents**: Use these official documentation links when updating the Anthropic adapter.
> Always verify against the latest API documentation before making changes.

### Core API Documentation

| Resource | URL | Description |
|----------|-----|-------------|
| **API Reference** | https://docs.anthropic.com/en/api | Main API documentation |
| **Messages API** | https://docs.anthropic.com/en/api/messages | Create messages endpoint |
| **Streaming** | https://docs.anthropic.com/en/api/messages-streaming | SSE streaming format |
| **Errors** | https://docs.anthropic.com/en/api/errors | Error codes and handling |

### Model Information

| Resource | URL | Description |
|----------|-----|-------------|
| **Models** | https://docs.anthropic.com/en/docs/about-claude/models | Available models and capabilities |
| **Pricing** | https://www.anthropic.com/pricing | Current pricing per model |
| **Model Versions** | https://docs.anthropic.com/en/docs/about-claude/models#model-names | Model ID naming conventions |

### Feature Documentation

| Feature | URL | Description |
|---------|-----|-------------|
| **Tool Use** | https://docs.anthropic.com/en/docs/build-with-claude/tool-use | Function/tool calling |
| **Vision** | https://docs.anthropic.com/en/docs/build-with-claude/vision | Image input support |
| **Prompt Caching** | https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching | Caching for cost reduction |
| **PDF Support** | https://docs.anthropic.com/en/docs/build-with-claude/pdf-support | PDF document handling |
| **Extended Thinking** | https://docs.anthropic.com/en/docs/build-with-claude/extended-thinking | Chain-of-thought reasoning |

### API Versioning

| Resource | URL | Description |
|----------|-----|-------------|
| **Versioning** | https://docs.anthropic.com/en/api/versioning | API version headers |
| **Changelog** | https://docs.anthropic.com/en/release-notes/api | API changes and updates |

## Configuration

```python
import arcllm

# Via environment variable (recommended)
# export ANTHROPIC_API_KEY="sk-ant-..."

response = arcllm.completion(
    model="anthropic/claude-3-5-sonnet-latest",
    messages=[{"role": "user", "content": "Hello!"}]
)

# Or with explicit API key
response = arcllm.completion(
    model="anthropic/claude-3-5-sonnet-latest",
    messages=[{"role": "user", "content": "Hello!"}],
    api_key="sk-ant-..."
)

# With prefix inference (when model name is unique)
response = arcllm.completion(
    model="claude-3-5-sonnet-latest",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

## Supported Models

### Claude 4.5 Series (Current Flagship - November 2025)

| Model ID | Context | Max Output | Vision | Tools | PDF | JSON Mode |
|----------|---------|------------|--------|-------|-----|-----------|
| `claude-4-5-opus-20251120` | 500K | 32,768 | ✅ | ✅ | ✅ | ✅ |
| `claude-4-5-opus-latest` | 500K | 32,768 | ✅ | ✅ | ✅ | ✅ |
| `claude-4-5-sonnet-20251015` | 500K | 16,384 | ✅ | ✅ | ✅ | ✅ |
| `claude-4-5-sonnet-latest` | 500K | 16,384 | ✅ | ✅ | ✅ | ✅ |
| `claude-4-5-haiku-20251201` | 500K | 16,384 | ✅ | ✅ | ✅ | ✅ |
| `claude-4-5-haiku-latest` | 500K | 16,384 | ✅ | ✅ | ✅ | ✅ |

### Claude 4 Series (June 2025)

| Model ID | Context | Max Output | Vision | Tools | PDF | JSON Mode |
|----------|---------|------------|--------|-------|-----|-----------|
| `claude-4-opus-20250615` | 300K | 16,384 | ✅ | ✅ | ✅ | ✅ |
| `claude-4-opus-latest` | 300K | 16,384 | ✅ | ✅ | ✅ | ✅ |
| `claude-4-sonnet-20250601` | 300K | 16,384 | ✅ | ✅ | ✅ | ✅ |
| `claude-4-sonnet-latest` | 300K | 16,384 | ✅ | ✅ | ✅ | ✅ |
| `claude-4-haiku-20250701` | 300K | 8,192 | ✅ | ✅ | ✅ | ✅ |
| `claude-4-haiku-latest` | 300K | 8,192 | ✅ | ✅ | ✅ | ✅ |

### Claude 3.5 Series (DEPRECATED - retiring March 2026)

| Model ID | Context | Max Output | Vision | Tools | PDF | JSON Mode |
|----------|---------|------------|--------|-------|-----|-----------|
| `claude-3-5-sonnet-20241022` | 200K | 8,192 | ✅ | ✅ | ✅ | ❌ |
| `claude-3-5-sonnet-latest` | 200K | 8,192 | ✅ | ✅ | ✅ | ❌ |
| `claude-3-5-haiku-20241022` | 200K | 8,192 | ✅ | ✅ | ❌ | ❌ |
| `claude-3-5-haiku-latest` | 200K | 8,192 | ✅ | ✅ | ❌ | ❌ |

### Claude 3 Series (DEPRECATED)

| Model ID | Context | Max Output | Vision | Tools |
|----------|---------|------------|--------|-------|
| `claude-3-opus-20240229` | 200K | 4,096 | ✅ | ✅ |
| `claude-3-opus-latest` | 200K | 4,096 | ✅ | ✅ |
| `claude-3-sonnet-20240229` | 200K | 4,096 | ✅ | ✅ |
| `claude-3-haiku-20240307` | 200K | 4,096 | ✅ | ✅ |

### Legacy Models (DEPRECATED - end of life)

| Model ID | Context | Max Output | Vision | Tools |
|----------|---------|------------|--------|-------|
| `claude-2.1` | 200K | 4,096 | ❌ | ❌ |

## Feature Support Matrix

| Feature | Supported | Notes |
|---------|-----------|-------|
| Streaming | ✅ | Full SSE support |
| Tool Calling | ✅ | Claude 3+ only, converted to Anthropic format |
| Vision | ✅ | Claude 3+ models, base64 and URL images |
| PDF Input | ✅ | Claude 3.5 Sonnet only |
| System Prompts | ✅ | Extracted to separate `system` parameter |
| Usage Reporting | ✅ | Input/output tokens in response |
| Prompt Caching | ✅ | Via `anthropic-beta` header |
| Structured Output | ⚠️ | No native JSON mode - use system prompts |
| Embeddings | ❌ | Not available |

## API Request/Response Mapping

### Request Format Conversion

arcllm converts OpenAI-style requests to Anthropic format:

```python
# OpenAI-style (what you write)
{
    "model": "claude-3-5-sonnet-latest",
    "messages": [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hello!"}
    ],
    "max_tokens": 100,
    "stop": ["END"]
}

# Anthropic format (what's sent)
{
    "model": "claude-3-5-sonnet-latest",
    "system": "You are helpful.",  # Extracted from messages
    "messages": [
        {"role": "user", "content": "Hello!"}
    ],
    "max_tokens": 100,
    "stop_sequences": ["END"]  # Renamed parameter
}
```

### Response Format Mapping

```python
# Anthropic response
{
    "id": "msg_123",
    "type": "message",
    "role": "assistant",
    "content": [{"type": "text", "text": "Hello!"}],
    "stop_reason": "end_turn",
    "usage": {"input_tokens": 10, "output_tokens": 5}
}

# Mapped to OpenAI-compatible ModelResponse
response.choices[0].message.content  # "Hello!"
response.choices[0].finish_reason    # "stop" (mapped from "end_turn")
response.usage.prompt_tokens         # 10
response.usage.completion_tokens     # 5
```

### Stop Reason Mapping

| Anthropic | OpenAI-style |
|-----------|--------------|
| `end_turn` | `stop` |
| `max_tokens` | `length` |
| `stop_sequence` | `stop` |
| `tool_use` | `tool_calls` |

## Tool Calling

### Format Conversion

arcllm converts OpenAI tool format to Anthropic format automatically:

```python
# OpenAI format (what you write)
tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get weather for a location",
        "parameters": {
            "type": "object",
            "properties": {"location": {"type": "string"}},
            "required": ["location"]
        }
    }
}]

# Anthropic format (converted internally)
# {
#     "name": "get_weather",
#     "description": "Get weather for a location",
#     "input_schema": {
#         "type": "object",
#         "properties": {"location": {"type": "string"}},
#         "required": ["location"]
#     }
# }
```

### Tool Choice Mapping

| OpenAI | Anthropic | Description |
|--------|-----------|-------------|
| `"auto"` | `{"type": "auto"}` | Model decides |
| `"required"` | `{"type": "any"}` | Must use a tool |
| `"none"` | (omitted) | Don't use tools |
| `{"function": {"name": "X"}}` | `{"type": "tool", "name": "X"}` | Force specific tool |

### Tool Response Format

```python
# Anthropic tool use response
{
    "content": [
        {"type": "text", "text": "Let me check."},
        {
            "type": "tool_use",
            "id": "toolu_123",
            "name": "get_weather",
            "input": {"location": "Paris"}
        }
    ]
}

# Converted to OpenAI-compatible format
message.content = "Let me check."
message.tool_calls = [
    ToolCall(
        id="toolu_123",
        type="function",
        function=FunctionCall(
            name="get_weather",
            arguments='{"location": "Paris"}'  # JSON string
        )
    )
]
```

## Streaming Events

### Event Types

| Event Type | Description | Maps To |
|------------|-------------|---------|
| `message_start` | Initial message metadata | `StreamChunk` with role |
| `content_block_start` | Start of text/tool block | `StreamChunk` |
| `content_block_delta` | Content chunk | `StreamChunk` with delta |
| `message_delta` | Final metadata, stop reason | `StreamChunk` with finish_reason |
| `message_stop` | End of stream | `None` (ignored) |

### Streaming Example

```python
stream = arcllm.completion(
    model="claude-3-5-sonnet-latest",
    messages=[{"role": "user", "content": "Hello"}],
    stream=True
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
    if chunk.choices[0].finish_reason:
        print(f"\n[Finished: {chunk.choices[0].finish_reason}]")
```

## Vision Support

### Supported Image Formats

- **Base64**: JPEG, PNG, GIF, WebP
- **URL**: Direct image URLs (Claude 3+)

### Image Input Example

```python
# Base64 encoded image
response = arcllm.completion(
    model="claude-3-5-sonnet-latest",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "What's in this image?"},
            {
                "type": "image_url",
                "image_url": {
                    "url": "data:image/png;base64,iVBORw0KGgo..."
                }
            }
        ]
    }]
)
```

### Image Conversion

```python
# OpenAI image format (what you write)
{
    "type": "image_url",
    "image_url": {"url": "data:image/png;base64,ABC123"}
}

# Anthropic format (converted internally)
{
    "type": "image",
    "source": {
        "type": "base64",
        "media_type": "image/png",
        "data": "ABC123"
    }
}
```

## Error Handling

### Error Code Mapping

| Status | Anthropic Error | arcllm Exception |
|--------|-----------------|------------------|
| 401 | `authentication_error` | `AuthenticationError` |
| 400 | `invalid_request_error` | `InvalidRequestError` |
| 404 | `not_found_error` | `UnsupportedModelError` |
| 429 | `rate_limit_error` | `RateLimitError` |
| 500+ | `api_error` | `ProviderAPIError` |

### Error Response Format

```python
# Anthropic error response
{
    "error": {
        "type": "authentication_error",
        "message": "Invalid API Key"
    }
}

# Accessible via exception
try:
    response = arcllm.completion(...)
except AuthenticationError as e:
    print(e.message)      # "Invalid API Key"
    print(e.provider)     # "anthropic"
    print(e.status_code)  # 401
```

## Headers

### Required Headers

```python
headers = {
    "x-api-key": "sk-ant-...",          # API key
    "Content-Type": "application/json",
    "anthropic-version": "2023-06-01",   # API version
}
```

### Optional Headers

```python
# Beta features
headers["anthropic-beta"] = "prompt-caching-2024-07-31"
```

## Important Notes

### `max_tokens` is Required

Anthropic requires `max_tokens` parameter. arcllm defaults to 4096 if not specified:

```python
# These are equivalent
response = arcllm.completion(model="claude-3-5-sonnet-latest", ...)
response = arcllm.completion(model="claude-3-5-sonnet-latest", max_tokens=4096, ...)
```

### JSON Mode Limitation

Anthropic does not support native `response_format={"type": "json_object"}`. To get JSON output, use a system prompt:

```python
response = arcllm.completion(
    model="claude-3-5-sonnet-latest",
    messages=[
        {"role": "system", "content": "Always respond with valid JSON only. No text outside JSON."},
        {"role": "user", "content": "Give me a person object with name and age"}
    ]
)
```

### Tool Call IDs

Anthropic uses `toolu_xxx` format for tool call IDs, which arcllm preserves:

```python
tool_call.id  # "toolu_01234567890abcdef"
```

## Testing

### Integration Test Model

- **Recommended**: `claude-3-5-haiku-20241022`
- **Why**: Fast, affordable, supports all features (tools, streaming, vision)

### Running Tests

```bash
# Set API key
export ANTHROPIC_API_KEY="sk-ant-..."

# Run unit tests
pytest tests/providers/test_anthropic.py -v

# Run integration tests
pytest tests/integration/test_anthropic_integration.py -v

# Run comprehensive model tests
python examples/test_anthropic_models.py
```

## Updating This Adapter

When Anthropic releases API updates:

1. **Check the changelog**: https://docs.anthropic.com/en/release-notes/api
2. **Update model lists** in `arcllm/capabilities/tables.py` and `arcllm/pricing/tables.py`
3. **Update API version** if needed in `AnthropicAdapter.__init__`
4. **Test with integration tests** before deploying
5. **Update this documentation** with any new features or changes

### Key Files to Update

| File | What to Update |
|------|----------------|
| `arcllm/providers/anthropic_adapter.py` | API logic, headers, conversions |
| `arcllm/capabilities/tables.py` | `ANTHROPIC_CAPABILITIES` dict |
| `arcllm/pricing/tables.py` | `ANTHROPIC_PRICING` dict |
| `docs/providers/anthropic.md` | This documentation |
| `tests/providers/test_anthropic.py` | Unit tests |
| `tests/integration/test_anthropic_integration.py` | Integration tests |
