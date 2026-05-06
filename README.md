<p align="center">
  <img src="https://raw.githubusercontent.com/dynamiq-ai/arcllm/main/docs/assets/logo.svg" alt="ArcLLM" width="400">
</p>

<h3 align="center">The arc connecting you to every LLM</h3>

<p align="center">
  <strong>Minimal dependencies. Maximum performance. One unified API.</strong>
</p>

<p align="center">
  <a href="https://pypi.org/project/arcllm-sdk/"><img src="https://img.shields.io/pypi/v/arcllm-sdk?color=blue&label=PyPI" alt="PyPI"></a>
  <a href="https://pypi.org/project/arcllm-sdk/"><img src="https://img.shields.io/pypi/pyversions/arcllm-sdk" alt="Python"></a>
  <a href="https://github.com/dynamiq-ai/arcllm/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-blue" alt="License"></a>
  <a href="https://github.com/dynamiq-ai/arcllm/actions"><img src="https://img.shields.io/github/actions/workflow/status/dynamiq-ai/arcllm/ci.yml?branch=main" alt="CI"></a>
</p>

<p align="center">
  <a href="#installation">Installation</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="#supported-providers">Providers</a> •
  <a href="#features">Features</a> •
  <a href="#documentation">Docs</a>
</p>

---

## Why ArcLLM?

ArcLLM ships a single unified, OpenAI-compatible surface across every major LLM provider with a tightly curated runtime footprint:

- **4 runtime deps**: `httpx[http2]`, `aiohttp`, `msgspec`, `orjson` — all chosen for raw speed.
- **OpenAI-compatible API** so existing client code keeps working.
- **Sync + async, streaming, tools, structured output, vision, embeddings** in one library.
- **Built-in cost + capability tracking** for every supported model.

Built for developers who want **speed**, **simplicity**, and **reliability** when working with LLMs.

## Installation

```bash
pip install arcllm-sdk
```

## Quick Start

```python
import arcllm

# Simple completion
response = arcllm.completion(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Hello!"}]
)
print(response.choices[0].message.content)
```

### Streaming

```python
stream = arcllm.completion(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Write a haiku about coding"}],
    stream=True
)
for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

### Async

```python
response = await arcllm.acompletion(
    model="anthropic/claude-sonnet-4-5",
    messages=[{"role": "user", "content": "Explain quantum computing"}]
)
```

### Different providers

```python
# OpenAI
arcllm.completion(model="gpt-4o", messages=messages)

# Anthropic
arcllm.completion(model="anthropic/claude-sonnet-4-5", messages=messages)

# Google Gemini
arcllm.completion(model="gemini/gemini-2.5-pro", messages=messages)

# Groq (ultra-fast inference)
arcllm.completion(model="groq/llama-3.3-70b-versatile", messages=messages)

# Together AI / Fireworks (open-weight flagships: Llama 4, Qwen 3, DeepSeek, Kimi, GLM, MiniMax)
arcllm.completion(model="together_ai/meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8", messages=messages)
arcllm.completion(model="fireworks_ai/accounts/fireworks/models/deepseek-v4-pro", messages=messages)

# Local with Ollama
arcllm.completion(model="ollama/llama3.3", messages=messages)
```

## Supported providers

| Provider | Prefix | Models | Auth |
|----------|--------|--------|------|
| **OpenAI** | `openai/` | GPT-5, GPT-4o, o-series reasoning | `OPENAI_API_KEY` |
| **Anthropic** | `anthropic/` | Claude Opus 4.7, Sonnet 4.6, Haiku 4.5 (incl. extended thinking) | `ANTHROPIC_API_KEY` |
| **Google Gemini** | `gemini/` | Gemini 2.5 / 3.x (with thinking config) | `GEMINI_API_KEY` |
| **Mistral** | `mistral/` | Mistral Large, Medium, Small, Codestral, Pixtral | `MISTRAL_API_KEY` |
| **Cohere** | `cohere/` | Command A, Command R+, Aya Vision, Embed v4 | `COHERE_API_KEY` |
| **Groq** | `groq/` | Llama 3.x / 4.x, GPT-OSS, Qwen 3 | `GROQ_API_KEY` |
| **Together AI** | `together_ai/` | Llama 4, Qwen 3, DeepSeek V4, Kimi, GLM, MiniMax | `TOGETHER_API_KEY` |
| **Fireworks AI** | `fireworks_ai/` | DeepSeek V4 Pro, Kimi K2.6, GLM 5.1, Llama, Qwen | `FIREWORKS_API_KEY` |
| **DeepSeek** | `deepseek/` | DeepSeek V4 Flash + Pro (reasoning + chat) | `DEEPSEEK_API_KEY` |
| **Perplexity** | `perplexity/` | Sonar, Sonar Pro, Sonar Reasoning, Deep Research | `PERPLEXITY_API_KEY` |
| **Ollama** | `ollama/` | Local: Llama, Qwen, Gemma, DeepSeek-R1, Phi | (local server) |
| **Azure** | `azure/` | OpenAI Service + AI Foundry serverless (Phi, Llama, Cohere, Mistral) | `AZURE_OPENAI_API_KEY` |
| **AWS Bedrock** | `bedrock/` | Anthropic, OpenAI GPT-OSS, Llama, Mistral, Cohere, Nova, Titan, AI21 | AWS SigV4 |
| **Google Vertex** | `vertex_ai/` | Gemini + Anthropic Claude + Mistral + Llama on Vertex | OAuth (gcloud / ADC) |
| **Databricks** | `databricks/` | Llama, Claude, Gemini, GPT-5 on Foundation Model APIs | `DATABRICKS_TOKEN` |

## Features

### 🛠️ Tool Calling

```python
tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get weather for a location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "City name"}
            },
            "required": ["location"]
        }
    }
}]

response = arcllm.completion(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "What's the weather in Tokyo?"}],
    tools=tools
)

if response.choices[0].message.tool_calls:
    for tool_call in response.choices[0].message.tool_calls:
        print(f"Call: {tool_call.function.name}({tool_call.function.arguments})")
```

### 📋 Structured Output

```python
response = arcllm.completion(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Generate a user profile"}],
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "user_profile",
            "schema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "age": {"type": "integer"},
                    "interests": {"type": "array", "items": {"type": "string"}}
                },
                "required": ["name", "age"]
            }
        }
    }
)
```

### 🖼️ Vision

```python
response = arcllm.completion(
    model="gpt-4o",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "What's in this image?"},
            {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}}
        ]
    }]
)
```

### 📄 PDF input (Anthropic, Gemini)

```python
response = arcllm.completion(
    model="anthropic/claude-haiku-4-5",
    messages=[{
        "role": "user",
        "content": [
            {"type": "input_file", "file": {
                "data": pdf_base64, "media_type": "application/pdf"
            }},
            {"type": "text", "text": "Summarise this document"},
        ],
    }],
    max_tokens=512,
)
```

### 🧠 Reasoning models (thinking budget + reasoning effort)

```python
# OpenAI o-series + GPT-5 hybrid: reasoning_effort
arcllm.completion(
    model="openai/o4-mini",
    messages=[{"role": "user", "content": "What is 7*8?"}],
    reasoning_effort="medium",
    max_completion_tokens=64,
)
# (passing temperature= here is dropped automatically with a warning —
#  o4-mini rejects temperature, and the capability table knows it)

# Anthropic Claude with extended thinking
arcllm.completion(
    model="anthropic/claude-opus-4-7",
    messages=[{"role": "user", "content": "Solve this hard problem"}],
    thinking_budget=2048,
    max_tokens=4096,
)

# Gemini 2.5+ with thinking config
arcllm.completion(
    model="gemini/gemini-2.5-pro",
    messages=[{"role": "user", "content": "Solve"}],
    thinking_budget=1024,
    include_thoughts=True,
)
```

### 🔎 Citations from grounded providers

```python
# Perplexity Sonar — search is implicit
response = arcllm.completion(
    model="perplexity/sonar-pro",
    messages=[{"role": "user", "content": "Latest news on small models?"}],
)
for c in response.choices[0].message.citations or []:
    print(f"{c.title or '(no title)'}: {c.url}")

# Anthropic + Gemini grounded responses populate the same field, sourced
# from `web_search_tool_result` blocks / `groundingMetadata` respectively.
```

### 🛡️ Built-in provider tools (pass-through)

```python
# Anthropic web search + code execution
arcllm.completion(
    model="anthropic/claude-sonnet-4-5",
    messages=[{"role": "user", "content": "Research arcllm and run a quick demo"}],
    tools=[
        {"type": "web_search_20250305", "name": "web_search"},
        {"type": "code_execution_20250825", "name": "code_execution"},
    ],
    max_tokens=1024,
)

# Gemini Google Search grounding
arcllm.completion(
    model="gemini/gemini-2.5-pro",
    messages=[{"role": "user", "content": "What happened in AI yesterday?"}],
    tools=[{"google_search": {}}],
)
```

### 📊 Embeddings

```python
response = arcllm.embedding(
    model="text-embedding-3-small",
    input=["Hello world", "Goodbye world"]
)
print(f"Dimensions: {len(response.data[0].embedding)}")
```

### 💰 Cost Tracking

```python
response = arcllm.completion(model="gpt-4o", messages=messages)

# Calculate cost
cost = arcllm.completion_cost(response)
print(f"Cost: ${cost:.6f}")

# Or get per-token pricing
input_cost, output_cost = arcllm.cost_per_token(
    model="gpt-4o",
    prompt_tokens=1000,
    completion_tokens=500
)
```

### 🔍 Model capabilities

```python
arcllm.supports_vision("gpt-4o")                       # True
arcllm.supports_pdf_input("claude-sonnet-4-5-20250929") # True
arcllm.supports_tools("gemini-2.5-pro")                # True
arcllm.supports_structured_output("gpt-4o")            # True

arcllm.get_max_tokens("gpt-4o")  # 16384
```

## Error Handling

```python
from arcllm import (
    ArcLLMError,
    AuthenticationError,
    RateLimitError,
    TimeoutError,
)

try:
    response = arcllm.completion(model="gpt-4o", messages=messages)
except AuthenticationError:
    print("Check your API key")
except RateLimitError as e:
    print(f"Rate limited. Retry after {e.retry_after}s")
except TimeoutError:
    print("Request timed out")
except ArcLLMError as e:
    print(f"Error: {e.message}")
```

## Configuration

```python
# Per-request configuration
response = arcllm.completion(
    model="gpt-4o",
    messages=messages,
    api_key="sk-...",           # Override API key
    api_base="https://...",     # Custom endpoint
    timeout=120.0,              # Request timeout
    max_retries=5,              # Retry count
)

# Azure OpenAI
response = arcllm.completion(
    model="azure/my-deployment",
    messages=messages,
    api_base="https://myresource.openai.azure.com",
    api_version="2024-10-21",
)
```

## Migration from LiteLLM

ArcLLM is designed as a drop-in replacement:

```python
# Before
import litellm
response = litellm.completion(model="gpt-4o", messages=messages)

# After
import arcllm
response = arcllm.completion(model="gpt-4o", messages=messages)

# Or alias it
import arcllm as litellm
response = litellm.completion(model="gpt-4o", messages=messages)
```

## Documentation

- [Adding a Provider](docs/ADDING_A_PROVIDER.md)
- [Provider Capabilities](docs/providers/CAPABILITIES.md)
- [Performance Guide](docs/PERF.md)
- [Contributing](CONTRIBUTING.md)

## Maintained by

[Dynamiq AI](https://github.com/dynamiq-ai). Issues and pull requests welcome.

## Why "Arc"?

An **arc** is the shortest path between two points. ArcLLM is the shortest path between your code and any LLM provider—minimal, direct, efficient.

## License

Apache 2.0 - see [LICENSE](LICENSE)

---

<p align="center">
  <sub>Built with ❤️ for developers who value simplicity</sub>
</p>
