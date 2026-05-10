<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/dynamiq-ai/arcllm/main/docs/assets/logo-dark.svg">
    <img src="https://raw.githubusercontent.com/dynamiq-ai/arcllm/main/docs/assets/logo-light.svg" alt="arcllm" width="440">
  </picture>
</p>

<h3 align="center">The arc connecting you to every LLM</h3>

<p align="center">
  <strong>Minimal dependencies. Maximum performance. One unified API.</strong>
</p>

<p align="center">
  <a href="https://pypi.org/project/arcllm-sdk/"><img src="https://img.shields.io/pypi/v/arcllm-sdk?color=blue&label=PyPI" alt="PyPI"></a>
  <a href="https://pypi.org/project/arcllm-sdk/"><img src="https://img.shields.io/pypi/pyversions/arcllm-sdk" alt="Python"></a>
  <a href="https://github.com/dynamiq-ai/arcllm/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-blue" alt="License"></a>
  <a href="https://github.com/dynamiq-ai/arcllm/actions/workflows/ci.yml"><img src="https://github.com/dynamiq-ai/arcllm/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
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
- **Drop-in for litellm** — the public surface (`completion`, `acompletion`, exception classes, `ModelResponse`, `EmbeddingResponse`, `Delta`, `token_counter`, `cost_per_token`, `get_model_info`, `get_supported_openai_params`, `image_generation`, `rerank`) matches name-for-name. Most projects swap with a single-import change.

Built for developers who want **speed**, **simplicity**, and **reliability** when working with LLMs.

## Installation

```bash
pip install arcllm-sdk
```

## Migrating from litellm

ArcLLM's public surface mirrors litellm's, so adopting it in an existing
codebase is usually one search-and-replace:

```python
# Before
from litellm import completion, acompletion
from litellm.exceptions import RateLimitError, BadRequestError

# After
from arcllm import completion, acompletion
from arcllm.exceptions import RateLimitError, BadRequestError
```

Submodule paths map as follows:

| litellm path                                | arcllm path                            |
|---                                          |---                                     |
| `from litellm import X`                     | `from arcllm import X`                 |
| `from litellm.exceptions import …`          | `from arcllm.exceptions import …`      |
| `from litellm.types.utils import Delta, ModelResponse, EmbeddingResponse` | `from arcllm.types import Delta, ModelResponse, EmbeddingResponse` |
| `from litellm.utils import supports_pdf_input` | `from arcllm import supports_pdf_input` |
| `import litellm` (then `litellm.X(...)`)    | `import arcllm` (then `arcllm.X(...)`) |

Validated against the open-source [`dynamiq`](https://github.com/dynamiq-ai/dynamiq)
agentic framework: 1148-test unit suite + 986-test integration suite pass
with arcllm in litellm's place. Exception classes accept both arcllm's
keyword-only construction *and* litellm's positional shape — e.g.
`BadRequestError("msg", "gpt-4o", "openai")` resolves correctly via a
`SUPPORTED_PROVIDERS` heuristic, so existing call patterns keep working.

## Quick Start

```python
import arcllm

# Simple completion
response = arcllm.completion(
    model="gpt-5.4-mini",
    messages=[{"role": "user", "content": "Hello!"}]
)
print(response.choices[0].message.content)
```

### Streaming

```python
stream = arcllm.completion(
    model="gpt-5.4-mini",
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

30 provider adapters, grouped by surface. The model prefix you pass to `arcllm.completion(model=...)` is shown in the **Prefix** column.

### First-party APIs

| Provider | Prefix | Highlights |
|---|---|---|
| **OpenAI** | `openai/` | GPT-5 family, GPT-4.1, GPT-4o, o-series reasoning, embeddings |
| **Anthropic** | `anthropic/` | Claude Opus 4.7, Sonnet 4.6, Haiku 4.5 (extended thinking) |
| **Google Gemini** | `gemini/` | Gemini 2.5 / 3.x with thinking config |
| **Mistral** | `mistral/` | Mistral Large/Medium/Small, Codestral, Pixtral, embeddings |
| **Cohere** | `cohere/` | Command A/R+/R, Aya Vision, Embed v4, Rerank v3.5 |
| **DeepSeek** | `deepseek/` | DeepSeek V4 Flash + Pro (chat + reasoner) |
| **xAI** | `xai/` | Grok-4 / 4.1 / 4.20 / 4.3 family + Grok-3 (legacy) |
| **Perplexity** | `perplexity/` | Sonar, Sonar Pro, Sonar Reasoning, Deep Research |
| **Groq** | `groq/` | Llama 3/4, GPT-OSS, Qwen 3 (LPU low-latency) |
| **Together AI** | `together_ai/` | Llama 4, Qwen 3, DeepSeek V4, Kimi, GLM, MiniMax |
| **Fireworks AI** | `fireworks_ai/` | DeepSeek V4 Pro, Kimi K2, GLM 5.1, Llama, Qwen |
| **Cerebras** | `cerebras/` | Llama 3.x, Qwen 3, GPT-OSS on CS-3 wafer-scale |
| **SambaNova** | `sambanova/` | Llama 3.x / Llama 4, DeepSeek, MiniMax on RDU |
| **DeepInfra** | `deepinfra/` | Full open-weights catalog: Llama, Qwen, DeepSeek, Phi, Gemma, Kimi |
| **AI21** | `ai21/` | Jamba 1.5 Large + Mini |
| **Nebius AI** | `nebius/` | Llama 3.x, Qwen 2.5/3, DeepSeek R1/V3, Mistral, Nemotron |
| **OVHcloud** | `ovhcloud/` | Llama 3.x, DeepSeek R1, Mistral, Qwen 3 — European GPU cloud |
| **Z.AI (GLM)** | `zai/` | GLM-4.5 / 4.6 / 5 family by Zhipu AI (incl. vision + reasoning) |
| **Moonshot AI** | `moonshot/` | Kimi K2.5 / K2.6 / K2-thinking (long-context, multimodal) |

### Cloud platforms

| Provider | Prefix | Highlights |
|---|---|---|
| **Azure** | `azure/` | OpenAI Service deployments + AI Foundry (Phi, Llama, Cohere, Mistral) |
| **AWS Bedrock** | `bedrock/` | Anthropic, OpenAI GPT-OSS, Llama, Mistral, Cohere, Nova, Titan, AI21 |
| **Google Vertex** | `vertex_ai/` | Gemini + Anthropic Claude + Mistral + Llama on Vertex |
| **Databricks** | `databricks/` | Llama, Claude, Gemini, GPT-5 on Foundation Model APIs |
| **IBM watsonx** | `watsonx/` | Granite, Llama, Mistral on IBM Cloud (auto IAM-token exchange) |
| **NVIDIA NIM** | `nvidia_nim/` | Llama, Nemotron, Mixtral, Phi on `build.nvidia.com` |

### Gateways, local & custom

| Provider | Prefix | Highlights |
|---|---|---|
| **OpenRouter** | `openrouter/` | Unified gateway over 300+ upstream models |
| **HuggingFace** | `huggingface/` | Hub Inference + Inference Endpoints (chat-completions API) |
| **Ollama** | `ollama/` | Local: Llama, Qwen, Gemma, DeepSeek-R1, Phi (no API key) |
| **Custom** | `custom/` | Any user-supplied OpenAI-compatible HTTP endpoint |

## Authentication

Every provider reads its key from a documented env var. You can also pass `api_key=` per-call to override.

| Provider | Env var(s) | Notes |
|---|---|---|
| OpenAI | `OPENAI_API_KEY` | |
| Anthropic | `ANTHROPIC_API_KEY` | |
| Gemini | `GEMINI_API_KEY` | AI Studio key |
| Mistral | `MISTRAL_API_KEY` | |
| Cohere | `COHERE_API_KEY` | v2 endpoints |
| DeepSeek | `DEEPSEEK_API_KEY` | direct API (`api.deepseek.com`) |
| xAI | `XAI_API_KEY` | |
| Perplexity | `PERPLEXITY_API_KEY` | |
| Groq | `GROQ_API_KEY` | |
| Together AI | `TOGETHER_API_KEY` | |
| Fireworks AI | `FIREWORKS_API_KEY` | |
| Cerebras | `CEREBRAS_API_KEY` | |
| SambaNova | `SAMBANOVA_API_KEY` | |
| DeepInfra | `DEEPINFRA_API_KEY` | |
| AI21 | `AI21_API_KEY` | Jamba family |
| Nebius AI | `NEBIUS_API_KEY` | |
| OVHcloud | `OVHCLOUD_API_KEY` | European AI Endpoints |
| Z.AI (GLM) | `ZAI_API_KEY` | |
| Moonshot AI | `MOONSHOT_API_KEY` | clamp `temperature` to [0, 1]; multimodal arrays only on Kimi vision/video models |
| Azure | `AZURE_OPENAI_API_KEY` | + `api_base` + `api_version` per call |
| AWS Bedrock | `AWS_ACCESS_KEY_ID` + `AWS_SECRET_ACCESS_KEY` | SigV4-signed; honors `AWS_REGION_NAME` / `AWS_SESSION_TOKEN` |
| Vertex AI | OAuth (gcloud ADC) | falls back to `GOOGLE_APPLICATION_CREDENTIALS` |
| Databricks | `DATABRICKS_TOKEN` | + `DATABRICKS_HOST` |
| IBM watsonx | `WATSONX_API_KEY` | raw IBM Cloud key (auto-exchanged for IAM JWT) **or** pre-exchanged JWT. Plus `WATSONX_URL` + `WATSONX_PROJECT_ID` |
| NVIDIA NIM | `NVIDIA_NIM_API_KEY` | |
| OpenRouter | `OPENROUTER_API_KEY` | optional `OPENROUTER_REFERER` + `OPENROUTER_APP_NAME` for app attribution |
| HuggingFace | `HUGGINGFACE_API_KEY` | works against router or custom Inference Endpoint URL |
| Ollama | none | uses local `OLLAMA_API_BASE` (default `http://localhost:11434`) |
| Custom | user-supplied | pass `api_base=` plus optional `api_key=` / `extra_headers={...}` |

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
    model="gpt-5.4-mini",
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
    model="gpt-5.4-mini",
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

Reasoning output is normalised into a single, cross-provider surface on
the response message:

```python
response = arcllm.completion(
    model="anthropic/claude-sonnet-4-5",
    messages=[{"role": "user", "content": "Solve 12 * 7 step by step."}],
    thinking_budget=2048,
    max_tokens=512,
)
msg = response.choices[0].message
print(msg.reasoning_content)   # flat-string CoT, populated for every reasoning provider
print(msg.thinking_blocks)     # Anthropic's structured form (signatures preserved)
```

`reasoning_content` is filled by OpenAI o-series, GPT-5 hybrid, DeepSeek-R1,
GLM-4.5+, Anthropic extended thinking, Gemini 2.5 with `include_thoughts`,
Groq DeepSeek/Qwen, Cerebras Qwen-thinking, Together / Fireworks DeepSeek-R1,
and Moonshot Kimi-thinking. `thinking_blocks` carries Anthropic's structured
blocks (with signatures intact for tool-use round-trips). Streaming deltas
expose the same fields per chunk.

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

### 🔁 Reranking

```python
response = arcllm.rerank(
    model="cohere/rerank-v3.5",
    query="Who created the Python programming language?",
    documents=[
        "Linus Torvalds created the Linux kernel in 1991.",
        "Guido van Rossum created the Python programming language in 1991.",
        "Dennis Ritchie designed the C programming language at Bell Labs.",
    ],
    top_n=2,
)
for r in response.results:
    print(f"#{r.index}  score={r.relevance_score:.3f}  {r.document}")
```

`arcllm.arerank(...)` is the async equivalent. Cohere is the supported
rerank provider; other adapters raise `UnsupportedModelError` when
called through this surface.

### 🖼️ Image generation

```python
# DALL-E 3 / gpt-image-1
img = arcllm.image_generation(
    model="openai/dall-e-3",
    prompt="a teal arc connecting two glowing endpoints, vector art",
    size="1024x1024",
    quality="standard",
)
print(img.data[0].url)

# Variation + edit (multipart) follow the same OpenAI shape
arcllm.image_variation(model="openai/dall-e-2", image=open("orig.png", "rb").read())
arcllm.image_edit(
    model="openai/gpt-image-1",
    image=open("orig.png", "rb").read(),
    mask=open("mask.png", "rb").read(),
    prompt="replace the sky with a starfield",
)
```

`aimage_generation`, `aimage_variation`, `aimage_edit` are async equivalents.

### 🔢 Token counting

```python
n = arcllm.token_counter(
    model="gpt-4o",
    messages=[{"role": "user", "content": "How many tokens?"}],
)
```

Without extras it falls back to a `chars / 4` heuristic and warns once.
For exact counts on OpenAI-family models install with the `tokenize`
extra:

```bash
pip install "arcllm-sdk[tokenize]"   # pulls in tiktoken
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

Pure-Python lookups against the bundled capability + pricing tables.
No network calls.

```python
# Boolean predicates
arcllm.supports_vision("gpt-4o")                          # True
arcllm.supports_pdf_input("claude-sonnet-4-5-20250929")   # True
arcllm.supports_tools("gemini-2.5-pro")                   # True
arcllm.supports_structured_output("gpt-4o")               # True
arcllm.supports_function_calling("openai/o4-mini")        # True (alias of supports_tools)

# Numbers + records
arcllm.get_max_tokens("gpt-4o")           # 16384
arcllm.get_model_pricing("gpt-4o")        # ModelPricing(input_cost_per_million=2.5, ...)
arcllm.get_model_info("gpt-4o")           # full dict (capabilities + pricing)

# Which OpenAI request params does this model accept?
arcllm.get_supported_openai_params("openai/o4-mini")
# -> ['messages', 'max_completion_tokens', 'reasoning_effort', 'tools', ...]
# (drops 'temperature' / 'top_p' / 'stop' for reasoning models that reject them)
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
