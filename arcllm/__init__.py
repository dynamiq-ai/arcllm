"""
ArcLLM - The arc connecting you to every LLM.

Minimal, curated runtime dependencies (`httpx`, `aiohttp`, `msgspec`, `orjson`).
Maximum performance. One unified, OpenAI-compatible API.

ArcLLM provides a high-performance interface for calling
multiple LLM providers with a unified OpenAI-compatible API.

Basic Usage:
    import arcllm

    # Simple completion
    response = arcllm.completion(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Hello!"}]
    )
    print(response.choices[0].message.content)

    # Async completion
    response = await arcllm.acompletion(
        model="anthropic/claude-3-5-sonnet-latest",
        messages=[{"role": "user", "content": "Hello!"}]
    )

    # Streaming
    stream = arcllm.completion(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Hello!"}],
        stream=True
    )
    for chunk in stream:
        print(chunk.choices[0].delta.content, end="")

    # Embeddings
    embeddings = arcllm.embedding(
        model="text-embedding-3-small",
        input=["Hello world", "Goodbye world"]
    )

Performance Tips:
    # For maximum async performance on Unix, install uvloop separately:
    # pip install uvloop
    #
    # Then at the start of your application:
    import uvloop
    uvloop.install()
    # Or use arcllm.install_uvloop() for automatic detection

Supported Providers:
    - OpenAI (openai/)
    - Azure OpenAI (azure/)
    - Anthropic (anthropic/)
    - Google Gemini (gemini/)
    - Google Vertex AI (vertex_ai/)
    - AWS Bedrock (bedrock/)
    - Mistral (mistral/)
    - Cohere (cohere/)
    - Groq (groq/)
    - Together AI (together_ai/)
    - Fireworks AI (fireworks_ai/)
    - DeepSeek (deepseek/)
    - Perplexity (perplexity/)
    - Databricks (databricks/)
    - Ollama (ollama/)

See README.md for complete documentation.
"""

from __future__ import annotations

__version__ = "0.4.9"
__all__ = [
    "APIConnectionError",
    "APIError",
    "ArcLLMError",
    "AuthenticationError",
    "BadRequestError",
    "BudgetExceededError",
    "Choice",
    "ChunkChoice",
    "ChunkDelta",
    "Citation",
    "ConnectionError",
    "ContentFilterError",
    "CustomStreamWrapper",
    "EmbeddingData",
    "EmbeddingResponse",
    "EmbeddingUsage",
    "FunctionCall",
    "ImageData",
    "ImageResponse",
    "InternalServerError",
    "InvalidRequestError",
    "Message",
    "ModelResponse",
    "ProviderAPIError",
    "RateLimitError",
    "RerankResponse",
    "RerankResult",
    "ResponseParseError",
    "ServiceUnavailableError",
    "StreamChunk",
    "StreamingResponse",
    "ThinkingBlock",
    "Timeout",
    "TimeoutError",
    "ToolCall",
    "UnsupportedModelError",
    "UnsupportedParameterError",
    "Usage",
    "__version__",
    "acompletion",
    "aembedding",
    "aimage_edit",
    "aimage_generation",
    "aimage_variation",
    "arerank",
    "completion",
    "completion_cost",
    "cost_per_token",
    "embedding",
    "get_max_tokens",
    "get_model_info",
    "get_model_pricing",
    "get_supported_openai_params",
    "image_edit",
    "image_generation",
    "image_variation",
    "install_uvloop",
    "rerank",
    "stream_chunk_builder",
    "supports_function_calling",
    "supports_pdf_input",
    "supports_structured_output",
    "supports_tools",
    "supports_vision",
    "token_counter",
]


# Capabilities
from arcllm.capabilities import (
    get_max_tokens,
    get_model_info,
    get_supported_openai_params,
    supports_function_calling,
    supports_pdf_input,
    supports_structured_output,
    supports_tools,
    supports_vision,
)

# Core API functions
from arcllm.core import (
    acompletion,
    aembedding,
    completion,
    embedding,
    stream_chunk_builder,
)

# Exceptions
from arcllm.exceptions import (
    APIConnectionError,
    APIError,
    ArcLLMError,
    AuthenticationError,
    BadRequestError,
    BudgetExceededError,
    ConnectionError,
    ContentFilterError,
    InternalServerError,
    InvalidRequestError,
    ProviderAPIError,
    RateLimitError,
    ResponseParseError,
    ServiceUnavailableError,
    Timeout,
    TimeoutError,
    UnsupportedModelError,
    UnsupportedParameterError,
)

# Image generation surface
from arcllm.images import (
    aimage_edit,
    aimage_generation,
    aimage_variation,
    image_edit,
    image_generation,
    image_variation,
)

# Pricing
from arcllm.pricing import (
    completion_cost,
    cost_per_token,
    get_model_pricing,
)

# Rerank surface
from arcllm.rerank import arerank, rerank

# Token counting (heuristic by default; tiktoken-precise with `arcllm-sdk[tokenize]`)
from arcllm.tokens import token_counter

# Providers are lazy-loaded when first accessed.
# Types
from arcllm.types import (
    Choice,
    ChunkChoice,
    ChunkDelta,
    Citation,
    CustomStreamWrapper,
    EmbeddingData,
    EmbeddingResponse,
    EmbeddingUsage,
    FunctionCall,
    ImageData,
    ImageResponse,
    Message,
    ModelResponse,
    RerankResponse,
    RerankResult,
    StreamChunk,
    StreamingResponse,
    ThinkingBlock,
    ToolCall,
    Usage,
)


def install_uvloop() -> bool:
    """
    Install uvloop as the default event loop policy for better async performance.

    uvloop is a fast, drop-in replacement for asyncio's event loop.
    It provides ~10-15% improvement for async operations on Unix systems.

    Returns:
        True if uvloop was installed, False if unavailable (e.g., on Windows)

    Example:
        import arcllm
        arcllm.install_uvloop()  # Call once at application startup

        # Then use async operations as normal
        response = await arcllm.acompletion(...)
    """
    try:
        import uvloop

        uvloop.install()
    except ImportError:
        return False
    except Exception:
        return False
    return True
