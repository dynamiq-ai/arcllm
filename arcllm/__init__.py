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

from typing import Any

__version__ = "0.4.9"
__all__ = [
    "APIConnectionError",
    "APIError",
    "ArcLLMError",
    "AuthenticationError",
    "BadRequestError",
    "BudgetExceededError",
    "ChatCompletionAssistantMessage",
    "ChatCompletionAssistantToolCall",
    "ChatCompletionDeltaToolCall",
    "ChatCompletionMessageToolCall",
    "ChatCompletionSystemMessage",
    "ChatCompletionToolMessage",
    "ChatCompletionUserMessage",
    "Choice",
    "Choices",
    "ChunkChoice",
    "ChunkDelta",
    "Citation",
    "ConnectionError",
    "ContentFilterError",
    "ContextWindowExceededError",
    "CustomStreamWrapper",
    "EmbeddingData",
    "EmbeddingResponse",
    "EmbeddingUsage",
    "FileObject",
    "Function",
    "FunctionCall",
    "ImageData",
    "ImageResponse",
    "InternalServerError",
    "InvalidRequestError",
    "Message",
    "ModelResponse",
    "ModelResponseStream",
    "OpenAIMessageContent",
    "ProviderAPIError",
    "RateLimitError",
    "RerankResponse",
    "RerankResult",
    "ResponseParseError",
    "Router",
    "ServiceUnavailableError",
    "StreamChunk",
    "StreamingChoices",
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
    "acreate_file",
    "add_function_to_prompt",
    "aembedding",
    "aimage_edit",
    "aimage_generation",
    "aimage_variation",
    "arerank",
    "completion",
    "audio_cost",
    "completion_cost",
    "cost_per_token",
    "embedding",
    "get_max_tokens",
    "get_model_info",
    "get_model_pricing",
    "get_supported_openai_params",
    "image_cost",
    "image_edit",
    "image_generation",
    "image_variation",
    "install_uvloop",
    "rerank",
    "rerank_cost",
    "stream_chunk_builder",
    "supports_function_calling",
    "supports_pdf_input",
    "supports_response_schema",
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
    supports_response_schema,
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
    ContextWindowExceededError,
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
    audio_cost,
    completion_cost,
    cost_per_token,
    get_model_pricing,
    image_cost,
    rerank_cost,
)

# Rerank surface
from arcllm.rerank import arerank, rerank

# Token counting (heuristic by default; tiktoken-precise with `arcllm-sdk[tokenize]`)
from arcllm.tokens import token_counter

# Providers are lazy-loaded when first accessed.
# Types
from arcllm.types import (
    ChatCompletionAssistantMessage,
    ChatCompletionAssistantToolCall,
    ChatCompletionDeltaToolCall,
    ChatCompletionMessageToolCall,
    ChatCompletionSystemMessage,
    ChatCompletionToolMessage,
    ChatCompletionUserMessage,
    Choice,
    Choices,
    ChunkChoice,
    ChunkDelta,
    Citation,
    CustomStreamWrapper,
    EmbeddingData,
    EmbeddingResponse,
    EmbeddingUsage,
    FileObject,
    Function,
    FunctionCall,
    ImageData,
    ImageResponse,
    Message,
    ModelResponse,
    ModelResponseStream,
    OpenAIMessageContent,
    RerankResponse,
    RerankResult,
    StreamChunk,
    StreamingChoices,
    StreamingResponse,
    ThinkingBlock,
    ToolCall,
    Usage,
)

# ---------------------------------------------------------------------------
# Litellm-compat module attributes.
# ---------------------------------------------------------------------------
#
# These exist purely so litellm-compat callers that mutate
# ``litellm.success_callback`` (or sibling lists) keep working after a
# ``litellm → arcllm`` import swap. arcllm does NOT invoke them — the SDK
# has no callback subsystem. Per-call observability should be done by
# reading ``response.usage`` from the return value of :func:`completion`
# / :func:`acompletion`.
#
# Keeping them as plain ``list`` instances means consumers may safely call
# ``.append()`` / ``.remove()`` / direct assignment without runtime errors.
success_callback: list[Any] = []
_async_success_callback: list[Any] = []
failure_callback: list[Any] = []
callbacks: list[Any] = []

# Module-level toggle that mirrors litellm's ``litellm.drop_params``.
# When ``True`` and no per-call ``drop_params=`` is passed, ``completion()``
# / ``acompletion()`` will silently drop unsupported request kwargs instead
# of raising :class:`arcllm.exceptions.UnsupportedParameterError`. The
# default is ``False`` so existing arcllm consumers see no behavior change.
drop_params: bool = False

# Litellm-compat: ``litellm.add_function_to_prompt = True`` instructs litellm
# to inject function/tool definitions into the system prompt for providers
# that lack native tool-calling (e.g. older Ollama models). arcllm does NOT
# implement this behavior — the flag exists so callers that mutate it at
# import time do not AttributeError. Setting it is a no-op; providers
# without native tool support will simply not advertise tools in the
# request.
add_function_to_prompt: bool = False


async def acreate_file(
    *,
    file: bytes | None = None,
    purpose: str = "assistants",
    custom_llm_provider: str | None = None,
    **_: Any,
) -> Any:
    """Stub for ``litellm.acreate_file``.

    litellm calls this to upload a file (PDF, docx, etc.) to a provider's
    Files API (e.g. OpenAI's ``/v1/files``) and returns an object with an
    ``.id`` attribute. The stub exists so litellm-compat callers that
    expect the symbol keep type-checking and reach a clear runtime
    failure point if they actually invoke it.

    arcllm does not yet implement file upload — text-only and inline-data
    (image_url / video_url) multimodal paths are unaffected.
    """
    raise NotImplementedError(
        "arcllm.acreate_file is not yet implemented. arcllm does not "
        "currently support uploading files to provider Files APIs (e.g. "
        "OpenAI ``/v1/files``). Inline image_url / video_url multimodal "
        "content works without it. Request the feature at "
        "https://github.com/dynamiq-ai/arcllm/issues if you need it."
    )


class Router:
    """Stub for ``litellm.Router``.

    litellm's Router provides client-side load balancing, fallbacks, and
    retry across a list of model deployments. arcllm does not yet
    implement routing — the class exists so code that imports
    ``litellm.Router`` (for type annotations or as a constructor argument)
    keeps loading and type-checking after a ``litellm → arcllm`` swap.

    Constructing it raises ``NotImplementedError`` with a pointer back
    here so the failure mode is local and obvious. Until routing is
    implemented, callers with a real load-balancing need should pass in
    a real ``litellm.Router`` instance.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError(
            "arcllm.Router is not yet implemented. arcllm currently routes "
            "a completion call to a single provider derived from the "
            "``model=`` string (e.g. ``openai/gpt-4o-mini``). If you need "
            "client-side load balancing, fallbacks, or retries across "
            "multiple deployments, keep using ``litellm.Router`` directly. "
            "Request the feature at "
            "https://github.com/dynamiq-ai/arcllm/issues if you need it."
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
