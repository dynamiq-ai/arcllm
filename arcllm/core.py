"""
Core API functions for arcllm.

This module provides the main entry points:
- completion(): Synchronous chat completion
- acompletion(): Asynchronous chat completion
- embedding(): Create embeddings
- stream_chunk_builder(): Build ModelResponse from stream chunks
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, overload

from arcllm.http.async_client import AsyncHTTPClient, AsyncHTTPResponse
from arcllm.http.client import HTTPClient, HTTPResponse
from arcllm.http.sse import AsyncSSEParser, SSEParser
from arcllm.providers.base import (
    Adapter,
    ProviderConfig,
    RequestData,
    get_provider,
    parse_model_string,
)
from arcllm.types import (
    Choice,
    EmbeddingResponse,
    FunctionCall,
    Message,
    ModelResponse,
    StreamChunk,
    StreamingResponse,
    ThinkingBlock,
    ToolCall,
    Usage,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterator

__all__ = [
    "acompletion",
    "aembedding",
    "completion",
    "embedding",
    "stream_chunk_builder",
]

def _apply_module_drop_params(kwargs: dict[str, Any]) -> None:
    """Honor the module-level ``arcllm.drop_params`` toggle when the caller
    didn't pass a per-call ``drop_params=``.

    litellm-compat: this mirrors ``litellm.drop_params = True`` being a
    once-and-done global. Per-call values always win — only used as the
    fallback when ``drop_params`` is absent from kwargs.
    """
    if "drop_params" in kwargs:
        return
    # ``import arcllm`` resolves via ``sys.modules`` after first import; the
    # cost is a single dict lookup per call (~sub-microsecond). Done lazily
    # to avoid a top-level circular import (core ↔ arcllm/__init__).
    import arcllm

    if arcllm.drop_params:
        kwargs["drop_params"] = True


# Pre-computed frozenset of config parameters to exclude from completion kwargs.
# Using frozenset at module level avoids recreating the set on every call.
_CONFIG_PARAMS: frozenset[str] = frozenset(
    {
        "api_key",
        "api_base",
        "base_url",
        "api_version",
        "organization",
        "project",
        "timeout",
        "max_retries",
        "provider",
        "azure_deployment",
        "azure_ad_token",
        "aws_region",
        "aws_access_key_id",
        "aws_secret_access_key",
        "aws_session_token",
        "vertex_project",
        "vertex_location",
        "extra_headers",
    }
)

# Subset for embedding requests (fewer params needed)
_EMBEDDING_CONFIG_PARAMS: frozenset[str] = frozenset(
    {
        "api_key",
        "api_base",
        "base_url",
        "api_version",
        "organization",
        "project",
        "timeout",
        "max_retries",
        "provider",
        "extra_headers",
    }
)

# Global HTTP clients (lazy initialized)
_http_client: HTTPClient | None = None
_async_http_client: AsyncHTTPClient | None = None
_async_client_loop_id: int | None = None  # Track which event loop the client was created for


def _get_http_client() -> HTTPClient:
    """Get or create global HTTP client."""
    global _http_client
    if _http_client is None:
        _http_client = HTTPClient()
    return _http_client


def _get_async_http_client() -> AsyncHTTPClient:
    """
    Get or create global async HTTP client.

    The async client is bound to the current event loop. If the loop changes
    (e.g., between pytest tests), we create a new client to avoid
    "Event loop is closed" errors.
    """
    import asyncio

    global _async_http_client, _async_client_loop_id

    try:
        loop = asyncio.get_running_loop()
        current_loop_id = id(loop)
    except RuntimeError:
        # No running loop - will be created when needed
        current_loop_id = None

    # Create new client if none exists or loop has changed
    if _async_http_client is None or _async_client_loop_id != current_loop_id:
        # Note: We don't close the old client here because:
        # 1. It might still have pending requests
        # 2. Calling close() from sync context is problematic
        # The old client will be garbage collected
        _async_http_client = AsyncHTTPClient()
        _async_client_loop_id = current_loop_id

    return _async_http_client


def _build_provider_config(**kwargs: Any) -> ProviderConfig:
    """Build ProviderConfig from kwargs."""
    return ProviderConfig(
        api_key=kwargs.get("api_key"),
        api_base=kwargs.get("api_base") or kwargs.get("base_url"),
        api_version=kwargs.get("api_version"),
        organization=kwargs.get("organization"),
        project=kwargs.get("project"),
        timeout=kwargs.get("timeout", 60.0),
        max_retries=kwargs.get("max_retries", 3),
        azure_deployment=kwargs.get("azure_deployment"),
        azure_ad_token=kwargs.get("azure_ad_token"),
        aws_region=kwargs.get("aws_region"),
        aws_access_key_id=kwargs.get("aws_access_key_id"),
        aws_secret_access_key=kwargs.get("aws_secret_access_key"),
        aws_session_token=kwargs.get("aws_session_token"),
        vertex_project=kwargs.get("vertex_project"),
        vertex_location=kwargs.get("vertex_location"),
        extra_headers=kwargs.get("extra_headers", {}),
    )


def _get_adapter(model: str, **kwargs: Any) -> tuple[Adapter, str]:
    """
    Get adapter for model string.

    Returns:
        Tuple of (adapter instance, model_id without provider prefix)
    """
    provider_name, model_id = parse_model_string(model)

    # Allow explicit provider override
    if "provider" in kwargs:
        provider_name = kwargs["provider"]

    config = _build_provider_config(**kwargs)
    adapter = get_provider(provider_name, config)

    return adapter, model_id


@overload
def completion(
    *,
    model: str,
    messages: list[dict[str, Any]],
    stream: Literal[False] = ...,
    **kwargs: Any,
) -> ModelResponse: ...


@overload
def completion(
    *,
    model: str,
    messages: list[dict[str, Any]],
    stream: Literal[True],
    **kwargs: Any,
) -> StreamingResponse: ...


def completion(
    *,
    model: str,
    messages: list[dict[str, Any]],
    stream: bool = False,
    **kwargs: Any,
) -> ModelResponse | StreamingResponse:
    """
    Create a chat completion.

    Args:
        model: Model identifier (e.g., "gpt-4", "openai/gpt-4", "anthropic/claude-3")
        messages: List of message dicts in OpenAI format
        stream: Whether to stream the response
        **kwargs: Additional parameters:
            - temperature: Sampling temperature (0-2)
            - top_p: Nucleus sampling parameter
            - max_tokens: Maximum tokens to generate
            - stop: Stop sequences
            - seed: Random seed for reproducibility
            - presence_penalty: Presence penalty (-2 to 2)
            - frequency_penalty: Frequency penalty (-2 to 2)
            - tools: List of tool definitions
            - tool_choice: Tool choice strategy
            - response_format: Response format specification
            - drop_params: If True, silently drop unsupported params
            - api_key: API key (or use env var)
            - api_base: Custom API base URL
            - timeout: Request timeout in seconds
            - stream_options: Stream options (e.g., {"include_usage": True})

    Returns:
        ModelResponse for non-streaming, StreamingResponse for streaming

    Raises:
        ArcLLMError: On any error
    """
    _apply_module_drop_params(kwargs)
    adapter, model_id = _get_adapter(model, **kwargs)

    # Remove config params from kwargs using pre-computed frozenset
    completion_kwargs = {k: v for k, v in kwargs.items() if k not in _CONFIG_PARAMS}

    # Build request
    request = adapter.build_request(
        model=model_id,
        messages=messages,
        stream=stream,
        **completion_kwargs,
    )

    client = _get_http_client()

    if stream:
        return _stream_completion(adapter, model_id, request, client)
    return _sync_completion(adapter, model_id, request, client)


def _sync_completion(
    adapter: Adapter,
    model_id: str,
    request: RequestData,
    client: HTTPClient,
) -> ModelResponse:
    """Execute synchronous completion request."""
    result = client.request(
        request.method,
        request.url,
        headers=request.headers,
        body=request.body,
        timeout=request.timeout,
        stream=False,
    )

    assert isinstance(result, HTTPResponse)

    if result.status_code >= 400:
        raise adapter.parse_error(result.status_code, result.body, result.request_id)

    return adapter.parse_response(result.body, model_id)


def _stream_completion(
    adapter: Adapter,
    model_id: str,
    request: RequestData,
    client: HTTPClient,
) -> StreamingResponse:
    """Execute streaming completion request."""
    result = client.request(
        request.method,
        request.url,
        headers=request.headers,
        body=request.body,
        timeout=request.timeout,
        stream=True,
    )

    # result is an Iterator[bytes] for streaming
    assert not isinstance(result, HTTPResponse)

    def chunk_generator() -> Iterator[StreamChunk]:
        parser = SSEParser()
        for chunk_bytes in result:
            for event in parser.feed(chunk_bytes):
                if event.is_done:
                    continue
                stream_chunk = adapter.parse_stream_event(event.data, model_id)
                if stream_chunk is not None:
                    yield stream_chunk

        # Flush any remaining
        final_event = parser.flush()
        if final_event and not final_event.is_done:
            stream_chunk = adapter.parse_stream_event(final_event.data, model_id)
            if stream_chunk is not None:
                yield stream_chunk

    return StreamingResponse(
        iterator=chunk_generator(),
        model=model_id,
    )


async def acompletion(
    *,
    model: str,
    messages: list[dict[str, Any]],
    stream: bool = False,
    **kwargs: Any,
) -> ModelResponse | AsyncIterator[StreamChunk]:
    """
    Create an async chat completion.

    This is a true async implementation using asyncio - it does not block
    the event loop.

    Args:
        Same as completion()

    Returns:
        ModelResponse for non-streaming, AsyncIterator[StreamChunk] for streaming

    Raises:
        ArcLLMError: On any error
    """
    _apply_module_drop_params(kwargs)
    adapter, model_id = _get_adapter(model, **kwargs)

    # Remove config params from kwargs using pre-computed frozenset
    completion_kwargs = {k: v for k, v in kwargs.items() if k not in _CONFIG_PARAMS}

    # Build request
    request = adapter.build_request(
        model=model_id,
        messages=messages,
        stream=stream,
        **completion_kwargs,
    )

    client = _get_async_http_client()

    if stream:
        return _astream_completion(adapter, model_id, request, client)
    return await _async_completion(adapter, model_id, request, client)


async def _async_completion(
    adapter: Adapter,
    model_id: str,
    request: RequestData,
    client: AsyncHTTPClient,
) -> ModelResponse:
    """Execute async completion request."""
    result = await client.request(
        request.method,
        request.url,
        headers=request.headers,
        body=request.body,
        timeout=request.timeout,
        stream=False,
    )

    assert isinstance(result, AsyncHTTPResponse)

    if result.status_code >= 400:
        raise adapter.parse_error(result.status_code, result.body, result.request_id)

    return adapter.parse_response(result.body, model_id)


async def _astream_completion(
    adapter: Adapter,
    model_id: str,
    request: RequestData,
    client: AsyncHTTPClient,
) -> AsyncIterator[StreamChunk]:
    """Execute async streaming completion request."""
    result = await client.request(
        request.method,
        request.url,
        headers=request.headers,
        body=request.body,
        timeout=request.timeout,
        stream=True,
    )

    # result is an AsyncIterator[bytes] for streaming
    assert not isinstance(result, AsyncHTTPResponse)

    parser = AsyncSSEParser()
    async for event in parser.parse(result):
        if event.is_done:
            continue
        stream_chunk = adapter.parse_stream_event(event.data, model_id)
        if stream_chunk is not None:
            yield stream_chunk


def embedding(
    *,
    model: str,
    input: list[str] | str,
    **kwargs: Any,
) -> EmbeddingResponse:
    """
    Create embeddings for text.

    Args:
        model: Model identifier (e.g., "text-embedding-3-small")
        input: Text or list of texts to embed
        **kwargs: Additional parameters

    Returns:
        EmbeddingResponse with embeddings and usage

    Raises:
        ArcLLMError: On any error
    """
    adapter, model_id = _get_adapter(model, **kwargs)

    # Normalize input to list
    if isinstance(input, str):
        input = [input]

    # Build request using pre-computed frozenset for param filtering
    request = adapter.build_embedding_request(
        model=model_id,
        input=input,
        **{k: v for k, v in kwargs.items() if k not in _EMBEDDING_CONFIG_PARAMS},
    )

    client = _get_http_client()

    result = client.request(
        request.method,
        request.url,
        headers=request.headers,
        body=request.body,
        timeout=request.timeout,
        stream=False,
    )

    assert isinstance(result, HTTPResponse)

    if result.status_code >= 400:
        raise adapter.parse_error(result.status_code, result.body, result.request_id)

    return adapter.parse_embedding_response(result.body, model_id)


async def aembedding(
    *,
    model: str,
    input: list[str] | str,
    **kwargs: Any,
) -> EmbeddingResponse:
    """
    Create embeddings for text (async).

    Args:
        Same as embedding()

    Returns:
        EmbeddingResponse with embeddings and usage

    Raises:
        ArcLLMError: On any error
    """
    adapter, model_id = _get_adapter(model, **kwargs)

    # Normalize input to list
    if isinstance(input, str):
        input = [input]

    # Build request using pre-computed frozenset for param filtering
    request = adapter.build_embedding_request(
        model=model_id,
        input=input,
        **{k: v for k, v in kwargs.items() if k not in _EMBEDDING_CONFIG_PARAMS},
    )

    client = _get_async_http_client()

    result = await client.request(
        request.method,
        request.url,
        headers=request.headers,
        body=request.body,
        timeout=request.timeout,
        stream=False,
    )

    assert isinstance(result, AsyncHTTPResponse)

    if result.status_code >= 400:
        raise adapter.parse_error(result.status_code, result.body, result.request_id)

    return adapter.parse_embedding_response(result.body, model_id)


def stream_chunk_builder(
    chunks: list[StreamChunk],
    messages: list[dict[str, Any]] | None = None,
) -> ModelResponse:
    """
    Build a complete ModelResponse from accumulated stream chunks.

    This helper accumulates content and tool calls from streaming chunks
    into a final response object, similar to LiteLLM's stream_chunk_builder.

    Performance optimizations:
    - Pre-allocates lists where size is known
    - Uses list.append() instead of string concatenation
    - Minimizes dict lookups with local variable caching
    - Avoids repeated attribute access in tight loops

    Args:
        chunks: List of StreamChunk objects from streaming
        messages: Original messages (optional, for context)

    Returns:
        ModelResponse with accumulated content and tool calls
    """
    if not chunks:
        raise ValueError("No chunks provided to stream_chunk_builder")

    # Get metadata from first chunk - cache for speed
    first_chunk = chunks[0]
    response_id = first_chunk.id
    model = first_chunk.model
    created = first_chunk.created
    system_fingerprint = first_chunk.system_fingerprint

    # Track content and tool calls per choice index
    # Use specialized structure for better performance
    choice_roles: dict[int, str | None] = {}
    choice_content: dict[int, list[str]] = {}
    choice_reasoning: dict[int, list[str]] = {}
    # Anthropic-style: per-choice ordered list of (thinking_text, signature)
    # blocks rebuilt from the stream so callers can replay them as input.
    choice_thinking_blocks: dict[int, list[list[str]]] = {}
    choice_tool_calls: dict[
        int, dict[int, list[Any]]
    ] = {}  # idx -> tc_idx -> [id, type, name_parts, arg_parts]
    choice_finish: dict[int, str | None] = {}
    choice_logprobs: dict[int, dict[str, Any] | None] = {}

    # Last usage (if present in any chunk)
    last_usage: Usage | None = None

    # Single pass through chunks
    for chunk in chunks:
        chunk_usage = chunk.usage
        if chunk_usage is not None:
            last_usage = chunk_usage

        for choice in chunk.choices:
            idx = choice.index
            delta = choice.delta

            # Initialize on first encounter
            if idx not in choice_content:
                choice_roles[idx] = None
                choice_content[idx] = []
                choice_reasoning[idx] = []
                choice_thinking_blocks[idx] = []
                choice_tool_calls[idx] = {}
                choice_finish[idx] = None
                choice_logprobs[idx] = None

            # Accumulate delta - direct attribute access
            delta_role = delta.role
            if delta_role:
                choice_roles[idx] = delta_role

            delta_content = delta.content
            if delta_content:
                choice_content[idx].append(delta_content)

            # Reasoning (DeepSeek/GLM/o-series style — flat string deltas).
            delta_reasoning = delta.reasoning_content
            if delta_reasoning:
                choice_reasoning[idx].append(delta_reasoning)

            # Anthropic-style thinking deltas — group by current open block.
            # A new block starts whenever a thinking delta arrives after a
            # signature delta (or first thinking delta of the stream).
            delta_thinking = delta.thinking
            delta_signature = delta.signature
            if delta_thinking is not None or delta_signature is not None:
                blocks = choice_thinking_blocks[idx]
                if not blocks or (blocks and blocks[-1][1]):
                    # Last block is closed (has signature) — start a new one.
                    blocks.append(["", ""])
                if delta_thinking:
                    blocks[-1][0] += delta_thinking
                if delta_signature:
                    blocks[-1][1] = delta_signature

            choice_finish_reason = choice.finish_reason
            if choice_finish_reason:
                choice_finish[idx] = choice_finish_reason

            choice_lp = choice.logprobs
            if choice_lp:
                choice_logprobs[idx] = choice_lp

            # Accumulate tool calls
            delta_tool_calls = delta.tool_calls
            if delta_tool_calls:
                tc_dict = choice_tool_calls[idx]
                for tc_delta in delta_tool_calls:
                    tc_index = tc_delta.get("index", 0)
                    if tc_index not in tc_dict:
                        # [id, type, name_parts, arg_parts]
                        tc_dict[tc_index] = ["", "function", [], []]

                    tc = tc_dict[tc_index]
                    tc_id = tc_delta.get("id")
                    if tc_id:
                        tc[0] = tc_id
                    tc_type = tc_delta.get("type")
                    if tc_type:
                        tc[1] = tc_type
                    tc_func = tc_delta.get("function")
                    if tc_func:
                        func_name = tc_func.get("name")
                        if func_name:
                            tc[2].append(func_name)
                        func_args = tc_func.get("arguments")
                        if func_args:
                            tc[3].append(func_args)

    # Build choices - pre-sort keys once
    sorted_indices = sorted(choice_content.keys())
    choices: list[Choice] = []

    for idx in sorted_indices:
        # Build tool calls if present
        tool_calls: list[ToolCall] | None = None
        tc_dict = choice_tool_calls[idx]
        if tc_dict:
            tool_calls = []
            for tc_idx in sorted(tc_dict.keys()):
                tc = tc_dict[tc_idx]
                tool_calls.append(
                    ToolCall(
                        id=tc[0],
                        type=tc[1],
                        function=FunctionCall(
                            name="".join(tc[2]),
                            arguments="".join(tc[3]),
                        ),
                    )
                )

        # Join content parts efficiently
        content_parts = choice_content[idx]
        content = "".join(content_parts) if content_parts else None

        reasoning_parts = choice_reasoning[idx]
        reasoning_content = "".join(reasoning_parts) if reasoning_parts else None

        thinking_blocks_assembled: list[ThinkingBlock] | None = None
        if choice_thinking_blocks[idx]:
            thinking_blocks_assembled = [
                ThinkingBlock(
                    type="thinking",
                    thinking=text,
                    signature=sig or None,
                )
                for text, sig in choice_thinking_blocks[idx]
                if text or sig
            ] or None
            # Fallback to populate the flat surface when only thinking blocks
            # arrived (Anthropic) — concatenate their text so callers reading
            # ``reasoning_content`` see the same string regardless of provider.
            if reasoning_content is None and thinking_blocks_assembled is not None:
                reasoning_content = (
                    "".join(b.thinking or "" for b in thinking_blocks_assembled) or None
                )

        message = Message(
            role=choice_roles[idx] or "assistant",
            content=content,
            tool_calls=tool_calls or None,
            reasoning_content=reasoning_content,
            thinking_blocks=thinking_blocks_assembled,
        )

        choices.append(
            Choice(
                index=idx,
                message=message,
                finish_reason=choice_finish[idx],
                logprobs=choice_logprobs[idx],
            )
        )

    return ModelResponse(
        id=response_id,
        object="chat.completion",
        created=created,
        model=model,
        choices=choices,
        usage=last_usage,
        system_fingerprint=system_fingerprint,
        model_extra={"usage": last_usage.model_dump() if last_usage else {}},
    )
