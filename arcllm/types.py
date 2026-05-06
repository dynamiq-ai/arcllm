"""
Core type definitions for arcllm.

All types use msgspec.Struct for maximum performance:
- 2.7x faster object creation than dataclasses
- 2.2x faster JSON serialization
- Memory efficient with __slots__ by default
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import msgspec

if TYPE_CHECKING:
    from collections.abc import Iterator


class _DictLike:
    """Litellm-compat mixin: lets ``msgspec.Struct`` types double as
    dict-like records.

    Litellm's response classes inherit from a Pydantic base that
    transparently supports both attribute and item access — call sites
    routinely do ``response["data"]`` *and* ``response.data``
    interchangeably, often inside test fixtures that set fields after
    construction. This mixin gives arcllm's strongly-typed
    ``msgspec.Struct`` types the same surface so a litellm-trained
    caller can keep working unchanged.

    The mixin only forwards key access to the underlying attributes —
    msgspec's type validation still applies to the canonical attribute
    path. ``__setitem__`` calls ``setattr`` directly, which lets
    fixtures populate fields with whatever shape the test wants
    (mirrors litellm's loose typing).
    """

    __slots__ = ()

    def __getitem__(self, key: str) -> Any:
        try:
            return getattr(self, key)
        except AttributeError as exc:
            raise KeyError(key) from exc

    def __setitem__(self, key: str, value: Any) -> None:
        # No type validation: this is the loose-typing escape hatch for
        # tests / litellm-style fixtures. Real provider code uses the
        # canonical Struct constructor and gets full type checking.
        setattr(self, key, value)

    def __contains__(self, key: str) -> bool:
        return hasattr(self, key) and getattr(self, key) is not None

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)


__all__ = [
    "Choice",
    "ChunkChoice",
    "ChunkDelta",
    "Citation",
    "CustomStreamWrapper",
    "Delta",
    "EmbeddingData",
    # Embedding types
    "EmbeddingResponse",
    "EmbeddingUsage",
    "FunctionCall",
    # Image types
    "ImageData",
    "ImageResponse",
    "Message",
    # Response types
    "ModelResponse",
    # Rerank types
    "RerankResponse",
    "RerankResult",
    "StreamChunk",
    "StreamingResponse",
    "ToolCall",
    "Usage",
]


# =============================================================================
# Tool Calling Types
# =============================================================================


class FunctionCall(_DictLike, msgspec.Struct):
    """Function call details within a tool call."""

    name: str
    arguments: str  # JSON string - call parse_arguments() for dict

    def parse_arguments(self) -> dict[str, Any]:
        """Parse the arguments JSON string into a dict. Raises ValueError on invalid JSON."""
        try:
            result: dict[str, Any] = msgspec.json.decode(self.arguments)
            return result
        except msgspec.DecodeError as e:
            raise ValueError(f"Invalid JSON in function arguments: {e}") from e

    def model_dump(self) -> dict[str, Any]:
        """Return dict representation for serialization."""
        return {"name": self.name, "arguments": self.arguments}


class ToolCall(_DictLike, msgspec.Struct):
    """A tool call from the model response."""

    id: str
    type: Literal["function"] = "function"
    function: FunctionCall | None = None

    def model_dump(self) -> dict[str, Any]:
        """Return dict representation for serialization."""
        result: dict[str, Any] = {"id": self.id, "type": self.type}
        if self.function is not None:
            result["function"] = self.function.model_dump()
        return result


class Citation(_DictLike, msgspec.Struct):
    """A single source citation attached to a model response.

    Different providers populate different subsets:

    - Perplexity Sonar: ``url`` always; ``title`` and ``snippet`` when the
      ``search_results`` block is returned (newer responses).
    - Gemini grounding: ``url`` and ``title`` from ``groundingChunks``;
      ``start_index``/``end_index`` from ``groundingSupports`` (segment of the
      assistant content the citation grounds).
    - Anthropic web-search: ``url``, ``title``, ``snippet`` from
      ``web_search_tool_result`` blocks; ``start_index``/``end_index`` from the
      ``citations`` annotation on the assistant text block.
    """

    url: str
    title: str | None = None
    snippet: str | None = None
    start_index: int | None = None
    end_index: int | None = None

    def model_dump(self) -> dict[str, Any]:
        """Return dict representation for serialization."""
        result: dict[str, Any] = {"url": self.url}
        if self.title is not None:
            result["title"] = self.title
        if self.snippet is not None:
            result["snippet"] = self.snippet
        if self.start_index is not None:
            result["start_index"] = self.start_index
        if self.end_index is not None:
            result["end_index"] = self.end_index
        return result


# =============================================================================
# Message Types
# =============================================================================


class Message(_DictLike, msgspec.Struct):
    """A message in a completion response."""

    role: str
    content: str | None = None
    tool_calls: list[ToolCall] | None = None
    function_call: FunctionCall | None = None  # Legacy, prefer tool_calls
    refusal: str | None = None
    # Source citations attached by search-grounded providers (Perplexity,
    # Gemini grounding, Anthropic web-search). ``None`` for non-grounded
    # responses; an empty list means "the provider was asked to ground but
    # returned no sources" (rare).
    citations: list[Citation] | None = None

    def model_dump(self) -> dict[str, Any]:
        """Return dict representation for serialization."""
        result: dict[str, Any] = {"role": self.role}
        if self.content is not None:
            result["content"] = self.content
        if self.tool_calls is not None:
            result["tool_calls"] = [tc.model_dump() for tc in self.tool_calls]
        if self.function_call is not None:
            result["function_call"] = self.function_call.model_dump()
        if self.refusal is not None:
            result["refusal"] = self.refusal
        if self.citations is not None:
            result["citations"] = [c.model_dump() for c in self.citations]
        return result


# =============================================================================
# Usage Types
# =============================================================================


class Usage(_DictLike, msgspec.Struct):
    """Token usage information from the provider.

    Cache-related fields (``cache_read_input_tokens`` /
    ``cache_creation_input_tokens``) are populated by providers that support
    prompt caching (Anthropic, Bedrock-Anthropic, Vertex-Anthropic, and
    Databricks-Claude). ``cache_read_input_tokens`` is the slice of
    ``prompt_tokens`` that was served from cache and billed at the
    ``cached_input_cost_per_million`` rate; ``cache_creation_input_tokens``
    is the slice that was newly written to the cache and billed at a higher
    rate (per Anthropic docs, ~25% above the base input rate). Both default
    to ``None`` for providers that don't expose cache tracking.
    """

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    # Extended fields for providers that report more detail
    prompt_tokens_details: dict[str, Any] | None = None
    completion_tokens_details: dict[str, Any] | None = None
    # Cache token tracking (Anthropic-family providers)
    cache_read_input_tokens: int | None = None
    cache_creation_input_tokens: int | None = None

    def model_dump(self) -> dict[str, Any]:
        """Return dict representation for serialization."""
        result: dict[str, Any] = {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }
        if self.prompt_tokens_details is not None:
            result["prompt_tokens_details"] = self.prompt_tokens_details
        if self.completion_tokens_details is not None:
            result["completion_tokens_details"] = self.completion_tokens_details
        if self.cache_read_input_tokens is not None:
            result["cache_read_input_tokens"] = self.cache_read_input_tokens
        if self.cache_creation_input_tokens is not None:
            result["cache_creation_input_tokens"] = self.cache_creation_input_tokens
        return result


# =============================================================================
# Choice Types
# =============================================================================


class Choice(_DictLike, msgspec.Struct):
    """A single choice in a completion response."""

    index: int
    message: Message
    finish_reason: str | None = None
    logprobs: dict[str, Any] | None = None

    def model_dump(self) -> dict[str, Any]:
        """Return dict representation for serialization."""
        result: dict[str, Any] = {
            "index": self.index,
            "message": self.message.model_dump(),
        }
        if self.finish_reason is not None:
            result["finish_reason"] = self.finish_reason
        if self.logprobs is not None:
            result["logprobs"] = self.logprobs
        return result


# =============================================================================
# Model Response
# =============================================================================


class ModelResponse(_DictLike, msgspec.Struct):
    """
    The unified response from a completion call.

    Compatible with LiteLLM's ModelResponse structure:
    - response.choices[0].message.content
    - response.choices[0].message.tool_calls
    - response.model_extra["usage"]
    """

    id: str
    object: str = "chat.completion"
    created: int = 0
    model: str = ""
    choices: list[Choice] = []
    usage: Usage | None = None
    system_fingerprint: str | None = None
    # Extra fields for debugging/compatibility
    model_extra: dict[str, Any] = {}

    def model_dump(self) -> dict[str, Any]:
        """Return dict representation for serialization."""
        result: dict[str, Any] = {
            "id": self.id,
            "object": self.object,
            "created": self.created,
            "model": self.model,
            "choices": [c.model_dump() for c in self.choices],
        }
        if self.usage is not None:
            result["usage"] = self.usage.model_dump()
        if self.system_fingerprint is not None:
            result["system_fingerprint"] = self.system_fingerprint
        return result


# =============================================================================
# Streaming Types
# =============================================================================


class ChunkDelta(_DictLike, msgspec.Struct):
    """Delta content in a streaming chunk."""

    role: str | None = None
    content: str | None = None
    tool_calls: list[dict[str, Any]] | None = None  # Partial tool call deltas
    function_call: dict[str, Any] | None = None
    # Citations as they arrive in the stream (typically on the final chunk
    # for grounded providers — Perplexity, Gemini grounding, Anthropic
    # web-search). None on intermediate chunks.
    citations: list[Citation] | None = None

    def model_dump(self) -> dict[str, Any]:
        """Return dict representation for serialization."""
        result: dict[str, Any] = {}
        if self.role is not None:
            result["role"] = self.role
        if self.content is not None:
            result["content"] = self.content
        if self.tool_calls is not None:
            result["tool_calls"] = self.tool_calls
        if self.function_call is not None:
            result["function_call"] = self.function_call
        if self.citations is not None:
            result["citations"] = [c.model_dump() for c in self.citations]
        return result


class ChunkChoice(_DictLike, msgspec.Struct):
    """A single choice in a streaming chunk."""

    index: int
    delta: ChunkDelta
    finish_reason: str | None = None
    logprobs: dict[str, Any] | None = None

    def model_dump(self) -> dict[str, Any]:
        """Return dict representation for serialization."""
        result: dict[str, Any] = {
            "index": self.index,
            "delta": self.delta.model_dump(),
        }
        if self.finish_reason is not None:
            result["finish_reason"] = self.finish_reason
        if self.logprobs is not None:
            result["logprobs"] = self.logprobs
        return result


class StreamChunk(_DictLike, msgspec.Struct):
    """A single chunk in a streaming response."""

    id: str
    object: str = "chat.completion.chunk"
    created: int = 0
    model: str = ""
    choices: list[ChunkChoice] = []
    usage: Usage | None = None  # Present in final chunk if include_usage=True
    system_fingerprint: str | None = None

    def model_dump(self) -> dict[str, Any]:
        """Return dict representation for serialization."""
        result: dict[str, Any] = {
            "id": self.id,
            "object": self.object,
            "created": self.created,
            "model": self.model,
            "choices": [c.model_dump() for c in self.choices],
        }
        if self.usage is not None:
            result["usage"] = self.usage.model_dump()
        if self.system_fingerprint is not None:
            result["system_fingerprint"] = self.system_fingerprint
        return result


class StreamingResponse:
    """
    Wrapper for streaming responses that yields StreamChunk objects.

    Implements iterator protocol for sync iteration and provides
    async iteration via __aiter__ when backed by async source.
    """

    __slots__ = ("_chunks", "_iterator", "_model", "_response_id", "_usage")

    def __init__(
        self,
        iterator: Iterator[StreamChunk],
        response_id: str = "",
        model: str = "",
    ) -> None:
        self._iterator = iterator
        self._response_id = response_id
        self._model = model
        self._usage: Usage | None = None
        self._chunks: list[StreamChunk] = []

    def __iter__(self) -> Iterator[StreamChunk]:
        for chunk in self._iterator:
            self._chunks.append(chunk)
            if chunk.usage is not None:
                self._usage = chunk.usage
            yield chunk

    @property
    def usage(self) -> Usage | None:
        """Return usage if available (typically after iteration completes)."""
        return self._usage

    @property
    def response_id(self) -> str:
        return self._response_id

    @property
    def model(self) -> str:
        return self._model


# =============================================================================
# Embedding Types
# =============================================================================


class EmbeddingUsage(_DictLike, msgspec.Struct):
    """Usage information for embedding requests."""

    prompt_tokens: int = 0
    total_tokens: int = 0

    def model_dump(self) -> dict[str, Any]:
        """Return dict representation for serialization."""
        return {
            "prompt_tokens": self.prompt_tokens,
            "total_tokens": self.total_tokens,
        }


class EmbeddingData(_DictLike, msgspec.Struct):
    """A single embedding result."""

    index: int
    embedding: list[float]
    object: str = "embedding"

    def model_dump(self) -> dict[str, Any]:
        """Return dict representation for serialization."""
        return {
            "index": self.index,
            "embedding": self.embedding,
            "object": self.object,
        }


class EmbeddingResponse(_DictLike, msgspec.Struct):
    """Response from an embedding request.

    All fields default — matches the litellm-compat contract where test
    fixtures construct ``EmbeddingResponse()`` with no args and populate
    fields after the fact. Adapters always set every field on real
    responses, so this is purely for caller ergonomics.
    """

    model: str = ""
    data: list[EmbeddingData] = []
    usage: EmbeddingUsage | None = None
    object: str = "list"

    def model_dump(self) -> dict[str, Any]:
        """Return dict representation for serialization."""
        return {
            "model": self.model,
            "data": [d.model_dump() for d in self.data],
            "usage": self.usage.model_dump() if self.usage is not None else None,
            "object": self.object,
        }


# =============================================================================
# Image Types
# =============================================================================


class ImageData(_DictLike, msgspec.Struct):
    """A single generated image returned by the provider.

    Either ``url`` or ``b64_json`` is set, depending on ``response_format``
    on the request. ``revised_prompt`` is populated by DALL-E 3 when the
    model rewrites the user prompt for safety/style; other providers leave
    it ``None``.
    """

    url: str | None = None
    b64_json: str | None = None
    revised_prompt: str | None = None

    def model_dump(self) -> dict[str, Any]:
        result: dict[str, Any] = {}
        if self.url is not None:
            result["url"] = self.url
        if self.b64_json is not None:
            result["b64_json"] = self.b64_json
        if self.revised_prompt is not None:
            result["revised_prompt"] = self.revised_prompt
        return result


class ImageResponse(_DictLike, msgspec.Struct):
    """Response from an image generation / variation / edit request."""

    created: int
    data: list[ImageData]
    model: str = ""

    def model_dump(self) -> dict[str, Any]:
        return {
            "created": self.created,
            "data": [d.model_dump() for d in self.data],
            "model": self.model,
        }


# =============================================================================
# Rerank Types
# =============================================================================


class RerankResult(_DictLike, msgspec.Struct):
    """A single reranked document hit.

    ``index`` is the 0-based position of this document in the ``documents``
    list passed to :func:`arcllm.rerank`. ``relevance_score`` is the
    provider-reported score (typically 0..1, but provider-specific). The
    ``document`` field is populated only when ``return_documents=True`` was
    passed on the request.
    """

    index: int
    relevance_score: float
    document: str | None = None

    def model_dump(self) -> dict[str, Any]:
        result: dict[str, Any] = {"index": self.index, "relevance_score": self.relevance_score}
        if self.document is not None:
            result["document"] = self.document
        return result


class RerankResponse(_DictLike, msgspec.Struct):
    """Response from a rerank request.

    ``results`` is sorted by descending relevance.
    """

    model: str
    results: list[RerankResult]
    id: str = ""

    def model_dump(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "model": self.model,
            "results": [r.model_dump() for r in self.results],
        }


# Litellm-compat aliases: dynamiq's tests + TYPE_CHECKING blocks reference
# names from litellm.utils / litellm. arcllm calls the equivalent types
# ``ChunkDelta`` and ``StreamingResponse``. Adding these aliases here lets
# the migration script swap the import path without renaming the symbol at
# the call sites.
Delta = ChunkDelta
CustomStreamWrapper = StreamingResponse
