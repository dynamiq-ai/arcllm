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

__all__ = [
    "Choice",
    "ChunkChoice",
    "ChunkDelta",
    "EmbeddingData",
    # Embedding types
    "EmbeddingResponse",
    "EmbeddingUsage",
    "FunctionCall",
    "Message",
    # Response types
    "ModelResponse",
    "StreamChunk",
    "StreamingResponse",
    "ToolCall",
    "Usage",
]


# =============================================================================
# Tool Calling Types
# =============================================================================


class FunctionCall(msgspec.Struct):
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


class ToolCall(msgspec.Struct):
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


# =============================================================================
# Message Types
# =============================================================================


class Message(msgspec.Struct):
    """A message in a completion response."""

    role: str
    content: str | None = None
    tool_calls: list[ToolCall] | None = None
    function_call: FunctionCall | None = None  # Legacy, prefer tool_calls
    refusal: str | None = None

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
        return result


# =============================================================================
# Usage Types
# =============================================================================


class Usage(msgspec.Struct):
    """Token usage information from provider."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    # Extended fields for providers that report more detail
    prompt_tokens_details: dict[str, Any] | None = None
    completion_tokens_details: dict[str, Any] | None = None

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
        return result


# =============================================================================
# Choice Types
# =============================================================================


class Choice(msgspec.Struct):
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


class ModelResponse(msgspec.Struct):
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


class ChunkDelta(msgspec.Struct):
    """Delta content in a streaming chunk."""

    role: str | None = None
    content: str | None = None
    tool_calls: list[dict[str, Any]] | None = None  # Partial tool call deltas
    function_call: dict[str, Any] | None = None

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
        return result


class ChunkChoice(msgspec.Struct):
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


class StreamChunk(msgspec.Struct):
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


class EmbeddingUsage(msgspec.Struct):
    """Usage information for embedding requests."""

    prompt_tokens: int = 0
    total_tokens: int = 0

    def model_dump(self) -> dict[str, Any]:
        """Return dict representation for serialization."""
        return {
            "prompt_tokens": self.prompt_tokens,
            "total_tokens": self.total_tokens,
        }


class EmbeddingData(msgspec.Struct):
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


class EmbeddingResponse(msgspec.Struct):
    """Response from an embedding request."""

    model: str
    data: list[EmbeddingData]
    usage: EmbeddingUsage
    object: str = "list"

    def model_dump(self) -> dict[str, Any]:
        """Return dict representation for serialization."""
        return {
            "model": self.model,
            "data": [d.model_dump() for d in self.data],
            "usage": self.usage.model_dump(),
            "object": self.object,
        }
