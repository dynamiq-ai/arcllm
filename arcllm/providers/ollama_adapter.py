"""
Ollama adapter for arcllm.

Ollama provides a local LLM server with an OpenAI-compatible API.
"""

from __future__ import annotations

import os
import time
from typing import Any

import orjson

from arcllm.exceptions import (
    ArcLLMError,
    ProviderAPIError,
    ResponseParseError,
)
from arcllm.providers.base import (
    COMMON_PARAMS,
    BaseAdapter,
    ProviderConfig,
    RequestData,
    register_provider,
)
from arcllm.types import (
    Choice,
    ChunkChoice,
    ChunkDelta,
    EmbeddingData,
    EmbeddingResponse,
    EmbeddingUsage,
    FunctionCall,
    Message,
    ModelResponse,
    StreamChunk,
    ToolCall,
    Usage,
)

__all__ = ["OllamaAdapter"]


class OllamaAdapter(BaseAdapter):
    """
    Adapter for Ollama local server.

    Ollama supports both its native API and OpenAI-compatible API.
    This adapter uses the OpenAI-compatible endpoint.
    """

    provider_name = "ollama"

    supported_params = COMMON_PARAMS | {
        "num_ctx",  # Context window size
        "num_predict",  # Max tokens to predict
        "repeat_penalty",
        "num_gpu",
        "main_gpu",
        "low_vram",
        "f16_kv",
        "logits_all",
        "vocab_only",
        "use_mmap",
        "use_mlock",
        "num_thread",
        "options",  # Raw Ollama options
    }

    def __init__(self, config: ProviderConfig) -> None:
        super().__init__(config)
        base = config.api_base or os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        self._api_base = base.rstrip("/")

    def _build_headers(self) -> dict[str, str]:
        """Get request headers (no auth required for local Ollama)."""
        headers = {"Content-Type": "application/json"}
        if self.config.extra_headers:
            headers.update(self.config.extra_headers)
        return headers

    def build_request(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        stream: bool = False,
        drop_params: bool = False,
        **kwargs: Any,
    ) -> RequestData:
        """Build Ollama chat request using OpenAI-compatible endpoint."""
        kwargs = self._check_params(drop_params, **kwargs)

        body: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": stream,
        }

        # Map OpenAI params to Ollama options
        options: dict[str, Any] = kwargs.get("options", {})

        if "temperature" in kwargs and kwargs["temperature"] is not None:
            options["temperature"] = kwargs["temperature"]
        if "top_p" in kwargs and kwargs["top_p"] is not None:
            options["top_p"] = kwargs["top_p"]
        if "seed" in kwargs and kwargs["seed"] is not None:
            options["seed"] = kwargs["seed"]
        if kwargs.get("stop"):
            body["stop"] = kwargs["stop"] if isinstance(kwargs["stop"], list) else [kwargs["stop"]]

        # Ollama uses num_predict instead of max_tokens
        if "max_tokens" in kwargs and kwargs["max_tokens"] is not None:
            options["num_predict"] = kwargs["max_tokens"]
        if "num_predict" in kwargs:
            options["num_predict"] = kwargs["num_predict"]
        if "num_ctx" in kwargs:
            options["num_ctx"] = kwargs["num_ctx"]
        if "repeat_penalty" in kwargs:
            options["repeat_penalty"] = kwargs["repeat_penalty"]

        if options:
            body["options"] = options

        # Handle tools (Ollama supports function calling)
        if kwargs.get("tools"):
            body["tools"] = kwargs["tools"]

        # Handle response format
        if kwargs.get("response_format"):
            rf = kwargs["response_format"]
            if rf.get("type") == "json_object" or rf.get("type") == "json_schema":
                body["format"] = "json"

        url = f"{self._api_base}/v1/chat/completions"
        body_bytes = orjson.dumps(body)

        return RequestData(
            method="POST",
            url=url,
            headers=self._get_headers(),
            body=body_bytes,
            timeout=self.config.timeout,
        )

    def parse_response(self, data: bytes, model: str) -> ModelResponse:
        """Parse Ollama chat response."""
        try:
            resp = orjson.loads(data)
        except (orjson.JSONDecodeError, UnicodeDecodeError) as e:
            raise ResponseParseError(
                f"Failed to parse response JSON: {e}",
                provider=self.provider_name,
                raw_data=data,
            ) from e

        # Cache timestamp once for this response
        now = int(time.time())

        # Ollama uses OpenAI-compatible format
        choices: list[Choice] = []
        for choice_data in resp.get("choices", []):
            message_data = choice_data.get("message", {})

            # Parse tool calls if present
            tool_calls: list[ToolCall] | None = None
            if message_data.get("tool_calls"):
                tool_calls = []
                for tc in message_data["tool_calls"]:
                    func_data = tc.get("function", {})
                    tool_calls.append(
                        ToolCall(
                            id=tc.get("id", ""),
                            type=tc.get("type", "function"),
                            function=FunctionCall(
                                name=func_data.get("name", ""),
                                arguments=func_data.get("arguments", ""),
                            ),
                        )
                    )

            message = Message(
                role=message_data.get("role", "assistant"),
                content=message_data.get("content"),
                tool_calls=tool_calls,
            )

            choices.append(
                Choice(
                    index=choice_data.get("index", 0),
                    message=message,
                    finish_reason=choice_data.get("finish_reason"),
                )
            )

        # Parse usage
        usage: Usage | None = None
        usage_data = resp.get("usage")
        if usage_data:
            usage = Usage(
                prompt_tokens=usage_data.get("prompt_tokens", 0),
                completion_tokens=usage_data.get("completion_tokens", 0),
                total_tokens=usage_data.get("total_tokens", 0),
            )

        return ModelResponse(
            id=resp.get("id", f"ollama-{now}"),
            object="chat.completion",
            created=resp.get("created", now),
            model=resp.get("model", model),
            choices=choices,
            usage=usage,
            model_extra={"usage": usage.model_dump() if usage else {}},
        )

    def parse_stream_event(self, data: str, model: str) -> StreamChunk | None:
        """Parse Ollama streaming event."""
        data = data.strip()
        if not data or data == "[DONE]":
            return None

        try:
            event = orjson.loads(data)
        except orjson.JSONDecodeError:
            return None

        # Cache timestamp once for this event
        now = int(time.time())
        choices: list[ChunkChoice] = []
        for choice_data in event.get("choices", []):
            delta_data = choice_data.get("delta", {})

            # Parse tool call deltas
            tool_calls: list[dict[str, Any]] | None = None
            if "tool_calls" in delta_data:
                tool_calls = delta_data["tool_calls"]

            delta = ChunkDelta(
                role=delta_data.get("role"),
                content=delta_data.get("content"),
                tool_calls=tool_calls,
            )

            choices.append(
                ChunkChoice(
                    index=choice_data.get("index", 0),
                    delta=delta,
                    finish_reason=choice_data.get("finish_reason"),
                )
            )

        return StreamChunk(
            id=event.get("id", ""),
            object="chat.completion.chunk",
            created=event.get("created", now),
            model=event.get("model", model),
            choices=choices,
        )

    def parse_error(
        self,
        status_code: int,
        data: bytes,
        request_id: str | None = None,
    ) -> ArcLLMError:
        """Parse Ollama error response."""
        try:
            error_data = orjson.loads(data)
            message = error_data.get("error", "Unknown error")
        except (orjson.JSONDecodeError, UnicodeDecodeError):
            message = data.decode("utf-8", errors="replace")

        return ProviderAPIError(
            message,
            provider=self.provider_name,
            status_code=status_code,
            request_id=request_id,
        )

    def build_embedding_request(
        self,
        *,
        model: str,
        input: list[str],
        **kwargs: Any,
    ) -> RequestData:
        """Build Ollama embedding request."""
        # Ollama uses a different endpoint for embeddings
        body: dict[str, Any] = {
            "model": model,
            "input": input,
        }

        url = f"{self._api_base}/v1/embeddings"
        body_bytes = orjson.dumps(body)

        return RequestData(
            method="POST",
            url=url,
            headers=self._get_headers(),
            body=body_bytes,
            timeout=self.config.timeout,
        )

    def parse_embedding_response(self, data: bytes, model: str) -> EmbeddingResponse:
        """Parse Ollama embedding response."""
        try:
            resp = orjson.loads(data)
        except (orjson.JSONDecodeError, UnicodeDecodeError) as e:
            raise ResponseParseError(
                f"Failed to parse embedding response: {e}",
                provider=self.provider_name,
                raw_data=data,
            ) from e

        embeddings: list[EmbeddingData] = []
        for item in resp.get("data", []):
            embeddings.append(
                EmbeddingData(
                    index=item.get("index", 0),
                    embedding=item.get("embedding", []),
                )
            )

        usage_data = resp.get("usage", {})
        return EmbeddingResponse(
            model=resp.get("model", model),
            data=embeddings,
            usage=EmbeddingUsage(
                prompt_tokens=usage_data.get("prompt_tokens", 0),
                total_tokens=usage_data.get("total_tokens", 0),
            ),
        )


# Register on import
register_provider("ollama", OllamaAdapter)
