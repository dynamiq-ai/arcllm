"""
Anthropic Claude adapter for arcllm.

Anthropic uses a different API format than OpenAI:
- Messages endpoint at /v1/messages
- Different message format (system separate from messages)
- Different tool calling format
- Different streaming format (SSE with different event types)

=============================================================================
OFFICIAL API DOCUMENTATION REFERENCES (for AI coding agents)
=============================================================================

Main API Documentation:
    https://docs.anthropic.com/en/api

Messages API (main endpoint):
    https://docs.anthropic.com/en/api/messages
    - POST /v1/messages
    - Required fields: model, messages, max_tokens
    - Optional: system, temperature, top_p, stop_sequences, stream, tools, etc.

Streaming:
    https://docs.anthropic.com/en/api/messages-streaming
    - Event types: message_start, content_block_start, content_block_delta,
                   message_delta, message_stop
    - SSE format with event/data pairs

Tool Use (Function Calling):
    https://docs.anthropic.com/en/docs/build-with-claude/tool-use
    - Tools defined with name, description, input_schema
    - Tool results sent as user messages with tool_result content type
    - Tool choice: auto, any, tool (specific)

Vision:
    https://docs.anthropic.com/en/docs/build-with-claude/vision
    - Images sent as content blocks with type "image"
    - Supports base64 and URL sources
    - Media types: image/jpeg, image/png, image/gif, image/webp

Models & Pricing:
    https://docs.anthropic.com/en/docs/about-claude/models
    https://www.anthropic.com/pricing

API Versioning:
    https://docs.anthropic.com/en/api/versioning
    - Current stable: 2023-06-01
    - Set via anthropic-version header

Error Handling:
    https://docs.anthropic.com/en/api/errors
    - 400: invalid_request_error
    - 401: authentication_error
    - 404: not_found_error
    - 429: rate_limit_error
    - 500+: api_error

Changelog (check for API updates):
    https://docs.anthropic.com/en/release-notes/api

=============================================================================
"""

from __future__ import annotations

import time
from typing import Any

import orjson

from arcllm.exceptions import (
    ArcLLMError,
    AuthenticationError,
    InvalidRequestError,
    ProviderAPIError,
    RateLimitError,
    ResponseParseError,
    UnsupportedModelError,
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
    EmbeddingResponse,
    FunctionCall,
    Message,
    ModelResponse,
    StreamChunk,
    ToolCall,
    Usage,
)

__all__ = ["AnthropicAdapter"]


class AnthropicAdapter(BaseAdapter):
    """Adapter for Anthropic Claude API."""

    provider_name = "anthropic"

    # Anthropic-specific supported params
    supported_params = COMMON_PARAMS | {
        "system",  # System prompt (separate in Anthropic)
        "metadata",
        "stop_sequences",  # Anthropic calls it stop_sequences
    }

    def __init__(self, config: ProviderConfig) -> None:
        super().__init__(config)
        self._api_base = config.api_base or "https://api.anthropic.com"
        self._api_version = config.api_version or "2023-06-01"

    def _build_headers(self) -> dict[str, str]:
        """Get request headers."""
        api_key = self._get_api_key("ANTHROPIC_API_KEY")
        headers = {
            "x-api-key": api_key,
            "Content-Type": "application/json",
            "anthropic-version": self._api_version,
        }
        # Enable beta features like prompt caching
        headers["anthropic-beta"] = "prompt-caching-2024-07-31"

        if self.config.extra_headers:
            headers.update(self.config.extra_headers)
        return headers

    def _convert_messages(
        self, messages: list[dict[str, Any]]
    ) -> tuple[str | None, list[dict[str, Any]]]:
        """
        Convert OpenAI-style messages to Anthropic format.

        Returns:
            Tuple of (system_prompt, anthropic_messages)
        """
        system_prompt: str | None = None
        anthropic_messages: list[dict[str, Any]] = []

        for msg in messages:
            role: str = msg.get("role", "")
            content: str | list[dict[str, Any]] = msg.get("content", "")

            if role == "system":
                # Anthropic takes system as a separate parameter
                content_str = content if isinstance(content, str) else ""
                if system_prompt:
                    system_prompt += "\n\n" + content_str
                else:
                    system_prompt = content_str
            elif role == "user":
                # Handle potential multimodal content
                if isinstance(content, list):
                    anthropic_content = self._convert_multimodal_content(content)
                    anthropic_messages.append({"role": "user", "content": anthropic_content})
                else:
                    anthropic_messages.append({"role": "user", "content": content})
            elif role == "assistant":
                # Check for tool use in assistant messages
                if msg.get("tool_calls"):
                    # Convert tool calls to Anthropic format
                    content_blocks: list[dict[str, Any]] = []
                    if content:
                        content_blocks.append({"type": "text", "text": content})
                    for tc in msg["tool_calls"]:
                        func = tc.get("function", {})
                        tool_use = {
                            "type": "tool_use",
                            "id": tc.get("id", ""),
                            "name": func.get("name", ""),
                            "input": orjson.loads(func.get("arguments", "{}")),
                        }
                        content_blocks.append(tool_use)
                    anthropic_messages.append({"role": "assistant", "content": content_blocks})
                else:
                    anthropic_messages.append({"role": "assistant", "content": content or ""})
            elif role == "tool":
                # Tool result message
                tool_result: dict[str, Any] = {
                    "type": "tool_result",
                    "tool_use_id": msg.get("tool_call_id", ""),
                    "content": content,
                }
                # Anthropic requires tool results in a user message
                if anthropic_messages and anthropic_messages[-1]["role"] == "user":
                    # Append to existing user message
                    existing: str | list[dict[str, Any]] = anthropic_messages[-1]["content"]
                    if isinstance(existing, list):
                        existing.append(tool_result)
                    else:
                        anthropic_messages[-1]["content"] = [
                            {"type": "text", "text": existing},
                            tool_result,
                        ]
                else:
                    anthropic_messages.append({"role": "user", "content": [tool_result]})

        return system_prompt, anthropic_messages

    def _convert_multimodal_content(self, content: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert OpenAI multimodal content to Anthropic format."""
        anthropic_content: list[dict[str, Any]] = []

        for part in content:
            if part.get("type") == "text":
                anthropic_content.append({"type": "text", "text": part.get("text", "")})
            elif part.get("type") == "image_url":
                image_url = part.get("image_url", {})
                url = image_url.get("url", "")

                if url.startswith("data:"):
                    # Base64 encoded image
                    # Format: data:image/jpeg;base64,....
                    header, data = url.split(",", 1)
                    media_type = header.split(";")[0].split(":")[1]
                    anthropic_content.append(
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": data,
                            },
                        }
                    )
                else:
                    # URL-based image
                    anthropic_content.append(
                        {
                            "type": "image",
                            "source": {
                                "type": "url",
                                "url": url,
                            },
                        }
                    )

        return anthropic_content

    def _convert_tools(self, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert OpenAI tool format to Anthropic format."""
        anthropic_tools: list[dict[str, Any]] = []

        for tool in tools:
            if tool.get("type") == "function":
                func = tool.get("function", {})
                anthropic_tools.append(
                    {
                        "name": func.get("name", ""),
                        "description": func.get("description", ""),
                        "input_schema": func.get(
                            "parameters", {"type": "object", "properties": {}}
                        ),
                    }
                )

        return anthropic_tools

    def build_request(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        stream: bool = False,
        drop_params: bool = False,
        **kwargs: Any,
    ) -> RequestData:
        """Build Anthropic messages request."""
        kwargs = self._check_params(drop_params, **kwargs)

        # Convert messages
        system_prompt, anthropic_messages = self._convert_messages(messages)

        # Build request body
        body: dict[str, Any] = {
            "model": model,
            "messages": anthropic_messages,
            "max_tokens": kwargs.get("max_tokens", 4096),
        }

        if stream:
            body["stream"] = True

        # Add system prompt if present
        if system_prompt or "system" in kwargs:
            body["system"] = kwargs.get("system") or system_prompt

        # Add optional parameters
        if "temperature" in kwargs and kwargs["temperature"] is not None:
            body["temperature"] = kwargs["temperature"]
        if "top_p" in kwargs and kwargs["top_p"] is not None:
            body["top_p"] = kwargs["top_p"]
        if kwargs.get("stop"):
            body["stop_sequences"] = (
                kwargs["stop"] if isinstance(kwargs["stop"], list) else [kwargs["stop"]]
            )
        if kwargs.get("stop_sequences"):
            body["stop_sequences"] = kwargs["stop_sequences"]
        if "metadata" in kwargs:
            body["metadata"] = kwargs["metadata"]

        # Handle tools
        if kwargs.get("tools"):
            body["tools"] = self._convert_tools(kwargs["tools"])
            if kwargs.get("tool_choice"):
                tc = kwargs["tool_choice"]
                if tc == "auto":
                    body["tool_choice"] = {"type": "auto"}
                elif tc == "required":
                    body["tool_choice"] = {"type": "any"}
                elif tc == "none":
                    # Don't send tool_choice, effectively disabling tools
                    pass
                elif isinstance(tc, dict) and "function" in tc:
                    body["tool_choice"] = {"type": "tool", "name": tc["function"]["name"]}

        url = f"{self._api_base}/v1/messages"
        body_bytes = orjson.dumps(body)

        return RequestData(
            method="POST",
            url=url,
            headers=self._get_headers(),
            body=body_bytes,
            timeout=self.config.timeout,
        )

    def parse_response(self, data: bytes, model: str) -> ModelResponse:
        """Parse Anthropic messages response."""
        try:
            resp = orjson.loads(data)
        except (orjson.JSONDecodeError, UnicodeDecodeError) as e:
            raise ResponseParseError(
                f"Failed to parse response JSON: {e}",
                provider=self.provider_name,
                raw_data=data,
            ) from e

        return self._build_model_response(resp, model)

    def _build_model_response(self, resp: dict[str, Any], model: str) -> ModelResponse:
        """Build ModelResponse from Anthropic response."""
        # Cache timestamp once for this response
        now = int(time.time())
        content_blocks = resp.get("content", [])

        # Extract text content and tool uses
        # Use list + join for efficient string building
        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []

        for block in content_blocks:
            if block.get("type") == "text":
                text_parts.append(block.get("text", ""))
            elif block.get("type") == "tool_use":
                tool_calls.append(
                    ToolCall(
                        id=block.get("id", ""),
                        type="function",
                        function=FunctionCall(
                            name=block.get("name", ""),
                            arguments=orjson.dumps(block.get("input", {})).decode(),
                        ),
                    )
                )

        # Join text parts efficiently
        text_content = "".join(text_parts) if text_parts else None

        message = Message(
            role=resp.get("role", "assistant"),
            content=text_content,
            tool_calls=tool_calls if tool_calls else None,
        )

        # Map Anthropic stop reasons to OpenAI format
        stop_reason = resp.get("stop_reason", "")
        finish_reason = {
            "end_turn": "stop",
            "max_tokens": "length",
            "stop_sequence": "stop",
            "tool_use": "tool_calls",
        }.get(stop_reason, stop_reason)

        choice = Choice(
            index=0,
            message=message,
            finish_reason=finish_reason,
        )

        # Parse usage
        usage_data = resp.get("usage", {})
        usage = Usage(
            prompt_tokens=usage_data.get("input_tokens", 0),
            completion_tokens=usage_data.get("output_tokens", 0),
            total_tokens=usage_data.get("input_tokens", 0) + usage_data.get("output_tokens", 0),
        )

        return ModelResponse(
            id=resp.get("id", ""),
            object="chat.completion",
            created=now,
            model=resp.get("model", model),
            choices=[choice],
            usage=usage,
            model_extra={"usage": usage.model_dump()},
        )

    def parse_stream_event(self, data: str, model: str) -> StreamChunk | None:
        """Parse Anthropic streaming event."""
        data = data.strip()
        if not data:
            return None

        try:
            event = orjson.loads(data)
        except orjson.JSONDecodeError:
            # Anthropic sometimes sends non-JSON events
            return None

        # Cache timestamp once for this event
        now = int(time.time())
        event_type = event.get("type", "")

        # Handle different event types
        if event_type == "message_start":
            # Initial message with metadata
            message = event.get("message", {})
            return StreamChunk(
                id=message.get("id", ""),
                object="chat.completion.chunk",
                created=now,
                model=message.get("model", model),
                choices=[
                    ChunkChoice(
                        index=0,
                        delta=ChunkDelta(role="assistant"),
                        finish_reason=None,
                    )
                ],
            )

        if event_type == "content_block_start":
            block = event.get("content_block", {})
            if block.get("type") == "text":
                return StreamChunk(
                    id="",
                    model=model,
                    choices=[
                        ChunkChoice(
                            index=0,
                            delta=ChunkDelta(content=""),
                            finish_reason=None,
                        )
                    ],
                )
            if block.get("type") == "tool_use":
                # Start of tool use
                return StreamChunk(
                    id="",
                    model=model,
                    choices=[
                        ChunkChoice(
                            index=0,
                            delta=ChunkDelta(
                                tool_calls=[
                                    {
                                        "index": event.get("index", 0),
                                        "id": block.get("id", ""),
                                        "type": "function",
                                        "function": {
                                            "name": block.get("name", ""),
                                            "arguments": "",
                                        },
                                    }
                                ]
                            ),
                            finish_reason=None,
                        )
                    ],
                )

        elif event_type == "content_block_delta":
            delta = event.get("delta", {})
            if delta.get("type") == "text_delta":
                return StreamChunk(
                    id="",
                    model=model,
                    choices=[
                        ChunkChoice(
                            index=0,
                            delta=ChunkDelta(content=delta.get("text", "")),
                            finish_reason=None,
                        )
                    ],
                )
            if delta.get("type") == "input_json_delta":
                # Tool argument delta
                return StreamChunk(
                    id="",
                    model=model,
                    choices=[
                        ChunkChoice(
                            index=0,
                            delta=ChunkDelta(
                                tool_calls=[
                                    {
                                        "index": event.get("index", 0),
                                        "function": {
                                            "arguments": delta.get("partial_json", ""),
                                        },
                                    }
                                ]
                            ),
                            finish_reason=None,
                        )
                    ],
                )

        elif event_type == "message_delta":
            # Final delta with stop reason and usage
            delta = event.get("delta", {})
            usage_data = event.get("usage", {})

            stop_reason = delta.get("stop_reason", "")
            finish_reason = {
                "end_turn": "stop",
                "max_tokens": "length",
                "stop_sequence": "stop",
                "tool_use": "tool_calls",
            }.get(stop_reason, stop_reason)

            usage = None
            if usage_data:
                usage = Usage(
                    prompt_tokens=0,  # Not provided in delta
                    completion_tokens=usage_data.get("output_tokens", 0),
                    total_tokens=usage_data.get("output_tokens", 0),
                )

            return StreamChunk(
                id="",
                model=model,
                choices=[
                    ChunkChoice(
                        index=0,
                        delta=ChunkDelta(),
                        finish_reason=finish_reason,
                    )
                ],
                usage=usage,
            )

        elif event_type == "message_stop":
            # End of stream
            return None

        return None

    def parse_error(
        self,
        status_code: int,
        data: bytes,
        request_id: str | None = None,
    ) -> ArcLLMError:
        """Parse Anthropic error response."""
        try:
            error_data = orjson.loads(data)
            error = error_data.get("error", {})
            message = error.get("message", "Unknown error")
            error_type = error.get("type", "")
        except (orjson.JSONDecodeError, UnicodeDecodeError):
            message = data.decode("utf-8", errors="replace")
            error_type = ""

        if status_code == 401:
            return AuthenticationError(
                message,
                provider=self.provider_name,
                status_code=status_code,
                request_id=request_id,
            )
        if status_code == 429:
            return RateLimitError(
                message,
                provider=self.provider_name,
                status_code=status_code,
                request_id=request_id,
            )
        if status_code == 400:
            return InvalidRequestError(
                message,
                provider=self.provider_name,
                status_code=status_code,
                request_id=request_id,
            )
        if status_code == 404:
            return UnsupportedModelError(
                message,
                provider=self.provider_name,
                status_code=status_code,
                request_id=request_id,
            )
        return ProviderAPIError(
            message,
            provider=self.provider_name,
            status_code=status_code,
            request_id=request_id,
            error_type=error_type,
        )

    def build_embedding_request(
        self,
        *,
        model: str,
        input: list[str],
        **kwargs: Any,
    ) -> RequestData:
        """Anthropic does not support embeddings."""
        raise UnsupportedModelError(
            "Anthropic does not provide an embeddings API",
            provider=self.provider_name,
        )

    def parse_embedding_response(self, data: bytes, model: str) -> EmbeddingResponse:
        """Anthropic does not support embeddings."""
        raise UnsupportedModelError(
            "Anthropic does not provide an embeddings API",
            provider=self.provider_name,
        )


# Register on import
register_provider("anthropic", AnthropicAdapter)
