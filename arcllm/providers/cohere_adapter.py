"""
Cohere adapter for arcllm.

Cohere uses a different API format than OpenAI with:
- Different endpoint (/v2/chat for chat completions)
- Different message format
- Different tool calling format
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

__all__ = ["CohereAdapter"]


class CohereAdapter(BaseAdapter):
    """Adapter for Cohere API."""

    provider_name = "cohere"

    supported_params = COMMON_PARAMS | {
        "preamble",
        "connectors",
        "documents",
        "citation_options",
        "safety_mode",
    }

    def __init__(self, config: ProviderConfig) -> None:
        super().__init__(config)
        self._api_base = config.api_base or "https://api.cohere.com/v2"

    def _build_headers(self) -> dict[str, str]:
        """Get request headers."""
        api_key = self._get_api_key("COHERE_API_KEY")
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        if self.config.extra_headers:
            headers.update(self.config.extra_headers)
        return headers

    def _convert_messages(
        self, messages: list[dict[str, Any]]
    ) -> tuple[str | None, list[dict[str, Any]]]:
        """Convert OpenAI messages to Cohere format."""
        system_message: str | None = None
        cohere_messages: list[dict[str, Any]] = []

        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")

            if role == "system":
                if system_message:
                    system_message += "\n\n" + content
                else:
                    system_message = content
            elif role == "user":
                cohere_messages.append({"role": "user", "content": content})
            elif role == "assistant":
                cohere_msg: dict[str, Any] = {"role": "assistant"}
                if content:
                    cohere_msg["content"] = content

                # Handle tool calls
                if msg.get("tool_calls"):
                    tool_calls: list[dict[str, Any]] = []
                    for tc in msg["tool_calls"]:
                        func: dict[str, Any] = tc.get("function", {})
                        tool_calls.append(
                            {
                                "id": tc.get("id", ""),
                                "type": "function",
                                "function": {
                                    "name": func.get("name", ""),
                                    "arguments": func.get("arguments", "{}"),
                                },
                            }
                        )
                    cohere_msg["tool_calls"] = tool_calls

                cohere_messages.append(cohere_msg)
            elif role == "tool":
                cohere_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": msg.get("tool_call_id", ""),
                        "content": content,
                    }
                )

        return system_message, cohere_messages

    def _convert_tools(self, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert OpenAI tools to Cohere format."""
        cohere_tools: list[dict[str, Any]] = []

        for tool in tools:
            if tool.get("type") == "function":
                func = tool.get("function", {})
                cohere_tools.append(
                    {
                        "type": "function",
                        "function": {
                            "name": func.get("name", ""),
                            "description": func.get("description", ""),
                            "parameters": func.get("parameters", {}),
                        },
                    }
                )

        return cohere_tools

    def build_request(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        stream: bool = False,
        drop_params: bool = False,
        **kwargs: Any,
    ) -> RequestData:
        """Build Cohere chat request."""
        kwargs = self._check_params(drop_params, **kwargs)

        system_message, cohere_messages = self._convert_messages(messages)

        body: dict[str, Any] = {
            "model": model,
            "messages": cohere_messages,
            "stream": stream,
        }

        # Add system message as preamble
        if system_message:
            body["preamble"] = system_message

        # Add optional parameters
        if "temperature" in kwargs and kwargs["temperature"] is not None:
            body["temperature"] = kwargs["temperature"]
        if "top_p" in kwargs and kwargs["top_p"] is not None:
            body["p"] = kwargs["top_p"]
        if "max_tokens" in kwargs and kwargs["max_tokens"] is not None:
            body["max_tokens"] = kwargs["max_tokens"]
        if kwargs.get("stop"):
            stop_val: str | list[str] = kwargs["stop"]
            stops: list[str] = stop_val if isinstance(stop_val, list) else [stop_val]
            body["stop_sequences"] = stops
        if "seed" in kwargs and kwargs["seed"] is not None:
            body["seed"] = kwargs["seed"]
        if "frequency_penalty" in kwargs and kwargs["frequency_penalty"] is not None:
            body["frequency_penalty"] = kwargs["frequency_penalty"]
        if "presence_penalty" in kwargs and kwargs["presence_penalty"] is not None:
            body["presence_penalty"] = kwargs["presence_penalty"]

        # Cohere-specific params
        if "preamble" in kwargs:
            body["preamble"] = kwargs["preamble"]
        if "connectors" in kwargs:
            body["connectors"] = kwargs["connectors"]
        if "documents" in kwargs:
            body["documents"] = kwargs["documents"]
        if "safety_mode" in kwargs:
            body["safety_mode"] = kwargs["safety_mode"]

        # Handle tools
        if kwargs.get("tools"):
            body["tools"] = self._convert_tools(kwargs["tools"])

        # Handle response_format
        if kwargs.get("response_format"):
            rf = kwargs["response_format"]
            if rf.get("type") == "json_object":
                body["response_format"] = {"type": "json_object"}
            elif rf.get("type") == "json_schema" and "json_schema" in rf:
                body["response_format"] = {
                    "type": "json_object",
                    "schema": rf["json_schema"].get("schema", {}),
                }

        url = f"{self._api_base}/chat"
        body_bytes = orjson.dumps(body)

        return RequestData(
            method="POST",
            url=url,
            headers=self._get_headers(),
            body=body_bytes,
            timeout=self.config.timeout,
        )

    def parse_response(self, data: bytes, model: str) -> ModelResponse:
        """Parse Cohere chat response."""
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
        """Build ModelResponse from Cohere response."""
        # Cache timestamp once for this response
        now = int(time.time())
        # Cohere v2 returns message directly
        message_data = resp.get("message", {})
        content_list = message_data.get("content", [])

        # Use list + join for efficient string building
        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []

        for item in content_list:
            if item.get("type") == "text":
                text_parts.append(item.get("text", ""))
            elif item.get("type") == "tool_call":
                tc = item
                tool_calls.append(
                    ToolCall(
                        id=tc.get("id", ""),
                        type="function",
                        function=FunctionCall(
                            name=tc.get("name", ""),
                            arguments=orjson.dumps(tc.get("parameters", {})).decode(),
                        ),
                    )
                )

        # Also check for tool_calls at message level
        if message_data.get("tool_calls"):
            for tc in message_data["tool_calls"]:
                func = tc.get("function", {})
                tool_calls.append(
                    ToolCall(
                        id=tc.get("id", ""),
                        type="function",
                        function=FunctionCall(
                            name=func.get("name", ""),
                            arguments=func.get("arguments", "{}"),
                        ),
                    )
                )

        text_content = "".join(text_parts) if text_parts else None
        message = Message(
            role="assistant",
            content=text_content,
            tool_calls=tool_calls if tool_calls else None,
        )

        # Map finish reason
        finish_reason_map = {
            "COMPLETE": "stop",
            "MAX_TOKENS": "length",
            "STOP_SEQUENCE": "stop",
            "TOOL_CALL": "tool_calls",
        }
        finish_reason = finish_reason_map.get(resp.get("finish_reason", ""), "stop")

        choice = Choice(
            index=0,
            message=message,
            finish_reason=finish_reason,
        )

        # Parse usage
        usage_data = resp.get("usage", {})
        billed = usage_data.get("billed_units", {})
        tokens = usage_data.get("tokens", {})

        usage = Usage(
            prompt_tokens=tokens.get("input_tokens", 0) or billed.get("input_tokens", 0),
            completion_tokens=tokens.get("output_tokens", 0) or billed.get("output_tokens", 0),
            total_tokens=(tokens.get("input_tokens", 0) or 0)
            + (tokens.get("output_tokens", 0) or 0),
        )

        return ModelResponse(
            id=resp.get("id", ""),
            object="chat.completion",
            created=now,
            model=model,
            choices=[choice],
            usage=usage,
            model_extra={"usage": usage.model_dump()},
        )

    def parse_stream_event(self, data: str, model: str) -> StreamChunk | None:
        """Parse Cohere streaming event."""
        data = data.strip()
        if not data:
            return None

        try:
            event = orjson.loads(data)
        except orjson.JSONDecodeError:
            return None

        event_type = event.get("type", "")

        if event_type == "message-start":
            return StreamChunk(
                id=event.get("id", ""),
                model=model,
                choices=[
                    ChunkChoice(
                        index=0,
                        delta=ChunkDelta(role="assistant"),
                        finish_reason=None,
                    )
                ],
            )

        if event_type == "content-start":
            return None  # Skip content start markers

        if event_type == "content-delta":
            delta_data = event.get("delta", {})
            message = delta_data.get("message", {})
            content = message.get("content", {})
            text = content.get("text", "")

            return StreamChunk(
                id="",
                model=model,
                choices=[
                    ChunkChoice(
                        index=0,
                        delta=ChunkDelta(content=text),
                        finish_reason=None,
                    )
                ],
            )

        if event_type == "tool-call-start":
            delta_data = event.get("delta", {})
            tc = delta_data.get("tool_call", {})
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
                                    "id": tc.get("id", ""),
                                    "type": "function",
                                    "function": {
                                        "name": tc.get("name", ""),
                                        "arguments": "",
                                    },
                                }
                            ]
                        ),
                        finish_reason=None,
                    )
                ],
            )

        if event_type == "tool-call-delta":
            delta_data = event.get("delta", {})
            tc = delta_data.get("tool_call", {})
            args = tc.get("parameters", "")

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
                                        "arguments": args,
                                    },
                                }
                            ]
                        ),
                        finish_reason=None,
                    )
                ],
            )

        if event_type == "message-end":
            delta_data = event.get("delta", {})
            finish_reason_map = {
                "COMPLETE": "stop",
                "MAX_TOKENS": "length",
                "STOP_SEQUENCE": "stop",
                "TOOL_CALL": "tool_calls",
            }
            finish_reason = finish_reason_map.get(delta_data.get("finish_reason", ""), "stop")

            # Usage in final event
            usage = None
            usage_data = delta_data.get("usage", {})
            if usage_data:
                billed = usage_data.get("billed_units", {})
                tokens = usage_data.get("tokens", {})
                usage = Usage(
                    prompt_tokens=tokens.get("input_tokens", 0) or billed.get("input_tokens", 0),
                    completion_tokens=tokens.get("output_tokens", 0)
                    or billed.get("output_tokens", 0),
                    total_tokens=(tokens.get("input_tokens", 0) or 0)
                    + (tokens.get("output_tokens", 0) or 0),
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

        return None

    def parse_error(
        self,
        status_code: int,
        data: bytes,
        request_id: str | None = None,
    ) -> ArcLLMError:
        """Parse Cohere error response."""
        try:
            error_data = orjson.loads(data)
            message = error_data.get("message", "Unknown error")
        except (orjson.JSONDecodeError, UnicodeDecodeError):
            message = data.decode("utf-8", errors="replace")

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
        )

    def build_embedding_request(
        self,
        *,
        model: str,
        input: list[str],
        **kwargs: Any,
    ) -> RequestData:
        """Build Cohere embedding request."""
        body: dict[str, Any] = {
            "model": model,
            "texts": input,
            "input_type": kwargs.get("input_type", "search_document"),
        }

        if "truncate" in kwargs:
            body["truncate"] = kwargs["truncate"]

        url = f"{self._api_base}/embed"
        body_bytes = orjson.dumps(body)

        return RequestData(
            method="POST",
            url=url,
            headers=self._get_headers(),
            body=body_bytes,
            timeout=self.config.timeout,
        )

    def parse_embedding_response(self, data: bytes, model: str) -> EmbeddingResponse:
        """Parse Cohere embedding response."""
        try:
            resp = orjson.loads(data)
        except (orjson.JSONDecodeError, UnicodeDecodeError) as e:
            raise ResponseParseError(
                f"Failed to parse embedding response: {e}",
                provider=self.provider_name,
                raw_data=data,
            ) from e

        embeddings: list[EmbeddingData] = []
        for i, emb in enumerate(resp.get("embeddings", [])):
            embeddings.append(
                EmbeddingData(
                    index=i,
                    embedding=emb,
                )
            )

        # Cohere provides billed_units for usage
        meta = resp.get("meta", {})
        billed = meta.get("billed_units", {})

        return EmbeddingResponse(
            model=model,
            data=embeddings,
            usage=EmbeddingUsage(
                prompt_tokens=billed.get("input_tokens", 0),
                total_tokens=billed.get("input_tokens", 0),
            ),
        )


# Register on import
register_provider("cohere", CohereAdapter)
