"""
Google Gemini (AI Studio) adapter for arcllm.

Google's Gemini API uses a different format than OpenAI:
- Different endpoint structure
- Different message format (parts-based)
- Different tool calling format
"""

from __future__ import annotations

import os
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

__all__ = ["GeminiAdapter"]


class GeminiAdapter(BaseAdapter):
    """Adapter for Google Gemini (AI Studio) API."""

    provider_name = "gemini"

    supported_params = COMMON_PARAMS | {
        "safety_settings",
        "generation_config",
    }

    def __init__(self, config: ProviderConfig) -> None:
        super().__init__(config)
        self._api_base = config.api_base or "https://generativelanguage.googleapis.com/v1beta"

    def _build_headers(self) -> dict[str, str]:
        """Build request headers (cached after first call)."""
        headers = {"Content-Type": "application/json"}
        if self.config.extra_headers:
            headers.update(self.config.extra_headers)
        return headers

    def _get_api_key(self, env_var: str = "GEMINI_API_KEY", param_key: str = "api_key") -> str:
        """Get API key, checking multiple env vars."""
        key = self.config.api_key
        if not key:
            key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not key:
            raise AuthenticationError(
                "Gemini API key not provided. Set GEMINI_API_KEY or GOOGLE_API_KEY",
                provider=self.provider_name,
            )
        return key

    def _convert_messages(
        self, messages: list[dict[str, Any]]
    ) -> tuple[str | None, list[dict[str, Any]]]:
        """
        Convert OpenAI-style messages to Gemini format.

        Returns:
            Tuple of (system_instruction, gemini_contents)
        """
        system_instruction: str | None = None
        gemini_contents: list[dict[str, Any]] = []

        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")

            if role == "system":
                if system_instruction:
                    system_instruction += "\n\n" + content
                else:
                    system_instruction = content
            elif role == "user":
                parts = self._convert_content_to_parts(content)
                gemini_contents.append({"role": "user", "parts": parts})
            elif role == "assistant":
                assistant_parts: list[dict[str, Any]] = []
                if content:
                    assistant_parts.append({"text": content})
                # Handle tool calls
                if msg.get("tool_calls"):
                    for tc in msg["tool_calls"]:
                        func = tc.get("function", {})
                        assistant_parts.append(
                            {
                                "functionCall": {
                                    "name": func.get("name", ""),
                                    "args": orjson.loads(func.get("arguments", "{}")),
                                }
                            }
                        )
                gemini_contents.append({"role": "model", "parts": assistant_parts})
            elif role == "tool":
                # Tool result
                gemini_contents.append(
                    {
                        "role": "function",
                        "parts": [
                            {
                                "functionResponse": {
                                    "name": msg.get("name", ""),
                                    "response": {"result": content},
                                }
                            }
                        ],
                    }
                )

        return system_instruction, gemini_contents

    def _convert_content_to_parts(
        self, content: str | list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Convert content to Gemini parts format."""
        if isinstance(content, str):
            return [{"text": content}]

        parts: list[dict[str, Any]] = []
        for item in content:
            if item.get("type") == "text":
                parts.append({"text": item.get("text", "")})
            elif item.get("type") == "image_url":
                image_url = item.get("image_url", {})
                url = image_url.get("url", "")
                if url.startswith("data:"):
                    # Base64 image
                    header, data = url.split(",", 1)
                    mime_type = header.split(";")[0].split(":")[1]
                    parts.append(
                        {
                            "inlineData": {
                                "mimeType": mime_type,
                                "data": data,
                            }
                        }
                    )
                else:
                    parts.append(
                        {
                            "fileData": {
                                "fileUri": url,
                            }
                        }
                    )
        return parts

    def _convert_tools(self, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert OpenAI tools to Gemini format."""
        function_declarations: list[dict[str, Any]] = []

        for tool in tools:
            if tool.get("type") == "function":
                func = tool.get("function", {})
                function_declarations.append(
                    {
                        "name": func.get("name", ""),
                        "description": func.get("description", ""),
                        "parameters": func.get("parameters", {}),
                    }
                )

        return [{"functionDeclarations": function_declarations}]

    def build_request(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        stream: bool = False,
        drop_params: bool = False,
        **kwargs: Any,
    ) -> RequestData:
        """Build Gemini generateContent request."""
        kwargs = self._check_params(drop_params, **kwargs)

        system_instruction, contents = self._convert_messages(messages)

        body: dict[str, Any] = {
            "contents": contents,
        }

        if system_instruction:
            body["systemInstruction"] = {"parts": [{"text": system_instruction}]}

        # Generation config
        generation_config: dict[str, Any] = {}
        if "temperature" in kwargs and kwargs["temperature"] is not None:
            generation_config["temperature"] = kwargs["temperature"]
        if "top_p" in kwargs and kwargs["top_p"] is not None:
            generation_config["topP"] = kwargs["top_p"]
        if "max_tokens" in kwargs and kwargs["max_tokens"] is not None:
            generation_config["maxOutputTokens"] = kwargs["max_tokens"]
        if kwargs.get("stop"):
            stop_val: str | list[str] = kwargs["stop"]
            stops: list[str] = stop_val if isinstance(stop_val, list) else [stop_val]
            generation_config["stopSequences"] = stops

        # Handle response_format for JSON mode
        if kwargs.get("response_format"):
            rf = kwargs["response_format"]
            if rf.get("type") == "json_object":
                generation_config["responseMimeType"] = "application/json"
            elif rf.get("type") == "json_schema" and "json_schema" in rf:
                generation_config["responseMimeType"] = "application/json"
                generation_config["responseSchema"] = rf["json_schema"].get("schema", {})

        if generation_config:
            body["generationConfig"] = generation_config

        # Safety settings
        if "safety_settings" in kwargs:
            body["safetySettings"] = kwargs["safety_settings"]

        # Tools
        if kwargs.get("tools"):
            body["tools"] = self._convert_tools(kwargs["tools"])

        api_key = self._get_api_key()
        method = "streamGenerateContent" if stream else "generateContent"
        url = f"{self._api_base}/models/{model}:{method}?key={api_key}"

        if stream:
            url += "&alt=sse"

        body_bytes = orjson.dumps(body)

        return RequestData(
            method="POST",
            url=url,
            headers=self._get_headers(),
            body=body_bytes,
            timeout=self.config.timeout,
        )

    def parse_response(self, data: bytes, model: str) -> ModelResponse:
        """Parse Gemini generateContent response."""
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
        """Build ModelResponse from Gemini response."""
        # Cache timestamp once for this response
        now = int(time.time())
        candidates = resp.get("candidates", [])
        choices: list[Choice] = []

        for i, candidate in enumerate(candidates):
            content = candidate.get("content", {})
            parts = content.get("parts", [])

            # Use list + join for efficient string building
            text_parts: list[str] = []
            tool_calls: list[ToolCall] = []

            for part in parts:
                if "text" in part:
                    text_parts.append(part["text"])
                elif "functionCall" in part:
                    fc = part["functionCall"]
                    tool_calls.append(
                        ToolCall(
                            id=f"call_{i}_{len(tool_calls)}",
                            type="function",
                            function=FunctionCall(
                                name=fc.get("name", ""),
                                arguments=orjson.dumps(fc.get("args", {})).decode(),
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
                "STOP": "stop",
                "MAX_TOKENS": "length",
                "SAFETY": "content_filter",
                "RECITATION": "content_filter",
                "OTHER": "stop",
            }
            finish_reason = finish_reason_map.get(candidate.get("finishReason", ""), "stop")

            choices.append(
                Choice(
                    index=i,
                    message=message,
                    finish_reason=finish_reason,
                )
            )

        # Parse usage
        usage_metadata = resp.get("usageMetadata", {})
        usage = Usage(
            prompt_tokens=usage_metadata.get("promptTokenCount", 0),
            completion_tokens=usage_metadata.get("candidatesTokenCount", 0),
            total_tokens=usage_metadata.get("totalTokenCount", 0),
        )

        return ModelResponse(
            id=f"gemini-{now}",
            object="chat.completion",
            created=now,
            model=model,
            choices=choices,
            usage=usage,
            model_extra={"usage": usage.model_dump()},
        )

    def parse_stream_event(self, data: str, model: str) -> StreamChunk | None:
        """Parse Gemini streaming event."""
        data = data.strip()
        if not data:
            return None

        try:
            event = orjson.loads(data)
        except orjson.JSONDecodeError:
            return None

        candidates = event.get("candidates", [])
        if not candidates:
            return None

        # Cache timestamp once for this event
        now = int(time.time())
        choices: list[ChunkChoice] = []
        for i, candidate in enumerate(candidates):
            content = candidate.get("content", {})
            parts = content.get("parts", [])

            # Use list + join for efficient string building
            text_parts: list[str] = []
            tool_call_deltas: list[dict[str, Any]] = []

            for part in parts:
                if "text" in part:
                    text_parts.append(part["text"])
                elif "functionCall" in part:
                    fc = part["functionCall"]
                    tool_call_deltas.append(
                        {
                            "index": len(tool_call_deltas),
                            "id": f"call_{i}_{len(tool_call_deltas)}",
                            "type": "function",
                            "function": {
                                "name": fc.get("name", ""),
                                "arguments": orjson.dumps(fc.get("args", {})).decode(),
                            },
                        }
                    )

            text_content = "".join(text_parts) if text_parts else None
            delta = ChunkDelta(
                content=text_content,
                tool_calls=tool_call_deltas if tool_call_deltas else None,
            )

            finish_reason = None
            if candidate.get("finishReason"):
                finish_reason_map = {
                    "STOP": "stop",
                    "MAX_TOKENS": "length",
                    "SAFETY": "content_filter",
                }
                finish_reason = finish_reason_map.get(candidate.get("finishReason", ""), "stop")

            choices.append(
                ChunkChoice(
                    index=i,
                    delta=delta,
                    finish_reason=finish_reason,
                )
            )

        # Usage in stream
        usage = None
        usage_metadata = event.get("usageMetadata")
        if usage_metadata:
            usage = Usage(
                prompt_tokens=usage_metadata.get("promptTokenCount", 0),
                completion_tokens=usage_metadata.get("candidatesTokenCount", 0),
                total_tokens=usage_metadata.get("totalTokenCount", 0),
            )

        return StreamChunk(
            id=f"gemini-{now}",
            model=model,
            choices=choices,
            usage=usage,
        )

    def parse_error(
        self,
        status_code: int,
        data: bytes,
        request_id: str | None = None,
    ) -> ArcLLMError:
        """Parse Gemini error response."""
        try:
            error_data = orjson.loads(data)
            error = error_data.get("error", {})
            message = error.get("message", "Unknown error")
            error_status = error.get("status", "")
        except (orjson.JSONDecodeError, UnicodeDecodeError):
            message = data.decode("utf-8", errors="replace")
            error_status = ""

        if status_code in {401, 403}:
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
            error_type=error_status,
        )

    def build_embedding_request(
        self,
        *,
        model: str,
        input: list[str],
        **kwargs: Any,
    ) -> RequestData:
        """Build Gemini embedding request."""
        api_key = self._get_api_key()

        # Gemini uses batch embedding endpoint
        requests = [
            {"model": f"models/{model}", "content": {"parts": [{"text": text}]}} for text in input
        ]

        body = {"requests": requests}
        url = f"{self._api_base}/models/{model}:batchEmbedContents?key={api_key}"

        body_bytes = orjson.dumps(body)

        return RequestData(
            method="POST",
            url=url,
            headers=self._get_headers(),
            body=body_bytes,
            timeout=self.config.timeout,
        )

    def parse_embedding_response(self, data: bytes, model: str) -> EmbeddingResponse:
        """Parse Gemini embedding response."""
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
                    embedding=emb.get("values", []),
                )
            )

        return EmbeddingResponse(
            model=model,
            data=embeddings,
            usage=EmbeddingUsage(prompt_tokens=0, total_tokens=0),
        )


# Register on import
register_provider("gemini", GeminiAdapter)
