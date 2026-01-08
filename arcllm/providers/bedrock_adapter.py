"""
AWS Bedrock adapter for arcllm.

Bedrock provides access to various models including Anthropic Claude,
Meta Llama, Amazon Titan, and more through AWS infrastructure.

Requires AWS credentials (access key, secret key, optional session token)
and signing requests with AWS Signature Version 4.
"""

from __future__ import annotations

import datetime
import hashlib
import hmac
import os
import time
from typing import Any
from urllib.parse import quote, urlparse

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

__all__ = ["BedrockAdapter"]


class BedrockAdapter(BaseAdapter):
    """
    Adapter for AWS Bedrock Runtime API.

    Supports Anthropic Claude, Meta Llama, Amazon Titan models.
    Uses AWS Signature Version 4 for authentication.
    """

    provider_name = "bedrock"

    supported_params = COMMON_PARAMS | {
        "anthropic_version",
        "top_k",
    }

    def __init__(self, config: ProviderConfig) -> None:
        super().__init__(config)
        self._region = config.aws_region or os.environ.get("AWS_REGION", "us-east-1")
        self._access_key = config.aws_access_key_id or os.environ.get("AWS_ACCESS_KEY_ID")
        self._secret_key = config.aws_secret_access_key or os.environ.get("AWS_SECRET_ACCESS_KEY")
        self._session_token = config.aws_session_token or os.environ.get("AWS_SESSION_TOKEN")
        self._api_base = config.api_base or f"https://bedrock-runtime.{self._region}.amazonaws.com"

    def _get_credentials(self) -> tuple[str, str, str | None]:
        """Get AWS credentials."""
        if not self._access_key or not self._secret_key:
            raise AuthenticationError(
                "AWS credentials not provided. Set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY.",
                provider=self.provider_name,
            )
        return self._access_key, self._secret_key, self._session_token

    def _sign_request(
        self,
        method: str,
        url: str,
        headers: dict[str, str],
        body: bytes,
    ) -> dict[str, str]:
        """
        Sign request with AWS Signature Version 4.

        This is a simplified implementation of AWS SigV4 signing.
        """
        access_key, secret_key, session_token = self._get_credentials()

        # Parse URL
        parsed = urlparse(url)
        host = parsed.hostname or ""
        path = parsed.path or "/"

        # Current time (timezone-aware UTC)
        now = datetime.datetime.now(datetime.UTC)
        amz_date = now.strftime("%Y%m%dT%H%M%SZ")
        date_stamp = now.strftime("%Y%m%d")

        # Create canonical request
        service = "bedrock"
        algorithm = "AWS4-HMAC-SHA256"
        credential_scope = f"{date_stamp}/{self._region}/{service}/aws4_request"

        # Headers to sign
        signed_headers = {
            "content-type": headers.get("Content-Type", "application/json"),
            "host": host,
            "x-amz-date": amz_date,
        }
        if session_token:
            signed_headers["x-amz-security-token"] = session_token

        # Sort headers
        sorted_header_names = sorted(signed_headers.keys())
        canonical_headers = "".join(f"{k}:{signed_headers[k]}\n" for k in sorted_header_names)
        signed_headers_str = ";".join(sorted_header_names)

        # Payload hash
        payload_hash = hashlib.sha256(body).hexdigest()

        # Canonical request
        canonical_request = "\n".join(
            [
                method,
                quote(path, safe="/-_.~"),
                "",  # query string
                canonical_headers,
                signed_headers_str,
                payload_hash,
            ]
        )

        # String to sign
        canonical_request_hash = hashlib.sha256(canonical_request.encode()).hexdigest()
        string_to_sign = "\n".join(
            [
                algorithm,
                amz_date,
                credential_scope,
                canonical_request_hash,
            ]
        )

        # Signing key
        def sign(key: bytes, msg: str) -> bytes:
            return hmac.new(key, msg.encode(), hashlib.sha256).digest()

        k_date = sign(f"AWS4{secret_key}".encode(), date_stamp)
        k_region = sign(k_date, self._region)
        k_service = sign(k_region, service)
        k_signing = sign(k_service, "aws4_request")

        # Signature
        signature = hmac.new(k_signing, string_to_sign.encode(), hashlib.sha256).hexdigest()

        # Authorization header
        authorization = (
            f"{algorithm} "
            f"Credential={access_key}/{credential_scope}, "
            f"SignedHeaders={signed_headers_str}, "
            f"Signature={signature}"
        )

        # Build final headers
        final_headers = {
            "Content-Type": headers.get("Content-Type", "application/json"),
            "Authorization": authorization,
            "X-Amz-Date": amz_date,
            "X-Amz-Content-Sha256": payload_hash,
        }
        if session_token:
            final_headers["X-Amz-Security-Token"] = session_token

        return final_headers

    def _get_model_family(self, model: str) -> str:
        """Determine model family from model ID."""
        model_lower = model.lower()
        if "anthropic" in model_lower or "claude" in model_lower:
            return "anthropic"
        if "meta" in model_lower or "llama" in model_lower:
            return "meta"
        if "amazon" in model_lower or "titan" in model_lower:
            return "amazon"
        if "cohere" in model_lower:
            return "cohere"
        if "mistral" in model_lower:
            return "mistral"
        if "ai21" in model_lower:
            return "ai21"
        return "anthropic"  # Default

    def _build_anthropic_body(
        self,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Build request body for Anthropic Claude models."""
        # Convert messages to Anthropic format
        system_prompt: str | None = None
        anthropic_messages: list[dict[str, Any]] = []

        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")

            if role == "system":
                if system_prompt:
                    system_prompt += "\n\n" + content
                else:
                    system_prompt = content
            elif role == "user":
                anthropic_messages.append({"role": "user", "content": content})
            elif role == "assistant":
                anthropic_messages.append({"role": "assistant", "content": content})
            elif role == "tool":
                # Tool results
                anthropic_messages.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": msg.get("tool_call_id", ""),
                                "content": content,
                            }
                        ],
                    }
                )

        body: dict[str, Any] = {
            "anthropic_version": kwargs.get("anthropic_version", "bedrock-2023-05-31"),
            "messages": anthropic_messages,
            "max_tokens": kwargs.get("max_tokens", 4096),
        }

        if system_prompt:
            body["system"] = system_prompt

        if "temperature" in kwargs and kwargs["temperature"] is not None:
            body["temperature"] = kwargs["temperature"]
        if "top_p" in kwargs and kwargs["top_p"] is not None:
            body["top_p"] = kwargs["top_p"]
        if "top_k" in kwargs and kwargs["top_k"] is not None:
            body["top_k"] = kwargs["top_k"]
        if kwargs.get("stop"):
            body["stop_sequences"] = (
                kwargs["stop"] if isinstance(kwargs["stop"], list) else [kwargs["stop"]]
            )

        # Handle tools
        if kwargs.get("tools"):
            tools: list[dict[str, Any]] = []
            for tool in kwargs["tools"]:
                if tool.get("type") == "function":
                    func: dict[str, Any] = tool.get("function", {})
                    tools.append(
                        {
                            "name": func.get("name", ""),
                            "description": func.get("description", ""),
                            "input_schema": func.get(
                                "parameters", {"type": "object", "properties": {}}
                            ),
                        }
                    )
            body["tools"] = tools

        return body

    def build_request(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        stream: bool = False,
        drop_params: bool = False,
        **kwargs: Any,
    ) -> RequestData:
        """Build Bedrock invoke request."""
        kwargs = self._check_params(drop_params, **kwargs)

        model_family = self._get_model_family(model)

        # Build body based on model family
        if model_family == "anthropic":
            body = self._build_anthropic_body(messages, **kwargs)
        else:
            # Generic body for other models (simplified)
            body = {
                "messages": messages,
                "max_tokens": kwargs.get("max_tokens", 4096),
            }
            if "temperature" in kwargs:
                body["temperature"] = kwargs["temperature"]

        # Determine endpoint
        endpoint = "invoke-with-response-stream" if stream else "invoke"

        url = f"{self._api_base}/model/{model}/{endpoint}"
        body_bytes = orjson.dumps(body)

        headers = self._sign_request("POST", url, {"Content-Type": "application/json"}, body_bytes)

        return RequestData(
            method="POST",
            url=url,
            headers=headers,
            body=body_bytes,
            timeout=self.config.timeout,
        )

    def parse_response(self, data: bytes, model: str) -> ModelResponse:
        """Parse Bedrock response."""
        try:
            resp = orjson.loads(data)
        except (orjson.JSONDecodeError, UnicodeDecodeError) as e:
            raise ResponseParseError(
                f"Failed to parse response JSON: {e}",
                provider=self.provider_name,
                raw_data=data,
            ) from e

        model_family = self._get_model_family(model)

        if model_family == "anthropic":
            return self._parse_anthropic_response(resp, model)
        # Generic parsing
        return self._parse_generic_response(resp, model)

    def _parse_anthropic_response(self, resp: dict[str, Any], model: str) -> ModelResponse:
        """Parse Anthropic Claude response from Bedrock."""
        # Cache timestamp once for this response
        now = int(time.time())
        content_blocks = resp.get("content", [])

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

        text_content = "".join(text_parts) if text_parts else None
        message = Message(
            role="assistant",
            content=text_content,
            tool_calls=tool_calls if tool_calls else None,
        )

        # Map stop reason
        stop_reason = resp.get("stop_reason", "")
        finish_reason = {
            "end_turn": "stop",
            "max_tokens": "length",
            "stop_sequence": "stop",
            "tool_use": "tool_calls",
        }.get(stop_reason, stop_reason)

        usage_data = resp.get("usage", {})
        usage = Usage(
            prompt_tokens=usage_data.get("input_tokens", 0),
            completion_tokens=usage_data.get("output_tokens", 0),
            total_tokens=usage_data.get("input_tokens", 0) + usage_data.get("output_tokens", 0),
        )

        return ModelResponse(
            id=resp.get("id", f"bedrock-{now}"),
            object="chat.completion",
            created=now,
            model=model,
            choices=[Choice(index=0, message=message, finish_reason=finish_reason)],
            usage=usage,
            model_extra={"usage": usage.model_dump()},
        )

    def _parse_generic_response(self, resp: dict[str, Any], model: str) -> ModelResponse:
        """Parse generic model response."""
        # Cache timestamp once for this response
        now = int(time.time())
        content = (
            resp.get("generation", "") or resp.get("outputText", "") or resp.get("completion", "")
        )

        message = Message(role="assistant", content=content)

        return ModelResponse(
            id=f"bedrock-{now}",
            object="chat.completion",
            created=now,
            model=model,
            choices=[Choice(index=0, message=message, finish_reason="stop")],
            usage=Usage(),
            model_extra={"usage": {}},
        )

    def parse_stream_event(self, data: str, model: str) -> StreamChunk | None:
        """Parse Bedrock streaming event."""
        data = data.strip()
        if not data:
            return None

        try:
            event = orjson.loads(data)
        except orjson.JSONDecodeError:
            return None

        # Bedrock streaming format varies by model
        model_family = self._get_model_family(model)

        if model_family == "anthropic":
            return self._parse_anthropic_stream_event(event, model)
        return self._parse_generic_stream_event(event, model)

    def _parse_anthropic_stream_event(
        self, event: dict[str, Any], model: str
    ) -> StreamChunk | None:
        """Parse Anthropic streaming event from Bedrock."""
        event_type = event.get("type", "")

        if event_type == "message_start":
            return StreamChunk(
                id=event.get("message", {}).get("id", ""),
                model=model,
                choices=[
                    ChunkChoice(
                        index=0,
                        delta=ChunkDelta(role="assistant"),
                        finish_reason=None,
                    )
                ],
            )

        if event_type == "content_block_delta":
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
                                        "function": {"arguments": delta.get("partial_json", "")},
                                    }
                                ]
                            ),
                            finish_reason=None,
                        )
                    ],
                )

        elif event_type == "message_delta":
            delta = event.get("delta", {})
            stop_reason = delta.get("stop_reason", "")
            finish_reason = {
                "end_turn": "stop",
                "max_tokens": "length",
                "tool_use": "tool_calls",
            }.get(stop_reason, stop_reason)

            usage = None
            usage_data = event.get("usage", {})
            if usage_data:
                usage = Usage(
                    prompt_tokens=0,
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

        return None

    def _parse_generic_stream_event(self, event: dict[str, Any], model: str) -> StreamChunk | None:
        """Parse generic streaming event."""
        text = (
            event.get("outputText", "")
            or event.get("generation", "")
            or event.get("completion", "")
        )

        if text:
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

        return None

    def parse_error(
        self,
        status_code: int,
        data: bytes,
        request_id: str | None = None,
    ) -> ArcLLMError:
        """Parse Bedrock error response."""
        try:
            error_data = orjson.loads(data)
            message = error_data.get("message", "Unknown error")
        except (orjson.JSONDecodeError, UnicodeDecodeError):
            message = data.decode("utf-8", errors="replace")

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
        )

    def build_embedding_request(
        self,
        *,
        model: str,
        input: list[str],
        **kwargs: Any,
    ) -> RequestData:
        """Build Bedrock embedding request for Titan Embeddings."""
        # Bedrock embeddings are one at a time
        body = {
            "inputText": input[0] if len(input) == 1 else input,
        }

        url = f"{self._api_base}/model/{model}/invoke"
        body_bytes = orjson.dumps(body)

        headers = self._sign_request("POST", url, {"Content-Type": "application/json"}, body_bytes)

        return RequestData(
            method="POST",
            url=url,
            headers=headers,
            body=body_bytes,
            timeout=self.config.timeout,
        )

    def parse_embedding_response(self, data: bytes, model: str) -> EmbeddingResponse:
        """Parse Bedrock embedding response."""
        try:
            resp = orjson.loads(data)
        except (orjson.JSONDecodeError, UnicodeDecodeError) as e:
            raise ResponseParseError(
                f"Failed to parse embedding response: {e}",
                provider=self.provider_name,
                raw_data=data,
            ) from e

        embedding = resp.get("embedding", [])

        return EmbeddingResponse(
            model=model,
            data=[EmbeddingData(index=0, embedding=embedding)],
            usage=EmbeddingUsage(
                prompt_tokens=resp.get("inputTextTokenCount", 0),
                total_tokens=resp.get("inputTextTokenCount", 0),
            ),
        )


# Register on import
register_provider("bedrock", BedrockAdapter)
