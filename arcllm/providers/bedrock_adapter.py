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
from typing import Any, cast
from urllib.parse import quote, urlparse

import orjson

from arcllm.exceptions import (
    ArcLLMError,
    AuthenticationError,
    BudgetExceededError,
    InternalServerError,
    InvalidRequestError,
    ProviderAPIError,
    RateLimitError,
    ResponseParseError,
    ServiceUnavailableError,
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


_ANTHROPIC_NATIVE_TOOL_PREFIXES = (
    "web_search_",
    "code_execution_",
    "text_editor_",
    "computer_use_",
    "bash_",
)


def _is_anthropic_native_tool(tool_type: str) -> bool:
    """True if ``tool_type`` matches one of Anthropic's server-side tools."""
    return any(tool_type.startswith(prefix) for prefix in _ANTHROPIC_NATIVE_TOOL_PREFIXES)


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
        # Lazy-initialised proxy adapters used only for response parsing on
        # cross-provider Bedrock paths (e.g. OpenAI-on-Bedrock). See
        # ``_parse_openai_response``.
        self._proxy_cache: dict[str, Any] = {}

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
        """Determine model family from a Bedrock model id.

        Bedrock model ids follow ``<provider>.<model>-<version>`` (or
        ``<region>.<provider>.<model>...`` for cross-region inference profiles).
        We classify by provider segment.
        """
        model_lower = model.lower()
        # OpenAI GPT-OSS family — Bedrock now hosts ``openai.gpt-oss-*``.
        if "openai" in model_lower or model_lower.startswith("gpt-oss"):
            return "openai"
        if "anthropic" in model_lower or "claude" in model_lower:
            return "anthropic"
        if "meta" in model_lower or "llama" in model_lower:
            return "meta"
        if "amazon" in model_lower or "titan" in model_lower or "nova" in model_lower:
            return "amazon"
        if "cohere" in model_lower:
            return "cohere"
        if "mistral" in model_lower:
            return "mistral"
        if "ai21" in model_lower:
            return "ai21"
        return "anthropic"  # Default fallback

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

        # Extended thinking on Bedrock-hosted Claude takes the same shape as
        # direct Anthropic. Drop temperature/top_p when thinking is on.
        if "thinking_budget" in kwargs and kwargs["thinking_budget"] is not None:
            body["thinking"] = {
                "type": "enabled",
                "budget_tokens": int(kwargs["thinking_budget"]),
            }
            body.pop("temperature", None)
            body.pop("top_p", None)

        # Handle tools
        if kwargs.get("tools"):
            tools: list[dict[str, Any]] = []
            for tool in kwargs["tools"]:
                tool_type = tool.get("type", "")
                if tool_type == "function":
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
                elif _is_anthropic_native_tool(tool_type):
                    tools.append(tool)
            body["tools"] = tools

            # Translate tool_choice to Anthropic shape (auto/any/tool/none).
            tc: Any = kwargs.get("tool_choice")
            if tc == "auto":
                body["tool_choice"] = {"type": "auto"}
            elif tc == "required":
                body["tool_choice"] = {"type": "any"}
            elif isinstance(tc, dict):
                tc_dict = cast("dict[str, Any]", tc)
                fn_raw: Any = tc_dict.get("function") or {}
                if isinstance(fn_raw, dict):
                    fn = cast("dict[str, Any]", fn_raw)
                    name = fn.get("name")
                    if name:
                        body["tool_choice"] = {"type": "tool", "name": name}

        # Anthropic-on-Bedrock honours ``response_format=json_schema`` via the
        # tool-result pattern: emit a function-style tool that captures the
        # schema and force the model to call it. This mirrors how the direct
        # Anthropic API encourages JSON-mode usage.
        rf_raw: Any = kwargs.get("response_format")
        if isinstance(rf_raw, dict) and cast("dict[str, Any]", rf_raw).get("type") == "json_schema":
            rf = cast("dict[str, Any]", rf_raw)
            json_schema = cast("dict[str, Any]", rf.get("json_schema") or {})
            schema: Any = json_schema.get("schema") or {}
            schema_name = str(json_schema.get("name") or "structured_output")
            tools = cast("list[dict[str, Any]]", body.get("tools") or [])
            tools.append(
                {
                    "name": schema_name,
                    "description": "Return the response as JSON matching the supplied schema.",
                    "input_schema": schema,
                }
            )
            body["tools"] = tools
            body["tool_choice"] = {"type": "tool", "name": schema_name}

        return body

    def _build_openai_body(
        self,
        messages: list[dict[str, Any]],
        *,
        stream: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Build an OpenAI Chat Completions body for ``openai.gpt-oss-*`` on Bedrock.

        Bedrock's GPT-OSS endpoint speaks the OpenAI wire format almost verbatim,
        so we only thread the params we know are accepted.
        """
        body: dict[str, Any] = {
            "messages": messages,
            "stream": stream,
        }
        for key in (
            "max_tokens",
            "max_completion_tokens",
            "temperature",
            "top_p",
            "stop",
            "seed",
            "presence_penalty",
            "frequency_penalty",
            "n",
            "user",
            "reasoning_effort",
        ):
            if key in kwargs and kwargs[key] is not None:
                body[key] = kwargs[key]
        if kwargs.get("tools"):
            body["tools"] = kwargs["tools"]
            if kwargs.get("tool_choice") is not None:
                body["tool_choice"] = kwargs["tool_choice"]
        if kwargs.get("response_format"):
            body["response_format"] = kwargs["response_format"]
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
        kwargs = self._check_params(model, drop_params, **kwargs)

        model_family = self._get_model_family(model)

        # Build body based on model family
        if model_family == "anthropic":
            body = self._build_anthropic_body(messages, **kwargs)
        elif model_family == "openai":
            body = self._build_openai_body(messages, stream=stream, **kwargs)
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
        if model_family == "openai":
            # OpenAI on Bedrock returns the standard Chat Completions shape.
            # Delegate to ``OpenAIAdapter.parse_response`` via a minimal stub
            # so we don't duplicate the logic. We can't subclass because we
            # already inherit from ``BaseAdapter``; reuse the helper instead.
            return self._parse_openai_response(resp, model)
        # Generic parsing
        return self._parse_generic_response(resp, model)

    def _parse_openai_response(self, resp: dict[str, Any], model: str) -> ModelResponse:
        """Parse an OpenAI-shape Bedrock response.

        Reuses ``OpenAIAdapter`` parsing via a lightweight proxy adapter
        cached on the instance (lazy to avoid circular import at module load).
        """
        from arcllm.providers.openai_adapter import OpenAIAdapter

        proxy = self._proxy_cache.get("openai")
        if proxy is None:
            proxy = OpenAIAdapter(ProviderConfig(api_key="bedrock-noop"))
            self._proxy_cache["openai"] = proxy
        response: ModelResponse = proxy.parse_response(orjson.dumps(resp), model)
        return response

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
            tool_calls=tool_calls or None,
        )

        # Map stop reason
        stop_reason = resp.get("stop_reason", "")
        finish_reason = {
            "end_turn": "stop",
            "max_tokens": "length",
            "stop_sequence": "stop",
            "tool_use": "tool_calls",
        }.get(stop_reason, stop_reason)

        # Anthropic-on-Bedrock returns the same usage shape as direct
        # Anthropic; capture the cache token fields when present.
        usage_data = resp.get("usage", {})
        cache_read = usage_data.get("cache_read_input_tokens")
        cache_creation = usage_data.get("cache_creation_input_tokens")
        base_input = usage_data.get("input_tokens", 0)
        prompt_total = base_input + (cache_read or 0) + (cache_creation or 0)
        completion_tokens = usage_data.get("output_tokens", 0)
        usage = Usage(
            prompt_tokens=prompt_total,
            completion_tokens=completion_tokens,
            total_tokens=prompt_total + completion_tokens,
            cache_read_input_tokens=cache_read,
            cache_creation_input_tokens=cache_creation,
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
        """Parse a non-Anthropic Bedrock response.

        Each Bedrock model family ships its own response shape; we dispatch on
        ``_get_model_family`` rather than blindly probing field names so that
        usage tokens are captured wherever the provider supplies them.
        """
        now = int(time.time())
        family = self._get_model_family(model)
        content, finish_reason, usage = self._extract_generic_body(family, resp)

        return ModelResponse(
            id=f"bedrock-{now}",
            object="chat.completion",
            created=now,
            model=model,
            choices=[
                Choice(
                    index=0,
                    message=Message(role="assistant", content=content),
                    finish_reason=finish_reason,
                )
            ],
            usage=usage,
            model_extra={"usage": usage.model_dump() if usage else {}},
        )

    @staticmethod
    def _extract_generic_body(
        family: str,
        resp: dict[str, Any],
    ) -> tuple[str, str | None, Usage | None]:
        """Return ``(content, finish_reason, usage)`` for non-Anthropic families.

        Field names per family:

        - ``meta`` (Llama):   ``generation`` / ``stop_reason`` /
          ``prompt_token_count`` + ``generation_token_count``.
        - ``amazon`` (Titan): ``results[0].outputText`` /
          ``completionReason`` / ``inputTextTokenCount`` +
          ``results[0].tokenCount``. Nova returns
          ``output.message.content[0].text`` and a top-level ``usage`` dict.
        - ``mistral``:        ``outputs[0].text`` / ``outputs[0].stop_reason``
          (no usage block).
        - ``cohere``:         ``generations[0].text`` /
          ``generations[0].finish_reason`` (no usage block).
        - ``ai21``:           ``completions[0].data.text`` /
          ``completions[0].finishReason.reason``.
        """

        def _str(value: Any) -> str:
            return str(value) if value else ""

        def _opt_str(value: Any) -> str | None:
            return str(value) if value else None

        def _int(value: Any) -> int:
            try:
                return int(value)
            except (TypeError, ValueError):
                return 0

        if family == "meta":
            content = _str(resp.get("generation"))
            finish = _opt_str(resp.get("stop_reason"))
            prompt_tokens = resp.get("prompt_token_count")
            completion_tokens = resp.get("generation_token_count")
            usage: Usage | None = None
            if prompt_tokens is not None or completion_tokens is not None:
                p = _int(prompt_tokens)
                c = _int(completion_tokens)
                usage = Usage(prompt_tokens=p, completion_tokens=c, total_tokens=p + c)
            return content, finish, usage

        # All non-Anthropic Bedrock response shapes are loosely-typed JSON. We
        # cast the wire dict to ``dict[str, Any]`` at every boundary so pyright
        # can narrow each nested ``.get()`` cleanly under strict mode.
        if family == "amazon":
            # Nova chat models: output.message.content[0].text + top-level usage.
            output_raw = resp.get("output")
            if isinstance(output_raw, dict) and "message" in output_raw:
                output = cast("dict[str, Any]", output_raw)
                msg = cast("dict[str, Any]", output.get("message") or {})
                parts = cast("list[Any]", msg.get("content") or [])
                content = "".join(
                    str(cast("dict[str, Any]", p).get("text") or "")
                    for p in parts
                    if isinstance(p, dict)
                )
                finish = _opt_str(resp.get("stopReason"))
                u = cast("dict[str, Any]", resp.get("usage") or {})
                if u:
                    in_tok = _int(u.get("inputTokens"))
                    out_tok = _int(u.get("outputTokens"))
                    total = _int(u.get("totalTokens")) or (in_tok + out_tok)
                    usage = Usage(
                        prompt_tokens=in_tok,
                        completion_tokens=out_tok,
                        total_tokens=total,
                    )
                else:
                    usage = None
                return content, finish, usage
            # Titan text models: results array.
            results = cast("list[Any]", resp.get("results") or [])
            content = "".join(
                str(cast("dict[str, Any]", r).get("outputText") or "")
                for r in results
                if isinstance(r, dict)
            )
            finish = (
                _opt_str(cast("dict[str, Any]", results[0]).get("completionReason"))
                if results and isinstance(results[0], dict)
                else None
            )
            input_tokens = resp.get("inputTextTokenCount")
            output_tokens = sum(
                _int(cast("dict[str, Any]", r).get("tokenCount"))
                for r in results
                if isinstance(r, dict)
            )
            if input_tokens is not None or output_tokens:
                in_tok = _int(input_tokens)
                usage = Usage(
                    prompt_tokens=in_tok,
                    completion_tokens=output_tokens,
                    total_tokens=in_tok + output_tokens,
                )
            else:
                usage = None
            return content, finish, usage

        if family == "mistral":
            outputs = cast("list[Any]", resp.get("outputs") or [])
            content = "".join(
                str(cast("dict[str, Any]", o).get("text") or "")
                for o in outputs
                if isinstance(o, dict)
            )
            finish = (
                _opt_str(cast("dict[str, Any]", outputs[0]).get("stop_reason"))
                if outputs and isinstance(outputs[0], dict)
                else None
            )
            return content, finish, None

        if family == "cohere":
            gens = cast("list[Any]", resp.get("generations") or [])
            content = "".join(
                str(cast("dict[str, Any]", g).get("text") or "")
                for g in gens
                if isinstance(g, dict)
            )
            finish = (
                _opt_str(cast("dict[str, Any]", gens[0]).get("finish_reason"))
                if gens and isinstance(gens[0], dict)
                else None
            )
            return content, finish, None

        if family == "ai21":
            completions = cast("list[Any]", resp.get("completions") or [])
            content = ""
            finish = None
            if completions and isinstance(completions[0], dict):
                first = cast("dict[str, Any]", completions[0])
                data = cast("dict[str, Any]", first.get("data") or {})
                content = str(data.get("text") or "")
                finish_reason = cast("dict[str, Any]", first.get("finishReason") or {})
                finish = _opt_str(finish_reason.get("reason"))
            return content, finish, None

        # Unknown family: best-effort extraction from any of the historical fields.
        content = _str(resp.get("generation") or resp.get("outputText") or resp.get("completion"))
        return content, None, None

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

        common_kwargs: dict[str, Any] = {
            "provider": self.provider_name,
            "status_code": status_code,
            "request_id": request_id,
        }
        message_lower = (message or "").lower()

        if status_code in {401, 403}:
            return AuthenticationError(message, **common_kwargs)
        if status_code == 402:
            return BudgetExceededError(message, **common_kwargs)
        if status_code == 429:
            if any(
                token in message_lower
                for token in ("quota", "throttling", "throttled", "credit", "billing")
            ):
                return BudgetExceededError(message, **common_kwargs)
            return RateLimitError(message, **common_kwargs)
        if status_code == 400:
            return InvalidRequestError(message, **common_kwargs)
        if status_code == 404:
            return UnsupportedModelError(message, **common_kwargs)
        if status_code == 503:
            return ServiceUnavailableError(message, **common_kwargs)
        if status_code >= 500:
            return InternalServerError(message, **common_kwargs)
        return ProviderAPIError(message, **common_kwargs)

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
