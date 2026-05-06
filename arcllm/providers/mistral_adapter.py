"""
Mistral AI adapter for arcllm.

Mistral uses an OpenAI-compatible API format with minor differences.
"""

from __future__ import annotations

from typing import Any

import orjson

from arcllm.providers.base import (
    COMMON_PARAMS,
    ProviderConfig,
    RequestData,
    register_provider,
)
from arcllm.providers.openai_adapter import OpenAIAdapter

__all__ = ["MistralAdapter"]


class MistralAdapter(OpenAIAdapter):
    """
    Adapter for Mistral AI API.

    Mistral uses OpenAI-compatible format with some differences.
    """

    provider_name = "mistral"

    # Mistral-specific extras on top of the OpenAI-compatible surface.
    supported_params = COMMON_PARAMS | {
        "safe_prompt",  # Mistral-only content moderation flag.
        "random_seed",  # Mistral's name for deterministic sampling.
    }

    def __init__(self, config: ProviderConfig) -> None:
        super().__init__(config)
        self._api_base = config.api_base or "https://api.mistral.ai/v1"

    def _build_headers(self) -> dict[str, str]:
        """Get request headers."""
        api_key = self._get_api_key("MISTRAL_API_KEY")
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
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
        """Build Mistral chat completion request."""
        # Use parent's implementation with Mistral-specific adjustments
        kwargs = self._check_params(model, drop_params, **kwargs)

        body: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": stream,
        }

        # Mistral-specific: safe_prompt for content moderation
        if "safe_prompt" in kwargs:
            body["safe_prompt"] = kwargs["safe_prompt"]

        # Add optional parameters
        optional_params = [
            "temperature",
            "top_p",
            "max_tokens",
            "random_seed",  # Mistral uses random_seed instead of seed
        ]

        for param in optional_params:
            if param in kwargs and kwargs[param] is not None:
                body[param] = kwargs[param]

        # Mistral accepts `stop` as a list of strings; map a bare string too.
        if kwargs.get("stop"):
            stop_val = kwargs["stop"]
            body["stop"] = stop_val if isinstance(stop_val, list) else [stop_val]

        # Map seed to random_seed
        if "seed" in kwargs and kwargs["seed"] is not None:
            body["random_seed"] = kwargs["seed"]

        # Handle tools
        if kwargs.get("tools"):
            body["tools"] = kwargs["tools"]
            if "tool_choice" in kwargs:
                body["tool_choice"] = kwargs["tool_choice"]

        # Handle response_format
        if kwargs.get("response_format"):
            body["response_format"] = kwargs["response_format"]

        url = f"{self._api_base}/chat/completions"
        body_bytes = orjson.dumps(body)

        return RequestData(
            method="POST",
            url=url,
            headers=self._get_headers(),
            body=body_bytes,
            timeout=self.config.timeout,
        )

    def build_embedding_request(
        self,
        *,
        model: str,
        input: list[str],
        **kwargs: Any,
    ) -> RequestData:
        """Build Mistral embedding request."""
        body: dict[str, Any] = {
            "model": model,
            "input": input,
        }

        if "encoding_format" in kwargs:
            body["encoding_format"] = kwargs["encoding_format"]

        url = f"{self._api_base}/embeddings"
        body_bytes = orjson.dumps(body)

        return RequestData(
            method="POST",
            url=url,
            headers=self._get_headers(),
            body=body_bytes,
            timeout=self.config.timeout,
        )


# Register on import
register_provider("mistral", MistralAdapter)
