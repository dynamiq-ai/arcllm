"""
Azure OpenAI adapter for arcllm.

Azure OpenAI uses the same API format as OpenAI but with different
authentication and endpoint structure.
"""

from __future__ import annotations

import os
from typing import Any

import orjson

from arcllm.exceptions import (
    ArcLLMError,
    AuthenticationError,
)
from arcllm.providers.base import (
    ProviderConfig,
    RequestData,
    register_provider,
)
from arcllm.providers.openai_adapter import OpenAIAdapter

__all__ = ["AzureOpenAIAdapter"]


class AzureOpenAIAdapter(OpenAIAdapter):
    """
    Adapter for Azure OpenAI Service.

    Inherits from OpenAIAdapter since Azure uses the same API format,
    but requires different authentication and endpoint handling.
    """

    provider_name = "azure"

    def __init__(self, config: ProviderConfig) -> None:
        super().__init__(config)
        self._api_version = config.api_version or "2024-10-21"
        self._deployment = config.azure_deployment
        self._ad_token = config.azure_ad_token

    def _get_api_base(self) -> str:
        """Get Azure API base URL."""
        base = self.config.api_base
        if not base:
            # Try environment variable
            base = os.environ.get("AZURE_OPENAI_ENDPOINT")
        if not base:
            raise ArcLLMError(
                "Azure OpenAI endpoint not provided. Set AZURE_OPENAI_ENDPOINT "
                "or pass api_base parameter.",
                provider=self.provider_name,
            )
        return base.rstrip("/")

    def _build_headers(self) -> dict[str, str]:
        """Get request headers for Azure."""
        headers = {"Content-Type": "application/json"}

        # Azure supports both API key and Azure AD token authentication
        if self._ad_token or os.environ.get("AZURE_OPENAI_AD_TOKEN"):
            token = self._ad_token or os.environ.get("AZURE_OPENAI_AD_TOKEN")
            headers["Authorization"] = f"Bearer {token}"
        else:
            api_key = self.config.api_key or os.environ.get("AZURE_OPENAI_API_KEY")
            if not api_key:
                raise AuthenticationError(
                    "Azure OpenAI API key not provided. Set AZURE_OPENAI_API_KEY "
                    "or pass api_key parameter.",
                    provider=self.provider_name,
                )
            headers["api-key"] = api_key

        if self.config.extra_headers:
            headers.update(self.config.extra_headers)

        return headers

    def _get_deployment(self, model: str) -> str:
        """Get deployment name from model or config."""
        # Deployment can be specified in config, or we use the model name
        return self._deployment or model

    def build_request(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        stream: bool = False,
        drop_params: bool = False,
        **kwargs: Any,
    ) -> RequestData:
        """Build Azure OpenAI chat completion request."""
        # Check params (same as OpenAI)
        kwargs = self._check_params(drop_params, **kwargs)

        # Build request body (same format as OpenAI)
        body: dict[str, Any] = {
            "messages": messages,
            "stream": stream,
        }

        # Add stream_options for usage in streaming
        if stream:
            stream_options = kwargs.pop("stream_options", None)
            if stream_options:
                body["stream_options"] = stream_options

        # Add optional parameters (same as OpenAI)
        optional_params = [
            "temperature",
            "top_p",
            "max_tokens",
            "max_completion_tokens",
            "stop",
            "seed",
            "presence_penalty",
            "frequency_penalty",
            "logit_bias",
            "n",
            "logprobs",
            "top_logprobs",
            "user",
            "parallel_tool_calls",
        ]

        for param in optional_params:
            if param in kwargs and kwargs[param] is not None:
                body[param] = kwargs[param]

        # Handle tools
        if kwargs.get("tools"):
            body["tools"] = kwargs["tools"]
            if "tool_choice" in kwargs:
                body["tool_choice"] = kwargs["tool_choice"]

        # Handle response_format
        if kwargs.get("response_format"):
            body["response_format"] = kwargs["response_format"]

        # Build Azure-specific URL
        api_base = self._get_api_base()
        deployment = self._get_deployment(model)
        url = f"{api_base}/openai/deployments/{deployment}/chat/completions?api-version={self._api_version}"

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
        """Build Azure OpenAI embedding request."""
        body: dict[str, Any] = {
            "input": input,
        }

        # Optional parameters
        if "encoding_format" in kwargs:
            body["encoding_format"] = kwargs["encoding_format"]
        if "dimensions" in kwargs:
            body["dimensions"] = kwargs["dimensions"]
        if "user" in kwargs:
            body["user"] = kwargs["user"]

        # Build Azure-specific URL
        api_base = self._get_api_base()
        deployment = self._get_deployment(model)
        url = (
            f"{api_base}/openai/deployments/{deployment}/embeddings?api-version={self._api_version}"
        )

        body_bytes = orjson.dumps(body)

        return RequestData(
            method="POST",
            url=url,
            headers=self._get_headers(),
            body=body_bytes,
            timeout=self.config.timeout,
        )


# Register on import
register_provider("azure", AzureOpenAIAdapter)
