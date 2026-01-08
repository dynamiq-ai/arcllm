"""
Databricks Model Serving adapter for arcllm.

Databricks provides model serving with an OpenAI-compatible API.
"""

from __future__ import annotations

import os
from typing import Any

import orjson

from arcllm.exceptions import AuthenticationError
from arcllm.providers.base import (
    ProviderConfig,
    RequestData,
    register_provider,
)
from arcllm.providers.openai_adapter import OpenAIAdapter

__all__ = ["DatabricksAdapter"]


class DatabricksAdapter(OpenAIAdapter):
    """
    Adapter for Databricks Model Serving.

    Uses OpenAI-compatible format with Databricks authentication.
    """

    provider_name = "databricks"

    def __init__(self, config: ProviderConfig) -> None:
        super().__init__(config)
        # Databricks requires workspace URL as base
        base = config.api_base or os.environ.get("DATABRICKS_HOST")
        if base:
            self._api_base = base.rstrip("/") + "/serving-endpoints"
        else:
            raise AuthenticationError(
                "Databricks host not provided. Set DATABRICKS_HOST or api_base parameter.",
                provider=self.provider_name,
            )

    def _build_headers(self) -> dict[str, str]:
        """Get request headers with Databricks authentication."""
        api_key = self.config.api_key or os.environ.get("DATABRICKS_TOKEN")
        if not api_key:
            raise AuthenticationError(
                "Databricks token not provided. Set DATABRICKS_TOKEN or api_key parameter.",
                provider=self.provider_name,
            )

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
        """Build Databricks model serving request."""
        kwargs = self._check_params(drop_params, **kwargs)

        body: dict[str, Any] = {
            "messages": messages,
            "stream": stream,
        }

        # Add optional parameters
        optional_params = [
            "temperature",
            "top_p",
            "max_tokens",
            "stop",
            "n",
        ]

        for param in optional_params:
            if param in kwargs and kwargs[param] is not None:
                body[param] = kwargs[param]

        # Handle tools
        if kwargs.get("tools"):
            body["tools"] = kwargs["tools"]
            if "tool_choice" in kwargs:
                body["tool_choice"] = kwargs["tool_choice"]

        # Databricks endpoint structure: /serving-endpoints/{endpoint_name}/invocations
        url = f"{self._api_base}/{model}/invocations"
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
        """Build Databricks embedding request."""
        body: dict[str, Any] = {
            "input": input,
        }

        url = f"{self._api_base}/{model}/invocations"
        body_bytes = orjson.dumps(body)

        return RequestData(
            method="POST",
            url=url,
            headers=self._get_headers(),
            body=body_bytes,
            timeout=self.config.timeout,
        )


# Register on import
register_provider("databricks", DatabricksAdapter)
