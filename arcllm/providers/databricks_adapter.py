"""
Databricks Foundation Model APIs adapter for arcllm.

Databricks normalises every hosted model — Llama, Claude (Anthropic), Gemini,
GPT-5, Gemma, OSS variants — to an OpenAI-compatible Chat Completions /
Embeddings wire shape. We therefore use one unified body builder regardless
of the underlying model family. The model id is the **endpoint name** the
user pre-deployed on Databricks (e.g. ``databricks-claude-sonnet-4-5``,
``databricks-meta-llama-3-3-70b-instruct``); it goes in the URL, not the body.

Modern reasoning + thinking params are pass-through:

- ``reasoning_effort`` for GPT-5 / o-series endpoints.
- ``thinking_budget`` for Claude endpoints (Databricks forwards the extra body
  key to the underlying Anthropic deployment).
- ``response_format`` for JSON-mode workflows.
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
        kwargs = self._check_params(model, drop_params, **kwargs)

        body: dict[str, Any] = {
            "messages": messages,
            "stream": stream,
        }

        # Pass-through OpenAI-compatible request params + reasoning extras.
        # Databricks forwards unknown body keys to the underlying model
        # endpoint, which is how we surface ``thinking_budget`` for Claude
        # endpoints and ``reasoning_effort`` for GPT-5 endpoints.
        optional_params = (
            "temperature",
            "top_p",
            "max_tokens",
            "max_completion_tokens",
            "stop",
            "seed",
            "presence_penalty",
            "frequency_penalty",
            "n",
            "user",
            "reasoning_effort",
            "thinking_budget",
        )
        for param in optional_params:
            if param in kwargs and kwargs[param] is not None:
                body[param] = kwargs[param]

        # Handle tools / response_format pass-through (Databricks accepts the
        # OpenAI shapes verbatim).
        if kwargs.get("tools"):
            body["tools"] = kwargs["tools"]
            if kwargs.get("tool_choice") is not None:
                body["tool_choice"] = kwargs["tool_choice"]
        if kwargs.get("response_format"):
            body["response_format"] = kwargs["response_format"]

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
