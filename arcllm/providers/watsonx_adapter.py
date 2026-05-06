"""
IBM watsonx.ai adapter for arcllm.

watsonx.ai exposes a chat-completions endpoint with an OpenAI-shape
``messages`` array, but renames ``model`` → ``model_id``, requires
``project_id`` (or ``space_id``), and pins an API version via query string.

Auth: Bearer IAM access token. Token exchange is **caller-managed** in
0.4.0 — exchange your IBM Cloud API key for an IAM token via
``https://iam.cloud.ibm.com/identity/token`` and pass the resulting token
as ``api_key``. Automatic IAM exchange is planned for a follow-up.

API Documentation:
    - watsonx.ai REST API: https://cloud.ibm.com/apidocs/watsonx-ai
    - Chat endpoint: https://cloud.ibm.com/apidocs/watsonx-ai#text-chat
    - Token exchange: https://cloud.ibm.com/docs/account?topic=account-iamtoken_from_apikey

Required configuration:
    - ``api_key``: pre-exchanged IAM access token
    - ``api_base``: regional URL, e.g. ``https://us-south.ml.cloud.ibm.com``
    - ``WATSONX_PROJECT_ID`` env or ``project_id`` kwarg
    - ``WATSONX_API_VERSION`` env (default ``2024-08-01``)
"""

from __future__ import annotations

import os
from typing import Any

import orjson

from arcllm.exceptions import InvalidRequestError
from arcllm.providers.base import (
    ProviderConfig,
    RequestData,
    register_provider,
)
from arcllm.providers.openai_adapter import OpenAIAdapter

__all__ = ["WatsonXAdapter"]


_DEFAULT_API_VERSION = "2024-08-01"


class WatsonXAdapter(OpenAIAdapter):
    """Adapter for IBM watsonx.ai chat-completions API."""

    provider_name = "watsonx"

    def __init__(self, config: ProviderConfig) -> None:
        super().__init__(config)
        self._api_base = config.api_base or os.environ.get(
            "WATSONX_URL",
            "https://us-south.ml.cloud.ibm.com",
        )
        self._api_version = config.api_version or os.environ.get(
            "WATSONX_API_VERSION", _DEFAULT_API_VERSION
        )

    def _build_headers(self) -> dict[str, str]:
        api_key = self._get_api_key("WATSONX_TOKEN")
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
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
        """Build watsonx.ai chat request.

        Reuses :class:`OpenAIAdapter`'s body construction (capability filter
        + optional-param mapping) and then renames ``model`` to ``model_id``
        and injects ``project_id`` / ``space_id``.
        """
        project_id = kwargs.pop("project_id", None) or os.environ.get("WATSONX_PROJECT_ID")
        space_id = kwargs.pop("space_id", None) or os.environ.get("WATSONX_SPACE_ID")
        if not project_id and not space_id:
            raise InvalidRequestError(
                "watsonx requires `project_id` (or `space_id`); set WATSONX_PROJECT_ID env or pass project_id kwarg",
                provider=self.provider_name,
            )

        # Reuse OpenAI body construction by calling super, then rewrite.
        base_request = super().build_request(
            model=model,
            messages=messages,
            stream=stream,
            drop_params=drop_params,
            **kwargs,
        )
        body: dict[str, Any] = orjson.loads(base_request.body or b"{}")
        body["model_id"] = body.pop("model")
        if project_id:
            body["project_id"] = project_id
        if space_id:
            body["space_id"] = space_id

        url = f"{self._api_base.rstrip('/')}/ml/v1/text/chat?version={self._api_version}"
        return RequestData(
            method="POST",
            url=url,
            headers=self._get_headers(),
            body=orjson.dumps(body),
            timeout=self.config.timeout,
        )


register_provider("watsonx", WatsonXAdapter)
