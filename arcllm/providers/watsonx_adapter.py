"""
IBM watsonx.ai adapter for arcllm.

watsonx.ai exposes a chat-completions endpoint with an OpenAI-shape
``messages`` array, but renames ``model`` → ``model_id``, requires
``project_id`` (or ``space_id``), and pins an API version via query string.

Auth: Bearer IAM access token. The adapter accepts either:

- A **raw IBM Cloud API key** (typical case) — exchanged for an IAM access
  token at ``https://iam.cloud.ibm.com/identity/token`` on first use, then
  cached in-memory until 60 seconds before expiry.
- A **pre-exchanged IAM access token** (starts with ``eyJ`` — JWT) — used
  verbatim. Useful in environments that already manage their own IAM
  tokens (corporate SSO, vault-backed credentials, etc.).

API Documentation:
    - watsonx.ai REST API: https://cloud.ibm.com/apidocs/watsonx-ai
    - Chat endpoint: https://cloud.ibm.com/apidocs/watsonx-ai#text-chat
    - Token exchange: https://cloud.ibm.com/docs/account?topic=account-iamtoken_from_apikey

Required configuration:
    - ``api_key``: IBM Cloud API key (preferred) **or** a pre-exchanged IAM
      access token. Falls back to ``WATSONX_API_KEY`` env var.
    - ``api_base``: regional URL, e.g. ``https://us-south.ml.cloud.ibm.com``.
      Falls back to ``WATSONX_URL`` env var.
    - ``WATSONX_PROJECT_ID`` env or ``project_id`` kwarg
    - ``WATSONX_API_VERSION`` env (default ``2024-08-01``)
"""

from __future__ import annotations

import os
import threading
import time
import urllib.parse
from typing import Any

import orjson

from arcllm.exceptions import (
    AuthenticationError,
    InvalidRequestError,
)
from arcllm.http.client import HTTPClient, HTTPResponse
from arcllm.providers.base import (
    ProviderConfig,
    RequestData,
    register_provider,
)
from arcllm.providers.openai_adapter import OpenAIAdapter

__all__ = ["WatsonXAdapter"]


_DEFAULT_API_VERSION = "2024-08-01"
_IAM_TOKEN_URL = "https://iam.cloud.ibm.com/identity/token"
# Refresh the IAM token a minute early to avoid races against expiry.
_IAM_REFRESH_LEEWAY_S = 60


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

        # IAM token cache. Populated on first request when an IBM Cloud API
        # key is supplied (rather than a pre-exchanged token). Guarded by a
        # lock for thread-safety in sync use; the same cache is reused
        # across both sync and async paths since the underlying lifetime is
        # an hour.
        self._iam_token: str | None = None
        self._iam_token_expires_at: float = 0.0
        self._iam_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Auth: IAM token exchange
    # ------------------------------------------------------------------

    @staticmethod
    def _looks_like_iam_token(value: str) -> bool:
        """JWT-shaped strings are pre-exchanged IAM tokens.

        IAM access tokens are JWTs and always start with ``eyJ`` (the
        base64url encoding of ``{"alg":...``). IBM Cloud API keys never
        start with that prefix, so the heuristic is safe and avoids one
        round-trip when the caller already has a token.
        """
        return value.startswith("eyJ")

    def _exchange_apikey_for_iam_token(self, apikey: str) -> tuple[str, float]:
        """POST the IBM Cloud API key to IAM and return ``(token, expires_at)``.

        Raises :class:`AuthenticationError` on non-2xx — IAM 4xx responses
        almost always mean a bad / revoked / wrong-account API key.
        """
        body = urllib.parse.urlencode(
            {
                "grant_type": "urn:ibm:params:oauth:grant-type:apikey",
                "apikey": apikey,
            }
        ).encode()
        request = RequestData(
            method="POST",
            url=_IAM_TOKEN_URL,
            headers={
                "Content-Type": "application/x-www-form-urlencoded",
                "Accept": "application/json",
            },
            body=body,
            timeout=min(self.config.timeout, 30.0),
        )

        client = HTTPClient()
        try:
            raw = client.request(
                method=request.method,
                url=request.url,
                headers=request.headers,
                body=request.body,
                timeout=request.timeout,
            )
            assert isinstance(raw, HTTPResponse)
            if raw.status_code >= 400:
                raise AuthenticationError(
                    f"watsonx IAM token exchange failed ({raw.status_code}): "
                    f"{raw.body.decode('utf-8', errors='replace')[:200]}",
                    provider=self.provider_name,
                    status_code=raw.status_code,
                )
            payload = orjson.loads(raw.body)
        finally:
            client.close()

        token = payload.get("access_token")
        expires_in = int(payload.get("expires_in", 3600))
        if not token:
            raise AuthenticationError(
                "watsonx IAM token exchange returned no access_token",
                provider=self.provider_name,
            )
        return token, time.time() + expires_in - _IAM_REFRESH_LEEWAY_S

    def _resolve_iam_token(self) -> str:
        """Return a usable IAM bearer token, exchanging the API key if needed.

        Caches the token in-memory until shortly before its declared
        expiry so we don't hit IAM on every request.
        """
        api_key = self._get_api_key("WATSONX_API_KEY")
        if self._looks_like_iam_token(api_key):
            return api_key

        # Raw IBM Cloud API key — exchange (or reuse cached token).
        with self._iam_lock:
            if self._iam_token and time.time() < self._iam_token_expires_at:
                return self._iam_token
            token, expires_at = self._exchange_apikey_for_iam_token(api_key)
            self._iam_token = token
            self._iam_token_expires_at = expires_at
            return token

    def _build_headers(self) -> dict[str, str]:
        token = self._resolve_iam_token()
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if self.config.extra_headers:
            headers.update(self.config.extra_headers)
        return headers

    def _get_headers(self) -> dict[str, str]:
        """Override the BaseAdapter cache so headers are rebuilt on token refresh.

        BaseAdapter caches headers indefinitely. Our auth header rotates
        when the IAM token expires, so we resolve fresh each call. The
        token itself is cached in :meth:`_resolve_iam_token`, so this is
        cheap (no network call when the token is still valid).
        """
        return self._build_headers()

    # ------------------------------------------------------------------
    # Chat
    # ------------------------------------------------------------------

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

        # watsonx.ai routes streaming requests to a separate endpoint
        # — sending stream=true to /chat returns a buffered response, not
        # SSE chunks. The /chat_stream endpoint accepts the same body.
        endpoint = "chat_stream" if stream else "chat"
        url = f"{self._api_base.rstrip('/')}/ml/v1/text/{endpoint}?version={self._api_version}"
        return RequestData(
            method="POST",
            url=url,
            headers=self._get_headers(),
            body=orjson.dumps(body),
            timeout=self.config.timeout,
        )


register_provider("watsonx", WatsonXAdapter)
