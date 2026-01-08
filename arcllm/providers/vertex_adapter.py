"""
Google Vertex AI adapter for arcllm.

Vertex AI provides access to Gemini models with different authentication
and endpoint structure than AI Studio.
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
from arcllm.providers.gemini_adapter import GeminiAdapter

__all__ = ["VertexAIAdapter"]


class VertexAIAdapter(GeminiAdapter):
    """
    Adapter for Google Vertex AI.

    Uses the same format as Gemini but with different auth and endpoints.
    Requires OAuth2 or service account authentication.
    """

    provider_name = "vertex_ai"

    def __init__(self, config: ProviderConfig) -> None:
        super().__init__(config)
        self._project = (
            config.vertex_project
            or os.environ.get("VERTEX_PROJECT")
            or os.environ.get("GOOGLE_CLOUD_PROJECT")
        )
        self._location = config.vertex_location or os.environ.get("VERTEX_LOCATION", "us-central1")
        self._api_base = config.api_base or f"https://{self._location}-aiplatform.googleapis.com/v1"

    def _get_api_key(self, env_var: str = "GOOGLE_API_KEY", param_key: str = "api_key") -> str:
        """
        Get access token for Vertex AI.

        Vertex AI uses OAuth2 tokens, not API keys.
        This can be obtained via:
        - gcloud auth print-access-token
        - Service account key
        - Application Default Credentials
        """
        # Check for explicit token
        token = (
            self.config.api_key
            or os.environ.get("VERTEX_API_KEY")
            or os.environ.get("GOOGLE_ACCESS_TOKEN")
        )

        if not token:
            # Try to get token from gcloud CLI or ADC
            try:
                import subprocess

                result = subprocess.run(
                    ["gcloud", "auth", "print-access-token"],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                if result.returncode == 0:
                    token = result.stdout.strip()
            except Exception:
                pass

        if not token:
            raise AuthenticationError(
                "Vertex AI access token not provided. Set GOOGLE_ACCESS_TOKEN, "
                "use 'gcloud auth print-access-token', or configure Application Default Credentials.",
                provider=self.provider_name,
            )
        return token

    def _get_project(self) -> str:
        """Get GCP project ID."""
        if not self._project:
            raise ArcLLMError(
                "Vertex AI project not provided. Set VERTEX_PROJECT or GOOGLE_CLOUD_PROJECT.",
                provider=self.provider_name,
            )
        return self._project

    def build_request(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        stream: bool = False,
        drop_params: bool = False,
        **kwargs: Any,
    ) -> RequestData:
        """Build Vertex AI generateContent request."""
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

        # Handle response_format
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

        # Build Vertex AI URL
        project = self._get_project()
        token = self._get_api_key()
        method = "streamGenerateContent" if stream else "generateContent"

        # Vertex AI uses publisher model path
        url = f"{self._api_base}/projects/{project}/locations/{self._location}/publishers/google/models/{model}:{method}"

        if stream:
            url += "?alt=sse"

        body_bytes = orjson.dumps(body)

        return RequestData(
            method="POST",
            url=url,
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            },
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
        """Build Vertex AI embedding request."""
        project = self._get_project()
        token = self._get_api_key()

        # Vertex AI uses instances format
        instances = [{"content": text} for text in input]
        body = {"instances": instances}

        url = f"{self._api_base}/projects/{project}/locations/{self._location}/publishers/google/models/{model}:predict"

        body_bytes = orjson.dumps(body)

        return RequestData(
            method="POST",
            url=url,
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            },
            body=body_bytes,
            timeout=self.config.timeout,
        )


# Register on import
register_provider("vertex_ai", VertexAIAdapter)
