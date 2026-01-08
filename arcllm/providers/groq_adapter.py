"""
Groq adapter for arcllm.

Groq provides an OpenAI-compatible API with ultra-fast inference
powered by their custom LPU (Language Processing Unit) hardware.

API Documentation References (for future updates):
    - Main Docs: https://console.groq.com/docs
    - Models: https://console.groq.com/docs/models
    - Chat API: https://console.groq.com/docs/api-reference#chat-create
    - Rate Limits: https://console.groq.com/docs/rate-limits
    - Pricing: https://groq.com/pricing/

Available Models (January 2026):
    - meta-llama/llama-4-maverick-17b-128e-instruct (Vision, tools, 128K ctx)
    - meta-llama/llama-4-scout-17b-16e-instruct (Vision, tools, 128K ctx)
    - openai/gpt-oss-120b (Strong reasoning, 128K ctx)
    - openai/gpt-oss-20b (Fast, tool use, 128K ctx)
    - moonshotai/kimi-k2-instruct (256K context)
    - qwen/qwen3-32b (General purpose)
    - groq/compound (Agentic with built-in tools)
    - groq/compound-mini (Lightweight agentic)
    - llama-3.3-70b-versatile (Stable flagship)
    - llama-3.1-8b-instant (Ultra-fast inference)

Key Features:
    - Ultra-low latency inference (fastest in the industry)
    - OpenAI-compatible API format
    - Built-in tools with Compound models (web search, code exec)
    - Wide model selection (Llama, GPT-OSS, Qwen, Kimi, etc.)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from arcllm.exceptions import UnsupportedModelError
from arcllm.providers.base import (
    ProviderConfig,
    RequestData,
    register_provider,
)
from arcllm.providers.openai_adapter import OpenAIAdapter

if TYPE_CHECKING:
    from arcllm.types import EmbeddingResponse

__all__ = ["GroqAdapter"]


class GroqAdapter(OpenAIAdapter):
    """
    Adapter for Groq API.

    Groq uses OpenAI-compatible format.
    """

    provider_name = "groq"

    def __init__(self, config: ProviderConfig) -> None:
        super().__init__(config)
        self._api_base = config.api_base or "https://api.groq.com/openai/v1"

    def _build_headers(self) -> dict[str, str]:
        """Get request headers."""
        api_key = self._get_api_key("GROQ_API_KEY")
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        if self.config.extra_headers:
            headers.update(self.config.extra_headers)
        return headers

    def build_embedding_request(
        self,
        *,
        model: str,
        input: list[str],
        **kwargs: Any,
    ) -> RequestData:
        """Groq does not support embeddings."""
        raise UnsupportedModelError(
            "Groq does not provide an embeddings API",
            provider=self.provider_name,
        )

    def parse_embedding_response(self, data: bytes, model: str) -> EmbeddingResponse:
        """Groq does not support embeddings."""
        raise UnsupportedModelError(
            "Groq does not provide an embeddings API",
            provider=self.provider_name,
        )


# Register on import
register_provider("groq", GroqAdapter)
