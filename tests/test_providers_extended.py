"""
Extended tests for provider adapters.

Tests to improve coverage for smaller provider adapters that
extend OpenAI-compatible patterns.
"""

from __future__ import annotations

import json

import pytest

from arcllm.providers.base import ProviderConfig


class TestDeepSeekAdapter:
    """Tests for DeepSeekAdapter."""

    @pytest.fixture
    def adapter(self):
        """Create DeepSeek adapter with test key."""
        from arcllm.providers.deepseek_adapter import DeepSeekAdapter

        config = ProviderConfig(api_key="test-key")
        return DeepSeekAdapter(config)

    def test_provider_name(self, adapter):
        """Test provider name is set correctly."""
        assert adapter.provider_name == "deepseek"

    def test_api_base(self, adapter):
        """Test default API base."""
        assert "deepseek" in adapter._api_base.lower()

    def test_build_request(self, adapter):
        """Test building request."""
        request = adapter.build_request(
            model="deepseek-chat",
            messages=[{"role": "user", "content": "Hi"}],
        )
        assert request.method == "POST"
        body = json.loads(request.body.decode())
        assert body["model"] == "deepseek-chat"


class TestFireworksAdapter:
    """Tests for FireworksAdapter."""

    @pytest.fixture
    def adapter(self):
        """Create Fireworks adapter with test key."""
        from arcllm.providers.fireworks_adapter import FireworksAdapter

        config = ProviderConfig(api_key="test-key")
        return FireworksAdapter(config)

    def test_provider_name(self, adapter):
        """Test provider name is set correctly."""
        assert adapter.provider_name == "fireworks_ai"

    def test_api_base(self, adapter):
        """Test default API base."""
        assert "fireworks" in adapter._api_base.lower()

    def test_build_request(self, adapter):
        """Test building request."""
        request = adapter.build_request(
            model="accounts/fireworks/models/llama-v3p3-70b-instruct",
            messages=[{"role": "user", "content": "Hi"}],
        )
        assert request.method == "POST"


class TestTogetherAdapter:
    """Tests for TogetherAdapter."""

    @pytest.fixture
    def adapter(self):
        """Create Together adapter with test key."""
        from arcllm.providers.together_adapter import TogetherAdapter

        config = ProviderConfig(api_key="test-key")
        return TogetherAdapter(config)

    def test_provider_name(self, adapter):
        """Test provider name is set correctly."""
        assert adapter.provider_name == "together_ai"

    def test_api_base(self, adapter):
        """Test default API base."""
        assert "together" in adapter._api_base.lower()

    def test_build_request(self, adapter):
        """Test building request."""
        request = adapter.build_request(
            model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
            messages=[{"role": "user", "content": "Hi"}],
        )
        assert request.method == "POST"


class TestGroqAdapter:
    """Tests for GroqAdapter."""

    @pytest.fixture
    def adapter(self):
        """Create Groq adapter with test key."""
        from arcllm.providers.groq_adapter import GroqAdapter

        config = ProviderConfig(api_key="test-key")
        return GroqAdapter(config)

    def test_provider_name(self, adapter):
        """Test provider name is set correctly."""
        assert adapter.provider_name == "groq"

    def test_api_base(self, adapter):
        """Test default API base."""
        assert "groq" in adapter._api_base.lower()

    def test_build_request(self, adapter):
        """Test building request."""
        request = adapter.build_request(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": "Hi"}],
        )
        assert request.method == "POST"
        body = json.loads(request.body.decode())
        assert body["model"] == "llama-3.3-70b-versatile"


class TestPerplexityAdapter:
    """Tests for PerplexityAdapter."""

    @pytest.fixture
    def adapter(self):
        """Create Perplexity adapter with test key."""
        from arcllm.providers.perplexity_adapter import PerplexityAdapter

        config = ProviderConfig(api_key="test-key")
        return PerplexityAdapter(config)

    def test_provider_name(self, adapter):
        """Test provider name is set correctly."""
        assert adapter.provider_name == "perplexity"

    def test_api_base(self, adapter):
        """Test default API base."""
        assert "perplexity" in adapter._api_base.lower()

    def test_build_request(self, adapter):
        """Test building request."""
        request = adapter.build_request(
            model="sonar-pro",
            messages=[{"role": "user", "content": "Hi"}],
        )
        assert request.method == "POST"


class TestCapabilitiesExtended:
    """Extended tests for capabilities module."""

    def test_get_model_capabilities_all_providers(self):
        """Test capabilities for models from all providers."""
        from arcllm.capabilities import get_model_capabilities

        models = [
            "gpt-4o",
            "claude-3-5-sonnet-20241022",
            "gemini-1.5-pro",
            "mistral-large-latest",
            "llama-3.3-70b-versatile",
            "deepseek-chat",
            "sonar-pro",
        ]
        for model in models:
            caps = get_model_capabilities(model)
            assert caps.max_tokens is not None or caps.context_window is not None

    def test_default_capabilities(self):
        """Test default capabilities for unknown models."""
        from arcllm.capabilities.tables import DEFAULT_CAPABILITIES

        assert DEFAULT_CAPABILITIES.max_tokens == 4096
        assert DEFAULT_CAPABILITIES.context_window == 8192
        assert DEFAULT_CAPABILITIES.supports_vision is False

    def test_capabilities_version(self):
        """Test capabilities version is defined."""
        from arcllm.capabilities import CAPABILITIES_VERSION

        assert CAPABILITIES_VERSION is not None
        assert len(CAPABILITIES_VERSION) > 0


class TestBaseProviderRegister:
    """Tests for provider registration in base module."""

    def test_register_all_providers_idempotent(self):
        """Test that register_all_providers can be called multiple times."""
        from arcllm.providers.base import _PROVIDERS, register_all_providers

        initial_count = len(_PROVIDERS)
        register_all_providers()
        # Should not change (providers already registered)
        assert len(_PROVIDERS) >= initial_count


class TestOpenAICacheTokenExtraction:
    """Pin OpenAI's cache + reasoning token extraction.

    Without these lifts, ``completion_cost()`` undercounts cache savings
    (treats cached prompt tokens at full rate) and over- / under-bills
    reasoning models (no separate reasoning rate applied).
    """

    @pytest.fixture
    def adapter(self):
        from arcllm.providers.openai_adapter import OpenAIAdapter

        return OpenAIAdapter(ProviderConfig(api_key="test"))

    def test_cached_tokens_promoted_to_cache_read_input_tokens(self, adapter):
        """``prompt_tokens_details.cached_tokens`` must be lifted into
        ``Usage.cache_read_input_tokens`` so ``cost_per_token`` applies
        the cached rate to those tokens."""
        body = json.dumps(
            {
                "id": "x",
                "object": "chat.completion",
                "created": 0,
                "model": "gpt-4o-mini",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "hi"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 1000,
                    "completion_tokens": 50,
                    "total_tokens": 1050,
                    "prompt_tokens_details": {"cached_tokens": 800},
                },
            }
        ).encode("utf-8")
        resp = adapter.parse_response(body, model="gpt-4o-mini")
        assert resp.usage is not None
        assert resp.usage.cache_read_input_tokens == 800
        # The full details dict is preserved too so other introspection
        # paths keep working.
        assert resp.usage.prompt_tokens_details == {"cached_tokens": 800}

    def test_reasoning_tokens_round_trip_through_details(self, adapter):
        """``completion_tokens_details.reasoning_tokens`` must round-trip
        so ``completion_cost`` can apply the reasoning rate when the
        model exposes one."""
        body = json.dumps(
            {
                "id": "x",
                "object": "chat.completion",
                "created": 0,
                "model": "o3-mini",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "answer"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 600,
                    "total_tokens": 700,
                    "completion_tokens_details": {"reasoning_tokens": 500},
                },
            }
        ).encode("utf-8")
        resp = adapter.parse_response(body, model="o3-mini")
        assert resp.usage is not None
        assert resp.usage.completion_tokens_details == {"reasoning_tokens": 500}

    def test_cached_tokens_none_when_no_details(self, adapter):
        """Absent ``prompt_tokens_details`` must leave
        ``cache_read_input_tokens`` as ``None`` (not a spurious zero) so
        downstream cost code can distinguish ``no caching`` from
        ``caching with no hits``."""
        body = json.dumps(
            {
                "id": "x",
                "object": "chat.completion",
                "created": 0,
                "model": "gpt-3.5-turbo",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "hi"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15,
                },
            }
        ).encode("utf-8")
        resp = adapter.parse_response(body, model="gpt-3.5-turbo")
        assert resp.usage is not None
        assert resp.usage.cache_read_input_tokens is None


class TestDeepSeekCacheTokenExtraction:
    """DeepSeek reports cache hits as a top-level ``prompt_cache_hit_tokens``
    field instead of OpenAI's nested ``prompt_tokens_details.cached_tokens``.
    The adapter must lift it into the canonical ``cache_read_input_tokens``
    so the 90%-off cache rate gets applied."""

    @pytest.fixture
    def adapter(self):
        from arcllm.providers.deepseek_adapter import DeepSeekAdapter

        return DeepSeekAdapter(ProviderConfig(api_key="test"))

    def test_prompt_cache_hit_tokens_lifted(self, adapter):
        body = json.dumps(
            {
                "id": "x",
                "object": "chat.completion",
                "created": 0,
                "model": "deepseek-chat",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "hi"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 1000,
                    "completion_tokens": 50,
                    "total_tokens": 1050,
                    "prompt_cache_hit_tokens": 800,
                    "prompt_cache_miss_tokens": 200,
                },
            }
        ).encode("utf-8")
        resp = adapter.parse_response(body, model="deepseek-chat")
        assert resp.usage is not None
        assert resp.usage.cache_read_input_tokens == 800


class TestDatabricksCacheTokenExtraction:
    """Databricks-hosted Claude returns Anthropic-shaped cache fields at
    the top level of ``usage``. The adapter (which inherits from
    OpenAIAdapter) must lift both ``cache_read_input_tokens`` and
    ``cache_creation_input_tokens``."""

    @pytest.fixture
    def adapter(self):
        from arcllm.providers.databricks_adapter import DatabricksAdapter

        return DatabricksAdapter(
            ProviderConfig(api_key="test", api_base="https://x.databricks.com")
        )

    def test_anthropic_cache_fields_lifted(self, adapter):
        body = json.dumps(
            {
                "id": "x",
                "object": "chat.completion",
                "created": 0,
                "model": "databricks-claude-sonnet-4-5",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "hi"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 1200,
                    "completion_tokens": 30,
                    "total_tokens": 1230,
                    "cache_read_input_tokens": 600,
                    "cache_creation_input_tokens": 400,
                },
            }
        ).encode("utf-8")
        resp = adapter.parse_response(body, model="databricks-claude-sonnet-4-5")
        assert resp.usage is not None
        assert resp.usage.cache_read_input_tokens == 600
        assert resp.usage.cache_creation_input_tokens == 400


class TestGeminiCacheTokenExtraction:
    """Gemini's context caching reports cache hits via
    ``usageMetadata.cachedContentTokenCount``. The adapter must lift it
    into ``cache_read_input_tokens`` so the 90%-off cache rate applies."""

    @pytest.fixture
    def adapter(self):
        from arcllm.providers.gemini_adapter import GeminiAdapter

        return GeminiAdapter(ProviderConfig(api_key="test"))

    def test_cached_content_token_count_lifted(self, adapter):
        body = json.dumps(
            {
                "candidates": [
                    {
                        "content": {"role": "model", "parts": [{"text": "hi"}]},
                        "finishReason": "STOP",
                    }
                ],
                "usageMetadata": {
                    "promptTokenCount": 1000,
                    "candidatesTokenCount": 50,
                    "totalTokenCount": 1050,
                    "cachedContentTokenCount": 800,
                },
            }
        ).encode("utf-8")
        resp = adapter.parse_response(body, model="gemini-2.5-pro")
        assert resp.usage is not None
        assert resp.usage.cache_read_input_tokens == 800

    def test_no_cached_content_means_none(self, adapter):
        body = json.dumps(
            {
                "candidates": [
                    {
                        "content": {"role": "model", "parts": [{"text": "hi"}]},
                        "finishReason": "STOP",
                    }
                ],
                "usageMetadata": {
                    "promptTokenCount": 10,
                    "candidatesTokenCount": 5,
                    "totalTokenCount": 15,
                },
            }
        ).encode("utf-8")
        resp = adapter.parse_response(body, model="gemini-2.5-pro")
        assert resp.usage is not None
        assert resp.usage.cache_read_input_tokens is None


class TestOpenRouterCostCapture:
    """OpenRouter returns the actual billed USD cost in the
    ``x-openrouter-cost`` response header — authoritative across the
    routed provider's markup. The adapter lifts it into
    ``ModelResponse.provider_reported_cost`` so ``completion_cost()``
    returns it directly without a static-table lookup."""

    @pytest.fixture
    def adapter(self):
        from arcllm.providers.openrouter_adapter import OpenRouterAdapter

        return OpenRouterAdapter(ProviderConfig(api_key="test"))

    @pytest.fixture
    def body(self):
        return json.dumps(
            {
                "id": "x",
                "object": "chat.completion",
                "created": 0,
                "model": "openai/gpt-4o-mini",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "hi"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 50,
                    "total_tokens": 150,
                },
            }
        ).encode("utf-8")

    def test_x_openrouter_cost_lifted_to_provider_reported_cost(self, adapter, body):
        resp = adapter.parse_response(body, model="openai/gpt-4o-mini")
        # Standard parse: no headers visible. post_process_response lifts
        # the header.
        resp = adapter.post_process_response(resp, {"x-openrouter-cost": "0.00042"})
        assert resp.provider_reported_cost == pytest.approx(0.00042)

    def test_post_process_response_handles_mixed_case_header(self, adapter, body):
        """httpx and some servers preserve original header casing in the
        dict; the adapter must do case-insensitive lookup."""
        resp = adapter.parse_response(body, model="openai/gpt-4o-mini")
        resp = adapter.post_process_response(resp, {"X-OpenRouter-Cost": "0.00100"})
        assert resp.provider_reported_cost == pytest.approx(0.00100)

    def test_post_process_response_no_cost_header_keeps_none(self, adapter, body):
        """Without the header, provider_reported_cost stays None and
        completion_cost falls back to the static table."""
        resp = adapter.parse_response(body, model="openai/gpt-4o-mini")
        resp = adapter.post_process_response(resp, {"content-type": "application/json"})
        assert resp.provider_reported_cost is None

    def test_post_process_response_ignores_malformed_cost(self, adapter, body):
        """Malformed header value (non-numeric) is swallowed — falling
        back to None so the static table takes over rather than crashing."""
        resp = adapter.parse_response(body, model="openai/gpt-4o-mini")
        resp = adapter.post_process_response(resp, {"x-openrouter-cost": "not-a-number"})
        assert resp.provider_reported_cost is None


class TestCorePostProcessHook:
    """Core's completion() path must invoke adapter.post_process_response
    so adapter-side header lifts (e.g. OpenRouter cost) take effect."""

    def test_completion_threads_response_headers_to_post_process(self):
        """End-to-end: mock the HTTP layer to return a response with
        the OpenRouter cost header set; verify the returned
        ModelResponse.provider_reported_cost reflects it."""
        from unittest.mock import patch

        from arcllm import completion
        from arcllm.http.client import HTTPResponse

        body = json.dumps(
            {
                "id": "x",
                "object": "chat.completion",
                "created": 0,
                "model": "openai/gpt-4o-mini",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "ok"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                },
            }
        ).encode("utf-8")
        http_response = HTTPResponse(
            status_code=200,
            headers={"x-openrouter-cost": "0.00025"},
            body=body,
        )

        class _FakeClient:
            def request(self, method, url, *, headers, body, timeout, stream):
                return http_response

        with (
            patch.dict("os.environ", {"OPENROUTER_API_KEY": "test"}),
            patch("arcllm.core._get_http_client", return_value=_FakeClient()),
        ):
            resp = completion(
                model="openrouter/openai/gpt-4o-mini",
                messages=[{"role": "user", "content": "hi"}],
            )

        assert resp.provider_reported_cost == pytest.approx(0.00025)
