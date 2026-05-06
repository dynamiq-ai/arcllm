"""
Common tests for provider adapters.

Tests base adapter functionality, provider registration, and common behaviors
across all provider adapters.
"""

from __future__ import annotations

import os
from typing import Any, ClassVar
from unittest.mock import patch

import pytest

from arcllm.exceptions import (
    ArcLLMError,
    UnsupportedModelError,
    UnsupportedParameterError,
)
from arcllm.providers.base import (
    COMMON_PARAMS,
    SUPPORTED_PROVIDERS,
    BaseAdapter,
    ProviderConfig,
    RequestData,
    get_provider,
    parse_model_string,
    register_all_providers,
    register_provider,
)


class TestParseModelString:
    """Tests for parse_model_string function."""

    @pytest.mark.parametrize(
        "model,expected_provider,expected_model",
        [
            ("openai/gpt-4o", "openai", "gpt-4o"),
            ("anthropic/claude-3-5-sonnet", "anthropic", "claude-3-5-sonnet"),
            ("gemini/gemini-1.5-pro", "gemini", "gemini-1.5-pro"),
            ("together_ai/llama-3.3", "together_ai", "llama-3.3"),
            ("together-ai/llama-3.3", "together_ai", "llama-3.3"),  # Hyphen variant
            ("azure/my-deployment", "azure", "my-deployment"),
            ("vertex_ai/gemini-pro", "vertex_ai", "gemini-pro"),
        ],
    )
    def test_explicit_provider(self, model, expected_provider, expected_model):
        """Test parsing model with explicit provider prefix."""
        provider, model_id = parse_model_string(model)
        assert provider == expected_provider
        assert model_id == expected_model

    @pytest.mark.parametrize(
        "model,expected_provider",
        [
            ("gpt-4o", "openai"),
            ("gpt-4o-mini", "openai"),
            ("gpt-3.5-turbo", "openai"),
            ("o1", "openai"),
            ("o1-mini", "openai"),
            ("claude-3-5-sonnet", "anthropic"),
            ("claude-2.1", "anthropic"),
            ("gemini-1.5-pro", "gemini"),
            ("gemini-pro", "gemini"),
            ("mistral-large", "mistral"),
            ("codestral-latest", "mistral"),
            ("command-r-plus", "cohere"),
            ("deepseek-chat", "deepseek"),
            ("llama-3.3-70b", "groq"),
            ("llama3-8b-8192", "groq"),
            ("pplx-online", "perplexity"),
        ],
    )
    def test_inferred_provider(self, model, expected_provider):
        """Test provider inference from model name."""
        provider, model_id = parse_model_string(model)
        assert provider == expected_provider
        assert model_id == model

    def test_unknown_model_defaults_to_openai(self):
        """Test that unknown models default to OpenAI."""
        provider, model_id = parse_model_string("some-random-model")
        assert provider == "openai"
        assert model_id == "some-random-model"


class TestProviderConfig:
    """Tests for ProviderConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = ProviderConfig()
        assert config.api_key is None
        assert config.api_base is None
        assert config.timeout == 60.0
        assert config.max_retries == 3
        assert config.extra_headers == {}

    def test_custom_config(self):
        """Test custom configuration."""
        config = ProviderConfig(
            api_key="test-key",
            api_base="https://custom.api.com",
            timeout=30.0,
            max_retries=5,
            extra_headers={"X-Custom": "header"},
        )
        assert config.api_key == "test-key"
        assert config.api_base == "https://custom.api.com"
        assert config.timeout == 30.0
        assert config.max_retries == 5
        assert config.extra_headers == {"X-Custom": "header"}

    def test_azure_specific_config(self):
        """Test Azure-specific configuration."""
        config = ProviderConfig(
            azure_deployment="my-deployment",
            azure_ad_token="ad-token",
        )
        assert config.azure_deployment == "my-deployment"
        assert config.azure_ad_token == "ad-token"

    def test_aws_specific_config(self):
        """Test AWS Bedrock-specific configuration."""
        config = ProviderConfig(
            aws_region="us-east-1",
            aws_access_key_id="AKIAIOSFODNN7EXAMPLE",
            aws_secret_access_key="secret",
            aws_session_token="session-token",
        )
        assert config.aws_region == "us-east-1"
        assert config.aws_access_key_id == "AKIAIOSFODNN7EXAMPLE"
        assert config.aws_session_token == "session-token"

    def test_vertex_specific_config(self):
        """Test Vertex AI-specific configuration."""
        config = ProviderConfig(
            vertex_project="my-project",
            vertex_location="us-central1",
        )
        assert config.vertex_project == "my-project"
        assert config.vertex_location == "us-central1"


class TestGetProvider:
    """Tests for get_provider function."""

    def test_get_registered_provider(self):
        """Test getting a registered provider."""
        # Ensure providers are registered
        register_all_providers()

        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            adapter = get_provider("openai")
            assert adapter.provider_name == "openai"

    def test_get_provider_case_insensitive(self):
        """Test that provider lookup is case-insensitive."""
        register_all_providers()

        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            adapter = get_provider("OPENAI")
            assert adapter.provider_name == "openai"

    def test_unknown_provider_raises_error(self):
        """Test that unknown provider raises UnsupportedModelError."""
        with pytest.raises(UnsupportedModelError) as exc_info:
            get_provider("unknown_provider_xyz")

        assert "unknown_provider_xyz" in str(exc_info.value)

    def test_provider_with_custom_config(self):
        """Test getting provider with custom configuration."""
        register_all_providers()

        config = ProviderConfig(
            api_key="custom-key",
            api_base="https://custom.api.com",
        )
        with patch.dict(os.environ, {}, clear=False):
            adapter = get_provider("openai", config)
            assert adapter.config.api_key == "custom-key"
            assert adapter.config.api_base == "https://custom.api.com"


class TestSupportedProviders:
    """Tests for supported providers list."""

    def test_supported_providers_list(self):
        """Test that SUPPORTED_PROVIDERS contains all expected providers."""
        expected = [
            "openai",
            "azure",
            "anthropic",
            "gemini",
            "vertex_ai",
            "bedrock",
            "mistral",
            "cohere",
            "groq",
            "together_ai",
            "fireworks_ai",
            "deepseek",
            "perplexity",
            "databricks",
            "ollama",
        ]
        for provider in expected:
            assert provider in SUPPORTED_PROVIDERS


class TestCommonParams:
    """Tests for common parameters set."""

    def test_common_params_contains_essential(self):
        """Test that COMMON_PARAMS contains essential parameters."""
        essential = [
            "model",
            "messages",
            "stream",
            "temperature",
            "top_p",
            "max_tokens",
            "stop",
            "tools",
            "tool_choice",
            "response_format",
        ]
        for param in essential:
            assert param in COMMON_PARAMS


class TestRegisterProvider:
    """Tests for provider registration."""

    def test_register_custom_provider(self):
        """Test registering a custom provider."""

        class CustomAdapter:
            provider_name = "custom"

            def __init__(self, config):
                self.config = config

        register_provider("custom", CustomAdapter)
        adapter = get_provider("custom", ProviderConfig())
        assert adapter.provider_name == "custom"


class TestRequestData:
    """Tests for RequestData dataclass."""

    def test_create_request_data(self):
        """Test creating request data."""
        request = RequestData(
            method="POST",
            url="https://api.example.com/v1/chat/completions",
            headers={"Authorization": "Bearer test-key"},
            body=b'{"model": "test"}',
            timeout=30.0,
        )
        assert request.method == "POST"
        assert request.url == "https://api.example.com/v1/chat/completions"
        assert request.headers["Authorization"] == "Bearer test-key"
        assert request.body == b'{"model": "test"}'
        assert request.timeout == 30.0

    def test_request_data_default_timeout(self):
        """Test default timeout value."""
        request = RequestData(
            method="GET",
            url="https://api.example.com",
            headers={},
        )
        assert request.timeout == 60.0

    def test_request_data_no_body(self):
        """Test request with no body."""
        request = RequestData(
            method="GET",
            url="https://api.example.com",
            headers={},
        )
        assert request.body is None


class _MockAdapter(BaseAdapter):
    """Mock adapter implementation for testing."""

    provider_name = "test"
    supported_params: ClassVar[set[str]] = {
        "temperature",
        "max_tokens",
        "model",
        "messages",
        "stream",
    }

    def build_request(self, **kwargs: Any) -> RequestData:
        return RequestData(method="POST", url="test", headers={})

    def parse_response(self, data: bytes, model: str) -> Any:
        return None

    def parse_stream_event(self, data: str, model: str) -> Any:
        return None

    def parse_error(
        self, status_code: int, data: bytes, request_id: str | None = None
    ) -> ArcLLMError:
        return ArcLLMError("test error")


class TestBaseAdapterParamChecking:
    """Tests for BaseAdapter parameter checking."""

    def test_check_params_passes_supported(self):
        """Test that supported params pass through."""
        config = ProviderConfig(api_key="test")
        adapter = _MockAdapter(config)

        result = adapter._check_params(
            "gpt-4o-mini",  # known model that accepts temperature
            drop_params=False,
            temperature=0.7,
            max_tokens=100,
        )
        assert result["temperature"] == 0.7
        assert result["max_tokens"] == 100

    def test_check_params_raises_unsupported(self):
        """Test that adapter-level unsupported params raise when drop_params=False."""
        config = ProviderConfig(api_key="test")
        adapter = _MockAdapter(config)

        with pytest.raises(UnsupportedParameterError) as exc_info:
            adapter._check_params(
                "gpt-4o-mini",
                drop_params=False,
                temperature=0.7,
                unsupported_param="value",
            )

        assert "unsupported_param" in str(exc_info.value)

    def test_check_params_drops_unsupported(self):
        """Test that adapter-level unsupported params are dropped when drop_params=True."""
        config = ProviderConfig(api_key="test")
        adapter = _MockAdapter(config)

        result = adapter._check_params(
            "gpt-4o-mini",
            drop_params=True,
            temperature=0.7,
            unsupported_param="value",
        )

        assert "temperature" in result
        assert "unsupported_param" not in result

    def test_capability_filter_drops_temperature_for_reasoning_models(self):
        """Reasoning models reject ``temperature``; the model-level filter drops it
        with a warning regardless of ``drop_params``."""
        import warnings

        config = ProviderConfig(api_key="test")
        adapter = _MockAdapter(config)

        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            result = adapter._check_params(
                "o4-mini",  # reasoning model in the capability table
                drop_params=False,
                temperature=0.5,
                max_tokens=100,
            )

        assert "temperature" not in result
        assert "max_tokens" in result
        assert any(
            "temperature" in str(w.message) and "o4-mini" in str(w.message) for w in captured
        ), (
            f"Expected a warning about temperature for o4-mini; got {[str(w.message) for w in captured]}"
        )

    def test_capability_filter_skips_unknown_models(self):
        """Unknown models bypass the capability filter (we trust the user)."""
        config = ProviderConfig(api_key="test")
        adapter = _MockAdapter(config)

        result = adapter._check_params(
            "some-future-model-not-in-table",
            drop_params=False,
            temperature=0.5,
            max_tokens=100,
        )
        assert result["temperature"] == 0.5
        assert result["max_tokens"] == 100

    def test_get_api_key_from_config(self):
        """Test getting API key from config."""
        config = ProviderConfig(api_key="config-key")
        adapter = _MockAdapter(config)

        key = adapter._get_api_key("TEST_API_KEY")
        assert key == "config-key"

    def test_get_api_key_from_env(self):
        """Test getting API key from environment."""
        config = ProviderConfig()
        adapter = _MockAdapter(config)

        with patch.dict(os.environ, {"TEST_API_KEY": "env-key"}):
            key = adapter._get_api_key("TEST_API_KEY")
            assert key == "env-key"

    def test_get_api_key_raises_when_missing(self):
        """Test that missing API key raises error."""
        config = ProviderConfig()
        adapter = _MockAdapter(config)

        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ArcLLMError) as exc_info:
                adapter._get_api_key("MISSING_API_KEY")

            assert "MISSING_API_KEY" in str(exc_info.value)
