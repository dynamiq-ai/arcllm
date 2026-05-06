"""
Extended tests for capabilities module to improve coverage.
"""

from __future__ import annotations

from arcllm.capabilities.tables import (
    ALL_CAPABILITIES,
    ANTHROPIC_CAPABILITIES,
    GEMINI_CAPABILITIES,
    GROQ_CAPABILITIES,
    OPENAI_CAPABILITIES,
    ModelCapabilities,
    _normalize_model_name,
)


class TestCapabilitiesNormalization:
    """Tests for model name normalization in capabilities."""

    def test_normalize_simple_model(self):
        """Test normalizing model without provider prefix."""
        provider, model = _normalize_model_name("gpt-4o-mini")
        assert provider is None
        assert model == "gpt-4o-mini"

    def test_normalize_with_provider_prefix(self):
        """Test normalizing model with provider prefix."""
        provider, model = _normalize_model_name("openai/gpt-4o-mini")
        assert provider == "openai"
        assert model == "gpt-4o-mini"

    def test_normalize_hyphenated_provider(self):
        """Test normalizing with hyphenated provider name."""
        provider, model = _normalize_model_name("together-ai/model")
        assert provider == "together_ai"
        assert model == "model"


class TestCapabilitiesTables:
    """Tests for capabilities tables content."""

    def test_openai_gpt4o_capabilities(self):
        """Test GPT-4o has correct capabilities."""
        caps = OPENAI_CAPABILITIES["gpt-4o"]
        assert caps.max_tokens == 16384
        assert caps.context_window == 128000
        assert caps.supports_vision is True
        assert caps.supports_tools is True
        assert caps.supports_structured_output is True

    def test_anthropic_claude_capabilities(self):
        """Test current Claude flagship has correct capabilities."""
        caps = ANTHROPIC_CAPABILITIES["claude-sonnet-4-5-20250929"]
        assert caps.supports_vision is True
        assert caps.supports_pdf_input is True
        assert caps.supports_tools is True

    def test_gemini_long_context(self):
        """Test Gemini flagship has long context window."""
        caps = GEMINI_CAPABILITIES["gemini-2.5-pro"]
        assert caps.context_window is not None
        assert caps.context_window >= 1_000_000  # at least 1M tokens

    def test_groq_llama_capabilities(self):
        """Test Groq Llama has correct capabilities."""
        caps = GROQ_CAPABILITIES["llama-3.3-70b-versatile"]
        assert caps.supports_tools is True
        assert caps.supports_structured_output is True

    def test_all_capabilities_structure(self):
        """Test ALL_CAPABILITIES has proper structure."""
        for caps in ALL_CAPABILITIES.values():
            assert isinstance(caps, dict)
            for model_caps in caps.values():
                assert isinstance(model_caps, ModelCapabilities)


class TestModelCapabilitiesDataclass:
    """Tests for ModelCapabilities dataclass."""

    def test_capabilities_with_all_fields(self):
        """Test creating capabilities with all fields."""
        caps = ModelCapabilities(
            max_tokens=4096,
            context_window=8192,
            supports_vision=True,
            supports_pdf_input=True,
            supports_tools=True,
            supports_structured_output=True,
        )
        assert caps.max_tokens == 4096
        assert caps.context_window == 8192
        assert caps.supports_vision is True
        assert caps.supports_pdf_input is True
        assert caps.supports_tools is True
        assert caps.supports_structured_output is True

    def test_capabilities_defaults(self):
        """Test default values for capabilities."""
        caps = ModelCapabilities(max_tokens=None, context_window=None)
        assert caps.supports_vision is False
        assert caps.supports_pdf_input is False
        assert caps.supports_tools is False
        assert caps.supports_structured_output is False


class TestGetModelInfo:
    """``get_model_info`` is a litellm-compat snapshot of caps + pricing."""

    def test_known_model_returns_full_payload(self):
        from arcllm.capabilities import get_model_info

        info = get_model_info("gpt-4o-mini")
        assert info["max_tokens"] == 16384
        assert info["max_input_tokens"] == 128000
        assert info["max_output_tokens"] == 16384
        assert info["supports_function_calling"] is True
        assert info["supports_response_schema"] is True
        assert info["kind"] == "chat"
        # Pricing is converted from per-million to per-token.
        assert info["input_cost_per_token"] is not None
        assert info["output_cost_per_token"] is not None
        # Cache fields are populated when the model has cached pricing.
        if info["cache_read_input_token_cost"] is not None:
            assert info["cache_read_input_token_cost"] > 0

    def test_unknown_model_returns_defaults(self):
        from arcllm.capabilities import get_model_info

        info = get_model_info("totally-fake-model-xyz")
        # Default capabilities are mostly Falsy; pricing fields are None.
        assert info["supports_function_calling"] is False
        assert info["input_cost_per_token"] is None
        assert info["output_cost_per_token"] is None


class TestSupportsFunctionCallingAlias:
    def test_alias_matches_supports_tools(self):
        from arcllm.capabilities import supports_function_calling, supports_tools

        for model in ("gpt-4o", "claude-sonnet-4-5-20250929", "gemini-2.5-pro"):
            assert supports_function_calling(model) == supports_tools(model)


class TestGetSupportedOpenAIParams:
    def test_chat_model_has_temperature_and_response_format(self):
        from arcllm.capabilities import get_supported_openai_params

        params = get_supported_openai_params("gpt-4o-mini")
        assert "temperature" in params
        assert "top_p" in params
        assert "tools" in params
        assert "response_format" in params
        # GPT-4o-mini is not a reasoning model, so reasoning_effort isn't listed.
        assert "reasoning_effort" not in params

    def test_reasoning_model_drops_temperature_and_lists_reasoning_effort(self):
        from arcllm.capabilities import get_supported_openai_params

        params = get_supported_openai_params("o4-mini")
        assert "temperature" not in params
        assert "top_p" not in params
        assert "reasoning_effort" in params

    def test_anthropic_drops_response_format(self):
        from arcllm.capabilities import get_supported_openai_params

        # Anthropic does not implement OpenAI-style response_format.
        params = get_supported_openai_params("claude-sonnet-4-5-20250929")
        assert "response_format" not in params
        assert "tools" in params  # tools are still supported
