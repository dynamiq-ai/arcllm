"""
Tests for arcllm.capabilities module.
"""

import pytest

from arcllm.capabilities import (
    CAPABILITIES_VERSION,
    ModelCapabilities,
    get_max_tokens,
    get_model_capabilities,
    supports_pdf_input,
    supports_structured_output,
    supports_tools,
    supports_vision,
)


class TestGetModelCapabilities:
    """Tests for get_model_capabilities function."""

    def test_get_openai_capabilities(self):
        """Test getting OpenAI model capabilities."""
        caps = get_model_capabilities("gpt-4o")
        assert caps.max_tokens == 16384
        assert caps.context_window == 128000
        assert caps.supports_vision is True
        assert caps.supports_tools is True
        assert caps.supports_structured_output is True

    def test_get_capabilities_with_prefix(self):
        """Test getting capabilities with provider prefix."""
        caps = get_model_capabilities("openai/gpt-4o-mini")
        assert caps.max_tokens == 16384

    def test_get_anthropic_capabilities(self):
        """Test getting Anthropic model capabilities."""
        caps = get_model_capabilities("claude-sonnet-4-5-20250929")
        assert caps.max_tokens == 64000
        assert caps.context_window == 200000
        assert caps.supports_vision is True
        assert caps.supports_pdf_input is True
        assert caps.supports_tools is True

    def test_get_gemini_capabilities(self):
        """Test getting Gemini model capabilities."""
        caps = get_model_capabilities("gemini-2.5-pro")
        assert caps.supports_vision is True
        assert caps.supports_pdf_input is True

    def test_unknown_model_returns_defaults(self):
        """Test unknown model returns default capabilities."""
        caps = get_model_capabilities("unknown-model-xyz")
        assert caps.max_tokens == 4096
        assert caps.context_window == 8192
        assert caps.supports_vision is False

    def test_capabilities_version_exists(self):
        """Test capabilities version is defined."""
        assert CAPABILITIES_VERSION is not None


class TestGetMaxTokens:
    """Tests for get_max_tokens function."""

    def test_get_max_tokens_gpt4o(self):
        """Test max tokens for GPT-4o."""
        max_tokens = get_max_tokens("gpt-4o")
        assert max_tokens == 16384

    def test_get_max_tokens_o1(self):
        """Test max tokens for o1 (large output)."""
        max_tokens = get_max_tokens("o1")
        assert max_tokens == 100000

    def test_get_max_tokens_unknown(self):
        """Test max tokens for unknown model returns default."""
        max_tokens = get_max_tokens("unknown-model")
        assert max_tokens == 4096


class TestSupportsVision:
    """Tests for supports_vision function."""

    def test_gpt4o_supports_vision(self):
        """Test GPT-4o supports vision."""
        assert supports_vision("gpt-4o") is True

    def test_gpt4o_mini_supports_vision(self):
        """Test GPT-4o-mini supports vision."""
        assert supports_vision("gpt-4o-mini") is True

    def test_text_only_cohere_no_vision(self):
        """Cohere command-r-plus is text-only — should not report vision."""
        assert supports_vision("command-r-plus-08-2024") is False

    def test_claude_supports_vision(self):
        """Test Claude supports vision."""
        assert supports_vision("claude-sonnet-4-5-20250929") is True

    def test_gemini_supports_vision(self):
        """Test Gemini supports vision."""
        assert supports_vision("gemini-2.5-pro") is True

    def test_unknown_model_no_vision(self):
        """Test unknown model doesn't support vision."""
        assert supports_vision("unknown-model") is False


class TestSupportsPdfInput:
    """Tests for supports_pdf_input function."""

    def test_claude_sonnet_supports_pdf(self):
        """Test Claude 3.5 Sonnet supports PDF."""
        assert supports_pdf_input("claude-sonnet-4-5-20250929") is True

    def test_gemini_supports_pdf(self):
        """Test Gemini supports PDF."""
        assert supports_pdf_input("gemini-2.5-pro") is True

    def test_gpt4o_no_pdf(self):
        """Test GPT-4o doesn't support PDF."""
        assert supports_pdf_input("gpt-4o") is False

    def test_unknown_model_no_pdf(self):
        """Test unknown model doesn't support PDF."""
        assert supports_pdf_input("unknown-model") is False


class TestSupportsTools:
    """Tests for supports_tools function."""

    def test_gpt4o_supports_tools(self):
        """Test GPT-4o supports tools."""
        assert supports_tools("gpt-4o") is True

    def test_gpt35_supports_tools(self):
        """Test GPT-3.5-turbo supports tools."""
        assert supports_tools("gpt-4o-mini") is True

    def test_claude_supports_tools(self):
        """Test Claude supports tools."""
        assert supports_tools("claude-sonnet-4-5-20250929") is True

    def test_o1_preview_no_tools(self):
        """Test o1-preview doesn't support tools."""
        assert supports_tools("o1-preview") is False

    def test_unknown_model_no_tools(self):
        """Test unknown model doesn't support tools."""
        assert supports_tools("unknown-model") is False


class TestSupportsStructuredOutput:
    """Tests for supports_structured_output function."""

    def test_gpt4o_supports_structured(self):
        """Test GPT-4o supports structured output."""
        assert supports_structured_output("gpt-4o") is True

    def test_gpt4o_mini_supports_structured(self):
        """Test GPT-4o-mini supports structured output."""
        assert supports_structured_output("gpt-4o-mini") is True

    def test_claude_does_not_advertise_native_structured_output(self):
        """Anthropic does not implement OpenAI-style ``response_format=json_schema``.

        Structured output is achieved via tool-call output instead, so the
        capability flag is False even on the latest Claude. arcllm callers
        should rely on ``supports_tools`` for Claude when targeting structured
        responses.
        """
        assert supports_structured_output("claude-sonnet-4-5-20250929") is False
        assert supports_structured_output("claude-opus-4-5-20251101") is False

    def test_gemini_supports_structured(self):
        """Test Gemini supports structured output."""
        assert supports_structured_output("gemini-2.5-pro") is True

    def test_unknown_model_no_structured(self):
        """Test unknown model doesn't support structured output."""
        assert supports_structured_output("unknown-model") is False


class TestModelCapabilities:
    """Tests for ModelCapabilities dataclass."""

    def test_capabilities_frozen(self):
        """Test ModelCapabilities is frozen."""
        caps = ModelCapabilities(max_tokens=100, context_window=1000)
        with pytest.raises(AttributeError):
            caps.max_tokens = 200

    def test_capabilities_defaults(self):
        """Test ModelCapabilities default values."""
        caps = ModelCapabilities(max_tokens=100, context_window=1000)
        assert caps.supports_vision is False
        assert caps.supports_pdf_input is False
        assert caps.supports_tools is False
        assert caps.supports_structured_output is False


class TestCapabilitiesCoverage:
    """Tests for capabilities coverage of common models."""

    @pytest.mark.parametrize(
        "model",
        [
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-4-turbo",
            "gpt-4o-mini",
            "o1",
            "o1-mini",
        ],
    )
    def test_openai_models_have_capabilities(self, model):
        """Test OpenAI models have capabilities defined."""
        caps = get_model_capabilities(model)
        assert caps.max_tokens is not None
        assert caps.context_window is not None

    @pytest.mark.parametrize(
        "model",
        [
            "claude-sonnet-4-5-20250929",
            "claude-opus-4-5-20251101",
            "claude-haiku-4-5-20251001",
        ],
    )
    def test_anthropic_models_have_capabilities(self, model):
        """Test Anthropic models have capabilities defined."""
        caps = get_model_capabilities(model)
        assert caps.max_tokens is not None

    @pytest.mark.parametrize(
        "model",
        [
            "gemini-2.5-pro",
            "gemini-2.5-flash",
        ],
    )
    def test_gemini_models_have_capabilities(self, model):
        """Test Gemini models have capabilities defined."""
        caps = get_model_capabilities(model)
        assert caps.supports_vision is True
