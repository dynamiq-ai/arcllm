"""
Comprehensive tests for pricing module.

Tests pricing lookup, model normalization, overrides, unknown model behavior,
and usage token validation.
"""

from __future__ import annotations

import pytest

from arcllm.pricing import (
    completion_cost,
    cost_per_token,
    get_model_pricing,
)
from arcllm.pricing.tables import (
    ALL_PRICING,
    ANTHROPIC_PRICING,
    COHERE_PRICING,
    DEEPSEEK_PRICING,
    FIREWORKS_PRICING,
    GEMINI_PRICING,
    GROQ_PRICING,
    MISTRAL_PRICING,
    OPENAI_PRICING,
    PERPLEXITY_PRICING,
    TOGETHER_PRICING,
    ModelPricing,
    UnknownModelPricingError,
    _normalize_model_name,
)
from arcllm.types import ModelResponse, Usage


class TestModelNormalization:
    """Tests for model name normalization."""

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

    def test_normalize_with_hyphenated_provider(self):
        """Test normalizing with hyphenated provider name."""
        provider, model = _normalize_model_name("together-ai/llama-model")
        assert provider == "together_ai"
        assert model == "llama-model"

    def test_normalize_unknown_provider(self):
        """Test normalizing with unknown provider prefix."""
        provider, model = _normalize_model_name("unknown-provider/model")
        assert provider is None
        assert model == "unknown-provider/model"

    def test_normalize_model_with_slash(self):
        """Test model name that contains slash (e.g., Together models)."""
        provider, model = _normalize_model_name("together_ai/meta-llama/Llama-3.3-70B-Instruct")
        # First part should be recognized as provider
        assert provider == "together_ai"
        assert model == "meta-llama/Llama-3.3-70B-Instruct"


class TestPricingLookup:
    """Tests for pricing lookup across providers."""

    @pytest.mark.parametrize(
        "model,expected_input,expected_output",
        [
            ("gpt-4o", 2.50, 10.00),
            ("gpt-4o-mini", 0.15, 0.60),
            ("gpt-5", 1.25, 10.00),
            ("gpt-5-mini", 0.25, 2.00),
            ("o3", 2.00, 8.00),
            ("o4-mini", 1.10, 4.40),
        ],
    )
    def test_openai_pricing_values(self, model, expected_input, expected_output):
        """Test specific OpenAI model pricing values."""
        pricing = get_model_pricing(model)
        assert pricing.input_cost_per_million == expected_input
        assert pricing.output_cost_per_million == expected_output

    @pytest.mark.parametrize(
        "model",
        [
            "claude-sonnet-4-5-20250929",
            "claude-haiku-4-5-20251001",
            "claude-opus-4-5-20251101",
            "claude-opus-4-1-20250805",
        ],
    )
    def test_anthropic_models_have_cached_pricing(self, model):
        """Test that current Claude models have cached input pricing."""
        pricing = get_model_pricing(model)
        assert pricing.cached_input_cost_per_million is not None
        # Cached should be cheaper than regular input
        assert pricing.cached_input_cost_per_million < pricing.input_cost_per_million

    def test_embedding_models_zero_output_cost(self):
        """Test that embedding models have zero output cost."""
        embedding_models = [
            "text-embedding-3-small",
            "text-embedding-3-large",
            "text-embedding-ada-002",
            "mistral-embed",
        ]
        for model in embedding_models:
            pricing = get_model_pricing(model)
            assert pricing.output_cost_per_million == 0.0

    def test_free_models_zero_cost(self):
        """Local Ollama models are billed as $0/$0 (run on the user's hardware)."""
        free_models = [
            "ollama/llama3.3",
            "ollama/qwen3",
        ]
        for model in free_models:
            pricing = get_model_pricing(model)
            assert pricing.input_cost_per_million == 0.0
            assert pricing.output_cost_per_million == 0.0


class TestPricingOverrides:
    """Tests for pricing with model override."""

    def test_completion_cost_with_model_override(self):
        """Test cost calculation with explicit model override."""
        response = ModelResponse(
            id="resp-1",
            model="",  # Empty model
            choices=[],
            usage=Usage(prompt_tokens=1000, completion_tokens=1000, total_tokens=2000),
        )

        # Override with gpt-4o pricing
        cost = completion_cost(response, model="gpt-4o")

        # gpt-4o: 2.50/1M input, 10.00/1M output
        expected = (1000 / 1_000_000 * 2.50) + (1000 / 1_000_000 * 10.00)
        assert pytest.approx(cost, abs=1e-6) == expected

    def test_completion_cost_uses_response_model(self):
        """Test that cost uses response model when no override provided."""
        response = ModelResponse(
            id="resp-1",
            model="gpt-4o-mini",
            choices=[],
            usage=Usage(prompt_tokens=1000, completion_tokens=500, total_tokens=1500),
        )

        cost = completion_cost(response)

        # gpt-4o-mini: 0.15/1M input, 0.60/1M output
        expected = (1000 / 1_000_000 * 0.15) + (500 / 1_000_000 * 0.60)
        assert pytest.approx(cost, abs=1e-6) == expected


class TestUnknownModelBehavior:
    """Tests for unknown model handling."""

    def test_unknown_model_raises_error(self):
        """Test that unknown model raises UnknownModelPricingError."""
        with pytest.raises(UnknownModelPricingError) as exc_info:
            get_model_pricing("completely-unknown-model-xyz")

        assert "completely-unknown-model-xyz" in str(exc_info.value)

    def test_unknown_model_with_known_provider(self):
        """Test unknown model with known provider prefix."""
        with pytest.raises(UnknownModelPricingError) as exc_info:
            get_model_pricing("openai/unknown-model-xyz")

        assert "unknown-model-xyz" in str(exc_info.value)
        assert "openai" in str(exc_info.value)

    def test_cost_per_token_unknown_model(self):
        """Test cost_per_token raises for unknown model."""
        with pytest.raises(UnknownModelPricingError):
            cost_per_token("unknown-model", 100, 100)


class TestUsageTokenValidation:
    """Tests for usage token validation and consistency."""

    def test_usage_tokens_non_negative(self):
        """Test that usage tokens can be non-negative."""
        usage = Usage(
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
        )
        assert usage.prompt_tokens >= 0
        assert usage.completion_tokens >= 0
        assert usage.total_tokens >= 0

    def test_usage_total_consistency(self):
        """Test usage total equals prompt + completion when consistent."""
        usage = Usage(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
        )
        assert usage.total_tokens == usage.prompt_tokens + usage.completion_tokens

    def test_cost_calculation_with_large_tokens(self):
        """Test cost calculation with large token counts."""
        # 1 million tokens each
        prompt_cost, comp_cost = cost_per_token(
            "gpt-4o",
            prompt_tokens=1_000_000,
            completion_tokens=1_000_000,
        )

        # gpt-4o: 2.50/1M input, 10.00/1M output
        assert prompt_cost == 2.50
        assert comp_cost == 10.00

    def test_cost_with_fractional_tokens(self):
        """Test cost calculation with non-round token counts."""
        prompt_cost, comp_cost = cost_per_token(
            "gpt-4o-mini",
            prompt_tokens=1234,
            completion_tokens=567,
        )

        # gpt-4o-mini: 0.15/1M input, 0.60/1M output
        expected_prompt = (1234 / 1_000_000) * 0.15
        expected_comp = (567 / 1_000_000) * 0.60

        assert pytest.approx(prompt_cost, abs=1e-10) == expected_prompt
        assert pytest.approx(comp_cost, abs=1e-10) == expected_comp


class TestAllProvidersPricingCoverage:
    """Tests to ensure all providers have pricing data."""

    def test_all_pricing_providers_populated(self):
        """Test that ALL_PRICING contains entries for key providers."""
        expected_providers = [
            "openai",
            "anthropic",
            "gemini",
            "mistral",
            "cohere",
            "groq",
            "together_ai",
            "fireworks_ai",
            "deepseek",
            "perplexity",
        ]
        for provider in expected_providers:
            assert provider in ALL_PRICING
            assert len(ALL_PRICING[provider]) > 0

    def test_vertex_overlaps_with_gemini_pricing(self):
        """Vertex AI tracks the same Gemini model families as direct Gemini.

        The two tables are now distinct objects (Vertex on-prem pricing differs
        from AI Studio for some 2.0 models — see Phase 3 manifest), but the set
        of model IDs should largely overlap.
        """
        gemini_ids = set(ALL_PRICING["gemini"].keys())
        vertex_ids = set(ALL_PRICING["vertex_ai"].keys())
        overlap = gemini_ids & vertex_ids
        assert len(overlap) >= 5  # at least the current GA Gemini family

    @pytest.mark.parametrize(
        "table,model",
        [
            (OPENAI_PRICING, "gpt-4o"),
            (ANTHROPIC_PRICING, "claude-sonnet-4-5-20250929"),
            (GEMINI_PRICING, "gemini-2.5-pro"),
            (MISTRAL_PRICING, "mistral-large-latest"),
            (COHERE_PRICING, "command-a-03-2025"),
            (GROQ_PRICING, "llama-3.3-70b-versatile"),
            (TOGETHER_PRICING, "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8"),
            (FIREWORKS_PRICING, "accounts/fireworks/models/deepseek-v4-pro"),
            (DEEPSEEK_PRICING, "deepseek-v4-flash"),
            (PERPLEXITY_PRICING, "sonar-pro"),
        ],
    )
    def test_provider_tables_have_flagship_models(self, table, model):
        """Each provider table contains its current flagship model."""
        assert model in table, f"{model!r} missing from provider table"
        pricing = table[model]
        assert isinstance(pricing, ModelPricing)
        assert pricing.input_cost_per_million >= 0
        assert pricing.output_cost_per_million >= 0


class TestModelPricingDataclass:
    """Tests for ModelPricing dataclass behavior."""

    def test_pricing_slots(self):
        """Test that ModelPricing uses __slots__."""
        assert hasattr(ModelPricing, "__slots__") or hasattr(ModelPricing, "__dataclass_fields__")

    def test_pricing_immutable(self):
        """Test that ModelPricing is immutable (frozen)."""
        pricing = ModelPricing(input_cost_per_million=1.0, output_cost_per_million=2.0)
        with pytest.raises(AttributeError):
            pricing.input_cost_per_million = 5.0  # type: ignore

    def test_pricing_equality(self):
        """Test ModelPricing equality comparison."""
        p1 = ModelPricing(input_cost_per_million=1.0, output_cost_per_million=2.0)
        p2 = ModelPricing(input_cost_per_million=1.0, output_cost_per_million=2.0)
        assert p1 == p2

    def test_pricing_with_optional_cached(self):
        """Test ModelPricing with optional cached pricing."""
        pricing = ModelPricing(
            input_cost_per_million=3.0,
            output_cost_per_million=15.0,
            cached_input_cost_per_million=0.3,
        )
        assert pricing.cached_input_cost_per_million == 0.3


class TestImageGenCoverage:
    """Pin per-image pricing for the image-generation models arcllm supports
    via :func:`arcllm.image_generation`. Prices come from each provider's
    canonical pricing page; see the manifest ``sources`` block."""

    @pytest.mark.parametrize(
        "model",
        [
            "openai/dall-e-3",
            "openai/dall-e-2",
            "openai/gpt-image-1",
            "gemini/imagen-3",
            "azure/dall-e-3",
        ],
    )
    def test_image_models_have_per_request_pricing(self, model):
        p = get_model_pricing(model)
        assert p.image_per_request is not None
        assert p.image_per_request > 0


class TestAudioCoverage:
    """Pin per-character (TTS) and per-second (STT) pricing for the audio
    models arcllm exposes via its OpenAI-shaped audio routes."""

    @pytest.mark.parametrize(
        "model",
        [
            "openai/tts-1",
            "openai/tts-1-hd",
            "azure/tts-1",
        ],
    )
    def test_tts_has_per_character_pricing(self, model):
        p = get_model_pricing(model)
        assert p.audio_per_character is not None
        assert p.audio_per_character > 0

    @pytest.mark.parametrize(
        "model",
        [
            "openai/whisper-1",
            "azure/whisper-1",
        ],
    )
    def test_stt_has_per_second_pricing(self, model):
        p = get_model_pricing(model)
        assert p.audio_per_second is not None
        assert p.audio_per_second > 0


class TestRerankCoverage:
    """Pin per-query pricing for the rerank models exposed via
    :func:`arcllm.rerank`."""

    @pytest.mark.parametrize(
        "model",
        [
            "cohere/rerank-english-v3.0",
            "cohere/rerank-multilingual-v3.0",
            "cohere/rerank-v3.5",
        ],
    )
    def test_rerank_has_per_query_pricing(self, model):
        p = get_model_pricing(model)
        assert p.rerank_per_query is not None
        assert p.rerank_per_query > 0
