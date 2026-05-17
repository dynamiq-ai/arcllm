"""
Tests for arcllm.pricing module.
"""

import pytest

from arcllm.pricing import (
    PRICING_VERSION,
    ModelPricing,
    completion_cost,
    cost_per_token,
    get_model_pricing,
)
from arcllm.pricing.tables import UnknownModelPricingError
from arcllm.types import Choice, Message, ModelResponse, Usage


class TestGetModelPricing:
    """Tests for get_model_pricing function."""

    def test_get_openai_pricing(self):
        """Test getting OpenAI model pricing."""
        pricing = get_model_pricing("gpt-4o-mini")
        assert pricing.input_cost_per_million == 0.15
        assert pricing.output_cost_per_million == 0.60

    def test_get_openai_pricing_with_prefix(self):
        """Test getting pricing with provider prefix."""
        pricing = get_model_pricing("openai/gpt-4o-mini")
        assert pricing.input_cost_per_million == 0.15

    def test_get_anthropic_pricing(self):
        """Test getting Anthropic model pricing."""
        pricing = get_model_pricing("claude-sonnet-4-5-20250929")
        assert pricing.input_cost_per_million == 3.00
        assert pricing.output_cost_per_million == 15.00
        assert pricing.cached_input_cost_per_million == 0.30

    def test_get_gemini_pricing(self):
        """Test getting Gemini model pricing."""
        pricing = get_model_pricing("gemini-2.5-pro")
        assert pricing.input_cost_per_million == 1.25

    def test_unknown_model_raises_error(self):
        """Test unknown model raises error."""
        with pytest.raises(UnknownModelPricingError):
            get_model_pricing("unknown-model-xyz")

    def test_unknown_model_with_provider_raises_error(self):
        """Test unknown model with provider raises error."""
        with pytest.raises(UnknownModelPricingError):
            get_model_pricing("openai/unknown-model-xyz")

    def test_pricing_version_exists(self):
        """Test pricing version is defined."""
        assert PRICING_VERSION is not None
        assert len(PRICING_VERSION) > 0


class TestCostPerToken:
    """Tests for cost_per_token function."""

    def test_calculate_cost_gpt4o_mini(self):
        """Test cost calculation for gpt-4o-mini."""
        prompt_cost, completion_cost = cost_per_token(
            "gpt-4o-mini", prompt_tokens=1000, completion_tokens=500
        )
        # Input: 0.15/1M * 1000 = 0.00015
        # Output: 0.60/1M * 500 = 0.0003
        assert pytest.approx(prompt_cost, abs=1e-6) == 0.00015
        assert pytest.approx(completion_cost, abs=1e-6) == 0.0003

    def test_calculate_cost_claude(self):
        """Test cost calculation for current Claude Sonnet 4.5."""
        prompt_cost, completion_cost = cost_per_token(
            "claude-sonnet-4-5-20250929", prompt_tokens=1000000, completion_tokens=1000000
        )
        # Input: 3.00/1M * 1M = 3.00
        # Output: 15.00/1M * 1M = 15.00
        assert prompt_cost == 3.00
        assert completion_cost == 15.00

    def test_zero_tokens(self):
        """Test cost calculation with zero tokens."""
        prompt_cost, completion_cost = cost_per_token("gpt-4o-mini")
        assert prompt_cost == 0.0
        assert completion_cost == 0.0

    def test_unknown_model_raises_error(self):
        """Test unknown model raises error."""
        with pytest.raises(UnknownModelPricingError):
            cost_per_token("unknown-model", 100, 100)


class TestCompletionCost:
    """Tests for completion_cost function."""

    def test_calculate_completion_cost(self):
        """Test calculating total cost from response."""
        response = ModelResponse(
            id="resp-1",
            model="gpt-4o-mini",
            choices=[Choice(index=0, message=Message(role="assistant", content="Hi"))],
            usage=Usage(prompt_tokens=1000, completion_tokens=500, total_tokens=1500),
        )

        cost = completion_cost(response)
        # Input: 0.00015, Output: 0.0003, Total: 0.00045
        assert pytest.approx(cost, abs=1e-6) == 0.00045

    def test_completion_cost_no_usage(self):
        """Test cost with no usage returns 0."""
        response = ModelResponse(id="resp-1", model="gpt-4o-mini", choices=[], usage=None)

        cost = completion_cost(response)
        assert cost == 0.0

    def test_completion_cost_with_model_override(self):
        """Test cost calculation with model override."""
        response = ModelResponse(
            id="resp-1",
            model="",  # Empty model
            choices=[],
            usage=Usage(prompt_tokens=1000, completion_tokens=1000, total_tokens=2000),
        )

        cost = completion_cost(response, model="gpt-4o")
        # gpt-4o: 2.50/1M input, 10.00/1M output
        expected = (1000 / 1_000_000 * 2.50) + (1000 / 1_000_000 * 10.00)
        assert pytest.approx(cost, abs=1e-6) == expected

    def test_completion_cost_missing_model_raises(self):
        """Test missing model raises ValueError."""
        response = ModelResponse(
            id="resp-1",
            model="",
            choices=[],
            usage=Usage(prompt_tokens=100, completion_tokens=100, total_tokens=200),
        )

        with pytest.raises(ValueError):
            completion_cost(response)


class TestModelPricing:
    """Tests for ModelPricing dataclass."""

    def test_model_pricing_frozen(self):
        """Test ModelPricing is frozen."""
        pricing = ModelPricing(input_cost_per_million=1.0, output_cost_per_million=2.0)
        with pytest.raises(AttributeError):
            pricing.input_cost_per_million = 5.0

    def test_model_pricing_with_cached(self):
        """Test ModelPricing with cached input pricing."""
        pricing = ModelPricing(
            input_cost_per_million=3.0,
            output_cost_per_million=15.0,
            cached_input_cost_per_million=0.3,
        )
        assert pricing.cached_input_cost_per_million == 0.3


class TestPricingCoverage:
    """Tests for pricing coverage of common models."""

    @pytest.mark.parametrize(
        "model",
        [
            "gpt-5",
            "gpt-5-mini",
            "gpt-5.4",
            "gpt-5.4-mini",
            "gpt-4o",
            "gpt-4o-mini",
            "o3",
            "o4-mini",
        ],
    )
    def test_openai_models_have_pricing(self, model):
        """Test current OpenAI GA models have pricing."""
        pricing = get_model_pricing(model)
        assert pricing.input_cost_per_million >= 0
        assert pricing.output_cost_per_million >= 0

    @pytest.mark.parametrize(
        "model",
        [
            "claude-sonnet-4-5-20250929",
            "claude-haiku-4-5-20251001",
            "claude-opus-4-5-20251101",
        ],
    )
    def test_anthropic_models_have_pricing(self, model):
        """Test current Anthropic GA models have pricing."""
        pricing = get_model_pricing(model)
        assert pricing.input_cost_per_million >= 0

    @pytest.mark.parametrize(
        "model",
        [
            "gemini-2.5-pro",
            "gemini-2.5-flash",
            "gemini-2.5-flash-lite",
        ],
    )
    def test_gemini_models_have_pricing(self, model):
        """Test current Gemini GA models have pricing."""
        pricing = get_model_pricing(model)
        assert pricing.input_cost_per_million >= 0

    @pytest.mark.parametrize(
        "model",
        [
            "mistral-large-latest",
            "mistral-small-latest",
        ],
    )
    def test_mistral_models_have_pricing(self, model):
        """Test Mistral models have pricing."""
        pricing = get_model_pricing(model)
        assert pricing.input_cost_per_million >= 0


class TestExtendedModelPricing:
    """Pin the optional modality / cache-write / reasoning fields on ModelPricing."""

    def test_cache_creation_cost_field(self):
        p = ModelPricing(
            input_cost_per_million=3.0,
            output_cost_per_million=15.0,
            cached_input_cost_per_million=0.3,
            cache_creation_cost_per_million=3.75,
        )
        assert p.cache_creation_cost_per_million == 3.75

    def test_output_cost_per_reasoning_token_field(self):
        p = ModelPricing(
            input_cost_per_million=1.0,
            output_cost_per_million=2.0,
            output_cost_per_reasoning_token=8.0,
        )
        assert p.output_cost_per_reasoning_token == 8.0

    def test_image_per_request_field(self):
        p = ModelPricing(
            input_cost_per_million=0,
            output_cost_per_million=0,
            image_per_request=0.04,
        )
        assert p.image_per_request == 0.04

    def test_audio_per_character_field(self):
        p = ModelPricing(
            input_cost_per_million=0,
            output_cost_per_million=0,
            audio_per_character=1.5e-05,
        )
        assert p.audio_per_character == 1.5e-05

    def test_audio_per_second_field(self):
        p = ModelPricing(
            input_cost_per_million=0,
            output_cost_per_million=0,
            audio_per_second=1e-04,
        )
        assert p.audio_per_second == 1e-04

    def test_rerank_per_query_field(self):
        p = ModelPricing(
            input_cost_per_million=0,
            output_cost_per_million=0,
            rerank_per_query=0.002,
        )
        assert p.rerank_per_query == 0.002


class TestCostFormulaWithExtendedFields:
    """The cost_per_token formula must honor the new optional fields."""

    def test_reasoning_tokens_use_separate_rate_when_provided(self):
        """If output_cost_per_reasoning_token is set, reasoning_tokens are
        billed at that rate instead of the standard output rate. Real
        provider responses (OpenAI o-series, DeepSeek-R1) include
        reasoning_tokens *within* completion_tokens, so we subtract first
        to avoid double-counting."""
        # Construct a synthetic pricing entry by monkey-patching the
        # lookup so the test doesn't depend on any specific real model.
        from unittest.mock import patch

        synthetic = ModelPricing(
            input_cost_per_million=1.0,
            output_cost_per_million=2.0,
            output_cost_per_reasoning_token=8.0,
        )
        with patch("arcllm.pricing.tables.get_model_pricing", return_value=synthetic):
            _prompt, completion = cost_per_token(
                "synthetic/model",
                prompt_tokens=1000,
                completion_tokens=2500,
                reasoning_tokens=2000,
            )
        # non-reasoning = 2500 - 2000 = 500 tokens at $2/M = 0.001
        # reasoning = 2000 tokens at $8/M = 0.016
        assert completion == pytest.approx(0.017, rel=1e-6)

    def test_reasoning_tokens_ignored_when_rate_unset(self):
        """If no reasoning rate is set, reasoning_tokens fall through to
        the standard output rate (and shouldn't double-bill — completion
        tokens already include them)."""
        from unittest.mock import patch

        synthetic = ModelPricing(
            input_cost_per_million=1.0,
            output_cost_per_million=2.0,
            # output_cost_per_reasoning_token=None
        )
        with patch("arcllm.pricing.tables.get_model_pricing", return_value=synthetic):
            _prompt, completion = cost_per_token(
                "synthetic/model",
                prompt_tokens=1000,
                completion_tokens=500,
                reasoning_tokens=200,
            )
        # All 500 tokens at $2/M = 0.001 (reasoning_tokens not surcharged)
        assert completion == pytest.approx(0.001, rel=1e-6)

    def test_cache_creation_uses_manifest_value_not_hardcoded_factor(self):
        """When ``cache_creation_cost_per_million`` is set on the model,
        cost_per_token must use it instead of the historical 1.25x
        fallback. We pick a synthetic rate that's NOT 1.25x so the bug
        would be detectable."""
        from unittest.mock import patch

        synthetic = ModelPricing(
            input_cost_per_million=4.0,
            output_cost_per_million=10.0,
            cached_input_cost_per_million=0.4,
            cache_creation_cost_per_million=6.0,  # 1.5x, not 1.25x
        )
        with patch("arcllm.pricing.tables.get_model_pricing", return_value=synthetic):
            prompt, _ = cost_per_token(
                "synthetic/model",
                prompt_tokens=10000,
                completion_tokens=0,
                cache_creation_input_tokens=5000,
            )
        # base = 5000 at $4/M = 0.020
        # creation = 5000 at $6/M = 0.030
        # total = 0.050
        assert prompt == pytest.approx(0.050, rel=1e-6)

    def test_cache_creation_falls_back_to_1_25x_when_unset(self):
        """Legacy entries without cache_creation_cost_per_million keep the
        historical Anthropic 1.25x surcharge so old data stays correct."""
        from unittest.mock import patch

        synthetic = ModelPricing(
            input_cost_per_million=4.0,
            output_cost_per_million=10.0,
            cached_input_cost_per_million=0.4,
            # cache_creation_cost_per_million=None
        )
        with patch("arcllm.pricing.tables.get_model_pricing", return_value=synthetic):
            prompt, _ = cost_per_token(
                "synthetic/model",
                prompt_tokens=10000,
                completion_tokens=0,
                cache_creation_input_tokens=5000,
            )
        # base = 5000 at $4/M = 0.020
        # creation = 5000 at $4*1.25/M = $5/M = 0.025
        # total = 0.045
        assert prompt == pytest.approx(0.045, rel=1e-6)


class TestCompletionCostExtractsReasoningTokens:
    def test_completion_cost_passes_reasoning_tokens_through(self):
        """completion_cost should read reasoning_tokens out of
        usage.completion_tokens_details and pass them into cost_per_token
        so the reasoning-rate path is exercised end-to-end."""
        from unittest.mock import patch

        synthetic = ModelPricing(
            input_cost_per_million=1.0,
            output_cost_per_million=2.0,
            output_cost_per_reasoning_token=8.0,
        )
        response = ModelResponse(
            id="x",
            object="chat.completion",
            created=0,
            model="synthetic/model",
            choices=[Choice(index=0, message=Message(role="assistant", content="hi"), finish_reason="stop")],
            usage=Usage(
                prompt_tokens=1000,
                completion_tokens=2500,
                total_tokens=3500,
                completion_tokens_details={"reasoning_tokens": 2000},
            ),
        )
        with patch("arcllm.pricing.tables.get_model_pricing", return_value=synthetic):
            total = completion_cost(response)
        # prompt = 1000 at $1/M = 0.001
        # non-reasoning completion = 500 at $2/M = 0.001
        # reasoning = 2000 at $8/M = 0.016
        # total = 0.018
        assert total == pytest.approx(0.018, rel=1e-6)


class TestProviderReportedCostWins:
    def test_completion_cost_prefers_provider_reported(self):
        """When response.provider_reported_cost is set (e.g. via the
        OpenRouter adapter from the x-openrouter-cost header), completion_cost
        returns it directly without a table lookup. This sidesteps the
        meta-router enumeration problem entirely."""
        resp = ModelResponse(
            id="x",
            object="chat.completion",
            created=0,
            model="some-unknown-meta-router-model",
            choices=[Choice(index=0, message=Message(role="assistant", content="hi"), finish_reason="stop")],
            usage=Usage(prompt_tokens=100, completion_tokens=50, total_tokens=150),
            provider_reported_cost=0.00042,
        )
        assert completion_cost(resp) == pytest.approx(0.00042)

    def test_completion_cost_falls_back_to_table_when_unset(self):
        """Without provider_reported_cost, completion_cost computes from
        the static table as before."""
        resp = ModelResponse(
            id="x",
            object="chat.completion",
            created=0,
            model="openai/gpt-4o-mini",
            choices=[Choice(index=0, message=Message(role="assistant", content="hi"), finish_reason="stop")],
            usage=Usage(prompt_tokens=1_000_000, completion_tokens=0, total_tokens=1_000_000),
        )
        # gpt-4o-mini input is non-zero — confirm we get a positive table-derived value.
        assert completion_cost(resp) > 0


class TestModalityHelpers:
    """Per-modality cost helpers built on top of get_model_pricing."""

    def test_image_cost_per_request(self):
        from arcllm.pricing import image_cost

        # DALL-E 3 standard 1024x1024 = $0.04/image (per the manifest).
        assert image_cost("openai/dall-e-3", n=1) == pytest.approx(0.04)
        assert image_cost("openai/dall-e-3", n=3) == pytest.approx(0.12)

    def test_image_cost_defaults_to_one_image(self):
        from arcllm.pricing import image_cost

        assert image_cost("openai/dall-e-3") == pytest.approx(0.04)

    def test_image_cost_unknown_model_raises(self):
        from arcllm.pricing import image_cost
        from arcllm.pricing.tables import UnknownModelPricingError

        with pytest.raises(UnknownModelPricingError):
            image_cost("openai/gpt-4o-mini")  # chat model, no image pricing

    def test_audio_speech_cost_per_character(self):
        from arcllm.pricing import audio_cost

        # TTS-1 = $15/1M chars → $0.015 per 1k chars.
        assert audio_cost("openai/tts-1", characters=1000) == pytest.approx(0.015)
        assert audio_cost("openai/tts-1-hd", characters=1000) == pytest.approx(0.030)

    def test_audio_transcription_cost_per_second(self):
        from arcllm.pricing import audio_cost

        # Whisper-1 = $0.006/min = $0.0001/sec → 600 sec = $0.06.
        assert audio_cost("openai/whisper-1", seconds=600) == pytest.approx(0.06)

    def test_audio_cost_requires_unit_kwarg(self):
        from arcllm.pricing import audio_cost
        from arcllm.pricing.tables import UnknownModelPricingError

        with pytest.raises(UnknownModelPricingError):
            # No characters/seconds passed — can't determine billing.
            audio_cost("openai/tts-1")

    def test_audio_cost_wrong_unit_for_modality_raises(self):
        from arcllm.pricing import audio_cost
        from arcllm.pricing.tables import UnknownModelPricingError

        with pytest.raises(UnknownModelPricingError):
            # TTS is per-character, not per-second.
            audio_cost("openai/tts-1", seconds=10)
        with pytest.raises(UnknownModelPricingError):
            # Whisper is per-second, not per-character.
            audio_cost("openai/whisper-1", characters=100)

    def test_rerank_cost_per_query(self):
        from arcllm.pricing import rerank_cost

        # rerank-v3.5 = $2 per 1000 queries → 5 queries = $0.01.
        assert rerank_cost("cohere/rerank-v3.5", queries=5) == pytest.approx(0.01)

    def test_rerank_cost_defaults_to_one_query(self):
        from arcllm.pricing import rerank_cost

        assert rerank_cost("cohere/rerank-v3.5") == pytest.approx(0.002)
