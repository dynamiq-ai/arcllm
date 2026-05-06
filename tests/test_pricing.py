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
