"""Parity guarantees between ``arcllm/pricing/tables.py`` and ``arcllm/capabilities/tables.py``.

Every model with pricing must have capabilities, and vice-versa. The two
tables are regenerated together by ``scripts/sync_tables.py``; these tests
catch any drift introduced by hand-edits.
"""

from __future__ import annotations

import pytest

from arcllm.capabilities.tables import (
    ALL_CAPABILITIES,
    CAPABILITIES_VERSION,
    ModelCapabilities,
)
from arcllm.pricing.tables import ALL_PRICING, PRICING_VERSION, ModelPricing


def test_pricing_and_capabilities_versions_match() -> None:
    """The two tables are generated from the same manifest set, so the
    version stamps must match."""
    assert PRICING_VERSION == CAPABILITIES_VERSION


def test_all_providers_present_in_both_tables() -> None:
    """Every provider with pricing also has a capability table, and vice-versa."""
    assert set(ALL_PRICING.keys()) == set(ALL_CAPABILITIES.keys())


@pytest.mark.parametrize("provider", sorted(ALL_PRICING.keys()))
def test_pricing_keys_are_subset_of_capability_keys(provider: str) -> None:
    """Every priced model has a capability entry."""
    pricing_ids = set(ALL_PRICING[provider].keys())
    caps_ids = set(ALL_CAPABILITIES[provider].keys())
    missing = pricing_ids - caps_ids
    assert not missing, f"{provider}: missing capability entries for {sorted(missing)[:5]}"


@pytest.mark.parametrize("provider", sorted(ALL_PRICING.keys()))
def test_capability_keys_are_subset_of_pricing_keys(provider: str) -> None:
    """Every model with capability info has pricing — so cost calculations
    don't silently fall back to ``UnknownModelPricingError`` for a model the
    rest of the SDK claims to support."""
    pricing_ids = set(ALL_PRICING[provider].keys())
    caps_ids = set(ALL_CAPABILITIES[provider].keys())
    missing = caps_ids - pricing_ids
    assert not missing, f"{provider}: missing pricing entries for {sorted(missing)[:5]}"


def test_every_provider_table_is_non_empty() -> None:
    """A provider with no models is almost certainly a regeneration bug.

    Exception: gateway-style providers (OpenRouter, HuggingFace Inference)
    whose catalogs span hundreds of upstream models with prices that vary
    per upstream — we don't enumerate them, so empty is fine.
    """
    # Gateways: arcllm doesn't enumerate their catalogs, so empty pricing
    # is intentional — cost_per_token raises UnknownModelPricingError, the
    # caller is expected to consult the gateway's own dashboard for cost.
    gateway_providers = {"openrouter", "huggingface"}
    for provider, table in ALL_PRICING.items():
        if provider in gateway_providers:
            continue
        assert len(table) > 0, f"{provider} pricing table is empty"
    for provider, table in ALL_CAPABILITIES.items():
        if provider in gateway_providers:
            continue
        assert len(table) > 0, f"{provider} capabilities table is empty"


def test_pricing_dataclass_shape() -> None:
    """Spot-check one entry per provider to confirm dataclass shape stays stable."""
    for provider, table in ALL_PRICING.items():
        if not table:
            continue  # gateway provider — see test_every_provider_table_is_non_empty
        sample = next(iter(table.values()))
        assert isinstance(sample, ModelPricing), f"{provider}: non-ModelPricing entry"
        assert sample.input_cost_per_million >= 0
        assert sample.output_cost_per_million >= 0


def test_capabilities_dataclass_shape() -> None:
    """Spot-check that every capability entry has the new ``kind`` field."""
    for provider, table in ALL_CAPABILITIES.items():
        for model_id, caps in table.items():
            assert isinstance(caps, ModelCapabilities), f"{provider}/{model_id}"
            assert caps.kind in {"chat", "reason", "embed"}, (
                f"{provider}/{model_id}: kind={caps.kind!r}"
            )


def test_embedding_models_have_dimensions() -> None:
    """Every model whose ``kind == 'embed'`` must declare a vector dimension."""
    for provider, table in ALL_CAPABILITIES.items():
        for model_id, caps in table.items():
            if caps.kind == "embed":
                assert caps.dimensions is not None, f"{provider}/{model_id}: missing dimensions"
                assert caps.dimensions > 0


def test_reasoning_models_drop_temperature() -> None:
    """Reasoning models (``kind == 'reason'``) must declare they don't support
    ``temperature`` so ``BaseAdapter._normalize_params`` knows to drop it.

    OpenAI o-series, GPT-5, etc. 400 if you pass ``temperature``. We catch this
    at the SDK layer.
    """
    for provider, table in ALL_CAPABILITIES.items():
        for model_id, caps in table.items():
            if caps.kind == "reason":
                assert not caps.supports_temperature, (
                    f"{provider}/{model_id}: reasoning model should not advertise "
                    "supports_temperature=True"
                )
                assert caps.supports_reasoning_effort, (
                    f"{provider}/{model_id}: reasoning model should advertise "
                    "supports_reasoning_effort=True"
                )


def test_embedding_models_reject_all_generation_params() -> None:
    """Embedding models can't take any chat-completion params."""
    for provider, table in ALL_CAPABILITIES.items():
        for model_id, caps in table.items():
            if caps.kind == "embed":
                assert not caps.supports_temperature, f"{provider}/{model_id}"
                assert not caps.supports_stop_sequences, f"{provider}/{model_id}"
                assert not caps.supports_reasoning_effort, f"{provider}/{model_id}"


def test_cost_per_token_supports_cache_read_tokens() -> None:
    """cost_per_token bills cache reads at the cached rate, not the base rate."""
    from arcllm.pricing.tables import cost_per_token, get_model_pricing

    pricing = get_model_pricing("claude-sonnet-4-5-20250929")
    assert pricing.cached_input_cost_per_million is not None, (
        "test fixture requires a model with prompt-caching pricing"
    )

    # Two calls: one without cache awareness, one with all input as cache read.
    base_prompt_cost, _ = cost_per_token(
        "claude-sonnet-4-5-20250929",
        prompt_tokens=1_000_000,
        completion_tokens=0,
    )
    cached_prompt_cost, _ = cost_per_token(
        "claude-sonnet-4-5-20250929",
        prompt_tokens=1_000_000,
        completion_tokens=0,
        cache_read_input_tokens=1_000_000,
    )
    # Cached rate is strictly cheaper.
    assert cached_prompt_cost < base_prompt_cost
    # And it equals the cached_input rate x tokens.
    expected = (1_000_000 / 1_000_000) * pricing.cached_input_cost_per_million
    assert abs(cached_prompt_cost - expected) < 1e-9


def test_cost_per_token_charges_cache_creation_premium() -> None:
    """Cache-write tokens are billed above the base input rate (~1.25x per Anthropic)."""
    from arcllm.pricing.tables import cost_per_token, get_model_pricing

    pricing = get_model_pricing("claude-sonnet-4-5-20250929")
    base_prompt_cost, _ = cost_per_token(
        "claude-sonnet-4-5-20250929",
        prompt_tokens=1_000_000,
    )
    creation_prompt_cost, _ = cost_per_token(
        "claude-sonnet-4-5-20250929",
        prompt_tokens=1_000_000,
        cache_creation_input_tokens=1_000_000,
    )
    # Creation premium is strictly higher than base.
    assert creation_prompt_cost > base_prompt_cost
    expected = (1_000_000 / 1_000_000) * pricing.input_cost_per_million * 1.25
    assert abs(creation_prompt_cost - expected) < 1e-9
