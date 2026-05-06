"""
Capability helpers — all pure-Python lookups, no API calls.

These are the functions you reach for when you need to know **before
calling** whether a model supports a feature: vision, tools, structured
output, PDF input, what its context window is, etc. arcllm ships a
generated capability table covering 300+ models — same shape as
``litellm.get_model_info`` for callers migrating from there.

Run::

    python examples/capability_helpers.py
"""

from __future__ import annotations

import arcllm


def boolean_predicates() -> None:
    """One-call yes/no predicates."""
    pairs = [
        ("gpt-4o", arcllm.supports_vision),
        ("anthropic/claude-haiku-4-5", arcllm.supports_pdf_input),
        ("gemini/gemini-2.5-pro", arcllm.supports_tools),
        ("gpt-4o", arcllm.supports_structured_output),
        ("openai/o4-mini", arcllm.supports_function_calling),
    ]
    for model, fn in pairs:
        print(f"  {fn.__name__:32s} {model:40s} -> {fn(model)}")


def numeric_lookups() -> None:
    """Token-budget + pricing lookups."""
    for model in ("gpt-4o", "anthropic/claude-opus-4-7", "gemini/gemini-2.5-flash"):
        print(f"\n{model}:")
        print(f"  max_tokens   = {arcllm.get_max_tokens(model)}")
        pricing = arcllm.get_model_pricing(model)
        print(f"  pricing.in   = ${pricing.input_cost_per_million}/M tokens")
        print(f"  pricing.out  = ${pricing.output_cost_per_million}/M tokens")
        if pricing.cached_input_cost_per_million is not None:
            print(f"  pricing.cache= ${pricing.cached_input_cost_per_million}/M tokens")


def full_model_info() -> None:
    """Litellm-compat ``get_model_info`` returns capability + pricing combined.

    The dict shape mirrors litellm's: per-token costs (``$/token``, not
    ``$/M tokens``) and the ``supports_function_calling`` /
    ``supports_response_schema`` litellm-style flag names. See
    :func:`arcllm.get_model_info` for the canonical schema.
    """
    info = arcllm.get_model_info("gpt-4o")
    print("\nget_model_info('gpt-4o'):")
    for k in (
        "max_input_tokens",
        "max_output_tokens",
        "supports_vision",
        "supports_function_calling",
        "supports_response_schema",
        "input_cost_per_token",
        "output_cost_per_token",
    ):
        if k in info:
            print(f"  {k}: {info[k]}")


def supported_params() -> None:
    """Which OpenAI request kwargs does this model accept?

    Reasoning models drop ``temperature``, ``top_p``, ``stop``; they take
    ``reasoning_effort`` and ``max_completion_tokens`` instead.
    """
    for model in ("gpt-4o", "openai/o4-mini", "anthropic/claude-haiku-4-5"):
        params = arcllm.get_supported_openai_params(model)
        print(f"\n{model}:")
        print(f"  supports {len(params)} params: {sorted(params)}")


if __name__ == "__main__":
    print("=== Boolean predicates ===")
    boolean_predicates()
    print("\n=== Numeric lookups ===")
    numeric_lookups()
    full_model_info()
    print("\n=== Supported params ===")
    supported_params()
