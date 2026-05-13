"""
Model capability helpers.

Public surface:

- ``get_max_tokens(model) -> int | None``
- ``supports_vision(model) -> bool``
- ``supports_pdf_input(model) -> bool``
- ``supports_tools(model) -> bool``
- ``supports_function_calling(model) -> bool`` — alias of ``supports_tools``,
  exposed for litellm-compat naming.
- ``supports_structured_output(model) -> bool``
- ``get_model_capabilities(model) -> ModelCapabilities``
- ``get_model_info(model) -> dict`` — full capability + pricing snapshot
  (litellm-compat shape).
- ``get_supported_openai_params(model) -> list[str]`` — the OpenAI-style
  request-kwarg names this model accepts. Driven by the per-model
  ``supports_temperature`` / ``supports_stop_sequences`` /
  ``supports_reasoning_effort`` flags so reasoning-only models advertise the
  right surface.
"""

from __future__ import annotations

from typing import Any

from arcllm.capabilities.tables import (
    CAPABILITIES_VERSION,
    ModelCapabilities,
    get_max_tokens,
    get_model_capabilities,
    supports_pdf_input,
    supports_structured_output,
    supports_tools,
    supports_vision,
)

__all__ = [
    "CAPABILITIES_VERSION",
    "ModelCapabilities",
    "get_max_tokens",
    "get_model_capabilities",
    "get_model_info",
    "get_supported_openai_params",
    "supports_function_calling",
    "supports_pdf_input",
    "supports_response_schema",
    "supports_structured_output",
    "supports_tools",
    "supports_vision",
]


def supports_function_calling(model: str) -> bool:
    """Alias of :func:`supports_tools` (litellm-compat naming).

    arcllm uses "tools" terminology internally — this helper exists so
    upstream code that previously called ``litellm.supports_function_calling``
    can be migrated without renaming.
    """
    return supports_tools(model)


def supports_response_schema(model: str) -> bool:
    """Alias of :func:`supports_structured_output` (litellm-compat naming).

    Upstream code that previously called ``litellm.utils.supports_response_schema``
    can call ``arcllm.supports_response_schema`` unchanged.
    """
    return supports_structured_output(model)


def get_model_info(model: str) -> dict[str, Any]:
    """Return a serialised snapshot of the model's capabilities + pricing.

    Drop-in for ``litellm.get_model_info``. Shape:

    .. code-block:: python

        {
            "max_tokens": 16384,
            "max_input_tokens": 128000,
            "max_output_tokens": 16384,
            "input_cost_per_token": 2.5e-06,
            "output_cost_per_token": 1e-05,
            "cache_read_input_token_cost": 1.25e-06,
            "supports_vision": True,
            "supports_pdf_input": False,
            "supports_function_calling": True,
            "supports_response_schema": True,
            "supports_reasoning": False,
            "kind": "chat",
            "dimensions": None,
        }

    For unknown models, returns the same shape with the default capabilities
    (every flag False, no pricing).
    """
    caps = get_model_capabilities(model)

    # Pricing lookup is best-effort — unknown models return default caps and
    # no pricing entry.
    input_cost_per_token: float | None = None
    output_cost_per_token: float | None = None
    cache_read_cost_per_token: float | None = None
    try:
        from arcllm.pricing.tables import get_model_pricing

        pricing = get_model_pricing(model)
        input_cost_per_token = pricing.input_cost_per_million / 1_000_000
        output_cost_per_token = pricing.output_cost_per_million / 1_000_000
        if pricing.cached_input_cost_per_million is not None:
            cache_read_cost_per_token = pricing.cached_input_cost_per_million / 1_000_000
    except Exception:
        pass

    return {
        "max_tokens": caps.max_tokens,
        "max_input_tokens": caps.context_window,
        "max_output_tokens": caps.max_tokens,
        "input_cost_per_token": input_cost_per_token,
        "output_cost_per_token": output_cost_per_token,
        "cache_read_input_token_cost": cache_read_cost_per_token,
        "supports_vision": caps.supports_vision,
        "supports_pdf_input": caps.supports_pdf_input,
        "supports_function_calling": caps.supports_tools,
        "supports_response_schema": caps.supports_structured_output,
        "supports_reasoning": caps.supports_reasoning_effort,
        "kind": caps.kind,
        "dimensions": caps.dimensions,
    }


# OpenAI request-kwarg names that arcllm forwards to providers. Driven from
# the unified ``COMMON_PARAMS`` set so we stay in sync with the actual
# pass-through surface.
_OPENAI_PARAM_NAMES = (
    "temperature",
    "top_p",
    "max_tokens",
    "max_completion_tokens",
    "stop",
    "seed",
    "presence_penalty",
    "frequency_penalty",
    "tools",
    "tool_choice",
    "response_format",
    "n",
    "logprobs",
    "top_logprobs",
    "user",
    "stream_options",
    "parallel_tool_calls",
)


def get_supported_openai_params(model: str) -> list[str]:
    """Return the OpenAI-style request kwargs this model accepts.

    Drop-in for ``litellm.get_supported_openai_params``. Filters
    ``_OPENAI_PARAM_NAMES`` against the per-model capability flags:

    - drops ``temperature`` / ``top_p`` for reasoning models
      (``supports_temperature=False``).
    - drops ``stop`` for models without stop-sequence support.
    - adds ``reasoning_effort`` for models that accept it.
    - drops ``response_format`` for models without structured-output
      support (Anthropic + Bedrock-Anthropic, etc.).
    - drops ``tools`` / ``tool_choice`` / ``parallel_tool_calls`` for models
      without tool support.

    Unknown models get the full default set (we trust the user).
    """
    caps = get_model_capabilities(model)
    out: list[str] = []
    for name in _OPENAI_PARAM_NAMES:
        if name in {"temperature", "top_p"} and not caps.supports_temperature:
            continue
        if name == "stop" and not caps.supports_stop_sequences:
            continue
        if name in {"tools", "tool_choice", "parallel_tool_calls"} and not caps.supports_tools:
            continue
        if name == "response_format" and not caps.supports_structured_output:
            continue
        out.append(name)
    if caps.supports_reasoning_effort:
        out.append("reasoning_effort")
    return out
