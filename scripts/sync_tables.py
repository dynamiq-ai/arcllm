"""
Regenerate `arcllm/pricing/tables.py` and `arcllm/capabilities/tables.py` from the
provider model manifests in `tmp/model_manifests/*.json`.

Manifests are produced by Phase 3 of the public-release prep work (see
plan: arcllm public release prep). Each manifest is a JSON document of shape:

    {
      "provider": "<id>",
      "as_of": "YYYY-MM-DD",
      "models": [{ "id", "kind", "context_window", "max_output_tokens",
                   "supports_vision", "supports_pdf_input", "supports_tools",
                   "supports_structured_output",
                   "input_per_m", "output_per_m", "cached_input_per_m",
                   "deprecated", "source_disagreements" }, ...],
      "embedding_models": [{ "id", "kind"="embed", "context_window",
                             "dimensions", "input_per_m" }, ...]
    }

This script is idempotent: running it twice with the same manifest set produces
byte-identical output. Run with `--dry-run` to print the diff without writing.

Usage:
    python scripts/sync_tables.py            # write
    python scripts/sync_tables.py --dry-run  # don't write, exit 1 if files would change
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_DIR = REPO_ROOT / "tmp" / "model_manifests"
PRICING_PATH = REPO_ROOT / "arcllm" / "pricing" / "tables.py"
CAPS_PATH = REPO_ROOT / "arcllm" / "capabilities" / "tables.py"

TODAY = date.today().isoformat().replace("-", ".")

# Provider keys we generate dicts for. Order matters for stable output.
PROVIDERS = [
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
    # 0.4.0 — Tier A OpenAI-compat
    "xai",
    "openrouter",
    "nvidia_nim",
    "cerebras",
    "sambanova",
    "deepinfra",
    # 0.4.0 — Tier B
    "huggingface",
    "watsonx",
    "ai21",
]

# Module-level variable name suffix per provider. The provider key (used in
# ``ALL_PRICING`` / ``ALL_CAPABILITIES``) keeps the canonical ``together_ai`` /
# ``fireworks_ai`` slugs; the Python variable name drops the trailing ``_AI``
# to match historical names (``TOGETHER_PRICING``, ``FIREWORKS_PRICING``) that
# external callers and tests already import.
VAR_NAME = {
    "openai": "OPENAI",
    "azure": "AZURE",
    "anthropic": "ANTHROPIC",
    "gemini": "GEMINI",
    "vertex_ai": "VERTEX_AI",
    "bedrock": "BEDROCK",
    "mistral": "MISTRAL",
    "cohere": "COHERE",
    "groq": "GROQ",
    "together_ai": "TOGETHER",
    "fireworks_ai": "FIREWORKS",
    "deepseek": "DEEPSEEK",
    "perplexity": "PERPLEXITY",
    "databricks": "DATABRICKS",
    "ollama": "OLLAMA",
    # 0.4.0 — Tier A OpenAI-compat
    "xai": "XAI",
    "openrouter": "OPENROUTER",
    "nvidia_nim": "NVIDIA_NIM",
    "cerebras": "CEREBRAS",
    "sambanova": "SAMBANOVA",
    "deepinfra": "DEEPINFRA",
    # 0.4.0 — Tier B
    "huggingface": "HUGGINGFACE",
    "watsonx": "WATSONX",
    "ai21": "AI21",
}


def _load_manifests() -> dict[str, dict]:
    """Load every manifest into a {provider: manifest_dict} map."""
    out: dict[str, dict] = {}
    for prov in PROVIDERS:
        path = MANIFEST_DIR / f"{prov}.json"
        if not path.exists():
            print(f"WARN: missing manifest {path}", file=sys.stderr)
            continue
        out[prov] = json.loads(path.read_text())
    return out


def _fmt_price(v: float | None) -> str:
    if v is None:
        return "None"
    if v == int(v):
        return f"{int(v)}.0"
    # Strip trailing zeros, keep up to 6 decimal places
    return f"{v:.6f}".rstrip("0").rstrip(".")


def _pricing_line(model: dict) -> str:
    cached = model.get("cached_input_per_m")
    if cached is None:
        return (
            f"    {model['id']!r}: "
            f"ModelPricing({_fmt_price(model['input_per_m'])}, "
            f"{_fmt_price(model['output_per_m'])}),"
        )
    return (
        f"    {model['id']!r}: "
        f"ModelPricing({_fmt_price(model['input_per_m'])}, "
        f"{_fmt_price(model['output_per_m'])}, "
        f"{_fmt_price(cached)}),"
    )


def _embed_pricing_line(model: dict) -> str:
    return f"    {model['id']!r}: ModelPricing({_fmt_price(model['input_per_m'])}, 0.0),"


def _caps_line(model: dict) -> str:
    """Emit one ModelCapabilities row, deriving param-restriction flags from kind.

    Reasoning models (``kind="reason"``) reject ``temperature``/``top_p`` and
    take ``reasoning_effort`` instead. We default to those rules; manifest
    entries can opt out via explicit ``supports_temperature: true`` etc.
    """
    kind = model.get("kind", "chat")
    is_reason = kind == "reason"
    supports_temperature = bool(model.get("supports_temperature", not is_reason))
    supports_stop_sequences = bool(model.get("supports_stop_sequences", True))
    supports_reasoning_effort = bool(model.get("supports_reasoning_effort", is_reason))
    return (
        f"    {model['id']!r}: ModelCapabilities("
        f"{model['max_output_tokens']!r}, "
        f"{model['context_window']!r}, "
        f"{bool(model.get('supports_vision'))}, "
        f"{bool(model.get('supports_pdf_input'))}, "
        f"{bool(model.get('supports_tools'))}, "
        f"{bool(model.get('supports_structured_output'))}, "
        f"kind={kind!r}, "
        f"supports_temperature={supports_temperature}, "
        f"supports_stop_sequences={supports_stop_sequences}, "
        f"supports_reasoning_effort={supports_reasoning_effort}),"
    )


def _embed_caps_line(model: dict) -> str:
    """Embedding models reject all generation params."""
    dims = model.get("dimensions")
    return (
        f"    {model['id']!r}: ModelCapabilities("
        f"None, "
        f"{model['context_window']!r}, "
        f"False, False, False, False, "
        f"kind='embed', "
        f"dimensions={dims!r}, "
        f"supports_temperature=False, "
        f"supports_stop_sequences=False, "
        f"supports_reasoning_effort=False),"
    )


# =============================================================================
# Pricing table generation
# =============================================================================

PRICING_HEADER = '''"""
Pricing tables for supported LLM models.

Prices are in USD per 1 million tokens.
Generated from `tmp/model_manifests/` by `scripts/sync_tables.py`.
Last updated: {as_of}

To update prices:
    1. Refresh the manifests: see plan + AGENTS.md for the agent-driven workflow.
    2. Run `python scripts/sync_tables.py`.
    3. Bump `PRICING_VERSION` if the format / public API changes.
    4. Run `pytest tests/test_pricing.py tests/test_tables_parity.py`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from arcllm.exceptions import ArcLLMError

if TYPE_CHECKING:
    from arcllm.types import ModelResponse

__all__ = [
    "PRICING_VERSION",
    "ModelPricing",
    "completion_cost",
    "cost_per_token",
    "get_model_pricing",
]


PRICING_VERSION = "{version}"


@dataclass(slots=True, frozen=True)
class ModelPricing:
    """Pricing information for a model (USD per 1M tokens)."""

    input_cost_per_million: float
    output_cost_per_million: float
    # Optional: cached input price for prompt caching (None if not supported).
    cached_input_cost_per_million: float | None = None


'''

PRICING_FOOTER = '''
ALL_PRICING: dict[str, dict[str, ModelPricing]] = {{
{registry_body}
}}


class UnknownModelPricingError(ArcLLMError):
    """Raised when pricing is not available for a model."""


def _normalize_model_name(model: str) -> tuple[str | None, str]:
    """Split a model string into ``(provider, model_id)``.

    Accepts ``provider/model`` or bare ``model``. Provider keys are matched
    against ``ALL_PRICING`` after normalising kebab-case to snake_case so that
    e.g. ``vertex-ai/gemini-2.5-pro`` resolves to ``vertex_ai``.
    """
    provider: str | None = None
    model_name = model

    if "/" in model:
        head, tail = model.split("/", 1)
        head_norm = head.lower().replace("-", "_")
        if head_norm in ALL_PRICING:
            provider = head_norm
            model_name = tail
        # Otherwise the slash belongs to the model id (e.g. Together's
        # ``meta-llama/Llama-4-...``); fall through and search every provider.

    return provider, model_name


def get_model_pricing(model: str) -> ModelPricing:
    """Return pricing for ``model``.

    Raises ``UnknownModelPricingError`` if no entry is found.
    """
    provider, model_name = _normalize_model_name(model)

    if provider is not None:
        pricing_table = ALL_PRICING.get(provider, {{}})
        if model_name in pricing_table:
            return pricing_table[model_name]
        raise UnknownModelPricingError(
            f"No pricing available for model {{model_name!r}} from provider {{provider!r}}",
            model=model,
            provider=provider,
        )

    for pricing_table in ALL_PRICING.values():
        if model_name in pricing_table:
            return pricing_table[model_name]

    raise UnknownModelPricingError(
        f"No pricing available for model {{model!r}}",
        model=model,
    )


def cost_per_token(
    model: str,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    *,
    cache_read_input_tokens: int = 0,
    cache_creation_input_tokens: int = 0,
) -> tuple[float, float]:
    """Return ``(prompt_cost, completion_cost)`` in USD.

    Prompt-side cost is split into three slices when the provider supports
    prompt caching:

    - ``cache_read_input_tokens``: billed at
      ``cached_input_cost_per_million`` (typically 10% of base).
    - ``cache_creation_input_tokens``: billed at 1.25x the base input rate
      (Anthropic's documented cache-write surcharge). Falls back to base
      when the model has no cached pricing entry.
    - The remainder (``prompt_tokens - cache_read - cache_creation``):
      billed at ``input_cost_per_million``.

    Callers that don't track cache state simply omit the cache args; cost
    falls back to the simple ``prompt_tokens * input_rate`` calculation.
    """
    pricing = get_model_pricing(model)
    input_rate = pricing.input_cost_per_million
    cached_rate = (
        pricing.cached_input_cost_per_million
        if pricing.cached_input_cost_per_million is not None
        else input_rate
    )
    creation_rate = input_rate * 1.25  # Anthropic cache-write surcharge

    base_prompt = max(
        0, prompt_tokens - cache_read_input_tokens - cache_creation_input_tokens
    )
    prompt_cost = (
        (base_prompt / 1_000_000) * input_rate
        + (cache_read_input_tokens / 1_000_000) * cached_rate
        + (cache_creation_input_tokens / 1_000_000) * creation_rate
    )
    completion_cost = (completion_tokens / 1_000_000) * pricing.output_cost_per_million
    return (prompt_cost, completion_cost)


def completion_cost(
    response: ModelResponse,
    model: str | None = None,
) -> float:
    """Return the total USD cost for ``response``.

    ``model`` overrides ``response.model`` when given. Returns ``0.0`` when
    ``response.usage`` is ``None`` (provider didn't report usage). When the
    response carries cache token counts (Anthropic-family providers), they
    are factored into the prompt-side cost at the cached rate.
    """
    model_name = model or response.model
    if not model_name:
        raise ValueError("Model name required but not provided")

    usage = response.usage
    if usage is None:
        return 0.0

    prompt_cost, comp_cost = cost_per_token(
        model_name,
        prompt_tokens=usage.prompt_tokens,
        completion_tokens=usage.completion_tokens,
        cache_read_input_tokens=usage.cache_read_input_tokens or 0,
        cache_creation_input_tokens=usage.cache_creation_input_tokens or 0,
    )
    return prompt_cost + comp_cost
'''


def _emit_pricing_dict(provider: str, manifest: dict) -> str:
    var = VAR_NAME[provider] + "_PRICING"
    body_lines: list[str] = []
    for m in manifest.get("models", []):
        body_lines.append(_pricing_line(m))
    for m in manifest.get("embedding_models", []):
        body_lines.append(_embed_pricing_line(m))
    if not body_lines:
        body = "    # No models in manifest"
    else:
        body = "\n".join(body_lines)
    return f"{var}: dict[str, ModelPricing] = {{\n{body}\n}}\n"


def _emit_pricing_registry(manifests: dict[str, dict]) -> str:
    lines: list[str] = []
    for prov in PROVIDERS:
        if prov in manifests:
            lines.append(f'    "{prov}": {VAR_NAME[prov]}_PRICING,')
    return "\n".join(lines)


def render_pricing_module(manifests: dict[str, dict]) -> str:
    """Build the full text of `arcllm/pricing/tables.py`."""
    as_of = max((m.get("as_of", TODAY) for m in manifests.values()), default=TODAY)
    parts: list[str] = [PRICING_HEADER.format(as_of=as_of, version=as_of.replace("-", "."))]
    for prov in PROVIDERS:
        if prov in manifests:
            parts.append(f"# {'=' * 77}\n# {prov}\n# {'=' * 77}\n")
            parts.append(_emit_pricing_dict(prov, manifests[prov]))
            parts.append("\n")
    parts.append(PRICING_FOOTER.format(registry_body=_emit_pricing_registry(manifests)))
    return "".join(parts)


# =============================================================================
# Capabilities table generation
# =============================================================================

CAPS_HEADER = '''"""
Model capability tables for supported LLM models.

Generated from `tmp/model_manifests/` by `scripts/sync_tables.py`.
Last updated: {as_of}

Capabilities tracked per entry:
    - max_tokens: max output tokens (None for embedding models)
    - context_window: max input context length
    - supports_vision / supports_pdf_input / supports_tools / supports_structured_output
    - kind: "chat" | "reason" | "embed"
    - dimensions: embedding vector size (None for non-embedding models)

To update:
    1. Refresh manifests, then run `python scripts/sync_tables.py`.
    2. Bump CAPABILITIES_VERSION when changing the dataclass shape.
    3. Run `pytest tests/test_capabilities.py tests/test_tables_parity.py`.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "CAPABILITIES_VERSION",
    "DEFAULT_CAPABILITIES",
    "ModelCapabilities",
    "get_max_tokens",
    "get_model_capabilities",
    "supports_pdf_input",
    "supports_structured_output",
    "supports_tools",
    "supports_vision",
]


CAPABILITIES_VERSION = "{version}"


@dataclass(slots=True, frozen=True)
class ModelCapabilities:
    """Capability information for a model."""

    max_tokens: int | None
    context_window: int | None
    supports_vision: bool = False
    supports_pdf_input: bool = False
    supports_tools: bool = False
    supports_structured_output: bool = False
    # ``kind`` classifies the model so callers can filter (chat vs. reason vs.
    # embed) without string-matching the id.
    kind: str = "chat"
    dimensions: int | None = None
    # Param-restriction flags. Reasoning models (o-series, GPT-5, Claude with
    # thinking, Gemini 2.5+ with thinking) reject ``temperature``/``top_p`` and
    # take ``reasoning_effort`` instead. arcllm uses these flags in
    # ``BaseAdapter._normalize_params`` to drop unsupported params with a
    # warning rather than letting the provider 400.
    supports_temperature: bool = True
    supports_stop_sequences: bool = True
    supports_reasoning_effort: bool = False


'''

CAPS_FOOTER = '''
ALL_CAPABILITIES: dict[str, dict[str, ModelCapabilities]] = {{
{registry_body}
}}

DEFAULT_CAPABILITIES = ModelCapabilities(
    max_tokens=4096,
    context_window=8192,
    supports_vision=False,
    supports_pdf_input=False,
    supports_tools=False,
    supports_structured_output=False,
)


def _normalize_model_name(model: str) -> tuple[str | None, str]:
    """Split a model string into ``(provider, model_id)``.

    Mirrors the implementation in ``arcllm/pricing/tables.py``.
    """
    provider: str | None = None
    model_name = model

    if "/" in model:
        head, tail = model.split("/", 1)
        head_norm = head.lower().replace("-", "_")
        if head_norm in ALL_CAPABILITIES:
            provider = head_norm
            model_name = tail

    return provider, model_name


def get_model_capabilities(model: str) -> ModelCapabilities:
    """Return capability info for ``model``, or ``DEFAULT_CAPABILITIES`` if unknown."""
    provider, model_name = _normalize_model_name(model)

    if provider is not None:
        cap_table = ALL_CAPABILITIES.get(provider, {{}})
        return cap_table.get(model_name, DEFAULT_CAPABILITIES)

    for cap_table in ALL_CAPABILITIES.values():
        if model_name in cap_table:
            return cap_table[model_name]
    return DEFAULT_CAPABILITIES


def get_max_tokens(model: str) -> int | None:
    return get_model_capabilities(model).max_tokens


def supports_vision(model: str) -> bool:
    return get_model_capabilities(model).supports_vision


def supports_pdf_input(model: str) -> bool:
    return get_model_capabilities(model).supports_pdf_input


def supports_tools(model: str) -> bool:
    return get_model_capabilities(model).supports_tools


def supports_structured_output(model: str) -> bool:
    return get_model_capabilities(model).supports_structured_output
'''


def _emit_caps_dict(provider: str, manifest: dict) -> str:
    var = VAR_NAME[provider] + "_CAPABILITIES"
    body_lines: list[str] = []
    for m in manifest.get("models", []):
        body_lines.append(_caps_line(m))
    for m in manifest.get("embedding_models", []):
        body_lines.append(_embed_caps_line(m))
    if not body_lines:
        body = "    # No models in manifest"
    else:
        body = "\n".join(body_lines)
    return f"{var}: dict[str, ModelCapabilities] = {{\n{body}\n}}\n"


def _emit_caps_registry(manifests: dict[str, dict]) -> str:
    lines: list[str] = []
    for prov in PROVIDERS:
        if prov in manifests:
            lines.append(f'    "{prov}": {VAR_NAME[prov]}_CAPABILITIES,')
    return "\n".join(lines)


def render_caps_module(manifests: dict[str, dict]) -> str:
    as_of = max((m.get("as_of", TODAY) for m in manifests.values()), default=TODAY)
    parts: list[str] = [CAPS_HEADER.format(as_of=as_of, version=as_of.replace("-", "."))]
    for prov in PROVIDERS:
        if prov in manifests:
            parts.append(f"# {'=' * 77}\n# {prov}\n# {'=' * 77}\n")
            parts.append(_emit_caps_dict(prov, manifests[prov]))
            parts.append("\n")
    parts.append(CAPS_FOOTER.format(registry_body=_emit_caps_registry(manifests)))
    return "".join(parts)


# =============================================================================
# Driver
# =============================================================================


def main() -> int:
    parser = argparse.ArgumentParser(description="Regenerate arcllm pricing/capabilities tables")
    parser.add_argument("--dry-run", action="store_true", help="Print diff without writing")
    args = parser.parse_args()

    manifests = _load_manifests()
    if not manifests:
        print("ERR: no manifests found", file=sys.stderr)
        return 2

    pricing_text = render_pricing_module(manifests)
    caps_text = render_caps_module(manifests)

    changed = False
    for path, new_text in [(PRICING_PATH, pricing_text), (CAPS_PATH, caps_text)]:
        old_text = path.read_text() if path.exists() else ""
        if old_text == new_text:
            print(f"[ok] {path.relative_to(REPO_ROOT)}: unchanged")
            continue
        changed = True
        if args.dry_run:
            print(f"[diff] {path.relative_to(REPO_ROOT)}: would change ({len(new_text)} bytes)")
        else:
            path.write_text(new_text)
            print(f"[wrote] {path.relative_to(REPO_ROOT)}: {len(new_text)} bytes")

    if args.dry_run and changed:
        return 1

    # Format the generated files so they pass `ruff format --check` in CI.
    # Run on every successful invocation, even when content was unchanged,
    # so an unformatted file already on disk gets fixed without forcing the
    # caller to re-edit a manifest.
    if not args.dry_run:
        import subprocess

        subprocess.run(
            ["ruff", "format", str(PRICING_PATH), str(CAPS_PATH)],
            check=False,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
