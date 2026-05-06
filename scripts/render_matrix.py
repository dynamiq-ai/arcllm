"""
Regenerate `docs/provider_model_matrix.md` from `arcllm/capabilities/tables.py`.

Run after `scripts/sync_tables.py`. The matrix is a single-page reference of
every model arcllm supports, grouped by provider, with capability flags +
context window + max output tokens + pricing.

Usage:
    python scripts/render_matrix.py            # write
    python scripts/render_matrix.py --check    # exit 1 if matrix is stale
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from arcllm.capabilities.tables import ALL_CAPABILITIES, CAPABILITIES_VERSION, ModelCapabilities
from arcllm.pricing.tables import ALL_PRICING, ModelPricing

REPO_ROOT = Path(__file__).resolve().parent.parent
MATRIX_PATH = REPO_ROOT / "docs" / "provider_model_matrix.md"


def _flag(value: bool) -> str:
    return "✅" if value else "—"


def _fmt_int(value: int | None) -> str:
    if value is None:
        return "—"
    if value >= 1000:
        return f"{value:,}"
    return str(value)


def _fmt_price(value: float | None) -> str:
    if value is None:
        return "—"
    if value == 0:
        return "free"
    if value < 0.01:
        return f"${value:.4f}"
    if value < 1:
        return f"${value:.3f}"
    return f"${value:.2f}"


def _row(model_id: str, caps: ModelCapabilities, pricing: ModelPricing | None) -> str:
    cells = [
        f"`{model_id}`",
        caps.kind,
        _fmt_int(caps.context_window),
        _fmt_int(caps.max_tokens),
        _flag(caps.supports_vision),
        _flag(caps.supports_pdf_input),
        _flag(caps.supports_tools),
        _flag(caps.supports_structured_output),
        _flag(caps.supports_reasoning_effort),
        _fmt_price(pricing.input_cost_per_million if pricing else None),
        _fmt_price(pricing.output_cost_per_million if pricing else None),
    ]
    return "| " + " | ".join(cells) + " |"


def render() -> str:
    parts: list[str] = []
    parts.append("# Provider model matrix\n")
    parts.append(
        f"_Auto-generated from `arcllm/capabilities/tables.py` "
        f"(version `{CAPABILITIES_VERSION}`)._\n"
    )
    parts.append("Run `python scripts/render_matrix.py` to refresh after updating the manifests.\n")
    parts.append(
        "\nColumns: `kind` is `chat` / `reason` / `embed`; ✅ marks a supported "
        "capability flag. Prices are USD per 1M tokens. `reasoning` indicates "
        "the model accepts the `reasoning_effort` parameter."
    )

    header = "| Model | Kind | Ctx | Out | Vision | PDF | Tools | Structured | Reasoning | Input/M | Output/M |"
    sep = "|" + "|".join(["---"] * 11) + "|"

    for provider in sorted(ALL_CAPABILITIES.keys()):
        caps_table = ALL_CAPABILITIES[provider]
        pricing_table = ALL_PRICING.get(provider, {})
        parts.append(f"\n## {provider} ({len(caps_table)} models)\n")
        parts.append(header)
        parts.append(sep)
        for model_id in sorted(caps_table.keys()):
            caps = caps_table[model_id]
            pricing = pricing_table.get(model_id)
            parts.append(_row(model_id, caps, pricing))

    return "\n".join(parts) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit non-zero if the matrix would change (for CI).",
    )
    args = parser.parse_args()

    new_text = render()
    old_text = MATRIX_PATH.read_text() if MATRIX_PATH.exists() else ""
    if new_text == old_text:
        print(f"[ok] {MATRIX_PATH.relative_to(REPO_ROOT)}: unchanged")
        return 0
    if args.check:
        print(f"[diff] {MATRIX_PATH.relative_to(REPO_ROOT)}: would change", file=sys.stderr)
        return 1
    MATRIX_PATH.write_text(new_text)
    print(f"[wrote] {MATRIX_PATH.relative_to(REPO_ROOT)}: {len(new_text)} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
