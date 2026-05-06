#!/usr/bin/env python3
"""
Live cross-provider latency benchmark.

Fires the same tiny prompt at every configured provider's flagship + cheapest
model. Measures:

    - TTFT (time-to-first-token) — proxy for streaming responsiveness.
    - Total latency end-to-end.
    - Tokens-per-second (output) when usage is reported.

Skips silently for providers whose API key isn't set. Outputs a stable JSON
document on stdout (no file artifacts) suitable for piping into ``jq`` or a
spreadsheet:

    python benchmarks/benchmark_provider_matrix.py | jq '.results[] | select(.error == null)'

Concurrency: ``--concurrent`` fan-outs each test under one ``asyncio.gather``
so the total wall-clock time roughly equals the slowest provider, not the sum.
Use ``--sequential`` for cleaner per-provider readings without contention.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from dataclasses import asdict, dataclass

import arcllm

# Two models per provider: a fast/cheap one and a flagship. Keep prompts tiny
# and ``max_tokens`` low so a full sweep runs for pennies. Pulled from the
# Phase 4 manifests, intentionally aligned with capabilities/pricing tables.
SCENARIOS: list[tuple[str, str, str, str]] = [
    # (provider, env_var, label, model_id)
    ("openai", "OPENAI_API_KEY", "cheap", "gpt-4o-mini"),
    ("openai", "OPENAI_API_KEY", "flagship", "gpt-5-mini"),
    ("anthropic", "ANTHROPIC_API_KEY", "cheap", "claude-haiku-4-5"),
    ("anthropic", "ANTHROPIC_API_KEY", "flagship", "claude-sonnet-4-6"),
    ("gemini", "GEMINI_API_KEY", "cheap", "gemini-2.5-flash-lite"),
    ("gemini", "GEMINI_API_KEY", "flagship", "gemini-2.5-pro"),
    ("mistral", "MISTRAL_API_KEY", "cheap", "mistral-small-latest"),
    ("mistral", "MISTRAL_API_KEY", "flagship", "mistral-large-latest"),
    (
        "fireworks_ai",
        "FIREWORKS_API_KEY",
        "cheap",
        "accounts/fireworks/models/llama-v3p3-70b-instruct",
    ),
    (
        "fireworks_ai",
        "FIREWORKS_API_KEY",
        "flagship",
        "accounts/fireworks/models/deepseek-v4-pro",
    ),
    ("perplexity", "PERPLEXITY_API_KEY", "cheap", "sonar"),
    ("perplexity", "PERPLEXITY_API_KEY", "flagship", "sonar-pro"),
    (
        "together_ai",
        "TOGETHER_API_KEY",
        "cheap",
        "meta-llama/Llama-3.3-70B-Instruct-Turbo",
    ),
    (
        "together_ai",
        "TOGETHER_API_KEY",
        "flagship",
        "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
    ),
    ("groq", "GROQ_API_KEY", "cheap", "llama-3.1-8b-instant"),
    ("groq", "GROQ_API_KEY", "flagship", "openai/gpt-oss-120b"),
    ("deepseek", "DEEPSEEK_API_KEY", "cheap", "deepseek-v4-flash"),
    ("cohere", "COHERE_API_KEY", "cheap", "command-r-08-2024"),
    ("cohere", "COHERE_API_KEY", "flagship", "command-a-03-2025"),
]


PROMPT = "Reply with exactly: ok"
MAX_TOKENS = 8


@dataclass(slots=True)
class Result:
    provider: str
    label: str
    model: str
    ttft_ms: float | None
    total_ms: float | None
    output_tokens: int | None
    tps_out: float | None
    error: str | None


async def measure(provider: str, label: str, model: str) -> Result:
    """Run a streaming completion and capture TTFT + total wall time."""
    full_model = f"{provider}/{model}"
    t_start = time.perf_counter()
    ttft: float | None = None
    output_tokens: int | None = None

    try:
        stream = await arcllm.acompletion(
            model=full_model,
            messages=[{"role": "user", "content": PROMPT}],
            max_tokens=MAX_TOKENS,
            temperature=0,
            stream=True,
        )
        async for chunk in stream:
            if (
                ttft is None
                and chunk.choices
                and chunk.choices[0].delta
                and chunk.choices[0].delta.content
            ):
                ttft = (time.perf_counter() - t_start) * 1000
            if chunk.usage and chunk.usage.completion_tokens:
                output_tokens = chunk.usage.completion_tokens
        total_ms = (time.perf_counter() - t_start) * 1000
        tps_out: float | None = None
        if output_tokens and total_ms:
            tps_out = output_tokens / (total_ms / 1000)
        return Result(provider, label, model, ttft, total_ms, output_tokens, tps_out, None)
    except Exception as exc:  # benchmark surfaces every failure mode
        return Result(
            provider,
            label,
            model,
            None,
            None,
            None,
            None,
            f"{type(exc).__name__}: {exc}",
        )


async def run_concurrent(scenarios: list[tuple[str, str, str, str]]) -> list[Result]:
    return await asyncio.gather(
        *(measure(prov, label, model) for prov, _env, label, model in scenarios)
    )


async def run_sequential(scenarios: list[tuple[str, str, str, str]]) -> list[Result]:
    out: list[Result] = []
    for prov, _env, label, model in scenarios:
        out.append(await measure(prov, label, model))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument(
        "--sequential",
        action="store_true",
        help="Run providers one at a time (cleaner per-provider readings).",
    )
    parser.add_argument(
        "--filter",
        type=str,
        default=None,
        help="Substring filter on provider name (e.g. 'gemini').",
    )
    args = parser.parse_args()

    scenarios = [s for s in SCENARIOS if os.environ.get(s[1])]
    if args.filter:
        scenarios = [s for s in scenarios if args.filter in s[0]]
    if not scenarios:
        print(json.dumps({"error": "no API keys configured for any provider"}))
        return 1

    runner = run_sequential if args.sequential else run_concurrent
    t0 = time.perf_counter()
    results = asyncio.run(runner(scenarios))
    wall_ms = (time.perf_counter() - t0) * 1000

    output = {
        "as_of": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "mode": "sequential" if args.sequential else "concurrent",
        "wall_ms": round(wall_ms, 1),
        "scenarios": len(scenarios),
        "results": [asdict(r) for r in results],
    }
    print(json.dumps(output, indent=2, default=str))
    failed = sum(1 for r in results if r.error)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
