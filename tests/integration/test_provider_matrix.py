"""
Unified live-API smoke matrix across every provider with a configured API key.

Why this exists alongside the per-provider integration files:
    - The per-provider files (``test_openai_integration.py`` etc.) lock in
      detailed correctness — every code path of a single provider.
    - This matrix file fans out a small smoke battery (basic completion, async
      completion, streaming, embedding) across **every** provider in parallel
      and asserts they all respond. It surfaces regressions that affect a
      class of providers (e.g. a wire-format change in our request builder)
      with one short test run.

Each test is parametrized over the providers configured below. Tests skip
silently for providers whose API-key env var is missing — so a developer with
only one or two keys can still get useful local coverage.

The async tests fire all configured providers concurrently via
``asyncio.gather`` and assert each one returns within budget.
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass

import pytest

import arcllm

# ---------------------------------------------------------------------------
# Provider matrix
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ProviderSpec:
    """One row of the live matrix."""

    name: str  # arcllm provider key (used in `model="<name>/<model_id>"`)
    env_var: str  # API key env variable name
    chat_model: str  # smallest / cheapest current chat model with a key
    embedding_model: str | None  # None = provider has no embeddings


# Models chosen for: cheapest currently-served GA option per provider, so the
# matrix runs fast and on a small token budget. Adjust if the provider sunsets
# the model (capabilities/pricing tables are the source of truth for what's GA).
PROVIDERS: list[ProviderSpec] = [
    ProviderSpec("openai", "OPENAI_API_KEY", "gpt-4o-mini", "text-embedding-3-small"),
    ProviderSpec("anthropic", "ANTHROPIC_API_KEY", "claude-haiku-4-5", None),
    ProviderSpec("gemini", "GEMINI_API_KEY", "gemini-2.5-flash-lite", "text-embedding-004"),
    ProviderSpec("mistral", "MISTRAL_API_KEY", "mistral-small-latest", "mistral-embed"),
    ProviderSpec(
        "fireworks_ai",
        "FIREWORKS_API_KEY",
        "accounts/fireworks/models/llama-v3p3-70b-instruct",
        None,
    ),
    ProviderSpec("perplexity", "PERPLEXITY_API_KEY", "sonar", None),
    ProviderSpec(
        "together_ai",
        "TOGETHER_API_KEY",
        "meta-llama/Llama-3.3-70B-Instruct-Turbo",
        None,
    ),
    ProviderSpec("groq", "GROQ_API_KEY", "llama-3.1-8b-instant", None),
    ProviderSpec("deepseek", "DEEPSEEK_API_KEY", "deepseek-v4-flash", None),
    ProviderSpec("cohere", "COHERE_API_KEY", "command-r-08-2024", "embed-v4.0"),
]


def _configured_providers() -> list[ProviderSpec]:
    """Return only the providers whose API key is currently set."""
    return [p for p in PROVIDERS if os.environ.get(p.env_var)]


# ---------------------------------------------------------------------------
# Sequential per-provider tests (parametrized — each skips independently)
# ---------------------------------------------------------------------------


def _skip_if_no_key(spec: ProviderSpec) -> None:
    if not os.environ.get(spec.env_var):
        pytest.skip(f"{spec.env_var} not set; skipping {spec.name}")


@pytest.mark.parametrize("spec", PROVIDERS, ids=lambda s: s.name)
@pytest.mark.smoke
def test_completion_sync(spec: ProviderSpec) -> None:
    """Each provider returns a non-empty response to a tiny prompt."""
    _skip_if_no_key(spec)
    response = arcllm.completion(
        model=f"{spec.name}/{spec.chat_model}",
        messages=[{"role": "user", "content": "Reply with the single word ok."}],
        max_tokens=10,
        temperature=0,
    )
    assert response.choices, f"{spec.name}: no choices in response"
    content = response.choices[0].message.content or ""
    assert content.strip(), f"{spec.name}: empty content"


@pytest.mark.parametrize("spec", PROVIDERS, ids=lambda s: s.name)
@pytest.mark.smoke
def test_completion_async(spec: ProviderSpec) -> None:
    """Async client returns a non-empty response."""
    _skip_if_no_key(spec)

    async def run() -> None:
        response = await arcllm.acompletion(
            model=f"{spec.name}/{spec.chat_model}",
            messages=[{"role": "user", "content": "Reply with the single word ok."}],
            max_tokens=10,
            temperature=0,
        )
        assert response.choices
        assert (response.choices[0].message.content or "").strip()

    asyncio.run(run())


@pytest.mark.parametrize("spec", PROVIDERS, ids=lambda s: s.name)
def test_streaming(spec: ProviderSpec) -> None:
    """Streaming completion yields non-empty content deltas + a usable last chunk."""
    _skip_if_no_key(spec)
    stream = arcllm.completion(
        model=f"{spec.name}/{spec.chat_model}",
        messages=[{"role": "user", "content": "Count: one two three"}],
        max_tokens=20,
        stream=True,
    )
    chunks = list(stream)
    assert chunks, f"{spec.name}: streamed no chunks"
    parts = [
        c.choices[0].delta.content
        for c in chunks
        if c.choices and c.choices[0].delta and c.choices[0].delta.content
    ]
    assert parts, f"{spec.name}: streamed no content deltas"


@pytest.mark.parametrize(
    "spec",
    [p for p in PROVIDERS if p.embedding_model],
    ids=lambda s: s.name,
)
def test_embedding(spec: ProviderSpec) -> None:
    """Embedding model returns a non-empty float vector."""
    _skip_if_no_key(spec)
    assert spec.embedding_model
    response = arcllm.embedding(
        model=f"{spec.name}/{spec.embedding_model}",
        input=["hello world"],
    )
    assert response.data, f"{spec.name}: no embedding rows"
    vec = response.data[0].embedding
    assert vec, f"{spec.name}: empty embedding vector"
    assert all(isinstance(v, float) for v in vec), f"{spec.name}: non-float values"


# ---------------------------------------------------------------------------
# Parallel run — all configured providers fire at once
# ---------------------------------------------------------------------------


@pytest.mark.smoke
def test_all_providers_in_parallel() -> None:
    """Fan out one async completion per configured provider and gather them.

    This is the canary for "did we accidentally serialise async calls?" and
    "does our async HTTP client survive 10 concurrent in-flight requests?"
    Skipped if no API keys are configured at all.
    """
    configured = _configured_providers()
    if not configured:
        pytest.skip("No provider API keys configured — set at least one to run")

    async def call(spec: ProviderSpec) -> tuple[str, str | None, BaseException | None]:
        try:
            response = await arcllm.acompletion(
                model=f"{spec.name}/{spec.chat_model}",
                messages=[{"role": "user", "content": "Reply with the single word ok."}],
                max_tokens=8,
                temperature=0,
                timeout=30.0,
            )
            return spec.name, (response.choices[0].message.content or "").strip(), None
        except BaseException as exc:  # surface every failure mode in the report
            return spec.name, None, exc

    async def run_all() -> list[tuple[str, str | None, BaseException | None]]:
        return await asyncio.gather(*(call(p) for p in configured))

    results = asyncio.run(run_all())

    failures = [(name, exc) for name, _, exc in results if exc is not None]
    empties = [name for name, content, exc in results if exc is None and not content]

    assert not failures, "Provider failures: " + ", ".join(
        f"{n}: {type(e).__name__}({e})" for n, e in failures
    )
    assert not empties, f"Providers returned empty content: {empties}"
