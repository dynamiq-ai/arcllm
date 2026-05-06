"""
Live integration smoke for arcllm 0.4.0.

Hits each provider that has an API key set in the environment with a
single low-cost completion. Designed to fail loudly and fast — one call
per provider, max 8 output tokens, default model is the provider's
cheapest entry. Skips any provider whose env var isn't set.

Usage::

    python scripts/smoke_live.py                # run all providers
    python scripts/smoke_live.py --only xai groq  # subset
    python scripts/smoke_live.py --skip cohere    # exclude

Loads .env from the repo root via python-dotenv if available; otherwise
relies on env already being set. Exits non-zero if any attempted
provider fails.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv() -> None:
    env_path = REPO_ROOT / ".env"
    if not env_path.exists():
        return
    for raw in env_path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        # Don't override values already set in the shell env.
        os.environ.setdefault(k.strip(), v.strip())


# (provider_name, env_var, model, extra_kwargs) for the chat smoke.
CHAT_TARGETS: list[tuple[str, str, str, dict[str, Any]]] = [
    ("openai", "OPENAI_API_KEY", "openai/gpt-4o-mini", {}),
    ("anthropic", "ANTHROPIC_API_KEY", "anthropic/claude-haiku-4-5", {"max_tokens": 16}),
    ("gemini", "GEMINI_API_KEY", "gemini/gemini-2.5-flash-lite", {}),
    ("mistral", "MISTRAL_API_KEY", "mistral/mistral-small-latest", {}),
    ("together_ai", "TOGETHER_API_KEY", "together_ai/meta-llama/Llama-3.3-70B-Instruct-Turbo", {}),
    (
        "fireworks_ai",
        "FIREWORKS_API_KEY",
        "fireworks_ai/accounts/fireworks/models/llama-v3p3-70b-instruct",
        {},
    ),
    ("perplexity", "PERPLEXITY_API_KEY", "perplexity/sonar", {}),
    ("groq", "GROQ_API_KEY", "groq/llama-3.1-8b-instant", {}),
    ("xai", "XAI_API_KEY", "xai/grok-3-mini", {}),
    ("openrouter", "OPENROUTER_API_KEY", "openrouter/openai/gpt-4o-mini", {}),
    ("sambanova", "SAMBANOVA_API_KEY", "sambanova/Meta-Llama-3.3-70B-Instruct", {}),
    ("cohere", "COHERE_API_KEY", "cohere/command-r-08-2024", {}),
    ("cerebras", "CEREBRAS_API_KEY", "cerebras/llama3.1-8b", {}),
    ("deepinfra", "DEEPINFRA_API_KEY", "deepinfra/meta-llama/Llama-3.3-70B-Instruct", {}),
    ("deepseek", "DEEPSEEK_API_KEY", "deepseek/deepseek-chat", {}),
    (
        "nvidia_nim",
        "NVIDIA_NIM_API_KEY",
        "nvidia_nim/meta/llama-3.3-70b-instruct",
        {},
    ),
    (
        "watsonx",
        "WATSONX_API_KEY",
        "watsonx/meta-llama/llama-3-3-70b-instruct",
        {"max_tokens": 16},
    ),
    (
        "bedrock",
        "AWS_ACCESS_KEY_ID",
        "bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0",
        {"max_tokens": 16},
    ),
]


# Embedding smoke targets: (provider, env, model)
EMBED_TARGETS: list[tuple[str, str, str]] = [
    ("openai", "OPENAI_API_KEY", "openai/text-embedding-3-small"),
    ("cohere", "COHERE_API_KEY", "cohere/embed-english-v3.0"),
    ("mistral", "MISTRAL_API_KEY", "mistral/mistral-embed"),
]


def _has_keys(env_vars: list[str]) -> bool:
    return all(os.environ.get(v) for v in env_vars)


def _result(label: str, ok: bool, detail: str, elapsed: float) -> None:
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {label:50s} {elapsed:5.2f}s  {detail}")


def smoke_chat(provider: str, env_var: str, model: str, kwargs: dict[str, Any]) -> bool:
    if not os.environ.get(env_var):
        _result(f"chat:{provider}", True, "skipped (no key)", 0.0)
        return True
    import arcllm

    started = time.monotonic()
    try:
        max_tokens = kwargs.get("max_tokens", 8)
        resp = arcllm.completion(
            model=model,
            messages=[{"role": "user", "content": "Say 'hi' and nothing else."}],
            max_tokens=max_tokens,
            temperature=0.0,
        )
        text = resp.choices[0].message.content or "<empty>"
        elapsed = time.monotonic() - started
        _result(f"chat:{provider}", True, f"got {text[:30]!r}", elapsed)
        return True
    except Exception as exc:
        elapsed = time.monotonic() - started
        _result(f"chat:{provider}", False, f"{type(exc).__name__}: {exc}", elapsed)
        return False


def smoke_embed(provider: str, env_var: str, model: str) -> bool:
    if not os.environ.get(env_var):
        _result(f"embed:{provider}", True, "skipped (no key)", 0.0)
        return True
    import arcllm

    started = time.monotonic()
    try:
        resp = arcllm.embedding(model=model, input=["hello world"])
        n = len(resp.data[0].embedding)
        elapsed = time.monotonic() - started
        _result(f"embed:{provider}", True, f"got {n}-dim vector", elapsed)
        return True
    except Exception as exc:
        elapsed = time.monotonic() - started
        _result(f"embed:{provider}", False, f"{type(exc).__name__}: {exc}", elapsed)
        return False


def smoke_ollama() -> bool:
    """Hit a local Ollama (default ``http://localhost:11434``).

    Skips if the env var ``OLLAMA_API_BASE`` is unset *and* the default
    URL doesn't respond. Model defaults to ``qwen2.5:0.5b`` — smallest
    serviceable Ollama model — but can be overridden via
    ``OLLAMA_TEST_MODEL``.
    """
    import urllib.error
    import urllib.request

    base = os.environ.get("OLLAMA_API_BASE", "http://localhost:11434")
    try:
        with urllib.request.urlopen(f"{base}/api/version", timeout=2) as resp:
            resp.read()
    except (urllib.error.URLError, ConnectionError, TimeoutError, OSError):
        _result("chat:ollama", True, f"skipped (no daemon at {base})", 0.0)
        return True

    import arcllm

    model = os.environ.get("OLLAMA_TEST_MODEL", "qwen2.5:0.5b")
    started = time.monotonic()
    try:
        resp = arcllm.completion(
            model=f"ollama/{model}",
            messages=[{"role": "user", "content": "Reply with just the word 'hi'."}],
            max_tokens=8,
            temperature=0.0,
            api_base=base,
        )
        text = resp.choices[0].message.content or "<empty>"
        elapsed = time.monotonic() - started
        _result("chat:ollama", True, f"({model}) got {text[:30]!r}", elapsed)
        return True
    except Exception as exc:
        elapsed = time.monotonic() - started
        _result("chat:ollama", False, f"{type(exc).__name__}: {exc}", elapsed)
        return False


def smoke_ollama_embed() -> bool:
    import urllib.error
    import urllib.request

    base = os.environ.get("OLLAMA_API_BASE", "http://localhost:11434")
    try:
        with urllib.request.urlopen(f"{base}/api/version", timeout=2) as resp:
            resp.read()
    except (urllib.error.URLError, ConnectionError, TimeoutError, OSError):
        _result("embed:ollama", True, f"skipped (no daemon at {base})", 0.0)
        return True

    import arcllm

    model = os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text")
    started = time.monotonic()
    try:
        resp = arcllm.embedding(
            model=f"ollama/{model}",
            input=["hello world"],
            api_base=base,
        )
        n = len(resp.data[0].embedding)
        elapsed = time.monotonic() - started
        _result("embed:ollama", True, f"({model}) got {n}-dim vector", elapsed)
        return True
    except Exception as exc:
        elapsed = time.monotonic() - started
        _result("embed:ollama", False, f"{type(exc).__name__}: {exc}", elapsed)
        return False


def smoke_rerank() -> bool:
    if not os.environ.get("COHERE_API_KEY"):
        _result("rerank:cohere", True, "skipped (no key)", 0.0)
        return True
    import arcllm

    started = time.monotonic()
    docs = [
        "Linus Torvalds created the Linux kernel in 1991.",
        "Guido van Rossum created the Python programming language in 1991.",
        "Dennis Ritchie designed the C programming language at Bell Labs.",
    ]
    try:
        resp = arcllm.rerank(
            model="cohere/rerank-v3.5",
            query="Who invented the Python programming language?",
            documents=docs,
            top_n=3,
        )
        elapsed = time.monotonic() - started
        top = resp.results[0]
        order = [r.index for r in resp.results]
        # Guido is at index 1; rerank should put him first.
        ok = top.index == 1
        detail = f"order={order} top_score={top.relevance_score:.3f}"
        _result("rerank:cohere", ok, detail, elapsed)
        return ok
    except Exception as exc:
        elapsed = time.monotonic() - started
        _result("rerank:cohere", False, f"{type(exc).__name__}: {exc}", elapsed)
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="*", help="only run these provider names")
    parser.add_argument("--skip", nargs="*", default=[], help="exclude these provider names")
    parser.add_argument("--no-chat", action="store_true", help="skip chat smoke")
    parser.add_argument("--no-embed", action="store_true", help="skip embedding smoke")
    parser.add_argument("--no-rerank", action="store_true", help="skip rerank smoke")
    parser.add_argument("--no-ollama", action="store_true", help="skip local Ollama smoke")
    args = parser.parse_args()

    _load_dotenv()

    only = set(args.only or [])
    skip = set(args.skip)

    def selected(name: str) -> bool:
        if only and name not in only:
            return False
        return name not in skip

    failures: list[str] = []

    if not args.no_chat:
        print("== Chat completion smoke ==")
        for provider, env_var, model, kwargs in CHAT_TARGETS:
            if not selected(provider):
                continue
            if not smoke_chat(provider, env_var, model, kwargs):
                failures.append(f"chat:{provider}")

    if not args.no_embed:
        print("\n== Embedding smoke ==")
        for provider, env_var, model in EMBED_TARGETS:
            if not selected(provider):
                continue
            if not smoke_embed(provider, env_var, model):
                failures.append(f"embed:{provider}")

    if not args.no_rerank and selected("cohere"):
        print("\n== Rerank smoke ==")
        if not smoke_rerank():
            failures.append("rerank:cohere")

    if not args.no_ollama and selected("ollama"):
        print("\n== Ollama (local docker) smoke ==")
        if not smoke_ollama():
            failures.append("chat:ollama")
        if not args.no_embed and not smoke_ollama_embed():
            failures.append("embed:ollama")

    print("\n== Summary ==")
    if failures:
        print(f"{len(failures)} failure(s): {', '.join(failures)}")
        return 1
    print("All attempted smokes passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
