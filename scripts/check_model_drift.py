"""
Detect drift between provider /v1/models endpoints and `arcllm/pricing/tables.py`.

Run weekly from `.github/workflows/model-drift.yml`. Skips silently for any
provider whose API key isn't configured. Emits a Markdown report on stdout.

Lines beginning with `- DRIFT` mark actionable drift that should open an issue.
Lines beginning with `- INFO` are informational (provider has no /v1/models API,
key missing, etc.) and do not trigger an issue.
"""

from __future__ import annotations

import os
import sys
import urllib.error
import urllib.request

import orjson

from arcllm.pricing.tables import ALL_PRICING


def _get(url: str, headers: dict[str, str], timeout: float = 15.0) -> dict | None:
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return orjson.loads(resp.read())
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as exc:
        print(f"- INFO: {url} unreachable ({type(exc).__name__})")
        return None
    except Exception as exc:
        print(f"- INFO: {url} failed ({type(exc).__name__}: {exc})")
        return None


def _diff(provider: str, upstream_ids: set[str], local_ids: set[str]) -> None:
    """Print Markdown bullets for any drift between upstream and local model lists."""
    new_upstream = upstream_ids - local_ids
    missing_locally = upstream_ids - local_ids  # alias for clarity
    dropped_upstream = local_ids - upstream_ids
    if new_upstream:
        print(
            f"- DRIFT: **{provider}** has {len(missing_locally)} model(s) upstream we don't track:"
        )
        for m in sorted(missing_locally)[:20]:
            print(f"    - `{m}`")
        if len(missing_locally) > 20:
            print(f"    - …and {len(missing_locally) - 20} more")
    if dropped_upstream:
        print(
            f"- DRIFT: **{provider}** has {len(dropped_upstream)} model(s) in our table "
            "missing from upstream `/v1/models`:"
        )
        for m in sorted(dropped_upstream)[:20]:
            print(f"    - `{m}`")
        if len(dropped_upstream) > 20:
            print(f"    - …and {len(dropped_upstream) - 20} more")
    if not new_upstream and not dropped_upstream:
        print(f"- OK: **{provider}** in sync ({len(upstream_ids)} models upstream).")


def check_openai() -> None:
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        print("- INFO: OPENAI_API_KEY not set; skipping openai")
        return
    data = _get("https://api.openai.com/v1/models", {"Authorization": f"Bearer {key}"})
    if not data:
        return
    ids = {m["id"] for m in data.get("data", [])}
    local = set(ALL_PRICING.get("openai", {}).keys())
    _diff("openai", ids, local)


def check_anthropic() -> None:
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        print("- INFO: ANTHROPIC_API_KEY not set; skipping anthropic")
        return
    data = _get(
        "https://api.anthropic.com/v1/models",
        {"x-api-key": key, "anthropic-version": "2023-06-01"},
    )
    if not data:
        return
    ids = {m["id"] for m in data.get("data", [])}
    local = set(ALL_PRICING.get("anthropic", {}).keys())
    _diff("anthropic", ids, local)


def check_gemini() -> None:
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        print("- INFO: GEMINI_API_KEY not set; skipping gemini")
        return
    data = _get(f"https://generativelanguage.googleapis.com/v1beta/models?key={key}", {})
    if not data:
        return
    # IDs come back as "models/gemini-2.5-pro" — strip the namespace.
    ids: set[str] = set()
    for m in data.get("models", []):
        name = m.get("name", "")
        name = name.removeprefix("models/")
        # Only chat-capable models — skip TTS / image-gen variants.
        methods = m.get("supportedGenerationMethods", []) or []
        if "generateContent" in methods or "embedContent" in methods:
            ids.add(name)
    local = set(ALL_PRICING.get("gemini", {}).keys())
    _diff("gemini", ids, local)


def check_mistral() -> None:
    key = os.environ.get("MISTRAL_API_KEY")
    if not key:
        print("- INFO: MISTRAL_API_KEY not set; skipping mistral")
        return
    data = _get("https://api.mistral.ai/v1/models", {"Authorization": f"Bearer {key}"})
    if not data:
        return
    ids = {m["id"] for m in data.get("data", []) if not m.get("deprecation")}
    local = set(ALL_PRICING.get("mistral", {}).keys())
    _diff("mistral", ids, local)


def check_together() -> None:
    key = os.environ.get("TOGETHER_API_KEY")
    if not key:
        print("- INFO: TOGETHER_API_KEY not set; skipping together_ai")
        return
    data = _get("https://api.together.xyz/v1/models", {"Authorization": f"Bearer {key}"})
    if not data:
        return
    # Together's response is a flat list, not {"data": [...]}.
    payload = data if isinstance(data, list) else data.get("data", [])
    ids = {
        m["id"] for m in payload if isinstance(m, dict) and m.get("type") in {"chat", "embedding"}
    }
    local = set(ALL_PRICING.get("together_ai", {}).keys())
    _diff("together_ai", ids, local)


def check_fireworks() -> None:
    key = os.environ.get("FIREWORKS_API_KEY")
    if not key:
        print("- INFO: FIREWORKS_API_KEY not set; skipping fireworks_ai")
        return
    data = _get(
        "https://api.fireworks.ai/inference/v1/models",
        {"Authorization": f"Bearer {key}"},
    )
    if not data:
        return
    ids = {m["id"] for m in data.get("data", [])}
    local = set(ALL_PRICING.get("fireworks_ai", {}).keys())
    _diff("fireworks_ai", ids, local)


def check_perplexity() -> None:
    # Perplexity does NOT expose /v1/models (returns 404). Skip programmatically.
    print("- INFO: perplexity has no /v1/models endpoint; track manually via docs.perplexity.ai")


def main() -> int:
    print(f"# Model drift report — {os.environ.get('GITHUB_RUN_ID', 'local')}")
    print()
    print("Compares each provider's `/v1/models` endpoint to the contents of")
    print("`arcllm/pricing/tables.py`. Run by `.github/workflows/model-drift.yml`.")
    print()
    for fn in (
        check_openai,
        check_anthropic,
        check_gemini,
        check_mistral,
        check_together,
        check_fireworks,
        check_perplexity,
    ):
        fn()
    return 0


if __name__ == "__main__":
    sys.exit(main())
