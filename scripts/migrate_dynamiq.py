"""
Mechanical migration of dynamiq from litellm to arcllm.

Operates on a checked-out dynamiq tree (default: ``vendor/dynamiq-ref/``).
Performs only the deterministic textual rewrites — symbol renames, import
swaps, exception path rewrites. Anything semantic (provider router,
fallback-trigger detection, test-mock patch targets) the script flags but
does not auto-fix.

Usage:
    python scripts/migrate_dynamiq.py [--root <dir>] [--check]

``--check`` exits non-zero if any file would change, useful for CI.

The full PR plan (which files, why) lives in
``/Users/vitalii.duk/.claude/plans/purrfect-sparking-church.md`` — phase B.

The arcllm public surface is a deliberate superset of the litellm symbols
that dynamiq imports, so most replacements are straight string swaps. The
notable exception is exception classes: arcllm uses
``arcllm.exceptions.{...}`` rather than the (unused-as-classpath)
``litellm.exceptions``.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DYNAMIQ_ROOT = REPO_ROOT / "vendor" / "dynamiq-ref"


# Each entry is a (regex, replacement) pair. Order matters — earlier rules
# rewrite more-specific phrases first so later, broader rules don't
# clobber them.
REPLACEMENTS: list[tuple[str, str]] = [
    # --- Exception path rewrites (litellm.exceptions → arcllm.exceptions) ---
    # ``Timeout`` and ``APIConnectionError`` are exported from
    # ``arcllm.exceptions`` as compat aliases of ``TimeoutError`` /
    # ``ConnectionError``, so we don't rename the symbols themselves —
    # only the import path.
    (r"\bfrom\s+litellm\.exceptions\s+import\s*\(", "from arcllm.exceptions import ("),
    (r"\bfrom\s+litellm\.exceptions\s+import\s+", "from arcllm.exceptions import "),
    (r"litellm\.exceptions\.", "arcllm.exceptions."),
    # --- litellm.utils.supports_pdf_input → arcllm.capabilities.supports_pdf_input ---
    (
        r"\bfrom\s+litellm\.utils\s+import\s+supports_pdf_input\b",
        "from arcllm.capabilities import supports_pdf_input",
    ),
    # ``litellm.utils.Delta`` and other type-ish symbols live on arcllm.types.
    # The Delta name is preserved via the ``Delta = ChunkDelta`` alias in
    # arcllm/types.py.
    (r"\bfrom\s+litellm\.utils\s+import\s+", "from arcllm.types import "),
    # --- litellm.types.utils → arcllm.types ---
    (r"\bfrom\s+litellm\.types\.utils\s+import\s+", "from arcllm.types import "),
    (r"\blitellm\.types\.utils\.", "arcllm.types."),
    # --- litellm.types or other submodules: re-route to arcllm.types ---
    (r"\bfrom\s+litellm\.types\s+import\s+", "from arcllm.types import "),
    (r"\blitellm\.types\.", "arcllm.types."),
    # --- Top-level imports: from litellm import X → from arcllm import X ---
    # Multi-line `from litellm import (`
    (r"\bfrom\s+litellm\s+import\s*\(", "from arcllm import ("),
    # Single-line `from litellm import X, Y, Z`
    (r"\bfrom\s+litellm\s+import\s+", "from arcllm import "),
    # Bare `import litellm` → `import arcllm as litellm`. Avoids cascading
    # symbol-by-symbol rewrites in files that just keep a module reference.
    # We use ``as litellm`` so call sites like ``litellm.completion`` keep
    # working — arcllm's surface is a superset of those calls.
    (r"^\s*import\s+litellm\s*$", "import arcllm as litellm"),
    # --- litellm.X token-counter style references (rarely seen but cheap to handle)
    (r"\blitellm\.completion\b", "arcllm.completion"),
    (r"\blitellm\.acompletion\b", "arcllm.acompletion"),
    (r"\blitellm\.embedding\b", "arcllm.embedding"),
    (r"\blitellm\.aembedding\b", "arcllm.aembedding"),
    (r"\blitellm\.token_counter\b", "arcllm.token_counter"),
    (r"\blitellm\.cost_per_token\b", "arcllm.cost_per_token"),
    (r"\blitellm\.completion_cost\b", "arcllm.completion_cost"),
    (r"\blitellm\.image_generation\b", "arcllm.image_generation"),
    (r"\blitellm\.image_variation\b", "arcllm.image_variation"),
    (r"\blitellm\.image_edit\b", "arcllm.image_edit"),
    (r"\blitellm\.rerank\b", "arcllm.rerank"),
    (r"\blitellm\.stream_chunk_builder\b", "arcllm.stream_chunk_builder"),
    (r"\blitellm\.get_model_info\b", "arcllm.get_model_info"),
    (r"\blitellm\.get_max_tokens\b", "arcllm.get_max_tokens"),
    (r"\blitellm\.supports_vision\b", "arcllm.supports_vision"),
    (r"\blitellm\.supports_pdf_input\b", "arcllm.supports_pdf_input"),
    (r"\blitellm\.supports_function_calling\b", "arcllm.supports_function_calling"),
    (r"\blitellm\.get_supported_openai_params\b", "arcllm.get_supported_openai_params"),
    # --- Test mocks: patch("litellm.X") → patch("arcllm.X") ---
    (r'patch\("litellm\.', 'patch("arcllm.'),
    (r"patch\('litellm\.", "patch('arcllm."),
    (r"patch\.object\(litellm,", "patch.object(arcllm,"),
    # --- Logger suppression: drop the LiteLLM logger lines completely ---
    # We replace the two-line block with empty so logger.py keeps its shape;
    # callers can re-introduce ARCLLM logger suppression if needed.
    (
        r'litellm_logger\s*=\s*logging\.getLogger\("LiteLLM"\)\n'
        r"litellm_logger\.setLevel\(logging\.ERROR\)\n",
        "",
    ),
]


# Files within the dynamiq tree that we touch. Glob applied under
# ``<root>/dynamiq/`` and ``<root>/tests/``.
GLOB_PATTERNS = ["**/*.py"]


# Files we deliberately do NOT touch — generated, vendored, or out of scope.
SKIP_DIRS = {
    "__pycache__",
    ".git",
    ".venv",
    "venv",
    "node_modules",
    "dist",
    "build",
    "vendor",
}


def _should_skip(path: Path, root: Path) -> bool:
    rel = path.relative_to(root)
    return any(part in SKIP_DIRS for part in rel.parts)


def _rewrite(text: str) -> tuple[str, int]:
    """Apply every rule in order. Return (new_text, total_replacements)."""
    total = 0
    for pattern, repl in REPLACEMENTS:
        new_text, n = re.subn(pattern, repl, text, flags=re.MULTILINE)
        total += n
        text = new_text
    return text, total


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_DYNAMIQ_ROOT,
        help="Path to the dynamiq checkout (default: vendor/dynamiq-ref/)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit non-zero if any file would change; do not write",
    )
    args = parser.parse_args()

    root: Path = args.root.resolve()
    if not root.exists():
        print(f"ERR: dynamiq root not found at {root}", file=sys.stderr)
        return 2

    targets: list[Path] = []
    for sub in ("dynamiq", "tests"):
        base = root / sub
        if not base.exists():
            continue
        for pattern in GLOB_PATTERNS:
            for p in base.glob(pattern):
                if p.is_file() and not _should_skip(p, root):
                    targets.append(p)

    changed: list[Path] = []
    total_replacements = 0
    for path in sorted(targets):
        original = path.read_text(encoding="utf-8")
        new_text, n = _rewrite(original)
        if n == 0 or new_text == original:
            continue
        total_replacements += n
        changed.append(path)
        if not args.check:
            path.write_text(new_text, encoding="utf-8")

    label = "would change" if args.check else "rewrote"
    for p in changed:
        print(f"  {label}: {p.relative_to(root)}")
    print(
        f"\n{label.capitalize()} {len(changed)} file(s) "
        f"with {total_replacements} replacement(s) under {root}."
    )

    # Manual-review reminders. These are not auto-fixed because they involve
    # behaviour, not just symbols.
    print("\n=== Manual review checklist ===")
    print(
        "  - dynamiq/pyproject.toml: replace `litellm = ...` with "
        "`arcllm = '>=0.4.0,<0.5'` (and remove litellm)."
    )
    print(
        "  - dynamiq/nodes/llms/replicate.py + tests: arcllm 0.4 does NOT yet ship "
        "a Replicate adapter (deferred to 0.4.1). Decide whether to keep the "
        "Replicate node on litellm temporarily, or drop the node from this PR."
    )
    print(
        "  - Fallback-trigger detection in nodes/llms/base.py: arcllm raises "
        "different exception subclasses than litellm. Verify "
        "`_should_trigger_fallback` still works — error-message string matching "
        "should remain intact, but check the type-tuple cases."
    )
    print(
        "  - utils/logger.py: the LiteLLM logger suppression was removed; "
        "arcllm has no equivalent noisy logger so no replacement is required."
    )

    if args.check and changed:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
