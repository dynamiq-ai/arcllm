"""
Integration tests for Ollama (local server).

Unlike the other providers we don't gate on an API key — Ollama runs locally.
Instead we gate on a probe of the OLLAMA_HOST endpoint. CI runners that don't
have an Ollama server reachable should skip, not fail.

To run locally:
    ollama pull llama3.2
    pytest tests/integration/test_ollama_integration.py
"""

from __future__ import annotations

import os
import socket
from urllib.parse import urlparse

import pytest

from tests.integration.base import IntegrationTestBase


def _ollama_reachable() -> bool:
    """Quick TCP probe to ``OLLAMA_HOST`` (default localhost:11434)."""
    base = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    parsed = urlparse(base)
    host = parsed.hostname or "localhost"
    port = parsed.port or 11434
    try:
        with socket.create_connection((host, port), timeout=0.5):
            return True
    except OSError:
        return False


class TestOllamaIntegration(IntegrationTestBase):
    """End-to-end integration coverage against a local Ollama daemon."""

    PROVIDER = "ollama"
    # IntegrationTestBase.setup_class checks ENV_VAR; setting it to a benign
    # value that is always present (PATH) lets us bypass the env-var gate and
    # do our own reachability check below.
    ENV_VAR = "PATH"

    # llama3.2 is a small, widely-pulled default. Override via OLLAMA_TEST_MODEL.
    PRIMARY_MODEL = os.environ.get("OLLAMA_TEST_MODEL", "llama3.2")
    EMBEDDING_MODEL = "nomic-embed-text"

    SUPPORTS_TOOLS = True
    SUPPORTS_STRUCTURED_OUTPUT = True
    SUPPORTS_STREAMING = True
    SUPPORTS_EMBEDDINGS = True

    @classmethod
    def setup_class(cls) -> None:
        super().setup_class()
        if not _ollama_reachable():
            pytest.skip(
                "Ollama daemon not reachable at OLLAMA_HOST "
                f"({os.environ.get('OLLAMA_HOST', 'http://localhost:11434')})"
            )
