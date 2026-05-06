"""
Integration tests for the Cohere provider.

Skipped automatically when ``COHERE_API_KEY`` is not in the environment.
Set the secret in CI to enable. Tests inherit from :class:`IntegrationTestBase`.

API docs: https://docs.cohere.com/v2/docs/models
"""

from __future__ import annotations

from tests.integration.base import IntegrationTestBase


class TestCohereIntegration(IntegrationTestBase):
    """End-to-end integration coverage for Cohere's chat + embedding APIs."""

    PROVIDER = "cohere"
    ENV_VAR = "COHERE_API_KEY"
    PRIMARY_MODEL = "command-a-03-2025"
    EMBEDDING_MODEL = "embed-v4.0"

    SUPPORTS_TOOLS = True
    SUPPORTS_STRUCTURED_OUTPUT = True
    SUPPORTS_STREAMING = True
    SUPPORTS_EMBEDDINGS = True
