"""
Integration tests for the DeepSeek provider.

Skipped automatically when ``DEEPSEEK_API_KEY`` is not in the environment.

API docs: https://api-docs.deepseek.com/
"""

from __future__ import annotations

from tests.integration.base import IntegrationTestBase


class TestDeepSeekIntegration(IntegrationTestBase):
    """End-to-end integration coverage for DeepSeek's V4-Flash chat surface.

    DeepSeek does not offer embeddings, so ``SUPPORTS_EMBEDDINGS`` stays False.
    """

    PROVIDER = "deepseek"
    ENV_VAR = "DEEPSEEK_API_KEY"
    PRIMARY_MODEL = "deepseek-v4-flash"

    SUPPORTS_TOOLS = True
    SUPPORTS_STRUCTURED_OUTPUT = True
    SUPPORTS_STREAMING = True
    SUPPORTS_EMBEDDINGS = False
