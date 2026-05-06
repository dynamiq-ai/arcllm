"""Unit tests for the Databricks Foundation Model APIs adapter.

Databricks normalises every hosted model to OpenAI-compatible shape, so the
adapter is a thin wrapper over OpenAI request/response handling. The unique
behaviour is the URL pattern (``/serving-endpoints/{endpoint_name}/invocations``)
and pass-through of reasoning / thinking extras.
"""

from __future__ import annotations

import orjson
import pytest

from arcllm.providers.base import ProviderConfig
from arcllm.providers.databricks_adapter import DatabricksAdapter


@pytest.fixture
def adapter() -> DatabricksAdapter:
    return DatabricksAdapter(
        ProviderConfig(api_key="dapi-token", api_base="https://example.databricks.com")
    )


class TestDatabricksRequestBuild:
    def test_chat_url_uses_endpoint_name(self, adapter: DatabricksAdapter) -> None:
        req = adapter.build_request(
            model="databricks-claude-sonnet-4-5",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=64,
        )
        assert req.url.endswith("/serving-endpoints/databricks-claude-sonnet-4-5/invocations")
        body = orjson.loads(req.body or b"")
        # Model id is in the URL, not the body.
        assert "model" not in body
        assert body["messages"][0]["role"] == "user"
        assert body["max_tokens"] == 64

    def test_thinking_budget_passes_through_for_claude_endpoints(
        self, adapter: DatabricksAdapter
    ) -> None:
        req = adapter.build_request(
            model="databricks-claude-opus-4-7",
            messages=[{"role": "user", "content": "solve"}],
            thinking_budget=2048,
            max_tokens=4096,
        )
        body = orjson.loads(req.body or b"")
        # Databricks forwards extra body keys to the underlying Anthropic
        # endpoint; we expose ``thinking_budget`` directly so callers don't
        # need to construct the Anthropic ``thinking`` block themselves.
        assert body["thinking_budget"] == 2048

    def test_reasoning_effort_passes_through_for_gpt5_endpoints(
        self, adapter: DatabricksAdapter
    ) -> None:
        req = adapter.build_request(
            model="databricks-gpt-5",
            messages=[{"role": "user", "content": "hi"}],
            reasoning_effort="high",
            max_tokens=64,
        )
        body = orjson.loads(req.body or b"")
        assert body["reasoning_effort"] == "high"

    def test_response_format_passes_through(self, adapter: DatabricksAdapter) -> None:
        rf = {"type": "json_schema", "json_schema": {"name": "x", "schema": {}}}
        req = adapter.build_request(
            model="databricks-meta-llama-3-3-70b-instruct",
            messages=[{"role": "user", "content": "hi"}],
            response_format=rf,
        )
        body = orjson.loads(req.body or b"")
        assert body["response_format"] == rf

    def test_tools_pass_through(self, adapter: DatabricksAdapter) -> None:
        tools = [{"type": "function", "function": {"name": "f", "parameters": {}}}]
        req = adapter.build_request(
            model="databricks-gpt-5",
            messages=[{"role": "user", "content": "hi"}],
            tools=tools,
            tool_choice="auto",
        )
        body = orjson.loads(req.body or b"")
        assert body["tools"] == tools
        assert body["tool_choice"] == "auto"


class TestDatabricksAuth:
    def test_uses_databricks_token_env_when_api_key_missing(self, monkeypatch) -> None:
        monkeypatch.setenv("DATABRICKS_TOKEN", "env-token")
        adapter = DatabricksAdapter(ProviderConfig(api_base="https://example.databricks.com"))
        headers = adapter._build_headers()
        assert headers["Authorization"] == "Bearer env-token"

    def test_explicit_config_token_wins(self) -> None:
        adapter = DatabricksAdapter(
            ProviderConfig(api_key="cfg-token", api_base="https://example.databricks.com")
        )
        headers = adapter._build_headers()
        assert headers["Authorization"] == "Bearer cfg-token"


class TestDatabricksEmbedding:
    def test_embedding_url_is_invocations(self, adapter: DatabricksAdapter) -> None:
        req = adapter.build_embedding_request(
            model="databricks-gte-large-en",
            input=["hello", "world"],
        )
        assert req.url.endswith("/serving-endpoints/databricks-gte-large-en/invocations")
        body = orjson.loads(req.body or b"")
        assert body["input"] == ["hello", "world"]
