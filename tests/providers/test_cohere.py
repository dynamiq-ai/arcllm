"""Unit tests for the Cohere v2 chat adapter.

Cohere v2 chat is OpenAI-shape compatible for messages content (it accepts
``image_url`` content arrays for vision-capable models like Aya Vision and
``command-a-vision-07-2025``). The adapter passes content through verbatim,
so these tests lock in:

- Multimodal pass-through (Aya Vision content array survives unchanged).
- Tool calling shape (Cohere v2 uses OpenAI-style ``tools`` and emits OpenAI-
  style ``tool_calls`` back).
- ``stop`` -> ``stop_sequences`` rename.
- ``response_format=json_object`` and ``json_schema`` translation.
"""

from __future__ import annotations

import orjson
import pytest

from arcllm.providers.base import ProviderConfig
from arcllm.providers.cohere_adapter import CohereAdapter


@pytest.fixture
def adapter() -> CohereAdapter:
    return CohereAdapter(ProviderConfig(api_key="test-key"))


class TestCohereRequestBuild:
    def test_basic_chat_request(self, adapter: CohereAdapter) -> None:
        req = adapter.build_request(
            model="command-r-08-2024",
            messages=[{"role": "user", "content": "hi"}],
        )
        body = orjson.loads(req.body or b"")
        assert body["model"] == "command-r-08-2024"
        assert body["messages"][0]["role"] == "user"
        assert body["messages"][0]["content"] == "hi"

    def test_system_message_collapses_into_preamble(self, adapter: CohereAdapter) -> None:
        """Cohere v2 lifts ``role=system`` messages into a top-level ``preamble``
        field rather than keeping them in the message stream."""
        req = adapter.build_request(
            model="command-r-08-2024",
            messages=[
                {"role": "system", "content": "You are helpful"},
                {"role": "user", "content": "hi"},
            ],
        )
        body = orjson.loads(req.body or b"")
        assert body.get("preamble") == "You are helpful"
        # System message must NOT appear in the messages array.
        assert all(m["role"] != "system" for m in body["messages"])
        assert any(m["role"] == "user" for m in body["messages"])

    def test_stop_param_renames_to_stop_sequences(self, adapter: CohereAdapter) -> None:
        req = adapter.build_request(
            model="command-r-08-2024",
            messages=[{"role": "user", "content": "hi"}],
            stop=["END", "STOP"],
        )
        body = orjson.loads(req.body or b"")
        assert body["stop_sequences"] == ["END", "STOP"]
        assert "stop" not in body

    def test_tools_pass_through_with_function_shape(self, adapter: CohereAdapter) -> None:
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the weather",
                    "parameters": {
                        "type": "object",
                        "properties": {"location": {"type": "string"}},
                    },
                },
            }
        ]
        req = adapter.build_request(
            model="command-a-03-2025",
            messages=[{"role": "user", "content": "hi"}],
            tools=tools,
        )
        body = orjson.loads(req.body or b"")
        assert body["tools"][0]["type"] == "function"
        assert body["tools"][0]["function"]["name"] == "get_weather"

    def test_response_format_json_object(self, adapter: CohereAdapter) -> None:
        req = adapter.build_request(
            model="command-a-03-2025",
            messages=[{"role": "user", "content": "hi"}],
            response_format={"type": "json_object"},
        )
        body = orjson.loads(req.body or b"")
        # Cohere accepts OpenAI-style {"type": "json_object"} natively.
        assert body["response_format"]["type"] == "json_object"

    def test_response_format_json_schema(self, adapter: CohereAdapter) -> None:
        rf = {
            "type": "json_schema",
            "json_schema": {
                "name": "user",
                "schema": {"type": "object", "properties": {"name": {"type": "string"}}},
            },
        }
        req = adapter.build_request(
            model="command-a-03-2025",
            messages=[{"role": "user", "content": "hi"}],
            response_format=rf,
        )
        body = orjson.loads(req.body or b"")
        # Cohere converts json_schema to json_object with schema.
        assert body["response_format"]["type"] == "json_object"
        assert "schema" in body["response_format"]


class TestCohereMultimodalPassThrough:
    """Aya / command-a-vision: OpenAI-shape image_url arrays pass through unchanged."""

    def test_image_url_content_array_passes_through(self, adapter: CohereAdapter) -> None:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "describe"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "https://example.com/cat.png"},
                    },
                ],
            }
        ]
        req = adapter.build_request(model="command-a-vision-07-2025", messages=messages)
        body = orjson.loads(req.body or b"")
        # Cohere v2 accepts content arrays directly — no conversion needed.
        assert body["messages"][0]["content"] == messages[0]["content"]


class TestCohereEmbedding:
    def test_embedding_endpoint(self, adapter: CohereAdapter) -> None:
        req = adapter.build_embedding_request(
            model="embed-v4.0",
            input=["hello"],
        )
        body = orjson.loads(req.body or b"")
        assert body["model"] == "embed-v4.0"
        # v2 default — request only float-typed embeddings so the response
        # shape is uniform regardless of caller.
        assert body["embedding_types"] == ["float"]
        assert body["input_type"] == "search_document"

    def test_parse_v2_embedding_response_dict_shape(self, adapter: CohereAdapter) -> None:
        """Cohere /v2/embed wraps vectors under embeddings.float."""
        import json

        payload = {
            "id": "x",
            "embeddings": {"float": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]},
            "meta": {"billed_units": {"input_tokens": 4}},
        }
        resp = adapter.parse_embedding_response(json.dumps(payload).encode(), "embed-english-v3.0")
        assert len(resp.data) == 2
        assert resp.data[0].embedding == [0.1, 0.2, 0.3]
        assert resp.data[1].embedding == [0.4, 0.5, 0.6]
        assert resp.usage.prompt_tokens == 4

    def test_parse_v1_legacy_list_shape_still_works(self, adapter: CohereAdapter) -> None:
        """A custom api_base can still hit v1; we accept the flat list."""
        import json

        payload = {
            "id": "x",
            "embeddings": [[0.1, 0.2], [0.3, 0.4]],
            "meta": {"billed_units": {"input_tokens": 2}},
        }
        resp = adapter.parse_embedding_response(json.dumps(payload).encode(), "embed-v1")
        assert len(resp.data) == 2
        assert resp.data[0].embedding == [0.1, 0.2]
