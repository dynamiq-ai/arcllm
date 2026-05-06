"""Unit tests for the Mistral adapter.

Mistral's API is OpenAI-shaped, so most of the request building is inherited
from ``OpenAIAdapter``. These tests lock in:

- Mistral-specific param mapping (``seed`` → ``random_seed``, ``safe_prompt``).
- Vision pass-through: Pixtral models accept OpenAI-style ``image_url`` content
  blocks unchanged. We don't convert them — we pass them straight through.
- Tool calling stays in OpenAI shape end-to-end.
- Embedding requests target the right endpoint.
"""

from __future__ import annotations

import orjson
import pytest

from arcllm.providers.base import ProviderConfig
from arcllm.providers.mistral_adapter import MistralAdapter


@pytest.fixture
def adapter() -> MistralAdapter:
    return MistralAdapter(ProviderConfig(api_key="test-key"))


class TestMistralRequestBuild:
    def test_basic_chat_request(self, adapter: MistralAdapter) -> None:
        req = adapter.build_request(
            model="mistral-large-latest",
            messages=[{"role": "user", "content": "hi"}],
        )
        assert req.method == "POST"
        assert req.url.endswith("/chat/completions")
        body = orjson.loads(req.body or b"")
        assert body["model"] == "mistral-large-latest"
        assert body["messages"] == [{"role": "user", "content": "hi"}]
        assert body["stream"] is False
        assert "Authorization" in req.headers
        assert req.headers["Authorization"].startswith("Bearer ")

    def test_seed_param_maps_to_random_seed(self, adapter: MistralAdapter) -> None:
        """Mistral uses ``random_seed`` rather than ``seed``."""
        req = adapter.build_request(
            model="mistral-small-latest",
            messages=[{"role": "user", "content": "hi"}],
            seed=42,
        )
        body = orjson.loads(req.body or b"")
        assert body["random_seed"] == 42
        assert "seed" not in body

    def test_safe_prompt_param_pass_through(self, adapter: MistralAdapter) -> None:
        req = adapter.build_request(
            model="mistral-large-latest",
            messages=[{"role": "user", "content": "hi"}],
            safe_prompt=True,
        )
        body = orjson.loads(req.body or b"")
        assert body["safe_prompt"] is True

    def test_tools_pass_through(self, adapter: MistralAdapter) -> None:
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]
        req = adapter.build_request(
            model="mistral-large-latest",
            messages=[{"role": "user", "content": "weather?"}],
            tools=tools,
            tool_choice="auto",
        )
        body = orjson.loads(req.body or b"")
        assert body["tools"] == tools
        assert body["tool_choice"] == "auto"

    def test_stop_param_is_passed_through(self, adapter: MistralAdapter) -> None:
        """Mistral accepts ``stop`` as a list. Earlier versions silently dropped it."""
        req = adapter.build_request(
            model="mistral-large-latest",
            messages=[{"role": "user", "content": "Count: one"}],
            stop="STOP",
        )
        body = orjson.loads(req.body or b"")
        assert body["stop"] == ["STOP"]

        req2 = adapter.build_request(
            model="mistral-large-latest",
            messages=[{"role": "user", "content": "Count: one"}],
            stop=["END", "STOP"],
        )
        body2 = orjson.loads(req2.body or b"")
        assert body2["stop"] == ["END", "STOP"]

    def test_response_format_pass_through(self, adapter: MistralAdapter) -> None:
        rf = {"type": "json_schema", "json_schema": {"name": "x", "schema": {}}}
        req = adapter.build_request(
            model="mistral-large-latest",
            messages=[{"role": "user", "content": "x"}],
            response_format=rf,
        )
        body = orjson.loads(req.body or b"")
        assert body["response_format"] == rf


class TestMistralVisionPassThrough:
    """Pixtral models accept OpenAI-style content arrays. We don't convert."""

    def test_image_url_content_array_passes_through(self, adapter: MistralAdapter) -> None:
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
        req = adapter.build_request(model="pixtral-large-latest", messages=messages)
        body = orjson.loads(req.body or b"")
        # Content array preserved exactly — Mistral API accepts this directly.
        assert body["messages"] == messages

    def test_base64_data_url_passes_through(self, adapter: MistralAdapter) -> None:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "what is this?"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="},
                    },
                ],
            }
        ]
        req = adapter.build_request(model="pixtral-12b-2409", messages=messages)
        body = orjson.loads(req.body or b"")
        assert body["messages"][0]["content"][1]["image_url"]["url"].startswith(
            "data:image/png;base64,"
        )


class TestMistralEmbedding:
    def test_embedding_endpoint(self, adapter: MistralAdapter) -> None:
        req = adapter.build_embedding_request(
            model="mistral-embed",
            input=["hello", "world"],
        )
        assert req.url.endswith("/embeddings")
        body = orjson.loads(req.body or b"")
        assert body["model"] == "mistral-embed"
        assert body["input"] == ["hello", "world"]
