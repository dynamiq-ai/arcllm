"""Unit tests for the Ollama adapter.

Ollama runs a local server. The adapter targets Ollama's OpenAI-compatible
``/v1/chat/completions`` endpoint, which accepts OpenAI-style ``image_url``
content arrays unchanged for vision-capable models (LLaVA, Llama 3.2-Vision,
Qwen-VL, Gemma 3 4B+, etc.).

These tests lock in:

- Default ``OLLAMA_HOST`` discovery and override.
- ``max_tokens`` → ``options.num_predict`` mapping (Ollama-specific).
- ``response_format`` mapping to Ollama's ``format: json`` field.
- Vision pass-through (no base64 → ``images`` field rewriting needed).
"""

from __future__ import annotations

import orjson
import pytest

from arcllm.providers.base import ProviderConfig
from arcllm.providers.ollama_adapter import OllamaAdapter


@pytest.fixture
def adapter() -> OllamaAdapter:
    return OllamaAdapter(ProviderConfig())


class TestOllamaRequestBuild:
    def test_default_host(self, adapter: OllamaAdapter) -> None:
        req = adapter.build_request(
            model="llama3.3",
            messages=[{"role": "user", "content": "hi"}],
        )
        assert req.url.startswith("http://localhost:11434/")

    def test_custom_api_base(self) -> None:
        a = OllamaAdapter(ProviderConfig(api_base="http://gpu-box:11434/"))
        req = a.build_request(
            model="qwen3",
            messages=[{"role": "user", "content": "hi"}],
        )
        assert req.url == "http://gpu-box:11434/v1/chat/completions"

    def test_max_tokens_maps_to_num_predict(self, adapter: OllamaAdapter) -> None:
        req = adapter.build_request(
            model="llama3.3",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=128,
        )
        body = orjson.loads(req.body or b"")
        assert body["options"]["num_predict"] == 128
        assert "max_tokens" not in body

    def test_temperature_top_p_seed_pass_to_options(self, adapter: OllamaAdapter) -> None:
        req = adapter.build_request(
            model="llama3.3",
            messages=[{"role": "user", "content": "hi"}],
            temperature=0.2,
            top_p=0.9,
            seed=42,
        )
        body = orjson.loads(req.body or b"")
        assert body["options"]["temperature"] == 0.2
        assert body["options"]["top_p"] == 0.9
        assert body["options"]["seed"] == 42

    def test_json_schema_response_format_maps_to_format_json(self, adapter: OllamaAdapter) -> None:
        req = adapter.build_request(
            model="llama3.3",
            messages=[{"role": "user", "content": "hi"}],
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "x", "schema": {}},
            },
        )
        body = orjson.loads(req.body or b"")
        assert body["format"] == "json"

    def test_tools_pass_through(self, adapter: OllamaAdapter) -> None:
        tools = [{"type": "function", "function": {"name": "f", "parameters": {}}}]
        req = adapter.build_request(
            model="llama3.3",
            messages=[{"role": "user", "content": "hi"}],
            tools=tools,
        )
        body = orjson.loads(req.body or b"")
        assert body["tools"] == tools


class TestOllamaVisionPassThrough:
    """Ollama's OpenAI-compat endpoint accepts OpenAI image_url content arrays."""

    def test_image_url_content_array_passes_through(self, adapter: OllamaAdapter) -> None:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "what's this?"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,iVBOR=="},
                    },
                ],
            }
        ]
        req = adapter.build_request(model="llava", messages=messages)
        body = orjson.loads(req.body or b"")
        assert body["messages"] == messages


class TestOllamaErrorParsing:
    def test_parse_error_with_json_body(self, adapter: OllamaAdapter) -> None:
        err = adapter.parse_error(500, b'{"error": "model not found"}')
        assert "model not found" in str(err)
        assert err.provider == "ollama"
        assert err.status_code == 500

    def test_parse_error_with_non_json_body(self, adapter: OllamaAdapter) -> None:
        err = adapter.parse_error(502, b"<html>bad gateway</html>")
        assert "bad gateway" in str(err)
        assert err.provider == "ollama"


class TestOllamaEmbedding:
    def test_embedding_endpoint(self, adapter: OllamaAdapter) -> None:
        req = adapter.build_embedding_request(
            model="nomic-embed-text",
            input=["hello"],
        )
        assert req.url.endswith("/v1/embeddings")
        body = orjson.loads(req.body or b"")
        assert body["model"] == "nomic-embed-text"
        assert body["input"] == ["hello"]
