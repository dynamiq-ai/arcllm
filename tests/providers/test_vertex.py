"""Per-family dispatch tests for the Vertex AI adapter.

Vertex hosts four model families: Gemini (native), Anthropic Claude,
Mistral, and Llama (Meta). Each goes through a different publisher path on
the Vertex API but shares one OAuth-bearer auth scheme. These tests confirm:

- Family detection from model id.
- Each family's wire body matches the underlying provider's shape.
- The publisher-scoped URL has the correct ``publishers/<id>/models/<model>:<method>`` path.
"""

from __future__ import annotations

import orjson
import pytest

from arcllm.providers.base import ProviderConfig
from arcllm.providers.vertex_adapter import VertexAIAdapter, _get_vertex_family


def _make_adapter() -> VertexAIAdapter:
    return VertexAIAdapter(
        ProviderConfig(
            api_key="vertex-token",
            vertex_project="my-project",
            vertex_location="us-central1",
        )
    )


class TestVertexFamilyDetection:
    @pytest.mark.parametrize(
        "model,expected",
        [
            ("gemini-2.5-pro", "gemini"),
            ("gemini-2.5-flash", "gemini"),
            ("text-embedding-005", "gemini"),
            ("claude-sonnet-4-5", "anthropic"),
            ("claude-haiku-4-5@anthropic", "anthropic"),
            ("mistral-large-2411", "mistral"),
            ("codestral-2508", "mistral"),
            ("llama-4-maverick-17b", "meta"),
            ("meta/llama-3.3-70b-instruct-maas", "meta"),
        ],
    )
    def test_family_dispatch(self, model: str, expected: str) -> None:
        assert _get_vertex_family(model) == expected


class TestVertexGeminiPath:
    def test_native_gemini_uses_publishers_google_endpoint(self) -> None:
        adapter = _make_adapter()
        req = adapter.build_request(
            model="gemini-2.5-pro",
            messages=[{"role": "user", "content": "hi"}],
        )
        assert "publishers/google/models/gemini-2.5-pro:generateContent" in req.url
        assert "us-central1-aiplatform.googleapis.com" in req.url
        assert req.headers["Authorization"].startswith("Bearer ")

    def test_native_gemini_streaming_url_has_alt_sse(self) -> None:
        adapter = _make_adapter()
        req = adapter.build_request(
            model="gemini-2.5-flash",
            messages=[{"role": "user", "content": "hi"}],
            stream=True,
        )
        assert ":streamGenerateContent" in req.url
        assert "alt=sse" in req.url


class TestVertexAnthropicPath:
    def test_anthropic_uses_publishers_anthropic_rawpredict(self) -> None:
        adapter = _make_adapter()
        req = adapter.build_request(
            model="claude-sonnet-4-5",
            messages=[
                {"role": "system", "content": "Be concise."},
                {"role": "user", "content": "hi"},
            ],
            max_tokens=64,
        )
        assert "publishers/anthropic/models/claude-sonnet-4-5:rawPredict" in req.url
        body = orjson.loads(req.body or b"")
        assert body["anthropic_version"] == "vertex-2023-10-16"
        assert body["max_tokens"] == 64
        # System prompt extracted into the top-level field.
        assert body["system"] == "Be concise."
        # User message went into Anthropic-shape messages array.
        assert body["messages"][0]["role"] == "user"

    def test_anthropic_thinking_budget_block(self) -> None:
        adapter = _make_adapter()
        req = adapter.build_request(
            model="claude-opus-4-7",
            messages=[{"role": "user", "content": "solve"}],
            max_tokens=4096,
            thinking_budget=2048,
            temperature=0.7,
        )
        body = orjson.loads(req.body or b"")
        assert body["thinking"] == {"type": "enabled", "budget_tokens": 2048}
        assert "temperature" not in body

    def test_anthropic_streaming_uses_streamRawPredict(self) -> None:
        adapter = _make_adapter()
        req = adapter.build_request(
            model="claude-haiku-4-5",
            messages=[{"role": "user", "content": "hi"}],
            stream=True,
        )
        assert ":streamRawPredict" in req.url


class TestVertexMistralPath:
    def test_mistral_uses_publishers_mistralai_rawpredict(self) -> None:
        adapter = _make_adapter()
        req = adapter.build_request(
            model="mistral-large-2411",
            messages=[{"role": "user", "content": "hi"}],
            temperature=0.5,
            max_tokens=64,
        )
        assert "publishers/mistralai/models/mistral-large-2411:rawPredict" in req.url
        body = orjson.loads(req.body or b"")
        assert body["model"] == "mistral-large-2411"
        assert body["temperature"] == 0.5
        assert body["max_tokens"] == 64
        assert body["messages"][0]["role"] == "user"


class TestVertexMetaPath:
    def test_meta_uses_publishers_meta_rawpredict(self) -> None:
        adapter = _make_adapter()
        req = adapter.build_request(
            model="llama-4-maverick-17b-128e-instruct-maas",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=64,
        )
        assert (
            "publishers/meta/models/llama-4-maverick-17b-128e-instruct-maas:rawPredict" in req.url
        )
        body = orjson.loads(req.body or b"")
        assert body["model"] == "llama-4-maverick-17b-128e-instruct-maas"
        assert body["max_tokens"] == 64


class TestVertexEmbeddings:
    def test_embedding_uses_predict_endpoint_and_instances_payload(self) -> None:
        adapter = _make_adapter()
        req = adapter.build_embedding_request(
            model="text-embedding-005",
            input=["hello", "world"],
        )
        assert "publishers/google/models/text-embedding-005:predict" in req.url
        body = orjson.loads(req.body or b"")
        assert body["instances"] == [{"content": "hello"}, {"content": "world"}]
