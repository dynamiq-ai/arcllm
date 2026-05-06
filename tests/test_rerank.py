"""Tests for the rerank public API.

Cohere is the reference implementation in 0.4.0; other providers raise
``UnsupportedModelError``.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import orjson
import pytest

from arcllm.exceptions import UnsupportedModelError
from arcllm.providers.base import ProviderConfig
from arcllm.providers.cohere_adapter import CohereAdapter

if TYPE_CHECKING:
    from arcllm.types import RerankResponse


@pytest.fixture
def cohere() -> CohereAdapter:
    return CohereAdapter(ProviderConfig(api_key="cohere-test-key"))


class TestCohereRerankRequest:
    def test_basic_request_targets_v2_rerank(self, cohere: CohereAdapter) -> None:
        req = cohere.build_rerank_request(
            model="rerank-v3.5",
            query="who invented the python language?",
            documents=["Guido van Rossum", "Linus Torvalds", "Dennis Ritchie"],
            top_n=2,
            return_documents=True,
        )
        assert req.method == "POST"
        # CohereAdapter._api_base already includes ``/v2``; rerank is at /v2/rerank.
        assert req.url.endswith("/v2/rerank")
        assert "/v2/v2/rerank" not in req.url
        body = orjson.loads(req.body or b"")
        assert body["model"] == "rerank-v3.5"
        assert body["query"].startswith("who invented")
        assert body["documents"] == ["Guido van Rossum", "Linus Torvalds", "Dennis Ritchie"]
        assert body["top_n"] == 2
        assert body["return_documents"] is True
        assert req.headers["Authorization"] == "Bearer cohere-test-key"

    def test_omit_top_n_and_return_documents(self, cohere: CohereAdapter) -> None:
        req = cohere.build_rerank_request(
            model="rerank-multilingual-v3.0",
            query="bonjour",
            documents=["hi", "hello", "salut"],
            return_documents=False,
        )
        body = orjson.loads(req.body or b"")
        assert "top_n" not in body
        assert body["return_documents"] is False


class TestCohereParseRerankResponse:
    def test_parse_results_with_documents(self, cohere: CohereAdapter) -> None:
        payload = {
            "id": "abc123",
            "results": [
                {"index": 0, "relevance_score": 0.95, "document": {"text": "Guido"}},
                {"index": 2, "relevance_score": 0.42, "document": {"text": "Dennis"}},
            ],
        }
        resp: RerankResponse = cohere.parse_rerank_response(
            json.dumps(payload).encode(), "rerank-v3.5"
        )
        assert resp.id == "abc123"
        assert resp.model == "rerank-v3.5"
        assert len(resp.results) == 2
        first = resp.results[0]
        assert first.index == 0
        assert first.relevance_score == 0.95
        assert first.document == "Guido"

    def test_parse_results_without_documents(self, cohere: CohereAdapter) -> None:
        payload = {
            "id": "xyz",
            "results": [{"index": 7, "relevance_score": 0.1}],
        }
        resp = cohere.parse_rerank_response(json.dumps(payload).encode(), "rerank-v3.5")
        assert resp.results[0].index == 7
        assert resp.results[0].document is None


class TestRerankResponseSerialisation:
    def test_model_dump_omits_unset_document(self) -> None:
        from arcllm.types import RerankResult

        r = RerankResult(index=0, relevance_score=0.5)
        d = r.model_dump()
        assert "document" not in d
        assert d["index"] == 0
        assert d["relevance_score"] == 0.5


class TestUnsupportedProvider:
    def test_openai_raises_unsupported(self) -> None:
        from arcllm.providers.openai_adapter import OpenAIAdapter

        adapter = OpenAIAdapter(ProviderConfig(api_key="x"))
        with pytest.raises(UnsupportedModelError):
            adapter.build_rerank_request(
                model="text-embedding-3-large",
                query="hi",
                documents=["a", "b"],
            )


class TestRerankPublicAPI:
    """End-to-end coverage of ``arcllm.rerank`` with a stubbed HTTP client."""

    def test_rerank_dispatches_to_cohere_via_provider_prefix(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import arcllm
        from arcllm.http.client import HTTPClient, HTTPResponse

        captured: dict[str, object] = {}

        def fake_request(
            self: HTTPClient, *, method: str, url: str, headers: dict[str, str], body, timeout
        ) -> HTTPResponse:
            captured["method"] = method
            captured["url"] = url
            return HTTPResponse(
                status_code=200,
                headers={"content-type": "application/json"},
                body=b'{"id":"r1","results":[{"index":0,"relevance_score":0.9,"document":{"text":"a"}}]}',
                request_id="req-test",
            )

        monkeypatch.setattr(HTTPClient, "request", fake_request)

        resp = arcllm.rerank(
            model="cohere/rerank-v3.5",
            query="hello",
            documents=["a", "b"],
            api_key="cohere-test",
            top_n=1,
        )
        assert captured["method"] == "POST"
        assert "/v2/rerank" in str(captured["url"])
        assert resp.results[0].index == 0
        assert resp.results[0].relevance_score == 0.9
        assert resp.results[0].document == "a"

    def test_rerank_propagates_provider_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import arcllm
        from arcllm.exceptions import RateLimitError
        from arcllm.http.client import HTTPClient, HTTPResponse

        def fake_429(
            self: HTTPClient, *, method: str, url: str, headers: dict[str, str], body, timeout
        ) -> HTTPResponse:
            return HTTPResponse(
                status_code=429,
                headers={},
                body=b'{"message":"Too many requests"}',
                request_id=None,
            )

        monkeypatch.setattr(HTTPClient, "request", fake_429)

        with pytest.raises(RateLimitError):
            arcllm.rerank(
                model="cohere/rerank-v3.5",
                query="x",
                documents=["a"],
                api_key="cohere-test",
            )

    @pytest.mark.asyncio
    async def test_arerank_dispatches_async(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import arcllm
        from arcllm.http.async_client import AsyncHTTPClient, AsyncHTTPResponse

        async def fake_async_request(
            self: AsyncHTTPClient,
            *,
            method: str,
            url: str,
            headers: dict[str, str],
            body,
            timeout,
        ) -> AsyncHTTPResponse:
            return AsyncHTTPResponse(
                status_code=200,
                headers={},
                body=b'{"id":"r2","results":[{"index":1,"relevance_score":0.5}]}',
                request_id="req-async",
            )

        monkeypatch.setattr(AsyncHTTPClient, "request", fake_async_request)

        resp = await arcllm.arerank(
            model="cohere/rerank-v3.5",
            query="hi",
            documents=["a", "b"],
            api_key="cohere-test",
        )
        assert resp.results[0].index == 1
        assert resp.results[0].document is None
