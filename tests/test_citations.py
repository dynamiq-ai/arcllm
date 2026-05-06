"""Citations parsing across grounded providers.

Each provider exposes a different citation surface; arcllm normalises all of
them into ``Message.citations: list[Citation]``. These fixture-based tests
lock in the parser for each shape.
"""

from __future__ import annotations

import orjson
import pytest

from arcllm.providers.anthropic_adapter import AnthropicAdapter
from arcllm.providers.base import ProviderConfig
from arcllm.providers.gemini_adapter import GeminiAdapter
from arcllm.providers.perplexity_adapter import PerplexityAdapter


@pytest.fixture
def perplexity() -> PerplexityAdapter:
    return PerplexityAdapter(ProviderConfig(api_key="test"))


@pytest.fixture
def gemini() -> GeminiAdapter:
    return GeminiAdapter(config=ProviderConfig(api_key="test"))


@pytest.fixture
def anthropic() -> AnthropicAdapter:
    return AnthropicAdapter(ProviderConfig(api_key="test"))


# ---------------------------------------------------------------------------
# Perplexity — top-level `citations` array (legacy + search_results shapes)
# ---------------------------------------------------------------------------


class TestPerplexityCitations:
    def test_legacy_url_only_citations(self, perplexity: PerplexityAdapter) -> None:
        body = {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 0,
            "model": "sonar",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "The sky is blue."},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            "citations": [
                "https://example.com/sky",
                "https://example.com/blue",
            ],
        }
        response = perplexity.parse_response(orjson.dumps(body), "sonar")
        msg = response.choices[0].message
        assert msg.citations is not None
        assert [c.url for c in msg.citations] == [
            "https://example.com/sky",
            "https://example.com/blue",
        ]
        # Legacy URL-only citations should have no title/snippet.
        assert all(c.title is None and c.snippet is None for c in msg.citations)

    def test_search_results_with_title_and_snippet(self, perplexity: PerplexityAdapter) -> None:
        body = {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 0,
            "model": "sonar-pro",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Latest news:"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            "search_results": [
                {
                    "url": "https://news.example.com/a",
                    "title": "Headline A",
                    "snippet": "Summary of A",
                },
                {
                    "url": "https://news.example.com/b",
                    "title": "Headline B",
                    "snippet": "Summary of B",
                },
            ],
        }
        response = perplexity.parse_response(orjson.dumps(body), "sonar-pro")
        msg = response.choices[0].message
        assert msg.citations is not None
        assert msg.citations[0].url == "https://news.example.com/a"
        assert msg.citations[0].title == "Headline A"
        assert msg.citations[0].snippet == "Summary of A"

    def test_no_citations_field_leaves_message_unchanged(
        self, perplexity: PerplexityAdapter
    ) -> None:
        body = {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 0,
            "model": "sonar",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "ok"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
        response = perplexity.parse_response(orjson.dumps(body), "sonar")
        assert response.choices[0].message.citations is None


# ---------------------------------------------------------------------------
# Gemini — groundingMetadata.{groundingChunks,groundingSupports}
# ---------------------------------------------------------------------------


class TestGeminiGrounding:
    def test_grounding_chunks_with_offsets(self, gemini: GeminiAdapter) -> None:
        body = {
            "candidates": [
                {
                    "content": {
                        "parts": [{"text": "The capital of France is Paris."}],
                        "role": "model",
                    },
                    "finishReason": "STOP",
                    "groundingMetadata": {
                        "groundingChunks": [
                            {
                                "web": {
                                    "uri": "https://en.wikipedia.org/wiki/Paris",
                                    "title": "Paris",
                                }
                            },
                            {"web": {"uri": "https://france.fr", "title": "France"}},
                        ],
                        "groundingSupports": [
                            {
                                "segment": {"startIndex": 25, "endIndex": 30, "text": "Paris"},
                                "groundingChunkIndices": [0],
                            }
                        ],
                    },
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 5,
                "candidatesTokenCount": 7,
                "totalTokenCount": 12,
            },
        }
        response = gemini.parse_response(orjson.dumps(body), "gemini-2.5-pro")
        msg = response.choices[0].message
        assert msg.citations is not None
        assert len(msg.citations) == 2

        # Chunk 0 has a grounding support → start/end set.
        first = next(c for c in msg.citations if c.url == "https://en.wikipedia.org/wiki/Paris")
        assert first.title == "Paris"
        assert first.start_index == 25
        assert first.end_index == 30

        # Chunk 1 has no support → offsets None.
        second = next(c for c in msg.citations if c.url == "https://france.fr")
        assert second.start_index is None
        assert second.end_index is None

    def test_no_grounding_metadata_means_no_citations(self, gemini: GeminiAdapter) -> None:
        body = {
            "candidates": [
                {
                    "content": {"parts": [{"text": "hi"}], "role": "model"},
                    "finishReason": "STOP",
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 1,
                "candidatesTokenCount": 1,
                "totalTokenCount": 2,
            },
        }
        response = gemini.parse_response(orjson.dumps(body), "gemini-2.5-flash")
        assert response.choices[0].message.citations is None


# ---------------------------------------------------------------------------
# Anthropic — web_search_tool_result blocks + text-block annotations
# ---------------------------------------------------------------------------


class TestAnthropicCitations:
    def test_web_search_tool_result_block(self, anthropic: AnthropicAdapter) -> None:
        body = {
            "id": "msg_test",
            "type": "message",
            "role": "assistant",
            "model": "claude-sonnet-4-5-20250929",
            "stop_reason": "end_turn",
            "content": [
                {
                    "type": "web_search_tool_result",
                    "tool_use_id": "srvtoolu_test",
                    "content": [
                        {
                            "type": "web_search_result",
                            "url": "https://example.com/news",
                            "title": "Example News",
                            "snippet": "Some snippet text",
                        }
                    ],
                },
                {
                    "type": "text",
                    "text": "Here is some breaking news.",
                },
            ],
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }
        response = anthropic.parse_response(orjson.dumps(body), "claude-sonnet-4-5-20250929")
        msg = response.choices[0].message
        assert msg.citations is not None
        assert len(msg.citations) == 1
        c = msg.citations[0]
        assert c.url == "https://example.com/news"
        assert c.title == "Example News"
        assert c.snippet == "Some snippet text"

    def test_text_block_annotation_takes_precedence(self, anthropic: AnthropicAdapter) -> None:
        """When the same URL appears in both the tool result and the text-block
        ``citations`` annotation, the annotation (with offsets) wins."""
        body = {
            "id": "msg_test",
            "type": "message",
            "role": "assistant",
            "model": "claude-sonnet-4-5-20250929",
            "stop_reason": "end_turn",
            "content": [
                {
                    "type": "web_search_tool_result",
                    "tool_use_id": "srvtoolu_test",
                    "content": [
                        {
                            "type": "web_search_result",
                            "url": "https://example.com/foo",
                            "title": "Foo (search title)",
                        }
                    ],
                },
                {
                    "type": "text",
                    "text": "According to Foo, the sky is blue.",
                    "citations": [
                        {
                            "url": "https://example.com/foo",
                            "title": "Foo (annotated title)",
                            "cited_text": "the sky is blue",
                            "start_char_index": 19,
                            "end_char_index": 34,
                        }
                    ],
                },
            ],
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }
        response = anthropic.parse_response(orjson.dumps(body), "claude-sonnet-4-5-20250929")
        msg = response.choices[0].message
        assert msg.citations is not None
        assert len(msg.citations) == 1
        c = msg.citations[0]
        assert c.url == "https://example.com/foo"
        # Annotation wins because it's processed first by the text block path.
        assert c.title == "Foo (annotated title)"
        assert c.start_index == 19
        assert c.end_index == 34

    def test_no_citations_when_no_grounded_content(self, anthropic: AnthropicAdapter) -> None:
        body = {
            "id": "msg_test",
            "type": "message",
            "role": "assistant",
            "model": "claude-haiku-4-5",
            "stop_reason": "end_turn",
            "content": [{"type": "text", "text": "Hi"}],
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }
        response = anthropic.parse_response(orjson.dumps(body), "claude-haiku-4-5")
        assert response.choices[0].message.citations is None
