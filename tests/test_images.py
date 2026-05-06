"""Tests for the image generation / variation / edit public API.

The OpenAI adapter is the reference implementation; other providers raise
``UnsupportedModelError`` until they're plumbed in.
"""

from __future__ import annotations

import json

import orjson
import pytest

from arcllm.exceptions import UnsupportedModelError
from arcllm.providers.base import ProviderConfig
from arcllm.providers.openai_adapter import OpenAIAdapter
from arcllm.types import ImageData, ImageResponse


@pytest.fixture
def openai() -> OpenAIAdapter:
    return OpenAIAdapter(ProviderConfig(api_key="test-key"))


class TestOpenAIImageGenerationRequest:
    def test_basic_request_targets_images_generations(self, openai: OpenAIAdapter) -> None:
        req = openai.build_image_generation_request(
            model="dall-e-3", prompt="A teal arch over a desert"
        )
        assert req.method == "POST"
        assert req.url.endswith("/images/generations")
        body = orjson.loads(req.body or b"")
        assert body["model"] == "dall-e-3"
        assert body["prompt"] == "A teal arch over a desert"
        assert req.headers["Authorization"] == "Bearer test-key"

    def test_optional_params_pass_through(self, openai: OpenAIAdapter) -> None:
        req = openai.build_image_generation_request(
            model="gpt-image-1",
            prompt="logo",
            n=2,
            size="1024x1024",
            quality="hd",
            style="vivid",
            response_format="b64_json",
            user="user-42",
            background="transparent",
        )
        body = orjson.loads(req.body or b"")
        assert body["n"] == 2
        assert body["size"] == "1024x1024"
        assert body["quality"] == "hd"
        assert body["style"] == "vivid"
        assert body["response_format"] == "b64_json"
        assert body["user"] == "user-42"
        assert body["background"] == "transparent"


class TestOpenAIImageVariationRequest:
    def test_multipart_body_includes_image_and_model(self, openai: OpenAIAdapter) -> None:
        req = openai.build_image_variation_request(
            model="dall-e-2",
            image=b"\x89PNG\r\n\x1a\n" + b"\x00" * 64,  # fake PNG bytes
            n=2,
            size="512x512",
        )
        assert req.url.endswith("/images/variations")
        assert req.headers["Content-Type"].startswith("multipart/form-data;")
        # Body must contain the image bytes verbatim and the model field.
        assert b'name="model"' in (req.body or b"")
        assert b"dall-e-2" in (req.body or b"")
        assert b'name="image"' in (req.body or b"")
        assert b'name="n"' in (req.body or b"")
        assert b'name="size"' in (req.body or b"")


class TestOpenAIImageEditRequest:
    def test_multipart_body_includes_prompt_and_optional_mask(self, openai: OpenAIAdapter) -> None:
        req = openai.build_image_edit_request(
            model="gpt-image-1",
            image=b"\x89PNG\r\n\x1a\n" + b"\x00" * 64,
            prompt="add a moon",
            mask=b"\x89PNG\r\n\x1a\n" + b"\xff" * 64,
            size="1024x1024",
        )
        assert req.url.endswith("/images/edits")
        body = req.body or b""
        assert b'name="prompt"' in body
        assert b"add a moon" in body
        assert b'name="mask"' in body


class TestOpenAIParseImageResponse:
    def test_dalle3_response_with_url_and_revised_prompt(self, openai: OpenAIAdapter) -> None:
        payload = {
            "created": 1715000000,
            "data": [
                {
                    "url": "https://example.com/img.png",
                    "revised_prompt": "A teal arch over a sunlit desert",
                }
            ],
        }
        resp = openai.parse_image_response(json.dumps(payload).encode(), "dall-e-3")
        assert isinstance(resp, ImageResponse)
        assert resp.created == 1715000000
        assert len(resp.data) == 1
        first: ImageData = resp.data[0]
        assert first.url == "https://example.com/img.png"
        assert first.revised_prompt == "A teal arch over a sunlit desert"
        assert first.b64_json is None

    def test_b64_response(self, openai: OpenAIAdapter) -> None:
        payload = {
            "created": 1715000000,
            "data": [{"b64_json": "iVBORw0KGgo="}],
        }
        resp = openai.parse_image_response(json.dumps(payload).encode(), "gpt-image-1")
        assert resp.data[0].b64_json == "iVBORw0KGgo="
        assert resp.data[0].url is None


class TestImageResponseSerialisation:
    def test_model_dump_omits_unset_fields(self) -> None:
        d = ImageData(url="https://example.com/x.png").model_dump()
        assert "url" in d
        assert "b64_json" not in d
        assert "revised_prompt" not in d


class TestUnsupportedProvider:
    def test_anthropic_raises_unsupported(self) -> None:
        from arcllm.providers.anthropic_adapter import AnthropicAdapter

        adapter = AnthropicAdapter(ProviderConfig(api_key="x"))
        with pytest.raises(UnsupportedModelError):
            adapter.build_image_generation_request(model="claude-haiku-4-5", prompt="impossible")
