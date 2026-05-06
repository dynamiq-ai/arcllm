"""Tests for ``arcllm.token_counter``.

When tiktoken is available (it's pulled in by ``arcllm[dev]``) we expect
exact counts for OpenAI-family models. Without tiktoken, the function falls
back to a chars/4 heuristic with a one-time warning.
"""

from __future__ import annotations

import warnings

import pytest

from arcllm import token_counter


class TestTokenCounter:
    def test_text_only(self):
        n = token_counter(model="gpt-4o-mini", text="hello world")
        assert isinstance(n, int)
        assert n >= 1
        # "hello world" is 11 chars; tiktoken gives 2, heuristic gives ceil(11/4)=3.
        # Either is fine for guard rails.
        assert n <= 5

    def test_messages(self):
        n = token_counter(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Tell me a joke."},
            ],
        )
        assert n > 5  # "system You are helpful.\nuser Tell me a joke."

    def test_empty_text(self):
        n = token_counter(model="gpt-4o-mini", text="")
        assert n == 0

    def test_requires_text_or_messages(self):
        with pytest.raises(ValueError, match="requires either"):
            token_counter(model="gpt-4o-mini")

    def test_rejects_both(self):
        with pytest.raises(ValueError, match="not both"):
            token_counter(model="gpt-4o-mini", text="hi", messages=[])

    def test_image_content_part_inflates_count(self):
        """Image parts should reserve room so guard rails don't undercount."""
        text_only = token_counter(
            model="gpt-4o", messages=[{"role": "user", "content": "describe"}]
        )
        with_image = token_counter(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "describe"},
                        {
                            "type": "image_url",
                            "image_url": {"url": "https://example.com/x.png"},
                        },
                    ],
                }
            ],
        )
        assert with_image > text_only

    def test_unknown_model_uses_heuristic_or_falls_back_cleanly(self):
        # We don't know whether tiktoken is installed, but either path must
        # return a non-negative int without raising.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            n = token_counter(model="unknown-provider/some-future-model", text="hello")
        assert n >= 1


class TestHeuristicFallback:
    """When tiktoken can't recognise the model, we use the heuristic."""

    def test_heuristic_path_produces_nonzero_count(self, monkeypatch):
        # Force the tiktoken path to return None by passing a model name that
        # ``_normalise_for_tiktoken`` doesn't recognise as OpenAI-family.
        from arcllm import tokens as tokens_mod

        monkeypatch.setattr(tokens_mod, "_heuristic_warned", True)
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # any warning becomes an exception
            n = tokens_mod._heuristic_count("hello world")
        assert n == 3  # ceil(11/4)
