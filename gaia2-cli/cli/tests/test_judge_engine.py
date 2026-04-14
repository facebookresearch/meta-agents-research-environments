# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
"""Tests for the CLI-owned judge engine factory."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest
from gaia2_cli.judge import RateLimitError, create_litellm_engine


def _mock_completion_response(text: str) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text))]
    )


def test_create_litellm_engine_normalizes_openai_compatible_requests(
    monkeypatch,
) -> None:
    calls: list[dict] = []

    def completion(**kwargs):
        calls.append(kwargs)
        return _mock_completion_response("hello")

    monkeypatch.setitem(sys.modules, "litellm", SimpleNamespace(completion=completion))

    engine = create_litellm_engine(
        model="judge-model",
        provider="openai",
        base_url="https://example.invalid/v1",
        validate=False,
        api_key="judge-key",
    )

    content, info = engine([{"role": "user", "content": "Say hello"}])

    assert content == "hello"
    assert info == {"model": "judge-model"}
    assert calls == [
        {
            "model": "openai/judge-model",
            "messages": [{"role": "user", "content": "Say hello"}],
            "api_base": "https://example.invalid/v1",
            "api_key": "judge-key",
            "max_retries": 12,
            "temperature": 0,
        }
    ]


def test_create_litellm_engine_raises_rate_limit_error(monkeypatch) -> None:
    def completion(**kwargs):
        raise RuntimeError("429 Too Many Requests")

    monkeypatch.setitem(sys.modules, "litellm", SimpleNamespace(completion=completion))

    engine = create_litellm_engine(
        model="openai/test-model",
        provider="openai",
        validate=False,
    )

    with pytest.raises(RateLimitError):
        engine([{"role": "user", "content": "Say hello"}])


def test_create_litellm_engine_uses_explicit_api_key(monkeypatch) -> None:
    calls: list[dict] = []

    def completion(**kwargs):
        calls.append(kwargs)
        return _mock_completion_response("hello")

    monkeypatch.setitem(sys.modules, "litellm", SimpleNamespace(completion=completion))

    engine = create_litellm_engine(
        model="judge-model",
        provider="anthropic",
        base_url="https://example.invalid/v1",
        validate=False,
        api_key="explicit-key",
    )

    content, info = engine([{"role": "user", "content": "Say hello"}])

    assert content == "hello"
    assert info == {"model": "judge-model"}
    assert calls[0]["model"] == "judge-model"
    assert calls[0]["api_key"] == "explicit-key"
