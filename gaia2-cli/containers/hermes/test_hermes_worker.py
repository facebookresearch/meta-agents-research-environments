# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
"""Unit tests for Hermes worker trace logging.

Run with:
    cd gaia2-cli
    python3 -m pytest containers/hermes/test_hermes_worker.py -v
"""

import json
import os
import sys
from types import SimpleNamespace

import pytest

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

import hermes_worker as worker  # noqa: E402


@pytest.fixture(autouse=True)
def trace_file(tmp_path, monkeypatch):
    trace_file = tmp_path / "trace.jsonl"
    monkeypatch.setattr(worker, "TRACE_FILE", str(trace_file))
    monkeypatch.setattr(worker, "_trace_seq", 0)
    if hasattr(worker._trace_depth, "value"):
        del worker._trace_depth.value
    yield trace_file
    if hasattr(worker._trace_depth, "value"):
        del worker._trace_depth.value


def _read_trace(trace_file):
    lines = trace_file.read_text().strip().splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def _make_openai_response():
    return SimpleNamespace(
        model="test-model",
        usage={"prompt_tokens": 7, "completion_tokens": 3},
        choices=[
            SimpleNamespace(
                index=0,
                finish_reason="stop",
                message=SimpleNamespace(
                    role="assistant",
                    content="Hello from Hermes.",
                    tool_calls=[
                        SimpleNamespace(
                            id="call_1",
                            type="function",
                            function=SimpleNamespace(
                                name="exec",
                                arguments='{"command":"date"}',
                            ),
                        )
                    ],
                ),
            )
        ],
    )


def test_streaming_wrapper_logs_single_entry_when_delegating(trace_file):
    class FakeAgent:
        api_mode = "chat_completions"

        def __init__(self):
            self._interruptible_api_call = self._api_call
            self._interruptible_streaming_api_call = self._streaming_call

        def _api_call(self, api_kwargs):
            return _make_openai_response()

        def _streaming_call(self, api_kwargs, *, on_first_delta=None):
            return self._interruptible_api_call(api_kwargs)

    agent = FakeAgent()
    worker._install_trace_logging(agent, "https://compat.example/v1")

    request = {
        "model": "test-model",
        "messages": [{"role": "user", "content": "hi"}],
    }
    response = agent._interruptible_streaming_api_call(request)

    assert response.choices[0].message.content == "Hello from Hermes."

    entries = _read_trace(trace_file)
    assert len(entries) == 1

    entry = entries[0]
    assert entry["seq"] == 1
    assert entry["type"] == "llm_call"
    assert entry["url"].endswith("/chat/completions")
    assert entry["http_status"] == 200
    assert entry["request"] == request
    assert "T" in entry["timestamp"]

    raw = json.loads(entry["raw_response"])
    assert raw["model"] == "test-model"
    assert raw["choices"][0]["message"]["content"] == "Hello from Hermes."
    assert raw["choices"][0]["message"]["tool_calls"][0]["function"]["name"] == "exec"


def test_error_path_writes_trace_entry_with_status(trace_file):
    class FakeStatusError(Exception):
        def __init__(self):
            super().__init__("rate limited")
            self.status_code = 429
            self.body = {"error": {"message": "rate limited"}}

    class FakeAgent:
        api_mode = "chat_completions"

        def __init__(self):
            self._interruptible_api_call = self._api_call
            self._interruptible_streaming_api_call = self._streaming_call

        def _api_call(self, api_kwargs):
            raise FakeStatusError()

        def _streaming_call(self, api_kwargs, *, on_first_delta=None):
            return self._api_call(api_kwargs)

    agent = FakeAgent()
    worker._install_trace_logging(agent, "https://api.openai.com/v1")

    with pytest.raises(FakeStatusError):
        agent._interruptible_api_call({"model": "gpt-5"})

    entries = _read_trace(trace_file)
    assert len(entries) == 1

    entry = entries[0]
    assert entry["http_status"] == 429
    assert entry["url"].endswith("/chat/completions")

    raw = json.loads(entry["raw_response"])
    assert raw["error"] == "rate limited"
    assert raw["type"] == "FakeStatusError"
    assert raw["body"]["error"]["message"] == "rate limited"


def test_anthropic_trace_uses_messages_endpoint(trace_file):
    class FakeAgent:
        api_mode = "anthropic_messages"

        def __init__(self):
            self._interruptible_api_call = self._api_call
            self._interruptible_streaming_api_call = self._streaming_call

        def _api_call(self, api_kwargs):
            return {
                "model": "claude-opus-4.6",
                "content": [{"type": "text", "text": "done"}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 10, "output_tokens": 2},
            }

        def _streaming_call(self, api_kwargs, *, on_first_delta=None):
            return self._api_call(api_kwargs)

    agent = FakeAgent()
    worker._install_trace_logging(agent, "https://api.anthropic.com")

    agent._interruptible_api_call({"model": "claude-opus-4.6"})

    entries = _read_trace(trace_file)
    assert len(entries) == 1
    assert entries[0]["url"] == "https://api.anthropic.com/v1/messages"


def test_resolve_base_url_prefers_base_url_env(monkeypatch):
    monkeypatch.setenv("BASE_URL", "https://proxy.example/v1/")

    assert worker._resolve_base_url("openai") == "https://proxy.example/v1"


def test_resolve_base_url_returns_empty_for_unknown_provider(monkeypatch):
    monkeypatch.delenv("BASE_URL", raising=False)

    assert worker._resolve_base_url("custom-provider") == ""


def test_normalize_chat_reasoning_request_adds_reasoning_effort():
    request = {
        "model": "fireworks-kimi-k2p5-od",
        "messages": [{"role": "user", "content": "hi"}],
    }

    normalized = worker._normalize_chat_reasoning_request(
        request,
        api_mode="chat_completions",
        thinking="high",
    )

    assert normalized is not request
    assert normalized["reasoning_effort"] == "high"
    assert "reasoning_effort" not in request


def test_normalize_chat_reasoning_request_translates_extra_body_reasoning():
    request = {
        "model": "fireworks-kimi-k2p5-od",
        "messages": [{"role": "user", "content": "hi"}],
        "extra_body": {
            "reasoning": {"effort": "medium"},
            "metadata": {"source": "test"},
        },
    }

    normalized = worker._normalize_chat_reasoning_request(
        request,
        api_mode="chat_completions",
        thinking="high",
    )

    assert normalized["reasoning_effort"] == "medium"
    assert normalized["extra_body"] == {"metadata": {"source": "test"}}


def test_normalize_chat_reasoning_request_skips_responses_api():
    request = {
        "model": "gpt-5.4",
        "input": [{"role": "user", "content": "hi"}],
    }

    normalized = worker._normalize_chat_reasoning_request(
        request,
        api_mode="codex_responses",
        thinking="high",
    )

    assert normalized is request
    assert "reasoning_effort" not in normalized


def test_patch_chat_reasoning_effort_mutates_api_call_request():
    calls = []

    class FakeAgent:
        api_mode = "chat_completions"

        def __init__(self):
            self._interruptible_api_call = self._api_call
            self._interruptible_streaming_api_call = self._streaming_call

        def _api_call(self, api_kwargs):
            calls.append(api_kwargs)
            return {"ok": True}

        def _streaming_call(self, api_kwargs, *, on_first_delta=None):
            return self._interruptible_api_call(api_kwargs)

    agent = FakeAgent()
    worker._patch_chat_reasoning_effort(agent, "high")

    agent._interruptible_api_call(
        {
            "model": "fireworks-kimi-k2p5-od",
            "messages": [{"role": "user", "content": "hi"}],
        }
    )

    assert calls == [
        {
            "model": "fireworks-kimi-k2p5-od",
            "messages": [{"role": "user", "content": "hi"}],
            "reasoning_effort": "high",
        }
    ]


def test_trace_logging_records_normalized_reasoning_effort(trace_file):
    class FakeAgent:
        api_mode = "chat_completions"

        def __init__(self):
            self._interruptible_api_call = self._api_call
            self._interruptible_streaming_api_call = self._streaming_call

        def _api_call(self, api_kwargs):
            return _make_openai_response()

        def _streaming_call(self, api_kwargs, *, on_first_delta=None):
            return self._interruptible_api_call(api_kwargs)

    agent = FakeAgent()
    worker._disable_hermes_streaming_for_gaia2(agent)
    worker._patch_chat_reasoning_effort(agent, "high")
    worker._install_trace_logging(agent, "https://compat.example/v1", thinking="high")

    agent._interruptible_streaming_api_call(
        {
            "model": "fireworks-kimi-k2p5-od",
            "messages": [{"role": "user", "content": "hi"}],
        }
    )

    entries = _read_trace(trace_file)
    assert len(entries) == 1
    assert entries[0]["request"]["reasoning_effort"] == "high"


def test_patch_hermes_tool_call_ids_normalizes_provider_objects():
    class FakeAgent:
        @staticmethod
        def _sanitize_api_messages(messages):
            return messages

        def _build_assistant_message(self, assistant_message, finish_reason):
            return {
                "role": "assistant",
                "content": "",
                "finish_reason": finish_reason,
                "tool_calls": [
                    {
                        "id": tool_call.id.strip(),
                        "type": "function",
                        "function": {
                            "name": tool_call.function.name,
                            "arguments": tool_call.function.arguments,
                        },
                    }
                    for tool_call in assistant_message.tool_calls
                ],
            }

    worker._patch_hermes_tool_call_ids(FakeAgent)

    agent = FakeAgent()
    assistant_message = SimpleNamespace(
        tool_calls=[
            SimpleNamespace(
                id=" functions.terminal:0 ",
                call_id=None,
                response_item_id=" fc_123 ",
                function=SimpleNamespace(name="terminal", arguments="{}"),
            )
        ]
    )

    message = agent._build_assistant_message(assistant_message, "tool_calls")

    assert message["tool_calls"][0]["id"] == "functions.terminal:0"
    assert assistant_message.tool_calls[0].id == "functions.terminal:0"
    assert assistant_message.tool_calls[0].response_item_id == "fc_123"


def test_patch_hermes_tool_call_ids_normalizes_stored_messages():
    class FakeAgent:
        def _build_assistant_message(self, assistant_message, finish_reason):
            return {}

        @staticmethod
        def _sanitize_api_messages(messages):
            return messages

    worker._patch_hermes_tool_call_ids(FakeAgent)

    messages = [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": " functions.terminal:0 ",
                    "call_id": " functions.terminal:0 ",
                    "response_item_id": " fc_123 ",
                    "type": "function",
                    "function": {"name": "terminal", "arguments": "{}"},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": " functions.terminal:0 ",
            "content": "ok",
        },
    ]

    sanitized = FakeAgent._sanitize_api_messages(messages)

    assert sanitized[0]["tool_calls"][0]["id"] == "functions.terminal:0"
    assert sanitized[0]["tool_calls"][0]["call_id"] == "functions.terminal:0"
    assert sanitized[0]["tool_calls"][0]["response_item_id"] == "fc_123"
    assert sanitized[1]["tool_call_id"] == "functions.terminal:0"


def test_disable_hermes_streaming_for_gaia2_uses_non_streaming_path():
    calls = []

    class FakeAgent:
        def __init__(self):
            self._interruptible_api_call = self._api_call
            self._interruptible_streaming_api_call = self._streaming_call

        def _api_call(self, api_kwargs):
            calls.append(("api", api_kwargs))
            return {"ok": True}

        def _streaming_call(self, api_kwargs, *, on_first_delta=None):
            calls.append(("stream", api_kwargs))
            return {"stream": True}

    agent = FakeAgent()
    worker._disable_hermes_streaming_for_gaia2(agent)

    result = agent._interruptible_streaming_api_call(
        {"model": "kimi"}, on_first_delta=lambda: None
    )

    assert result == {"ok": True}
    assert calls == [("api", {"model": "kimi"})]
