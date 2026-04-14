# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
"""Tests for OpenClaw's Gaia2 adapter."""

from __future__ import annotations

import os
import sys
import types
from unittest import mock

import pytest

# Mock websockets before importing the adapter module.
_fake_ws = types.ModuleType("websockets")
_fake_ws.connect = mock.AsyncMock()
_fake_ws_exc = types.ModuleType("websockets.exceptions")
_fake_ws_exc.ConnectionClosed = type("ConnectionClosed", (Exception,), {})
_fake_ws.exceptions = _fake_ws_exc
sys.modules.setdefault("websockets", _fake_ws)
sys.modules.setdefault("websockets.exceptions", _fake_ws_exc)

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
_SHARED_DIR = os.path.join(_REPO_ROOT, "shared")
for _path in (
    _THIS_DIR,
    _REPO_ROOT,
    os.path.join(_REPO_ROOT, "cli"),
    os.path.join(_REPO_ROOT, "core"),
    _SHARED_DIR,
):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import gaia2_adapter as adapter  # noqa: E402
import gaia2_adapter_base  # noqa: E402


@pytest.fixture(autouse=True)
def reset_adapter_state(monkeypatch: pytest.MonkeyPatch) -> dict[str, mock.MagicMock]:
    """Reset mutable adapter globals and patch side effects for each test."""

    aui_mock = mock.MagicMock()
    buffer_mock = mock.MagicMock(side_effect=lambda response: {"seq": 1, **response})
    on_response = mock.MagicMock()

    monkeypatch.setattr(adapter, "write_aui_event", aui_mock)
    monkeypatch.setattr(adapter._state, "buffer_and_broadcast", buffer_mock)
    adapter._latest_chat_delta.clear()
    adapter._on_response = on_response

    yield {
        "aui": aui_mock,
        "buffer": buffer_mock,
        "on_response": on_response,
    }

    adapter._on_response = None
    adapter._latest_chat_delta.clear()


class TestMessageText:
    def test_message_text_from_string(self) -> None:
        assert adapter._message_text("hello") == "hello"

    @pytest.mark.parametrize(
        ("message", "expected"),
        [
            ({"text": "short answer"}, "short answer"),
            ({"content": "short answer"}, "short answer"),
            (
                {
                    "content": [
                        {"type": "tool_use", "name": "exec"},
                        {"type": "text", "text": "first"},
                        {"type": "text", "text": " second"},
                    ]
                },
                "first second",
            ),
        ],
    )
    def test_message_text_from_supported_shapes(
        self,
        message: object,
        expected: str,
    ) -> None:
        assert adapter._message_text(message) == expected

    def test_message_text_ignores_non_text_payloads(self) -> None:
        assert adapter._message_text({"content": [{"type": "tool_use"}]}) == ""
        assert adapter._message_text(123) == ""

    def test_response_text_reads_message_field(self) -> None:
        response = {"message": {"content": [{"type": "text", "text": "done"}]}}
        assert adapter._response_text(response) == "done"


class TestToolMessageDetection:
    @pytest.mark.parametrize(
        "message",
        [
            {"tool_calls": [{"id": "call-1"}]},
            {"toolUses": [{"id": "call-1"}]},
            {"content": [{"type": "tool_use", "name": "exec"}]},
            {"content": [{"toolCall": {"id": "call-1"}}]},
        ],
    )
    def test_tool_messages_are_detected(self, message: dict) -> None:
        assert adapter._is_tool_message(message) is True

    @pytest.mark.parametrize(
        "message",
        [
            {"content": [{"type": "text", "text": "hello"}]},
            {"content": "plain text"},
            "plain text",
        ],
    )
    def test_non_tool_messages_are_not_detected(self, message: object) -> None:
        assert adapter._is_tool_message(message) is False


class TestHandleChatEvent:
    def test_final_event_is_forwarded(
        self,
        reset_adapter_state: dict[str, mock.MagicMock],
    ) -> None:
        adapter._handle_chat_event(
            {
                "state": "final",
                "runId": "run-1",
                "sessionKey": "sess-1",
                "message": "The answer is 42.",
            }
        )

        reset_adapter_state["on_response"].assert_called_once_with(
            {
                "run_id": "run-1",
                "runId": "run-1",
                "sessionKey": "sess-1",
                "state": "final",
                "message": "The answer is 42.",
            }
        )

    def test_error_event_keeps_error_message(
        self,
        reset_adapter_state: dict[str, mock.MagicMock],
    ) -> None:
        adapter._handle_chat_event(
            {
                "state": "error",
                "runId": "run-2",
                "message": "",
                "errorMessage": "LLM timeout",
            }
        )

        call = reset_adapter_state["on_response"].call_args.args[0]
        assert call["state"] == "error"
        assert call["errorMessage"] == "LLM timeout"

    def test_empty_final_reuses_cached_delta(
        self,
        reset_adapter_state: dict[str, mock.MagicMock],
    ) -> None:
        adapter._handle_chat_event(
            {
                "state": "delta",
                "runId": "run-3",
                "message": {"content": [{"type": "text", "text": "cached answer"}]},
            }
        )
        adapter._handle_chat_event(
            {
                "state": "final",
                "runId": "run-3",
                "sessionKey": "sess-3",
                "message": {"content": []},
            }
        )

        forwarded = reset_adapter_state["on_response"].call_args.args[0]
        assert forwarded["state"] == "final"
        assert adapter._response_text(forwarded) == "cached answer"

    def test_terminal_tool_message_is_ignored(
        self,
        reset_adapter_state: dict[str, mock.MagicMock],
    ) -> None:
        adapter._handle_chat_event(
            {
                "state": "final",
                "runId": "run-4",
                "message": {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "type": "function",
                            "function": {"name": "exec", "arguments": "{}"},
                        }
                    ],
                },
            }
        )

        reset_adapter_state["on_response"].assert_not_called()

    def test_non_terminal_events_are_ignored(
        self,
        reset_adapter_state: dict[str, mock.MagicMock],
    ) -> None:
        adapter._handle_chat_event({"state": "in_progress", "runId": "run-5"})
        adapter._handle_chat_event({"state": "streaming", "runId": "run-6"})
        adapter._handle_chat_event({})

        reset_adapter_state["on_response"].assert_not_called()


class TestBackendResponse:
    def test_final_response_buffers_and_writes_turn_boundary(
        self,
        reset_adapter_state: dict[str, mock.MagicMock],
    ) -> None:
        response = {
            "state": "final",
            "run_id": "run-1",
            "message": {"content": [{"type": "text", "text": "Here are the results."}]},
        }

        adapter.on_backend_response(response)

        reset_adapter_state["buffer"].assert_called_once_with(response)
        reset_adapter_state["aui"].assert_called_once_with(
            "send_message_to_user",
            "Here are the results.",
        )

    def test_final_empty_text_still_emits_boundary(
        self,
        reset_adapter_state: dict[str, mock.MagicMock],
    ) -> None:
        response = {"state": "final", "run_id": "run-2", "message": ""}

        adapter.on_backend_response(response)

        reset_adapter_state["buffer"].assert_called_once_with(response)
        reset_adapter_state["aui"].assert_called_once_with("send_message_to_user", "")

    def test_error_with_text_emits_boundary(
        self,
        reset_adapter_state: dict[str, mock.MagicMock],
    ) -> None:
        response = {"state": "error", "run_id": "run-3", "message": "Something failed."}

        adapter.on_backend_response(response)

        reset_adapter_state["aui"].assert_called_once_with(
            "send_message_to_user",
            "Something failed.",
        )

    def test_error_with_error_message_emits_boundary(
        self,
        reset_adapter_state: dict[str, mock.MagicMock],
    ) -> None:
        response = {
            "state": "error",
            "run_id": "run-err",
            "message": "",
            "errorMessage": "Unknown error (no error details in response)",
        }

        adapter.on_backend_response(response)

        reset_adapter_state["buffer"].assert_called_once_with(response)
        reset_adapter_state["aui"].assert_called_once_with(
            "send_message_to_user",
            "Unknown error (no error details in response)",
        )

    def test_aborted_without_text_does_not_emit_boundary(
        self,
        reset_adapter_state: dict[str, mock.MagicMock],
    ) -> None:
        response = {"state": "aborted", "run_id": "run-4", "message": ""}

        adapter.on_backend_response(response)

        reset_adapter_state["buffer"].assert_called_once_with(response)
        reset_adapter_state["aui"].assert_not_called()

    @pytest.mark.parametrize("text", ["HEARTBEAT_OK", "NO_REPLY"])
    def test_internal_final_acks_are_dropped(
        self,
        text: str,
        reset_adapter_state: dict[str, mock.MagicMock],
    ) -> None:
        adapter.on_backend_response(
            {
                "state": "final",
                "run_id": "run-drop",
                "message": text,
            }
        )

        reset_adapter_state["buffer"].assert_not_called()
        reset_adapter_state["aui"].assert_not_called()


class TestNotifySent:
    def test_notify_sent_writes_agent_event(
        self,
        reset_adapter_state: dict[str, mock.MagicMock],
    ) -> None:
        adapter._on_notify_sent("Find the cheapest flight to Paris.")

        reset_adapter_state["aui"].assert_called_once_with(
            "send_message_to_agent",
            "Find the cheapest flight to Paris.",
        )


class TestWriteAuiEventSimT:
    _real_write_aui = staticmethod(gaia2_adapter_base.write_aui_event)

    def test_includes_sim_t_when_faketime_exists(self, tmp_path) -> None:
        import builtins
        import json

        events_path = tmp_path / "events.jsonl"
        events_path.touch()
        faketime_path = tmp_path / "faketime.rc"
        faketime_path.write_text("2024-10-15 07:02:30\n")

        real_open = builtins.open

        def patched_open(path, *args, **kwargs):
            if str(path) == "/tmp/faketime.rc":
                return real_open(str(faketime_path), *args, **kwargs)
            return real_open(path, *args, **kwargs)

        with (
            mock.patch.object(gaia2_adapter_base, "EVENTS_JSONL", str(events_path)),
            mock.patch("builtins.open", side_effect=patched_open),
        ):
            self._real_write_aui("send_message_to_user", "Done!")

        event = json.loads(events_path.read_text().strip())
        assert event["sim_t"] == "2024-10-15 07:02:30"
        assert event["fn"] == "send_message_to_user"

    def test_no_sim_t_without_faketime(self, tmp_path) -> None:
        import json

        events_path = tmp_path / "events.jsonl"
        events_path.touch()

        with mock.patch.object(gaia2_adapter_base, "EVENTS_JSONL", str(events_path)):
            self._real_write_aui("send_message_to_agent", "Hello")

        event = json.loads(events_path.read_text().strip())
        assert "sim_t" not in event
        assert event["fn"] == "send_message_to_agent"
