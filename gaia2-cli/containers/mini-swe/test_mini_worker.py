# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
"""Unit tests for the gaia2-mini-swe worker.

Run with:
    cd gaia2-cli
    PYTHONPATH=. python3 -m pytest containers/mini-swe/test_mini_worker.py -v

Tests requiring the real mini-swe-agent (``minisweagent.exceptions``) are
skipped if the package isn't installed in the test interpreter.
"""

from __future__ import annotations

import io
import os
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

# Importable without minisweagent because the from-imports live inside
# ``_build_mini_agent`` (deferred).
import mini_worker as worker  # noqa: E402

# ─────────────────────────────────────────────────────────────────────
#  _run_task: maps DefaultAgent.run() output → (state, message)
# ─────────────────────────────────────────────────────────────────────


def _stub_agent(submission: str, exit_status: str = "Submitted") -> MagicMock:
    """Build a stub agent whose .run() returns the given submission dict."""
    agent = MagicMock()
    agent.run.return_value = {
        "submission": submission,
        "exit_status": exit_status,
    }
    return agent


def test_run_task_final_state_returns_submission():
    agent = _stub_agent("Lianying")
    state, msg = worker._run_task(agent, "What is the oldest name?", fresh=True)
    assert state == "final"
    assert msg == "Lianying"
    agent.run.assert_called_once_with("What is the oldest name?")


def test_run_task_strips_whitespace_in_submission():
    agent = _stub_agent("  Lianying  \n")
    state, msg = worker._run_task(agent, "task", fresh=True)
    assert state == "final"
    assert msg == "Lianying"


def test_run_task_empty_submission_is_error():
    agent = _stub_agent("", exit_status="LimitsExceeded")
    state, msg = worker._run_task(agent, "task", fresh=True)
    assert state == "error"
    assert "LimitsExceeded" in msg
    assert "empty submission" in msg


def test_run_task_propagates_exceptions_as_error_state():
    agent = MagicMock()
    agent.run.side_effect = RuntimeError("connection refused")
    state, msg = worker._run_task(agent, "task", fresh=True)
    assert state == "error"
    assert "RuntimeError" in msg
    assert "connection refused" in msg


def test_run_task_fresh_false_uses_resume_path(monkeypatch):
    """fresh=False must NOT call agent.run; must call _resume_agent instead."""
    agent = MagicMock()
    agent.run.side_effect = AssertionError(
        "agent.run should not be called when fresh=False"
    )

    fake_result = {"submission": "Resumed reply", "exit_status": "Submitted"}
    captured = {}

    def fake_resume(a, notification_text):
        captured["agent"] = a
        captured["text"] = notification_text
        return fake_result

    monkeypatch.setattr(worker, "_resume_agent", fake_resume)
    state, msg = worker._run_task(agent, "[Notifications]", fresh=False)
    assert state == "final"
    assert msg == "Resumed reply"
    assert captured["text"] == "[Notifications]"
    assert captured["agent"] is agent


# ─────────────────────────────────────────────────────────────────────
#  _resume_agent: preserves history across notifies + heals dangling
#  tool_use blocks left by the COMPLETE_TASK submission marker
# ─────────────────────────────────────────────────────────────────────


def _resume_agent_or_skip():
    """Skip the test if minisweagent isn't installed."""
    pytest.importorskip("minisweagent.exceptions")


def _make_fake_agent_with_history(history: list[dict]):
    """A fake DefaultAgent stub the resume loop can drive.

    Tracks add_messages / step calls; step() injects an "exit" message
    after one iteration so the loop terminates.
    """
    agent = SimpleNamespace()
    agent.messages = list(history)
    agent.config = SimpleNamespace(output_path=None)

    def add_messages(*msgs):
        agent.messages.extend(msgs)
        return list(msgs)

    def get_template_vars(**_):
        return {}

    def step():
        # Emit an "exit" message so the resume loop terminates after one step.
        agent.messages.append(
            {"role": "exit", "content": "ok", "extra": {"submission": "Resumed-ok"}}
        )

    def handle_uncaught_exception(_):
        agent.messages.append({"role": "exit", "content": "error", "extra": {}})

    def save(_):
        pass

    agent.add_messages = add_messages
    agent.get_template_vars = get_template_vars
    agent.step = step
    agent.handle_uncaught_exception = handle_uncaught_exception
    agent.save = save

    # Stub model with the format_message / format_observation_messages API
    model = SimpleNamespace()
    model.format_message = lambda *, role, content, extra=None: {
        "role": role,
        "content": content,
        **({"extra": extra} if extra else {}),
    }
    model.format_observation_messages = lambda msg, outputs, tvars: [
        {"role": "tool", "content": str(outputs)}
    ]
    agent.model = model
    return agent


def test_resume_agent_pops_trailing_exit_and_appends_user_msg():
    _resume_agent_or_skip()
    history = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "first task"},
        {"role": "assistant", "content": "doing it"},
        {"role": "exit", "content": "Done", "extra": {"submission": "Done"}},
    ]
    agent = _make_fake_agent_with_history(history)
    result = worker._resume_agent(agent, "follow-up notification")
    # Old exit message gone
    roles = [m.get("role") for m in agent.messages]
    assert roles.count("exit") == 1, (
        f"expected exactly the new exit only, got roles={roles}"
    )
    # The follow-up user message is in the history
    user_msgs = [m for m in agent.messages if m.get("role") == "user"]
    assert any(m.get("content") == "follow-up notification" for m in user_msgs)
    # System / first user / earlier assistant are preserved
    assert agent.messages[0]["role"] == "system"
    assert agent.messages[1]["content"] == "first task"
    assert any(m.get("role") == "assistant" for m in agent.messages)
    # Returns the new exit's extra
    assert result.get("submission") == "Resumed-ok"


def test_resume_agent_synthesises_tool_result_for_dangling_tool_use():
    _resume_agent_or_skip()
    # Simulates the state after a Submitted exception: assistant emitted a
    # tool_call (the COMPLETE_TASK echo) but no tool_result was appended
    # before the exit message landed.
    dangling_assistant = {
        "role": "assistant",
        "tool_calls": [{"id": "toolu_abc", "function": {"name": "bash"}}],
    }
    history = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "first task"},
        dangling_assistant,
        {"role": "exit", "content": "Done", "extra": {"submission": "Done"}},
    ]
    agent = _make_fake_agent_with_history(history)
    worker._resume_agent(agent, "notify-2")
    # After resume: must have the placeholder tool observation between the
    # dangling assistant tool_use and the new user message.
    msgs = agent.messages
    assistant_idx = next(i for i, m in enumerate(msgs) if m is dangling_assistant)
    after = msgs[assistant_idx + 1 :]
    # Next message should be the synthetic tool_result, before the user notify.
    assert after[0]["role"] == "tool"
    # Then the new user notification.
    user_msgs = [m for m in after if m.get("role") == "user"]
    assert any(m.get("content") == "notify-2" for m in user_msgs)


def test_resume_agent_no_tool_use_skips_synthesis():
    _resume_agent_or_skip()
    # Last assistant message has NO tool_calls — no synthesis should happen.
    history = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "task"},
        {"role": "assistant", "content": "plain reply"},
        {"role": "exit", "content": "Done", "extra": {"submission": "Done"}},
    ]
    agent = _make_fake_agent_with_history(history)
    worker._resume_agent(agent, "notify")
    # The "tool" role should not appear at all (no synthesised observation).
    assert not any(m.get("role") == "tool" for m in agent.messages)


# ─────────────────────────────────────────────────────────────────────
#  Socket protocol: _send and _recv_lines (no minisweagent dep)
# ─────────────────────────────────────────────────────────────────────


class _FakeSock:
    """Just enough of the socket.socket API for _send + _recv_lines."""

    def __init__(self, recv_data: bytes = b"") -> None:
        self._sendbuf = io.BytesIO()
        self._recvbuf = io.BytesIO(recv_data)

    def sendall(self, data: bytes) -> None:
        self._sendbuf.write(data)

    def recv(self, n: int) -> bytes:
        return self._recvbuf.read(n)

    def sent(self) -> bytes:
        return self._sendbuf.getvalue()


def test_send_writes_one_json_line_terminated_by_newline():
    sock = _FakeSock()
    worker._send(sock, {"type": "ready"})
    payload = sock.sent()
    assert payload.endswith(b"\n")
    assert b'"type": "ready"' in payload or b'"type":"ready"' in payload


def test_recv_lines_parses_multiple_json_objects_in_one_buffer():
    payload = b'{"type":"message","text":"hi","run_id":"r1"}\n{"type":"interrupt","text":"x"}\n'
    sock = _FakeSock(recv_data=payload)
    out = list(worker._recv_lines(sock))
    assert len(out) == 2
    assert out[0]["type"] == "message"
    assert out[0]["run_id"] == "r1"
    assert out[1]["type"] == "interrupt"


# ─────────────────────────────────────────────────────────────────────
#  Trace logging: trace.jsonl writer + LitellmModel wrapper
# ─────────────────────────────────────────────────────────────────────


def _read_trace(path) -> list[dict]:
    with open(path) as f:
        return [__import__("json").loads(line) for line in f if line.strip()]


def test_install_trace_logging_wraps_query_and_writes_entry(monkeypatch, tmp_path):
    """Successful query → one trace.jsonl line with status 200, request, response."""
    trace_file = tmp_path / "trace.jsonl"
    monkeypatch.setattr(worker, "TRACE_FILE", str(trace_file))
    monkeypatch.setattr(worker, "_trace_seq", 0)

    fake_response = MagicMock()
    fake_response.model_dump_json.return_value = (
        '{"choices":[{"message":{"content":"hi"}}]}'
    )

    model = MagicMock()
    model._query = MagicMock(return_value=fake_response)
    # Reset the install sentinel so the wrap actually takes effect.
    if hasattr(model, "_gaia2_trace_logging_installed"):
        del model._gaia2_trace_logging_installed

    worker._install_trace_logging(
        model, "https://example.com/v1", "anthropic/claude-test"
    )
    # Now calling _query should write a trace entry.
    out = model._query([{"role": "user", "content": "hi"}])
    assert out is fake_response

    entries = _read_trace(trace_file)
    assert len(entries) == 1
    e = entries[0]
    assert e["seq"] == 1
    assert e["type"] == "llm_call"
    assert e["http_status"] == 200
    assert "request" in e and "raw_response" in e
    assert "/chat/completions" in e["url"]


def test_install_trace_logging_records_errors_with_non_200_status(
    monkeypatch, tmp_path
):
    """Failed query → trace entry with non-200 status + error body, exception re-raised."""
    trace_file = tmp_path / "trace.jsonl"
    monkeypatch.setattr(worker, "TRACE_FILE", str(trace_file))
    monkeypatch.setattr(worker, "_trace_seq", 0)

    def fail(_messages, **_kwargs):
        err = RuntimeError("upstream rejected")
        err.status_code = 503
        raise err

    model = MagicMock()
    model._query = fail
    if hasattr(model, "_gaia2_trace_logging_installed"):
        del model._gaia2_trace_logging_installed

    worker._install_trace_logging(model, "", "kimi-2.6")

    with pytest.raises(RuntimeError, match="upstream rejected"):
        model._query([{"role": "user", "content": "x"}])

    entries = _read_trace(trace_file)
    assert len(entries) == 1
    e = entries[0]
    assert e["http_status"] == 503
    assert "upstream rejected" in e["raw_response"]


def test_install_trace_logging_is_idempotent(monkeypatch, tmp_path):
    """Calling install twice must not double-wrap (would record each call twice)."""
    trace_file = tmp_path / "trace.jsonl"
    monkeypatch.setattr(worker, "TRACE_FILE", str(trace_file))
    monkeypatch.setattr(worker, "_trace_seq", 0)

    fake_response = MagicMock()
    fake_response.model_dump_json.return_value = "{}"
    model = MagicMock()
    model._query = MagicMock(return_value=fake_response)
    if hasattr(model, "_gaia2_trace_logging_installed"):
        del model._gaia2_trace_logging_installed

    worker._install_trace_logging(model, "", "test-model")
    worker._install_trace_logging(model, "", "test-model")  # 2nd install is a no-op
    model._query([{"role": "user", "content": "x"}])

    assert len(_read_trace(trace_file)) == 1


def test_install_trace_logging_skipped_when_TRACE_FILE_empty(monkeypatch, tmp_path):
    """If GAIA2_TRACE_FILE is unset, the wrapper does nothing."""
    monkeypatch.setattr(worker, "TRACE_FILE", "")
    model = MagicMock()
    orig_query = model._query
    worker._install_trace_logging(model, "https://x", "m")
    # _query must NOT have been replaced
    assert model._query is orig_query


def test_recv_lines_skips_blank_and_unparseable_lines():
    payload = b'\n{"type":"ready"}\nnot-json\n\n{"type":"message","text":"ok"}\n'
    sock = _FakeSock(recv_data=payload)
    out = list(worker._recv_lines(sock))
    # 2 valid messages, blanks and unparseable line dropped
    assert len(out) == 2
    assert out[0]["type"] == "ready"
    assert out[1]["type"] == "message"
