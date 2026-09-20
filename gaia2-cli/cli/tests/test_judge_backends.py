# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
"""Contract and integration tests for interchangeable semantic backends."""

from __future__ import annotations

import io
import json
import sys
from types import SimpleNamespace
from urllib.error import HTTPError, URLError

import pytest
from gaia2_cli.judge import Judge, RateLimitError, create_checker_factory
from gaia2_cli.judge.typesafe import TypeSafeCheckerFactory
from gaia2_core.judge.checkers import build_llm_checkers, build_soft_checkers
from gaia2_core.judge.prompts import LLMFunctionTemplates
from gaia2_core.types import CompletedEvent, EventAction, OracleEvent

TEMPLATES = LLMFunctionTemplates(
    system_prompt_template="Compare expected and actual content. Return [[Success]] or [[Failure]].",
    user_prompt_template="Expected: {{expected}}; actual: {{actual}}",
    assistant_prompt_template="{{verdict}}",
    examples=[
        {
            "input": {"expected": "one", "actual": "1"},
            "output": {"verdict": "[[Success]]"},
        }
    ],
)


def reply(probability=0.9):
    return {
        "model": "jev-test-version",
        "answers": {"verdict": {"type": "noul", "noul": probability}},
    }


def mock_api(monkeypatch, response):
    calls = []

    def urlopen(request, timeout):
        calls.append((request, timeout))
        return io.BytesIO(json.dumps(response).encode())

    monkeypatch.setattr("gaia2_cli.judge.typesafe.urlopen", urlopen)
    return calls


def factory(**kwargs):
    return TypeSafeCheckerFactory(model="jev-test", api_key="test-secret", **kwargs)


@pytest.mark.parametrize(
    "probability, expected", [(0, False), (0.49, False), (0.5, True), (1, True)]
)
def test_typed_verdict_and_audit(monkeypatch, tmp_path, probability, expected):
    calls = mock_api(monkeypatch, reply(probability))
    path = tmp_path / "judge_decisions.jsonl"
    checker = factory(audit_path=path)(TEMPLATES, 1, "[[Success]]", "[[Failure]]")
    assert (
        checker({"expected": "one", "actual": "1", "full_task": "Count the files"})
        is expected
    )
    request, timeout = calls[0]
    body = json.loads(request.data)
    assert request.full_url == "https://api.typesafe.ai/v1/systemone"
    assert request.get_header("Authorization") == "Bearer test-secret"
    assert timeout == 30
    assert body["state"]["full_task"] == "Count the files"
    assert len(body["state"]["messages"]) == 4
    assert body["state"]["messages"][-1]["content"] == "Expected: one; actual: 1"
    assert body["questions"]["verdict"]["type"] == "noul"
    assert "[[Success]]" in body["questions"]["verdict"]["criteria"]["true"]
    record = json.loads(path.read_text())
    assert record["model"] == "jev-test-version"
    assert record["probability"] == probability
    assert record["passed"] is expected
    assert len(record["input_sha256"]) == 64
    assert "test-secret" not in path.read_text()
    assert json.loads(checker.last_response) == [record]


@pytest.mark.parametrize(
    "value", [None, True, "0.9", -1, 2, float("nan"), float("inf")]
)
def test_invalid_probabilities_are_errors(monkeypatch, value):
    mock_api(monkeypatch, reply(value))
    with pytest.raises(RuntimeError, match="invalid verdict"):
        factory()(TEMPLATES, 1, "yes", "no")({})


@pytest.mark.parametrize(
    "response",
    [
        {},
        [],
        {"answers": {}},
        {"model": "x", "answers": {"verdict": {"type": "choice", "noul": 0.9}}},
    ],
)
def test_malformed_responses_are_errors(monkeypatch, response):
    mock_api(monkeypatch, response)
    with pytest.raises(RuntimeError, match="invalid verdict"):
        factory()(TEMPLATES, 1, "yes", "no")({})


@pytest.mark.parametrize(
    "error, kind",
    [
        (HTTPError("url", 429, "test-secret", {}, None), RateLimitError),
        (HTTPError("url", 401, "test-secret", {}, None), RuntimeError),
        (URLError("test-secret"), RuntimeError),
        (TimeoutError("test-secret"), RuntimeError),
    ],
)
def test_transport_errors_do_not_become_votes_or_leak_keys(monkeypatch, error, kind):
    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr("gaia2_cli.judge.typesafe.urlopen", fail)
    with pytest.raises(kind) as caught:
        factory()(TEMPLATES, 1, "yes", "no")({})
    assert "test-secret" not in str(caught.value)


@pytest.mark.parametrize(
    "option, value",
    [
        ("threshold", 0),
        ("threshold", 1),
        ("threshold", float("nan")),
        ("threshold", True),
        ("threshold", "0.5"),
        ("timeout", "30"),
        ("timeout", 0),
        ("timeout", float("inf")),
    ],
)
def test_invalid_options(option, value):
    with pytest.raises(ValueError):
        factory(**{option: value})


def test_provider_selection_is_lazy_and_uses_existing_key(monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "existing-key")
    monkeypatch.setitem(sys.modules, "litellm", None)
    backend = create_checker_factory(
        "jev-test", "typesafe", extra_body={"threshold": 0.8}
    )
    assert isinstance(backend, TypeSafeCheckerFactory)
    assert backend.api_key == "existing-key"
    assert backend.threshold == 0.8
    with pytest.raises(ValueError, match="Unknown TypeSafe"):
        create_checker_factory("jev-test", "typesafe", extra_body={"typo": 1})
    monkeypatch.delenv("TYPESAFE_API_KEY")
    with pytest.raises(ValueError, match="TYPESAFE_API_KEY"):
        create_checker_factory("jev-test", "typesafe")


def test_existing_provider_keeps_litellm_request_and_verdict(monkeypatch):
    calls = []

    def completion(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="[[Success]]"))]
        )

    monkeypatch.setitem(sys.modules, "litellm", SimpleNamespace(completion=completion))
    backend = create_checker_factory(
        "existing",
        "openai",
        api_key="test-secret",
        extra_body={"reasoning_effort": "low"},
    )
    assert (
        backend(TEMPLATES, 1, "[[Success]]", "[[Failure]]")(
            {"expected": "one", "actual": "1"}
        )
        is True
    )
    assert calls[0]["model"] == "existing"
    assert calls[0]["extra_body"] == {"reasoning_effort": "low"}
    assert calls[0]["temperature"] == 0


def test_rubric_overrides_votes_and_all_checkers_are_preserved():
    captured = []

    def backend(templates, votes, success, failure):
        captured.append((templates, votes, success, failure))
        return SimpleNamespace(last_response=None)

    checkers = build_soft_checkers(
        backend, num_votes=3, prompt_template_overrides={"email_checker": TEMPLATES}
    )
    assert len(checkers) == 9
    assert (TEMPLATES, 3, "[[Success]]", "[[Failure]]") in captured
    assert sum(item[1] == 1 for item in captured) == 3
    with pytest.raises(KeyError):
        build_soft_checkers(backend, prompt_template_overrides={"unknown": TEMPLATES})
    legacy = build_llm_checkers(lambda messages: ("[[TRUE]]", {}))
    assert (
        legacy["signature_checker"]({"agent_action_call": "Hi", "user_name": "Alice"})
        is True
    )


def judge_and_event(backend, recipient="correct@example.org"):
    oracle_args = {
        "recipients": ["correct@example.org"],
        "content": "The meeting is at 3 PM.",
        "subject": "Meeting",
        "cc": [],
        "attachment_paths": [],
    }
    agent_args = {
        **oracle_args,
        "recipients": [recipient],
        "content": "We meet at 15:00.",
    }
    oracle = OracleEvent(
        event_id="o1",
        event_type="AGENT",
        event_time=100,
        action=EventAction(
            app_name="EmailClientApp",
            class_name="EmailClientApp",
            function_name="send_email",
            operation_type="write",
            args=oracle_args,
        ),
        args=oracle_args,
    )
    agent = CompletedEvent(
        event_id="a1",
        event_type="AGENT",
        event_time=100,
        action=EventAction(
            app_name="EmailClientApp",
            class_name="EmailClientApp",
            function_name="send_email",
            operation_type="write",
            args=agent_args,
        ),
    )
    judge = Judge(
        [[oracle]],
        [{"o1": []}],
        ["Tell Alice the meeting is at 3 PM."],
        checker_factory=backend,
    )
    return judge, agent


def test_real_judge_preserves_hard_checks_and_uses_jev_for_semantics(monkeypatch):
    calls = mock_api(monkeypatch, reply())
    judge, agent = judge_and_event(factory(), "wrong@example.org")
    assert judge.judge_turn(0, [agent]).success is False
    assert calls == []
    judge, agent = judge_and_event(factory())
    assert judge.judge_turn(0, [agent]).success is True
    assert len(calls) == 2  # signature and email; placeholder check remains Python
    assert all(json.loads(call[0].data)["state"]["full_task"] for call in calls)
    mock_api(monkeypatch, reply(0.1))
    judge, agent = judge_and_event(factory())
    assert judge.judge_turn(0, [agent]).success is False


def test_backend_failure_never_degrades_to_hard_only_success(monkeypatch):
    def broken(*args):
        raise RuntimeError("backend unavailable")

    with pytest.raises(RuntimeError, match="backend unavailable"):
        judge_and_event(broken)
    mock_api(monkeypatch, {})
    judge, agent = judge_and_event(factory())
    with pytest.raises(RuntimeError, match="invalid verdict"):
        judge.judge_turn(0, [agent])
    with pytest.raises(ValueError, match="either engine"):
        Judge([], [], [], engine=lambda _: ("ok", {}), checker_factory=factory())


def test_real_http_transport(monkeypatch):
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from threading import Thread

    received = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            received.append(
                (
                    self.path,
                    self.headers["Authorization"],
                    json.loads(self.rfile.read(int(self.headers["Content-Length"]))),
                )
            )
            body = json.dumps(reply()).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    monkeypatch.setenv("NO_PROXY", "127.0.0.1")
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        checker = factory(base_url=f"http://127.0.0.1:{server.server_port}/v1")(
            TEMPLATES, 1, "yes", "no"
        )
        assert checker({"expected": "one", "actual": "1"}) is True
        assert received[0][0] == "/v1/systemone"
        assert received[0][1] == "Bearer test-secret"
        assert received[0][2]["model"] == "jev-test"
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
