# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
"""Judge-only task text as seen by its consumers: loader, judge, gaia2-eventd.

An audio-prompt scenario sends the agent a neutral carrier string; the
original prompt sits in a judge context (``gaia2_core.judge_context``) that
only the judge may read.  These tests import nothing from that module at
collection time, so against a build without judge-context support each one
fails on its own assertion (the judge grading the carrier, a bad file not
being refused, ...).  ``JudgeContextError`` is a ``ValueError``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

SECRET = "Find the txt file about Copa America 2024 in my news folder."


def _write(path: Path, data: object, mode: int = 0o600) -> Path:
    path.write_text(data if isinstance(data, str) else json.dumps(data))
    os.chmod(path, mode)
    return path


def _context(**turns: str) -> dict:
    return {"version": 1, "turns": {k: {"text": v} for k, v in turns.items()}}


# ---------------------------------------------------------------------------
# Scenario fixtures shared by the loader, judge and daemon tests below
# ---------------------------------------------------------------------------

CARRIER = "[User request provided as audio]"
FOLLOW_UP = "Now forward it to Bob."


def _arg(name: str, value: str, value_type: str = "str") -> dict:
    return {"name": name, "value": value, "value_type": value_type}


def _user_event(event_id: str, content: str, deps: list[str]) -> dict:
    return {
        "event_id": event_id,
        "event_type": "USER",
        "event_time": None,
        "event_relative_time": None,
        "dependencies": deps,
        "action": {
            "app": "AgentUserInterface",
            "class_name": "AgentUserInterface",
            "function": "send_message_to_agent",
            "operation_type": "READ",
            "args": [_arg("content", content), _arg("attachments", "[]", "list")],
        },
    }


def _smu_oracle(event_id: str, content: str, deps: list[str]) -> dict:
    return {
        "event_id": event_id,
        "event_type": "AGENT",
        "event_time": None,
        "event_relative_time": None,
        "dependencies": deps,
        "action": {
            "app": "AgentUserInterface",
            "class_name": "AgentUserInterface",
            "function": "send_message_to_user",
            "operation_type": "WRITE",
            "args": [_arg("content", content)],
        },
    }


def _write_scenario(tmp_path: Path, *, multi_turn: bool = False) -> Path:
    """An audio-prompt scenario: the first USER turn carries only the carrier."""
    events = [
        _user_event("Event-USER-1", CARRIER, []),
        _smu_oracle(
            "OracleEvent-AGENT-1", "The file is copa_america.txt.", ["Event-USER-1"]
        ),
    ]
    if multi_turn:
        events += [
            _user_event("Event-USER-2", FOLLOW_UP, ["OracleEvent-AGENT-1"]),
            _smu_oracle("OracleEvent-AGENT-2", "Forwarded to Bob.", ["Event-USER-2"]),
        ]
    data = {
        "metadata": {"definition": {"scenario_id": "audio_s1", "start_time": 0}},
        "apps": [{"name": "AgentUserInterface", "class_name": "AgentUserInterface"}],
        "events": events,
    }
    path = tmp_path / "audio_s1.json"
    path.write_text(json.dumps(data))
    return path


def _oracle_tasks(loader) -> list[str]:
    from gaia2_core.event_loop import EventProcessor

    processor = EventProcessor.from_loader(loader)
    processor.build_event_id_to_turn_idx()
    return loader.extract_oracle_data(
        processor.event_id_to_turn_idx, processor.nb_turns
    )[2]


# ---------------------------------------------------------------------------
# ScenarioLoader: the judge's task text
# ---------------------------------------------------------------------------


class TestLoaderJudgeTexts:
    def test_sidecar_text_replaces_the_carrier_for_the_judge_only(
        self, tmp_path: Path
    ) -> None:
        from gaia2_core.loader import ScenarioLoader

        scenario = _write_scenario(tmp_path, multi_turn=True)
        _write(
            tmp_path / "audio_s1.judge_context.json",
            _context(**{"Event-USER-1": SECRET}),
        )
        loader = ScenarioLoader(str(scenario))

        # The judge grades turn 0 against the original prompt; turn 1, not
        # listed in the judge context, keeps its own content.
        assert _oracle_tasks(loader) == [SECRET + "\n", FOLLOW_UP + "\n"]
        # What the agent is sent is untouched.
        assert loader.task == CARRIER
        first = next(ev for ev in loader.events if ev.event_id == "Event-USER-1")
        assert first.action.args["content"] == CARRIER

    def test_without_a_judge_context_the_judge_inputs_are_unchanged(
        self, tmp_path: Path
    ) -> None:
        from gaia2_core.loader import ScenarioLoader

        # Passes on builds without judge-context support too: it pins that
        # nothing changes for scenarios that have no judge context.
        scenario = _write_scenario(tmp_path, multi_turn=True)
        loader = ScenarioLoader(str(scenario))
        assert _oracle_tasks(loader) == [CARRIER + "\n", FOLLOW_UP + "\n"]
        assert getattr(loader, "judge_texts", {}) == {}

    def test_set_judge_context_replaces_or_clears_the_sidecar_text(
        self, tmp_path: Path
    ) -> None:
        from gaia2_core.loader import ScenarioLoader

        scenario = _write_scenario(tmp_path)
        _write(
            tmp_path / "audio_s1.judge_context.json",
            _context(**{"Event-USER-1": SECRET}),
        )
        loader = ScenarioLoader(str(scenario))
        assert _oracle_tasks(loader) == [SECRET + "\n"]

        loader.set_judge_context({"Event-USER-1": "Original."})
        assert _oracle_tasks(loader) == ["Original.\n"]
        loader.set_judge_context({})
        assert _oracle_tasks(loader) == [CARRIER + "\n"]

    @pytest.mark.parametrize(
        "event_id", ["Event-USER-typo", "OracleEvent-AGENT-1", "Event-ENV-9"]
    )
    def test_ids_that_are_not_user_turns_fail_loudly(
        self, tmp_path: Path, event_id: str
    ) -> None:
        from gaia2_core.loader import ScenarioLoader

        scenario = _write_scenario(tmp_path)
        _write(
            tmp_path / "audio_s1.judge_context.json",
            _context(**{event_id: SECRET}),
        )
        with pytest.raises(ValueError, match=event_id) as excinfo:
            ScenarioLoader(str(scenario))
        assert SECRET not in str(excinfo.value)

    def test_malformed_sidecar_fails_loudly(self, tmp_path: Path) -> None:
        from gaia2_core.loader import ScenarioLoader

        scenario = _write_scenario(tmp_path)
        _write(tmp_path / "audio_s1.judge_context.json", {"version": 2, "turns": {}})
        with pytest.raises(ValueError, match="unsupported version 2"):
            ScenarioLoader(str(scenario))


# ---------------------------------------------------------------------------
# Judge: the soft checkers receive the original text
# ---------------------------------------------------------------------------


def _agent_smu(event_id: str, content: str, t: float):
    from gaia2_core.types import CompletedEvent, EventAction

    return CompletedEvent(
        event_id=event_id,
        event_type="AGENT",
        event_time=t,
        action=EventAction(
            app_name="AgentUserInterface",
            class_name="AgentUserInterface",
            function_name="send_message_to_user",
            args={"content": content},
            operation_type="write",
        ),
    )


class TestJudgeSeesOriginalText:
    def test_judge_turn_passes_the_original_text_to_the_checkers(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import gaia2_core.judge.checkers as checkers
        from gaia2_core.event_loop import EventProcessor
        from gaia2_core.judge import Judge
        from gaia2_core.loader import ScenarioLoader

        scenario = _write_scenario(tmp_path)
        _write(
            tmp_path / "audio_s1.judge_context.json",
            _context(**{"Event-USER-1": SECRET}),
        )
        loader = ScenarioLoader(str(scenario))
        processor = EventProcessor.from_loader(loader)
        processor.build_event_id_to_turn_idx()
        oracle_events, graph, tasks, _ = loader.extract_oracle_data(
            processor.event_id_to_turn_idx, processor.nb_turns
        )

        seen: list[list[str] | None] = []
        real_mild_compare = checkers.mild_compare

        def spy(**kwargs):
            seen.append(kwargs.get("tasks"))
            return real_mild_compare(**kwargs)

        monkeypatch.setattr(checkers, "mild_compare", spy)
        judge = Judge(
            turn_to_oracle_events=oracle_events,
            turn_to_oracle_graph=graph,
            tasks=tasks,
            app_name_to_class=loader.app_name_to_class,
        )
        result = judge.judge_turn(0, [_agent_smu("AGENT-1", "copa_america.txt", 1.0)])

        assert result.success
        assert seen == [[SECRET + "\n"]]

    def test_subtask_extraction_prompt_gets_the_original_text(
        self, tmp_path: Path
    ) -> None:
        # soft_compare is where the task text meets an LLM prompt (the
        # subtask extractor). Drive it with the loader's tasks and an engine.
        from gaia2_core.judge.checkers import soft_compare
        from gaia2_core.judge.config import SoftCheckerType
        from gaia2_core.loader import ScenarioLoader

        scenario = _write_scenario(tmp_path)
        _write(
            tmp_path / "audio_s1.judge_context.json",
            _context(**{"Event-USER-1": SECRET}),
        )
        tasks = _oracle_tasks(ScenarioLoader(str(scenario)))

        prompts: list[str] = []

        def engine(messages, **_kwargs):
            prompts.append(messages[-1]["content"])
            return "<subtask>Tell the user the file name.</subtask>", {}

        class AcceptingChecker:
            last_response = "[[Success]]"

            def __call__(self, user_prompt_args):
                return True

        soft_compare(
            agent_args={"content": "copa_america.txt"},
            oracle_args={"content": "The file is copa_america.txt."},
            tool_name="AgentUserInterface__send_message_to_user",
            soft_checker_types=[SoftCheckerType.user_message_checker],
            llm_checkers={"user_message_checker": AcceptingChecker()},
            tasks=tasks,
            engine=engine,
        )
        assert len(prompts) == 1
        assert SECRET in prompts[0]
        assert CARRIER not in prompts[0]


# ---------------------------------------------------------------------------
# gaia2-eventd: reads the private copy, fails loudly, never leaks the text
# ---------------------------------------------------------------------------


@pytest.fixture
def daemon_factory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Build a Gaia2EventDaemon over an audio-prompt scenario (no setup())."""
    import logging

    from gaia2_cli.daemon import eventd

    monkeypatch.setenv("GAIA2_JUDGE_FINAL_TURN", "1")
    monkeypatch.setattr("gaia2_cli.daemon.eventd.time.sleep", lambda *_: None)
    eventd_logger = logging.getLogger(eventd.__name__)
    level = eventd_logger.level
    handlers = list(eventd_logger.handlers)

    def make(*, multi_turn: bool = False, **kwargs):
        state_dir = tmp_path / "state"
        state_dir.mkdir(exist_ok=True)
        (state_dir / "events.jsonl").touch()
        scenario = _write_scenario(tmp_path, multi_turn=multi_turn)
        return eventd.Gaia2EventDaemon(
            scenario_path=str(scenario),
            state_dir=str(state_dir),
            poll_interval=0.01,
            idle_timeout=5.0,
            log_path=str(state_dir / "eventd.log"),
            **kwargs,
        )

    yield make
    for handler in eventd_logger.handlers[:]:
        if handler not in handlers:
            eventd_logger.removeHandler(handler)
            handler.close()
    eventd_logger.setLevel(level)


def _private_context(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, **turns) -> Path:
    """Write a 0600 judge context outside the state dir; point the daemon at it."""
    private = tmp_path / "private"
    private.mkdir(exist_ok=True)
    path = _write(private / "judge_context.json", _context(**turns))
    monkeypatch.setenv("GAIA2_JUDGE_CONTEXT", str(path))
    return path


def _status(daemon) -> dict:
    return json.loads((daemon.state_dir / "daemon_status.json").read_text())


class TestDaemonJudgeContext:
    def test_judge_grades_against_the_private_text(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, daemon_factory
    ) -> None:
        _private_context(tmp_path, monkeypatch, **{"Event-USER-1": SECRET})
        daemon = daemon_factory(multi_turn=True)
        daemon.setup()

        assert daemon._processor._judge.tasks == [SECRET + "\n", FOLLOW_UP + "\n"]
        # The agent is still sent the carrier.
        assert daemon._task == CARRIER
        # daemon_status.json (served to the runner) says it was applied, by
        # count only.
        status = _status(daemon)
        assert status["judge_context_turns"] == 1
        assert SECRET not in json.dumps(status)

    def test_without_a_judge_context_nothing_changes(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, daemon_factory
    ) -> None:
        # Passes on builds without judge-context support too: the judge's
        # inputs must be exactly what the scenario alone gives.
        from gaia2_core.loader import ScenarioLoader

        monkeypatch.setenv("GAIA2_JUDGE_CONTEXT", str(tmp_path / "absent.json"))
        daemon = daemon_factory(multi_turn=True)
        daemon.setup()

        judge = daemon._processor._judge
        loader = ScenarioLoader(str(daemon.scenario_path))
        oracle_events, graph, tasks, _ = loader.extract_oracle_data(
            daemon._processor.event_id_to_turn_idx, daemon._processor.nb_turns
        )
        assert judge.tasks == tasks == [CARRIER + "\n", FOLLOW_UP + "\n"]
        assert [[e.model_dump() for e in t] for t in judge.turn_to_oracle_events] == [
            [e.model_dump() for e in t] for t in oracle_events
        ]
        assert judge.turn_to_oracle_graph == graph
        assert set(_status(daemon)) == {"status", "turn", "nb_turns", "num_events"}
        assert "judge context" not in (daemon.state_dir / "eventd.log").read_text()

    @pytest.mark.parametrize(
        "content, mode, expected",
        [
            (_context(**{"Event-USER-1": SECRET}), 0o640, "must be 0600"),
            (_context(**{"Event-USER-1": SECRET}), 0o644, "must be 0600"),
            (
                {"version": 2, "turns": {"Event-USER-1": {"text": SECRET}}},
                0o600,
                "unsupported version 2",
            ),
            (_context(**{"Event-USER-typo": SECRET}), 0o600, "Event-USER-typo"),
            ('{"version": 1, "turns": {', 0o600, "not valid JSON"),
        ],
    )
    def test_a_bad_judge_context_stops_the_daemon_with_a_reason(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        daemon_factory,
        content: object,
        mode: int,
        expected: str,
    ) -> None:
        private = tmp_path / "private"
        private.mkdir()
        path = _write(private / "judge_context.json", content, mode)
        monkeypatch.setenv("GAIA2_JUDGE_CONTEXT", str(path))
        daemon = daemon_factory()

        with pytest.raises(ValueError, match=expected):
            daemon.setup()

        status = _status(daemon)
        assert status["status"] == "error"
        assert expected in status["error"]
        assert SECRET not in status["error"]
        assert SECRET not in (daemon.state_dir / "eventd.log").read_text()

    def test_a_judge_context_owned_by_another_user_is_refused(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, daemon_factory
    ) -> None:
        _private_context(tmp_path, monkeypatch, **{"Event-USER-1": SECRET})
        monkeypatch.setattr(
            "gaia2_core.judge_context.os.geteuid", lambda: os.getuid() + 1
        )
        daemon = daemon_factory()
        with pytest.raises(ValueError, match="must be owned by the daemon"):
            daemon.setup()
        assert _status(daemon)["status"] == "error"

    def test_the_daemon_ignores_a_judge_context_next_to_the_scenario(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        daemon_factory,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        # Only the privacy-checked copy counts inside the container; the
        # sidecar is the host-side (offline re-judging) convention.
        monkeypatch.setenv("GAIA2_JUDGE_CONTEXT", str(tmp_path / "absent.json"))
        daemon = daemon_factory()
        _write(
            tmp_path / "audio_s1.judge_context.json",
            _context(**{"Event-USER-1": SECRET}),
            0o644,
        )
        with caplog.at_level("WARNING"):
            daemon.setup()
        assert daemon._processor._judge.tasks == [CARRIER + "\n"]
        assert (
            "Ignoring" in caplog.text and "audio_s1.judge_context.json" in caplog.text
        )
        assert SECRET not in caplog.text


class TestDaemonNeverLeaksTheText:
    """Run a judged scenario and grep everything the daemon produced.

    The judge engine echoes every prompt back as a failing verdict, the worst
    case for leakage: checker responses go to the daemon log (``/tmp``, which
    the agent can read) and to judgments.jsonl (which the agent can read
    under launchers that put it in the daemon's group). The judge-only text
    reaches no checker prompt at this commit (``Judge`` does not hand its
    engine to the subtask extractor), so nothing can be echoed. If that ever
    changes, this test fails on purpose: the judge's responses would then
    need redacting before they are logged.
    """

    @pytest.mark.parametrize("delivery", ["file", "http"])
    def test_no_artifact_contains_the_judge_only_text(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        daemon_factory,
        caplog: pytest.LogCaptureFixture,
        delivery: str,
    ) -> None:
        import gaia2_cli.judge as judge_module

        _private_context(tmp_path, monkeypatch, **{"Event-USER-1": SECRET})

        prompts: list[str] = []

        def echo_engine(messages, **_kwargs):
            # Echo the rendered user prompt (where a task would appear) as a
            # failing verdict, wrapped so that a subtask extractor would pass
            # it on too. Verdicts are logged and kept in judgments.jsonl.
            prompts.append(messages[-1]["content"])
            echoed = messages[-1]["content"]
            return f"[[Failure]] <subtask>{echoed}</subtask>", {}

        monkeypatch.setattr(
            judge_module, "create_litellm_engine", lambda **_kwargs: echo_engine
        )
        payloads: list[bytes] = []
        kwargs = {"judge_model": "echo-judge", "judge_provider": "openai"}
        if delivery == "http":
            kwargs["notify_url"] = "http://127.0.0.1:9"
        daemon = daemon_factory(**kwargs)

        class _Response:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def read(self):
                return b'{"ok": true, "runId": "r1"}'

        def fake_urlopen(url, data=None, timeout=10):
            payloads.append(data or b"")
            return _Response()

        monkeypatch.setattr(daemon, "_wait_for_notify_url", lambda: True)
        monkeypatch.setattr(daemon, "_noproxy_urlopen", fake_urlopen)

        caplog.set_level("DEBUG")
        daemon.setup()
        assert daemon._processor._judge.tasks == [SECRET + "\n"]  # it was applied
        with open(daemon.events_jsonl, "a") as f:
            f.write(
                json.dumps(
                    {
                        "t": 1.0,
                        "app": "AgentUserInterface",
                        "fn": "send_message_to_user",
                        "args": {"content": "copa_america.txt"},
                        "w": True,
                    }
                )
                + "\n"
            )
        daemon.run()

        status = _status(daemon)
        assert status["status"] == "stopped"  # the echoing judge failed it
        assert status["judgment"]["success"] is False
        assert prompts, "the judge LLM was never called"
        if delivery == "http":
            assert payloads == [json.dumps({"message": CARRIER}).encode()]
        else:
            notifications = (daemon.state_dir / "notifications.jsonl").read_text()
            assert json.loads(notifications) == {
                "type": "user_message",
                "content": CARRIER,
                "turn": 0,
            }

        produced = {
            str(p.relative_to(tmp_path)): p.read_text(errors="replace")
            for p in daemon.state_dir.rglob("*")
            if p.is_file()
        }
        assert {
            "state/eventd.log",
            "state/daemon_status.json",
            "state/judgments.jsonl",
        } <= set(produced)
        produced["<log records>"] = caplog.text
        produced["<adapter payloads>"] = b"\n".join(payloads).decode()
        for name, text in produced.items():
            assert SECRET not in text, name
            assert "Copa America" not in text, name
