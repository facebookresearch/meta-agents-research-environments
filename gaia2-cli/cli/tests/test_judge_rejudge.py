# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
"""Tests for offline re-scoring of a completed scenario (``gaia2_cli.judge.rejudge``).

These exercise the engine-free path: with no ``judge_model`` the Judge falls back
to deterministic argument matching, so a replay is fully reproducible in CI.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from gaia2_cli.judge.rejudge import main, rejudge

# ---------------------------------------------------------------------------
# Fixture scenario: one turn, two oracle AGENT events (send_email + SMU)
# ---------------------------------------------------------------------------

SEND_EMAIL_ARGS = {"recipients": ["bo@example.invalid"], "subject": "Report"}

_SCENARIO = {
    "apps": [
        {"name": "AgentUserInterface", "class_name": "AgentUserInterface"},
        {"name": "EmailClientV2", "class_name": "EmailClientV2"},
    ],
    "events": [
        {
            "event_id": "user-1",
            "event_type": "USER",
            "event_time": 0.0,
            "action": {
                "app": "AgentUserInterface",
                "function": "send_message_to_agent",
                "operation_type": "write",
                "args": [
                    {
                        "name": "content",
                        "value": "Send the report email.",
                        "value_type": "str",
                    }
                ],
            },
        },
        {
            "event_id": "oracle-1",
            "event_type": "AGENT",
            "event_time": 1.0,
            "dependencies": ["user-1"],
            "action": {
                "app": "EmailClientV2",
                "function": "send_email",
                "operation_type": "write",
                "args": [
                    {
                        "name": "recipients",
                        "value": '["bo@example.invalid"]',
                        "value_type": "list[str]",
                    },
                    {"name": "subject", "value": "Report", "value_type": "str"},
                ],
            },
        },
        {
            "event_id": "oracle-2",
            "event_type": "AGENT",
            "event_time": 2.0,
            "dependencies": ["oracle-1"],
            "action": {
                "app": "AgentUserInterface",
                "function": "send_message_to_user",
                "operation_type": "write",
                "args": [{"name": "content", "value": "Done.", "value_type": "str"}],
            },
        },
    ],
    "metadata": {
        "definition": {
            "start_time": 0.0,
            "duration": 100.0,
            "time_increment_in_seconds": 1.0,
        }
    },
}

_SMU_ACTION = {
    "t": 2.0,
    "sim_t": "1970-01-01 00:00:02",
    "app": "AgentUserInterface",
    "fn": "send_message_to_user",
    "args": {"content": "Done."},
    "w": True,
    "ret": None,
}

_SEND_EMAIL_ACTION = {
    "t": 1.0,
    "sim_t": "1970-01-01 00:00:01",
    "app": "EmailClientV2",
    "fn": "send_email",
    "args": SEND_EMAIL_ARGS,
    "w": True,
    "ret": "ok",
}


def _make_trajectory(
    tmp_path: Path, actions: list[dict], scenario_file: str | None = None
) -> Path:
    """Write a scenario JSON plus a completed-trajectory dir, return the dir."""
    scenario_path = tmp_path / "scenario.json"
    scenario_path.write_text(json.dumps(_SCENARIO))

    scenario_dir = tmp_path / "traj"
    scenario_dir.mkdir()
    (scenario_dir / "result.json").write_text(
        json.dumps({"scenario_file": scenario_file or str(scenario_path)})
    )
    with (scenario_dir / "events.jsonl").open("w") as f:
        for action in actions:
            f.write(json.dumps(action) + "\n")
    return scenario_dir


def _no_engine_kwargs() -> dict:
    return {
        "judge_model": None,
        "judge_provider": None,
        "judge_base_url": None,
        "judge_api_key": None,
        "judge_prompt_version": None,
    }


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_rejudge_rescores_a_completed_trajectory(tmp_path: Path) -> None:
    scenario_dir = _make_trajectory(tmp_path, [_SEND_EMAIL_ACTION, _SMU_ACTION])
    events_before = (scenario_dir / "events.jsonl").read_text()
    files_before = sorted(p.name for p in scenario_dir.iterdir())

    out_path = tmp_path / "replay" / "replay_judgments.jsonl"
    summary = rejudge(scenario_dir, out_path, _no_engine_kwargs())

    assert summary["num_turns_replayed"] == 1
    assert summary["success"] is True
    assert summary["results"] == [{"turn": 0, "success": True, "failure_reason": ""}]

    written = [json.loads(line) for line in out_path.read_text().splitlines() if line]
    assert written == summary["results"]

    # The agent is never re-run: the recorded action stream is only read, and
    # no new artifacts appear in the original trajectory dir.
    assert (scenario_dir / "events.jsonl").read_text() == events_before
    assert sorted(p.name for p in scenario_dir.iterdir()) == files_before


def test_rejudge_fails_a_trajectory_missing_an_oracle_action(tmp_path: Path) -> None:
    """Same scenario, but the agent never sent the email — verdict must flip."""
    scenario_dir = _make_trajectory(tmp_path, [_SMU_ACTION])

    out_path = tmp_path / "replay" / "replay_judgments.jsonl"
    summary = rejudge(scenario_dir, out_path, _no_engine_kwargs())

    assert summary["num_turns_replayed"] == 1
    assert summary["success"] is False
    assert summary["results"][0]["failure_reason"]


def test_rejudge_applies_the_requested_prompt_version(tmp_path: Path) -> None:
    """``judge_prompt_version`` is resolved through the override registry."""
    scenario_dir = _make_trajectory(tmp_path, [_SEND_EMAIL_ACTION, _SMU_ACTION])

    kwargs = _no_engine_kwargs() | {"judge_prompt_version": "omnigaia"}
    summary = rejudge(
        scenario_dir, tmp_path / "replay" / "replay_judgments.jsonl", kwargs
    )
    assert summary["success"] is True

    with pytest.raises(KeyError, match="Unknown judge prompt version"):
        rejudge(
            scenario_dir,
            tmp_path / "replay2" / "replay_judgments.jsonl",
            _no_engine_kwargs() | {"judge_prompt_version": "nope"},
        )


# ---------------------------------------------------------------------------
# Missing artifacts
# ---------------------------------------------------------------------------


def test_rejudge_errors_cleanly_when_the_scenario_file_is_gone(tmp_path: Path) -> None:
    scenario_dir = _make_trajectory(
        tmp_path,
        [_SEND_EMAIL_ACTION, _SMU_ACTION],
        scenario_file=str(tmp_path / "deleted.json"),
    )

    with pytest.raises(SystemExit, match="scenario_file does not exist"):
        rejudge(
            scenario_dir,
            tmp_path / "replay" / "replay_judgments.jsonl",
            _no_engine_kwargs(),
        )


def test_rejudge_errors_cleanly_when_result_json_has_no_scenario_file(
    tmp_path: Path,
) -> None:
    scenario_dir = _make_trajectory(tmp_path, [_SMU_ACTION])
    (scenario_dir / "result.json").write_text(json.dumps({"status": "COMPLETED"}))

    with pytest.raises(SystemExit, match="missing scenario_file"):
        rejudge(
            scenario_dir,
            tmp_path / "replay" / "replay_judgments.jsonl",
            _no_engine_kwargs(),
        )


def test_rejudge_cli_errors_cleanly_on_a_dir_without_events(
    tmp_path: Path, monkeypatch
) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.setattr(
        "sys.argv",
        ["rejudge", "--scenario-dir", str(empty), "--out", str(tmp_path / "o.jsonl")],
    )

    with pytest.raises(SystemExit, match="No events.jsonl"):
        main()


def test_rejudge_cli_errors_cleanly_on_a_missing_dir(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(
        "sys.argv", ["rejudge", "--scenario-dir", str(tmp_path / "nope")]
    )

    with pytest.raises(SystemExit, match="Not a directory"):
        main()
