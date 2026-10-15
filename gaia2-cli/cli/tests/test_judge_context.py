# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
"""Tests for the judge-only task text side file (gaia2_core.judge_context)."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from gaia2_core.judge_context import (
    DEFAULT_JUDGE_CONTEXT_PATH,
    JudgeContextError,
    judge_context_sidecar_path,
    load_judge_context,
    parse_judge_context,
    resolve_judge_context_path,
)

SECRET = "Find the txt file about Copa America 2024 in my news folder."


def _write(path: Path, data: object, mode: int = 0o600) -> Path:
    path.write_text(data if isinstance(data, str) else json.dumps(data))
    os.chmod(path, mode)
    return path


def _context(**turns: str) -> dict:
    return {"version": 1, "turns": {k: {"text": v} for k, v in turns.items()}}


class TestParse:
    def test_valid_context_maps_event_ids_to_text(self) -> None:
        data = {
            "version": 1,
            "turns": {
                "Event-USER-a188eb04": {"text": SECRET},
                "Event-USER-2": {"text": "Then email it to Bob."},
            },
        }
        assert parse_judge_context(data) == {
            "Event-USER-a188eb04": SECRET,
            "Event-USER-2": "Then email it to Bob.",
        }

    @pytest.mark.parametrize(
        "data, expected",
        [
            ([], "must be a JSON object"),
            ({"turns": {}}, "missing: ['version']"),
            (
                {"version": 1, "turns": {}, "transcript": "x"},
                "unexpected: ['transcript']",
            ),
            ({"version": 2, "turns": {"E": {"text": "x"}}}, "unsupported version 2"),
            (
                {"version": "1", "turns": {"E": {"text": "x"}}},
                "unsupported version string",
            ),
            (
                {"version": True, "turns": {"E": {"text": "x"}}},
                "unsupported version boolean",
            ),
            ({"version": 1, "turns": []}, "'turns' must be an object"),
            ({"version": 1, "turns": {}}, "'turns' is empty"),
            ({"version": 1, "turns": {"E": "x"}}, "exactly one key, 'text'"),
            (
                {"version": 1, "turns": {"E": {"text": "x", "role": "a"}}},
                "exactly one key",
            ),
            ({"version": 1, "turns": {"E": {"text": 3}}}, "'text' must be a string"),
            ({"version": 1, "turns": {"E": {"text": "  \n"}}}, "'text' is empty"),
        ],
    )
    def test_malformed_context_is_rejected(self, data: object, expected: str) -> None:
        with pytest.raises(JudgeContextError, match=None) as excinfo:
            parse_judge_context(data)
        assert expected in str(excinfo.value)

    def test_errors_never_echo_the_text(self) -> None:
        # The text, wherever it is misplaced, must not reach an error message:
        # the daemon's log is readable by the agent.
        cases = [
            {"version": 1, "turns": {"E": {"text": SECRET, "extra": 1}}},
            {"version": 1, "turns": {SECRET: SECRET}},
            {"version": 1, "turns": {"E": {"text": SECRET}}, SECRET: 1},
            {"version": 9, "turns": {"E": {"text": SECRET}}},
        ]
        for data in cases:
            with pytest.raises(JudgeContextError) as excinfo:
                parse_judge_context(data)
            assert SECRET not in str(excinfo.value)
            assert "Copa" not in str(excinfo.value)


class TestLoad:
    def test_missing_file(self, tmp_path: Path) -> None:
        path = tmp_path / "judge_context.json"
        assert load_judge_context(path, missing_ok=True) is None
        with pytest.raises(JudgeContextError, match="not found"):
            load_judge_context(path)

    def test_loads_a_private_file(self, tmp_path: Path) -> None:
        path = _write(tmp_path / "judge_context.json", _context(**{"E-1": SECRET}))
        assert load_judge_context(path, require_private=True) == {"E-1": SECRET}

    @pytest.mark.parametrize("mode", [0o640, 0o644, 0o604, 0o660, 0o610])
    def test_require_private_refuses_group_or_other_access(
        self, tmp_path: Path, mode: int
    ) -> None:
        # Under some launchers the agent shares the daemon's group.
        path = _write(tmp_path / "judge_context.json", _context(E=SECRET), mode)
        with pytest.raises(JudgeContextError, match="must be 0600") as excinfo:
            load_judge_context(path, require_private=True)
        assert SECRET not in str(excinfo.value)
        # Without the privacy requirement (host-side re-judging) it loads.
        assert load_judge_context(path) == {"E": SECRET}

    def test_invalid_json_and_duplicate_keys(self, tmp_path: Path) -> None:
        bad = _write(tmp_path / "bad.json", '{"version": 1, "turns": {"E": {"text": ')
        with pytest.raises(JudgeContextError, match="not valid JSON") as excinfo:
            load_judge_context(bad)
        assert str(bad) in str(excinfo.value)

        dup = _write(
            tmp_path / "dup.json",
            '{"version": 1, "turns": {"E": {"text": "a"}, "E": {"text": "b"}}}',
        )
        with pytest.raises(JudgeContextError, match="duplicate key E"):
            load_judge_context(dup)

    def test_directory_is_reported_as_unreadable(self, tmp_path: Path) -> None:
        with pytest.raises(JudgeContextError, match="cannot read judge context"):
            load_judge_context(tmp_path)


class TestPaths:
    def test_sidecar_sits_next_to_the_scenario(self) -> None:
        assert judge_context_sidecar_path("/data/search/s1.json") == Path(
            "/data/search/s1.judge_context.json"
        )

    def test_in_container_path_honours_the_env_var(self) -> None:
        assert resolve_judge_context_path({}) == Path(DEFAULT_JUDGE_CONTEXT_PATH)
        assert DEFAULT_JUDGE_CONTEXT_PATH == "/var/gaia2/judge_context.json"
        assert resolve_judge_context_path({"GAIA2_JUDGE_CONTEXT": "/x/j.json"}) == Path(
            "/x/j.json"
        )
