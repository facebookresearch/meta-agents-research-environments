# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from __future__ import annotations

import json
from pathlib import Path

from gaia2_runner.serve import _is_multi_run, _refresh_fingerprint


def test_is_multi_run_treats_split_root_as_single_run(tmp_path: Path) -> None:
    scenario_dir = tmp_path / "search" / "scenario_1"
    scenario_dir.mkdir(parents=True)
    (scenario_dir / "result.json").write_text(
        json.dumps({"scenario_id": "scenario_1", "success": True})
    )

    assert _is_multi_run(str(tmp_path)) is False


def test_is_multi_run_detects_pass_at_root(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_1"
    run_dir.mkdir()
    (run_dir / "run_config.json").write_text("{}")

    assert _is_multi_run(str(tmp_path)) is True


def test_is_multi_run_detects_nested_experiment_root(tmp_path: Path) -> None:
    run_dir = tmp_path / "experiment_a" / "run_1"
    run_dir.mkdir(parents=True)
    (run_dir / "run_config.json").write_text("{}")

    assert _is_multi_run(str(tmp_path)) is True


def test_refresh_fingerprint_changes_for_relevant_artifacts(tmp_path: Path) -> None:
    scenario_dir = tmp_path / "scenario_1"
    scenario_dir.mkdir()
    trace_path = scenario_dir / "trace.jsonl"
    trace_path.write_text('{"seq": 1}\n')

    before = _refresh_fingerprint(tmp_path)

    trace_path.write_text('{"seq": 1}\n{"seq": 2}\n')

    assert _refresh_fingerprint(tmp_path) != before


def test_refresh_fingerprint_ignores_unrelated_files(tmp_path: Path) -> None:
    scenario_dir = tmp_path / "scenario_1"
    scenario_dir.mkdir()
    (scenario_dir / "daemon_status.json").write_text('{"status": "running"}')

    before = _refresh_fingerprint(tmp_path)

    (tmp_path / "notes.txt").write_text("hello")

    assert _refresh_fingerprint(tmp_path) == before
