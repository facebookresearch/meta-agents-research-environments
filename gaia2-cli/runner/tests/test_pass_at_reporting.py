# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from __future__ import annotations

import json
from pathlib import Path

from gaia2_runner import cli as runner_cli
from gaia2_runner.cli import (
    RunStats,
    _load_run_stats_from_disk,
    _print_aggregate_summary,
    _summarize_results_by_split,
)


def _write_result(
    base: Path,
    run_name: str,
    scenario_id: str,
    success: bool | None,
) -> None:
    result_dir = base / run_name / scenario_id
    result_dir.mkdir(parents=True, exist_ok=True)
    (result_dir / "result.json").write_text(
        json.dumps({"scenario_id": scenario_id, "success": success}) + "\n"
    )


def test_load_run_stats_from_disk_counts_nested_results(tmp_path: Path) -> None:
    _write_result(tmp_path, "run_1", "scenario_a", True)
    _write_result(tmp_path, "run_1", "scenario_b", False)
    _write_result(tmp_path, "run_1", "scenario_c", None)

    stats = _load_run_stats_from_disk(str(tmp_path / "run_1"))

    assert stats.total == 3
    assert stats.passed == 1
    assert stats.failed == 1
    assert stats.errors == 1


def test_print_aggregate_summary_reports_avg_and_pass_at(
    tmp_path: Path, capsys
) -> None:
    run_1 = RunStats()
    run_1.record({"scenario_id": "scenario_a", "success": True})
    run_1.record({"scenario_id": "scenario_b", "success": False})

    run_2 = RunStats()
    run_2.record({"scenario_id": "scenario_a", "success": False})
    run_2.record({"scenario_id": "scenario_b", "success": True})

    _print_aggregate_summary([(1, run_1), (2, run_2)], str(tmp_path))

    captured = capsys.readouterr().out
    assert "Aggregate pass@2 summary" in captured
    assert "run_1: 1/2 passed" in captured
    assert "run_2: 1/2 passed" in captured
    assert "avg@2: 50.0% ± 0.0%" in captured
    assert "pass@2: 100.0% ± 0.0%" in captured


def test_summarize_results_by_split_uses_dataset_root_and_split_order(
    tmp_path: Path,
) -> None:
    dataset_root = tmp_path / "dataset"
    results = [
        {
            "scenario_id": "scenario_a",
            "success": True,
            "scenario_file": str(dataset_root / "time" / "scenario_a.json"),
        },
        {
            "scenario_id": "scenario_b",
            "success": False,
            "scenario_file": str(dataset_root / "adaptability" / "scenario_b.json"),
        },
        {
            "scenario_id": "scenario_c",
            "success": True,
            "scenario_file": str(dataset_root / "adaptability" / "scenario_c.json"),
        },
    ]

    split_stats = _summarize_results_by_split(results, dataset_root=dataset_root)

    assert list(split_stats) == ["adaptability", "time"]
    assert split_stats["adaptability"].total == 2
    assert split_stats["adaptability"].passed == 1
    assert split_stats["adaptability"].failed == 1
    assert split_stats["time"].total == 1
    assert split_stats["time"].passed == 1


def test_single_run_retry_summary_rebuilds_from_disk(
    tmp_path: Path, monkeypatch
) -> None:
    dataset_root = tmp_path / "dataset"

    batch_stats = RunStats()
    batch_stats.record(
        {
            "scenario_id": "scenario_a",
            "success": True,
            "scenario_file": str(dataset_root / "search" / "scenario_a.json"),
        }
    )
    batch_stats.record(
        {
            "scenario_id": "scenario_b",
            "success": False,
            "scenario_file": str(dataset_root / "search" / "scenario_b.json"),
        }
    )

    merged_stats = RunStats()
    merged_stats.record(
        {
            "scenario_id": "scenario_a",
            "success": True,
            "scenario_file": str(dataset_root / "search" / "scenario_a.json"),
        }
    )
    merged_stats.record(
        {
            "scenario_id": "scenario_b",
            "success": False,
            "scenario_file": str(dataset_root / "search" / "scenario_b.json"),
        }
    )
    merged_stats.record(
        {
            "scenario_id": "scenario_c",
            "success": True,
            "scenario_file": str(dataset_root / "time" / "scenario_c.json"),
        }
    )
    merged_stats.record(
        {
            "scenario_id": "scenario_d",
            "success": True,
            "scenario_file": str(dataset_root / "time" / "scenario_d.json"),
        }
    )
    merged_stats.record(
        {
            "scenario_id": "scenario_e",
            "success": False,
            "scenario_file": str(dataset_root / "time" / "scenario_e.json"),
        }
    )

    captured: dict[str, object] = {}

    monkeypatch.setattr(
        runner_cli, "_save_dataset_run_config", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        runner_cli,
        "_run_scenarios_concurrent",
        lambda **kwargs: batch_stats,
    )
    monkeypatch.setattr(
        runner_cli,
        "_load_run_stats_from_disk",
        lambda output_dir: merged_stats,
    )
    monkeypatch.setattr(
        runner_cli,
        "_generate_trace_viewer_if_possible",
        lambda *args, **kwargs: None,
    )

    def fake_write_results_jsonl(
        output_file: str | Path,
        results: list[dict],
        *,
        log_label: str = "Results",
    ) -> None:
        captured["output_file"] = str(output_file)
        captured["written_count"] = len(results)
        captured["log_label"] = log_label

    def fake_print_stats_summary(
        title: str,
        stats: RunStats,
        *,
        output_dir: str | None = None,
        split_stats: dict[str, RunStats] | None = None,
    ) -> None:
        captured["title"] = title
        captured["summary_total"] = stats.total
        captured["summary_passed"] = stats.passed
        captured["summary_failed"] = stats.failed
        captured["summary_errors"] = stats.errors
        captured["summary_output_dir"] = output_dir
        captured["summary_split_keys"] = list(split_stats or {})

    monkeypatch.setattr(runner_cli, "_write_results_jsonl", fake_write_results_jsonl)
    monkeypatch.setattr(runner_cli, "_print_stats_summary", fake_print_stats_summary)

    summary = runner_cli._run_dataset_once(
        scenario_paths=[Path("scenario_a.json"), Path("scenario_b.json")],
        execution_config=object(),
        adapter_port=8090,
        concurrency=2,
        output_dir=str(tmp_path),
        output_file=str(tmp_path / "results.jsonl"),
        run_config_base={},
        dataset_root=dataset_root,
        retry=True,
    )

    assert captured["written_count"] == 5
    assert captured["log_label"] == "Merged results"
    assert captured["summary_total"] == 5
    assert captured["summary_passed"] == 3
    assert captured["summary_failed"] == 2
    assert captured["summary_split_keys"] == ["search", "time"]
    assert summary["total"] == 5
    assert summary["passed"] == 3
    assert summary["failed"] == 2
    assert summary["by_split"]["search"]["total"] == 2
    assert summary["by_split"]["search"]["passed"] == 1
    assert summary["by_split"]["time"]["total"] == 3
    assert summary["by_split"]["time"]["passed"] == 2
    assert summary["retry_batch"]["total"] == 2
    assert summary["retry_batch"]["passed"] == 1
    assert summary["retry_batch"]["failed"] == 1
    assert summary["retry_batch"]["by_split"]["search"]["total"] == 2
