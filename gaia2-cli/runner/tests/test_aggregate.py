# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms of the Apache 2.0 license
# found in the LICENSE file in the root directory of this source tree.

"""Tests for shard aggregation: merge results.jsonl across shard_*/ subdirs."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner
from gaia2_runner.aggregate import aggregate_shards
from gaia2_runner.cli import main


def _write_shard(
    root: Path, shard_id: int, num_shards: int, results: list[dict]
) -> Path:
    shard_dir = root / f"shard_{shard_id:02d}_of_{num_shards:02d}"
    shard_dir.mkdir(parents=True, exist_ok=True)
    with (shard_dir / "results.jsonl").open("w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    return shard_dir


class TestAggregateShards:
    def test_concatenates_results_across_shards(self, tmp_path: Path):
        _write_shard(
            tmp_path,
            0,
            3,
            [{"scenario_id": "s00", "success": True, "reward": 1.0}],
        )
        _write_shard(
            tmp_path,
            1,
            3,
            [{"scenario_id": "s01", "success": False, "reward": 0.0}],
        )
        _write_shard(
            tmp_path,
            2,
            3,
            [
                {"scenario_id": "s02", "success": True, "reward": 1.0},
                {"scenario_id": "s05", "success": True, "reward": 0.5},
            ],
        )

        summary = aggregate_shards(tmp_path)

        merged_path = tmp_path / "results.jsonl"
        assert merged_path.exists()
        lines = merged_path.read_text().splitlines()
        assert len(lines) == 4
        rows = [json.loads(line) for line in lines]
        ids = sorted(r["scenario_id"] for r in rows)
        assert ids == ["s00", "s01", "s02", "s05"]

        assert summary["num_shards"] == 3
        assert summary["num_results"] == 4

    def test_overlapping_scenario_ids_across_shards_raises(self, tmp_path: Path):
        _write_shard(tmp_path, 0, 2, [{"scenario_id": "s00", "success": True}])
        _write_shard(tmp_path, 1, 2, [{"scenario_id": "s00", "success": False}])
        with pytest.raises(ValueError, match="overlapping scenario"):
            aggregate_shards(tmp_path)

    def test_missing_shards_subdir_raises(self, tmp_path: Path):
        # tmp_path has no shard_*_of_* subdirs.
        with pytest.raises(FileNotFoundError, match="no shard"):
            aggregate_shards(tmp_path)

    def test_ignores_non_shard_subdirs(self, tmp_path: Path):
        _write_shard(tmp_path, 0, 2, [{"scenario_id": "s0", "success": True}])
        _write_shard(tmp_path, 1, 2, [{"scenario_id": "s1", "success": True}])
        # An extraneous directory that does NOT match the shard pattern.
        (tmp_path / "logs").mkdir()
        (tmp_path / "logs" / "results.jsonl").write_text(
            json.dumps({"scenario_id": "should_be_ignored"}) + "\n"
        )

        aggregate_shards(tmp_path)
        rows = [
            json.loads(line)
            for line in (tmp_path / "results.jsonl").read_text().splitlines()
        ]
        ids = sorted(r["scenario_id"] for r in rows)
        assert ids == ["s0", "s1"]

    def test_shard_with_no_results_jsonl_is_skipped(self, tmp_path: Path):
        # Shard ran but crashed before writing results.
        (tmp_path / "shard_00_of_02").mkdir(parents=True)
        _write_shard(tmp_path, 1, 2, [{"scenario_id": "s1", "success": True}])
        summary = aggregate_shards(tmp_path)
        assert summary["num_shards_with_results"] == 1
        assert summary["num_results"] == 1


class TestAggregateCli:
    def test_aggregate_subcommand_writes_merged_results_jsonl(self, tmp_path: Path):
        _write_shard(tmp_path, 0, 2, [{"scenario_id": "s0", "success": True}])
        _write_shard(tmp_path, 1, 2, [{"scenario_id": "s1", "success": False}])

        result = CliRunner().invoke(
            main,
            ["aggregate", "--output-dir", str(tmp_path)],
            catch_exceptions=False,
        )
        assert result.exit_code == 0, result.output
        assert (tmp_path / "results.jsonl").exists()
        rows = [
            json.loads(line)
            for line in (tmp_path / "results.jsonl").read_text().splitlines()
        ]
        assert sorted(r["scenario_id"] for r in rows) == ["s0", "s1"]
