# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms of the Apache 2.0 license
# found in the LICENSE file in the root directory of this source tree.

"""Tests for the sharding hook in _load_dataset_scenarios."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from gaia2_runner.cli import _load_dataset_scenarios


def _make_dataset(root: Path, num_scenarios: int) -> Path:
    """Create ``num_scenarios`` dummy scenario JSON files under ``root``."""
    root.mkdir(parents=True, exist_ok=True)
    for i in range(num_scenarios):
        (root / f"scenario_{i:03d}.json").write_text(
            json.dumps(
                {
                    "metadata": {"definition": {"scenario_id": f"scenario_{i:03d}"}},
                    "events": [],
                }
            )
            + "\n"
        )
    return root


class TestLoadDatasetScenariosSharding:
    def test_shard_0_of_4_returns_disjoint_subset(self, tmp_path: Path):
        dataset = _make_dataset(tmp_path / "ds", 12)
        paths, _, _ = _load_dataset_scenarios(
            str(dataset), limit=None, shard_id=0, num_shards=4
        )
        assert len(paths) == 3
        # round-robin on sorted paths: shard 0 gets indices 0, 4, 8.
        names = sorted(p.name for p in paths)
        assert names == [
            "scenario_000.json",
            "scenario_004.json",
            "scenario_008.json",
        ]

    def test_all_shards_together_cover_dataset_exactly_once(self, tmp_path: Path):
        dataset = _make_dataset(tmp_path / "ds", 20)
        all_names: list[str] = []
        for i in range(5):
            paths, _, _ = _load_dataset_scenarios(
                str(dataset), limit=None, shard_id=i, num_shards=5
            )
            all_names.extend(p.name for p in paths)
        assert sorted(all_names) == sorted(f"scenario_{i:03d}.json" for i in range(20))
        assert len(all_names) == len(set(all_names))

    def test_sharding_applies_after_limit(self, tmp_path: Path):
        # limit=8 keeps the first 8 (sorted) scenarios; then 4 shards of 8 -> 2 each.
        dataset = _make_dataset(tmp_path / "ds", 20)
        paths, _, _ = _load_dataset_scenarios(
            str(dataset), limit=8, shard_id=0, num_shards=4
        )
        assert len(paths) == 2
        names = sorted(p.name for p in paths)
        # round-robin on the FIRST EIGHT sorted scenarios -> shard 0 gets 0,4.
        assert names == ["scenario_000.json", "scenario_004.json"]

    def test_rejects_out_of_range_shard_id(self, tmp_path: Path):
        dataset = _make_dataset(tmp_path / "ds", 4)
        with pytest.raises(ValueError, match="shard_id"):
            _load_dataset_scenarios(str(dataset), limit=None, shard_id=5, num_shards=4)
