# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms of the Apache 2.0 license
# found in the LICENSE file in the root directory of this source tree.

"""Tests for [run].shard_id / [run].num_shards in the TOML loader, and the
multi-split sharding behavior of _load_run_config_dataset_scenarios."""

from __future__ import annotations

import json
from pathlib import Path

import click
import pytest
from gaia2_runner.cli import _load_run_config_dataset_scenarios
from gaia2_runner.config import load_runner_toml_config


def _write_dataset_with_splits(root: Path, splits: dict[str, int]) -> None:
    for split, n in splits.items():
        split_dir = root / split
        split_dir.mkdir(parents=True, exist_ok=True)
        for i in range(n):
            (split_dir / f"scenario_{split}_{i:03d}.json").write_text(
                json.dumps(
                    {
                        "metadata": {"definition": {"scenario_id": f"{split}_{i:03d}"}},
                        "events": [],
                    }
                )
                + "\n"
            )


def _write_config(
    path: Path, dataset_root: Path, *, splits: str, run_extras: str = ""
) -> Path:
    path.write_text(
        f"""
[target]
dataset_root = "{dataset_root}"
splits = {splits}

[agent]
image = "localhost/gaia2-oc:latest"
provider = "openai-compat"
model = "stub"
api_key = "stub"

[judge]
provider = "openai-compat"
model = "stub-judge"
api_key = "stub"

[run]
output_dir = "{path.parent / "out"}"
{run_extras}
"""
    )
    return path


class TestRunConfigShardKeys:
    def test_defaults_when_omitted(self, tmp_path: Path):
        _write_dataset_with_splits(tmp_path / "ds", {"search": 4})
        cfg_path = _write_config(
            tmp_path / "eval.toml", tmp_path / "ds", splits='"search"'
        )
        cfg = load_runner_toml_config(str(cfg_path))
        assert cfg.run.shard_id == 0
        assert cfg.run.num_shards == 1

    def test_reads_shard_id_and_num_shards(self, tmp_path: Path):
        _write_dataset_with_splits(tmp_path / "ds", {"search": 4})
        cfg_path = _write_config(
            tmp_path / "eval.toml",
            tmp_path / "ds",
            splits='"search"',
            run_extras="shard_id = 2\nnum_shards = 4",
        )
        cfg = load_runner_toml_config(str(cfg_path))
        assert cfg.run.shard_id == 2
        assert cfg.run.num_shards == 4

    def test_rejects_out_of_range_shard_id(self, tmp_path: Path):
        _write_dataset_with_splits(tmp_path / "ds", {"search": 4})
        cfg_path = _write_config(
            tmp_path / "eval.toml",
            tmp_path / "ds",
            splits='"search"',
            run_extras="shard_id = 4\nnum_shards = 4",
        )
        with pytest.raises(click.UsageError, match="shard_id"):
            load_runner_toml_config(str(cfg_path))

    def test_rejects_non_positive_num_shards(self, tmp_path: Path):
        _write_dataset_with_splits(tmp_path / "ds", {"search": 4})
        cfg_path = _write_config(
            tmp_path / "eval.toml",
            tmp_path / "ds",
            splits='"search"',
            run_extras="num_shards = 0",
        )
        with pytest.raises(click.UsageError, match="num_shards"):
            load_runner_toml_config(str(cfg_path))


class TestRunConfigDatasetScenariosWithSharding:
    """The multi-split path must apply sharding AFTER concatenating splits,
    so that shard 0 of (search ∪ execution) is a single coherent partition,
    not (shard 0 of search ∪ shard 0 of execution).
    """

    def test_multi_split_sharding_applies_after_concatenation(self, tmp_path: Path):
        # 6 scenarios per split * 2 splits = 12 total. With 4 shards, each
        # shard gets 3 scenarios, regardless of how scenarios spread across splits.
        _write_dataset_with_splits(tmp_path / "ds", {"search": 6, "execution": 6})
        cfg_path = _write_config(
            tmp_path / "eval.toml",
            tmp_path / "ds",
            splits='["search", "execution"]',
            run_extras="shard_id = 0\nnum_shards = 4",
        )
        cfg = load_runner_toml_config(str(cfg_path))
        scenario_paths, _, _, _ = _load_run_config_dataset_scenarios(cfg)
        assert len(scenario_paths) == 3

    def test_multi_split_shards_disjoint_and_cover_all(self, tmp_path: Path):
        _write_dataset_with_splits(tmp_path / "ds", {"search": 6, "execution": 6})
        all_names: list[str] = []
        for i in range(4):
            cfg_path = _write_config(
                tmp_path / f"eval_{i}.toml",
                tmp_path / "ds",
                splits='["search", "execution"]',
                run_extras=f"shard_id = {i}\nnum_shards = 4",
            )
            cfg = load_runner_toml_config(str(cfg_path))
            paths, _, _, _ = _load_run_config_dataset_scenarios(cfg)
            all_names.extend(p.name for p in paths)
        assert sorted(all_names) == sorted(
            [f"scenario_search_{i:03d}.json" for i in range(6)]
            + [f"scenario_execution_{i:03d}.json" for i in range(6)]
        )
        assert len(all_names) == len(set(all_names))

    def test_single_split_no_sharding_unchanged(self, tmp_path: Path):
        _write_dataset_with_splits(tmp_path / "ds", {"search": 10})
        cfg_path = _write_config(
            tmp_path / "eval.toml",
            tmp_path / "ds",
            splits='"search"',
        )
        cfg = load_runner_toml_config(str(cfg_path))
        paths, _, _, _ = _load_run_config_dataset_scenarios(cfg)
        assert len(paths) == 10


class TestShardedOutputDirNamespacing:
    """When ``num_shards > 1``, the configured ``output_dir`` is namespaced
    with ``shard_{id:02d}_of_{n:02d}`` so concurrent shards don't trip
    the runner's "refuses to overwrite an existing output_dir" check."""

    def test_unsharded_output_dir_unchanged(self, tmp_path: Path):
        _write_dataset_with_splits(tmp_path / "ds", {"search": 4})
        cfg_path = _write_config(
            tmp_path / "eval.toml", tmp_path / "ds", splits='"search"'
        )
        cfg = load_runner_toml_config(str(cfg_path))
        assert cfg.run.output_dir == str((tmp_path / "out").resolve())

    def test_sharded_output_dir_gets_shard_subdir(self, tmp_path: Path):
        _write_dataset_with_splits(tmp_path / "ds", {"search": 4})
        cfg_path = _write_config(
            tmp_path / "eval.toml",
            tmp_path / "ds",
            splits='"search"',
            run_extras="shard_id = 2\nnum_shards = 8",
        )
        cfg = load_runner_toml_config(str(cfg_path))
        assert cfg.run.output_dir == str(
            (tmp_path / "out" / "shard_02_of_08").resolve()
        )

    def test_shard_subdir_only_appended_when_num_shards_gt_1(self, tmp_path: Path):
        _write_dataset_with_splits(tmp_path / "ds", {"search": 4})
        cfg_path = _write_config(
            tmp_path / "eval.toml",
            tmp_path / "ds",
            splits='"search"',
            run_extras="shard_id = 0\nnum_shards = 1",
        )
        cfg = load_runner_toml_config(str(cfg_path))
        assert cfg.run.output_dir == str((tmp_path / "out").resolve())
