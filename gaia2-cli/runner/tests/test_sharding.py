# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms of the Apache 2.0 license
# found in the LICENSE file in the root directory of this source tree.

"""Tests for scenario sharding — pure function, no I/O."""

from __future__ import annotations

from pathlib import Path

import pytest
from gaia2_runner.sharding import select_shard


def _paths(n: int) -> list[Path]:
    # Use scenario-id-shaped filenames so the sort is meaningful.
    return [Path(f"/tmp/scenario_{i:03d}.json") for i in range(n)]


class TestSelectShard:
    def test_num_shards_one_returns_all_paths(self):
        paths = _paths(10)
        assert select_shard(paths, shard_id=0, num_shards=1) == paths

    def test_shards_partition_the_input_with_no_overlap(self):
        paths = _paths(20)
        shards = [select_shard(paths, shard_id=i, num_shards=4) for i in range(4)]
        # Each path appears exactly once across all shards.
        flat = [p for shard in shards for p in shard]
        assert sorted(flat) == sorted(paths)
        assert len(flat) == len(set(flat))

    def test_round_robin_assignment_for_even_split(self):
        paths = _paths(8)
        # round-robin: shard 0 takes 0,4; shard 1 takes 1,5; etc.
        # _paths returns already sorted, so this is the expected assignment.
        assert select_shard(paths, shard_id=0, num_shards=4) == [
            Path("/tmp/scenario_000.json"),
            Path("/tmp/scenario_004.json"),
        ]
        assert select_shard(paths, shard_id=1, num_shards=4) == [
            Path("/tmp/scenario_001.json"),
            Path("/tmp/scenario_005.json"),
        ]
        assert select_shard(paths, shard_id=3, num_shards=4) == [
            Path("/tmp/scenario_003.json"),
            Path("/tmp/scenario_007.json"),
        ]

    def test_result_is_independent_of_input_order(self):
        # Same paths, different input orderings should produce the same shard.
        sorted_paths = _paths(20)
        reversed_paths = list(reversed(sorted_paths))
        shuffled_paths = [
            sorted_paths[i]
            for i in [
                5,
                0,
                12,
                3,
                9,
                17,
                1,
                8,
                14,
                2,
                6,
                18,
                4,
                7,
                11,
                19,
                13,
                15,
                10,
                16,
            ]
        ]
        a = select_shard(sorted_paths, shard_id=2, num_shards=5)
        b = select_shard(reversed_paths, shard_id=2, num_shards=5)
        c = select_shard(shuffled_paths, shard_id=2, num_shards=5)
        assert a == b == c

    def test_uneven_split_distributes_remainder(self):
        # 10 paths into 3 shards -> 4, 3, 3.
        paths = _paths(10)
        shards = [select_shard(paths, shard_id=i, num_shards=3) for i in range(3)]
        sizes = [len(s) for s in shards]
        assert sorted(sizes) == [3, 3, 4]
        # Still a complete cover.
        assert sorted(p for s in shards for p in s) == sorted(paths)

    def test_empty_input_returns_empty(self):
        assert select_shard([], shard_id=0, num_shards=4) == []
        assert select_shard([], shard_id=3, num_shards=4) == []

    def test_more_shards_than_scenarios_gives_empty_for_high_ids(self):
        paths = _paths(3)
        # shard 0..2 each get 1, shard 3..4 get 0.
        assert len(select_shard(paths, shard_id=0, num_shards=5)) == 1
        assert len(select_shard(paths, shard_id=1, num_shards=5)) == 1
        assert len(select_shard(paths, shard_id=2, num_shards=5)) == 1
        assert select_shard(paths, shard_id=3, num_shards=5) == []
        assert select_shard(paths, shard_id=4, num_shards=5) == []

    def test_rejects_shard_id_out_of_range(self):
        paths = _paths(10)
        with pytest.raises(ValueError, match="shard_id"):
            select_shard(paths, shard_id=4, num_shards=4)
        with pytest.raises(ValueError, match="shard_id"):
            select_shard(paths, shard_id=-1, num_shards=4)

    def test_rejects_non_positive_num_shards(self):
        paths = _paths(10)
        with pytest.raises(ValueError, match="num_shards"):
            select_shard(paths, shard_id=0, num_shards=0)
        with pytest.raises(ValueError, match="num_shards"):
            select_shard(paths, shard_id=0, num_shards=-1)
