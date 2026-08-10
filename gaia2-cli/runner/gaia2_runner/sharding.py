# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms of the Apache 2.0 license
# found in the LICENSE file in the root directory of this source tree.

"""Scenario sharding for parallel benchmark runs.

A single logical evaluation is split across K SLURM array tasks (or any K
independent runner processes); each task processes a disjoint slice of
scenarios. Aggregation happens after all shards finish via the
``aggregate`` subcommand (see :mod:`gaia2_runner.aggregate`).

The partition is round-robin on the sorted scenario paths — keeps load
balanced even if scenarios in a single split happen to have skewed runtime
(round-robin spreads any cluster of slow scenarios across all shards).
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence


def select_shard(
    scenario_paths: Sequence[Path],
    *,
    shard_id: int,
    num_shards: int,
) -> list[Path]:
    """Return the subset of ``scenario_paths`` belonging to shard ``shard_id``.

    Round-robin on the sorted paths so the assignment is deterministic and
    independent of the input order. ``num_shards=1, shard_id=0`` returns all
    paths unchanged.

    Args:
        scenario_paths: All paths to consider, in any order.
        shard_id: 0-based index of this shard. Must satisfy
            ``0 <= shard_id < num_shards``.
        num_shards: Total number of shards. Must be ``>= 1``.

    Raises:
        ValueError: if ``num_shards < 1`` or ``shard_id`` is out of range.
    """
    if num_shards < 1:
        raise ValueError(f"num_shards must be >= 1, got {num_shards}")
    if not 0 <= shard_id < num_shards:
        raise ValueError(f"shard_id must be in [0, {num_shards}), got {shard_id}")
    ordered = sorted(scenario_paths)
    return ordered[shard_id::num_shards]
