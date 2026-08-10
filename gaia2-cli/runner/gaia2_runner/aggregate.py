# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms of the Apache 2.0 license
# found in the LICENSE file in the root directory of this source tree.

"""Aggregate the results of a sharded run.

When a benchmark is split across K shards (each writing to
``out_root/shard_{id:02d}_of_{n:02d}/``), this module merges their
``results.jsonl`` files into a top-level ``out_root/results.jsonl`` and
returns a small summary dict.

It does NOT regenerate the trace viewer HTML — for that, run
``gaia2-runner serve --output-dir <out_root>`` against the merged dir.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_SHARD_DIR_RE = re.compile(r"^shard_(\d+)_of_(\d+)$")


def aggregate_shards(out_root: Path) -> dict[str, Any]:
    """Merge ``out_root/shard_*_of_*/results.jsonl`` into ``out_root/results.jsonl``.

    Args:
        out_root: The parent directory containing per-shard subdirs.

    Returns:
        A summary dict with keys ``num_shards``, ``num_shards_with_results``,
        ``num_results``.

    Raises:
        FileNotFoundError: if no ``shard_*_of_*`` subdirs exist.
        ValueError: if the same ``scenario_id`` appears in more than one shard.
    """
    shard_dirs = sorted(
        p for p in out_root.iterdir() if p.is_dir() and _SHARD_DIR_RE.match(p.name)
    )
    if not shard_dirs:
        raise FileNotFoundError(f"no shard_*_of_* subdirs found under {out_root}")

    seen_ids: set[str] = set()
    merged_rows: list[dict[str, Any]] = []
    shards_with_results = 0

    for shard_dir in shard_dirs:
        results_file = shard_dir / "results.jsonl"
        if not results_file.exists():
            logger.warning("shard %s has no results.jsonl — skipping", shard_dir.name)
            continue
        shards_with_results += 1
        with results_file.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                scenario_id = row.get("scenario_id")
                if scenario_id is not None:
                    if scenario_id in seen_ids:
                        raise ValueError(
                            f"overlapping scenario_id {scenario_id!r} appears in more than one shard"
                        )
                    seen_ids.add(scenario_id)
                merged_rows.append(row)

    out_file = out_root / "results.jsonl"
    with out_file.open("w") as f:
        for row in merged_rows:
            f.write(json.dumps(row) + "\n")

    summary = {
        "num_shards": len(shard_dirs),
        "num_shards_with_results": shards_with_results,
        "num_results": len(merged_rows),
    }
    logger.info(
        "aggregated %d results from %d/%d shards into %s",
        summary["num_results"],
        summary["num_shards_with_results"],
        summary["num_shards"],
        out_file,
    )
    return summary
