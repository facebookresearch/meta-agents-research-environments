#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Convert translated parquet files back to individual JSON scenario files.

Usage:
    python scripts/parquet_to_json.py <parquet_dir> <json_dir> [--subsets s1 s2 ...]

Reads each row's ``data`` column (a JSON string) and writes it as a
pretty-printed JSON file, using the original filename from the source
dataset when available, or ``scenario_{idx}.json`` otherwise.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq


def convert(parquet_dir: Path, json_dir: Path, subsets: list[str]) -> None:
    for subset in subsets:
        subset_parquet_dir = parquet_dir / subset
        parquet_files = sorted(subset_parquet_dir.glob("*.parquet"))
        if not parquet_files:
            print(f"  Skipping {subset}: no parquet files in {subset_parquet_dir}")
            continue

        out_dir = json_dir / subset
        if out_dir.exists() and any(out_dir.glob("*.json")):
            print(f"  {subset}: JSON files already exist, skipping")
            continue

        out_dir.mkdir(parents=True, exist_ok=True)
        total = 0
        for pf in parquet_files:
            table = pq.read_table(str(pf))
            for _idx, data_val in enumerate(table.column("data")):
                scenario_json = json.loads(data_val.as_py())
                filename = f"scenario_{total:04d}.json"
                (out_dir / filename).write_text(
                    json.dumps(scenario_json, indent=2, ensure_ascii=False) + "\n"
                )
                total += 1

        print(f"  {subset}: {total} scenarios -> {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "parquet_dir", type=Path, help="Directory with subset/parquet structure"
    )
    parser.add_argument(
        "json_dir", type=Path, help="Output directory for JSON scenario files"
    )
    parser.add_argument(
        "--subsets",
        nargs="+",
        default=["adaptability", "ambiguity", "execution", "search"],
        help="Subsets to convert (default: all four)",
    )
    args = parser.parse_args()
    convert(args.parquet_dir, args.json_dir, args.subsets)
