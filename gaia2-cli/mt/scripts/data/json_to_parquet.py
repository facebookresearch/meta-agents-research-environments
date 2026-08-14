#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Convert local JSON scenario files to HuggingFace-compatible parquet.

Usage:
    python scripts/json_to_parquet.py <json_dir> <parquet_dir> [--subsets s1 s2 ...]

Handles nested directories produced by ``manifold getr``.
Output matches the format produced by the translation pipeline
(one ``train-00000-of-00001.parquet`` per subset with a ``data`` column).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from datasets import Dataset


def convert(json_dir: Path, parquet_dir: Path, subsets: list[str]) -> None:
    # Handle nested directory from manifold getr
    if (json_dir / json_dir.name).is_dir():
        json_dir = json_dir / json_dir.name

    for subset in subsets:
        subset_dir = json_dir / subset
        if not subset_dir.is_dir():
            print(f"  Skipping {subset}: {subset_dir} not found")
            continue

        out_dir = parquet_dir / subset
        parquet_path = out_dir / "train-00000-of-00001.parquet"
        if parquet_path.exists():
            print(f"  {subset}: parquet already exists, skipping")
            continue

        rows = [{"data": f.read_text()} for f in sorted(subset_dir.glob("*.json"))]
        ds = Dataset.from_list(rows)
        out_dir.mkdir(parents=True, exist_ok=True)
        ds.to_parquet(str(parquet_path))
        print(f"  {subset}: {len(rows)} scenarios -> {parquet_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "json_dir", type=Path, help="Directory with JSON scenario files"
    )
    parser.add_argument("parquet_dir", type=Path, help="Output directory for parquets")
    parser.add_argument(
        "--subsets",
        nargs="+",
        default=["adaptability", "ambiguity", "execution", "search"],
        help="Subsets to convert (default: all four)",
    )
    args = parser.parse_args()
    convert(args.json_dir, args.parquet_dir, args.subsets)
