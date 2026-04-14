#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
"""Export the GAIA2 CLI HuggingFace dataset to individual scenario JSON files.

The benchmark dataset is stored in Parquet format on HuggingFace. This script
downloads it and exports each scenario as a standalone JSON file, organized by
split. The output is directly consumable by the runner via ``--dataset <dir>``
or ``dataset_root`` in a TOML config.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

DEFAULT_REPO_ID = "meta-agents-research-environments/gaia2-cli"
DEFAULT_DEST = Path.home() / "gaia2_datasets" / "gaia2-cli"
AVAILABLE_SPLITS = ("execution", "search", "ambiguity", "adaptability", "time")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download the GAIA2 CLI benchmark dataset from HuggingFace and "
            "export as individual scenario JSON files."
        )
    )
    parser.add_argument(
        "--repo-id",
        default=DEFAULT_REPO_ID,
        help=f"HuggingFace dataset repo (default: {DEFAULT_REPO_ID})",
    )
    parser.add_argument(
        "--dest",
        default=str(DEFAULT_DEST),
        help=f"Local destination directory (default: {DEFAULT_DEST})",
    )
    parser.add_argument(
        "--splits",
        default="all",
        help=(
            "Comma-separated splits to download, or 'all' "
            f"(default: all). Available: {', '.join(AVAILABLE_SPLITS)}"
        ),
    )
    parser.add_argument(
        "--token",
        default=None,
        help=(
            "Optional HuggingFace token. Defaults to the current HF login or "
            "HF_TOKEN environment variable."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing JSON files.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dest = Path(args.dest).expanduser()

    if args.splits == "all":
        splits = list(AVAILABLE_SPLITS)
    else:
        splits = [s.strip() for s in args.splits.split(",")]
        unknown = set(splits) - set(AVAILABLE_SPLITS)
        if unknown:
            print(f"error: unknown split(s): {', '.join(unknown)}", file=sys.stderr)
            print(f"available: {', '.join(AVAILABLE_SPLITS)}", file=sys.stderr)
            return 1

    try:
        from datasets import load_dataset
    except ImportError:
        print(
            "error: the 'datasets' package is required. Install it with:\n"
            "  pip install datasets",
            file=sys.stderr,
        )
        return 2

    import os

    token = args.token or os.environ.get("HF_TOKEN")

    print(f"Dataset: {args.repo_id}")
    print(f"Destination: {dest}")
    print(f"Splits: {', '.join(splits)}")
    print()

    total = 0
    for split in splits:
        split_dir = dest / split
        split_dir.mkdir(parents=True, exist_ok=True)

        ds = load_dataset(args.repo_id, split, split="test", token=token)

        written = 0
        skipped = 0
        for row in ds:
            scenario_id = row["scenario_id"]
            out_path = split_dir / f"{scenario_id}.json"

            if out_path.exists() and not args.force:
                skipped += 1
                continue

            out_path.write_text(row["scenario"])
            written += 1

        total += written + skipped
        msg = f"  {split}: {written} written"
        if skipped:
            msg += f", {skipped} skipped (use --force to overwrite)"
        print(msg)

    print()
    print(f"Dataset ready at: {dest}")
    print(f"Total scenarios: {total}")
    print()
    print("Usage with the runner:")
    print(f"  gaia2-runner run-dataset --dataset {dest}/search --image <image> ...")
    print()
    print("Or in a TOML config:")
    print(f'  dataset_root = "{dest}"')
    print('  splits = "all"')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
