# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""HuggingFace dataset download and materialization for the Gaia2 runner."""

from __future__ import annotations

import logging
import os
from pathlib import Path

from gaia2_runner.config import CANONICAL_SPLITS

logger = logging.getLogger(__name__)

_CACHE_DIR = Path(
    os.environ.get(
        "GAIA2_HF_CACHE",
        Path.home() / ".cache" / "gaia2" / "hf_datasets",
    )
)


def is_hf_dataset(dataset: str) -> bool:
    """Return True if *dataset* looks like a HuggingFace dataset ID (``org/name``).

    A HF dataset ID has exactly one ``/``, no path separators beyond that,
    and does not start with ``/``, ``~``, or ``.``.
    """
    if dataset.startswith(("/", "~", ".")):
        return False
    parts = dataset.split("/")
    return len(parts) == 2 and all(parts) and not Path(dataset).exists()


def download_hf_dataset(
    dataset_id: str,
    splits: list[str] | None = None,
    token: str | None = None,
) -> str:
    """Download a HuggingFace dataset and materialize it as scenario JSON files.

    Returns the path to a cache directory containing one subdirectory per
    split, each holding individual ``<scenario_id>.json`` files that the
    existing runner pipeline can consume directly.

    Results are cached under ``~/.cache/gaia2/hf_datasets/`` (override with
    ``$GAIA2_HF_CACHE``).  Subsequent runs reuse the cached JSON files
    without re-downloading or re-materializing.

    Parameters
    ----------
    dataset_id:
        HuggingFace dataset identifier, e.g.
        ``meta-agents-research-environments/gaia2-cli``.
    splits:
        List of split/config names to download.  ``None`` or ``["all"]``
        downloads all canonical splits.
    token:
        Optional HuggingFace API token.  Falls back to ``$HF_TOKEN``.
    """
    configs = splits or list(CANONICAL_SPLITS)
    if configs == ["all"]:
        configs = list(CANONICAL_SPLITS)

    cache_dir = _CACHE_DIR / dataset_id.replace("/", "_")

    # Return early if all requested splits are already cached.
    if cache_dir.exists() and all(
        (cache_dir / c).is_dir() and any((cache_dir / c).iterdir()) for c in configs
    ):
        logger.info("Using cached dataset at %s", cache_dir)
        return str(cache_dir)

    # Download and materialize.
    try:
        from datasets import load_dataset
    except ImportError:
        raise SystemExit(
            "The 'datasets' package is required for HuggingFace dataset support. "
            "Install it with: pip install datasets"
        )

    if token is None:
        token = os.environ.get("HF_TOKEN")

    cache_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading HF dataset %s (configs: %s)", dataset_id, configs)

    logging.getLogger("httpx").setLevel(logging.WARNING)

    for config_name in configs:
        split_dir = cache_dir / config_name
        if split_dir.is_dir() and any(split_dir.iterdir()):
            logger.info("  %s: cached", config_name)
            continue

        logger.info("  %s: downloading ...", config_name)
        split_dir.mkdir(exist_ok=True)
        ds = load_dataset(dataset_id, config_name, split="test", token=token)

        for row in ds:
            out_path = split_dir / f"{row['scenario_id']}.json"
            out_path.write_text(row["scenario"])

        logger.info("  %s: %d scenarios", config_name, len(ds))

    return str(cache_dir)
