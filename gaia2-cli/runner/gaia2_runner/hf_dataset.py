# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""HuggingFace dataset download and materialization for the Gaia2 runner."""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path

import click

from gaia2_runner.config import CANONICAL_SPLITS, MULTILINGUAL_SPLITS

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


def _available_config_names(dataset_id: str, token: str | None) -> str | None:
    """Best-effort listing of a dataset's HF config names, for error messages."""
    try:
        from huggingface_hub import get_dataset_config_names

        return ", ".join(get_dataset_config_names(dataset_id, token=token))
    except Exception:  # pragma: no cover - network/offline degradation
        return None


def download_hf_dataset(
    dataset_id: str,
    splits: list[str] | None = None,
    token: str | None = None,
    language: str | None = None,
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
        List of split names to download.  ``None`` or ``["all"]`` downloads
        every split the dataset provides.
    token:
        Optional HuggingFace API token.  Falls back to ``$HF_TOKEN``.
    language:
        Optional language code for datasets published with one config per
        language, e.g. ``facebook/omnilingual-gaia2``.  When set, the HF config
        loaded for each split is ``f"{language}_{split}"`` and the cache is
        keyed per language.  Split *directories* keep their bare names either
        way, which is what the runner's per-split reporting relies on.
    """
    available = MULTILINGUAL_SPLITS if language else CANONICAL_SPLITS
    split_names = list(splits) if splits else list(available)
    if split_names == ["all"]:
        split_names = list(available)

    def config_name_for(split: str) -> str:
        return f"{language}_{split}" if language else split

    cache_key = dataset_id.replace("/", "_")
    if language:
        cache_key = f"{cache_key}_{language.replace('/', '_')}"
    cache_dir = _CACHE_DIR / cache_key

    # Return early if all requested splits are already cached.
    if cache_dir.exists() and all(
        (cache_dir / s).is_dir() and any((cache_dir / s).iterdir()) for s in split_names
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
    logger.info(
        "Downloading HF dataset %s (configs: %s)",
        dataset_id,
        [config_name_for(s) for s in split_names],
    )

    logging.getLogger("httpx").setLevel(logging.WARNING)

    for split in split_names:
        # The directory is always the bare split name; only the HF config name
        # carries the language prefix.  _infer_result_split() reads the split
        # from the first path component under the dataset root.
        split_dir = cache_dir / split
        config_name = config_name_for(split)
        if split_dir.is_dir() and any(split_dir.iterdir()):
            logger.info("  %s: cached", config_name)
            continue

        logger.info("  %s: downloading ...", config_name)
        split_dir.mkdir(exist_ok=True)
        try:
            ds = load_dataset(dataset_id, config_name, split="test", token=token)

            for row in ds:
                out_path = split_dir / f"{row['scenario_id']}.json"
                out_path.write_text(row["scenario"])
        except BaseException as exc:
            # The directory was created before the download, so leaving a
            # partial one behind would make the next run's cache check pass and
            # silently reuse an incomplete split. BaseException rather than
            # Exception so a Ctrl-C mid-download cleans up too.
            shutil.rmtree(split_dir, ignore_errors=True)
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            message = (
                f"Failed to load HuggingFace config {config_name!r} from "
                f"{dataset_id}: {exc}"
            )
            configs = _available_config_names(dataset_id, token)
            if configs:
                message += f"\nAvailable configs: {configs}"
            if language:
                message += "\n(check [target].language / --language)"
            raise click.UsageError(message) from exc

        logger.info("  %s: %d scenarios", config_name, len(ds))

    return str(cache_dir)
