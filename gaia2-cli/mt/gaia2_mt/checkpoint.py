# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Checkpoint infrastructure for resumable pipeline execution.

Saves and loads intermediate pipeline results as JSON files so that
expensive LLM-based steps can be skipped on re-runs.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any


logger: logging.Logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Tuple-key dict serialization
# ─────────────────────────────────────────────────────────────────────────────


def serialize_tuple_key_dict(d: dict[tuple, str]) -> list[dict]:
    """Convert ``{(0, 1, "content"): "val"}`` → ``[{"key": [0, 1, "content"], "value": "val"}]``."""
    return [{"key": list(k), "value": v} for k, v in d.items()]


def deserialize_tuple_key_dict(entries: list[dict]) -> dict[tuple, str]:
    """Reverse of :func:`serialize_tuple_key_dict`."""
    return {tuple(e["key"]): e["value"] for e in entries}


def serialize_nested_tuple_key_dict(
    d: dict[str, dict[tuple, str]],
) -> dict[str, list[dict]]:
    """Serialize ``{hash: {(app_idx, *path): val}}`` for universe translations."""
    return {k: serialize_tuple_key_dict(v) for k, v in d.items()}


def deserialize_nested_tuple_key_dict(
    d: dict[str, list[dict]],
) -> dict[str, dict[tuple, str]]:
    """Reverse of :func:`serialize_nested_tuple_key_dict`."""
    return {k: deserialize_tuple_key_dict(v) for k, v in d.items()}


# ─────────────────────────────────────────────────────────────────────────────
# CheckpointManager
# ─────────────────────────────────────────────────────────────────────────────


class CheckpointManager:
    """Save and load intermediate pipeline results as JSON files.

    Directory layout::

        base_dir/
          subset/
            universe_fields.json          # global steps (no split_name)
            universe_translations.json
            split_name/
              translated_prompts.json     # per-split steps
              prompt_reviews.json
              oracle_args_translated.json
              oracle_args_reviewed.json
    """

    def __init__(self, base_dir: Path, subset: str) -> None:
        self._root = base_dir / subset
        self._root.mkdir(parents=True, exist_ok=True)

    def _path(self, step_name: str, split_name: str | None = None) -> Path:
        if split_name:
            d = self._root / split_name
            d.mkdir(parents=True, exist_ok=True)
            return d / f"{step_name}.json"
        return self._root / f"{step_name}.json"

    def exists(self, step_name: str, split_name: str | None = None) -> bool:
        return self._path(step_name, split_name).exists()

    def save(self, step_name: str, data: Any, split_name: str | None = None) -> None:
        path = self._path(step_name, split_name)
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2))
        label = f"{split_name}/{step_name}" if split_name else step_name
        logger.info(f"Saved checkpoint: {label} → {path}")

    def load(self, step_name: str, split_name: str | None = None) -> Any | None:
        path = self._path(step_name, split_name)
        if not path.exists():
            return None
        label = f"{split_name}/{step_name}" if split_name else step_name
        logger.info(f"Resuming: loaded '{label}' from checkpoint {path}")
        return json.loads(path.read_text())
