# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Helpers for loading runner env files via ``python-dotenv``."""

from __future__ import annotations

import os
from pathlib import Path

import click

_BLOCKED_DOTENV_KEYS = {
    "GAIA2_JUDGE_PROVIDER",
    "GAIA2_JUDGE_MODEL",
    "GAIA2_JUDGE_BASE_URL",
    "GAIA2_JUDGE_API_KEY",
}


def load_env_file(path: str | Path, *, override: bool = False) -> list[str]:
    """Load dotenv-style assignments into ``os.environ``.

    Returns the list of env var names written to the process environment.
    Existing values are preserved unless ``override`` is true.
    """
    try:
        from dotenv import dotenv_values
    except ImportError as exc:
        raise click.ClickException(
            "python-dotenv is required for .env loading. Install gaia2-runner dependencies."
        ) from exc

    env_path = Path(path).expanduser().resolve()
    loaded_keys: list[str] = []
    parsed = dotenv_values(env_path)
    ignored_keys: list[str] = []

    for key in parsed:
        if key is None:
            continue
        if key in _BLOCKED_DOTENV_KEYS:
            ignored_keys.append(key)
            continue
        existing_value = os.environ.get(key)
        if not override and existing_value not in (None, ""):
            continue

        value = parsed[key]
        os.environ[key] = "" if value is None else value
        loaded_keys.append(key)

    if ignored_keys:
        click.secho(
            "Ignoring judge defaults from .env; keep judge config in TOML or CLI: "
            + ", ".join(sorted(ignored_keys)),
            fg="yellow",
            err=True,
        )

    return loaded_keys
