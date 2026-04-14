# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
"""
Shared state I/O, event logging, and JSON helpers for standalone Gaia2 CLI tools.
"""

import fcntl
import json
import os
import re
import time
from dataclasses import fields, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Module-level app name, set once per CLI module via set_app()
# ---------------------------------------------------------------------------

_APP_NAME: str = ""
_LOG_CLASS: str = ""
_APP_CLASS_MAP: dict[str, str] | None = None


def set_app(name: str) -> None:
    """Set the app name and auto-resolve the scenario class_name.

    Reads ``app_class_map.json`` (written by ``gaia2-init``) to map the CLI
    module's canonical ``APP_NAME`` (e.g. ``Chats``) to the scenario's actual
    ``class_name`` (e.g. ``MessagingAppV2``).  Falls back to empty string if no
    map file exists (standalone CLI usage).
    """
    global _APP_NAME, _LOG_CLASS
    _APP_NAME = name
    _LOG_CLASS = _get_app_class_map().get(name, "")


def set_log_class(name: str) -> None:
    """Set the scenario class_name for events.jsonl (for judge matching).

    Prefer relying on ``set_app()`` which auto-resolves from
    ``app_class_map.json``.  This function is kept for backwards compatibility
    and is a no-op when the map already resolved a class name.
    """
    global _LOG_CLASS
    if not _LOG_CLASS:
        _LOG_CLASS = name


def _get_app_class_map() -> dict[str, str]:
    """Lazy-load app_class_map.json from the state directory.

    This file is written by ``gaia2-init`` and maps each CLI module's
    ``APP_NAME`` (e.g. ``Chats``) to the scenario's actual ``class_name``
    (e.g. ``MessagingAppV2``).  Loaded once per process and cached.
    """
    global _APP_CLASS_MAP
    if _APP_CLASS_MAP is None:
        try:
            with open(get_state_dir() / "app_class_map.json") as f:
                _APP_CLASS_MAP = json.load(f)
        except Exception:
            _APP_CLASS_MAP = {}
    return _APP_CLASS_MAP


# ---------------------------------------------------------------------------
# State directory helpers
# ---------------------------------------------------------------------------


def get_state_dir() -> Path:
    """Return the state directory from GAIA2_STATE_DIR env var."""
    state_dir = os.environ.get("GAIA2_STATE_DIR")
    if not state_dir:
        raise RuntimeError(
            "GAIA2_STATE_DIR environment variable is not set. "
            "Set it to the directory containing per-app state files."
        )
    return Path(state_dir)


def state_file_for_app(app_name: str, state_dir: Path | None = None) -> Path:
    """Return the path to the state JSON file for a given app."""
    if state_dir is None:
        state_dir = get_state_dir()
    filename = normalize_app_name(app_name) + ".json"
    return state_dir / filename


def events_file(state_dir: Path | None = None) -> Path:
    """Return the path to events.jsonl."""
    if state_dir is None:
        state_dir = get_state_dir()
    return state_dir / "events.jsonl"


def normalize_app_name(name: str) -> str:
    """Convert app name to a filesystem-safe lowercase form.

    Examples:
        Calendar -> calendar
        RentAFlat -> rent_a_flat
        MessagingAppV2 -> messaging_app_v2
        Chats -> chats
        CabApp -> cab_app
    """
    result = []
    for i, ch in enumerate(name):
        if ch.isupper() and i > 0:
            prev = name[i - 1]
            if prev.islower():
                result.append("_")
            elif prev.isupper() and i + 1 < len(name) and name[i + 1].islower():
                result.append("_")
        result.append(ch.lower())
    return "".join(result)


# ---------------------------------------------------------------------------
# Cloud-drive path resolution
# ---------------------------------------------------------------------------


def resolve_sandbox_path(path: str) -> str:
    """Resolve a cloud-drive path to a real filesystem path.

    Always resolves through the Gaia2 filesystem sandbox rooted at
    ``$GAIA2_FS_ROOT`` (or ``$GAIA2_STATE_DIR/filesystem/``).
    Mirrors the ``_validate_path`` logic in ``files_app.py``.
    """
    state_dir = os.environ.get("GAIA2_STATE_DIR", "/workspace/state")
    fs_root = os.environ.get("GAIA2_FS_ROOT", os.path.join(state_dir, "filesystem"))
    fs_root = os.path.abspath(fs_root)

    rel = re.sub(r"^~/?", "home/userhome/", path)
    rel = re.sub(r"^~([^/]+)/?", r"home/\1/", rel)
    rel = rel.lstrip("/")

    resolved = os.path.abspath(os.path.join(fs_root, rel))

    if not (resolved == fs_root or resolved.startswith(fs_root + os.sep)):
        raise ValueError(f"Path escapes sandbox: {path}")

    return resolved


# ---------------------------------------------------------------------------
# Input validation helpers
# ---------------------------------------------------------------------------


def validate_email_address(value: str, param_name: str = "recipient") -> None:
    """Validate that a string looks like an email address.

    Raises ValueError with a helpful message directing the agent to look up
    the contact's email address first.
    """
    if "@" not in value:
        raise ValueError(f"'{value}' is not a valid email address for {param_name}.")


def validate_email_list(values: list[str], param_name: str = "recipients") -> None:
    """Validate that every item in a list looks like an email address."""
    for v in values:
        validate_email_address(v, param_name)


# ---------------------------------------------------------------------------
# State I/O
# ---------------------------------------------------------------------------


def load_app_state(app_name: str, state_dir: Path | None = None) -> dict[str, Any]:
    """Load app state from its JSON file with an exclusive file lock.

    The lock is held until ``save_app_state`` is called for the same app,
    or until the process exits (whichever comes first).  This serializes
    the entire read-modify-write cycle across concurrent CLI processes.

    In production each CLI invocation is a short-lived subprocess, so the
    lock is automatically released on process exit even if ``save_app_state``
    is never called (read-only operations).
    """

    path = state_file_for_app(app_name, state_dir)

    # Release any previously held lock for this path (re-entrant safety).
    # In production each CLI call is a separate process so this never fires.
    # In tests, multiple load/save cycles happen in the same process.
    _release_lock(path)

    lock_path = path.with_suffix(".lock")
    lock_fd = open(lock_path, "w")
    fcntl.flock(lock_fd, fcntl.LOCK_EX)
    _active_locks[str(path)] = lock_fd

    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def save_app_state(
    app_name: str, state: dict[str, Any], state_dir: Path | None = None
) -> None:
    """Write app state atomically (temp + rename) and release the file lock.

    Must be called after ``load_app_state`` for the same app.  The exclusive
    lock acquired by ``load_app_state`` ensures that no other process can
    read stale data between our load and save.
    """
    path = state_file_for_app(app_name, state_dir)
    tmp_path = path.with_suffix(".tmp")
    with open(tmp_path, "w") as f:
        json.dump(make_serializable(state), f, indent=2)
        f.write("\n")
    tmp_path.rename(path)
    _release_lock(path)


# Active lock file descriptors — auto-released on process exit.
_active_locks: dict[str, Any] = {}


def _release_lock(path: Path) -> None:
    """Release the file lock for the given state file, if held."""

    key = str(path)
    lock_fd = _active_locks.pop(key, None)
    if lock_fd is not None:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        lock_fd.close()


# ---------------------------------------------------------------------------
# Event logging (Phase 2 minimal format)
# ---------------------------------------------------------------------------


def log_action(fn: str, args: dict, ret: Any = None, write: bool = False) -> None:
    """Log a successful action to events.jsonl (minimal format).

    The grader-side resource converts these entries to full CompletedEvent dicts.

    If the ``GAIA2_EVENT_ID`` environment variable is set, its value is included
    as ``event_id`` in the entry.  Replay and the daemon set this variable
    before each CLI subprocess call so that return values can be looked up by
    oracle event ID directly from events.jsonl.
    """
    entry = {
        "t": time.time(),
        "app": _LOG_CLASS or _APP_NAME,
        "fn": fn,
        "args": make_serializable(args),
        "w": write,
        "ret": make_serializable(ret),
    }
    event_id = os.environ.get("GAIA2_EVENT_ID")
    if event_id:
        entry["event_id"] = event_id
    # Include simulated time from faketime.rc if available.
    # File may not exist (non-faketime containers) — skip silently.
    # File should never be empty (daemon uses atomic write) — warn if it is.
    try:
        with open("/tmp/faketime.rc") as f:
            sim_t = f.read().strip()
        if sim_t:
            entry["sim_t"] = sim_t
        else:
            import logging

            logging.getLogger(__name__).warning("faketime.rc exists but is empty")
    except FileNotFoundError:
        pass
    line = json.dumps(entry) + "\n"
    ef = events_file()
    lock_path = ef.with_suffix(".lock")
    with open(lock_path, "w") as lock_fd:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        try:
            with open(ef, "a") as f:
                f.write(line)
        finally:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------


def make_serializable(value: Any) -> Any:
    """Convert data to a JSON-safe format.

    Handles dataclasses, Enums, lists, dicts, and primitive types.
    For unsupported types, removes non-deterministic memory addresses
    from the string representation.
    """
    if is_dataclass(value) and not isinstance(value, type):
        return {
            f.name: make_serializable(getattr(value, f.name)) for f in fields(value)
        }
    elif isinstance(value, Enum):
        return value.value
    elif isinstance(value, list):
        return [make_serializable(item) for item in value]
    elif isinstance(value, tuple):
        return [make_serializable(item) for item in value]
    elif isinstance(value, dict):
        return {str(k): make_serializable(v) for k, v in value.items()}
    elif isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    elif type(value) in (str, int, float, bool) or value is None:
        return value
    else:
        stringified: str = re.sub(r" at 0x[a-f0-9]{8,16}", "", str(value))
        return stringified


def json_output(data: Any) -> None:
    """Print data as formatted JSON to stdout."""
    print(json.dumps(make_serializable(data), indent=2))


# ---------------------------------------------------------------------------
# Compatibility helpers used by some app modules
# ---------------------------------------------------------------------------


def cli_error(msg: str) -> None:
    """Print error to stderr and exit with code 1."""
    import sys

    print(msg, file=sys.stderr)
    sys.exit(1)


def build_schema(group: Any) -> list[dict[str, Any]]:
    """Build a machine-readable schema list from a Click group."""
    import click

    schema_data = []
    for name, cmd in group.commands.items():
        if name == "schema" or getattr(cmd, "hidden", False):
            continue
        # callback_name is the Python function name (e.g. add_calendar_event),
        # which matches the Gaia2 oracle function name and log_action() name.
        # "function" is the kebab→snake of the CLI command for backwards compat.
        callback_name = (
            cmd.callback.__name__ if cmd.callback else name.replace("-", "_")
        )
        tool_schema: dict[str, Any] = {
            "command": name,
            "function": name.replace("-", "_"),
            "oracle_function": callback_name,
            "description": cmd.help or "",
            "parameters": [],
        }
        for param in cmd.params:
            if not isinstance(param, click.Option):
                continue
            param_info: dict[str, Any] = {
                "name": param.human_readable_name,
                "type": (
                    param.type.name if hasattr(param.type, "name") else str(param.type)
                ),
                "description": getattr(param, "help", "") or "",
                "required": param.required,
            }
            if param.default is not None:
                param_info["default"] = make_serializable(param.default)
            tool_schema["parameters"].append(param_info)
        schema_data.append(tool_schema)
    return schema_data
