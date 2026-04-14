# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
"""Shared fixtures and helpers for Gaia2 CLI unit tests."""

import json
from pathlib import Path

import pytest
from click.testing import CliRunner, Result

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def cli_env(tmp_path, monkeypatch):
    """Set up GAIA2_STATE_DIR pointing to tmp_path and create events.jsonl.

    Returns (state_dir: Path, runner: CliRunner).
    """
    monkeypatch.setenv("GAIA2_STATE_DIR", str(tmp_path))
    (tmp_path / "events.jsonl").touch()
    return tmp_path, make_cli_runner()


@pytest.fixture
def fixed_uuid(monkeypatch):
    """Patch uuid.uuid4() to return deterministic hex IDs.

    Returns a callable ``get_id(n)`` that gives the hex for call *n* (0-based).
    """
    call_count = {"n": 0}

    class _FakeUUID:
        def __init__(self, n):
            self._hex = f"{n:032x}"

        @property
        def hex(self):
            return self._hex

    def _fake_uuid4():
        n = call_count["n"]
        call_count["n"] += 1
        return _FakeUUID(n)

    monkeypatch.setattr("uuid.uuid4", _fake_uuid4)

    def get_id(n: int) -> str:
        return f"{n:032x}"

    return get_id


@pytest.fixture
def fixed_time(monkeypatch):
    """Patch time.time() to return a controllable value.

    Returns a callable ``set_time(ts)`` to change the returned timestamp.
    """
    current = {"ts": 1522479600.0}  # 2018-03-31 09:00:00 UTC

    def _fake_time():
        return current["ts"]

    monkeypatch.setattr("time.time", _fake_time)

    def set_time(ts: float) -> None:
        current["ts"] = ts

    return set_time


# ---------------------------------------------------------------------------
# Helper functions (not fixtures — import or call directly)
# ---------------------------------------------------------------------------


def seed_state(state_dir: Path, app_name: str, state: dict) -> None:
    """Write a state JSON file for an app into state_dir."""
    from gaia2_cli.shared import normalize_app_name

    filename = normalize_app_name(app_name) + ".json"
    path = state_dir / filename
    with open(path, "w") as f:
        json.dump(state, f, indent=2)
        f.write("\n")


def make_cli_runner() -> CliRunner:
    """Create a Click test runner.

    We intentionally use Click's default mixed-output behavior because
    Click 8.1.x drops stderr text when ``mix_stderr=False`` under
    ``CliRunner.invoke``. The compatibility shim below makes
    ``result.stderr`` behave sensibly across Click versions.
    """
    return CliRunner()


def _compat_result_stderr(self: Result) -> str:
    """Return stderr text across Click versions.

    Click >= 8.2 always exposes ``stderr``. Click 8.1.x raises when stderr
    wasn't separately captured, even though the user-visible mixed output
    still contains the error text. In that older case, fall back to
    ``result.output`` so existing assertions remain meaningful.
    """
    stderr_bytes = getattr(self, "stderr_bytes", None)
    if stderr_bytes is None:
        return self.output
    return stderr_bytes.decode(self.runner.charset, "replace").replace("\r\n", "\n")


Result.stderr = property(_compat_result_stderr)


def read_events(state_dir: Path) -> list[dict]:
    """Read all events from events.jsonl, returning a list of dicts."""
    events_path = state_dir / "events.jsonl"
    if not events_path.exists():
        return []
    lines = events_path.read_text().strip().splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def parse_output(result) -> dict:
    """Parse JSON from a CliRunner result's stdout."""
    return json.loads(result.output)


def assert_event(event: dict, app: str, fn: str, write: bool) -> None:
    """Assert the essential fields of an event dict."""
    assert event["app"] == app, f"Expected app={app}, got {event['app']}"
    assert event["fn"] == fn, f"Expected fn={fn}, got {event['fn']}"
    assert event["w"] == write, f"Expected w={write}, got {event['w']}"
    assert "t" in event
    assert "args" in event
    assert "ret" in event
