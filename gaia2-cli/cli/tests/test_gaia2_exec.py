# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
"""The setuid gaia2-exec wrapper must not let the agent steer the gaia2 tools.

The agent runs every Gaia2 CLI through ``shared/gaia2-exec.c``, which execs the
real tool as the gaia2 user.  Anything the agent puts in the environment and
the tool honours runs with gaia2's rights: ``GAIA2_FS_ROOT`` moved the
cloud-drive sandbox onto ``/var/gaia2`` (scenario JSON, judge context), and
``PYTHONPATH`` ran the agent's own modules as gaia2.

The wrapper is compiled from source with the tool directory pointed at a temp
dir (no setuid bit is needed: dropping to the effective ids is then a no-op).
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

SOURCE = Path(__file__).resolve().parents[2] / "shared" / "gaia2-exec.c"
CC = shutil.which("cc") or shutil.which("gcc")

pytestmark = pytest.mark.skipif(CC is None, reason="no C compiler")


@pytest.fixture
def wrapped_tool(tmp_path: Path) -> Path:
    """Build gaia2-exec against a fake ``cloud-drive`` that prints its env."""
    tool_dir = tmp_path / "bin"
    tool_dir.mkdir()
    tool = tool_dir / "cloud-drive"
    tool.write_text("#!/bin/sh\nexec env\n")
    tool.chmod(0o755)

    source = SOURCE.read_text()
    assert source.count('"/usr/local/bin/%s"') == 1
    patched = tmp_path / "gaia2-exec.c"
    patched.write_text(source.replace('"/usr/local/bin/%s"', f'"{tool_dir}/%s"'))
    binary = tmp_path / "gaia2-exec"
    subprocess.run([CC, "-O2", "-o", str(binary), str(patched)], check=True)

    link = tmp_path / "agent-bin" / "cloud-drive"
    link.parent.mkdir()
    link.symlink_to(binary)
    return link


def _run(link: Path, extra_env: dict[str, str]) -> dict[str, str]:
    env = {"PATH": "/usr/bin:/bin", "HOME": "/home/agent", **extra_env}
    out = subprocess.run(
        [str(link)], env=env, capture_output=True, text=True, check=True
    ).stdout
    return dict(line.split("=", 1) for line in out.splitlines() if "=" in line)


def test_caller_cannot_redirect_the_tool(wrapped_tool: Path) -> None:
    seen = _run(
        wrapped_tool,
        {
            "GAIA2_FS_ROOT": "/var/gaia2",
            "GAIA2_STATE_DIR": "/home/agent/fake_state",
            "GAIA2_EVENT_ID": "Event-ENV-forged",
            "GAIA2_JUDGE_CONTEXT": "/home/agent/x.json",
            "PYTHONPATH": "/home/agent/evil",
            "PYTHONSTARTUP": "/home/agent/evil.py",
            "PYTHONHOME": "/home/agent/py",
        },
    )
    assert seen["GAIA2_STATE_DIR"] == "/var/gaia2/state"
    assert seen["PYTHONNOUSERSITE"] == "1"
    for name in (
        "GAIA2_FS_ROOT",
        "GAIA2_EVENT_ID",
        "GAIA2_JUDGE_CONTEXT",
        "PYTHONPATH",
        "PYTHONSTARTUP",
        "PYTHONHOME",
    ):
        assert name not in seen, name


def test_harmless_variables_pass_through(wrapped_tool: Path) -> None:
    seen = _run(
        wrapped_tool,
        {"PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8", "LANG": "C.UTF-8"},
    )
    assert seen["PYTHONUNBUFFERED"] == "1"
    assert seen["PYTHONIOENCODING"] == "utf-8"
    assert seen["LANG"] == "C.UTF-8"
    assert seen["HOME"] == "/home/agent"


def test_unknown_tools_are_still_refused(wrapped_tool: Path, tmp_path: Path) -> None:
    other = tmp_path / "agent-bin" / "sh"
    other.symlink_to(os.readlink(wrapped_tool))
    proc = subprocess.run([str(other)], capture_output=True, text=True)
    assert proc.returncode == 1
    assert "not allowed" in proc.stderr
