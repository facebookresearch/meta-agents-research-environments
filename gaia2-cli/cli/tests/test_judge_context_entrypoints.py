# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
"""The init entrypoints install the judge context privately for gaia2-eventd.

The launcher mounts ``<scenario stem>.judge_context.json`` read-only at
``/var/gaia2/judge_context.staged.json``; each image's init entrypoint (run as
root) must install a gaia2-owned ``0600`` copy where gaia2-eventd reads it, and
the agent images must refuse to start when the agent could read the staged
mount.  The install step is cut out of the real scripts and run under bash
with ``su`` / ``install`` stubbed, so no container is needed.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import pytest

CONTAINERS = Path(__file__).resolve().parents[2] / "containers"
AGENT_IMAGES = ("openclaw", "hermes", "mini-swe")
ALL_IMAGES = (*AGENT_IMAGES, "oracle")


def _script(image: str) -> str:
    return (CONTAINERS / image / "gaia2-init-entrypoint.sh").read_text()


def _install_step(image: str) -> str:
    """Return the shell that installs the judge context in *image*'s entrypoint."""
    text = _script(image)
    if image == "openclaw":
        m = re.search(r"^install_judge_context\(\) \{\n.*?^\}\n", text, re.M | re.S)
        assert m, "install_judge_context() not found"
        return m.group(0) + "install_judge_context\n"
    m = re.search(
        r"^# ── 1b\. Install the judge-only task text.*?(?=^# ── 2\.)",
        text,
        re.M | re.S,
    )
    assert m, "judge context install block not found"
    return m.group(0)


def _run(image: str, tmp_path: Path, *, staged: bool, agent_can_read: bool):
    staged_path = tmp_path / "judge_context.staged.json"
    if staged:
        staged_path.write_text('{"version": 1, "turns": {}}')
    log = tmp_path / "calls.log"
    stubs = f"""
su() {{ echo "su $*" >> '{log}'; [ -n "$AGENT_CAN_READ" ]; }}
install() {{ echo "install $*" >> '{log}'; }}
set -eo pipefail
JUDGE_CONTEXT_STAGED='{staged_path}'
JUDGE_CONTEXT_PATH='{tmp_path}/judge_context.json'
"""
    env = {**os.environ, "AGENT_CAN_READ": "1" if agent_can_read else ""}
    proc = subprocess.run(
        ["bash", "-c", stubs + _install_step(image) + "\necho REACHED-END\n"],
        capture_output=True,
        text=True,
        env=env,
    )
    calls = log.read_text().splitlines() if log.exists() else []
    return proc, calls, staged_path


@pytest.mark.parametrize("image", ALL_IMAGES)
def test_defaults_and_eventd_env(image: str) -> None:
    text = _script(image)
    assert "JUDGE_CONTEXT_STAGED=/var/gaia2/judge_context.staged.json\n" in text
    assert (
        'JUDGE_CONTEXT_PATH="${GAIA2_JUDGE_CONTEXT:-/var/gaia2/judge_context.json}"\n'
        in text
    )
    # gaia2-eventd is told where the private copy is.
    eventd_block = text[text.index("/usr/local/bin/gaia2-eventd") - 600 :]
    assert "export GAIA2_JUDGE_CONTEXT='$JUDGE_CONTEXT_PATH'" in eventd_block[:700]


@pytest.mark.parametrize("image", ALL_IMAGES)
def test_installs_a_private_gaia2_copy(image: str, tmp_path: Path) -> None:
    proc, calls, staged = _run(image, tmp_path, staged=True, agent_can_read=False)
    assert proc.returncode == 0, proc.stderr
    assert "REACHED-END" in proc.stdout
    assert (
        f"install -o gaia2 -g gaia2 -m 0600 {staged} {tmp_path}/judge_context.json"
        in calls
    )


@pytest.mark.parametrize("image", ALL_IMAGES)
def test_nothing_happens_without_a_judge_context(image: str, tmp_path: Path) -> None:
    proc, calls, _ = _run(image, tmp_path, staged=False, agent_can_read=True)
    assert proc.returncode == 0, proc.stderr
    assert "REACHED-END" in proc.stdout
    assert calls == []


@pytest.mark.parametrize("image", AGENT_IMAGES)
def test_refuses_to_start_when_the_agent_can_read_the_staged_copy(
    image: str, tmp_path: Path
) -> None:
    proc, calls, staged = _run(image, tmp_path, staged=True, agent_can_read=True)
    assert proc.returncode == 1
    assert "REACHED-END" not in proc.stdout
    assert "the agent user can read" in proc.stderr
    assert f"su -s /usr/bin/bash agent -c test -r '{staged}'" in calls
    assert not any(c.startswith("install ") for c in calls)


def test_openclaw_installs_after_locking_down_and_before_eventd() -> None:
    main = _script("openclaw").split("\nmain() {\n", 1)[1].split("\n}\n", 1)[0]
    steps = [line.strip() for line in main.splitlines() if line.strip()]
    assert steps.index("init_state") < steps.index("install_judge_context")
    assert steps.index("install_judge_context") < steps.index("start_eventd")


@pytest.mark.parametrize("image", ("hermes", "mini-swe", "oracle"))
def test_inline_install_runs_after_init_and_before_eventd(image: str) -> None:
    text = _script(image)
    install = text.index("# ── 1b. Install the judge-only task text")
    assert text.index("chown -R gaia2:gaia2 /var/gaia2/state") < install
    assert install < text.index("/usr/local/bin/gaia2-eventd")
