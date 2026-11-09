# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
"""Tests for the Time-scenario section of render_agent_prompt.py."""

import json
import sys
from pathlib import Path

import pytest

_SHARED = Path(__file__).resolve().parents[2] / "shared"
sys.path.insert(0, str(_SHARED))
TEMPLATE = _SHARED / "AGENTS_TEMPLATE.md"

from render_agent_prompt import _TIME_SECTION, render  # noqa: E402


def _render(tmp_path, metadata):
    scenario = {"apps": [{"name": "Contacts", "class_name": "Contacts"}]}
    scenario["metadata"] = metadata
    path = tmp_path / "scenario.json"
    path.write_text(json.dumps(scenario))
    return render(str(path), str(TEMPLATE))


def _base_prompt():
    return (
        TEMPLATE.read_text()
        .replace("{{TOOL_LIST}}", "- `contacts` — Contacts")
        .replace("{{MESSAGING_TOOLS}}", "messages, chats")
        .replace("{{EXEC_TOOL}}", "exec")
    )


@pytest.mark.parametrize(
    "metadata", [None, {"definition": {"tags": ["Universe", "Adaptability"]}}]
)
def test_other_scenarios_render_unchanged(tmp_path, metadata) -> None:
    assert _render(tmp_path, metadata) == _base_prompt()


def test_time_scenarios_get_the_section(tmp_path) -> None:
    metadata = {"definition": {"tags": ["Universe", "Time"]}}
    assert _render(tmp_path, metadata) == _base_prompt() + _TIME_SECTION
