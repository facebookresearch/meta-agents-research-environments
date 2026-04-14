# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
"""Tests for the lightweight scenario loader in gaia2-core."""

from __future__ import annotations

import json
from pathlib import Path

from gaia2_core.loader import ScenarioLoader


def _write_scenario(tmp_path: Path, apps: list[dict]) -> Path:
    scenario_path = tmp_path / "scenario.json"
    scenario_path.write_text(json.dumps({"apps": apps, "events": []}))
    return scenario_path


def test_extract_user_details_from_internal_contacts(tmp_path: Path) -> None:
    scenario_path = _write_scenario(
        tmp_path,
        [
            {
                "name": "InternalContacts",
                "class_name": "InternalContacts",
                "app_state": {
                    "contacts": {
                        "user": {
                            "first_name": "Astrid",
                            "last_name": "Lundqvist",
                            "address": "Ostermalmsgatan 12",
                            "is_user": True,
                        }
                    }
                },
            }
        ],
    )

    loader = ScenarioLoader(str(scenario_path))
    user_details = loader._extract_user_details()

    assert user_details is not None
    assert user_details.first_name == "Astrid"
    assert user_details.last_name == "Lundqvist"
    assert user_details.address == "Ostermalmsgatan 12"
