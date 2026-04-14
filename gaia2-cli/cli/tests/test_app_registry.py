# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
"""Tests for the gaia2_cli app registry."""

import json

from gaia2_cli.app_registry import (
    APP_NAME_TO_STATE,
    APP_TO_CLI,
    resolve_scenario_tools,
)


def test_legacy_tree_and_sqlite_aliases_removed() -> None:
    removed_aliases = {
        "CalendarTreeApp",
        "CalendarSqliteApp",
        "ContactsTreeApp",
        "ContactSqliteApp",
        "EmailClientTreeApp",
        "EmailClientSqliteApp",
        "EmailClientV2SqliteApp",
        "MessagingTreeApp",
        "MessagingSqliteApp",
        "MessagingTreeAppV2",
        "MessagingV2SqliteApp",
        "ApartmentListingTreeApp",
        "ApartmentSqliteApp",
        "CabTreeApp",
        "RideSqliteApp",
    }

    for alias in removed_aliases:
        assert alias not in APP_NAME_TO_STATE
        assert alias not in APP_TO_CLI


def test_public_messaging_tools_resolve(tmp_path) -> None:
    scenario_path = tmp_path / "scenario.json"
    scenario_path.write_text(
        json.dumps(
            {
                "apps": [
                    {"name": "Messages", "class_name": "MessagingAppV2"},
                    {"name": "Chats", "class_name": "MessagingAppV2"},
                ]
            }
        )
    )

    result = resolve_scenario_tools(str(scenario_path))

    assert result["tools"] == {
        "messages": "Messages",
        "chats": "Chat conversations",
    }
