# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Gaia2 app registry — single source of truth for app names, CLI commands,
aliases, descriptions, and notification formatters.

Used by:
- ``eventd.py`` — CLI mapping, notification formatting
- ``init_cmd.py`` — scenario app name → state file resolution
- ``render_agent_prompt.py`` — agent system prompt generation
"""

from __future__ import annotations

import json
from typing import Any, Callable

# ---------------------------------------------------------------------------
# App Registry
# ---------------------------------------------------------------------------
# To add a new app: add one entry here. Everything else is derived.
#
# Fields:
#   canonical    — authoritative Gaia2 app class name (e.g. "CalendarApp")
#   cli          — real installed binary name (e.g. "calendar")
#   agent_cli    — name the agent sees and types; may be a symlink to cli
#   module       — Python module path for the CLI app
#   description  — human-readable description shown in agent system prompt
#   state_name   — APP_NAME constant in the CLI module (determines state file name)
#   aliases      — alternative Gaia2 class names that map to this entry
#   formatters   — ENV notification formatters {fn_name: lambda args -> str}

APP_REGISTRY: list[dict[str, Any]] = [
    {
        "canonical": "CalendarApp",
        "cli": "calendar",
        "agent_cli": "calendar",
        "module": "gaia2_cli.apps.calendar",
        "description": "Calendar events",
        "state_name": "Calendar",
        "aliases": ["Calendar"],
        "formatters": {
            "add_calendar_event_by_attendee": lambda a: (
                f"New calendar event added by {a.get('who_add', 'unknown')}"
            ),
            "delete_calendar_event_by_attendee": lambda a: (
                f"Calendar event deleted by {a.get('who_delete', 'unknown')}"
            ),
        },
    },
    {
        "canonical": "ContactsApp",
        "cli": "contacts",
        "agent_cli": "contacts",
        "module": "gaia2_cli.apps.contacts",
        "description": "Contacts",
        "state_name": "Contacts",
        "aliases": ["Contacts"],
    },
    {
        "canonical": "EmailClientV2",
        "cli": "emails",
        "agent_cli": "emails",
        "module": "gaia2_cli.apps.email",
        "description": "Emails",
        "state_name": "EmailClientV2",
        "aliases": [
            "EmailClientApp",
            "Mail",
            "Emails",
        ],
        "formatters": {
            "create_and_add_email": lambda a: (
                f"New email received from {a.get('sender', 'unknown')}"
            ),
            "send_email_to_user_only": lambda a: (
                f"New email received from {a.get('sender', 'unknown')}"
            ),
            "reply_to_email_from_user": lambda a: "New email received",
        },
    },
    {
        "canonical": "MessagingAppV2",
        "cli": "messages",
        "agent_cli": "messages",
        "module": "gaia2_cli.apps.messages",
        "description": "Messages",
        "state_name": "Messages",
        "aliases": ["Messages"],
        "formatters": {
            "create_and_add_message": lambda a: (
                f"New message received in conversation {a.get('conversation_id', '')}"
            ),
            "add_participant_to_conversation": lambda a: (
                f"New participant added to conversation {a.get('conversation_id', '')}"
            ),
        },
    },
    {
        "canonical": "Chats",
        "cli": "chats",
        "agent_cli": "chats",
        "module": "gaia2_cli.apps.chats",
        "description": "Chat conversations",
        "state_name": "Chats",
        "aliases": [],
        "formatters": {
            "create_and_add_message": lambda a: (
                f"New message received in conversation {a.get('conversation_id', '')}"
            ),
            "add_participant_to_conversation": lambda a: (
                f"New participant added to chat conversation {a.get('conversation_id', '')}"
            ),
        },
    },
    {
        "canonical": "ApartmentListingApp",
        "cli": "rent-a-flat",
        "agent_cli": "rent-a-flat",
        "module": "gaia2_cli.apps.apartment",
        "description": "Apartment listings",
        "state_name": "RentAFlat",
        "aliases": ["RentAFlat"],
        "formatters": {
            "add_new_apartment": lambda a: (
                f"New apartment available: {a.get('name', '')}"
            ),
            "update_apartment": lambda a: (
                f"Apartment {a.get('apartment_id', '')[:8]} updated"
                + (f" — new price: {a['new_price']}" if "new_price" in a else "")
            ),
        },
    },
    {
        "canonical": "ShoppingApp",
        "cli": "shopping",
        "agent_cli": "shopping",
        "module": "gaia2_cli.apps.shopping",
        "description": "Shopping",
        "state_name": "Shopping",
        "aliases": ["Shopping"],
        "formatters": {
            "add_product": lambda a: f"New product added: {a.get('name', '')}",
            "add_item_to_product": lambda a: (
                f"New item added to product {a.get('product_id', '')}"
            ),
            "cancel_order": lambda a: f"Order cancelled (Id {a.get('order_id', '')})",
            "update_order_status": lambda a: (
                f"Order updated (Id {a.get('order_id', '')})"
            ),
            "add_discount_code": lambda a: (
                f"New discount code for item {a.get('item_id', '')}"
            ),
            "update_item": lambda a: (
                f"Item {a.get('item_id', '')} updated"
                + (f" — new price: {a['new_price']}" if "new_price" in a else "")
            ),
        },
    },
    {
        "canonical": "CabApp",
        "cli": "cabs",
        "agent_cli": "cabs",
        "module": "gaia2_cli.apps.cab",
        "description": "Cab rides",
        "state_name": "CabApp",
        "aliases": ["Cabs"],
        "formatters": {
            "cancel_ride": lambda a: "Ride cancelled",
            "user_cancel_ride": lambda a: "Ride cancelled",
            "update_ride_status": lambda a: "Ride status updated",
            "end_ride": lambda a: "Ride completed",
        },
    },
    {
        "canonical": "CityApp",
        "cli": "city",
        "agent_cli": "city",
        "module": "gaia2_cli.apps.city",
        "description": "City information and services",
        "state_name": "CityApp",
        "aliases": ["City"],
        "formatters": {
            "update_crime_rate": lambda a: (
                f"Crime rate updated for {a.get('city_name', 'a city')}"
            ),
        },
    },
    {
        "canonical": "CloudDriveApp",
        "cli": "cloud-drive",
        "agent_cli": "cloud-drive",
        "module": "gaia2_cli.apps.files",
        "description": "Cloud storage (like Google Drive), separate from the local filesystem",
        "state_name": "Files",
        "aliases": [
            "CloudDrive",
            "Files",
            "SandboxLocalFileSystem",
            "VirtualFileSystem",
        ],
    },
]


# ---------------------------------------------------------------------------
# Derived mappings (built once at import time)
# ---------------------------------------------------------------------------

APP_TO_CLI: dict[str, str] = {}
"""Maps every app name (canonical + aliases) → real CLI binary name."""

CLI_TO_MODULE: dict[str, str] = {}
"""Maps CLI binary name → Python module path."""

NOTIFICATION_FORMATTERS: dict[str, dict[str, Callable[[dict[str, Any]], str]]] = {}
"""Maps app name → {function_name: notification formatter lambda}."""

APP_TO_AGENT_CLI: dict[str, str] = {}
"""Maps every app name (canonical + aliases) → agent-visible CLI name."""

APP_NAME_TO_STATE: dict[str, str] = {}
"""Maps every app name (canonical + aliases) → state file name (APP_NAME constant)."""

for _app in APP_REGISTRY:
    _cli = _app["cli"]
    _agent_cli = _app.get("agent_cli", _cli)
    _state = _app.get("state_name", _app["canonical"])
    CLI_TO_MODULE[_cli] = _app["module"]
    _fmts = _app.get("formatters")
    for _name in [_app["canonical"]] + _app.get("aliases", []):
        APP_TO_CLI[_name] = _cli
        APP_TO_AGENT_CLI[_name] = _agent_cli
        APP_NAME_TO_STATE[_name] = _state
        if _fmts:
            NOTIFICATION_FORMATTERS[_name] = _fmts
del _app, _cli, _agent_cli, _state, _fmts, _name


# ---------------------------------------------------------------------------
# Scenario resolution
# ---------------------------------------------------------------------------


def resolve_scenario_tools(scenario_path: str) -> dict[str, Any]:
    """Given a scenario JSON, return agent-visible tools and tools to remove.

    For each app in the scenario:
    1. Look up class_name (or name) in registry → get agent_cli and description
    2. Collect all agent_cli names the agent should see
    3. Compute which symlinks to remove (all known agent_cli names NOT in the set)

    Returns:
        {
            "tools": {"chats": "Chat conversations", ...},
            "remove": ["chats", "messages", ...],
        }
    """
    with open(scenario_path) as f:
        scenario = json.load(f)

    apps = scenario.get("apps", [])

    # Build the set of agent-visible CLI commands for this scenario
    tools: dict[str, str] = {}
    for app_entry in apps:
        if isinstance(app_entry, dict):
            class_name = app_entry.get("class_name", app_entry.get("name", ""))
            app_name = app_entry.get("name", "")
        else:
            class_name = getattr(app_entry, "class_name", "")
            app_name = getattr(app_entry, "name", "")

        # Try app_name first (e.g. "Chats"), then class_name (e.g. "MessagingAppV2").
        # app_name maps directly to the CLI module and determines the agent_cli.
        for name in (app_name, class_name):
            if name in APP_TO_AGENT_CLI:
                agent_cli = APP_TO_AGENT_CLI[name]
                # Find the description from the registry entry
                desc = _get_description(name)
                tools[agent_cli] = desc
                break

    # Compute which CLI names to remove from /home/agent/bin/:
    # all known cli and agent_cli names not in the tools set
    all_bin_names: set[str] = set()
    for app in APP_REGISTRY:
        all_bin_names.add(app["cli"])
        all_bin_names.add(app.get("agent_cli", app["cli"]))
    remove = sorted(all_bin_names - set(tools.keys()))

    return {"tools": tools, "remove": remove}


def _get_description(app_name: str) -> str:
    """Get the description for an app name from the registry."""
    for app in APP_REGISTRY:
        if app["canonical"] == app_name or app_name in app.get("aliases", []):
            return app.get("description", app["canonical"])
    return app_name
