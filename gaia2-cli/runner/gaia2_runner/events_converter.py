# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Convert events.jsonl entries (from gaia2-cli tools) into CompletedEvent objects.

The gaia2-cli tools log every action to events.jsonl via shared.log_action() with
the format: {"t": <timestamp>, "app": <app_name>, "fn": <function_name>,
"args": {...}, "w": <is_write>, "ret": <return_value>}

This module converts those lightweight CLI log entries into CompletedEvent
objects (from gaia2_core.types) that the standalone Judge can compare
against oracle events.
"""

from __future__ import annotations

import json
import logging
import uuid
from typing import Any

from gaia2_core.types import CompletedEvent, EventAction

logger: logging.Logger = logging.getLogger(__name__)


def _entry_to_completed_event(entry: dict[str, Any]) -> CompletedEvent:
    """Convert a minimal CLI log entry to a CompletedEvent.

    The CLI writes compact entries to events.jsonl:
        {"t": <float>, "app": "Contacts", "fn": "add_new_contact",
         "args": {...}, "w": true, "ret": "abc123"}
    """
    app = entry["app"]
    fn = entry["fn"]
    action = EventAction(
        app_name=app,
        class_name=app,
        function_name=fn,
        args=entry.get("args", {}),
        operation_type="write" if entry.get("w") else "read",
    )
    eid = (
        entry.get("event_id")
        or entry.get("eid")
        or f"AGENT-{app}.{fn}-{uuid.uuid4().hex[:8]}"
    )
    return CompletedEvent(
        event_id=eid,
        event_type="ENV" if eid.startswith("Event-ENV-") else "AGENT",
        event_time=float(entry.get("t", 0.0)),
        action=action,
        return_value=entry.get("ret"),
    )


def _dict_to_completed_event(d: dict[str, Any]) -> CompletedEvent:
    """Convert a full CompletedEvent dict to a CompletedEvent.

    Handles the dict format from the Gaia2 framework's CompletedEvent.to_dict()
    or any future full-form entries in events.jsonl.
    """
    action_d = d.get("action", {})
    action = None
    if action_d:
        action = EventAction(
            app_name=action_d.get("app_name", action_d.get("class_name", "")),
            class_name=action_d.get("class_name", action_d.get("app_name", "")),
            function_name=action_d.get("function_name", ""),
            args=action_d.get("args", {}),
            operation_type=action_d.get("operation_type", "read"),
        )

    metadata = d.get("metadata", {})
    return CompletedEvent(
        event_id=d.get("event_id", ""),
        event_type=d.get("event_type", "AGENT"),
        event_time=float(d.get("event_time", 0.0)),
        action=action,
        return_value=metadata.get("return_value") if metadata else None,
    )


def action_to_event_dict(action: dict[str, Any]) -> dict[str, Any]:
    """Convert a minimal CLI action entry to a full CompletedEvent dict.

    Kept for backwards compatibility — callers that need a dict representation
    can use this, but convert_events_jsonl() now constructs CompletedEvent
    objects directly.
    """
    app = action["app"]
    fn = action["fn"]
    return {
        "class_name": "CompletedEvent",
        "event_type": "AGENT",
        "event_time": action["t"],
        "event_id": f"AGENT-{app}.{fn}-{uuid.uuid4().hex[:8]}",
        "action": {
            "class_name": app,
            "app_name": app,
            "function_name": fn,
            "args": action.get("args", {}),
            "resolved_args": {},
            "operation_type": "write" if action.get("w") else "read",
            "action_id": f"{app}.{fn}-{uuid.uuid4().hex[:8]}",
        },
        "metadata": {
            "return_value": action.get("ret"),
            "exception": None,
            "exception_stack_trace": None,
            "completed": True,
        },
        "successors": [],
        "dependencies": [],
    }


def parse_events_jsonl(raw: str) -> list[dict[str, Any]]:
    """Parse events.jsonl content into a list of dicts.

    Skips blank lines and malformed JSON lines (logging a warning).
    """
    entries = []
    for lineno, line in enumerate(raw.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError as exc:
            logger.warning(
                "Skipping malformed line %d in events.jsonl: %s", lineno, exc
            )
    return entries


def convert_events_jsonl(raw: str) -> list[CompletedEvent]:
    """Convert raw events.jsonl content to a list of CompletedEvent objects.

    This is the main entry point for the converter.  Each line is either:
    - Minimal CLI format (has "app" and "fn" keys) — converted via
      _entry_to_completed_event().
    - Full CompletedEvent dict — converted via _dict_to_completed_event().

    Args:
        raw: Full contents of events.jsonl (newline-delimited JSON).

    Returns:
        List of CompletedEvent objects in log order.
    """
    entries = parse_events_jsonl(raw)
    events: list[CompletedEvent] = []
    for entry in entries:
        if "app" in entry and "fn" in entry:
            events.append(_entry_to_completed_event(entry))
        else:
            events.append(_dict_to_completed_event(entry))
    return events
