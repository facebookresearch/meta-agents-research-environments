# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Tests for gaia2_runner.events_converter."""

import json

import pytest
from gaia2_runner.events_converter import (
    action_to_event_dict,
    convert_events_jsonl,
    parse_events_jsonl,
)

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _make_action(
    app: str = "Contacts",
    fn: str = "add_new_contact",
    t: float = 1522479600.0,
    args: dict | None = None,
    w: bool | None = True,
    ret: str | None = "abc123",
) -> dict:
    """Build a minimal CLI action dict for testing."""
    action: dict = {"t": t, "app": app, "fn": fn, "args": args or {}}
    if w is not None:
        action["w"] = w
    if ret is not None:
        action["ret"] = ret
    return action


# ---------------------------------------------------------------------------
# parse_events_jsonl
# ---------------------------------------------------------------------------


class TestParseEventsJsonl:
    def test_single_entry(self):
        line = json.dumps(_make_action())
        entries = parse_events_jsonl(line)
        assert len(entries) == 1
        assert entries[0]["app"] == "Contacts"

    def test_multiple_entries(self):
        lines = "\n".join(json.dumps(_make_action(t=float(i))) for i in range(3))
        entries = parse_events_jsonl(lines)
        assert len(entries) == 3

    def test_empty_input(self):
        assert parse_events_jsonl("") == []

    def test_blank_lines_skipped(self):
        line = json.dumps(_make_action())
        raw = f"\n\n{line}\n\n"
        entries = parse_events_jsonl(raw)
        assert len(entries) == 1

    def test_malformed_line_skipped(self, caplog):
        good = json.dumps(_make_action())
        raw = f"{good}\nNOT VALID JSON\n{good}\n"
        entries = parse_events_jsonl(raw)
        assert len(entries) == 2
        assert "malformed" in caplog.text.lower()


# ---------------------------------------------------------------------------
# action_to_event_dict
# ---------------------------------------------------------------------------


class TestActionToEventDict:
    def test_write_action(self):
        action = _make_action(w=True)
        result = action_to_event_dict(action)

        assert result["action"]["operation_type"] == "write"
        assert result["class_name"] == "CompletedEvent"
        assert result["event_type"] == "AGENT"
        assert result["event_time"] == 1522479600.0
        assert result["action"]["function_name"] == "add_new_contact"
        assert result["action"]["class_name"] == "Contacts"
        assert result["metadata"]["return_value"] == "abc123"

    def test_read_action(self):
        action = _make_action(w=False)
        result = action_to_event_dict(action)
        assert result["action"]["operation_type"] == "read"

    def test_missing_w_defaults_to_read(self):
        action = _make_action(w=None)
        result = action_to_event_dict(action)
        assert result["action"]["operation_type"] == "read"

    def test_missing_ret_defaults_to_none(self):
        action = _make_action(ret=None)
        result = action_to_event_dict(action)
        assert result["metadata"]["return_value"] is None

    @pytest.mark.parametrize(
        "app_name",
        [
            "Calendar",
            "Contacts",
            "EmailClientV2",
            "MessagingAppV2",
            "Chats",
            "RentAFlat",
            "CabApp",
            "Shopping",
        ],
    )
    def test_all_app_names(self, app_name: str):
        action = _make_action(app=app_name)
        result = action_to_event_dict(action)
        assert result["action"]["class_name"] == app_name
        assert result["action"]["app_name"] == app_name

    def test_output_structure(self):
        result = action_to_event_dict(_make_action())

        # Top-level keys
        for key in (
            "class_name",
            "event_type",
            "event_time",
            "event_id",
            "action",
            "metadata",
            "successors",
            "dependencies",
        ):
            assert key in result

        # Action nested keys
        for key in (
            "class_name",
            "app_name",
            "function_name",
            "args",
            "resolved_args",
            "operation_type",
            "action_id",
        ):
            assert key in result["action"]

        # Metadata nested keys
        meta = result["metadata"]
        assert meta["completed"] is True
        assert meta["exception"] is None
        assert meta["exception_stack_trace"] is None

    def test_event_id_format(self):
        action = _make_action(app="Calendar", fn="get_events")
        result = action_to_event_dict(action)
        assert result["event_id"].startswith("AGENT-Calendar.get_events-")


# ---------------------------------------------------------------------------
# convert_events_jsonl (end-to-end, requires gaia2.apps importable)
# ---------------------------------------------------------------------------


class TestConvertEventsJsonl:
    def test_empty_file(self):
        assert convert_events_jsonl("") == []

    def test_end_to_end(self):
        entries = [
            _make_action(
                app="Contacts",
                fn="get_contacts",
                t=100.0,
                w=False,
                ret='[{"name": "Alice"}]',
            ),
            _make_action(
                app="Calendar",
                fn="add_calendar_event",
                t=101.0,
                args={"title": "Lunch"},
                w=True,
                ret="ev1",
            ),
        ]
        raw = "\n".join(json.dumps(e) for e in entries)
        events = convert_events_jsonl(raw)

        assert len(events) == 2
        assert events[0].tool_name == "Contacts__get_contacts"
        assert events[0].event_time == 100.0
        assert events[1].tool_name == "Calendar__add_calendar_event"
        assert events[1].get_args()["title"] == "Lunch"
