# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
"""Unit tests for gaia2_cli/calendar_app.py (Calendar CLI)."""

import json
from datetime import datetime, timezone

import pytest
from conftest import assert_event, parse_output, read_events, seed_state
from gaia2_cli.apps.calendar import cli

APP_NAME = "Calendar"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# "2024-01-15 10:00:00" UTC
TS_JAN15_10 = datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc).timestamp()
# "2024-01-15 11:00:00" UTC
TS_JAN15_11 = datetime(2024, 1, 15, 11, 0, 0, tzinfo=timezone.utc).timestamp()
# "2024-01-15 12:00:00" UTC
TS_JAN15_12 = datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc).timestamp()
# "2024-01-16 10:00:00" UTC
TS_JAN16_10 = datetime(2024, 1, 16, 10, 0, 0, tzinfo=timezone.utc).timestamp()
# "2024-01-16 12:00:00" UTC
TS_JAN16_12 = datetime(2024, 1, 16, 12, 0, 0, tzinfo=timezone.utc).timestamp()
# "2024-01-17 10:00:00" UTC
TS_JAN17_10 = datetime(2024, 1, 17, 10, 0, 0, tzinfo=timezone.utc).timestamp()
# "2024-01-17 12:00:00" UTC
TS_JAN17_12 = datetime(2024, 1, 17, 12, 0, 0, tzinfo=timezone.utc).timestamp()


def _strftime(ts: float) -> str:
    """Mirror the display format from calendar_app."""
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%A, %Y-%m-%d %H:%M:%S")


def _make_event(
    event_id: str,
    title: str,
    start_ts: float,
    end_ts: float,
    tag=None,
    description=None,
    location=None,
    attendees=None,
) -> dict:
    """Build a calendar event dict matching the stored state format."""
    return {
        "event_id": event_id,
        "title": title,
        "start_datetime": start_ts,
        "end_datetime": end_ts,
        "tag": tag,
        "description": description,
        "location": location,
        "attendees": attendees or [],
        "start_strftime": _strftime(start_ts),
        "end_strftime": _strftime(end_ts),
    }


# ---------------------------------------------------------------------------
# Autouse fixture — ensure set_app("Calendar") before each test
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _set_calendar_app():
    """Ensure the module-level app name is set to Calendar for every test."""
    from gaia2_cli.shared import set_app

    set_app(APP_NAME)


# ---------------------------------------------------------------------------
# 1. add-event happy path
# ---------------------------------------------------------------------------


class TestAddCalendarEvent:
    def test_happy_path(self, cli_env, fixed_uuid):
        state_dir, runner = cli_env
        result = runner.invoke(
            cli,
            [
                "add-event",
                "--title",
                "Team Standup",
                "--start-datetime",
                "2024-01-15 10:00:00",
                "--end-datetime",
                "2024-01-15 11:00:00",
            ],
        )
        assert result.exit_code == 0, result.stderr
        out = parse_output(result)
        assert out["status"] == "success"
        assert out["event_id"] == fixed_uuid(0)

    # 2. add-event with default times (uses time.time() via fixed_time)
    def test_default_times(self, cli_env, fixed_uuid, fixed_time):
        state_dir, runner = cli_env
        # 2024-01-15 10:00:00 UTC
        fixed_time(TS_JAN15_10)
        result = runner.invoke(
            cli,
            ["add-event", "--title", "Quick Chat"],
        )
        assert result.exit_code == 0, result.stderr
        out = parse_output(result)
        assert out["status"] == "success"

        # Verify stored event has start=10:00 end=11:00 (1h default duration)
        state_path = state_dir / "calendar.json"
        state = json.loads(state_path.read_text())
        eid = out["event_id"]
        ev = state["events"][eid]
        assert ev["start_datetime"] == TS_JAN15_10
        assert ev["end_datetime"] == TS_JAN15_11

    # 3. add-event with attendees as JSON list
    def test_with_attendees(self, cli_env, fixed_uuid):
        state_dir, runner = cli_env
        attendees_json = '["Alice Smith", "Bob Jones"]'
        result = runner.invoke(
            cli,
            [
                "add-event",
                "--title",
                "Planning",
                "--start-datetime",
                "2024-01-15 10:00:00",
                "--end-datetime",
                "2024-01-15 11:00:00",
                "--attendees",
                attendees_json,
            ],
        )
        assert result.exit_code == 0, result.stderr
        out = parse_output(result)
        eid = out["event_id"]

        state = json.loads((state_dir / "calendar.json").read_text())
        assert state["events"][eid]["attendees"] == ["Alice Smith", "Bob Jones"]

    # 4. add-event invalid datetime format
    def test_invalid_datetime_format(self, cli_env):
        state_dir, runner = cli_env
        result = runner.invoke(
            cli,
            [
                "add-event",
                "--title",
                "Bad Date",
                "--start-datetime",
                "not-a-date",
                "--end-datetime",
                "2024-01-15 11:00:00",
            ],
        )
        assert result.exit_code == 1

    # 5. add-event start > end
    def test_start_after_end(self, cli_env):
        state_dir, runner = cli_env
        result = runner.invoke(
            cli,
            [
                "add-event",
                "--title",
                "Backwards",
                "--start-datetime",
                "2024-01-15 12:00:00",
                "--end-datetime",
                "2024-01-15 10:00:00",
            ],
        )
        assert result.exit_code == 1

    # 6. add-event events.jsonl logging
    def test_events_jsonl_logging(self, cli_env, fixed_uuid, fixed_time):
        state_dir, runner = cli_env
        fixed_time(TS_JAN15_10)
        result = runner.invoke(
            cli,
            [
                "add-event",
                "--title",
                "Logged Event",
                "--start-datetime",
                "2024-01-15 10:00:00",
                "--end-datetime",
                "2024-01-15 11:00:00",
                "--tag",
                "work",
                "--description",
                "A test event",
                "--location",
                "Room 42",
                "--attendees",
                '["Eve"]',
            ],
        )
        assert result.exit_code == 0, result.stderr

        events = read_events(state_dir)
        assert len(events) == 1
        ev = events[0]
        assert_event(ev, app="Calendar", fn="add_calendar_event", write=True)

        # All 7 parameters must be present in args
        args = ev["args"]
        assert args["title"] == "Logged Event"
        assert args["start_datetime"] == "2024-01-15 10:00:00"
        assert args["end_datetime"] == "2024-01-15 11:00:00"
        assert args["tag"] == "work"
        assert args["description"] == "A test event"
        assert args["location"] == "Room 42"
        assert args["attendees"] == ["Eve"]

        # ret should be the event_id
        assert ev["ret"] == fixed_uuid(0)


# ---------------------------------------------------------------------------
# 7-8. get-event
# ---------------------------------------------------------------------------


class TestGetCalendarEvent:
    def test_happy_path(self, cli_env):
        state_dir, runner = cli_env
        event = _make_event("evt1", "Lunch", TS_JAN15_10, TS_JAN15_11, tag="personal")
        seed_state(state_dir, APP_NAME, {"events": {"evt1": event}})

        result = runner.invoke(cli, ["get-event", "--event-id", "evt1"])
        assert result.exit_code == 0, result.stderr
        out = parse_output(result)
        assert out["event_id"] == "evt1"
        assert out["title"] == "Lunch"
        assert out["tag"] == "personal"

    def test_not_found(self, cli_env):
        state_dir, runner = cli_env
        seed_state(state_dir, APP_NAME, {"events": {}})
        result = runner.invoke(cli, ["get-event", "--event-id", "nonexistent"])
        assert result.exit_code == 1


# ---------------------------------------------------------------------------
# 9. delete-event
# ---------------------------------------------------------------------------


class TestDeleteCalendarEvent:
    def test_happy_path(self, cli_env):
        state_dir, runner = cli_env
        event = _make_event("evt1", "Delete Me", TS_JAN15_10, TS_JAN15_11)
        seed_state(state_dir, APP_NAME, {"events": {"evt1": event}})

        result = runner.invoke(cli, ["delete-event", "--event-id", "evt1"])
        assert result.exit_code == 0, result.stderr
        out = parse_output(result)
        assert out["status"] == "success"

        # Event removed from state
        state = json.loads((state_dir / "calendar.json").read_text())
        assert "evt1" not in state["events"]

        # Event logged as write
        events = read_events(state_dir)
        assert len(events) == 1
        assert_event(events[0], app="Calendar", fn="delete_calendar_event", write=True)


# ---------------------------------------------------------------------------
# 10. get-events overlap filter
# ---------------------------------------------------------------------------


class TestGetCalendarEventsFromTo:
    def test_overlap_filter(self, cli_env):
        state_dir, runner = cli_env
        # Event A: Jan 15 10-11 (inside query range)
        ev_a = _make_event("a", "Event A", TS_JAN15_10, TS_JAN15_11)
        # Event B: Jan 16 10-12 (inside query range)
        ev_b = _make_event("b", "Event B", TS_JAN16_10, TS_JAN16_12)
        # Event C: Jan 17 10-12 (outside query range)
        ev_c = _make_event("c", "Event C", TS_JAN17_10, TS_JAN17_12)

        seed_state(
            state_dir,
            APP_NAME,
            {"events": {"a": ev_a, "b": ev_b, "c": ev_c}},
        )

        # Query range: Jan 15 00:00 - Jan 17 00:00 (should match A and B, not C)
        result = runner.invoke(
            cli,
            [
                "get-events",
                "--start-date",
                "2024-01-15 00:00:00",
                "--end-date",
                "2024-01-17 00:00:00",
            ],
        )
        assert result.exit_code == 0, result.stderr
        out = parse_output(result)
        assert out["total"] == 2
        returned_ids = {e["event_id"] for e in out["events"]}
        assert returned_ids == {"a", "b"}


# ---------------------------------------------------------------------------
# 11. today-events
# ---------------------------------------------------------------------------


class TestReadTodayCalendarEvents:
    def test_returns_today_events(self, cli_env, fixed_time):
        state_dir, runner = cli_env
        # "Today" is 2024-01-15 when we set time to 10:00 on that day
        fixed_time(TS_JAN15_10)

        # Event on "today" (Jan 15)
        ev_today = _make_event("today1", "Today Event", TS_JAN15_10, TS_JAN15_12)
        # Event on "tomorrow" (Jan 16)
        ev_tomorrow = _make_event(
            "tomorrow1", "Tomorrow Event", TS_JAN16_10, TS_JAN16_12
        )

        seed_state(
            state_dir,
            APP_NAME,
            {"events": {"today1": ev_today, "tomorrow1": ev_tomorrow}},
        )

        result = runner.invoke(cli, ["today-events"])
        assert result.exit_code == 0, result.stderr
        out = parse_output(result)
        assert out["total"] == 1
        assert out["events"][0]["event_id"] == "today1"


# ---------------------------------------------------------------------------
# 12. get-all-tags
# ---------------------------------------------------------------------------


class TestGetAllTags:
    def test_unique_tags(self, cli_env):
        state_dir, runner = cli_env
        ev1 = _make_event("e1", "E1", TS_JAN15_10, TS_JAN15_11, tag="work")
        ev2 = _make_event("e2", "E2", TS_JAN15_11, TS_JAN15_12, tag="personal")
        ev3 = _make_event("e3", "E3", TS_JAN16_10, TS_JAN16_12, tag="work")
        # ev4 has no tag — should not appear
        ev4 = _make_event("e4", "E4", TS_JAN17_10, TS_JAN17_12, tag=None)

        seed_state(
            state_dir,
            APP_NAME,
            {"events": {"e1": ev1, "e2": ev2, "e3": ev3, "e4": ev4}},
        )

        result = runner.invoke(cli, ["get-all-tags"])
        assert result.exit_code == 0, result.stderr
        out = parse_output(result)
        assert set(out) == {"work", "personal"}


# ---------------------------------------------------------------------------
# 13. get-events-by-tag
# ---------------------------------------------------------------------------


class TestGetCalendarEventsByTag:
    def test_filter_by_tag(self, cli_env):
        state_dir, runner = cli_env
        ev1 = _make_event("e1", "E1", TS_JAN15_10, TS_JAN15_11, tag="work")
        ev2 = _make_event("e2", "E2", TS_JAN15_11, TS_JAN15_12, tag="personal")
        ev3 = _make_event("e3", "E3", TS_JAN16_10, TS_JAN16_12, tag="work")

        seed_state(
            state_dir,
            APP_NAME,
            {"events": {"e1": ev1, "e2": ev2, "e3": ev3}},
        )

        result = runner.invoke(cli, ["get-events-by-tag", "--tag", "work"])
        assert result.exit_code == 0, result.stderr
        out = parse_output(result)
        assert len(out) == 2
        returned_ids = {e["event_id"] for e in out}
        assert returned_ids == {"e1", "e3"}


# ---------------------------------------------------------------------------
# 14. search-events by title substring
# ---------------------------------------------------------------------------


class TestSearchEvents:
    def test_search_by_title(self, cli_env):
        state_dir, runner = cli_env
        ev1 = _make_event("e1", "Team Standup", TS_JAN15_10, TS_JAN15_11)
        ev2 = _make_event("e2", "Lunch Break", TS_JAN15_11, TS_JAN15_12)
        ev3 = _make_event(
            "e3",
            "Weekly Team Sync",
            TS_JAN16_10,
            TS_JAN16_12,
            description="Discuss team goals",
        )

        seed_state(
            state_dir,
            APP_NAME,
            {"events": {"e1": ev1, "e2": ev2, "e3": ev3}},
        )

        # "team" should match e1 (title) and e3 (title + description)
        result = runner.invoke(cli, ["search-events", "--query", "team"])
        assert result.exit_code == 0, result.stderr
        out = parse_output(result)
        assert len(out) == 2
        returned_ids = {e["event_id"] for e in out}
        assert returned_ids == {"e1", "e3"}
