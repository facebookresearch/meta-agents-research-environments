# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
"""
Standalone CLI for the Gaia2 Calendar app.

Binary name: ``calendar``
App class name (for events): ``Calendar``
"""

import json
import time
import uuid
from datetime import datetime, timedelta, timezone

import click

from gaia2_cli.shared import (
    build_schema,
    cli_error,
    json_output,
    load_app_state,
    log_action,
    save_app_state,
    set_app,
)

APP_NAME = "Calendar"

set_app(APP_NAME)

DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"
DISPLAY_FORMAT = "%A, %Y-%m-%d %H:%M:%S"
DEFAULT_DURATION_HOURS = 1


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_state() -> dict:
    """Load calendar state, returning a default if missing."""
    state = load_app_state(APP_NAME)
    if not state:
        return {"events": {}}
    if "events" not in state:
        state["events"] = {}
    return state


def _parse_datetime(dt_str: str) -> float:
    """Parse 'YYYY-MM-DD HH:MM:SS' into a UTC timestamp (float)."""
    try:
        dt = datetime.strptime(dt_str, DATETIME_FORMAT)
        return dt.replace(tzinfo=timezone.utc).timestamp()
    except ValueError:
        raise ValueError("Invalid datetime format. Please use YYYY-MM-DD HH:MM:SS")


def _strftime(ts: float) -> str:
    """Format a UTC timestamp into the human-readable display format."""
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime(DISPLAY_FORMAT)


def _event_dict(event_id: str, event: dict) -> dict:
    """Return a canonical event dict with all fields present."""
    return {
        "event_id": event_id,
        "title": event.get("title", ""),
        "start_datetime": event.get("start_datetime", 0.0),
        "end_datetime": event.get("end_datetime", 0.0),
        "tag": event.get("tag"),
        "description": event.get("description"),
        "location": event.get("location"),
        "attendees": event.get("attendees", []),
        "start_strftime": event.get("start_strftime", ""),
        "end_strftime": event.get("end_strftime", ""),
    }


def _build_args_dict(**kwargs) -> dict:
    """Build an args dict — always include all keys.

    The grader's HardToolJudge accesses args by key directly (not .get()),
    so all function parameters must be present even when None.
    """
    return dict(kwargs)


# ---------------------------------------------------------------------------
# CLI group
# ---------------------------------------------------------------------------


@click.group(context_settings={"terminal_width": 10000, "max_content_width": 10000})
def cli():
    """Calendar - Gaia2 Calendar app CLI."""
    pass


# ---------------------------------------------------------------------------
# add-event (WRITE)
# ---------------------------------------------------------------------------


@cli.command("add-event")
@click.option("--title", default="Event", type=str, help="Title of the event.")
@click.option(
    "--start-datetime",
    "start_datetime",
    default=None,
    type=str,
    help="Start datetime in YYYY-MM-DD HH:MM:SS format.",
)
@click.option(
    "--end-datetime",
    "end_datetime",
    default=None,
    type=str,
    help="End datetime in YYYY-MM-DD HH:MM:SS format.",
)
@click.option(
    "--tag", default=None, type=str, help="Tag of the event. Defaults to None."
)
@click.option(
    "--description",
    default=None,
    type=str,
    help="Description of the event. Defaults to None.",
)
@click.option(
    "--location",
    default=None,
    type=str,
    help="Location of the event. Defaults to None.",
)
@click.option(
    "--attendees",
    default=None,
    type=str,
    help='JSON list of attendees full names, e.g. \'["Alice", "Bob"]\'.',
)
def add_calendar_event(
    title, start_datetime, end_datetime, tag, description, location, attendees
):
    """Add a calendar event to the calendar. Unless specified otherwise in the task, the default week starts on Monday and ends on Sunday."""
    # Parse attendees JSON
    if attendees is not None:
        try:
            attendees = json.loads(attendees)
        except json.JSONDecodeError:
            attendees = []
    else:
        attendees = []

    # Handle default times using time.time() (intercepted by libfaketime)
    now = time.time()
    if start_datetime is None:
        start_datetime = datetime.fromtimestamp(now, tz=timezone.utc).strftime(
            DATETIME_FORMAT
        )
    if end_datetime is None:
        end_datetime = (
            datetime.fromtimestamp(now, tz=timezone.utc)
            + timedelta(hours=DEFAULT_DURATION_HOURS)
        ).strftime(DATETIME_FORMAT)

    # Parse and validate datetimes
    try:
        start_ts = _parse_datetime(start_datetime)
        end_ts = _parse_datetime(end_datetime)
    except ValueError as e:
        cli_error(str(e))

    if start_ts > end_ts:
        cli_error("Start time cannot be after end time.")

    state = _load_state()
    events = state["events"]

    event_id = uuid.uuid4().hex
    new_event = {
        "event_id": event_id,
        "title": title,
        "start_datetime": start_ts,
        "end_datetime": end_ts,
        "tag": tag,
        "description": description,
        "location": location,
        "attendees": attendees,
        "start_strftime": _strftime(start_ts),
        "end_strftime": _strftime(end_ts),
    }

    events[event_id] = new_event
    state["events"] = events
    save_app_state(APP_NAME, state)

    args = _build_args_dict(
        title=title,
        start_datetime=start_datetime,
        end_datetime=end_datetime,
        tag=tag,
        description=description,
        location=location,
        attendees=attendees,
    )
    log_action("add_calendar_event", args, ret=event_id, write=True)
    json_output({"status": "success", "event_id": event_id})


# ---------------------------------------------------------------------------
# get-event (READ)
# ---------------------------------------------------------------------------


@cli.command("get-event")
@click.option(
    "--event-id", "event_id", required=True, type=str, help="Id of the event to read."
)
def get_calendar_event(event_id):
    """Read a calendar event from the calendar."""
    state = _load_state()
    events = state["events"]

    if event_id not in events:
        cli_error(f"Calendar Event with id {event_id} does not exist.")

    result = _event_dict(event_id, events[event_id])
    log_action("get_calendar_event", {"event_id": event_id}, ret=result)
    json_output(result)


# ---------------------------------------------------------------------------
# delete-event (WRITE)
# ---------------------------------------------------------------------------


@cli.command("delete-event")
@click.option(
    "--event-id", "event_id", required=True, type=str, help="Id of the event to delete."
)
def delete_calendar_event(event_id):
    """Delete a calendar event from the calendar."""
    state = _load_state()
    events = state["events"]

    if event_id not in events:
        cli_error(f"Calendar Event with id {event_id} does not exist.")

    del events[event_id]
    state["events"] = events
    save_app_state(APP_NAME, state)

    result = f"Event {event_id} successfully deleted."
    log_action("delete_calendar_event", {"event_id": event_id}, ret=result, write=True)
    json_output({"status": "success", "message": result})


# ---------------------------------------------------------------------------
# get-events (READ)
# ---------------------------------------------------------------------------


@cli.command("get-events")
@click.option(
    "--start-date",
    "start_datetime",
    required=True,
    type=str,
    help="Start datetime of the range in YYYY-MM-DD HH:MM:SS format.",
)
@click.option(
    "--end-date",
    "end_datetime",
    required=True,
    type=str,
    help="End datetime of the range in YYYY-MM-DD HH:MM:SS format.",
)
@click.option(
    "--offset", default=0, type=int, help="Offset to start listing from, default is 0."
)
@click.option(
    "--limit", default=10, type=int, help="Number of events to list, default is 10."
)
def get_calendar_events_from_to(start_datetime, end_datetime, offset, limit):
    """Get calendar events that have any time overlap with the specified date range (excludes events that only touch at boundaries). Unless specified otherwise in the task, the default week starts on Monday and ends on Sunday."""
    try:
        start_ts = _parse_datetime(start_datetime)
        end_ts = _parse_datetime(end_datetime)
    except ValueError as e:
        cli_error(str(e))

    if start_ts > end_ts:
        cli_error("Start time cannot be after end time.")

    state = _load_state()
    events = state["events"]

    # Find overlapping events: event.start < range.end AND event.end > range.start
    overlapping = [
        _event_dict(eid, ev)
        for eid, ev in events.items()
        if ev.get("start_datetime", 0) < end_ts and ev.get("end_datetime", 0) > start_ts
    ]

    start_index = offset
    end_index = offset + limit

    result = {
        "events": overlapping[start_index:end_index],
        "range": [start_index, min(end_index, len(overlapping))],
        "total": len(overlapping),
    }

    args = _build_args_dict(
        start_datetime=start_datetime,
        end_datetime=end_datetime,
        offset=offset,
        limit=limit,
    )
    log_action("get_calendar_events_from_to", args, ret=result)
    json_output(result)


# ---------------------------------------------------------------------------
# today-events (READ)
# ---------------------------------------------------------------------------


@cli.command("today-events")
def read_today_calendar_events():
    """Read today's calendar events from the calendar."""
    now = time.time()
    today = datetime.fromtimestamp(now, tz=timezone.utc)
    today = today.replace(hour=0, minute=0, second=0, microsecond=0)
    tomorrow = today + timedelta(days=1)

    today_ts = today.timestamp()
    tomorrow_ts = tomorrow.timestamp()

    state = _load_state()
    events = state["events"]

    overlapping = [
        _event_dict(eid, ev)
        for eid, ev in events.items()
        if ev.get("start_datetime", 0) < tomorrow_ts
        and ev.get("end_datetime", 0) > today_ts
    ]

    result = {
        "events": overlapping,
        "range": [0, len(overlapping)],
        "total": len(overlapping),
    }

    log_action("read_today_calendar_events", {}, ret=result)
    json_output(result)


# ---------------------------------------------------------------------------
# get-all-tags (READ)
# ---------------------------------------------------------------------------


@cli.command("get-all-tags")
def get_all_tags():
    """Get all tags from the calendar."""
    state = _load_state()
    events = state["events"]

    tags = list({ev.get("tag") for ev in events.values() if ev.get("tag")})

    log_action("get_all_tags", {}, ret=tags)
    json_output(tags)


# ---------------------------------------------------------------------------
# get-events-by-tag (READ)
# ---------------------------------------------------------------------------


@cli.command("get-events-by-tag")
@click.option("--tag", required=True, type=str, help="Tag for which to get the events.")
def get_calendar_events_by_tag(tag):
    """Get calendar events from the calendar by tag."""
    if not tag:
        cli_error("Tag cannot be empty.")

    state = _load_state()
    events = state["events"]

    results = [
        _event_dict(eid, ev) for eid, ev in events.items() if ev.get("tag") == tag
    ]

    log_action("get_calendar_events_by_tag", {"tag": tag}, ret=results)
    json_output(results)


# ---------------------------------------------------------------------------
# search-events (READ)
# ---------------------------------------------------------------------------


@cli.command("search-events")
@click.option("--query", required=True, type=str, help="The search query string.")
def search_events(query):
    """Searches for calendar events based on a query string. The search looks for partial matches in title, description, location, and attendees."""
    state = _load_state()
    events = state["events"]
    query_lower = query.lower()

    results = []
    for eid, ev in events.items():
        if (
            query_lower in (ev.get("title") or "").lower()
            or (ev.get("description") and query_lower in ev["description"].lower())
            or (ev.get("location") and query_lower in ev["location"].lower())
            or any(query_lower in att.lower() for att in ev.get("attendees", []))
        ):
            results.append(_event_dict(eid, ev))

    log_action("search_events", {"query": query}, ret=results)
    json_output(results)


# ===========================================================================
# Hidden ENV-tool commands (used by gaia2-eventd, not visible to agents)
# ===========================================================================


@cli.command("add-calendar-event-by-attendee", hidden=True)
@click.option(
    "--who-add", required=True, type=str, help="Name of the attendee adding the event."
)
@click.option("--title", default="Event", type=str, help="Title of the event.")
@click.option(
    "--start-datetime",
    "start_datetime",
    default=None,
    type=str,
    help="Start datetime (YYYY-MM-DD HH:MM:SS).",
)
@click.option(
    "--end-datetime",
    "end_datetime",
    default=None,
    type=str,
    help="End datetime (YYYY-MM-DD HH:MM:SS).",
)
@click.option("--tag", default=None, type=str, help="Tag of the event.")
@click.option("--description", default=None, type=str, help="Description of the event.")
@click.option("--location", default=None, type=str, help="Location of the event.")
@click.option("--attendees", default=None, type=str, help="JSON list of attendees.")
def add_calendar_event_by_attendee(
    who_add, title, start_datetime, end_datetime, tag, description, location, attendees
):
    """[ENV] Add a calendar event on behalf of another attendee."""
    if attendees is not None:
        try:
            attendees = json.loads(attendees)
        except json.JSONDecodeError:
            attendees = []
    else:
        attendees = []

    now = time.time()
    if start_datetime is None:
        start_datetime = datetime.fromtimestamp(now, tz=timezone.utc).strftime(
            DATETIME_FORMAT
        )
    if end_datetime is None:
        end_datetime = (
            datetime.fromtimestamp(now, tz=timezone.utc)
            + timedelta(hours=DEFAULT_DURATION_HOURS)
        ).strftime(DATETIME_FORMAT)

    try:
        start_ts = _parse_datetime(start_datetime)
        end_ts = _parse_datetime(end_datetime)
    except ValueError as e:
        cli_error(str(e))

    if start_ts > end_ts:
        cli_error("Start time cannot be after end time.")

    # Mimic Gaia2 app: auto-add who_add, prefix title/description
    if who_add not in attendees:
        attendees.append(who_add)
    title = f"Event created by {who_add}: {title}"
    description = (
        f"Event created by {who_add}: {description}"
        if description
        else f"Event created by {who_add}"
    )

    state = _load_state()
    events = state["events"]

    event_id = uuid.uuid4().hex
    new_event = {
        "event_id": event_id,
        "title": title,
        "start_datetime": start_ts,
        "end_datetime": end_ts,
        "tag": tag,
        "description": description,
        "location": location,
        "attendees": attendees,
        "start_strftime": _strftime(start_ts),
        "end_strftime": _strftime(end_ts),
    }

    events[event_id] = new_event
    state["events"] = events
    save_app_state(APP_NAME, state)

    args = _build_args_dict(
        who_add=who_add,
        title=title,
        start_datetime=start_datetime,
        end_datetime=end_datetime,
        tag=tag,
        description=description,
        location=location,
        attendees=attendees,
    )
    log_action("add_calendar_event_by_attendee", args, ret=event_id, write=True)
    json_output({"status": "success", "event_id": event_id})


@cli.command("delete-calendar-event-by-attendee", hidden=True)
@click.option(
    "--event-id", "event_id", required=True, type=str, help="Id of the event to delete."
)
@click.option(
    "--who-delete",
    required=True,
    type=str,
    help="Name of the attendee deleting the event.",
)
def delete_calendar_event_by_attendee(event_id, who_delete):
    """[ENV] Delete a calendar event on behalf of an attendee."""
    state = _load_state()
    events = state["events"]

    if event_id not in events:
        cli_error(f"Calendar Event with id {event_id} does not exist.")

    ev = events[event_id]
    if who_delete not in ev.get("attendees", []):
        cli_error(f"{who_delete} is not an attendee of the event.")

    del events[event_id]
    state["events"] = events
    save_app_state(APP_NAME, state)

    result = f"Event {event_id} successfully deleted by {who_delete}."
    log_action(
        "delete_calendar_event_by_attendee",
        {"event_id": event_id, "who_delete": who_delete},
        ret=result,
        write=True,
    )
    json_output({"status": "success", "message": result})


# ---------------------------------------------------------------------------
# schema
# ---------------------------------------------------------------------------


@cli.command("schema")
def schema():
    """Output machine-readable JSON schema of all commands."""
    json_output(build_schema(cli))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    cli()


if __name__ == "__main__":
    main()
