# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
"""Unit tests for gaia2_cli/shared.py."""

from dataclasses import dataclass
from enum import Enum

import click
import pytest
from conftest import read_events  # noqa: F401 (used by tests indirectly)
from gaia2_cli.shared import (
    build_schema,
    cli_error,
    get_state_dir,
    load_app_state,
    log_action,
    make_serializable,
    normalize_app_name,
    save_app_state,
    set_app,
    state_file_for_app,
)

# ---------------------------------------------------------------------------
# normalize_app_name
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "app_name, expected",
    [
        ("Calendar", "calendar"),
        ("Contacts", "contacts"),
        ("EmailClientV2", "email_client_v2"),
        ("MessagingAppV2", "messaging_app_v2"),
        ("Chats", "chats"),
        ("RentAFlat", "rent_a_flat"),
        ("CityApp", "city_app"),
        ("CabApp", "cab_app"),
        ("Shopping", "shopping"),
    ],
)
def test_normalize_app_name(app_name, expected):
    assert normalize_app_name(app_name) == expected


# ---------------------------------------------------------------------------
# state_file_for_app
# ---------------------------------------------------------------------------


def test_state_file_for_app(tmp_path):
    result = state_file_for_app("Calendar", state_dir=tmp_path)
    assert result == tmp_path / "calendar.json"


# ---------------------------------------------------------------------------
# get_state_dir
# ---------------------------------------------------------------------------


def test_get_state_dir_with_env(cli_env):
    state_dir, _ = cli_env
    result = get_state_dir()
    assert result == state_dir


def test_get_state_dir_without_env(monkeypatch):
    monkeypatch.delenv("GAIA2_STATE_DIR", raising=False)
    with pytest.raises(RuntimeError, match="GAIA2_STATE_DIR"):
        get_state_dir()


# ---------------------------------------------------------------------------
# load_app_state / save_app_state roundtrip
# ---------------------------------------------------------------------------


def test_load_save_roundtrip(cli_env):
    state_dir, _ = cli_env
    original = {"contacts": {"abc": {"name": "Alice"}}, "counter": 42}

    save_app_state("Contacts", original, state_dir=state_dir)
    loaded = load_app_state("Contacts", state_dir=state_dir)

    assert loaded == original


def test_load_missing_file_returns_empty_dict(cli_env):
    state_dir, _ = cli_env
    result = load_app_state("Calendar", state_dir=state_dir)
    assert result == {}


# ---------------------------------------------------------------------------
# log_action
# ---------------------------------------------------------------------------


def test_log_action_writes_correct_format(cli_env, fixed_time):
    state_dir, _ = cli_env
    set_app("Contacts")
    fixed_time(1522479600.0)

    log_action(
        fn="add_new_contact",
        args={"first_name": "Katie", "last_name": "Lee"},
        ret="abc123",
        write=True,
    )

    events = read_events(state_dir)
    assert len(events) == 1

    event = events[0]
    assert event["t"] == 1522479600.0
    assert event["app"] == "Contacts"
    assert event["fn"] == "add_new_contact"
    assert event["args"] == {"first_name": "Katie", "last_name": "Lee"}
    assert event["w"] is True
    assert event["ret"] == "abc123"
    # Exactly 6 fields
    assert set(event.keys()) == {"t", "app", "fn", "args", "w", "ret"}


def test_log_action_appends_multiple_events(cli_env, fixed_time):
    state_dir, _ = cli_env
    set_app("Calendar")

    fixed_time(1000.0)
    log_action(fn="create_event", args={"title": "Meeting"}, ret="e1", write=True)

    fixed_time(2000.0)
    log_action(
        fn="get_event", args={"event_id": "e1"}, ret={"title": "Meeting"}, write=False
    )

    events = read_events(state_dir)
    assert len(events) == 2
    assert events[0]["t"] == 1000.0
    assert events[0]["fn"] == "create_event"
    assert events[1]["t"] == 2000.0
    assert events[1]["fn"] == "get_event"


# ---------------------------------------------------------------------------
# make_serializable
# ---------------------------------------------------------------------------


def test_make_serializable_dataclass():
    @dataclass
    class Point:
        x: int
        y: int

    result = make_serializable(Point(x=1, y=2))
    assert result == {"x": 1, "y": 2}


def test_make_serializable_enum():
    class Color(Enum):
        RED = "red"
        BLUE = "blue"

    assert make_serializable(Color.RED) == "red"
    assert make_serializable(Color.BLUE) == "blue"


def test_make_serializable_bytes():
    assert make_serializable(b"hello") == "hello"
    # Non-UTF-8 bytes use replacement character
    result = make_serializable(b"\xff\xfe")
    assert isinstance(result, str)


def test_make_serializable_nested_dict():
    @dataclass
    class Inner:
        val: int

    class Status(Enum):
        ACTIVE = "active"

    data = {
        "dc": Inner(val=5),
        "enum": Status.ACTIVE,
        "raw": b"data",
        "nested": {"key": [1, 2, 3]},
    }

    result = make_serializable(data)
    assert result == {
        "dc": {"val": 5},
        "enum": "active",
        "raw": "data",
        "nested": {"key": [1, 2, 3]},
    }


def test_make_serializable_tuple_to_list():
    assert make_serializable((1, "two", 3)) == [1, "two", 3]
    # Nested tuples
    assert make_serializable(((1, 2), (3, 4))) == [[1, 2], [3, 4]]


# ---------------------------------------------------------------------------
# cli_error
# ---------------------------------------------------------------------------


def test_cli_error_exits_with_code_1(capsys):
    with pytest.raises(SystemExit) as exc_info:
        cli_error("something went wrong")

    assert exc_info.value.code == 1
    captured = capsys.readouterr()
    assert "something went wrong" in captured.err


# ---------------------------------------------------------------------------
# build_schema
# ---------------------------------------------------------------------------


def test_build_schema_returns_correct_structure():
    @click.group()
    def sample_group():
        pass

    @sample_group.command(help="Greet someone by name.")
    @click.option("--name", required=True, type=str, help="Person's name")
    @click.option("--count", default=1, type=int, help="Number of greetings")
    def greet(name, count):
        pass

    @sample_group.command(name="schema", help="Print schema.")
    def schema_cmd():
        pass

    result = build_schema(sample_group)

    # The "schema" command should be excluded
    assert len(result) == 1

    entry = result[0]
    assert entry["command"] == "greet"
    assert entry["function"] == "greet"
    assert entry["description"] == "Greet someone by name."
    assert len(entry["parameters"]) == 2

    # Click's human_readable_name strips the '--' prefix.
    # Decorators stack in reverse, so --count appears first.
    params_by_name = {p["name"]: p for p in entry["parameters"]}

    name_param = params_by_name["name"]
    assert name_param["required"] is True
    assert name_param["description"] == "Person's name"

    count_param = params_by_name["count"]
    assert count_param["required"] is False
    assert count_param["default"] == 1
    assert count_param["description"] == "Number of greetings"
