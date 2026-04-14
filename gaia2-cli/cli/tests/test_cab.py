# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
"""Unit tests for gaia2_cli/cab_app.py (CabApp)."""

import pytest
from conftest import assert_event, parse_output, read_events, seed_state
from gaia2_cli.apps.cab import cli
from gaia2_cli.shared import set_app

APP_NAME = "CabApp"

RIDE_TIME = "2024-01-15 10:00:00"

EMPTY_STATE = {
    "ride_history": [],
    "quotation_history": [],
    "d_service_config": {
        "Default": {
            "nb_seats": 4,
            "price_per_km": 1.0,
            "base_delay_min": 5,
            "max_distance_km": 25,
        },
        "Premium": {
            "nb_seats": 4,
            "price_per_km": 2.0,
            "base_delay_min": 3,
            "max_distance_km": 25,
        },
        "Van": {
            "nb_seats": 6,
            "price_per_km": 1.5,
            "base_delay_min": 7,
            "max_distance_km": 25,
        },
    },
    "on_going_ride": None,
}


@pytest.fixture(autouse=True)
def _set_app():
    set_app(APP_NAME)


# ---------------------------------------------------------------------------
# 1. get-quotation happy path
# ---------------------------------------------------------------------------


def test_get_quotation_happy(cli_env, fixed_time):
    state_dir, runner = cli_env
    seed_state(state_dir, APP_NAME, EMPTY_STATE)

    result = runner.invoke(
        cli,
        [
            "get-quotation",
            "--start-location",
            "Home",
            "--end-location",
            "Office",
            "--service-type",
            "Default",
            "--ride-time",
            RIDE_TIME,
        ],
    )

    assert result.exit_code == 0, result.output
    data = parse_output(result)
    assert data["status"] == "success"
    ride = data["data"]
    assert "price" in ride
    assert "distance_km" in ride
    assert "delay" in ride
    assert ride["service_type"] == "Default"
    assert ride["start_location"] == "Home"
    assert ride["end_location"] == "Office"
    assert ride["price"] > 0
    assert ride["distance_km"] > 0
    assert ride["delay"] > 0


# ---------------------------------------------------------------------------
# 2. get-quotation invalid service type
# ---------------------------------------------------------------------------


def test_get_quotation_invalid_service(cli_env, fixed_time):
    state_dir, runner = cli_env
    seed_state(state_dir, APP_NAME, EMPTY_STATE)

    result = runner.invoke(
        cli,
        [
            "get-quotation",
            "--start-location",
            "Home",
            "--end-location",
            "Office",
            "--service-type",
            "Helicopter",
            "--ride-time",
            RIDE_TIME,
        ],
    )

    assert result.exit_code == 1


# ---------------------------------------------------------------------------
# 3. get-quotation saves to quotation_history in state
# ---------------------------------------------------------------------------


def test_get_quotation_saves_to_history(cli_env, fixed_time):
    state_dir, runner = cli_env
    seed_state(state_dir, APP_NAME, EMPTY_STATE)

    result = runner.invoke(
        cli,
        [
            "get-quotation",
            "--start-location",
            "Home",
            "--end-location",
            "Office",
            "--service-type",
            "Default",
            "--ride-time",
            RIDE_TIME,
        ],
    )

    assert result.exit_code == 0, result.output

    # Read persisted state and verify quotation_history is populated
    import json

    from gaia2_cli.shared import normalize_app_name

    state_path = state_dir / (normalize_app_name(APP_NAME) + ".json")
    saved = json.loads(state_path.read_text())
    assert len(saved["quotation_history"]) == 1
    assert saved["quotation_history"][0]["service_type"] == "Default"


# ---------------------------------------------------------------------------
# 4. list-rides: returns quotations for all service types
# ---------------------------------------------------------------------------


def test_list_rides(cli_env, fixed_time):
    state_dir, runner = cli_env
    seed_state(state_dir, APP_NAME, EMPTY_STATE)

    result = runner.invoke(
        cli,
        [
            "list-rides",
            "--start-location",
            "Home",
            "--end-location",
            "Airport",
            "--ride-time",
            RIDE_TIME,
        ],
    )

    assert result.exit_code == 0, result.output
    data = parse_output(result)
    assert data["status"] == "success"
    rides = data["data"]
    assert len(rides) == 3
    service_types = {r["service_type"] for r in rides}
    assert service_types == {"Default", "Premium", "Van"}

    # All rides share the same distance (same start/end)
    distances = {r["distance_km"] for r in rides}
    assert len(distances) == 1


# ---------------------------------------------------------------------------
# 5. order-ride happy path
# ---------------------------------------------------------------------------


def test_order_ride_happy(cli_env, fixed_time):
    state_dir, runner = cli_env
    seed_state(state_dir, APP_NAME, EMPTY_STATE)

    result = runner.invoke(
        cli,
        [
            "order-ride",
            "--start-location",
            "Home",
            "--end-location",
            "Mall",
            "--service-type",
            "Premium",
            "--ride-time",
            RIDE_TIME,
        ],
    )

    assert result.exit_code == 0, result.output
    data = parse_output(result)
    assert data["status"] == "success"
    ride = data["data"]
    assert ride["status"] == "BOOKED"
    assert ride["service_type"] == "Premium"
    # Delay must not be visible to the user (matches original Gaia2 app behavior)
    assert "delay" not in ride
    assert "delay_history" not in ride

    # Verify persisted state
    import json

    from gaia2_cli.shared import normalize_app_name

    state_path = state_dir / (normalize_app_name(APP_NAME) + ".json")
    saved = json.loads(state_path.read_text())
    assert len(saved["ride_history"]) == 1
    assert saved["ride_history"][0]["status"] == "BOOKED"
    assert saved["on_going_ride"] is not None
    assert saved["on_going_ride"]["ride_id"] == ride["ride_id"]
    # Delay is hidden from output but still persisted in state for ENV events
    assert saved["on_going_ride"]["delay"] is not None
    assert len(saved["on_going_ride"]["delay_history"]) > 0

    # Check event
    events = read_events(state_dir)
    order_events = [e for e in events if e["fn"] == "order_ride"]
    assert len(order_events) == 1
    assert_event(order_events[0], APP_NAME, "order_ride", write=True)


# ---------------------------------------------------------------------------
# 6. order-ride while ongoing ride exists
# ---------------------------------------------------------------------------


def test_order_ride_while_ongoing(cli_env, fixed_time):
    state_dir, runner = cli_env
    seed_state(state_dir, APP_NAME, EMPTY_STATE)

    # First order
    runner.invoke(
        cli,
        [
            "order-ride",
            "--start-location",
            "Home",
            "--end-location",
            "Mall",
            "--service-type",
            "Default",
            "--ride-time",
            RIDE_TIME,
        ],
    )

    # Second order should fail
    result = runner.invoke(
        cli,
        [
            "order-ride",
            "--start-location",
            "Home",
            "--end-location",
            "Park",
            "--service-type",
            "Default",
            "--ride-time",
            RIDE_TIME,
        ],
    )

    assert result.exit_code == 1
    assert (
        "on-going ride" in result.stderr.lower()
        or "on-going ride" in result.output.lower()
    )


# ---------------------------------------------------------------------------
# 7. user-cancel-ride happy path
# ---------------------------------------------------------------------------


def test_user_cancel_ride_happy(cli_env, fixed_time):
    state_dir, runner = cli_env
    seed_state(state_dir, APP_NAME, EMPTY_STATE)

    # First order a ride
    order_result = runner.invoke(
        cli,
        [
            "order-ride",
            "--start-location",
            "Home",
            "--end-location",
            "Station",
            "--service-type",
            "Default",
            "--ride-time",
            RIDE_TIME,
        ],
    )
    assert order_result.exit_code == 0

    # Now cancel
    result = runner.invoke(cli, ["user-cancel-ride"])

    assert result.exit_code == 0, result.output
    data = parse_output(result)
    assert data["status"] == "success"
    assert "cancelled" in data["message"].lower()

    # Verify persisted state
    import json

    from gaia2_cli.shared import normalize_app_name

    state_path = state_dir / (normalize_app_name(APP_NAME) + ".json")
    saved = json.loads(state_path.read_text())
    assert saved["on_going_ride"] is None
    assert saved["ride_history"][0]["status"] == "CANCELLED"

    # Check event
    events = read_events(state_dir)
    cancel_events = [e for e in events if e["fn"] == "user_cancel_ride"]
    assert len(cancel_events) == 1
    assert_event(cancel_events[0], APP_NAME, "user_cancel_ride", write=True)


# ---------------------------------------------------------------------------
# 8. user-cancel-ride no ongoing ride
# ---------------------------------------------------------------------------


def test_user_cancel_ride_no_ongoing(cli_env, fixed_time):
    state_dir, runner = cli_env
    seed_state(state_dir, APP_NAME, EMPTY_STATE)

    result = runner.invoke(cli, ["user-cancel-ride"])

    assert result.exit_code == 1


# ---------------------------------------------------------------------------
# 9. get-current-ride-status with ongoing ride
# ---------------------------------------------------------------------------


def test_get_current_ride_status_happy(cli_env, fixed_time):
    state_dir, runner = cli_env
    seed_state(state_dir, APP_NAME, EMPTY_STATE)

    # Order a ride first
    runner.invoke(
        cli,
        [
            "order-ride",
            "--start-location",
            "Home",
            "--end-location",
            "Gym",
            "--service-type",
            "Van",
            "--ride-time",
            RIDE_TIME,
        ],
    )

    result = runner.invoke(cli, ["get-current-ride-status"])

    assert result.exit_code == 0, result.output
    data = parse_output(result)
    assert data["status"] == "success"
    ride = data["data"]
    assert ride["start_location"] == "Home"
    assert ride["end_location"] == "Gym"
    assert ride["service_type"] == "Van"

    # Check event
    events = read_events(state_dir)
    status_events = [e for e in events if e["fn"] == "get_current_ride_status"]
    assert len(status_events) == 1
    assert_event(status_events[0], APP_NAME, "get_current_ride_status", write=False)


# ---------------------------------------------------------------------------
# 10. get-current-ride-status no ride
# ---------------------------------------------------------------------------


def test_get_current_ride_status_no_ride(cli_env, fixed_time):
    state_dir, runner = cli_env
    seed_state(state_dir, APP_NAME, EMPTY_STATE)

    result = runner.invoke(cli, ["get-current-ride-status"])

    assert result.exit_code == 1


# ---------------------------------------------------------------------------
# 11. get-ride by index happy path
# ---------------------------------------------------------------------------


def test_get_ride_by_index(cli_env, fixed_time):
    state_dir, runner = cli_env
    seed_state(state_dir, APP_NAME, EMPTY_STATE)

    # Order a ride to populate ride_history
    order_result = runner.invoke(
        cli,
        [
            "order-ride",
            "--start-location",
            "A",
            "--end-location",
            "B",
            "--service-type",
            "Default",
            "--ride-time",
            RIDE_TIME,
        ],
    )
    assert order_result.exit_code == 0
    ordered_ride = parse_output(order_result)["data"]

    # Cancel so we can order another
    runner.invoke(cli, ["user-cancel-ride"])

    # Get ride at index 0
    result = runner.invoke(cli, ["get-ride", "--idx", "0"])

    assert result.exit_code == 0, result.output
    data = parse_output(result)
    assert data["status"] == "success"
    assert data["data"]["ride_id"] == ordered_ride["ride_id"]

    # Check event
    events = read_events(state_dir)
    get_events = [e for e in events if e["fn"] == "get_ride"]
    assert len(get_events) == 1
    assert_event(get_events[0], APP_NAME, "get_ride", write=False)


# ---------------------------------------------------------------------------
# 12. get-ride-history pagination
# ---------------------------------------------------------------------------


def test_get_ride_history_pagination(cli_env, fixed_time):
    state_dir, runner = cli_env

    # Seed state with 3 rides already in history
    import copy
    import hashlib

    state = copy.deepcopy(EMPTY_STATE)
    rides = []
    for i in range(3):
        ride_id = hashlib.md5(f"ride{i}".encode()).hexdigest()
        rides.append(
            {
                "ride_id": ride_id,
                "status": "COMPLETED",
                "service_type": "Default",
                "start_location": f"Loc{i}",
                "end_location": f"Dest{i}",
                "price": 10.0 + i,
                "duration": 15.0,
                "time_stamp": 1705312800.0 + i * 3600,
                "distance_km": 10.0,
                "delay": 6.0,
                "delay_history": [],
            }
        )
    state["ride_history"] = rides
    seed_state(state_dir, APP_NAME, state)

    # Get with offset=1, limit=1 -> should return only the second ride
    result = runner.invoke(cli, ["get-ride-history", "--offset", "1", "--limit", "1"])

    assert result.exit_code == 0, result.output
    data = parse_output(result)
    assert data["status"] == "success"
    history = data["data"]
    assert history["total"] == 3
    assert history["range"] == [1, 2]
    assert len(history["rides"]) == 1
    assert "1" in history["rides"]
    assert history["rides"]["1"]["ride_id"] == rides[1]["ride_id"]


# ---------------------------------------------------------------------------
# 13. get-ride-history-length
# ---------------------------------------------------------------------------


def test_get_ride_history_length(cli_env, fixed_time):
    state_dir, runner = cli_env

    import copy
    import hashlib

    state = copy.deepcopy(EMPTY_STATE)
    rides = []
    for i in range(5):
        ride_id = hashlib.md5(f"ride{i}".encode()).hexdigest()
        rides.append(
            {
                "ride_id": ride_id,
                "status": "COMPLETED",
                "service_type": "Default",
                "start_location": f"Loc{i}",
                "end_location": f"Dest{i}",
                "price": 10.0,
                "duration": 15.0,
                "time_stamp": 1705312800.0 + i * 3600,
                "distance_km": 10.0,
                "delay": 6.0,
                "delay_history": [],
            }
        )
    state["ride_history"] = rides
    seed_state(state_dir, APP_NAME, state)

    result = runner.invoke(cli, ["get-ride-history-length"])

    assert result.exit_code == 0, result.output
    data = parse_output(result)
    assert data["status"] == "success"
    assert data["data"] == 5

    # Check event
    events = read_events(state_dir)
    length_events = [e for e in events if e["fn"] == "get_ride_history_length"]
    assert len(length_events) == 1
    assert_event(length_events[0], APP_NAME, "get_ride_history_length", write=False)
