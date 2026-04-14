# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
"""Unit tests for gaia2_cli/city_app.py (CityApp CLI)."""

import json

import pytest
from conftest import assert_event, parse_output, read_events, seed_state
from gaia2_cli.apps.city import cli
from gaia2_cli.shared import set_app

APP = "CityApp"


@pytest.fixture(autouse=True)
def _reset_app_name():
    """Ensure shared._APP_NAME is always CityApp for every test."""
    set_app(APP)


# ---------------------------------------------------------------------------
# 1. get-crime-rate happy path
# ---------------------------------------------------------------------------


def test_get_crime_rate_happy_path(cli_env):
    state_dir, runner = cli_env
    seed_state(
        state_dir,
        APP,
        {
            "crime_data": {
                "12345": {"violent_crime": 0.5, "property_crime": 0.3},
            },
            "api_call_count": 0,
            "api_call_limit": 100,
        },
    )
    result = runner.invoke(cli, ["get-crime-rate", "--zip-code", "12345"])
    assert result.exit_code == 0, result.output + (result.stderr or "")
    data = parse_output(result)
    assert data["status"] == "success"
    assert data["data"]["violent_crime"] == 0.5
    assert data["data"]["property_crime"] == 0.3


# ---------------------------------------------------------------------------
# 2. get-crime-rate unknown zip code
# ---------------------------------------------------------------------------


def test_get_crime_rate_unknown_zip(cli_env):
    state_dir, runner = cli_env
    seed_state(
        state_dir,
        APP,
        {
            "crime_data": {
                "12345": {"violent_crime": 0.5, "property_crime": 0.3},
            },
            "api_call_count": 0,
            "api_call_limit": 100,
        },
    )
    result = runner.invoke(cli, ["get-crime-rate", "--zip-code", "99999"])
    assert result.exit_code == 1
    assert "Zip code does not exist" in (result.stderr or result.output)


# ---------------------------------------------------------------------------
# 3. get-crime-rate increments api_call_count
# ---------------------------------------------------------------------------


def test_get_crime_rate_increments_api_call_count(cli_env):
    state_dir, runner = cli_env
    seed_state(
        state_dir,
        APP,
        {
            "crime_data": {
                "12345": {"violent_crime": 0.1, "property_crime": 0.2},
            },
            "api_call_count": 5,
            "api_call_limit": 100,
        },
    )
    result = runner.invoke(cli, ["get-crime-rate", "--zip-code", "12345"])
    assert result.exit_code == 0, result.output + (result.stderr or "")

    # Read persisted state and confirm api_call_count incremented from 5 to 6
    state_path = state_dir / "city_app.json"
    with open(state_path) as f:
        state = json.load(f)
    assert state["api_call_count"] == 6


# ---------------------------------------------------------------------------
# 4. Rate limiting: api_call_count at limit -> error
# ---------------------------------------------------------------------------


def test_get_crime_rate_rate_limited(cli_env, fixed_time):
    state_dir, runner = cli_env
    fixed_time(1522479600.0)
    seed_state(
        state_dir,
        APP,
        {
            "crime_data": {
                "12345": {"violent_crime": 0.1, "property_crime": 0.2},
            },
            "api_call_count": 100,
            "api_call_limit": 100,
        },
    )
    result = runner.invoke(cli, ["get-crime-rate", "--zip-code", "12345"])
    assert result.exit_code == 1
    stderr = result.stderr or result.output
    assert "100 API calls" in stderr
    assert "30 minutes" in stderr


# ---------------------------------------------------------------------------
# 5. Rate limit cooldown: old rate_limit_time resets and call succeeds
# ---------------------------------------------------------------------------


def test_rate_limit_cooldown_expired(cli_env, fixed_time):
    state_dir, runner = cli_env
    now = 1522479600.0
    fixed_time(now)
    seed_state(
        state_dir,
        APP,
        {
            "crime_data": {
                "12345": {"violent_crime": 0.4, "property_crime": 0.6},
            },
            "api_call_count": 0,
            "api_call_limit": 100,
            # rate_limit_time is well over 1800s ago -> should reset
            "rate_limit_time": now - 2000,
            "rate_limit_exceeded": True,
        },
    )
    result = runner.invoke(cli, ["get-crime-rate", "--zip-code", "12345"])
    assert result.exit_code == 0, result.output + (result.stderr or "")
    data = parse_output(result)
    assert data["status"] == "success"
    assert data["data"]["violent_crime"] == 0.4


# ---------------------------------------------------------------------------
# 6. get-api-call-count returns the count from state
# ---------------------------------------------------------------------------


def test_get_api_call_count(cli_env):
    state_dir, runner = cli_env
    seed_state(state_dir, APP, {"api_call_count": 42})
    result = runner.invoke(cli, ["get-api-call-count"])
    assert result.exit_code == 0, result.output + (result.stderr or "")
    data = parse_output(result)
    assert data["status"] == "success"
    assert data["data"] == 42


# ---------------------------------------------------------------------------
# 7. get-api-call-limit returns default 100 and custom value
# ---------------------------------------------------------------------------


def test_get_api_call_limit_default(cli_env):
    state_dir, runner = cli_env
    seed_state(state_dir, APP, {})
    result = runner.invoke(cli, ["get-api-call-limit"])
    assert result.exit_code == 0, result.output + (result.stderr or "")
    data = parse_output(result)
    assert data["status"] == "success"
    assert data["data"] == 100


def test_get_api_call_limit_custom(cli_env):
    state_dir, runner = cli_env
    seed_state(state_dir, APP, {"api_call_limit": 250})
    result = runner.invoke(cli, ["get-api-call-limit"])
    assert result.exit_code == 0, result.output + (result.stderr or "")
    data = parse_output(result)
    assert data["status"] == "success"
    assert data["data"] == 250


# ---------------------------------------------------------------------------
# 8. Event format: app=CityApp, fn=get_crime_rate, w=False for reads
# ---------------------------------------------------------------------------


def test_event_format_get_crime_rate(cli_env, fixed_time):
    state_dir, runner = cli_env
    fixed_time(1522479600.0)
    seed_state(
        state_dir,
        APP,
        {
            "crime_data": {
                "12345": {"violent_crime": 0.5, "property_crime": 0.3},
            },
            "api_call_count": 0,
            "api_call_limit": 100,
        },
    )
    result = runner.invoke(cli, ["get-crime-rate", "--zip-code", "12345"])
    assert result.exit_code == 0, result.output + (result.stderr or "")

    events = read_events(state_dir)
    assert len(events) == 1
    ev = events[0]
    assert_event(ev, app="CityApp", fn="get_crime_rate", write=False)
    assert ev["args"] == {"zip_code": "12345"}
    assert ev["ret"] == {"violent_crime": 0.5, "property_crime": 0.3}
    assert ev["t"] == 1522479600.0
