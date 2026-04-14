# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
"""Unit tests for gaia2_cli/apartment_app.py (RentAFlat CLI)."""

import pytest
from conftest import assert_event, parse_output, read_events, seed_state
from gaia2_cli.apps.apartment import cli
from gaia2_cli.shared import set_app

APP_NAME = "RentAFlat"

# ---------------------------------------------------------------------------
# Sample state
# ---------------------------------------------------------------------------

_APT1 = {
    "name": "Sunny Studio",
    "location": "Downtown",
    "zip_code": "10001",
    "price": 1500.0,
    "bedrooms": 1,
    "bathrooms": 1,
    "property_type": "Apartment",
    "square_footage": 500,
    "furnished_status": "Furnished",
    "floor_level": "Upper floors",
    "pet_policy": "Pets allowed",
    "lease_term": "1 year",
    "amenities": ["WiFi", "Gym"],
}

_APT2 = {
    "name": "Cozy Loft",
    "location": "Suburbs",
    "zip_code": "10002",
    "price": 2500.0,
    "bedrooms": 2,
    "bathrooms": 2,
    "property_type": "Condo",
    "square_footage": 800,
    "furnished_status": "Unfurnished",
    "floor_level": "Penthouse",
    "pet_policy": "No pets",
    "lease_term": "6 months",
    "amenities": ["Pool", "Parking"],
}


def _two_apt_state(saved=None):
    """Return a state dict with two apartments and an optional saved list."""
    return {
        "apartments": {"apt1": _APT1, "apt2": _APT2},
        "saved_apartments": saved or [],
    }


# ---------------------------------------------------------------------------
# Autouse fixture — reset the shared module's app name before each test
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _set_app():
    set_app(APP_NAME)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestListAllApartments:
    def test_returns_both_apartments(self, cli_env, fixed_time):
        state_dir, runner = cli_env
        seed_state(state_dir, APP_NAME, _two_apt_state())

        result = runner.invoke(cli, ["list-all-apartments"])

        assert result.exit_code == 0
        data = parse_output(result)
        assert data["status"] == "success"
        assert "apt1" in data["data"]
        assert "apt2" in data["data"]
        assert data["data"]["apt1"]["name"] == "Sunny Studio"
        assert data["data"]["apt2"]["name"] == "Cozy Loft"

        events = read_events(state_dir)
        assert len(events) == 1
        assert_event(events[0], APP_NAME, "list_all_apartments", write=False)


class TestGetApartmentDetails:
    def test_happy_path(self, cli_env, fixed_time):
        state_dir, runner = cli_env
        seed_state(state_dir, APP_NAME, _two_apt_state())

        result = runner.invoke(cli, ["get-apartment-details", "--apartment-id", "apt1"])

        assert result.exit_code == 0
        data = parse_output(result)
        assert data["status"] == "success"
        assert data["data"]["name"] == "Sunny Studio"
        assert data["data"]["apartment_id"] == "apt1"

        events = read_events(state_dir)
        assert len(events) == 1
        assert_event(events[0], APP_NAME, "get_apartment_details", write=False)
        assert events[0]["args"]["apartment_id"] == "apt1"

    def test_not_found(self, cli_env, fixed_time):
        state_dir, runner = cli_env
        seed_state(state_dir, APP_NAME, _two_apt_state())

        result = runner.invoke(cli, ["get-apartment-details", "--apartment-id", "nope"])

        assert result.exit_code == 1


class TestSaveApartment:
    def test_happy_path(self, cli_env, fixed_time):
        state_dir, runner = cli_env
        seed_state(state_dir, APP_NAME, _two_apt_state())

        result = runner.invoke(cli, ["save-apartment", "--apartment-id", "apt1"])

        assert result.exit_code == 0
        data = parse_output(result)
        assert data["status"] == "success"

        events = read_events(state_dir)
        assert len(events) == 1
        assert_event(events[0], APP_NAME, "save_apartment", write=True)

        # Verify saved_apartments list was updated in state
        import json

        state_path = state_dir / "rent_a_flat.json"
        persisted = json.loads(state_path.read_text())
        assert "apt1" in persisted["saved_apartments"]

    def test_idempotent(self, cli_env, fixed_time):
        state_dir, runner = cli_env
        seed_state(state_dir, APP_NAME, _two_apt_state())

        # Save twice
        runner.invoke(cli, ["save-apartment", "--apartment-id", "apt1"])
        result = runner.invoke(cli, ["save-apartment", "--apartment-id", "apt1"])

        assert result.exit_code == 0

        # Should not duplicate in the saved list
        import json

        state_path = state_dir / "rent_a_flat.json"
        persisted = json.loads(state_path.read_text())
        assert persisted["saved_apartments"].count("apt1") == 1


class TestRemoveSavedApartment:
    def test_happy_path(self, cli_env, fixed_time):
        state_dir, runner = cli_env
        seed_state(state_dir, APP_NAME, _two_apt_state(saved=["apt1"]))

        result = runner.invoke(
            cli, ["remove-saved-apartment", "--apartment-id", "apt1"]
        )

        assert result.exit_code == 0
        data = parse_output(result)
        assert data["status"] == "success"

        events = read_events(state_dir)
        assert len(events) == 1
        assert_event(events[0], APP_NAME, "remove_saved_apartment", write=True)

        # Verify state
        import json

        state_path = state_dir / "rent_a_flat.json"
        persisted = json.loads(state_path.read_text())
        assert "apt1" not in persisted["saved_apartments"]

    def test_not_in_saved(self, cli_env, fixed_time):
        state_dir, runner = cli_env
        seed_state(state_dir, APP_NAME, _two_apt_state())

        result = runner.invoke(
            cli, ["remove-saved-apartment", "--apartment-id", "apt1"]
        )

        assert result.exit_code == 1


class TestListSavedApartments:
    def test_returns_only_saved(self, cli_env, fixed_time):
        state_dir, runner = cli_env
        seed_state(state_dir, APP_NAME, _two_apt_state(saved=["apt2"]))

        result = runner.invoke(cli, ["list-saved-apartments"])

        assert result.exit_code == 0
        data = parse_output(result)
        assert data["status"] == "success"
        assert "apt2" in data["data"]
        assert "apt1" not in data["data"]
        assert data["data"]["apt2"]["name"] == "Cozy Loft"

        events = read_events(state_dir)
        assert len(events) == 1
        assert_event(events[0], APP_NAME, "list_saved_apartments", write=False)


class TestSearchApartments:
    def test_by_location_case_insensitive(self, cli_env, fixed_time):
        state_dir, runner = cli_env
        seed_state(state_dir, APP_NAME, _two_apt_state())

        result = runner.invoke(cli, ["search-apartments", "--location", "downtown"])

        assert result.exit_code == 0
        data = parse_output(result)
        assert data["status"] == "success"
        assert "apt1" in data["data"]
        assert "apt2" not in data["data"]

    def test_by_price_range(self, cli_env, fixed_time):
        state_dir, runner = cli_env
        seed_state(state_dir, APP_NAME, _two_apt_state())

        result = runner.invoke(
            cli,
            ["search-apartments", "--min-price", "2000", "--max-price", "3000"],
        )

        assert result.exit_code == 0
        data = parse_output(result)
        assert data["status"] == "success"
        # Only apt2 (price=2500) is in range [2000, 3000]
        assert "apt2" in data["data"]
        assert "apt1" not in data["data"]

    def test_by_amenities(self, cli_env, fixed_time):
        state_dir, runner = cli_env
        seed_state(state_dir, APP_NAME, _two_apt_state())

        result = runner.invoke(cli, ["search-apartments", "--amenities", "Pool"])

        assert result.exit_code == 0
        data = parse_output(result)
        assert data["status"] == "success"
        # Only apt2 has Pool
        assert "apt2" in data["data"]
        assert "apt1" not in data["data"]

    def test_saved_only_flag(self, cli_env, fixed_time):
        state_dir, runner = cli_env
        seed_state(state_dir, APP_NAME, _two_apt_state(saved=["apt1"]))

        result = runner.invoke(cli, ["search-apartments", "--saved-only"])

        assert result.exit_code == 0
        data = parse_output(result)
        assert data["status"] == "success"
        # Only apt1 is saved, so only it should appear
        assert "apt1" in data["data"]
        assert "apt2" not in data["data"]

        events = read_events(state_dir)
        assert len(events) == 1
        assert_event(events[0], APP_NAME, "search_apartments", write=False)
        assert events[0]["args"]["saved_only"] is True
