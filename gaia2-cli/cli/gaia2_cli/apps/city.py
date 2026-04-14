# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
"""
CityApp — standalone CLI for city information (crime rates).

Binary: city
"""

import time
from dataclasses import asdict, dataclass
from typing import Any

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

APP_NAME = "CityApp"
_RATE_LIMIT_COOLDOWN_SECONDS = 1800  # 30 minutes

set_app(APP_NAME)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class CrimeDataPoint:
    violent_crime: float
    property_crime: float


def _load_crime_data(state: dict[str, Any]) -> dict[str, CrimeDataPoint]:
    """Load crime data from state dict."""
    crime_data: dict[str, CrimeDataPoint] = {}
    raw = state.get("crime_data", {})
    for zip_code, data in raw.items():
        crime_data[zip_code] = CrimeDataPoint(**data)
    return crime_data


def _get_api_call_limit(state: dict[str, Any]) -> int:
    """Get API call limit from state, default 100."""
    return state.get("api_call_limit", 100)


def _reset_rate_limit_if_expired(state: dict[str, Any]) -> None:
    """Reset rate limit if cooldown period has passed."""
    rate_limit_time = state.get("rate_limit_time")
    if rate_limit_time is not None:
        if time.time() - rate_limit_time >= _RATE_LIMIT_COOLDOWN_SECONDS:
            state["rate_limit_time"] = None
            state["rate_limit_exceeded"] = False


def _is_rate_limited(state: dict[str, Any]) -> bool:
    """Check if API calls are currently rate limited."""
    api_call_count = state.get("api_call_count", 0)
    api_call_limit = _get_api_call_limit(state)
    rate_limit_time = state.get("rate_limit_time")

    # First time hitting the limit
    if api_call_count >= api_call_limit and rate_limit_time is None:
        return True

    # Still in cooldown period
    if rate_limit_time is not None:
        time_since_limit = time.time() - rate_limit_time
        return time_since_limit < _RATE_LIMIT_COOLDOWN_SECONDS

    return False


def _enforce_rate_limit(state: dict[str, Any]) -> None:
    """Enforce rate limiting by setting state and raising via cli_error."""
    if state.get("rate_limit_time") is None:
        state["rate_limit_time"] = time.time()

    state["api_call_count"] = 0
    state["rate_limit_exceeded"] = True
    save_app_state(APP_NAME, state)

    api_call_limit = _get_api_call_limit(state)
    cli_error(
        f"Free version only supports {api_call_limit} API calls. "
        f"Please try again after 30 minutes or upgrade to pro-version."
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@click.group(context_settings={"terminal_width": 10000, "max_content_width": 10000})
def cli():
    """CityApp - city information app."""
    pass


@cli.command("schema")
def schema_cmd():
    """Output machine-readable JSON schema of commands."""
    json_output(build_schema(cli))


@cli.command("get-crime-rate")
@click.option("--zip-code", required=True, help="Zip code to get crime rate for")
def get_crime_rate(zip_code: str):
    """Get crime rate for a given zip code (rate-limited)."""
    state = load_app_state(APP_NAME) or {}
    crime_data = _load_crime_data(state)

    # Reset rate limit if cooldown expired, then check/enforce
    _reset_rate_limit_if_expired(state)
    if _is_rate_limited(state):
        _enforce_rate_limit(state)

    state["api_call_count"] = state.get("api_call_count", 0) + 1
    save_app_state(APP_NAME, state)

    if zip_code not in crime_data:
        cli_error("Zip code does not exist in our database")

    result = asdict(crime_data[zip_code])
    args = {"zip_code": zip_code}
    log_action("get_crime_rate", args, ret=result, write=False)
    json_output({"status": "success", "data": result})


@cli.command("get-api-call-count")
def get_api_call_count():
    """Get the current API call count."""
    state = load_app_state(APP_NAME) or {}
    result = state.get("api_call_count", 0)
    log_action("get_api_call_count", {}, ret=result, write=False)
    json_output({"status": "success", "data": result})


@cli.command("get-api-call-limit")
def get_api_call_limit():
    """Get the API call limit for the service."""
    state = load_app_state(APP_NAME) or {}
    result = _get_api_call_limit(state)
    log_action("get_api_call_limit", {}, ret=result, write=False)
    json_output({"status": "success", "data": result})


# ===========================================================================
# Hidden ENV-tool commands (used by gaia2-eventd, not visible to agents)
# ===========================================================================


@cli.command("add-crime-rate", hidden=True)
@click.option("--zip-code", required=True, help="Zip code.")
@click.option(
    "--violent-crime-rate", required=True, type=float, help="Violent crime rate."
)
@click.option(
    "--property-crime-rate", required=True, type=float, help="Property crime rate."
)
def env_add_crime_rate(
    zip_code: str, violent_crime_rate: float, property_crime_rate: float
):
    """[ENV] Add crime rate data for a zip code."""
    state = load_app_state(APP_NAME) or {}
    crime_data = _load_crime_data(state)

    crime_data[zip_code] = CrimeDataPoint(
        violent_crime=violent_crime_rate,
        property_crime=property_crime_rate,
    )

    state["crime_data"] = {zc: asdict(dp) for zc, dp in crime_data.items()}
    save_app_state(APP_NAME, state)

    log_action(
        "add_crime_rate",
        {
            "zip_code": zip_code,
            "violent_crime_rate": violent_crime_rate,
            "property_crime_rate": property_crime_rate,
        },
        ret="Added Successfully",
        write=True,
    )
    json_output({"status": "success", "message": "Added Successfully"})


@cli.command("update-crime-rate", hidden=True)
@click.option("--zip-code", required=True, help="Zip code.")
@click.option(
    "--new-violent-crime-rate", default=None, type=float, help="New violent crime rate."
)
@click.option(
    "--new-property-crime-rate",
    default=None,
    type=float,
    help="New property crime rate.",
)
def env_update_crime_rate(
    zip_code: str,
    new_violent_crime_rate: float | None,
    new_property_crime_rate: float | None,
):
    """[ENV] Update crime rate data for a zip code."""
    if new_violent_crime_rate is None and new_property_crime_rate is None:
        cli_error("No update provided")

    state = load_app_state(APP_NAME) or {}
    crime_data = _load_crime_data(state)

    if zip_code not in crime_data:
        cli_error("Zip code does not exist in our database")

    dp = crime_data[zip_code]
    if new_violent_crime_rate is not None:
        dp.violent_crime = new_violent_crime_rate
    if new_property_crime_rate is not None:
        dp.property_crime = new_property_crime_rate

    state["crime_data"] = {zc: asdict(d) for zc, d in crime_data.items()}
    save_app_state(APP_NAME, state)

    log_action(
        "update_crime_rate",
        {
            "zip_code": zip_code,
            "new_violent_crime_rate": new_violent_crime_rate,
            "new_property_crime_rate": new_property_crime_rate,
        },
        ret="Updated Successfully",
        write=True,
    )
    json_output({"status": "success", "message": "Updated Successfully"})


def main():
    cli()


if __name__ == "__main__":
    main()
