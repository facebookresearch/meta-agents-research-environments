# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
"""
CabApp — standalone CLI for the cab/ride service.

Binary: cabs
"""

import dataclasses
import hashlib
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
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

APP_NAME = "CabApp"

set_app(APP_NAME)

DEFAULT_SERVICE_CONFIG: dict[str, dict[str, float]] = {
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
}


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class Ride:
    ride_id: str = ""
    status: str | None = None
    service_type: str | None = None
    start_location: str | None = None
    end_location: str | None = None
    price: float | None = None
    duration: float | None = None
    time_stamp: float | None = None
    distance_km: float | None = None
    delay: float | None = None
    delay_history: list[dict] = field(default_factory=list)


# ---------------------------------------------------------------------------
# State helpers
# ---------------------------------------------------------------------------


def _load_state() -> dict[str, Any]:
    """Load full CabApp state from disk."""
    state = load_app_state(APP_NAME) or {}
    return state


_RIDE_FIELDS = {f.name for f in dataclasses.fields(Ride)}


def _filter_ride(raw: dict) -> Ride:
    """Build a Ride from a dict, ignoring unknown keys."""
    return Ride(**{k: v for k, v in raw.items() if k in _RIDE_FIELDS})


def _load_rides(raw_list: list[dict]) -> list[Ride]:
    """Parse a list of ride dicts into Ride objects."""
    return [_filter_ride(rd) for rd in raw_list]


def _build_full_state(
    ride_history: list[Ride],
    quotation_history: list[Ride],
    d_service_config: dict[str, dict[str, float]],
    on_going_ride: Ride | None,
) -> dict[str, Any]:
    """Reconstruct the full state dict."""
    result: dict[str, Any] = {
        "ride_history": [asdict(r) for r in ride_history],
        "quotation_history": [asdict(r) for r in quotation_history],
        "d_service_config": d_service_config,
    }
    if on_going_ride is not None:
        result["on_going_ride"] = asdict(on_going_ride)
    else:
        result["on_going_ride"] = None
    return result


def _parse_ride_time(ride_time: str | None) -> tuple[str, float]:
    """Parse ride_time string, default to current system time."""
    if ride_time is None:
        ride_time = datetime.fromtimestamp(
            time.time(),
            tz=timezone.utc,
        ).strftime("%Y-%m-%d %H:%M:%S")

    try:
        ts = (
            datetime.strptime(ride_time, "%Y-%m-%d %H:%M:%S")
            .replace(tzinfo=timezone.utc)
            .timestamp()
        )
    except ValueError:
        raise ValueError(
            "Invalid datetime format for the ride time. Please use YYYY-MM-DD HH:MM:SS"
        )
    return ride_time, ts


def _deterministic_distance(start_location: str, end_location: str) -> float:
    """Deterministic distance from location names (5-20 km range)."""
    h = int(hashlib.md5((start_location + end_location).encode()).hexdigest(), 16)
    return (h % 1500) / 100.0 + 5.0


def _calculate_distance(
    start_location: str,
    end_location: str,
    quotation_history: list[Ride],
) -> float:
    """Look up distance from quotation history; fall back to deterministic hash."""
    for ride in quotation_history:
        if ride.start_location == start_location and ride.end_location == end_location:
            return ride.distance_km
    return _deterministic_distance(start_location, end_location)


def _calculate_price(
    start_location: str,
    end_location: str,
    distance_km: float,
    service_type: str,
    time_stamp: float,
    quotation_history: list[Ride],
    d_service_config: dict[str, dict[str, float]],
) -> float:
    """Calculate ride price, applying variance if a previous quotation exists."""
    prev = None
    for ride in quotation_history:
        if (
            ride.start_location == start_location
            and ride.end_location == end_location
            and ride.service_type == service_type
        ):
            prev = ride
            break

    if prev and prev.price:
        variance = 0.01 * (time_stamp - prev.time_stamp) / 3600  # 1% per hour
        variance = min(max(variance, 0.5), 1.5)
        # Deterministic "random" based on hash for reproducibility
        h = int(
            hashlib.md5(
                f"{start_location}{end_location}{service_type}{time_stamp}".encode()
            ).hexdigest(),
            16,
        )
        factor = ((h % 2000) / 1000.0 - 1.0) * variance  # range: -variance to +variance
        price = prev.price * (1 + factor)
    else:
        price = distance_km * d_service_config[service_type]["price_per_km"]
    return price


def _make_ride_id(
    start_location: str, end_location: str, service_type: str, time_stamp: float
) -> str:
    """Generate a deterministic ride ID."""
    h = hashlib.md5(
        f"{start_location}{end_location}{service_type}{time_stamp}".encode()
    ).hexdigest()
    return h


def _get_quotation(
    start_location: str,
    end_location: str,
    service_type: str,
    ride_time: str | None,
    quotation_history: list[Ride],
    d_service_config: dict[str, dict[str, float]],
) -> Ride:
    """Compute a quotation and append to quotation_history (mutates it)."""
    _, time_stamp = _parse_ride_time(ride_time)

    if service_type not in d_service_config:
        raise ValueError("Invalid service type.")

    distance_km = _calculate_distance(start_location, end_location, quotation_history)
    if distance_km > d_service_config[service_type]["max_distance_km"]:
        raise ValueError("Distance exceeds maximum allowed.")

    price = _calculate_price(
        start_location,
        end_location,
        distance_km,
        service_type,
        time_stamp,
        quotation_history,
        d_service_config,
    )
    base_delay = d_service_config[service_type]["base_delay_min"]
    # Deterministic delay offset 1-5
    h = int(
        hashlib.md5(
            f"delay{start_location}{end_location}{service_type}{time_stamp}".encode()
        ).hexdigest(),
        16,
    )
    delay = base_delay + (h % 5) + 1

    # Duration: assume avg speed 50km/h, plus small deterministic extra
    h2 = int(
        hashlib.md5(
            f"dur{start_location}{end_location}{time_stamp}".encode()
        ).hexdigest(),
        16,
    )
    duration = distance_km / 50 * 60 + (h2 % 1000) / 100.0

    ride_id = _make_ride_id(start_location, end_location, service_type, time_stamp)
    ride = Ride(
        ride_id=ride_id,
        service_type=service_type,
        start_location=start_location,
        end_location=end_location,
        price=price,
        duration=duration,
        time_stamp=time_stamp,
        distance_km=distance_km,
        delay=delay,
    )
    quotation_history.append(ride)
    return ride


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@click.group(context_settings={"terminal_width": 10000, "max_content_width": 10000})
def cli():
    """CabApp - cab/ride service."""
    pass


@cli.command("schema")
def schema_cmd():
    """Output machine-readable JSON schema of commands."""
    json_output(build_schema(cli))


@cli.command("get-quotation")
@click.option("--start-location", required=True, help="Starting point of the ride")
@click.option("--end-location", required=True, help="Ending point of the ride")
@click.option(
    "--service-type", required=True, help="Type of service (Default, Premium, Van)"
)
@click.option(
    "--ride-time", default=None, help="Ride time in YYYY-MM-DD HH:MM:SS format"
)
def get_quotation(
    start_location: str, end_location: str, service_type: str, ride_time: str | None
):
    """Calculate price and estimated delay for a ride."""
    state = _load_state()
    ride_history = _load_rides(state.get("ride_history", []))
    quotation_history = _load_rides(state.get("quotation_history", []))
    d_service_config = state.get("d_service_config", DEFAULT_SERVICE_CONFIG)
    on_going_ride_raw = state.get("on_going_ride")
    on_going_ride = _filter_ride(on_going_ride_raw) if on_going_ride_raw else None

    event_args: dict[str, Any] = {
        "start_location": start_location,
        "end_location": end_location,
        "service_type": service_type,
        "ride_time": ride_time,
    }

    try:
        ride = _get_quotation(
            start_location,
            end_location,
            service_type,
            ride_time,
            quotation_history,
            d_service_config,
        )
    except ValueError as e:
        cli_error(str(e))

    # Save updated quotation history
    new_state = _build_full_state(
        ride_history, quotation_history, d_service_config, on_going_ride
    )
    save_app_state(APP_NAME, new_state)

    result = asdict(ride)
    log_action("get_quotation", event_args, ret=result)
    json_output({"status": "success", "data": result})


@cli.command("list-rides")
@click.option("--start-location", required=True, help="Starting point of the ride")
@click.option("--end-location", required=True, help="Ending point of the ride")
@click.option(
    "--ride-time", default=None, help="Ride time in YYYY-MM-DD HH:MM:SS format"
)
def list_rides(start_location: str, end_location: str, ride_time: str | None):
    """List all rides available between two locations."""
    state = _load_state()
    ride_history = _load_rides(state.get("ride_history", []))
    quotation_history = _load_rides(state.get("quotation_history", []))
    d_service_config = state.get("d_service_config", DEFAULT_SERVICE_CONFIG)
    on_going_ride_raw = state.get("on_going_ride")
    on_going_ride = _filter_ride(on_going_ride_raw) if on_going_ride_raw else None

    # Normalize ride_time so all quotations use the same string
    ride_time_str, _ = _parse_ride_time(ride_time)

    event_args: dict[str, Any] = {
        "start_location": start_location,
        "end_location": end_location,
        "ride_time": ride_time,
    }

    all_rides = []
    try:
        for svc in d_service_config:
            ride = _get_quotation(
                start_location,
                end_location,
                svc,
                ride_time_str,
                quotation_history,
                d_service_config,
            )
            all_rides.append(ride)
    except ValueError as e:
        cli_error(str(e))

    new_state = _build_full_state(
        ride_history, quotation_history, d_service_config, on_going_ride
    )
    save_app_state(APP_NAME, new_state)

    result = [asdict(r) for r in all_rides]
    log_action("list_rides", event_args, ret=result)
    json_output({"status": "success", "data": result})


@cli.command("order-ride")
@click.option("--start-location", required=True, help="Starting point of the ride")
@click.option("--end-location", required=True, help="Ending point of the ride")
@click.option(
    "--service-type", required=True, help="Type of service (Default, Premium, Van)"
)
@click.option(
    "--ride-time", default=None, help="Ride time in YYYY-MM-DD HH:MM:SS format"
)
def order_ride(
    start_location: str, end_location: str, service_type: str, ride_time: str | None
):
    """Order a ride and return ride details."""
    state = _load_state()
    ride_history = _load_rides(state.get("ride_history", []))
    quotation_history = _load_rides(state.get("quotation_history", []))
    d_service_config = state.get("d_service_config", DEFAULT_SERVICE_CONFIG)
    on_going_ride_raw = state.get("on_going_ride")
    on_going_ride = _filter_ride(on_going_ride_raw) if on_going_ride_raw else None

    event_args: dict[str, Any] = {
        "start_location": start_location,
        "end_location": end_location,
        "service_type": service_type,
        "ride_time": ride_time,
    }

    if on_going_ride is not None:
        cli_error("You have an on-going ride.")

    ride_time_str, _ = _parse_ride_time(ride_time)

    try:
        ride = _get_quotation(
            start_location,
            end_location,
            service_type,
            ride_time_str,
            quotation_history,
            d_service_config,
        )
    except ValueError as e:
        cli_error(str(e))

    ride.status = "BOOKED"
    ride.delay_history.append({"delay": ride.delay, "time_stamp": ride.time_stamp})
    ride_history.append(ride)
    on_going_ride = ride

    new_state = _build_full_state(
        ride_history, quotation_history, d_service_config, on_going_ride
    )
    save_app_state(APP_NAME, new_state)

    result = asdict(ride)
    # Hide delay from user (matches original Gaia2 app behavior —
    # "the app looks for a cab but the user is not aware of the delay")
    user_result = {
        k: v for k, v in result.items() if k not in ("delay", "delay_history")
    }
    log_action("order_ride", event_args, ret=user_result, write=True)
    json_output({"status": "success", "data": user_result})


@cli.command("user-cancel-ride")
def user_cancel_ride():
    """Cancel the current ride."""
    state = _load_state()
    ride_history = _load_rides(state.get("ride_history", []))
    quotation_history = _load_rides(state.get("quotation_history", []))
    d_service_config = state.get("d_service_config", DEFAULT_SERVICE_CONFIG)
    on_going_ride_raw = state.get("on_going_ride")
    on_going_ride = _filter_ride(on_going_ride_raw) if on_going_ride_raw else None

    if on_going_ride is None:
        cli_error("You have no on-going ride.")

    # Mark the ride as cancelled in history too
    for r in ride_history:
        if r.ride_id == on_going_ride.ride_id:
            r.status = "CANCELLED"
            break
    on_going_ride = None

    new_state = _build_full_state(
        ride_history, quotation_history, d_service_config, on_going_ride
    )
    save_app_state(APP_NAME, new_state)

    message = "Ride has been cancelled, sorry to see you go."
    log_action("user_cancel_ride", {}, ret=message, write=True)
    json_output({"status": "success", "message": message})


@cli.command("get-current-ride-status")
def get_current_ride_status():
    """Check the status of the current ride."""
    state = _load_state()
    on_going_ride_raw = state.get("on_going_ride")
    on_going_ride = _filter_ride(on_going_ride_raw) if on_going_ride_raw else None

    if on_going_ride is None:
        cli_error("No ride ordered.")

    # Update delay (matches original Ride.update_delay logic)
    if on_going_ride.time_stamp is not None and on_going_ride.delay is not None:
        current_ts = time.time()
        delta_time = current_ts - on_going_ride.time_stamp
        # Deterministic delay update using hash
        h = int(
            hashlib.md5(
                f"delay_update{on_going_ride.ride_id}{current_ts}".encode()
            ).hexdigest(),
            16,
        )
        rng_factor = ((h % 2500) / 1000.0) - 1.5  # range: -1.5 to +1.0
        on_going_ride.delay = min(
            0, on_going_ride.delay - delta_time * (1 + rng_factor)
        )
        on_going_ride.delay_history.append(
            {"delay": on_going_ride.delay, "time_stamp": on_going_ride.time_stamp}
        )

        # Persist updated delay to state
        ride_history = _load_rides(state.get("ride_history", []))
        quotation_history = _load_rides(state.get("quotation_history", []))
        d_service_config = state.get("d_service_config", DEFAULT_SERVICE_CONFIG)
        # Update the matching ride in history
        for r in ride_history:
            if r.ride_id == on_going_ride.ride_id:
                r.delay = on_going_ride.delay
                r.delay_history = on_going_ride.delay_history
                break
        new_state = _build_full_state(
            ride_history, quotation_history, d_service_config, on_going_ride
        )
        save_app_state(APP_NAME, new_state)

    result = asdict(on_going_ride)
    log_action("get_current_ride_status", {}, ret=result)
    json_output({"status": "success", "data": result})


@cli.command("get-ride")
@click.option("--idx", required=True, type=int, help="Index of the ride in history")
def get_ride(idx: int):
    """Get a specific ride from ride history by index."""
    state = _load_state()
    ride_history = _load_rides(state.get("ride_history", []))

    if not (0 <= idx < len(ride_history)):
        cli_error("Ride does not exist.")

    result = asdict(ride_history[idx])
    log_action("get_ride", {"idx": idx}, ret=result)
    json_output({"status": "success", "data": result})


@cli.command("get-ride-history")
@click.option("--offset", type=int, default=0, help="Starting index (default 0)")
@click.option(
    "--limit", type=int, default=10, help="Max number of rides to return (default 10)"
)
def get_ride_history(offset: int, limit: int):
    """Get ride history with pagination."""
    state = _load_state()
    ride_history = _load_rides(state.get("ride_history", []))

    subset = ride_history[offset : offset + limit]
    result = {
        "rides": {str(offset + i): asdict(r) for i, r in enumerate(subset)},
        "range": [offset, min(offset + limit, len(ride_history))],
        "total": len(ride_history),
    }

    log_action("get_ride_history", {"offset": offset, "limit": limit}, ret=result)
    json_output({"status": "success", "data": result})


@cli.command("get-ride-history-length")
def get_ride_history_length():
    """Get the total number of rides in history."""
    state = _load_state()
    ride_history = state.get("ride_history", [])
    result = len(ride_history)

    log_action("get_ride_history_length", {}, ret=result)
    json_output({"status": "success", "data": result})


# ===========================================================================
# Hidden ENV-tool commands (used by gaia2-eventd, not visible to agents)
# ===========================================================================


@cli.command("cancel-ride", hidden=True)
@click.option("--who-cancel", default="driver", help="Who cancels: 'driver' or 'user'.")
@click.option("--message", default=None, help="Optional cancellation message.")
def env_cancel_ride(who_cancel: str, message: str | None):
    """[ENV] Cancel the current ride (by driver or user)."""
    if who_cancel not in ["driver", "user"]:
        cli_error("who_cancel must be either 'driver' or 'user'.")

    state = _load_state()
    on_going = state.get("on_going_ride")
    if on_going is None:
        cli_error("No on-going ride to cancel.")

    on_going["status"] = "CANCELLED"
    ride_history = state.get("ride_history", [])
    if ride_history and ride_history[-1].get("ride_id") == on_going.get("ride_id"):
        ride_history[-1]["status"] = "CANCELLED"
    state["on_going_ride"] = None

    save_app_state(APP_NAME, state)

    result = message or "Ride has been cancelled."
    log_action(
        "cancel_ride",
        {"who_cancel": who_cancel, "message": message},
        ret=result,
        write=True,
    )
    json_output({"status": "success", "message": result})


@cli.command("end-ride", hidden=True)
def env_end_ride():
    """[ENV] End the current ride (mark as COMPLETED)."""
    state = _load_state()
    on_going = state.get("on_going_ride")
    if on_going is None:
        cli_error("No on-going ride to end.")

    on_going["status"] = "COMPLETED"
    ride_history = state.get("ride_history", [])
    if ride_history and ride_history[-1].get("ride_id") == on_going.get("ride_id"):
        ride_history[-1]["status"] = "COMPLETED"
    state["on_going_ride"] = None

    save_app_state(APP_NAME, state)

    result = "Ride has been completed."
    log_action("end_ride", {}, ret=result, write=True)
    json_output({"status": "success", "message": result})


@cli.command("update-ride-status", hidden=True)
@click.option(
    "--status",
    "new_status",
    required=True,
    help="New status: DELAYED, IN_PROGRESS, or ARRIVED_AT_PICKUP.",
)
@click.option("--message", default=None, help="Optional driver message.")
def env_update_ride_status(new_status: str, message: str | None):
    """[ENV] Update the status of the current ride."""
    valid = ["DELAYED", "IN_PROGRESS", "ARRIVED_AT_PICKUP"]
    if new_status not in valid:
        cli_error(f"status must be one of {valid}")

    state = _load_state()
    on_going = state.get("on_going_ride")
    if on_going is None:
        cli_error("No on-going ride to update.")

    on_going["status"] = new_status
    ride_history = state.get("ride_history", [])
    if ride_history and ride_history[-1].get("ride_id") == on_going.get("ride_id"):
        ride_history[-1]["status"] = new_status

    save_app_state(APP_NAME, state)

    if message:
        result = f"Ride status has been updated to {new_status}. Message from your driver: {message}"
    else:
        result = f"Ride status has been updated to {new_status}."

    log_action(
        "update_ride_status",
        {"status": new_status, "message": message},
        ret=result,
        write=True,
    )
    json_output({"status": "success", "message": result})


def main():
    cli()


if __name__ == "__main__":
    main()
