# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
"""
RentAFlat — standalone CLI for the apartment listing app.

Binary: rent-a-flat
"""

import dataclasses
import json
from dataclasses import asdict, dataclass, field
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

APP_NAME = "RentAFlat"

set_app(APP_NAME)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class Apartment:
    name: str
    location: str
    zip_code: str
    price: float
    bedrooms: int
    bathrooms: int
    property_type: str
    square_footage: int
    furnished_status: str | None = ""
    floor_level: str | None = ""
    pet_policy: str | None = ""
    lease_term: str | None = ""
    apartment_id: str = ""
    amenities: list[str] | None = field(default_factory=list)
    saved: bool = False


def _load_apartments(state: dict[str, Any]) -> dict[str, Apartment]:
    """Load apartments from state dict into Apartment objects."""
    apartments: dict[str, Apartment] = {}
    raw = state.get("apartments", {})
    valid_fields = {f.name for f in dataclasses.fields(Apartment)}
    for apt_id, apt_data in raw.items():
        apt_data["apartment_id"] = apt_id
        filtered = {k: v for k, v in apt_data.items() if k in valid_fields}
        apartments[apt_id] = Apartment(**filtered)
    return apartments


def _load_saved(state: dict[str, Any], apartments: dict[str, Apartment]) -> list[str]:
    """Load saved apartments list and sync saved flag."""
    saved = state.get("saved_apartments", [])
    for apt_id in saved:
        if apt_id in apartments:
            apartments[apt_id].saved = True
    return saved


def _build_state(
    apartments: dict[str, Apartment], saved_apartments: list[str]
) -> dict[str, Any]:
    """Reconstruct state dict from objects."""
    return {
        "apartments": {aid: asdict(a) for aid, a in apartments.items()},
        "saved_apartments": saved_apartments,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@click.group(context_settings={"terminal_width": 10000, "max_content_width": 10000})
def cli():
    """RentAFlat - apartment listing app."""
    pass


@cli.command("schema")
def schema_cmd():
    """Output machine-readable JSON schema of commands."""
    json_output(build_schema(cli))


@cli.command("list-all-apartments")
def list_all_apartments():
    """List all apartments in the catalog."""
    state = load_app_state(APP_NAME) or {}
    apartments = _load_apartments(state)
    _load_saved(state, apartments)

    result = {aid: asdict(a) for aid, a in apartments.items()}

    log_action("list_all_apartments", {}, ret=result)
    json_output({"status": "success", "data": result})


@cli.command("get-apartment-details")
@click.option("--apartment-id", required=True, help="Apartment ID to look up")
def get_apartment_details(apartment_id: str):
    """Get apartment details for a given apartment id."""
    state = load_app_state(APP_NAME) or {}
    apartments = _load_apartments(state)
    _load_saved(state, apartments)

    if apartment_id not in apartments:
        cli_error("Apartment does not exist")

    result = asdict(apartments[apartment_id])
    log_action("get_apartment_details", {"apartment_id": apartment_id}, ret=result)
    json_output({"status": "success", "data": result})


@cli.command("save-apartment")
@click.option("--apartment-id", required=True, help="Apartment ID to save to favorites")
def save_apartment(apartment_id: str):
    """Save an apartment to favorites."""
    state = load_app_state(APP_NAME) or {}
    apartments = _load_apartments(state)
    saved_apartments = _load_saved(state, apartments)

    if apartment_id not in apartments:
        cli_error("Apartment does not exist")

    if apartment_id not in saved_apartments:
        saved_apartments.append(apartment_id)
        apartments[apartment_id].saved = True

    new_state = _build_state(apartments, saved_apartments)
    save_app_state(APP_NAME, new_state)

    log_action("save_apartment", {"apartment_id": apartment_id}, write=True)
    json_output({"status": "success", "message": "Successfully saved apartment"})


@cli.command("remove-saved-apartment")
@click.option(
    "--apartment-id", required=True, help="Apartment ID to remove from favorites"
)
def remove_saved_apartment(apartment_id: str):
    """Remove an apartment from favorites."""
    state = load_app_state(APP_NAME) or {}
    apartments = _load_apartments(state)
    saved_apartments = _load_saved(state, apartments)

    if apartment_id not in saved_apartments:
        cli_error("Apartment not in saved list")

    saved_apartments.remove(apartment_id)
    apartments[apartment_id].saved = False

    new_state = _build_state(apartments, saved_apartments)
    save_app_state(APP_NAME, new_state)

    log_action("remove_saved_apartment", {"apartment_id": apartment_id}, write=True)
    json_output(
        {
            "status": "success",
            "message": "Successfully removed apartment from favorites",
        }
    )


@cli.command("list-saved-apartments")
def list_saved_apartments():
    """List apartments saved to favorites."""
    state = load_app_state(APP_NAME) or {}
    apartments = _load_apartments(state)
    saved_apartments = _load_saved(state, apartments)

    result = {aid: asdict(apartments[aid]) for aid in saved_apartments}

    log_action("list_saved_apartments", {}, ret=result)
    json_output({"status": "success", "data": result})


@cli.command("search-apartments")
@click.option("--name", default=None, help="Name of the apartment")
@click.option("--location", default=None, help="Desired location")
@click.option("--zip-code", default=None, help="Zip code of the location")
@click.option("--min-price", type=float, default=None, help="Minimum price")
@click.option("--max-price", type=float, default=None, help="Maximum price")
@click.option("--number-of-bedrooms", type=int, default=None, help="Number of bedrooms")
@click.option(
    "--number-of-bathrooms", type=int, default=None, help="Number of bathrooms"
)
@click.option(
    "--property-type",
    default=None,
    help="Type of property (Apartment, Condo, House, Studio)",
)
@click.option("--square-footage", type=int, default=None, help="Minimum square footage")
@click.option(
    "--furnished-status", default=None, help="Furnished, Unfurnished, or Semi-furnished"
)
@click.option(
    "--floor-level",
    default=None,
    help="Ground floor, Upper floors, Penthouse, Basement",
)
@click.option(
    "--pet-policy",
    default=None,
    help="Pets allowed, No pets, Cats allowed, Dogs allowed",
)
@click.option(
    "--lease-term",
    default=None,
    help="Month-to-month, 6 months, 1 year, Long term lease",
)
@click.option(
    "--amenities",
    default=None,
    help="Comma-separated list of desired amenities, or JSON array",
)
@click.option(
    "--saved-only", is_flag=True, default=False, help="Search only saved apartments"
)
def search_apartments(  # noqa: C901
    name,
    location,
    zip_code,
    min_price,
    max_price,
    number_of_bedrooms,
    number_of_bathrooms,
    property_type,
    square_footage,
    furnished_status,
    floor_level,
    pet_policy,
    lease_term,
    amenities,
    saved_only,
):
    """Search for apartments based on multiple filters."""
    state = load_app_state(APP_NAME) or {}
    all_apartments = _load_apartments(state)
    saved_list = _load_saved(state, all_apartments)

    # Parse amenities
    amenities_list: list[str] | None = None
    if amenities is not None:
        try:
            amenities_list = json.loads(amenities)
        except json.JSONDecodeError:
            amenities_list = [a.strip() for a in amenities.split(",")]

    # Determine apartment set to search
    if saved_only:
        apartments_to_search = {aid: all_apartments[aid] for aid in saved_list}
    else:
        apartments_to_search = all_apartments

    # Prepare case-insensitive search terms
    name_lower = name.lower() if name else None
    location_lower = location.lower() if location else None
    property_type_lower = property_type.lower() if property_type else None
    furnished_status_lower = furnished_status.lower() if furnished_status else None
    pet_policy_lower = pet_policy.lower() if pet_policy else None
    lease_term_lower = lease_term.lower() if lease_term else None
    amenities_lower = [am.lower() for am in amenities_list] if amenities_list else None

    # Apply filters
    filtered: dict[str, Any] = {}
    for apt_id, apt in apartments_to_search.items():
        # Numeric filters
        if min_price and apt.price < min_price:
            continue
        if max_price and apt.price > max_price:
            continue
        if number_of_bedrooms and apt.bedrooms != number_of_bedrooms:
            continue
        if number_of_bathrooms and apt.bathrooms != number_of_bathrooms:
            continue
        if square_footage and apt.square_footage < square_footage:
            continue

        # Exact match filters
        if zip_code and apt.zip_code != zip_code:
            continue
        if floor_level and apt.floor_level != floor_level:
            continue

        # String filters (case-insensitive substring)
        if name_lower and name_lower not in apt.name.lower():
            continue
        if location_lower and location_lower not in apt.location.lower():
            continue
        if property_type_lower and apt.property_type.lower() != property_type_lower:
            continue
        if (
            furnished_status_lower
            and apt.furnished_status
            and apt.furnished_status.lower() != furnished_status_lower
        ):
            continue
        if (
            pet_policy_lower
            and apt.pet_policy
            and apt.pet_policy.lower() != pet_policy_lower
        ):
            continue
        if (
            lease_term_lower
            and apt.lease_term
            and apt.lease_term.lower() != lease_term_lower
        ):
            continue

        # Amenities filter
        if amenities_lower:
            apt_amenities_lower = [t.lower() for t in (apt.amenities or [])]
            if not all(am in apt_amenities_lower for am in amenities_lower):
                continue

        filtered[apt_id] = asdict(apt)

    # Build args for event logging — always include all parameters
    event_args: dict[str, Any] = {
        "name": name,
        "location": location,
        "zip_code": zip_code,
        "min_price": min_price,
        "max_price": max_price,
        "number_of_bedrooms": number_of_bedrooms,
        "number_of_bathrooms": number_of_bathrooms,
        "property_type": property_type,
        "square_footage": square_footage,
        "furnished_status": furnished_status,
        "floor_level": floor_level,
        "pet_policy": pet_policy,
        "lease_term": lease_term,
        "amenities": amenities_list,
        "saved_only": saved_only,
    }

    log_action("search_apartments", event_args, ret=filtered)
    json_output({"status": "success", "data": filtered})


# ===========================================================================
# Hidden ENV-tool commands (used by gaia2-eventd, not visible to agents)
# ===========================================================================


@cli.command("add-new-apartment", hidden=True)
@click.option("--name", required=True, help="Apartment name.")
@click.option("--location", required=True, help="Location.")
@click.option("--zip-code", required=True, help="Zip code.")
@click.option("--price", required=True, type=float, help="Monthly price.")
@click.option(
    "--number-of-bedrooms", required=True, type=int, help="Number of bedrooms."
)
@click.option(
    "--number-of-bathrooms", required=True, type=int, help="Number of bathrooms."
)
@click.option("--square-footage", required=True, type=int, help="Square footage.")
@click.option("--property-type", default="", help="Property type.")
@click.option("--furnished-status", default="", help="Furnished status.")
@click.option("--floor-level", default="", help="Floor level.")
@click.option("--pet-policy", default="", help="Pet policy.")
@click.option("--lease-term", default="", help="Lease term.")
@click.option("--amenities", default=None, help="JSON list of amenities.")
def env_add_new_apartment(
    name,
    location,
    zip_code,
    price,
    number_of_bedrooms,
    number_of_bathrooms,
    square_footage,
    property_type,
    furnished_status,
    floor_level,
    pet_policy,
    lease_term,
    amenities,
):
    """[ENV] Add a new apartment listing."""
    import uuid as _uuid

    amenities_list = []
    if amenities:
        try:
            amenities_list = json.loads(amenities)
        except json.JSONDecodeError:
            amenities_list = []

    state = load_app_state(APP_NAME) or {}
    apartments = _load_apartments(state)
    saved_apartments = _load_saved(state, apartments)

    apt_id = _uuid.uuid4().hex
    apartments[apt_id] = Apartment(
        name=name,
        location=location,
        zip_code=zip_code,
        price=price,
        bedrooms=number_of_bedrooms,
        bathrooms=number_of_bathrooms,
        property_type=property_type,
        square_footage=square_footage,
        furnished_status=furnished_status,
        floor_level=floor_level,
        pet_policy=pet_policy,
        lease_term=lease_term,
        amenities=amenities_list,
        apartment_id=apt_id,
    )

    save_app_state(APP_NAME, _build_state(apartments, saved_apartments))

    log_action(
        "add_new_apartment",
        {
            "name": name,
            "location": location,
            "zip_code": zip_code,
            "price": price,
            "number_of_bedrooms": number_of_bedrooms,
            "number_of_bathrooms": number_of_bathrooms,
            "square_footage": square_footage,
            "property_type": property_type,
            "furnished_status": furnished_status,
            "floor_level": floor_level,
            "pet_policy": pet_policy,
            "lease_term": lease_term,
            "amenities": amenities_list,
        },
        ret=apt_id,
        write=True,
    )
    json_output({"status": "success", "apartment_id": apt_id})


@cli.command("update-apartment", hidden=True)
@click.option("--apartment-id", required=True, help="Apartment ID to update.")
@click.option("--new-price", required=True, type=float, help="New price.")
def env_update_apartment(apartment_id: str, new_price: float):
    """[ENV] Update an apartment price."""
    state = load_app_state(APP_NAME) or {}
    apartments = _load_apartments(state)
    saved_apartments = _load_saved(state, apartments)

    if apartment_id not in apartments:
        cli_error("Apartment does not exist")

    apartments[apartment_id].price = new_price
    save_app_state(APP_NAME, _build_state(apartments, saved_apartments))

    log_action(
        "update_apartment",
        {"apartment_id": apartment_id, "new_price": new_price},
        ret=apartment_id,
        write=True,
    )
    json_output({"status": "success", "apartment_id": apartment_id})


@cli.command("delete-apartment", hidden=True)
@click.option("--apartment-id", required=True, help="Apartment ID to delete.")
def env_delete_apartment(apartment_id: str):
    """[ENV] Delete an apartment listing."""
    state = load_app_state(APP_NAME) or {}
    apartments = _load_apartments(state)
    saved_apartments = _load_saved(state, apartments)

    if apartment_id not in apartments:
        cli_error("Apartment does not exist.")

    del apartments[apartment_id]
    if apartment_id in saved_apartments:
        saved_apartments.remove(apartment_id)

    save_app_state(APP_NAME, _build_state(apartments, saved_apartments))

    log_action(
        "delete_apartment",
        {"apartment_id": apartment_id},
        ret=apartment_id,
        write=True,
    )
    json_output({"status": "success", "apartment_id": apartment_id})


def main():
    cli()


if __name__ == "__main__":
    main()
