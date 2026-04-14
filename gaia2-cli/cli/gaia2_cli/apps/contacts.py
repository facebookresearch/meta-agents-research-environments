# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
"""
Standalone CLI for the Gaia2 Contacts app.

Binary name: ``contacts``
App class name (for events): ``Contacts``
"""

import json
import uuid

import click

from gaia2_cli.shared import (
    build_schema,
    cli_error,
    json_output,
    load_app_state,
    log_action,
    save_app_state,
    set_app,
    validate_email_address,
)

APP_NAME = "Contacts"

set_app(APP_NAME)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_state() -> dict:
    """Load contacts state, returning a default if missing."""
    state = load_app_state(APP_NAME)
    if not state:
        return {"contacts": {}, "view_limit": 10}
    return state


def _contact_dict(contact: dict) -> dict:
    """Return a contact dict with all canonical fields present."""
    return {
        "contact_id": contact.get("contact_id", ""),
        "first_name": contact.get("first_name", ""),
        "last_name": contact.get("last_name", ""),
        "is_user": contact.get("is_user", False),
        "gender": contact.get("gender", "Unknown"),
        "age": contact.get("age"),
        "nationality": contact.get("nationality"),
        "city_living": contact.get("city_living"),
        "country": contact.get("country"),
        "status": contact.get("status", "Unknown"),
        "job": contact.get("job"),
        "description": contact.get("description"),
        "phone": contact.get("phone"),
        "email": contact.get("email"),
        "address": contact.get("address"),
    }


# ---------------------------------------------------------------------------
# CLI group
# ---------------------------------------------------------------------------


@click.group(context_settings={"terminal_width": 10000, "max_content_width": 10000})
def cli():
    """Contacts - Gaia2 Contacts app CLI."""
    pass


# ---------------------------------------------------------------------------
# get-contacts (READ)
# ---------------------------------------------------------------------------


@cli.command("get-contacts")
@click.option(
    "--offset",
    type=int,
    default=0,
    help="Starting point to retrieve contacts from, default is 0.",
)
def get_contacts(offset):
    """Gets the list of contacts starting from a specified offset. There is a view limit. Use offset to scroll through the contacts."""
    state = _load_state()
    contacts = state.get("contacts", {})
    view_limit = state.get("view_limit", 10)

    contact_list = list(contacts.values())
    sliced = contact_list[offset : offset + view_limit]

    result = {
        "contacts": [_contact_dict(c) for c in sliced],
        "metadata": {
            "range": [offset, min(offset + view_limit, len(contact_list))],
            "total": len(contact_list),
        },
    }

    log_action("get_contacts", {"offset": offset}, ret=result)
    json_output(result)


# ---------------------------------------------------------------------------
# get-contact (READ)
# ---------------------------------------------------------------------------


@cli.command("get-contact")
@click.option(
    "--contact-id", required=True, type=str, help="ID of the contact to retrieve."
)
def get_contact(contact_id):
    """Gets a specific contact by contact_id."""
    state = _load_state()
    contacts = state.get("contacts", {})

    if contact_id not in contacts:
        cli_error(f"Contact {contact_id} does not exist.")

    result = _contact_dict(contacts[contact_id])
    log_action("get_contact", {"contact_id": contact_id}, ret=result)
    json_output(result)


# ---------------------------------------------------------------------------
# get-current-user-details (READ)
# ---------------------------------------------------------------------------


@cli.command("get-current-user-details")
def get_current_user_details():
    """Gets the current user's details including name, phone number, email address, nationality, country living."""
    state = _load_state()
    contacts = state.get("contacts", {})

    for contact in contacts.values():
        if contact.get("is_user", False):
            result = _contact_dict(contact)
            log_action("get_current_user_details", {}, ret=result)
            json_output(result)
            return

    cli_error("User Contact does not exist.")


# ---------------------------------------------------------------------------
# add-new-contact (WRITE)
# ---------------------------------------------------------------------------


@cli.command("add-new-contact")
@click.option(
    "--first-name", required=True, type=str, help="First name of the contact, required."
)
@click.option(
    "--last-name", required=True, type=str, help="Last name of the contact, required."
)
@click.option(
    "--gender",
    default=None,
    type=str,
    help="Gender of the contact (Male, Female, Other, Unknown), optional default is Unknown.",
)
@click.option("--age", default=None, type=int, help="Age of the contact, optional.")
@click.option(
    "--nationality",
    default=None,
    type=str,
    help="Nationality of the contact, optional.",
)
@click.option(
    "--city-living",
    default=None,
    type=str,
    help="City where the contact is living, optional.",
)
@click.option(
    "--country",
    default=None,
    type=str,
    help="Country where the contact is living, optional.",
)
@click.option(
    "--status",
    default=None,
    type=str,
    help="Status of the contact (Student, Employed, Unemployed, Retired, Unknown), optional default is Unknown.",
)
@click.option("--job", default=None, type=str, help="Job of the contact, optional.")
@click.option(
    "--description",
    default=None,
    type=str,
    help="Description of the contact, optional.",
)
@click.option(
    "--phone", default=None, type=str, help="Phone number of the contact, optional."
)
@click.option(
    "--email", default=None, type=str, help="Email address of the contact, optional."
)
@click.option(
    "--address", default=None, type=str, help="Address of the contact, optional."
)
def add_new_contact(
    first_name,
    last_name,
    gender,
    age,
    nationality,
    city_living,
    country,
    status,
    job,
    description,
    phone,
    email,
    address,
):
    """Adds a new contact to the contacts app."""
    state = _load_state()
    contacts = state.get("contacts", {})

    # Check for duplicate by full name (case-insensitive)
    names_to_id = {
        c.get("first_name", "").lower() + " " + c.get("last_name", "").lower(): cid
        for cid, c in contacts.items()
    }
    key = first_name.lower() + " " + last_name.lower()
    if key in names_to_id:
        err = f"Contact already exists with contact id - {names_to_id[key]}."
        cli_error(err)

    # Validate gender
    valid_genders = {"Male", "Female", "Other", "Unknown"}
    resolved_gender = "Unknown"
    if gender is not None:
        if gender not in valid_genders:
            err = f"'{gender}' is not a valid Gender"
            cli_error(err)
        resolved_gender = gender

    # Validate status
    valid_statuses = {"Student", "Employed", "Unemployed", "Retired", "Unknown"}
    resolved_status = "Unknown"
    if status is not None:
        if status not in valid_statuses:
            err = f"'{status}' is not a valid Status"
            cli_error(err)
        resolved_status = status

    # Validate age
    if age is not None and not (1 <= age <= 100):
        err = f"Invalid age. {age}"
        cli_error(err)

    # Validate email
    if email is not None:
        validate_email_address(email, "email")

    contact_id = uuid.uuid4().hex
    new_contact = {
        "contact_id": contact_id,
        "first_name": first_name,
        "last_name": last_name,
        "is_user": False,
        "gender": resolved_gender,
        "age": age,
        "nationality": nationality,
        "city_living": city_living,
        "country": country,
        "status": resolved_status,
        "job": job,
        "description": description,
        "phone": phone,
        "email": email,
        "address": address,
    }

    contacts[contact_id] = new_contact
    state["contacts"] = contacts
    save_app_state(APP_NAME, state)

    args = _build_args_dict(
        first_name=first_name,
        last_name=last_name,
        gender=gender,
        age=age,
        nationality=nationality,
        city_living=city_living,
        country=country,
        status=status,
        job=job,
        description=description,
        phone=phone,
        email=email,
        address=address,
    )
    log_action("add_new_contact", args, ret=contact_id, write=True)
    json_output({"status": "success", "contact_id": contact_id})


def _build_args_dict(**kwargs) -> dict:
    """Build an args dict — always include all keys.

    The grader's HardToolJudge accesses args by key directly (not .get()),
    so all function parameters must be present even when None.
    """
    return dict(kwargs)


# ---------------------------------------------------------------------------
# edit-contact (WRITE)
# ---------------------------------------------------------------------------


@cli.command("edit-contact")
@click.option(
    "--contact-id", required=True, type=str, help="ID of the contact to edit."
)
@click.option(
    "--updates",
    required=True,
    type=str,
    help="JSON dictionary of fields to update. Valid fields: first_name, last_name, gender (Male/Female/Other/Unknown), age (1-100), nationality, city_living, country, status (Student/Employed/Unemployed/Retired/Unknown), job, description, phone, email, address.",
)
def edit_contact(contact_id, updates):  # noqa: C901
    """Edits specific fields of a contact by contact_id."""
    # Parse the updates JSON
    try:
        updates_dict = json.loads(updates)
    except json.JSONDecodeError as e:
        cli_error(f"Invalid JSON for updates: {e}")

    state = _load_state()
    contacts = state.get("contacts", {})

    if contact_id not in contacts:
        cli_error("Contact does not exist.")

    contact = contacts[contact_id]
    valid_fields = {
        "first_name",
        "last_name",
        "gender",
        "age",
        "nationality",
        "city_living",
        "country",
        "status",
        "job",
        "description",
        "phone",
        "email",
        "address",
    }

    for key, _value in updates_dict.items():
        if key not in valid_fields:
            cli_error(f"{key} is not a valid attribute of Contact.")

    # Validate gender if being updated
    if "gender" in updates_dict:
        valid_genders = {"Male", "Female", "Other", "Unknown"}
        if updates_dict["gender"] not in valid_genders:
            err = f"'{updates_dict['gender']}' is not a valid Gender"
            cli_error(err)

    # Validate status if being updated
    if "status" in updates_dict:
        valid_statuses = {"Student", "Employed", "Unemployed", "Retired", "Unknown"}
        if updates_dict["status"] not in valid_statuses:
            err = f"'{updates_dict['status']}' is not a valid Status"
            cli_error(err)

    # Validate age if being updated
    if "age" in updates_dict and updates_dict["age"] is not None:
        if not (1 <= updates_dict["age"] <= 100):
            err = f"Invalid age. {updates_dict['age']}"
            cli_error(err)

    # Validate first_name / last_name not None
    if "first_name" in updates_dict and updates_dict["first_name"] is None:
        err = "Invalid first name."
        cli_error(err)

    if "last_name" in updates_dict and updates_dict["last_name"] is None:
        err = "Invalid last name."
        cli_error(err)

    # Validate email if being updated
    if "email" in updates_dict and updates_dict["email"] is not None:
        validate_email_address(updates_dict["email"], "email")

    # Apply updates
    for key, value in updates_dict.items():
        contact[key] = value

    contacts[contact_id] = contact
    state["contacts"] = contacts
    save_app_state(APP_NAME, state)

    result = f"Contact {contact_id} updated successfully."
    log_action(
        "edit_contact",
        {"contact_id": contact_id, "updates": updates_dict},
        ret=result,
        write=True,
    )
    json_output({"status": "success", "message": result})


# ---------------------------------------------------------------------------
# delete-contact (WRITE)
# ---------------------------------------------------------------------------


@cli.command("delete-contact")
@click.option(
    "--contact-id", required=True, type=str, help="ID of the contact to delete."
)
def delete_contact(contact_id):
    """Deletes a specific contact by contact_id."""
    state = _load_state()
    contacts = state.get("contacts", {})

    if contact_id not in contacts:
        cli_error("Contact does not exist.")

    del contacts[contact_id]
    state["contacts"] = contacts
    save_app_state(APP_NAME, state)

    result = f"Contact {contact_id} successfully deleted."
    log_action("delete_contact", {"contact_id": contact_id}, ret=result, write=True)
    json_output({"status": "success", "message": result})


# ---------------------------------------------------------------------------
# search-contacts (READ)
# ---------------------------------------------------------------------------


@cli.command("search-contacts")
@click.option("--query", required=True, type=str, help="The search query string.")
def search_contacts(query):
    """Searches for contacts based on a query string. The search looks for partial matches in first name, last name, phone number, and email."""
    state = _load_state()
    contacts = state.get("contacts", {})
    query_lower = query.lower()

    results = []
    for contact in contacts.values():
        first = (contact.get("first_name") or "").lower()
        last = (contact.get("last_name") or "").lower()
        phone = (contact.get("phone") or "").lower()
        email = (contact.get("email") or "").lower()
        if (
            query_lower in first
            or query_lower in last
            or query_lower in phone
            or query_lower in email
            or query_lower in first + " " + last
            or query_lower in last + " " + first
        ):
            results.append(_contact_dict(contact))

    log_action("search_contacts", {"query": query}, ret=results)
    json_output(results)


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
