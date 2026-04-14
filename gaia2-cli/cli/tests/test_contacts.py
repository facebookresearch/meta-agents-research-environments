# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
"""Unit tests for gaia2_cli/contacts_app.py."""

import json

import pytest
from conftest import assert_event, parse_output, read_events, seed_state
from gaia2_cli.apps.contacts import cli

APP = "Contacts"


@pytest.fixture(autouse=True)
def _set_app():
    """Ensure the module-level app name is set for every test."""
    from gaia2_cli.shared import set_app

    set_app(APP)


# ---- helpers ---------------------------------------------------------------


def _make_contact(
    contact_id="id1",
    first_name="John",
    last_name="Doe",
    is_user=False,
    gender="Male",
    age=30,
    **overrides,
):
    base = {
        "contact_id": contact_id,
        "first_name": first_name,
        "last_name": last_name,
        "is_user": is_user,
        "gender": gender,
        "age": age,
        "nationality": None,
        "city_living": None,
        "country": None,
        "status": "Unknown",
        "job": None,
        "description": None,
        "phone": None,
        "email": None,
        "address": None,
    }
    base.update(overrides)
    return base


def _seed_contacts(state_dir, contacts_dict, view_limit=10):
    seed_state(state_dir, APP, {"contacts": contacts_dict, "view_limit": view_limit})


# ---- 1. add-new-contact happy path ----------------------------------------


def test_add_new_contact_happy_path(cli_env, fixed_uuid, fixed_time):
    state_dir, runner = cli_env
    result = runner.invoke(
        cli, ["add-new-contact", "--first-name", "Katie", "--last-name", "Smith"]
    )

    assert result.exit_code == 0, result.stderr
    out = parse_output(result)
    assert out["status"] == "success"
    assert out["contact_id"] == fixed_uuid(0)


# ---- 2. add-new-contact all optional fields --------------------------------


def test_add_new_contact_all_fields(cli_env, fixed_uuid, fixed_time):
    state_dir, runner = cli_env
    result = runner.invoke(
        cli,
        [
            "add-new-contact",
            "--first-name",
            "Alice",
            "--last-name",
            "Wonderland",
            "--gender",
            "Female",
            "--age",
            "25",
            "--nationality",
            "British",
            "--city-living",
            "London",
            "--country",
            "UK",
            "--status",
            "Employed",
            "--job",
            "Engineer",
            "--description",
            "Curious person",
            "--phone",
            "+44123456",
            "--email",
            "alice@example.com",
            "--address",
            "1 Rabbit Hole Lane",
        ],
    )

    assert result.exit_code == 0, result.stderr
    out = parse_output(result)
    assert out["status"] == "success"
    assert out["contact_id"] == fixed_uuid(0)

    # Verify the contact was persisted with all fields
    state_path = state_dir / "contacts.json"
    state = json.loads(state_path.read_text())
    cid = fixed_uuid(0)
    contact = state["contacts"][cid]
    assert contact["first_name"] == "Alice"
    assert contact["gender"] == "Female"
    assert contact["age"] == 25
    assert contact["nationality"] == "British"
    assert contact["city_living"] == "London"
    assert contact["country"] == "UK"
    assert contact["status"] == "Employed"
    assert contact["job"] == "Engineer"
    assert contact["phone"] == "+44123456"
    assert contact["email"] == "alice@example.com"
    assert contact["address"] == "1 Rabbit Hole Lane"


# ---- 3. add-new-contact duplicate name ------------------------------------


def test_add_new_contact_duplicate_name(cli_env, fixed_uuid, fixed_time):
    state_dir, runner = cli_env
    _seed_contacts(
        state_dir,
        {"id1": _make_contact(contact_id="id1", first_name="John", last_name="Doe")},
    )

    result = runner.invoke(
        cli, ["add-new-contact", "--first-name", "john", "--last-name", "doe"]
    )

    assert result.exit_code == 1
    assert "already exists" in result.stderr


# ---- 4. add-new-contact invalid gender ------------------------------------


def test_add_new_contact_invalid_gender(cli_env, fixed_uuid, fixed_time):
    state_dir, runner = cli_env
    result = runner.invoke(
        cli,
        [
            "add-new-contact",
            "--first-name",
            "X",
            "--last-name",
            "Y",
            "--gender",
            "Alien",
        ],
    )

    assert result.exit_code == 1
    assert "not a valid Gender" in result.stderr


# ---- 5. add-new-contact invalid status ------------------------------------


def test_add_new_contact_invalid_status(cli_env, fixed_uuid, fixed_time):
    state_dir, runner = cli_env
    result = runner.invoke(
        cli,
        [
            "add-new-contact",
            "--first-name",
            "X",
            "--last-name",
            "Y",
            "--status",
            "Dancing",
        ],
    )

    assert result.exit_code == 1
    assert "not a valid Status" in result.stderr


# ---- 6. add-new-contact invalid age (0 and 101) ---------------------------


@pytest.mark.parametrize("bad_age", [0, 101])
def test_add_new_contact_invalid_age(cli_env, fixed_uuid, fixed_time, bad_age):
    state_dir, runner = cli_env
    result = runner.invoke(
        cli,
        [
            "add-new-contact",
            "--first-name",
            "X",
            "--last-name",
            "Y",
            "--age",
            str(bad_age),
        ],
    )

    assert result.exit_code == 1
    assert "Invalid age" in result.stderr


# ---- 7. Event log for add-new-contact -------------------------------------


def test_add_new_contact_event_log(cli_env, fixed_uuid, fixed_time):
    state_dir, runner = cli_env
    result = runner.invoke(
        cli, ["add-new-contact", "--first-name", "Eve", "--last-name", "Adams"]
    )

    assert result.exit_code == 0, result.stderr
    events = read_events(state_dir)
    assert len(events) == 1
    ev = events[0]
    assert_event(ev, APP, "add_new_contact", write=True)

    # All parameter keys must be present, even when None
    args = ev["args"]
    for key in [
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
    ]:
        assert key in args, f"Missing key '{key}' in event args"

    assert args["first_name"] == "Eve"
    assert args["last_name"] == "Adams"
    assert args["gender"] is None
    assert args["age"] is None
    assert ev["ret"] == fixed_uuid(0)


# ---- 8. get-contacts with pagination --------------------------------------


def test_get_contacts_pagination(cli_env, fixed_time):
    state_dir, runner = cli_env
    contacts = {}
    for i in range(3):
        cid = f"cid{i}"
        contacts[cid] = _make_contact(
            contact_id=cid, first_name=f"User{i}", last_name="Test"
        )
    _seed_contacts(state_dir, contacts, view_limit=10)

    result = runner.invoke(cli, ["get-contacts"])

    assert result.exit_code == 0, result.stderr
    out = parse_output(result)
    assert len(out["contacts"]) == 3
    assert out["metadata"]["total"] == 3
    assert out["metadata"]["range"] == [0, 3]


# ---- 9. get-contact happy path --------------------------------------------


def test_get_contact_happy_path(cli_env, fixed_time):
    state_dir, runner = cli_env
    _seed_contacts(state_dir, {"id1": _make_contact()})

    result = runner.invoke(cli, ["get-contact", "--contact-id", "id1"])

    assert result.exit_code == 0, result.stderr
    out = parse_output(result)
    assert out["contact_id"] == "id1"
    assert out["first_name"] == "John"
    assert out["last_name"] == "Doe"


# ---- 10. get-contact not found --------------------------------------------


def test_get_contact_not_found(cli_env, fixed_time):
    state_dir, runner = cli_env
    _seed_contacts(state_dir, {})

    result = runner.invoke(cli, ["get-contact", "--contact-id", "nonexistent"])

    assert result.exit_code == 1
    assert "does not exist" in result.stderr


# ---- 11. get-current-user-details happy path --------------------------------


def test_get_current_user_details(cli_env, fixed_time):
    state_dir, runner = cli_env
    user_contact = _make_contact(
        contact_id="user1", first_name="Me", last_name="Myself", is_user=True
    )
    _seed_contacts(state_dir, {"user1": user_contact})

    result = runner.invoke(cli, ["get-current-user-details"])

    assert result.exit_code == 0, result.stderr
    out = parse_output(result)
    assert out["contact_id"] == "user1"
    assert out["first_name"] == "Me"
    assert out["is_user"] is True


# ---- 12. get-current-user-details not found --------------------------------


def test_get_current_user_details_not_found(cli_env, fixed_time):
    state_dir, runner = cli_env
    non_user = _make_contact(contact_id="id1", is_user=False)
    _seed_contacts(state_dir, {"id1": non_user})

    result = runner.invoke(cli, ["get-current-user-details"])

    assert result.exit_code == 1
    assert "User Contact does not exist" in result.stderr


# ---- 13. edit-contact happy path -------------------------------------------


def test_edit_contact_happy_path(cli_env, fixed_time):
    state_dir, runner = cli_env
    _seed_contacts(state_dir, {"id1": _make_contact()})

    updates = json.dumps({"first_name": "Jane"})
    result = runner.invoke(
        cli, ["edit-contact", "--contact-id", "id1", "--updates", updates]
    )

    assert result.exit_code == 0, result.stderr
    out = parse_output(result)
    assert out["status"] == "success"

    # Verify the edit persisted
    state = json.loads((state_dir / "contacts.json").read_text())
    assert state["contacts"]["id1"]["first_name"] == "Jane"
    # Other fields unchanged
    assert state["contacts"]["id1"]["last_name"] == "Doe"

    events = read_events(state_dir)
    assert len(events) == 1
    assert_event(events[0], APP, "edit_contact", write=True)


# ---- 14. edit-contact invalid field ----------------------------------------


def test_edit_contact_invalid_field(cli_env, fixed_time):
    state_dir, runner = cli_env
    _seed_contacts(state_dir, {"id1": _make_contact()})

    updates = json.dumps({"favorite_color": "blue"})
    result = runner.invoke(
        cli, ["edit-contact", "--contact-id", "id1", "--updates", updates]
    )

    assert result.exit_code == 1
    assert "not a valid attribute" in result.stderr


# ---- 15. delete-contact + search-contacts by name --------------------------


def test_delete_contact_and_search(cli_env, fixed_time):
    state_dir, runner = cli_env
    c1 = _make_contact(contact_id="id1", first_name="Alice", last_name="Smith")
    c2 = _make_contact(contact_id="id2", first_name="Bob", last_name="Jones")
    _seed_contacts(state_dir, {"id1": c1, "id2": c2})

    # Search finds Alice
    result = runner.invoke(cli, ["search-contacts", "--query", "Alice"])
    assert result.exit_code == 0, result.stderr
    out = parse_output(result)
    assert len(out) == 1
    assert out[0]["first_name"] == "Alice"

    # Delete Alice
    result = runner.invoke(cli, ["delete-contact", "--contact-id", "id1"])
    assert result.exit_code == 0, result.stderr
    assert parse_output(result)["status"] == "success"

    # Search no longer finds Alice
    result = runner.invoke(cli, ["search-contacts", "--query", "Alice"])
    assert result.exit_code == 0, result.stderr
    out = parse_output(result)
    assert len(out) == 0

    # Bob is still there
    result = runner.invoke(cli, ["search-contacts", "--query", "Bob"])
    assert result.exit_code == 0, result.stderr
    out = parse_output(result)
    assert len(out) == 1
    assert out[0]["first_name"] == "Bob"
