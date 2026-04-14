# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
"""Unit tests for gaia2_cli/shopping_app.py (Shopping CLI)."""

import pytest
from conftest import assert_event, parse_output, read_events, seed_state
from gaia2_cli.apps.shopping import cli
from gaia2_cli.shared import set_app

APP = "Shopping"


@pytest.fixture(autouse=True)
def _set_app():
    """Ensure the shared module's app name is set for every test."""
    set_app(APP)


# ---------------------------------------------------------------------------
# Seed data
# ---------------------------------------------------------------------------


def _base_state(**overrides):
    """Return a minimal Shopping state with sensible defaults."""
    state = {
        "products": {
            "prod1": {
                "name": "Widget",
                "product_id": "prod1",
                "variants": {
                    "item1": {
                        "item_id": "item1",
                        "price": 10.0,
                        "available": True,
                        "options": {"color": "red"},
                    },
                    "item2": {
                        "item_id": "item2",
                        "price": 15.0,
                        "available": False,
                        "options": {"color": "blue"},
                    },
                },
            },
            "prod2": {
                "name": "Gadget",
                "product_id": "prod2",
                "variants": {
                    "item3": {
                        "item_id": "item3",
                        "price": 25.0,
                        "available": True,
                        "options": {"size": "large"},
                    },
                },
            },
        },
        "cart": {},
        "orders": {},
        "discount_codes": {
            "item1": {"SAVE10": 10.0},
        },
    }
    state.update(overrides)
    return state


# ---------------------------------------------------------------------------
# 1. list-all-products
# ---------------------------------------------------------------------------


def test_list_all_products(cli_env, fixed_time):
    state_dir, runner = cli_env
    seed_state(state_dir, APP, _base_state())

    result = runner.invoke(cli, ["list-all-products"])
    assert result.exit_code == 0, result.output

    out = parse_output(result)
    assert out["status"] == "success"
    # Two products: Gadget and Widget (sorted alphabetically)
    assert len(out["products"]) == 2
    assert "Gadget" in out["products"]
    assert "Widget" in out["products"]
    assert out["metadata"]["total"] == 2

    events = read_events(state_dir)
    assert len(events) == 1
    assert_event(events[0], APP, "list_all_products", write=False)


# ---------------------------------------------------------------------------
# 2. get-product-details happy path
# ---------------------------------------------------------------------------


def test_get_product_details(cli_env, fixed_time):
    state_dir, runner = cli_env
    seed_state(state_dir, APP, _base_state())

    result = runner.invoke(cli, ["get-product-details", "--product-id", "prod1"])
    assert result.exit_code == 0, result.output

    out = parse_output(result)
    assert out["status"] == "success"
    assert out["product"]["name"] == "Widget"
    assert "item1" in out["product"]["variants"]

    events = read_events(state_dir)
    assert_event(events[0], APP, "get_product_details", write=False)


# ---------------------------------------------------------------------------
# 3. get-product-details not found
# ---------------------------------------------------------------------------


def test_get_product_details_not_found(cli_env, fixed_time):
    state_dir, runner = cli_env
    seed_state(state_dir, APP, _base_state())

    result = runner.invoke(cli, ["get-product-details", "--product-id", "nope"])
    assert result.exit_code == 1


# ---------------------------------------------------------------------------
# 4. search-products by name substring (case-insensitive)
# ---------------------------------------------------------------------------


def test_search_product(cli_env, fixed_time):
    state_dir, runner = cli_env
    seed_state(state_dir, APP, _base_state())

    # "wid" matches "Widget" (case-insensitive)
    result = runner.invoke(cli, ["search-products", "--product-name", "wid"])
    assert result.exit_code == 0, result.output

    out = parse_output(result)
    assert out["status"] == "success"
    assert len(out["products"]) == 1
    assert out["products"][0]["name"] == "Widget"

    events = read_events(state_dir)
    assert_event(events[0], APP, "search_product", write=False)


# ---------------------------------------------------------------------------
# 5. add-to-cart happy path
# ---------------------------------------------------------------------------


def test_add_to_cart(cli_env, fixed_time):
    state_dir, runner = cli_env
    seed_state(state_dir, APP, _base_state())

    result = runner.invoke(cli, ["add-to-cart", "--item-id", "item1"])
    assert result.exit_code == 0, result.output

    out = parse_output(result)
    assert out["status"] == "success"
    assert out["item_id"] == "item1"
    assert out["quantity"] == 1

    events = read_events(state_dir)
    assert_event(events[0], APP, "add_to_cart", write=True)


# ---------------------------------------------------------------------------
# 6. add-to-cart increment quantity (add twice)
# ---------------------------------------------------------------------------


def test_add_to_cart_increment(cli_env, fixed_time):
    state_dir, runner = cli_env
    seed_state(state_dir, APP, _base_state())

    runner.invoke(cli, ["add-to-cart", "--item-id", "item1", "--quantity", "2"])
    result = runner.invoke(
        cli, ["add-to-cart", "--item-id", "item1", "--quantity", "3"]
    )
    assert result.exit_code == 0, result.output

    out = parse_output(result)
    assert out["quantity"] == 5  # 2 + 3


# ---------------------------------------------------------------------------
# 7. add-to-cart unavailable item
# ---------------------------------------------------------------------------


def test_add_to_cart_unavailable(cli_env, fixed_time):
    state_dir, runner = cli_env
    seed_state(state_dir, APP, _base_state())

    # item2 has available=False
    result = runner.invoke(cli, ["add-to-cart", "--item-id", "item2"])
    assert result.exit_code == 1


# ---------------------------------------------------------------------------
# 8. remove-from-cart happy path
# ---------------------------------------------------------------------------


def test_remove_from_cart(cli_env, fixed_time):
    state_dir, runner = cli_env
    seed_state(state_dir, APP, _base_state())

    # Add 3, then remove 1
    runner.invoke(cli, ["add-to-cart", "--item-id", "item1", "--quantity", "3"])
    result = runner.invoke(
        cli, ["remove-from-cart", "--item-id", "item1", "--quantity", "1"]
    )
    assert result.exit_code == 0, result.output

    out = parse_output(result)
    assert out["status"] == "success"
    assert out["item_id"] == "item1"

    # Verify cart still has quantity 2
    cart_result = runner.invoke(cli, ["list-cart"])
    cart_out = parse_output(cart_result)
    assert cart_out["cart"]["item1"]["quantity"] == 2

    events = read_events(state_dir)
    remove_events = [e for e in events if e["fn"] == "remove_from_cart"]
    assert len(remove_events) == 1
    assert_event(remove_events[0], APP, "remove_from_cart", write=True)


# ---------------------------------------------------------------------------
# 9. remove-from-cart exceeds quantity
# ---------------------------------------------------------------------------


def test_remove_from_cart_exceeds(cli_env, fixed_time):
    state_dir, runner = cli_env
    seed_state(state_dir, APP, _base_state())

    runner.invoke(cli, ["add-to-cart", "--item-id", "item1", "--quantity", "1"])
    result = runner.invoke(
        cli, ["remove-from-cart", "--item-id", "item1", "--quantity", "5"]
    )
    assert result.exit_code == 1


# ---------------------------------------------------------------------------
# 10. list-cart
# ---------------------------------------------------------------------------


def test_list_cart(cli_env, fixed_time):
    state_dir, runner = cli_env
    seed_state(state_dir, APP, _base_state())

    runner.invoke(cli, ["add-to-cart", "--item-id", "item1", "--quantity", "2"])
    runner.invoke(cli, ["add-to-cart", "--item-id", "item3", "--quantity", "1"])

    result = runner.invoke(cli, ["list-cart"])
    assert result.exit_code == 0, result.output

    out = parse_output(result)
    assert out["status"] == "success"
    assert "item1" in out["cart"]
    assert out["cart"]["item1"]["quantity"] == 2
    assert "item3" in out["cart"]
    assert out["cart"]["item3"]["quantity"] == 1

    events = read_events(state_dir)
    cart_events = [e for e in events if e["fn"] == "list_cart"]
    assert len(cart_events) == 1
    assert_event(cart_events[0], APP, "list_cart", write=False)


# ---------------------------------------------------------------------------
# 11. checkout happy path
# ---------------------------------------------------------------------------


def test_checkout(cli_env, fixed_uuid, fixed_time):
    state_dir, runner = cli_env
    seed_state(state_dir, APP, _base_state())

    runner.invoke(cli, ["add-to-cart", "--item-id", "item1", "--quantity", "2"])

    result = runner.invoke(cli, ["checkout"])
    assert result.exit_code == 0, result.output

    out = parse_output(result)
    assert out["status"] == "success"
    assert out["order_id"] == fixed_uuid(0)
    assert out["total"] == 20.0  # 10.0 * 2

    # Cart should now be empty
    cart_result = runner.invoke(cli, ["list-cart"])
    cart_out = parse_output(cart_result)
    assert cart_out["cart"] == {}

    events = read_events(state_dir)
    checkout_events = [e for e in events if e["fn"] == "checkout"]
    assert len(checkout_events) == 1
    assert_event(checkout_events[0], APP, "checkout", write=True)


# ---------------------------------------------------------------------------
# 12. checkout with discount code
# ---------------------------------------------------------------------------


def test_checkout_with_discount(cli_env, fixed_uuid, fixed_time):
    state_dir, runner = cli_env
    seed_state(state_dir, APP, _base_state())

    # item1 has discount SAVE10 = 10%
    runner.invoke(cli, ["add-to-cart", "--item-id", "item1", "--quantity", "1"])

    result = runner.invoke(cli, ["checkout", "--discount-code", "SAVE10"])
    assert result.exit_code == 0, result.output

    out = parse_output(result)
    assert out["status"] == "success"
    # 10.0 * 1 * (1 - 10/100) = 9.0
    assert out["total"] == 9.0


# ---------------------------------------------------------------------------
# 13. checkout invalid discount code
# ---------------------------------------------------------------------------


def test_checkout_invalid_discount(cli_env, fixed_time):
    state_dir, runner = cli_env
    seed_state(state_dir, APP, _base_state())

    runner.invoke(cli, ["add-to-cart", "--item-id", "item1", "--quantity", "1"])

    result = runner.invoke(cli, ["checkout", "--discount-code", "BOGUS"])
    assert result.exit_code == 1


# ---------------------------------------------------------------------------
# 14. checkout empty cart
# ---------------------------------------------------------------------------


def test_checkout_empty_cart(cli_env, fixed_time):
    state_dir, runner = cli_env
    seed_state(state_dir, APP, _base_state())

    result = runner.invoke(cli, ["checkout"])
    assert result.exit_code == 1


# ---------------------------------------------------------------------------
# 15. get-order-details + list-orders
# ---------------------------------------------------------------------------


def test_get_order_details_and_list_orders(cli_env, fixed_uuid, fixed_time):
    state_dir, runner = cli_env
    seed_state(state_dir, APP, _base_state())

    runner.invoke(cli, ["add-to-cart", "--item-id", "item1", "--quantity", "1"])
    runner.invoke(cli, ["checkout"])

    order_id = fixed_uuid(0)

    # get-order-details
    result = runner.invoke(cli, ["get-order-details", "--order-id", order_id])
    assert result.exit_code == 0, result.output

    out = parse_output(result)
    assert out["status"] == "success"
    assert out["order"]["order_id"] == order_id
    assert out["order"]["order_status"] == "processed"
    assert out["order"]["order_total"] == 10.0

    events = read_events(state_dir)
    detail_events = [e for e in events if e["fn"] == "get_order_details"]
    assert len(detail_events) == 1
    assert_event(detail_events[0], APP, "get_order_details", write=False)

    # list-orders
    result2 = runner.invoke(cli, ["list-orders"])
    assert result2.exit_code == 0, result2.output

    out2 = parse_output(result2)
    assert out2["status"] == "success"
    assert order_id in out2["orders"]

    events2 = read_events(state_dir)
    list_events = [e for e in events2 if e["fn"] == "list_orders"]
    assert len(list_events) == 1
    assert_event(list_events[0], APP, "list_orders", write=False)


# ---------------------------------------------------------------------------
# 16. cancel-order: processed -> cancelled OK; delivered -> exit 1
# ---------------------------------------------------------------------------


def test_cancel_order_processed(cli_env, fixed_uuid, fixed_time):
    state_dir, runner = cli_env
    seed_state(state_dir, APP, _base_state())

    runner.invoke(cli, ["add-to-cart", "--item-id", "item1", "--quantity", "1"])
    runner.invoke(cli, ["checkout"])

    order_id = fixed_uuid(0)

    result = runner.invoke(cli, ["cancel-order", "--order-id", order_id])
    assert result.exit_code == 0, result.output

    out = parse_output(result)
    assert out["status"] == "success"
    assert out["order_id"] == order_id

    # Verify order is cancelled
    detail = runner.invoke(cli, ["get-order-details", "--order-id", order_id])
    detail_out = parse_output(detail)
    assert detail_out["order"]["order_status"] == "cancelled"

    events = read_events(state_dir)
    cancel_events = [e for e in events if e["fn"] == "cancel_order"]
    assert len(cancel_events) == 1
    assert_event(cancel_events[0], APP, "cancel_order", write=True)


def test_cancel_order_delivered_fails(cli_env, fixed_uuid, fixed_time):
    state_dir, runner = cli_env

    # Seed state with a delivered order directly
    state = _base_state(
        orders={
            "order99": {
                "order_id": "order99",
                "order_status": "delivered",
                "order_date": 1522479600.0,
                "order_total": 10.0,
                "order_items": {},
            },
        }
    )
    seed_state(state_dir, APP, state)

    result = runner.invoke(cli, ["cancel-order", "--order-id", "order99"])
    assert result.exit_code == 1


# ---------------------------------------------------------------------------
# ENV commands (hidden, invoked by gaia2-eventd)
# ---------------------------------------------------------------------------


def test_env_add_product(cli_env, fixed_uuid, fixed_time):
    state_dir, runner = cli_env
    seed_state(state_dir, APP, _base_state())

    result = runner.invoke(cli, ["add-product", "--name", "Gizmo"])
    assert result.exit_code == 0, result.output

    out = parse_output(result)
    assert out["status"] == "success"
    product_id = out["product_id"]
    assert product_id == fixed_uuid(0)

    # Verify product is in catalog
    detail = runner.invoke(cli, ["get-product-details", "--product-id", product_id])
    detail_out = parse_output(detail)
    assert detail_out["product"]["name"] == "Gizmo"
    assert detail_out["product"]["variants"] == {}

    events = read_events(state_dir)
    add_events = [e for e in events if e["fn"] == "add_product"]
    assert len(add_events) == 1
    assert_event(add_events[0], APP, "add_product", write=True)


def test_env_add_item_to_product(cli_env, fixed_uuid, fixed_time):
    state_dir, runner = cli_env
    seed_state(state_dir, APP, _base_state())

    result = runner.invoke(
        cli,
        [
            "add-item-to-product",
            "--product-id",
            "prod1",
            "--price",
            "19.99",
            "--options",
            '{"color": "green", "size": "M"}',
            "--available",
            "True",
        ],
    )
    assert result.exit_code == 0, result.output

    out = parse_output(result)
    assert out["status"] == "success"
    item_id = out["item_id"]

    # Verify item is in product variants
    detail = runner.invoke(cli, ["get-product-details", "--product-id", "prod1"])
    detail_out = parse_output(detail)
    assert item_id in detail_out["product"]["variants"]
    item = detail_out["product"]["variants"][item_id]
    assert item["price"] == 19.99
    assert item["available"] is True
    assert item["options"]["color"] == "green"

    events = read_events(state_dir)
    add_events = [e for e in events if e["fn"] == "add_item_to_product"]
    assert len(add_events) == 1
    assert_event(add_events[0], APP, "add_item_to_product", write=True)


def test_env_add_item_to_product_not_found(cli_env, fixed_time):
    state_dir, runner = cli_env
    seed_state(state_dir, APP, _base_state())

    result = runner.invoke(
        cli,
        ["add-item-to-product", "--product-id", "nope", "--price", "5.0"],
    )
    assert result.exit_code == 1


def test_env_update_item_price(cli_env, fixed_time):
    state_dir, runner = cli_env
    seed_state(state_dir, APP, _base_state())

    result = runner.invoke(
        cli,
        ["update-item", "--item-id", "item1", "--new-price", "7.50"],
    )
    assert result.exit_code == 0, result.output

    out = parse_output(result)
    assert out["status"] == "success"
    assert out["item_id"] == "item1"

    # Verify price changed
    detail = runner.invoke(cli, ["get-product-details", "--product-id", "prod1"])
    detail_out = parse_output(detail)
    assert detail_out["product"]["variants"]["item1"]["price"] == 7.50

    events = read_events(state_dir)
    upd_events = [e for e in events if e["fn"] == "update_item"]
    assert len(upd_events) == 1
    assert_event(upd_events[0], APP, "update_item", write=True)


def test_env_update_item_availability(cli_env, fixed_time):
    state_dir, runner = cli_env
    seed_state(state_dir, APP, _base_state())

    # item2 is unavailable, make it available
    result = runner.invoke(
        cli,
        ["update-item", "--item-id", "item2", "--new-availability", "True"],
    )
    assert result.exit_code == 0, result.output

    detail = runner.invoke(cli, ["get-product-details", "--product-id", "prod1"])
    detail_out = parse_output(detail)
    assert detail_out["product"]["variants"]["item2"]["available"] is True


def test_env_update_item_not_found(cli_env, fixed_time):
    state_dir, runner = cli_env
    seed_state(state_dir, APP, _base_state())

    result = runner.invoke(
        cli,
        ["update-item", "--item-id", "nope", "--new-price", "1.0"],
    )
    assert result.exit_code == 1


def test_env_update_item_no_update(cli_env, fixed_time):
    state_dir, runner = cli_env
    seed_state(state_dir, APP, _base_state())

    result = runner.invoke(cli, ["update-item", "--item-id", "item1"])
    assert result.exit_code == 1


def test_env_add_discount_code(cli_env, fixed_time):
    state_dir, runner = cli_env
    seed_state(state_dir, APP, _base_state())

    result = runner.invoke(
        cli,
        [
            "add-discount-code",
            "--item-id",
            "item1",
            "--discount-code",
            '{"SAVE20": 20.0}',
        ],
    )
    assert result.exit_code == 0, result.output

    out = parse_output(result)
    assert out["status"] == "success"

    # Verify discount code was added (original SAVE10 should still be there)
    codes = runner.invoke(cli, ["get-all-discount-codes"])
    codes_out = parse_output(codes)
    assert "SAVE10" in codes_out["discount_codes"]["item1"]
    assert "SAVE20" in codes_out["discount_codes"]["item1"]
    assert codes_out["discount_codes"]["item1"]["SAVE20"] == 20.0

    events = read_events(state_dir)
    dc_events = [e for e in events if e["fn"] == "add_discount_code"]
    assert len(dc_events) == 1
    assert_event(dc_events[0], APP, "add_discount_code", write=True)


def test_env_add_discount_code_item_not_found(cli_env, fixed_time):
    state_dir, runner = cli_env
    seed_state(state_dir, APP, _base_state())

    result = runner.invoke(
        cli,
        ["add-discount-code", "--item-id", "nope", "--discount-code", '{"X": 5.0}'],
    )
    assert result.exit_code == 1


def test_env_update_order_status(cli_env, fixed_uuid, fixed_time):
    state_dir, runner = cli_env
    seed_state(state_dir, APP, _base_state())

    # Create an order first
    runner.invoke(cli, ["add-to-cart", "--item-id", "item1", "--quantity", "1"])
    runner.invoke(cli, ["checkout"])
    order_id = fixed_uuid(0)

    result = runner.invoke(
        cli,
        ["update-order-status", "--order-id", order_id, "--status", "shipped"],
    )
    assert result.exit_code == 0, result.output

    out = parse_output(result)
    assert out["status"] == "success"
    assert out["order_id"] == order_id

    # Verify status changed
    detail = runner.invoke(cli, ["get-order-details", "--order-id", order_id])
    detail_out = parse_output(detail)
    assert detail_out["order"]["order_status"] == "shipped"

    events = read_events(state_dir)
    upd_events = [e for e in events if e["fn"] == "update_order_status"]
    assert len(upd_events) == 1
    assert_event(upd_events[0], APP, "update_order_status", write=True)


def test_env_update_order_status_invalid(cli_env, fixed_uuid, fixed_time):
    state_dir, runner = cli_env
    seed_state(state_dir, APP, _base_state())

    runner.invoke(cli, ["add-to-cart", "--item-id", "item1", "--quantity", "1"])
    runner.invoke(cli, ["checkout"])
    order_id = fixed_uuid(0)

    result = runner.invoke(
        cli,
        ["update-order-status", "--order-id", order_id, "--status", "bogus"],
    )
    assert result.exit_code == 1


def test_env_update_order_status_not_found(cli_env, fixed_time):
    state_dir, runner = cli_env
    seed_state(state_dir, APP, _base_state())

    result = runner.invoke(
        cli,
        ["update-order-status", "--order-id", "nope", "--status", "shipped"],
    )
    assert result.exit_code == 1
