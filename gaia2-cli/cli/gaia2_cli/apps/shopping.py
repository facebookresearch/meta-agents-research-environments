# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
"""
Standalone CLI for the Gaia2 Shopping app.

Binary name: ``shopping``
App class name (for events): ``Shopping``

State structure:
    products: dict[product_id -> {name, product_id, variants: {item_id -> {price, available, item_id, options}}}]
    cart: dict[item_id -> {item_id, quantity, price, available, options}]
    orders: dict[order_id -> {order_id, order_status, order_date, order_total, order_items: {...}}]
    discount_codes: dict[item_id -> {code -> discount_pct}]
"""

import time
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
)

APP_NAME = "Shopping"

set_app(APP_NAME)


def _load_state() -> dict:
    """Load Shopping state, returning a default skeleton if absent."""
    state = load_app_state(APP_NAME)
    if not state:
        state = {}
    state.setdefault("products", {})
    state.setdefault("cart", {})
    state.setdefault("orders", {})
    state.setdefault("discount_codes", {})
    return state


def _get_item(state: dict, item_id: str) -> dict:
    """Find an item (variant) by item_id across all products.

    Returns a merged dict with product name/id and item fields, or empty dict.
    """
    for product_id, product in state["products"].items():
        variants = product.get("variants", {})
        if item_id in variants:
            item = variants[item_id]
            return {
                "name": product.get("name", ""),
                "product_id": product_id,
                **(item if isinstance(item, dict) else {}),
            }
    return {}


# ---------------------------------------------------------------------------
# Click CLI
# ---------------------------------------------------------------------------


@click.group(context_settings={"terminal_width": 10000, "max_content_width": 10000})
def cli():
    """Shopping - Gaia2 app CLI"""
    pass


# ---- READ commands --------------------------------------------------------


@cli.command("list-all-products")
@click.option("--offset", type=int, default=0, help="Offset to start listing from.")
@click.option("--limit", type=int, default=10, help="Number of products to list.")
def list_all_products(offset: int, limit: int):
    """List all products in the catalog."""
    state = _load_state()
    products = state["products"]

    # Build name->id mapping, sort by name, paginate
    product_dict = {info["name"]: pid for pid, info in products.items()}
    sorted_items = sorted(product_dict.items())
    page = dict(sorted_items[offset : offset + limit])

    result = {
        "products": page,
        "metadata": {
            "range": [offset, min(offset + limit, len(products))],
            "total": len(products),
        },
    }

    log_action("list_all_products", {"offset": offset, "limit": limit}, ret=result)
    json_output({"status": "success", **result})


@cli.command("get-product-details")
@click.option("--product-id", required=True, help="Product ID to get details for.")
def get_product_details(product_id: str):
    """Get product details for a given product ID."""
    state = _load_state()
    products = state["products"]

    if product_id not in products:
        cli_error("Product does not exist")

    product = products[product_id]
    log_action("get_product_details", {"product_id": product_id}, ret=product)
    json_output({"status": "success", "product": product})


@cli.command("search-products")
@click.option(
    "--product-name",
    required=True,
    help="Product name to search for (case-insensitive substring match).",
)
@click.option("--offset", type=int, default=0, help="Offset to start listing from.")
@click.option("--limit", type=int, default=10, help="Number of products to list.")
def search_product(product_name: str, offset: int, limit: int):
    """Search for products by name (case-insensitive substring match)."""
    state = _load_state()
    products = state["products"]

    results = []
    for _pid, info in products.items():
        name = info.get("name", "")
        if product_name.lower() in name.lower():
            results.append(info)

    paginated = results[offset : offset + limit]

    log_action(
        "search_product",
        {"product_name": product_name, "offset": offset, "limit": limit},
        ret=paginated,
    )
    json_output({"status": "success", "products": paginated})


# ---- CART commands --------------------------------------------------------


@cli.command("add-to-cart")
@click.option("--item-id", required=True, help="Item (variant) ID to add to cart.")
@click.option("--quantity", type=int, default=1, help="Quantity to add (default 1).")
def add_to_cart(item_id: str, quantity: int):
    """Add an item to the cart."""
    state = _load_state()

    if quantity <= 0:
        cli_error("Quantity cannot be negative or zero")

    item = _get_item(state, item_id)
    if not item:
        cli_error("Product does not exist")

    if not item.get("available", True):
        cli_error("Product is not available")

    cart = state["cart"]
    if item_id in cart:
        cart[item_id]["quantity"] += quantity
    else:
        cart[item_id] = {
            "item_id": item.get("item_id", item_id),
            "quantity": quantity,
            "price": item.get("price", 0.0),
            "available": item.get("available", True),
            "options": item.get("options", {}),
        }

    save_app_state(APP_NAME, state)
    log_action(
        "add_to_cart",
        {"item_id": item_id, "quantity": quantity},
        ret=item_id,
        write=True,
    )
    json_output(
        {"status": "success", "item_id": item_id, "quantity": cart[item_id]["quantity"]}
    )


@cli.command("remove-from-cart")
@click.option("--item-id", required=True, help="Item (variant) ID to remove from cart.")
@click.option("--quantity", type=int, default=1, help="Quantity to remove (default 1).")
def remove_from_cart(item_id: str, quantity: int):
    """Remove an item from the cart."""
    state = _load_state()
    cart = state["cart"]

    if quantity < 0:
        cli_error("Quantity cannot be negative")

    if item_id not in cart:
        cli_error("Product not in cart")

    if quantity > cart[item_id]["quantity"]:
        cli_error("Quantity exceeds available quantity")

    cart[item_id]["quantity"] -= quantity
    if cart[item_id]["quantity"] == 0:
        del cart[item_id]

    save_app_state(APP_NAME, state)
    log_action(
        "remove_from_cart",
        {"item_id": item_id, "quantity": quantity},
        ret=item_id,
        write=True,
    )
    json_output({"status": "success", "item_id": item_id})


@cli.command("list-cart")
def list_cart():
    """List the contents of the cart."""
    state = _load_state()
    cart = state["cart"]

    log_action("list_cart", {}, ret=cart)
    json_output({"status": "success", "cart": cart})


# ---- CHECKOUT / ORDERS ---------------------------------------------------


@cli.command("checkout")
@click.option("--discount-code", default=None, help="Optional discount code to apply.")
def checkout(discount_code: str | None):
    """Checkout the current cart and create an order."""
    state = _load_state()
    cart = state["cart"]
    discount_codes = state["discount_codes"]

    if not cart:
        cli_error("Cart is empty")

    # Validate discount code against all items in cart (all-or-nothing)
    if discount_code is not None and len(discount_code) > 0:
        for iid in cart:
            item_discounts = discount_codes.get(iid, {})
            if discount_code not in item_discounts:
                cli_error(
                    f"The provided discount code is not valid for the item with id - {iid}"
                )

    # Calculate total
    order_total = 0.0
    for iid, cart_item in cart.items():
        price = cart_item.get("price", 0.0)
        qty = cart_item.get("quantity", 1)
        item_total = price * qty

        if discount_code is not None and len(discount_code) > 0:
            item_discounts = discount_codes.get(iid, {})
            discount_pct = item_discounts.get(discount_code, 0.0)
            item_total = item_total * (1 - discount_pct / 100)

        order_total += item_total

    order_id = uuid.uuid4().hex
    order = {
        "order_id": order_id,
        "order_status": "processed",
        "order_date": time.time(),
        "order_total": order_total,
        "order_items": dict(cart),
    }

    state["orders"][order_id] = order
    state["cart"] = {}

    save_app_state(APP_NAME, state)

    log_action("checkout", {"discount_code": discount_code}, ret=order_id, write=True)
    json_output({"status": "success", "order_id": order_id, "total": order_total})


@cli.command("get-discount-code-info")
@click.option("--discount-code", required=True, help="Discount code to look up.")
def get_discount_code_info(discount_code: str):
    """Get information about a specific discount code."""
    state = _load_state()
    discount_codes = state["discount_codes"]

    # Aggregate which items accept this code
    result = {}
    for iid, codes in discount_codes.items():
        if discount_code in codes:
            result[iid] = codes[discount_code]

    log_action("get_discount_code_info", {"discount_code": discount_code}, ret=result)
    json_output({"status": "success", "discount_code": discount_code, "info": result})


@cli.command("get-all-discount-codes")
def get_all_discount_codes():
    """List all available discount codes."""
    state = _load_state()
    discount_codes = state["discount_codes"]

    log_action("get_all_discount_codes", {}, ret=discount_codes)
    json_output({"status": "success", "discount_codes": discount_codes})


@cli.command("list-orders")
def list_orders():
    """List all orders."""
    state = _load_state()
    orders = state["orders"]

    log_action("list_orders", {}, ret=orders)
    json_output({"status": "success", "orders": orders})


@cli.command("get-order-details")
@click.option("--order-id", required=True, help="Order ID to get details for.")
def get_order_details(order_id: str):
    """Get details for a specific order."""
    state = _load_state()
    orders = state["orders"]

    if order_id not in orders:
        cli_error("Order does not exist")

    order = orders[order_id]
    log_action("get_order_details", {"order_id": order_id}, ret=order)
    json_output({"status": "success", "order": order})


@cli.command("cancel-order")
@click.option("--order-id", required=True, help="Order ID to cancel.")
def cancel_order(order_id: str):
    """Cancel an order. Only orders with status 'processed' or 'shipped' can be cancelled."""
    state = _load_state()
    orders = state["orders"]

    if order_id not in orders:
        cli_error("Order does not exist")

    current_status = orders[order_id].get("order_status", "")
    if current_status not in ("processed", "shipped"):
        cli_error(
            f"Order {order_id} cannot be cancelled, current status: {current_status}"
        )

    orders[order_id]["order_status"] = "cancelled"
    save_app_state(APP_NAME, state)

    log_action("cancel_order", {"order_id": order_id}, ret=order_id, write=True)
    json_output({"status": "success", "order_id": order_id})


# ---- ENV commands (hidden, invoked by gaia2-eventd) -------------------------


@cli.command("add-product", hidden=True)
@click.option("--name", required=True, help="Product name.")
def env_add_product(name: str):
    """[ENV] Add a new product to the catalog."""
    state = _load_state()

    product_id = uuid.uuid4().hex
    state["products"][product_id] = {
        "name": name,
        "product_id": product_id,
        "variants": {},
    }

    save_app_state(APP_NAME, state)
    log_action("add_product", {"name": name}, ret=product_id, write=True)
    json_output({"status": "success", "product_id": product_id})


@cli.command("add-item-to-product", hidden=True)
@click.option("--product-id", required=True, help="Product ID to add item to.")
@click.option("--price", required=True, type=float, help="Item price.")
@click.option("--options", default="{}", help="JSON dict of item options.")
@click.option("--available", default="True", help="Whether item is available.")
def env_add_item_to_product(
    product_id: str, price: float, options: str, available: str
):
    """[ENV] Add an item variant to a product."""
    import json as _json

    state = _load_state()
    products = state["products"]

    if product_id not in products:
        cli_error("Product does not exist")

    parsed_options = {}
    if options:
        try:
            parsed_options = _json.loads(options)
        except _json.JSONDecodeError:
            parsed_options = {}

    avail = str(available).lower() not in ("false", "0", "no")

    item_id = uuid.uuid4().hex
    products[product_id].setdefault("variants", {})[item_id] = {
        "item_id": item_id,
        "price": price,
        "available": avail,
        "options": parsed_options,
    }

    save_app_state(APP_NAME, state)
    log_action(
        "add_item_to_product",
        {
            "product_id": product_id,
            "price": price,
            "options": parsed_options,
            "available": avail,
        },
        ret=item_id,
        write=True,
    )
    json_output({"status": "success", "item_id": item_id})


@cli.command("update-item", hidden=True)
@click.option("--item-id", required=True, help="Item ID to update.")
@click.option("--new-price", type=float, default=None, help="New price.")
@click.option("--new-availability", default=None, help="New availability (True/False).")
def env_update_item(
    item_id: str, new_price: float | None, new_availability: str | None
):
    """[ENV] Update an item's price and/or availability."""
    if new_price is None and new_availability is None:
        cli_error("No update provided")

    state = _load_state()
    products = state["products"]

    for pid, product in products.items():
        variants = product.get("variants", {})
        if item_id in variants:
            item = variants[item_id]
            if new_price is not None:
                item["price"] = new_price
            if new_availability is not None:
                item["available"] = str(new_availability).lower() not in (
                    "false",
                    "0",
                    "no",
                )
            save_app_state(APP_NAME, state)
            log_action(
                "update_item",
                {
                    "item_id": item_id,
                    "new_price": new_price,
                    "new_availability": new_availability,
                },
                ret=item_id,
                write=True,
            )
            json_output({"status": "success", "item_id": item_id})
            return

    cli_error(f"Item with id {item_id} does not exist")


@cli.command("add-discount-code", hidden=True)
@click.option("--item-id", required=True, help="Item ID for the discount.")
@click.option("--discount-code", required=True, help="JSON dict {code: discount_pct}.")
def env_add_discount_code(item_id: str, discount_code: str):
    """[ENV] Add a discount code for an item."""
    import json as _json

    state = _load_state()
    products = state["products"]

    # Validate item exists
    found = False
    for pid, product in products.items():
        if item_id in product.get("variants", {}):
            found = True
            break
    if not found:
        cli_error(f"Item {item_id} does not exist in the inventory.")

    try:
        parsed = _json.loads(discount_code)
    except _json.JSONDecodeError:
        cli_error("Invalid JSON for discount_code")
    state["discount_codes"].setdefault(item_id, {}).update(parsed)

    save_app_state(APP_NAME, state)
    log_action(
        "add_discount_code",
        {"item_id": item_id, "discount_code": parsed},
        ret="Successfully added the discount code",
        write=True,
    )
    json_output(
        {"status": "success", "message": "Successfully added the discount code"}
    )


@cli.command("update-order-status", hidden=True)
@click.option("--order-id", required=True, help="Order ID to update.")
@click.option("--status", default="shipped", help="New status.")
def env_update_order_status(order_id: str, status: str):
    """[ENV] Update an order's status."""
    state = _load_state()
    orders = state["orders"]

    if order_id not in orders:
        cli_error("Order does not exist")

    if status not in ("processed", "shipped", "delivered", "cancelled"):
        cli_error("Invalid status")

    orders[order_id]["order_status"] = status

    save_app_state(APP_NAME, state)
    log_action(
        "update_order_status",
        {"order_id": order_id, "status": status},
        ret=order_id,
        write=True,
    )
    json_output({"status": "success", "order_id": order_id})


# ---- SCHEMA ---------------------------------------------------------------


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
