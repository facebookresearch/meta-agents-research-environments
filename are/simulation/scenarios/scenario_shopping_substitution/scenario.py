# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


"""
Shopping substitution scenario.

The agent is asked for a specific product. Partway through the episode the exact
item goes out of stock, so the agent has to notice and choose a substitute.

What makes this different from a static shopping task is that the catalog holds
near-duplicates which differ only on the attributes the user named. Selecting the
first plausible search hit fails, because the closest name match is the wrong
product. Validation derives the acceptable set from the user's stated constraints
rather than pinning a single gold item, so any genuinely equivalent product passes.
"""

from typing import Any

from are.simulation.apps.agent_user_interface import AgentUserInterface
from are.simulation.apps.shopping import ShoppingApp
from are.simulation.scenarios.scenario import Scenario, ScenarioValidationResult
from are.simulation.scenarios.utils.registry import register_scenario
from are.simulation.types import EventRegisterer, disable_events

# The attributes the user names in their request, and the values they asked for.
# An item is an acceptable substitute only if it matches every one of them.
# Brand is deliberately absent: the user asked for a brand, but an equivalent
# product from another brand is a reasonable substitution once theirs is gone.
REQUESTED: dict[str, str] = {"organic": "yes", "fat": "2%", "size": "2L"}

REQUESTED_PRODUCT = "Clover Organic Milk 2L"
SUBSTITUTE_PRODUCT = "Horizon Organic Milk 2L"


def matches_requested_attributes(
    options: dict[str, Any], requested: dict[str, str]
) -> bool:
    """
    Whether an item satisfies every attribute the user named.

    Deliberately generic: it takes plain dicts and knows nothing about shopping.
    Any scenario that wants to score a substitution decision against stated
    constraints, rather than against one predetermined answer, can use this
    shape. Happy to move it to scenarios/utils/validation_utils.py if that is
    where you would rather it live.
    """
    return all(options.get(name) == value for name, value in requested.items())


# name -> list of variants. Every variant carries the full attribute set so the
# agent has to inspect variants rather than trust the product name.
CATALOG: dict[str, list[dict[str, Any]]] = {
    # The requested product. Goes out of stock mid-episode.
    REQUESTED_PRODUCT: [
        {"price": 5.49, "organic": "yes", "fat": "2%", "size": "2L"},
        {"price": 5.49, "organic": "yes", "fat": "whole", "size": "2L"},
    ],
    # Acceptable substitutes: same attributes, different brand.
    SUBSTITUTE_PRODUCT: [
        {"price": 5.99, "organic": "yes", "fat": "2%", "size": "2L"},
        {"price": 5.99, "organic": "yes", "fat": "1%", "size": "2L"},
    ],
    "Straus Organic Milk 2L": [
        {"price": 6.49, "organic": "yes", "fat": "2%", "size": "2L"},
    ],
    # Trap: same brand, same fat, same size, but not organic. This is the
    # closest match by name and by brand loyalty, and it is wrong.
    "Clover Milk 2L": [
        {"price": 4.29, "organic": "no", "fat": "2%", "size": "2L"},
    ],
    # Trap: same brand, organic, right fat, wrong size.
    "Clover Organic Milk 1L": [
        {"price": 3.49, "organic": "yes", "fat": "2%", "size": "1L"},
    ],
}

USER_REQUEST = (
    "We're out of milk. Can you reorder the Clover organic 2% milk, the 2 litre "
    "one? If Clover doesn't have it, anything equivalent is fine, but it does "
    "need to be organic 2% and the 2 litre size."
)


@register_scenario("scenario_shopping_substitution")
class ScenarioShoppingSubstitution(Scenario):
    start_time: float | None = 0
    duration: float | None = 60

    # item_id of the requested item, resolved during population.
    requested_item_id: str | None = None
    # item_id the oracle buys once the requested one is gone.
    substitute_item_id: str | None = None

    def init_and_populate_apps(self, *args, **kwargs) -> None:
        aui = AgentUserInterface()
        shopping = ShoppingApp()

        for product_name, variants in CATALOG.items():
            product_id = shopping.add_product(name=product_name)
            for variant in variants:
                options = {k: v for k, v in variant.items() if k != "price"}
                item_id = shopping.add_item_to_product(
                    product_id=product_id,
                    price=variant["price"],
                    options=options,
                )
                if product_name == REQUESTED_PRODUCT and self._matches(options):
                    self.requested_item_id = item_id
                elif product_name == SUBSTITUTE_PRODUCT and self._matches(options):
                    self.substitute_item_id = item_id

        self.apps = [aui, shopping]

    def build_events_flow(self) -> None:
        aui = self.get_typed_app(AgentUserInterface)
        shopping = self.get_typed_app(ShoppingApp)

        assert self.requested_item_id is not None
        assert self.substitute_item_id is not None

        with EventRegisterer.capture_mode():
            request = aui.send_message_to_agent(
                content=USER_REQUEST,
            ).depends_on(None, delay_seconds=1)

            # The requested item sells out after the agent has started working.
            # This is the point of the scenario: the answer changes mid-episode.
            sold_out = shopping.update_item(
                item_id=self.requested_item_id,
                new_availability=False,
            ).depends_on(request, delay_seconds=10)

            oracle_cart = (
                shopping.add_to_cart(item_id=self.substitute_item_id, quantity=1)
                .oracle()
                .depends_on(sold_out, delay_seconds=1)
            )
            oracle_checkout = (
                shopping.checkout().oracle().depends_on(oracle_cart, delay_seconds=1)
            )

        self.events = [request, sold_out, oracle_cart, oracle_checkout]

    @staticmethod
    def _matches(options: dict[str, Any]) -> bool:
        return matches_requested_attributes(options, REQUESTED)

    def validate(self, env) -> ScenarioValidationResult:
        """
        The agent passes if it ordered exactly one item and that item satisfies
        every attribute the user named.

        Ordering the sold-out item, the non-organic Clover, the 1L, or the wrong
        fat percentage all fail. Any organic 2% 2L product passes regardless of
        brand, so the agent is scored on the decision rather than on matching a
        single predetermined answer.
        """
        try:
            shopping: ShoppingApp = env.get_app(ShoppingApp.__name__)  # type: ignore[assignment]

            with disable_events():
                orders = shopping.list_orders()

            if len(orders) != 1:
                return ScenarioValidationResult(
                    success=False,
                    exception=AssertionError(
                        f"expected exactly 1 order, found {len(orders)}"
                    ),
                )

            order = next(iter(orders.values()))
            if len(order.order_items) != 1:
                return ScenarioValidationResult(
                    success=False,
                    exception=AssertionError(
                        f"expected exactly 1 item in the order, "
                        f"found {len(order.order_items)}"
                    ),
                )

            purchased = next(iter(order.order_items.values()))
            options = purchased.options
            if not self._matches(options):
                mismatched = {
                    name: options.get(name)
                    for name, value in REQUESTED.items()
                    if options.get(name) != value
                }
                return ScenarioValidationResult(
                    success=False,
                    exception=AssertionError(
                        f"purchased item does not match the request on {mismatched}"
                    ),
                )

            return ScenarioValidationResult(success=True)

        except Exception as e:
            return ScenarioValidationResult(success=False, exception=e)


if __name__ == "__main__":
    from are.simulation.scenarios.utils.cli_utils import run_and_validate

    run_and_validate(ScenarioShoppingSubstitution())
