# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


"""
Tests for the shopping substitution scenario.

These cover the part that is easy to get wrong: the validator has to accept any
product matching the attributes the user named, and reject the near-duplicates
that differ on one of them.
"""

import pytest

from are.simulation.apps.agent_user_interface import AgentUserInterface
from are.simulation.apps.shopping import ShoppingApp
from are.simulation.environment import Environment, EnvironmentConfig
from are.simulation.scenarios.scenario_shopping_substitution.scenario import (
    REQUESTED_PRODUCT,
    SUBSTITUTE_PRODUCT,
    ScenarioShoppingSubstitution,
)


def _scenario_with_env() -> tuple[
    ScenarioShoppingSubstitution, ShoppingApp, Environment
]:
    """Build the scenario and register its apps on a real environment."""
    scenario = ScenarioShoppingSubstitution()
    scenario.init_and_populate_apps()
    shopping = next(a for a in scenario.apps or [] if isinstance(a, ShoppingApp))
    aui = next(a for a in scenario.apps or [] if isinstance(a, AgentUserInterface))

    env = Environment(EnvironmentConfig(start_time=0, duration=scenario.duration))
    env.register_apps([shopping, aui])
    return scenario, shopping, env


def _item_id(shopping: ShoppingApp, product_name: str, variant_index: int = 0) -> str:
    product = next(p for p in shopping.products.values() if p.name == product_name)
    return list(product.variants.keys())[variant_index]


def _buy_and_validate(
    product_name: str,
    variant_index: int = 0,
    also_buy: tuple[str, int] | None = None,
) -> bool:
    scenario, shopping, env = _scenario_with_env()
    shopping.add_to_cart(item_id=_item_id(shopping, product_name, variant_index))
    if also_buy is not None:
        shopping.add_to_cart(item_id=_item_id(shopping, *also_buy))
    shopping.checkout()
    return scenario.validate(env).success is True


@pytest.mark.parametrize(
    "product_name,variant_index",
    [
        (REQUESTED_PRODUCT, 0),  # the requested item itself
        (SUBSTITUTE_PRODUCT, 0),  # equivalent, different brand
        ("Straus Organic Milk 2L", 0),  # equivalent, different brand
    ],
)
def test_accepts_any_item_matching_the_named_attributes(product_name, variant_index):
    assert _buy_and_validate(product_name, variant_index) is True


@pytest.mark.parametrize(
    "product_name,variant_index,reason",
    [
        ("Clover Milk 2L", 0, "same brand and fat and size, but not organic"),
        ("Clover Organic Milk 1L", 0, "wrong size"),
        (REQUESTED_PRODUCT, 1, "whole milk rather than 2%"),
        (SUBSTITUTE_PRODUCT, 1, "1% rather than 2%"),
    ],
)
def test_rejects_near_duplicates(product_name, variant_index, reason):
    assert _buy_and_validate(product_name, variant_index) is False, reason


def test_rejects_ordering_more_than_one_item():
    assert (
        _buy_and_validate(SUBSTITUTE_PRODUCT, 0, also_buy=("Straus Organic Milk 2L", 0))
        is False
    )


def test_rejects_ordering_nothing():
    scenario, _, env = _scenario_with_env()
    assert scenario.validate(env).success is not True


def test_sold_out_item_cannot_be_purchased():
    """The scenario's premise: once it sells out, the cart refuses it."""
    _, shopping, _ = _scenario_with_env()
    item_id = _item_id(shopping, REQUESTED_PRODUCT, 0)
    shopping.update_item(item_id=item_id, new_availability=False)
    with pytest.raises(ValueError, match="not available"):
        shopping.add_to_cart(item_id=item_id)


def test_oracle_path_satisfies_the_validator():
    """
    Run the scenario's own event graph in oracle mode and check it passes.

    This is what catches a wiring mistake in build_events_flow, such as the
    oracle buying an item that the validator would reject.
    """
    scenario = ScenarioShoppingSubstitution()
    scenario.initialize()

    env = Environment(EnvironmentConfig(oracle_mode=True, queue_based_loop=True))
    env.exit_when_no_events = True
    env.run(scenario)

    assert scenario.validate(env).success is True
