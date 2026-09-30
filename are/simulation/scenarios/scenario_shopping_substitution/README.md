# ScenarioShoppingSubstitution

## Overview

The `scenario_shopping_substitution` scenario tests whether an agent picks the
*right* product, not just whether it can complete a purchase.

The user asks for a specific item: Clover organic 2% milk in the 2 litre size.
Ten seconds into the episode, that exact item goes out of stock. The agent has
to notice and choose a substitute.

## Why this is different from a static shopping task

`ShoppingApp`'s tool surface is transactional: search, cart, checkout, orders,
cancel. An agent that searches, adds the first result and checks out completes a
conventional shopping task whether or not it chose correctly. This scenario is
built so that shortcut fails.

The catalog holds near-duplicates that differ only on the attributes the user
named:

| Product | organic | fat | size | Acceptable |
|---|---|---|---|---|
| Clover Organic Milk 2L | yes | 2% | 2L | yes, but it sells out mid-episode |
| Horizon Organic Milk 2L | yes | 2% | 2L | yes |
| Straus Organic Milk 2L | yes | 2% | 2L | yes |
| Clover Milk 2L | **no** | 2% | 2L | no |
| Clover Organic Milk 1L | yes | 2% | **1L** | no |
| Clover Organic Milk 2L (whole) | yes | **whole** | 2L | no |
| Horizon Organic Milk 2L (1%) | yes | **1%** | 2L | no |

`search_product` matches on product name only, so the closest match by name and
by brand loyalty is `Clover Milk 2L`, which is identical to the request except
that it is not organic. Choosing it is the characteristic failure of production
shopping agents, and it is the one this scenario is designed to catch.

## Validation

`validate` does not pin a single gold item. It derives the acceptable set from
the attributes the user named, so any organic 2% 2L product passes on merit
regardless of brand. Brand is deliberately excluded from the required
attributes: substituting brand is the reasonable thing to do once the requested
one is unavailable.

The agent passes if it ordered exactly one item and that item matches every
named attribute. Ordering nothing, ordering more than one item, or ordering any
of the near-duplicates fails, with the mismatched attributes reported in the
validation exception.

The helper that performs this check, `matches_requested_attributes`, takes plain
dicts and knows nothing about shopping. It is written to be reusable by any
scenario that wants to score a substitution decision against stated constraints
rather than against one predetermined answer.

## Running

```bash
python -m are.simulation.scenarios.scenario_shopping_substitution.scenario
```

This runs the scenario in oracle mode and validates the reference solution.
