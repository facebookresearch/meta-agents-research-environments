# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


import importlib
import inspect
from json import JSONDecodeError
from typing import Type

from are.simulation.scenarios import Scenario


def import_scenario(scenario_file: str) -> Type[Scenario]:
    """
    Get a scenario from scenario_file by dynamically importing module and searching scenario in it
    :param scenario_file: file to import scenario from
    :returns: scenario type
    """
    try:
        m = importlib.import_module(f"are.simulation.scenarios.{scenario_file}")
    except JSONDecodeError:
        raise ImportError(
            f"Failed to import {scenario_file} because of JSONDecodeError. This may be due to missing JSON files in are_simulation/datasets/scenarios. Ensure git LFS is installed and execute 'git lfs fetch --all' followed by 'git lfs pull'."
        )

    m_scenarios = inspect.getmembers(
        m,
        lambda obj: (
            inspect.isclass(obj)
            and obj.__module__
            == m.__name__  # ignore classes that are imported from other modules
            and hasattr(obj, "scenario_id")
            and obj.scenario_id
        ),
    )
    if len(m_scenarios) != 1:
        raise ValueError(
            f"Expected exactly 1 scenario in {scenario_file} got {len(m_scenarios)}"
        )
    return m_scenarios[0][1]
