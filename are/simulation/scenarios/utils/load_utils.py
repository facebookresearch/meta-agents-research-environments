# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


import logging
import os
from typing import Callable

from fsspec.core import url_to_fs

from are.simulation.apps.app import App
from are.simulation.data_handler.importer import JsonScenarioImporter
from are.simulation.dataset_helpers import get_data_path
from are.simulation.scenarios.scenario import Scenario
from are.simulation.types import CapabilityTag
from are.simulation.utils.huggingface import (
    get_scenario_from_huggingface,
    list_huggingface_scenarios,
)

logger = logging.getLogger(__name__)

# HuggingFace dataset constants for demo universes
HF_DEMO_DATASET_NAME = os.environ.get(
    "HF_DEMO_DATASET_NAME", "meta-agents-research-environments/gaia2"
)
HF_DEMO_CONFIG = os.environ.get("HF_DEMO_CONFIG", "demo")
HF_DEMO_SPLIT = os.environ.get("HF_DEMO_SPLIT", "validation")


def load_scenario(
    path: str,
    apps_to_skip: list[str] | None = None,
    apps_to_keep: list[str] | None = None,
) -> Scenario:
    scenario_importer = JsonScenarioImporter()
    fs, path = url_to_fs(path)
    with fs.open(path, "r") as f:
        scenario_json = f.read()
    scenario, _, _ = scenario_importer.import_from_json(
        scenario_json,
        apps_to_skip=apps_to_skip,
        apps_to_keep=apps_to_keep,
        load_completed_events=False,
    )
    return scenario


def load_template_apps_and_time(
    universe_path: str,
    apps_to_skip: list[str] | None = None,
    apps_to_keep: list[str] | None = None,
) -> tuple[list[App], float | None]:
    scenario = load_scenario(
        universe_path, apps_to_skip=apps_to_skip, apps_to_keep=apps_to_keep
    )
    scenario.initialize()
    return scenario.apps or [], scenario.start_time


def load_and_setup_scenario(
    file_path: str, scenario_id: str, tags: tuple = ()
) -> Scenario:
    try:
        logger.info(f"Loading scenario {file_path} into {scenario_id}")
        scenario = load_scenario(file_path)
        scenario.scenario_id = scenario_id
        if tags:
            scenario.tags = tags
        return scenario
    except Exception as e:
        logger.error(f"Error loading scenario from {file_path}: {str(e)}")
        raise e


def load_scenarios_from_directory(
    directory: str, file_filter: Callable[[str], bool], tags: tuple = ()
) -> dict[str, Callable[[], Scenario]]:
    scenarios = {}
    fs, path = url_to_fs(directory)
    for file in fs.ls(path):
        filename_with_ext = os.path.basename(file)
        if filename_with_ext.endswith(".json") and file_filter(filename_with_ext):
            filename = os.path.splitext(filename_with_ext)[0]
            scenario_id = f"scenario_{filename}"
            scenarios[scenario_id] = lambda path=file, id=scenario_id, t=tags: (
                load_and_setup_scenario(path, id, t)
            )
            logger.debug(
                f"Prepared to load scenario {filename_with_ext} into {scenario_id}"
            )
    return scenarios


def load_scenarios_from_huggingface(
    dataset_name: str, dataset_config: str, dataset_split: str, tags: tuple = ()
) -> dict[str, Callable[[], Scenario]]:
    """Load scenarios from a HuggingFace dataset using lazy loading for memory efficiency.

    This function only loads scenario IDs upfront and fetches scenario data on-demand
    when scenarios are actually needed, minimizing memory usage.

    Args:
        dataset_name: The name of the HuggingFace dataset
        dataset_config: The config (subset) of the dataset
        dataset_split: The split of the dataset
        tags: Tuple of tags to apply to all scenarios

    Returns:
        Dictionary mapping scenario IDs to callable functions that return Scenario objects
    """
    scenarios = {}

    # Get list of scenario IDs (lightweight operation)
    scenario_ids = list_huggingface_scenarios(
        dataset_name, dataset_config, dataset_split
    )

    if scenario_ids is None:
        logger.error(
            f"Failed to list scenarios from HuggingFace dataset {dataset_name}/{dataset_config}/{dataset_split}"
        )
        return scenarios

    logger.info(
        f"Found {len(scenario_ids)} scenarios in HuggingFace dataset {dataset_name}/{dataset_config}/{dataset_split}"
    )

    # Define the lazy loader function once
    def create_lazy_loader(dataset: str, config: str, split: str, sid: str, t: tuple):
        def load_scenario_on_demand() -> Scenario:
            try:
                logger.info(f"Loading HuggingFace scenario {sid} on-demand")
                scenario_data = get_scenario_from_huggingface(
                    dataset, config, split, sid
                )
                if scenario_data is None:
                    raise ValueError(f"Scenario {sid} not found in HuggingFace dataset")

                scenario_importer = JsonScenarioImporter()
                scenario, _, _ = scenario_importer.import_from_json(
                    scenario_data,
                    apps_to_skip=None,
                    apps_to_keep=None,
                    load_completed_events=False,
                )
                scenario.scenario_id = sid
                if t:
                    scenario.tags = t
                return scenario
            except Exception as e:
                logger.error(f"Error loading HuggingFace scenario {sid}: {str(e)}")
                raise e

        return load_scenario_on_demand

    # Create lazy loaders for each scenario
    for scenario_id in scenario_ids:
        scenarios[scenario_id] = create_lazy_loader(
            dataset_name, dataset_config, dataset_split, scenario_id, tags
        )
        logger.debug(f"Prepared lazy loader for HuggingFace scenario {scenario_id}")

    return scenarios


def load_universe(path: str):
    scenario = load_scenario(path)
    scenario.tags = (CapabilityTag.Universe,)
    return scenario


def load_all_universes() -> dict[str, Callable[[], Scenario]]:
    directory = os.path.join(get_data_path(), "scenarios")
    return load_scenarios_from_directory(
        directory, lambda f: "synthetic" not in f, tags=(CapabilityTag.Universe,)
    )


def load_hf_demo_universes() -> dict[str, Callable[[], Scenario]]:
    """Load demo universes from HuggingFace dataset.

    Returns:
        Dictionary mapping scenario IDs to callable functions that return Scenario objects
        with CapabilityTag.Universe tag applied.
    """
    return load_scenarios_from_huggingface(
        HF_DEMO_DATASET_NAME,
        HF_DEMO_CONFIG,
        HF_DEMO_SPLIT,
        tags=(CapabilityTag.Universe,),
    )
