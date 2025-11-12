# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


import logging
import re
from typing import cast

from are.simulation.apps import SystemApp
from are.simulation.data_handler.exporter import JsonScenarioExporter
from are.simulation.data_handler.importer import JsonScenarioImporter
from are.simulation.data_handler.models import ExportedApp
from are.simulation.environment import Environment, EnvironmentConfig
from are.simulation.scenarios.config import ScenarioRunnerConfig
from are.simulation.scenarios.utils.scenario_expander import EnvEventsConfig
from are.simulation.scenarios.utils.turn_conditions import (
    is_conditional_event,
    is_send_message_to_user,
)
from are.simulation.types import (
    AbstractEvent,
    CapabilityTag,
    CompletedEvent,
    Event,
    EventType,
    ExecutionMetadata,
    OracleEvent,
    PlaceholderMetadata,
    ToolAugmentationConfig,
)
from are.simulation.validation import BaseJudgeConfig, JudgeFactory

from .benchmark_scenario import BenchmarkScenarioImportedFromJson

ORACLE_EVENT_CLASS_NAME = "OracleEvent"

logger: logging.Logger = logging.getLogger(__name__)


def preprocess_scenario(
    scenario: BenchmarkScenarioImportedFromJson,
    judge_config: BaseJudgeConfig | None = None,
    max_scenario_duration: int | None = None,
    offline_validation: bool = False,
    tool_augmentation_config: ToolAugmentationConfig | None = None,
    env_events_config: EnvEventsConfig | None = None,
):
    """
    Preprocess the scenario by running it in oracle mode and attaching the oracle run events to the scenario.
    These oracle events will  be used to validate the scenario.
    It then initializes the turns of the scenario by triggering the judge and creates a validation function.
    """
    # If SystemApp not in apps, add it
    if scenario.serialized_apps and "SystemApp" not in [
        serialized_app.name for serialized_app in scenario.serialized_apps
    ]:
        scenario.serialized_apps.append(
            ExportedApp(name="SystemApp", class_name="SystemApp", app_state={})
        )

    if scenario.apps is not None and "SystemApp" not in [
        app.name for app in scenario.apps
    ]:
        scenario.apps.append(SystemApp())

    if max_scenario_duration is not None:
        if scenario.duration is not None:
            logger.warning(
                f"Scenario duration overridden to {max_scenario_duration} instead of {scenario.duration} seconds"
            )
        else:
            logger.info(f"Scenario duration set to {max_scenario_duration} seconds")
        scenario.duration = max_scenario_duration

    scenario.tool_augmentation_config = tool_augmentation_config
    scenario.env_events_config = env_events_config

    # Initialize the scenario
    scenario.initialize()
    # Set the judge
    judge = None
    has_oracle_events = any(isinstance(e, OracleEvent) for e in scenario.events)
    if not has_oracle_events:
        logger.info(
            f"{scenario.scenario_id}: scenario has no oracle events, judge is not set"
        )
    if judge_config and has_oracle_events:
        # Run the scenario in oracle mode to get the oracle completed events
        env = Environment(
            EnvironmentConfig(
                oracle_mode=True, queue_based_loop=True, start_time=scenario.start_time
            )
        )
        env.run(scenario)
        env.stop()
        oracle_run_event_log = env.event_log.list_view()
        # Check clean run
        if any(e.failed() for e in oracle_run_event_log):
            raise Exception(
                f"Oracle run failed: {[e.metadata.exception for e in oracle_run_event_log if e.failed()]}"
            )
        # Soft reset the scenario
        scenario.soft_reset()
        # Attach the oracle run events to the scenario for the judge
        scenario.oracle_run_event_log = env.event_log.list_view()
        # Instantiate the judge
        judge_factory = JudgeFactory()
        judge = judge_factory(judge_config)
        scenario.judge = judge  # type: ignore
    else:
        logger.info(f"{scenario.scenario_id}: judge is not set")
    # Initialize the turns of the scenario
    # The different cases are:
    # 1. Test scenario creation: no validation, dummy trigger condition
    # 2. Test scenario execution: no validation, no trigger condition (already present in the scenario events)
    # 3. Test scenario evaluation: offline validation, no trigger condition (scenario will be run in oracle mode)
    # 4. Validation scenario execution/evaluation: online validation, judge trigger condition
    if judge is not None:
        validation_mode = "offline" if offline_validation else "online"
        validation_fn = (
            judge.validate_current_turn if offline_validation else judge.validate
        )
        trigger_mode = (
            "without trigger condition"
            if offline_validation
            else "with judge trigger condition"
        )
        trigger_condition = None if offline_validation else judge.trigger_condition
        is_end_of_turn_event = is_send_message_to_user
    else:
        validation_mode = "no validation"
        validation_fn = None
        trigger_mode = (
            "with dummy trigger condition"
            if has_oracle_events
            else "without trigger condition"
        )
        trigger_condition = (lambda _, __: (True, {})) if has_oracle_events else None
        is_end_of_turn_event = (
            is_send_message_to_user if has_oracle_events else is_conditional_event
        )
    scenario.initialize_turns(
        trigger_condition=trigger_condition,  # type: ignore
        validation_fn=validation_fn,  # type: ignore
        offline_validation=offline_validation,  # Not used in this case
        is_end_of_turn_event=is_end_of_turn_event,
    )
    if judge is not None:
        # Initialize the state of the judge
        judge.initialize_state(scenario)

    logger.info(f"[{scenario.scenario_id}]: Initializing turns {trigger_mode}")
    logger.info(f"[{scenario.scenario_id}]: Validation mode {validation_mode}")
    logger.info(f"[{scenario.scenario_id}]: Scenario has {scenario.nb_turns} turns")


def is_oracle_event(event: dict) -> bool:
    """Check if an event is an oracle event."""
    return event.get("class_name") == "OracleEvent"


def create_placeholder(
    placeholder_event: AbstractEvent,
    parent_event: OracleEvent,
    event_id_to_turn_idx: dict[str, int],
) -> PlaceholderMetadata:
    """
    Create a placeholder event metadata.

    Args:
        placeholder_event: The event containing the placeholder
        parent_event: The parent event that the placeholder refers to
        event_id_to_turn_idx: Dictionary mapping event IDs to turn indices

    Returns:
        PlaceholderMetadata object with information about the placeholder
    """
    # Extract the parent tool name from the parent event
    function_name = getattr(parent_event.action_desc, "function", None)
    if function_name is None:
        raise Exception(f"Function name not found for event: {parent_event.event_id}")
    app_name = getattr(parent_event.action_desc, "app", None)
    if app_name is None:
        raise Exception(f"App name not found for event: {parent_event.event_id}")
    parent_tool_name = f"{app_name}__{function_name}"

    # Get turn indices for both events
    parent_turn_idx = event_id_to_turn_idx.get(parent_event.event_id, -1)
    placeholder_turn_idx = event_id_to_turn_idx.get(placeholder_event.event_id, -1)

    # Create and return the placeholder metadata
    return PlaceholderMetadata(
        parent_tool_name=parent_tool_name,
        parent_turn_idx=parent_turn_idx,
        parent_event_id=parent_event.event_id,
        placeholder_turn_idx=placeholder_turn_idx,
        placeholder_event_id=placeholder_event.event_id,
    )


def flag_same_tool_used_in_same_turn(
    placeholder: PlaceholderMetadata,
    scenario: BenchmarkScenarioImportedFromJson,
) -> bool:
    """
    Determine if the same tool is used in the same turn by another agent event.

    Args:
        placeholder (PlaceholderMetadata): Metadata of the placeholder event.
        scenario (BenchmarkScenarioImportedFromJson): The scenario containing events and their mappings.

    Returns:
        bool: True if the same tool is used in the same turn by another agent event, False otherwise.
    """
    if not scenario.event_id_to_turn_idx:
        raise ValueError("Event id to turn not found")

    turn_idx = placeholder.parent_turn_idx
    event_id = placeholder.parent_event_id
    tool_name = placeholder.parent_tool_name
    app_name, function_name = tool_name.split("__")

    for event in scenario.events:
        if (
            isinstance(event, OracleEvent)
            and event.event_id != event_id
            and scenario.event_id_to_turn_idx.get(event.event_id) == turn_idx
            and getattr(event.action_desc, "function", None) == function_name
            and getattr(event.action_desc, "app", None) == app_name
        ):
            logger.info(
                f"Scenario: {scenario.scenario_id}: Same tool {tool_name} used in same turn {turn_idx} for event: {event.event_id}, placeholder: {event_id}"
            )
            return True
    return False


def extract_placeholders(
    scenario: BenchmarkScenarioImportedFromJson,
) -> tuple[list[PlaceholderMetadata], bool]:
    """
    Extract placeholder events and convert them to PlaceholderMetadata.

    Args:
        scenario: The scenario object containing events and event_id_to_turn_idx mapping

    Returns:
        List of PlaceholderMetadata objects
    """
    if scenario.event_id_to_turn_idx is None:
        raise ValueError("Event id to turn not found")
    placeholders = []
    same_tool_flag = False
    for abstract_event in scenario.events:
        # Only consider USER or ENV events
        if abstract_event.event_type not in {EventType.USER, EventType.ENV}:
            continue

        # Cast to Event type which has action attribute
        event = cast(Event, abstract_event)

        # Skip events without action or args
        if not (
            hasattr(event, "action")
            and event.action
            and hasattr(event.action, "args")
            and event.action.args
        ):
            continue

        # Check each argument for placeholder syntax
        for arg_value in event.action.args.values():
            if isinstance(arg_value, str):
                # Look for {{event_id}} pattern
                match = re.match(r"^\{\{(.*?)\}\}$", arg_value)
                if match:
                    resolved_event_id = match.group(1).split(".")[0]

                    # Find the referenced parent event
                    for past_event in scenario.events:
                        # We only care about Oracle events
                        if resolved_event_id == past_event.event_id and isinstance(
                            past_event, OracleEvent
                        ):
                            # Convert events to placeholder metadata
                            placeholders.append(
                                create_placeholder(
                                    placeholder_event=abstract_event,
                                    parent_event=past_event,
                                    event_id_to_turn_idx=scenario.event_id_to_turn_idx,
                                )
                            )
                            # Check if the same tool is used in the same turn
                            if flag_same_tool_used_in_same_turn(
                                placeholder=placeholders[-1], scenario=scenario
                            ):
                                same_tool_flag = True
                            break
    return (placeholders, same_tool_flag)


def extract_execution_metadata(
    scenario: BenchmarkScenarioImportedFromJson,
) -> ExecutionMetadata:
    """
    Extract execution metadata from a scenario.

    Args:
        scenario: The scenario object to extract metadata from.
    """
    # Extract placeholders and same tool flag
    placeholders, same_tool_flag = extract_placeholders(scenario)
    return ExecutionMetadata(
        placeholders=placeholders,
        has_placeholder_conflicts=same_tool_flag,
    )


def preprocess_scenario_for_execution_without_oracle(
    scenario: BenchmarkScenarioImportedFromJson,
) -> BenchmarkScenarioImportedFromJson:
    # Preprocess the scenario to add dummy turn triggers
    preprocess_scenario(
        scenario,
        judge_config=None,
    )

    # Extract execution metadata
    execution_metadata = extract_execution_metadata(scenario)

    # Append the execution metadata to the scenario
    scenario.execution_metadata = execution_metadata

    # Return the updated scenario
    return scenario


def preprocess_scenario_str_for_execution_without_oracle(
    scenario_str: str,
) -> str:
    scenario, _, _ = JsonScenarioImporter().import_from_json_to_benchmark(
        scenario_str, load_completed_events=False
    )
    scenario = preprocess_scenario_for_execution_without_oracle(scenario)
    # Create empty env
    env = Environment(
        EnvironmentConfig(
            oracle_mode=True,
            queue_based_loop=True,
            start_time=scenario.start_time,
        )
    )
    # Convert the scenario back to a string
    sceanrio_str = JsonScenarioExporter().export_to_json(
        env=env,
        scenario=scenario,
        scenario_id=scenario.scenario_id,
        runner_config=None,
    )
    return sceanrio_str


def get_scenario_duration(
    scenario: BenchmarkScenarioImportedFromJson,
    max_time_scenario_duration: int,
    max_scenario_duration: int,
) -> int:
    if CapabilityTag.Time in scenario.tags:
        # Scenario from the Time capability
        return max_time_scenario_duration
    return max_scenario_duration


def preprocess_scenario_from_config(
    scenario: BenchmarkScenarioImportedFromJson,
    config: ScenarioRunnerConfig,
):
    # Preprocess the scenario
    if not config.oracle:
        # Create judge configuration using provided parameters
        # If judge_engine_config is None, skip judge creation entirely
        judge_config = None
        if config.judge_engine_config is not None:
            from are.simulation.validation.configs import (
                GraphPerEventJudgeConfig,
                create_judge_engine,
            )

            judge_engine = create_judge_engine(config.judge_engine_config)
            judge_config = GraphPerEventJudgeConfig(engine=judge_engine)

        preprocess_scenario(
            scenario=scenario,
            judge_config=judge_config,
            offline_validation=config.judge_only,
            max_scenario_duration=get_scenario_duration(
                scenario,
                config.max_time_scenario_duration,
                config.max_scenario_duration,
            ),
            tool_augmentation_config=config.tool_augmentation_config,
            env_events_config=config.env_events_config,
        )
    else:
        scenario.initialize()
        scenario.patch_oracle_user_message_order()


def load_and_preprocess_scenario_str(
    config: ScenarioRunnerConfig,
    sceanrio_str: str,
) -> tuple[BenchmarkScenarioImportedFromJson, list[CompletedEvent] | None]:
    from are.simulation.data_handler.importer import JsonScenarioImporter

    # Load the scenario
    (
        scenario,
        completed_events,
        _,
    ) = JsonScenarioImporter().import_from_json_to_benchmark(
        sceanrio_str, load_completed_events=True
    )
    if len(completed_events) == 0:
        completed_events = None

    # Preprocess the scenario
    preprocess_scenario_from_config(
        scenario=scenario,
        config=config,
    )
    return scenario, completed_events
