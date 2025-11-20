# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


import json
import logging
import os
import tempfile
from collections import defaultdict
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Iterable

from are.simulation.agents.agent_log import LLMOutputThoughtActionLog
from are.simulation.agents.are_simulation_agent import BaseAgentLog
from are.simulation.agents.default_agent.base_agent import (
    DEFAULT_STEP_2_MESSAGE,
    DEFAULT_STEP_2_ROLE,
    format_message,
)
from are.simulation.data_handler.models import (
    TRACE_V1_VERSION,
    ExportedAction,
    ExportedActionArg,
    ExportedApp,
    ExportedCompletedEvent,
    ExportedEvent,
    ExportedEventMetadata,
    ExportedExecutionMetadata,
    ExportedHint,
    ExportedHuggingFaceMetadata,
    ExportedOracleEvent,
    ExportedTrace,
    ExportedTraceAnnotationMetadata,
    ExportedTraceBase,
    ExportedTraceDefinitionMetadata,
    ExportedTraceMetadata,
    ExportedTraceSimulationMetadata,
)
from are.simulation.environment import Environment
from are.simulation.scenarios import Scenario
from are.simulation.scenarios.config import ScenarioRunnerConfig
from are.simulation.scenarios.scenario_imported_from_json.scenario import (
    ScenarioImportedFromJson,
)
from are.simulation.scenarios.utils.caching import get_run_id
from are.simulation.types import (
    AbstractEvent,
    Action,
    ActionDescription,
    CompletedEvent,
    ConditionCheckAction,
    ConditionCheckEvent,
    Event,
    EventMetadata,
    EventType,
    OracleEvent,
)

logger = logging.getLogger(__name__)


def build_history_from_logs(
    world_logs,
    exclude_log_types=["tool_call", "rationale", "action"],
    message_dict=DEFAULT_STEP_2_MESSAGE,
    role_dict=DEFAULT_STEP_2_ROLE,
):
    history = []
    step_order = list(role_dict.keys())
    i = 0
    for log in world_logs:
        role = log.get_type()
        content_for_llm = log.get_content_for_llm()
        step_messages = defaultdict(list)
        if (
            role not in role_dict
            or role in exclude_log_types
            or content_for_llm is None
            or content_for_llm == ""
        ):
            continue
        if role in ["observation", "error"]:
            i += 1
        content = format_message(message_dict, role, content_for_llm, i)
        attachments = log.get_attachments_for_llm()
        step_messages[role].append(
            {
                "role": role_dict[role],
                "content": content,
                "attachments": attachments,
                "timestamp": log.timestamp,
            }
        )
        for step in step_order:
            if step in step_messages:
                history.extend(step_messages[step])
    return history


def extract_llm_usage_stats_from_logs(
    world_logs: list[BaseAgentLog],
):
    total_llm_calls = 0
    prompt_tokens = []
    completion_tokens = []
    total_tokens = []
    reasoning_tokens = []
    duration_per_call = []
    for log in world_logs:
        if isinstance(log, LLMOutputThoughtActionLog):
            total_llm_calls += 1
            prompt_tokens.append(log.prompt_tokens)
            completion_tokens.append(log.completion_tokens)
            total_tokens.append(log.total_tokens)
            reasoning_tokens.append(log.reasoning_tokens)
            duration_per_call.append(log.completion_duration)

    return {
        "total_llm_calls": total_llm_calls,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "reasoning_tokens": reasoning_tokens,
        "completion_duration": duration_per_call,
    }


class JsonScenarioExporter:
    """
    JSON Scenario Exporter.
    """

    @staticmethod
    def convert_app(
        app_name: str, app_class: str, app_state: dict[str, Any]
    ) -> ExportedApp:
        return ExportedApp(
            name=app_name,
            class_name=app_class,
            app_state=JsonScenarioExporter.process_state(app_state),
        )

    @staticmethod
    def process_state(state: dict[str, Any] | None) -> dict[str, Any] | None:
        if state is None:
            return None

        def process_item(item):
            if isinstance(item, Enum):
                return item.value
            elif isinstance(item, dict):
                return {k: process_item(v) for k, v in item.items()}
            elif isinstance(item, list):
                return [process_item(v) for v in item]  # Process each item in the list
            elif isinstance(item, set):
                return [process_item(v) for v in item]  # Convert set to list
            else:
                return item

        return {key: process_item(value) for key, value in state.items()}

    @staticmethod
    def convert_event(event: AbstractEvent) -> ExportedEvent | ExportedOracleEvent:
        event_time = event.event_time if event.event_relative_time is None else None

        if (
            isinstance(event, Event) or isinstance(event, ConditionCheckEvent)
        ) and event.action is not None:
            action = JsonScenarioExporter.convert_action(event.action)
        elif isinstance(event, OracleEvent) and event.action_desc is not None:
            action = JsonScenarioExporter.convert_oracle_action(event.action_desc)
        else:
            action = None

        if isinstance(event, OracleEvent):
            return ExportedOracleEvent(
                class_name=event.__class__.__name__,
                event_type=event.event_type.name,
                event_time=event_time,
                event_time_comparator=(
                    event.event_time_comparator.value
                    if event.event_time_comparator
                    else None
                ),
                event_id=event.event_id,
                event_relative_time=event.event_relative_time,
                dependencies=[dependency.event_id for dependency in event.dependencies],
                action=action,
            )

        else:
            return ExportedEvent(
                class_name=event.__class__.__name__,
                event_type=event.event_type.name,
                event_time=event_time,
                event_id=event.event_id,
                event_relative_time=event.event_relative_time,
                dependencies=[dependency.event_id for dependency in event.dependencies],
                action=action,
            )

    @staticmethod
    def convert_completed_event(event: CompletedEvent) -> ExportedCompletedEvent:
        action = JsonScenarioExporter.convert_action(event.action)

        return ExportedCompletedEvent(
            class_name=event.__class__.__name__,
            event_type=event.event_type.name,
            event_time=event.event_time if event.event_time else 0,
            event_id=event.event_id,
            event_relative_time=event.event_relative_time,
            dependencies=[dependency.event_id for dependency in event.dependencies],
            action=action,
            metadata=JsonScenarioExporter.convert_event_metadata(event.metadata),
        )

    @staticmethod
    def convert_event_metadata(metadata: EventMetadata) -> ExportedEventMetadata:
        return ExportedEventMetadata(
            return_value=(
                str(metadata.return_value) if metadata.return_value else None
            ),
            return_value_type=(
                type(metadata.return_value).__name__ if metadata.return_value else None
            ),
            exception=metadata.exception,
            exception_stack_trace=metadata.exception_stack_trace,
        )

    @staticmethod
    def convert_action(action: Action | ConditionCheckAction) -> ExportedAction:
        if isinstance(action, Action):
            args = action.resolved_args if action.resolved_args else action.args
            return ExportedAction(
                action_id=action.action_id,
                app=action.app.name if action.app else action.app.__class__.__name__,
                function=action.function.__name__,
                operation_type=(
                    action.operation_type.name if action.operation_type else None
                ),
                args=[
                    JsonScenarioExporter.convert_action_args(key, value)
                    for key, value in args.items()
                    if key != "self"
                ],
            )
        elif isinstance(action, ConditionCheckAction):
            return ExportedAction(
                action_id=action.action_id,
                function=action.function.__name__,
            )
        else:
            raise ValueError("Invalid action type")

    @staticmethod
    def convert_action_args(name: str, value: Any) -> ExportedActionArg:
        value_str = (
            json.dumps(value)
            if isinstance(value, (list, dict))
            else str(value)
            if value is not None
            else None
        )
        value_type = type(value).__name__ if value is not None else None

        return ExportedActionArg(
            name=name,
            value=value_str,
            value_type=value_type,
        )

    @staticmethod
    def convert_oracle_action(action_desc: ActionDescription) -> ExportedAction:
        return ExportedAction(
            action_id="",
            app=action_desc.app,
            function=action_desc.function,
            operation_type=None,
            args=[
                ExportedActionArg(
                    name=arg["name"], value=arg["value"], value_type=arg["value_type"]
                )
                for arg in action_desc.args
            ],
        )

    def export_to_json_file(
        self,
        env: Environment,
        scenario: Scenario,
        model_id: str | None = None,
        agent_id: str | None = None,
        validation_decision: str | None = None,
        validation_rationale: str | None = None,
        run_duration: float | None = None,
        output_dir: str | None = None,
        export_apps: bool = True,
        trace_dump_format: str = "hf",
        scenario_exception: Exception | None = None,
        runner_config: ScenarioRunnerConfig | None = None,
    ) -> tuple[bool, str | None]:
        """
        Export trace data from the environment to a JSON file.
        :param env: Environment
        :param scenario: Scenario
        :param model_id: Model ID
        :param agent_id: Agent ID
        :param validation_decision: Validation decision
        :param validation_rationale: Validation rationale
        :param run_duration: End-to-end run duration
        :param output_dir: Output directory
        :param export_apps: Whether to export apps in the trace
        :param config: ScenarioRunnerConfig or MultiScenarioRunnerConfig for filename generation and trace storage
        :return: Tuple containing success status and file path if successful
        """
        try:
            if trace_dump_format == "hf":
                json_str = self.export_to_json(
                    env,
                    scenario,
                    scenario.scenario_id,
                    runner_config,
                    model_id,
                    agent_id,
                    validation_decision,
                    export_apps=export_apps,
                    scenario_exception=scenario_exception,
                    world_logs=env.world_logs,
                )
            elif trace_dump_format == "lite":
                logging.warning(
                    "Exporting trace as in **lite format**, outputs will not be "
                    "uploadable to HuggingFace."
                )
                json_str = self.export_to_json_lite(
                    env,
                    scenario,
                    scenario.scenario_id,
                    model_id,
                    agent_id,
                    validation_decision,
                    validation_rationale,
                    run_duration=run_duration,
                )
            elif trace_dump_format == "both":
                # Generate both formats
                hf_json_str = self.export_to_json(
                    env,
                    scenario,
                    scenario.scenario_id,
                    runner_config,
                    model_id,
                    agent_id,
                    validation_decision,
                    export_apps=export_apps,
                    scenario_exception=scenario_exception,
                    world_logs=env.world_logs,
                )
                lite_json_str = self.export_to_json_lite(
                    env,
                    scenario,
                    scenario.scenario_id,
                    model_id,
                    agent_id,
                    validation_decision,
                    validation_rationale,
                    run_duration,
                )

                # Save both formats with subdirectories
                if output_dir is None:
                    output_dir = tempfile.gettempdir()
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)

                # Create subdirectories and file paths
                hf_dir = os.path.join(output_dir, "hf")
                lite_dir = os.path.join(output_dir, "lite")
                os.makedirs(hf_dir, exist_ok=True)
                os.makedirs(lite_dir, exist_ok=True)

                base_filename = f"{get_run_id(scenario, runner_config)}.json"
                hf_file_path = os.path.join(hf_dir, base_filename)
                lite_file_path = os.path.join(lite_dir, base_filename)

                # Write both files
                with open(hf_file_path, "w", encoding="utf-8") as f:
                    f.write(hf_json_str)
                    f.flush()
                with open(lite_file_path, "w", encoding="utf-8") as f:
                    f.write(lite_json_str)
                    f.flush()

                return True, hf_file_path  # Return HF path for backward compatibility
            else:
                raise ValueError(
                    f"{trace_dump_format} is an invalid dump format, must be 'hf', 'lite', or 'both'"
                )
        except Exception as e:
            logger.exception(f"Failed to export trace: {e}")
            return False, None

        # Original single format handling
        if output_dir is None:
            output_dir = tempfile.gettempdir()
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Get path
        base_filename = f"{get_run_id(scenario, runner_config)}.json"
        file_path = os.path.join(output_dir, base_filename)
        try:
            with open(file_path, "w", encoding="utf-8") as file:
                file.write(json_str)
                file.flush()
            return True, file_path
        except Exception as e:
            logger.error(f"Failed to write to file: {e}")
            return False, None

    def export_to_json(
        self,
        env: Environment,
        scenario: Scenario,
        scenario_id: str,
        runner_config: ScenarioRunnerConfig | None = None,
        model_id: str | None = None,
        agent_id: str | None = None,
        validation_decision: str | None = None,
        annotation_id: str | None = None,
        annotator_name: str | None = None,
        context: str | None = None,
        comment: str | None = None,
        indent: int | None = None,
        apps_state: dict[str, Any] | None = None,
        world_logs: list[BaseAgentLog] | None = None,
        export_apps: bool = True,
        scenario_exception: Exception | None = None,
        **kwargs: Any,
    ) -> str:
        """
        Export trace data from the environment to a JSON string.

        :param env: Environment
        :param scenario: Scenario
        :param scenario_id: Scenario ID
        :param model_id: Model ID
        :param agent_id: Agent ID
        :param validation_decision: Validation decision
        :param annotation_id: Annotation ID
        :param annotator_name: Annotator name
        :param context: Context for bug report
        :param comment: Annotation comment
        :param indent: Indentation level for the JSON output
        :param apps_state: State of apps to persist in trace. If set to `None`,
            the initial scenario app state is used. If this value is set, we assume
            the exported trace should be a Red Team replay trace.
        :param world_logs: List of world logs
        :param export_apps: Whether to export apps in the trace
        :param kwargs: Additional parameters for internal use
        :return: JSON string representation of the trace data
        """

        trace_data: ExportedTraceBase = self._get_trace(
            env,
            scenario,
            scenario_id,
            model_id,
            agent_id,
            validation_decision,
            annotation_id,
            annotator_name,
            context,
            comment,
            apps_state,
            world_logs,
            export_apps,
            scenario_exception,
            runner_config=runner_config,
            **kwargs,
        )
        return trace_data.model_dump_json(indent=indent)

    def export_to_json_lite(
        self,
        env: Environment,
        scenario: Scenario,
        scenario_id: str,
        model_id: str | None = None,
        agent_id: str | None = None,
        validation_decision: str | None = None,
        validation_rationale: str | None = None,
        run_duration: float | None = None,
    ):
        # Build agent histories from world logs
        agent_histories, llm_usage_stats = self._extract_data_from_world_logs(env)

        # Dump to json
        return json.dumps(
            {
                "scenario_id": scenario_id,
                "model_id": model_id,
                "main_agent_type": agent_id,
                "validation_decision": validation_decision,
                "validation_rationale": validation_rationale,
                "run_duration": run_duration,
                "per_agent_interaction_histories": agent_histories,
                "per_agent_llm_usage_stats": llm_usage_stats,
            }
        )

    def _extract_data_from_world_logs(
        self,
        env: Environment,
    ):
        # Unpack world logs by agent ID --> does not assume that execution is always synchronous
        world_logs_by_agent = {}

        for world_log in env.world_logs:
            agent_id = world_log.agent_id
            if agent_id == "unknown":
                continue
            if agent_id not in world_logs_by_agent:
                world_logs_by_agent[agent_id] = [world_log]
            else:
                world_logs_by_agent[agent_id].append(world_log)

        # Convert each agent's world logs into a history
        agent_histories = {
            f"agent_{aid}": build_history_from_logs(world_logs)
            for aid, world_logs in world_logs_by_agent.items()
        }

        # Extract per agent LLM usage stats
        llm_usage_stats = {
            f"agent_{aid}": extract_llm_usage_stats_from_logs(world_logs)
            for aid, world_logs in world_logs_by_agent.items()
        }

        return agent_histories, llm_usage_stats

    def _get_trace(
        self,
        env: Environment,
        scenario: Scenario,
        scenario_id: str,
        model_id: str | None,
        agent_id: str | None,
        validation_decision: str | None,
        annotation_id: str | None,
        annotator_name: str | None,
        context: str | None,
        comment: str | None,
        apps_state: dict[str, Any] | None,
        world_logs: list[BaseAgentLog] | None = None,
        export_apps: bool = True,
        scenario_exception: Exception | None = None,
        runner_config: ScenarioRunnerConfig | None = None,
        **kwargs: Any,
    ) -> ExportedTraceBase:
        # Check if scenario has HuggingFace metadata
        hf_metadata = None
        # Use getattr to safely access hf_metadata attribute
        scenario_hf_metadata = getattr(scenario, "hf_metadata", None)
        if scenario_hf_metadata is not None:
            hf_metadata = ExportedHuggingFaceMetadata(
                dataset=scenario_hf_metadata.dataset,
                split=scenario_hf_metadata.split,
                revision=scenario_hf_metadata.revision,
            )

        execution_metadata = None
        if isinstance(scenario, ScenarioImportedFromJson):
            execution_metadata = scenario.execution_metadata

        definition_metadata = ExportedTraceDefinitionMetadata(
            scenario_id=scenario_id,
            seed=scenario.seed,
            duration=scenario.duration,
            time_increment_in_seconds=scenario.time_increment_in_seconds,
            start_time=scenario.start_time,
            run_number=getattr(scenario, "run_number", None),
            hints=[
                ExportedHint(
                    hint_type=hint.hint_type.value,
                    content=hint.content,
                    associated_event_id=hint.associated_event_id,
                )
                for hint in scenario.hints or []
            ],
            config=getattr(scenario, "config", None),
            has_a2a_augmentation=getattr(scenario, "has_a2a_augmentation", False),
            has_tool_augmentation=getattr(scenario, "tool_augmentation_config", None)
            is not None,
            has_env_events_augmentation=getattr(scenario, "env_events_config", None)
            is not None,
            has_exception=scenario_exception is not None,
            exception_type=(
                type(scenario_exception).__name__ if scenario_exception else None
            ),
            exception_message=(str(scenario_exception) if scenario_exception else None),
            tags=[tag.value for tag in scenario.tags],
            hf_metadata=hf_metadata,
        )

        simulation_metadata = ExportedTraceSimulationMetadata(
            agent_id=agent_id,
            model_id=model_id if agent_id else None,
        )
        annotation_metadata = ExportedTraceAnnotationMetadata(
            annotation_id=annotation_id,
            annotator=annotator_name,
            validation_decision=validation_decision,
            comment=comment,
            date=datetime.now(timezone.utc).timestamp(),
        )
        execution_metadata = (
            ExportedExecutionMetadata.from_metadata(execution_metadata)
            if execution_metadata
            else None
        )

        events: Iterable[AbstractEvent]
        events = scenario.events
        metadata = ExportedTraceMetadata(
            definition=definition_metadata,
            simulation=simulation_metadata,
            annotation=annotation_metadata,
            execution=execution_metadata,
            runner_config=runner_config,
        )

        apps_state = scenario._initial_apps or {}

        # Determine whether to include apps in the export
        apps_to_export = []
        if not export_apps and not hf_metadata:
            # Warn if export_apps=False but there's no HuggingFace metadata
            logger.warning(
                "export_apps=False but no HuggingFace metadata found. "
                "Apps will still be included in the export as there's no way to recover them later."
            )
            export_apps = True

        if export_apps:
            apps_to_export = [
                self.convert_app(
                    app_name,
                    app_data["class_name"],
                    json.loads(app_data["serialized_state"]),
                )
                for app_name, app_data in (apps_state or {}).items()
            ]

        world_logs_data = [agent_log.serialize() for agent_log in world_logs or []]
        events_data = [
            self.convert_event(event)
            for event in events
            if event.event_type
            != EventType.VALIDATION  # Validation events have Python functions that we can't serialize
        ]
        completed_events = [
            self.convert_completed_event(event)
            for event in env.event_log.list_view()
            if event.event_type
            != EventType.VALIDATION  # Validation events have Python functions that we can't serialize
        ]
        return ExportedTrace(
            metadata=metadata,
            world_logs=world_logs_data,
            events=events_data,
            completed_events=completed_events,
            apps=apps_to_export,
            version=TRACE_V1_VERSION,
            context=context,
        )
