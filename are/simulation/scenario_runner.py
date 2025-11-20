# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


import logging
import random
import time

from are.simulation.agents.agent_builder import (
    AbstractAgentBuilder,
    AgentBuilder,
    AppAgentBuilder,
)
from are.simulation.agents.agent_config_builder import (
    AbstractAgentConfigBuilder,
    AgentConfigBuilder,
    AppAgentConfigBuilder,
)
from are.simulation.agents.are_simulation_agent import RunnableARESimulationAgent
from are.simulation.agents.are_simulation_agent_config import (
    LLMEngineConfig,
    MainAgentConfig,
    RunnableARESimulationAgentConfig,
)
from are.simulation.data_handler.exporter import JsonScenarioExporter
from are.simulation.environment import Environment, EnvironmentConfig
from are.simulation.notification_system import VerboseNotificationSystem
from are.simulation.scenarios import Scenario
from are.simulation.scenarios.config import ScenarioRunnerConfig
from are.simulation.scenarios.scenario import ScenarioStatus, ScenarioValidationResult
from are.simulation.scenarios.scenario_imported_from_json.utils import (
    load_and_preprocess_scenario_str,
)
from are.simulation.types import (
    CompletedEvent,
    EnvironmentType,
    EventLog,
    SimulatedGenerationTimeConfig,
)

logger = logging.getLogger(__name__)


def _apply_a2a_config(
    config: ScenarioRunnerConfig,
    scenario: Scenario,
    env: Environment,
    app_agent_config_builder: AppAgentConfigBuilder = AppAgentConfigBuilder(),
    app_agent_builder: AppAgentBuilder = AppAgentBuilder(),
) -> Scenario:
    if config.a2a_app_prop <= 0:
        return scenario

    rng = random.Random(scenario.seed)

    assert scenario.apps is not None, "Scenario apps must be set"

    # We never switch "InternalContacts" or "AgentUserInterface" or "SystemApp" because it would not make sense
    filtered_apps = [
        app
        for app in scenario.apps
        if app.name not in ["InternalContacts", "AgentUserInterface", "SystemApp"]
    ]
    if len(filtered_apps) == 0:
        raise ValueError("No valid apps available in the scenario after filtering.")

    # Calculate how many apps to transform for this agent config
    num_apps_to_transform = int(len(filtered_apps) * config.a2a_app_prop)

    if num_apps_to_transform > 0:
        # Sample apps for this model config
        apps_to_transform = rng.sample(filtered_apps, num_apps_to_transform)

        # Create the base App agent config
        agent_config = app_agent_config_builder.build(
            agent_name=config.a2a_app_agent,
        )
        model_name = config.a2a_model if config.a2a_model is not None else config.model
        model_provider = (
            config.a2a_model_provider
            if config.a2a_model_provider is not None
            else config.model_provider
        )
        endpoint = (
            config.a2a_endpoint if config.a2a_endpoint is not None else config.endpoint
        )
        agent_config.llm_engine_config = LLMEngineConfig(
            model_name=model_name,
            provider=model_provider,
            endpoint=endpoint,
        )

        new_apps = [app for app in scenario.apps if app not in apps_to_transform]

        # Set the app to agent mode with the specific model config
        for app in apps_to_transform:
            env.register_apps([app])
            logger.warning(
                f"Scenario {scenario.scenario_id} - setting app {', '.join(app.name for app in apps_to_transform)} to Agent to Agent mode with agent {config.a2a_app_agent}, model {model_name}, provider {model_provider}, and endpoint {endpoint}"
            )
            logger.warning(
                f"App {app.name} has env callback {app.add_event_callbacks.keys()}"
            )
            new_apps.append(
                app_agent_builder.build(
                    agent_config=agent_config,
                    env=env,
                    app=app,
                )
            )

        scenario.reset_apps(new_apps)

        scenario.has_a2a_augmentation = True
    return scenario


class ScenarioRunner:
    def __init__(
        self,
        agent_config_builder: AbstractAgentConfigBuilder | None = None,
        agent_builder: AbstractAgentBuilder | None = None,
        app_agent_config_builder: AppAgentConfigBuilder | None = None,
        app_agent_builder: AppAgentBuilder | None = None,
    ):
        self.agent_config_builder = agent_config_builder or AgentConfigBuilder()
        self.agent_builder = agent_builder or AgentBuilder()
        self.app_agent_config_builder = (
            app_agent_config_builder or AppAgentConfigBuilder()
        )
        self.app_agent_builder = app_agent_builder or AppAgentBuilder()

    # Custom logging functions removed as they're no longer needed
    # The ScenarioAwareFormatter in logging_config.py now handles adding the scenario ID prefix

    def _judge(
        self, scenario: Scenario, completed_events: list[CompletedEvent]
    ) -> ScenarioValidationResult:
        logger.info("Judging scenario")
        environment = Environment()
        environment.event_log = EventLog.from_list_view(completed_events)
        return scenario.validate(environment)

    def _export_trace(
        self,
        env: Environment,
        scenario: Scenario,
        model_id: str,
        agent_id: str | None,
        validation_result: ScenarioValidationResult,
        run_duration: float,
        runner_config: ScenarioRunnerConfig,
        output_dir: str | None = None,
        export_apps: bool = True,
        trace_dump_format: str = "hf",
    ) -> str | None:
        """
        Exports the given environment, scenario, and model data to a JSON file.
        :param env: Environment
        :param scenario: Scenario
        :param model_id: Model identifier
        :param agent_id: Agent identifier
        :param validation_result: Validation result
        :param run_duration: Duration of the run
        :param output_dir: Output directory
        :param export_apps: Whether to export the apps or not
        :param trace_dump_format: Format for trace dump
        :param config: ScenarioRunnerConfig for filename generation and trace storage
        """

        validation_decision = (
            ScenarioStatus.Valid.value
            if validation_result.success
            else ScenarioStatus.Invalid.value
        )
        validation_rationale = validation_result.rationale

        scenario_exporter = JsonScenarioExporter()
        success, export_path = scenario_exporter.export_to_json_file(
            env,
            scenario,
            model_id,
            agent_id,
            validation_decision,
            validation_rationale,
            run_duration=run_duration,
            output_dir=output_dir,
            export_apps=export_apps,
            trace_dump_format=trace_dump_format,
            scenario_exception=validation_result.exception,
            runner_config=runner_config,
        )

        if success:
            logger.info(f"Trace exported to {export_path}")
        else:
            logger.error("Failed to export trace")

        return export_path

    def _run_without_agent(
        self, scenario_id: str, scenario: Scenario, env: Environment
    ) -> ScenarioValidationResult:
        logger.info("Running without Agent")
        env.join()
        logger.info("Validating...")
        validation_result = scenario.validate(env)
        logger.info(f"Validation {validation_result} EnvState={env.state}")
        return validation_result

    def _run_with_agent(
        self,
        scenario_id: str,
        scenario: Scenario,
        env: Environment,
        agent: str,
        model: str,
        provider: str | None = None,
        endpoint: str | None = None,
        max_turns: int | None = None,
        simulated_generation_time_mode: str = "measured",
        use_custom_logger: bool = True,
    ) -> ScenarioValidationResult:
        agent_config: RunnableARESimulationAgentConfig = (
            self.agent_config_builder.build(agent_name=agent)
        )

        # Set the use_custom_logger parameter in the base agent config
        agent_config.get_base_agent_config().use_custom_logger = use_custom_logger
        agent_config.get_base_agent_config().llm_engine_config = LLMEngineConfig(
            model_name=model, provider=provider, endpoint=endpoint
        )

        # Create SimulatedGenerationTimeConfig from the mode
        simulated_generation_time_config = SimulatedGenerationTimeConfig(
            mode=simulated_generation_time_mode  # type: ignore
        )
        agent_config.get_base_agent_config().simulated_generation_time_config = (
            simulated_generation_time_config
        )

        if isinstance(agent_config, MainAgentConfig) and max_turns is not None:
            agent_config.max_turns = max_turns

        are_simulation_agent: RunnableARESimulationAgent = self.agent_builder.build(
            agent_config=agent_config, env=env
        )
        logger.info(f"Running with Agent {agent}")

        result = are_simulation_agent.run_scenario(
            scenario=scenario, notification_system=env.notification_system
        )
        output = result.output
        logger.info(f"Agent Output {output}")
        logger.info("Validating...")
        validation_result = scenario.validate(env)
        logger.info(f"Validation {validation_result} EnvState={env.state}")
        return validation_result

    def _run(
        self,
        config: ScenarioRunnerConfig,
        scenario: Scenario,
    ) -> ScenarioValidationResult:
        env_config = EnvironmentConfig(
            oracle_mode=config.oracle,
            queue_based_loop=config.oracle,
            wait_for_user_input_timeout=config.wait_for_user_input_timeout,
            dump_dir=config.output_dir if config.oracle else None,
            time_increment_in_seconds=scenario.time_increment_in_seconds,
            exit_when_no_events=config.agent
            is None,  # Only exit when no events if running without an agent
        )
        if scenario.start_time and scenario.start_time > 0:
            env_config.start_time = scenario.start_time
        env = Environment(
            environment_type=EnvironmentType.CLI,
            config=env_config,
            notification_system=VerboseNotificationSystem(),
        )
        scenario = _apply_a2a_config(
            config, scenario, env, self.app_agent_config_builder, self.app_agent_builder
        )
        env.run(scenario, wait_for_end=False)

        # Run the agent
        try:
            if config.agent is None:
                validation_result = self._run_without_agent(
                    scenario.scenario_id, scenario, env
                )
            else:
                validation_result = self._run_with_agent(
                    scenario.scenario_id,
                    scenario,
                    env,
                    config.agent,
                    config.model,
                    config.model_provider,
                    config.endpoint,
                    config.max_turns,
                    config.simulated_generation_time_mode,
                    config.use_custom_logger,
                )
        except Exception as exception:
            logger.exception(f"Failed to run agent: {exception}")
            validation_result = ScenarioValidationResult(
                success=None, exception=exception
            )

        run_duration = env.time_manager.time_passed()

        if config.export:
            # Check if scenario has HuggingFace metadata to determine whether to export apps
            has_hf_metadata = getattr(scenario, "hf_metadata", None) is not None
            export_path = self._export_trace(
                env,
                scenario,
                config.model,
                config.agent,
                validation_result,
                run_duration,
                config,
                output_dir=config.output_dir,
                export_apps=not has_hf_metadata,  # Don't export apps if scenario has HF metadata
                trace_dump_format=config.trace_dump_format,
            )
            validation_result.export_path = export_path
        env.stop()
        return validation_result

    def run(
        self,
        config: ScenarioRunnerConfig,
        scenario: Scenario | str,
        completed_events: list[CompletedEvent] | None = None,
    ) -> ScenarioValidationResult:
        # Set the scenario ID and run number for the current thread if not already set
        # This ensures that all logs from this thread will include the scenario ID and run number
        from are.simulation.logging_config import (
            get_logger_scenario_id,
            set_logger_scenario_id,
        )

        start_time = time.time()
        if isinstance(scenario, str):
            # Load the scenario
            scenario, completed_events = load_and_preprocess_scenario_str(
                config, scenario
            )

        run_number = getattr(scenario, "run_number", None)
        if get_logger_scenario_id() != scenario.scenario_id:
            set_logger_scenario_id(scenario.scenario_id, run_number)

        try:
            if config.judge_only:
                assert completed_events is not None
                result = self._judge(scenario, completed_events)
            else:
                result = self._run(config, scenario)
        except Exception as exception:
            logger.exception(f"Failed to run scenario: {exception}")
            result = ScenarioValidationResult(success=None, exception=exception)
        logger.info(
            f"{'✅' if result.success is True else '❌' if result.success is False else '⚠️'} Result: {result}"
        )
        # Convert exception into failure
        if result.success is None and result.exception is not None:
            result.success = False
        # Add run duration to result
        result.duration = time.time() - start_time
        return result
