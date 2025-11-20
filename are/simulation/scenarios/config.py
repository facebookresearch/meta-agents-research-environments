# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


import hashlib
import json

from pydantic import BaseModel

from are.simulation.agents.are_simulation_agent_config import LLMEngineConfig
from are.simulation.scenarios.utils.scenario_expander import EnvEventsConfig
from are.simulation.types import ToolAugmentationConfig
from are.simulation.utils import DEFAULT_APP_AGENT, DEFAULT_MODEL, DEFAULT_PROVIDER

MAX_SCENARIO_DURATION = 1800  # 30 minutes
MAX_TIME_SCENARIO_DURATION = 420  # 7 minutes
DEFAULT_SCENARIO_TIMEOUT = 1860  # 31 minutes


class HuggingFaceConfig(BaseModel):
    # Upload configuration
    upload_dataset_name: str | None = None
    upload_dataset_config: str | None = None
    upload_dataset_split: str | None = None
    upload_private: bool = False

    # Original dataset information
    original_dataset_name: str | None = None
    original_dataset_config: str | None = None
    original_dataset_split: str | None = None
    original_dataset_revision: str | None = None


class ScenarioRunnerConfig(BaseModel):
    # Model to use for scenario execution (default: "llama3-70b-instruct")
    model: str = DEFAULT_MODEL

    # Optional model provider to use(default: "huggingface")
    model_provider: str | None = DEFAULT_PROVIDER

    # Optional agent to use for running the scenarios(default: None)
    agent: str | None = None

    # Parameters for scenario creation (default: empty dictionary)
    scenario_creation_params: str = "{}"

    # Parameters for creating multiple of the same scenario with different state (default: [{}])
    scenario_multi_creation_params: str = "[{}]"

    # Parameters for scenario initialization (default: empty dictionary)
    scenario_initialization_params: str = "{}"

    # Parameters for initializng multiple of the same scenario with different state (default: [{}])
    scenario_multi_initialization_params: str = "[{}]"

    # Flag indicating whether to run the scenarios in Oracle mode where oracle events (i.e. user defined agent events) are ran. (default: False)
    oracle: bool = False

    # Flag indicating whether to export traces to a JSON file. (default: False)
    export: bool = False

    # Timeout for user inputs in seconds (no timeout by default).
    wait_for_user_input_timeout: float | None = None

    # Directory to output the scenario states, traces and logs.
    output_dir: str | None = None

    # URL of the endpoint to contact for running the agent's model
    endpoint: str | None = None

    # Maximum number of turns of the conversation between the user and the agent.
    max_turns: int | None = 1

    # Whether to run only the judge for scenarios.
    judge_only: bool = False

    # Toggle to enable/disable agent2agent mode. Values greater than 0 will enable agent2agent mode. (default: 0)
    a2a_app_prop: float = 0.0

    # [Agent2Agent] Agent to use for App agent instances
    a2a_app_agent: str = DEFAULT_APP_AGENT

    # [Agent2Agent] Model to use for App agent instances
    a2a_model: str | None = DEFAULT_MODEL

    # [Agent2Agent] Optional model provider to use for App agent instances (if not specified, uses the same provider as the main model)
    a2a_model_provider: str | None = None

    # [Agent2Agent] URL of the endpoint to contact for running the App agent model instances (if not specified, will try to use the same endpoint as the main model)
    a2a_endpoint: str | None = None

    # Toggles scenario JSON export format -- must be one of "hf" or "lite"
    trace_dump_format: str = "lite"

    # Whether to use the custom logger in the agent (default: True)
    use_custom_logger: bool = True

    # Simulated generation time mode
    simulated_generation_time_mode: str = "measured"

    # Tool augmentation configuration for noise injection
    tool_augmentation_config: ToolAugmentationConfig | None = None

    # Environment events configuration for noise injection
    env_events_config: EnvEventsConfig | None = None

    # Judge engine configuration
    judge_engine_config: LLMEngineConfig | None = None

    # Max scenario duration in seconds
    max_scenario_duration: int = MAX_SCENARIO_DURATION

    # Max time scenario duration in seconds
    max_time_scenario_duration: int = MAX_TIME_SCENARIO_DURATION

    def get_config_hash(self) -> str:
        """
        Generate a hash of the relevant config parameters that affect scenario execution.
        Excludes parameters that only affect parallel execution or logging.
        """

        # Exclude fields specific to multi-threading and not to scenario execution
        exclude_fields = {
            "max_concurrent_scenarios",
            "timeout_seconds",
            "log_level",
        }

        # Use pydantic's model_dump with exclude parameter, then serialize to JSON
        config_dict = self.model_dump(exclude=exclude_fields)
        config_str = json.dumps(config_dict, sort_keys=True, default=str)
        return hashlib.md5(config_str.encode()).hexdigest()[:8]


class MultiScenarioRunnerConfig(ScenarioRunnerConfig):
    # Maximum number of concurrent scenarios to run. If not specified, automatically sets based on the number of CPUs.
    max_concurrent_scenarios: int | None = None

    # Timeout for individual scenarios in seconds. If not specified, no timeout is applied.
    timeout_seconds: int | None = None

    # Type of executor to use for running scenarios, options: "sequential", "thread", "process"
    executor_type: str = "thread"

    # Logging level to use for the runner and worker threads
    log_level: str = "INFO"

    # Enable scenario result caching to skip re-running identical scenarios
    enable_caching: bool = True
