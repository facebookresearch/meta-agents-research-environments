# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


from abc import ABC, abstractmethod

from are.simulation.agents.are_simulation_agent import RunnableARESimulationAgent
from are.simulation.agents.are_simulation_agent_config import (
    ARESimulationReactAgentConfig,
    ARESimulationReactAppAgentConfig,
    RunnableARESimulationAgentConfig,
)
from are.simulation.agents.default_agent.app_agent import AppAgent
from are.simulation.agents.llm.llm_engine_builder import LLMEngineBuilder
from are.simulation.apps import App
from are.simulation.environment import Environment


class AbstractAgentBuilder(ABC):
    """
    Abstract class for building agents.
    """

    @abstractmethod
    def list_agents(self) -> list[str]:
        """
        Method to list all available agents.
        :returns: A list of agent names.
        """

    @abstractmethod
    def build(
        self,
        agent_config: RunnableARESimulationAgentConfig,
        env: Environment | None = None,
        mock_responses: list[str] | None = None,
    ) -> RunnableARESimulationAgent:
        """
        Method to build an agent.
        :param agent_config: Configuration for the agent to be built.
        :param env: Optional environment in which the agent will operate.
        :param mock_responses: Optional list of mock responses to be used for the agent.
        :returns: An instance of the agent.
        """


class AgentBuilder(AbstractAgentBuilder):
    def __init__(self, llm_engine_builder: LLMEngineBuilder | None = None):
        self.llm_engine_builder = llm_engine_builder or LLMEngineBuilder()

    def list_agents(self) -> list[str]:
        return ["default", "browser"]

    def build(
        self,
        agent_config: RunnableARESimulationAgentConfig,
        env: Environment | None = None,
        mock_responses: list[str] | None = None,
    ) -> RunnableARESimulationAgent:
        match agent_config.get_agent_name():
            case "default":
                from are.simulation.agents.default_agent.agent_factory import (
                    are_simulation_react_json_agent,
                )
                from are.simulation.agents.default_agent.are_simulation_main import (
                    ARESimulationAgent,
                )

                assert env is not None, "Environment must be provided"
                assert env.time_manager is not None, "Time manager must be provided"
                assert env.append_to_world_logs is not None, (
                    "Log callback must be provided"
                )

                llm_engine = self.llm_engine_builder.create_engine(
                    engine_config=agent_config.get_base_agent_config().llm_engine_config,
                    mock_responses=mock_responses,
                )

                if isinstance(agent_config, ARESimulationReactAgentConfig):
                    return ARESimulationAgent(
                        log_callback=env.append_to_world_logs,
                        llm_engine=llm_engine,
                        base_agent=are_simulation_react_json_agent(
                            llm_engine=llm_engine,
                            base_agent_config=agent_config.base_agent_config,
                        ),
                        time_manager=env.time_manager,
                        max_turns=agent_config.max_turns,
                        pause_env=env.pause,
                        resume_env=env.resume_with_offset,
                        simulated_generation_time_config=(
                            agent_config.get_base_agent_config().simulated_generation_time_config
                        ),
                    )
                else:
                    raise ValueError(
                        f"Agent {agent_config.get_agent_name()} requires a ARESimulationReactBaseAgentConfig"
                    )

            case "browser":
                from are.simulation.agents.default_agent.agent_factory import (
                    are_simulation_browser_react_agent,
                )
                from are.simulation.agents.default_agent.are_simulation_main import (
                    ARESimulationAgent,
                )

                assert env is not None, "Environment must be provided"
                assert env.time_manager is not None, "Time manager must be provided"
                assert env.append_to_world_logs is not None, (
                    "Log callback must be provided"
                )

                llm_engine = self.llm_engine_builder.create_engine(
                    engine_config=agent_config.get_base_agent_config().llm_engine_config,
                    mock_responses=mock_responses,
                )

                if isinstance(agent_config, ARESimulationReactAgentConfig):
                    # Browser will be injected later from scenario during run_scenario()
                    # It's passed as None here and set in prepare_are_simulation_run()
                    return ARESimulationAgent(
                        log_callback=env.append_to_world_logs,
                        llm_engine=llm_engine,
                        base_agent=are_simulation_browser_react_agent(
                            llm_engine=llm_engine,
                            base_agent_config=agent_config.base_agent_config,
                            browser=None,  # Will be set from scenario later
                        ),
                        time_manager=env.time_manager,
                        max_turns=agent_config.max_turns,
                        pause_env=env.pause,
                        resume_env=env.resume_with_offset,
                        simulated_generation_time_config=(
                            agent_config.get_base_agent_config().simulated_generation_time_config
                        ),
                    )
                else:
                    raise ValueError(
                        f"Agent {agent_config.get_agent_name()} requires a ARESimulationReactBaseAgentConfig"
                    )

            case _:
                raise ValueError(f"Agent {agent_config.get_agent_name()} not found")


class AbstractAppAgentBuilder(ABC):
    """
    Abstract class for building agents.
    """

    @abstractmethod
    def list_agents(self) -> list[str]:
        """
        Method to list all available agents.
        :returns: A list of agent names.
        """

    @abstractmethod
    def build(
        self,
        agent_config: ARESimulationReactAppAgentConfig,
        app: App,
        env: Environment | None = None,
        mock_responses: list[str] | None = None,
    ) -> AppAgent:
        """
        Method to build an agent.
        :param agent_config: Configuration for the agent to be built.
        :param app: The app to be used by the agent.
        :param mock_responses: Optional list of mock responses to be used for the agent.
        :returns: An instance of the agent.
        """


class AppAgentBuilder(AbstractAppAgentBuilder):
    def __init__(self, llm_engine_builder: LLMEngineBuilder | None = None):
        self.llm_engine_builder = llm_engine_builder or LLMEngineBuilder()

    def list_agents(self) -> list[str]:
        return [
            "default_app_agent",
        ]

    def build(
        self,
        agent_config: ARESimulationReactAppAgentConfig,
        app: App,
        env: Environment | None = None,
        mock_responses: list[str] | None = None,
    ) -> AppAgent:
        match agent_config.get_agent_name():
            case "default_app_agent":
                from are.simulation.agents.default_agent.agent_factory import (
                    are_simulation_react_json_app_agent,
                )
                from are.simulation.agents.default_agent.default_tools import (
                    FinalAnswerTool,
                )
                from are.simulation.tool_utils import AppToolAdapter

                assert env is not None, "Environment must be provided"
                assert env.append_to_world_logs is not None, (
                    "Log callback must be provided"
                )

                llm_engine = self.llm_engine_builder.create_engine(
                    engine_config=agent_config.llm_engine_config,
                    mock_responses=mock_responses,
                )
                app_agent = AppAgent(
                    app_agent=are_simulation_react_json_app_agent(
                        llm_engine, agent_config, env.append_to_world_logs
                    ),
                    tools={
                        tool.name: tool
                        for tool in [AppToolAdapter(tool) for tool in app.get_tools()]
                        + [FinalAnswerTool()]
                    },
                    name=app.name,
                )
                return app_agent
            case _:
                raise ValueError(f"App agent {agent_config.get_agent_name()} not found")
