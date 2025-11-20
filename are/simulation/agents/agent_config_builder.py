# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


from abc import ABC, abstractmethod
from typing import Any

from are.simulation.agents.are_simulation_agent_config import (
    ARESimulationReactAgentConfig,
    ARESimulationReactAppAgentConfig,
    ARESimulationReactBaseAgentConfig,
    RunnableARESimulationAgentConfig,
)
from are.simulation.agents.default_agent.prompts import (
    DEFAULT_ARE_SIMULATION_APP_AGENT_REACT_JSON_SYSTEM_PROMPT,
    DEFAULT_ARE_SIMULATION_REACT_JSON_SYSTEM_PROMPT,
)


class AbstractAgentConfigBuilder(ABC):
    """
    Abstract class for building agent configs.
    """

    @abstractmethod
    def build(
        self,
        agent_name: str,
    ) -> Any:
        """
        Method to build a config.
        :param agent_name: Name of the agent that affects the config type.
        :returns: An instance of the config.
        """


class AgentConfigBuilder(AbstractAgentConfigBuilder):
    def build(
        self,
        agent_name: str,
    ) -> RunnableARESimulationAgentConfig:
        match agent_name:
            case "default":
                return ARESimulationReactAgentConfig(
                    agent_name=agent_name,
                    base_agent_config=ARESimulationReactBaseAgentConfig(
                        system_prompt=str(
                            DEFAULT_ARE_SIMULATION_REACT_JSON_SYSTEM_PROMPT
                        ),
                        max_iterations=80,
                    ),
                )

            case "browser":
                return ARESimulationReactAgentConfig(
                    agent_name=agent_name,
                    base_agent_config=ARESimulationReactBaseAgentConfig(
                        system_prompt=str(
                            DEFAULT_ARE_SIMULATION_REACT_JSON_SYSTEM_PROMPT
                        ),
                        max_iterations=80,
                    ),
                )

            case _:
                raise ValueError(f"Agent {agent_name} not found")


class AppAgentConfigBuilder(AbstractAgentConfigBuilder):
    def build(
        self,
        agent_name: str,
    ) -> ARESimulationReactAppAgentConfig:
        match agent_name:
            case "default_app_agent":
                return ARESimulationReactAppAgentConfig(
                    agent_name=agent_name,
                    system_prompt=str(
                        DEFAULT_ARE_SIMULATION_APP_AGENT_REACT_JSON_SYSTEM_PROMPT
                    ),
                    max_iterations=80,
                )
            case _:
                raise ValueError(f"Sub agent {agent_name} not found")
