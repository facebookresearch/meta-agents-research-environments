# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import asyncio

from browser_use import Browser

from are.simulation.agents.are_simulation_agent_config import (
    ARESimulationReactAppAgentConfig,
    ARESimulationReactBaseAgentConfig,
)
from are.simulation.agents.custom_agents.browser_agent import BrowserAgent
from are.simulation.agents.default_agent.base_agent import BaseAgent
from are.simulation.agents.default_agent.steps.are_simulation import (
    get_are_simulation_update_pre_step,
)
from are.simulation.agents.default_agent.termination_methods.are_simulation import (
    get_gaia2_termination_step,
    termination_step_are_simulation_final_answer,
)
from are.simulation.agents.default_agent.tools.json_action_executor import (
    JsonActionExecutor,
)
from are.simulation.agents.llm.llm_engine import LLMEngine


def are_simulation_react_json_agent(
    llm_engine: LLMEngine, base_agent_config: ARESimulationReactBaseAgentConfig
):
    return BaseAgent(
        llm_engine=llm_engine,
        tools={},
        system_prompts={
            "system_prompt": str(base_agent_config.system_prompt),
        },
        termination_step=get_gaia2_termination_step(),
        max_iterations=base_agent_config.max_iterations,
        action_executor=JsonActionExecutor(
            use_custom_logger=base_agent_config.use_custom_logger
        ),
        conditional_pre_steps=[get_are_simulation_update_pre_step()],
        use_custom_logger=base_agent_config.use_custom_logger,
    )


def are_simulation_react_json_app_agent(
    llm_engine: LLMEngine,
    app_agent_config: ARESimulationReactAppAgentConfig,
    log_callback=None,
):
    return BaseAgent(
        llm_engine=llm_engine,
        tools={},
        system_prompts={
            "system_prompt": app_agent_config.system_prompt,
        },
        termination_step=termination_step_are_simulation_final_answer(),
        max_iterations=app_agent_config.max_iterations,
        action_executor=JsonActionExecutor(
            use_custom_logger=app_agent_config.use_custom_logger
        ),
        use_custom_logger=app_agent_config.use_custom_logger,
        log_callback=log_callback,
    )


def are_simulation_browser_react_agent(
    llm_engine: LLMEngine,
    base_agent_config: ARESimulationReactBaseAgentConfig,
    browser: Browser | None = None,
    loop: asyncio.AbstractEventLoop | None = None,
):
    """
    Creates a browser-enabled ReAct agent with the same configuration as
    are_simulation_react_json_agent, but with browser state injection.

    This agent automatically captures and includes browser state (DOM + screenshot)
    before each LLM call, enabling web automation tasks.

    Args:
        llm_engine: The LLM engine to use for agent reasoning
        base_agent_config: Base configuration (same as standard agent)
        browser: Optional Browser instance to share. If None, agent creates its own.
        loop: Optional event loop to use. If None, uses asyncio.get_event_loop().

    Returns:
        BrowserAgent configured for ARE simulation with browser capabilities
    """
    return BrowserAgent(
        llm_engine=llm_engine,
        browser=browser,
        loop=loop,
        tools={},
        system_prompts={
            "system_prompt": str(base_agent_config.system_prompt),
        },
        termination_step=get_gaia2_termination_step(),
        max_iterations=base_agent_config.max_iterations,
        action_executor=JsonActionExecutor(
            use_custom_logger=base_agent_config.use_custom_logger
        ),
        # BrowserAgent will automatically add browser state capture to this list
        conditional_pre_steps=[get_are_simulation_update_pre_step()],
        use_custom_logger=base_agent_config.use_custom_logger,
    )
