# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


import logging
import time
from datetime import datetime, timezone
from types import MethodType
from typing import Any, Callable

from are.simulation.agents.adapters import register_event
from are.simulation.agents.agent_execution_result import AgentExecutionResult
from are.simulation.agents.are_simulation_agent import RunnableARESimulationAgent
from are.simulation.agents.default_agent.base_agent import (
    BaseAgent,
    BaseAgentLog,
    RunningState,
)
from are.simulation.agents.default_agent.default_tools import Tool
from are.simulation.agents.default_agent.prompts import get_notification_system_prompt
from are.simulation.agents.llm.llm_engine import LLMEngine
from are.simulation.agents.llm.types import MMObservation
from are.simulation.agents.multimodal import Attachment
from are.simulation.apps import AgentUserInterface
from are.simulation.notification_system import (
    BaseNotificationSystem,
    Message,
    MessageType,
)
from are.simulation.scenarios import Scenario
from are.simulation.time_manager import TimeManager
from are.simulation.tool_utils import AppTool, AppToolAdapter
from are.simulation.types import SimulatedGenerationTimeConfig

logger: logging.Logger = logging.getLogger(__name__)


class ARESimulationAgent(RunnableARESimulationAgent):
    def __init__(
        self,
        log_callback: Callable[[BaseAgentLog], None],
        pause_env: Callable[[], None] | None,
        resume_env: Callable[[float], None] | None,
        llm_engine: LLMEngine,
        base_agent: BaseAgent,
        time_manager: TimeManager,
        tools: list[Tool] | None = None,
        max_iterations: int = 80,
        max_turns: int | None = None,
        simulated_generation_time_config: SimulatedGenerationTimeConfig | None = None,
    ):
        super().__init__()

        if tools is None:
            tools = []

        # Main agent model
        self.llm_engine = llm_engine
        self.tools = tools
        self.max_iterations = max_iterations
        self.time_manager = time_manager
        # Max turns refers to the number of turns of the conversation between the user and the agent.
        # 1 turn for instance means one task from the user and one response from the agent.
        # If this is None, we will loop forever waiting for the user message, or env notification.
        # Until the environment sends an ENVIRONMENT_STOP message.
        self.max_turns = max_turns

        # Main agent
        self.react_agent = base_agent
        self.react_agent.max_iterations = max_iterations
        self.react_agent.llm_engine = self.llm_engine
        self.react_agent.time_manager = self.time_manager
        self.react_agent.log_callback = log_callback

        # Environment methods to handle simulation time.
        self.simulated_generation_time_config = simulated_generation_time_config
        self.pause_env = pause_env
        self.resume_env = resume_env
        self.react_agent.simulated_generation_time_config = (
            self.simulated_generation_time_config
        )

        self.sub_agents = []
        self.set_subagents()

        self._initialized = False

    @property
    def agent_framework(self) -> str:
        return "ARESimulationAgent"

    @property
    def model(self) -> str:
        return self.llm_engine.model_name

    def run_scenario(
        self,
        scenario: Scenario,
        notification_system: BaseNotificationSystem | None = None,
        initial_agent_logs: list[BaseAgentLog] | None = None,
    ) -> AgentExecutionResult:
        if not self._initialized:
            self.prepare_are_simulation_run(
                scenario=scenario,
                notification_system=notification_system,
                initial_agent_logs=initial_agent_logs,
            )
        else:
            logger.warning(
                "Agent already ready for run. Skipping prepare_are_simulation_run()."
            )

        max_turns = self.max_turns

        if scenario.nb_turns is not None:
            logger.warning(f"Setting agent max_turns to {scenario.nb_turns}")
            max_turns = scenario.nb_turns

        result = self.agent_loop(
            max_turns=max_turns, initial_agent_logs=initial_agent_logs
        )

        return AgentExecutionResult(output=result)

    def init_tools(self, scenario: Scenario):
        app_tools = scenario.get_tools()
        logger.info(
            f"Found {len(app_tools)} tools: {[tool.name for tool in app_tools]}"
        )
        app_tools = self.remove_aui_irrelevant_tools(app_tools)
        are_simulation_tools = [AppToolAdapter(tool) for tool in app_tools]
        self.tools += are_simulation_tools
        self.react_agent.tools = {tool.name: tool for tool in self.tools}

    def init_system_prompt(self, scenario: Scenario):
        additional_system_prompt = scenario.additional_system_prompt
        logger.info(f"Additional System Prompt: {additional_system_prompt}")

        if additional_system_prompt is not None:
            self.react_agent.init_system_prompts["system_prompt"] += (
                "\n\n" + additional_system_prompt
            )

        notification_system_prompt = get_notification_system_prompt(
            self.react_agent.notification_system, scenario.apps
        )
        self.react_agent.init_system_prompts["system_prompt"] = (
            self.react_agent.init_system_prompts["system_prompt"].replace(
                "<<notification_system_description>>", notification_system_prompt
            )
        )

        date_str = datetime.fromtimestamp(
            scenario.start_time or 0, tz=timezone.utc
        ).strftime("%Y-%m-%d %H")
        self.react_agent.init_system_prompts["system_prompt"] = (
            self.react_agent.init_system_prompts["system_prompt"].replace(
                "<<curent_time_description>>",
                f"Today's date in 'YYYY-MM-DD HH' format is {date_str}",
            )
        )

        self.react_agent.init_system_prompts["system_prompt"] = (
            self.react_agent.init_system_prompts["system_prompt"].replace(
                "<<agent_reminder_description>>",
                "",
            )
        )

    def init_notification_system(
        self,
        notification_system: BaseNotificationSystem | None = None,
    ):
        if notification_system is not None:
            logger.info(
                f"Setting notification system for Agent to provided one {notification_system}"
            )
            self.react_agent.notification_system = notification_system

    def prepare_are_simulation_run(
        self,
        scenario: Scenario,
        notification_system: BaseNotificationSystem | None = None,
        initial_agent_logs: list[BaseAgentLog] | None = None,
    ):
        # Inject browser and loop from scenario if available
        if hasattr(scenario, "browser") and scenario.browser is not None:
            from are.simulation.agents.custom_agents.browser_agent import (
                BrowserAgent,
            )

            if isinstance(self.react_agent, BrowserAgent):
                self.react_agent.browser = scenario.browser
                logger.info("Injected browser from scenario into BrowserAgent")

                # Inject event loop - prefer scenario's loop, create fallback if needed
                if hasattr(scenario, "_loop") and scenario._loop is not None:
                    self.react_agent._loop = scenario._loop
                    logger.info("Injected event loop from scenario into BrowserAgent")
                elif self.react_agent._loop is None:
                    # BrowserAgent needs a loop - create one if not provided
                    import asyncio

                    self.react_agent._loop = asyncio.get_event_loop()
                    logger.warning(
                        "Using default event loop for BrowserAgent (scenario didn't provide one)"
                    )

        self.init_tools(scenario)
        self.init_notification_system(notification_system)
        self.init_system_prompt(scenario)
        self._initialized = True
        # Sync the base agent time manager
        if initial_agent_logs is not None and len(initial_agent_logs) > 0:
            self.react_agent.replay(initial_agent_logs)
        # Pause/resume env functions

        if self.simulated_generation_time_config is not None:
            if self.pause_env is None or self.resume_env is None:
                raise Exception(
                    "Pause and resume environment functions must be provided if simulated generation time config is set"
                )
        self.react_agent.pause_env = self.pause_env
        self.react_agent.resume_env = self.resume_env

    def remove_aui_irrelevant_tools(self, app_tools: list[AppTool]) -> list[AppTool]:
        aui_tool = next(tool for tool in app_tools if "AgentUserInterface" in tool.name)

        if aui_tool is not None:
            aui: AgentUserInterface = aui_tool.class_instance  # type: ignore
            # We set this to True here because all the messages from the user are going to be received by the Agent as notifications
            # And thus handled as new tasks, instead of the Agent blocking when sending a message to the user waiting for a response.
            logger.warning(
                "Setting wait_for_user_response to False in AgentUserInterface"
            )
            aui.wait_for_user_response = False

            # Here we remove these tools, because all user messages will be injected to Agent
            # And thus he won't need to use these tools to get the messages.
            tools_to_remove = {
                "AgentUserInterface__get_last_message_from_user",
                "AgentUserInterface__get_last_message_from_agent",
                "AgentUserInterface__get_last_unread_messages",
                "AgentUserInterface__get_all_messages",
            }
            logger.warning(f"Removing tools {tools_to_remove} from app_tools")
            app_tools = [tool for tool in app_tools if tool.name not in tools_to_remove]
        return app_tools

    def agent_loop(
        self,
        initial_task: str | None = None,
        max_turns: int | None = None,
        initial_agent_logs: list[BaseAgentLog] | None = None,
    ) -> str | MMObservation | None:
        """
        This is a completely synchronous Agentic loop.
        Where the agent will run on a given task, where task = user message or env notification, until it returns a result.
        Then the Agent will run again on a nother task given by the user (or env) in the next turn.
        This loop will run until max_turns is reached or ENVIRONMENT_STOP message is received.
        """
        iterations = 0
        result = ""

        if self.react_agent.notification_system is None:
            raise Exception("Agent Notification system not set")

        if initial_task is not None and initial_agent_logs is not None:
            raise Exception(
                "Cannot provide both initial_task and initial_agent_logs. Please provide only one."
            )

        if initial_task is not None:
            self.react_agent.notification_system.message_queue.put(
                Message(
                    message_type=MessageType.USER_MESSAGE,
                    message=initial_task,
                    timestamp=datetime.fromtimestamp(
                        self.time_manager.time(), tz=timezone.utc
                    ),
                )
            )

        # Before checking for new messages, finish the current turn
        if initial_agent_logs:
            self.react_agent.init_tools()
            result = self.react_agent.execute_agent_loop()
            iterations += 1

        reset = True
        while max_turns is None or iterations < max_turns:
            new_user_messages, new_notifications, env_stop_messages = (
                self.get_notifications(self.react_agent.notification_system)
            )

            if len(env_stop_messages) > 0:
                logger.warning(
                    f"Environment stop message received - Stopping Agent: {env_stop_messages}"
                )
                break
            if len(new_user_messages) > 0 or len(new_notifications) > 0:
                # Get task from user, the env notifications are already handled by the base agent.
                task = self.build_task_from_notifications(new_user_messages)
                attachments: list[Attachment] = [
                    attachment
                    for user_message in new_user_messages
                    for attachment in user_message.attachments
                ]
                logger.debug(
                    f"Running agent with task '{task}' at iteration {iterations} and reset {reset} with attachments {attachments}"
                )
                result = self.react_agent.run(
                    task=task, hint=None, reset=reset, attachments=attachments
                )
                reset = False
                running_state = self.react_agent.custom_state.get("running_state", None)
                if running_state == RunningState.TERMINATED:
                    # Agent is at the end of a turn
                    iterations += 1
                    logger.debug(f"End of turn {iterations}")
                elif running_state == RunningState.PAUSED:
                    # Agent is paused within a turn
                    logger.debug("Agent paused")
                elif running_state == RunningState.FAILED:
                    agent_logs = self.react_agent.get_agent_logs()
                    error_message = (
                        f"Last agent log: {agent_logs[-1]}"
                        if len(agent_logs) > 0
                        else "No agent logs"
                    )
                    raise Exception(f"Agent failed. {error_message}")
                else:
                    raise Exception(f"Unknown running state: {running_state}")
            else:
                logger.debug("No new messages from user or environment")
                # Sleep for 1 second to avoid busy looping
                time.sleep(1)

        if max_turns is not None and iterations >= max_turns:
            logger.warning(f"Max iterations reached - Stopping Agent: {max_turns}")

        # Make sure all the sub agents are stopped
        for sub_agent in self.sub_agents:
            sub_agent.stop()

        return result

    def stop(self) -> None:
        self.react_agent.stop()

    def get_notifications(
        self, notification_system: BaseNotificationSystem
    ) -> tuple[list[Message], list[Message], list[Message]]:
        if notification_system is None:
            raise Exception("Agent Notification system not set")

        new_messages = notification_system.message_queue.get_by_timestamp(
            datetime.fromtimestamp(self.time_manager.time(), tz=timezone.utc)
        )
        new_user_messages = [
            message
            for message in new_messages
            if message.message_type == MessageType.USER_MESSAGE
        ]
        new_notifications = [
            message
            for message in new_messages
            if message.message_type == MessageType.ENVIRONMENT_NOTIFICATION
        ]
        # Put back the notifications to the queue.
        # They should be processed by the preprocessing step.
        for message in new_notifications:
            notification_system.message_queue.put(message)
        env_stop_messages = [
            message
            for message in new_messages
            if message.message_type == MessageType.ENVIRONMENT_STOP
        ]

        return new_user_messages, new_notifications, env_stop_messages

    def build_task_from_notifications(self, user_notifications: list[Message]) -> str:
        if len(user_notifications) > 0:
            logger.info(f"New messages from user {user_notifications}")
        task = "\n".join([message.message for message in user_notifications])
        return task

    def set_subagents(self):
        for tool in self.tools:
            for attr_name, attr_value in vars(tool).items():
                if isinstance(attr_value, BaseAgent):
                    logger.info(
                        f"Found inner Agent of {self} {attr_name} - setting parent field"
                    )
                    attr_value.parent = self.react_agent
                    attr_value.time_manager = self.react_agent.time_manager
                    if attr_value not in self.sub_agents:
                        self.sub_agents.append(attr_value)
                    if self.simulated_generation_time_config is not None:
                        attr_value.simulated_generation_time_config = (
                            self.simulated_generation_time_config
                        )
                        attr_value.pause_env = self.pause_env
                        attr_value.resume_env = self.resume_env

    def _register_sub_agent_events(
        self,
        add_to_event_log: Callable,
        tools_to_decorate: list[str],
        time_manager: TimeManager,
    ):
        for sub_agent in self.sub_agents:
            for tool_name, tool in sub_agent.tools.items():
                if tool_name in tools_to_decorate:
                    logger.info(
                        f"Registering tool for sub-agent {tool_name}: {tool_name}"
                    )
                    decorated_forward = register_event(
                        time_manager=time_manager, add_to_event_log=add_to_event_log
                    )(tool.forward)
                    tool.forward = MethodType(decorated_forward, tool)

    def _get_add_to_event_log(self, app_tools: list[Any]) -> Callable:
        aui_tool = next(tool for tool in app_tools if "AgentUserInterface" in tool.name)
        aui: AgentUserInterface = aui_tool.class_instance  # type: ignore
        return aui.add_event
