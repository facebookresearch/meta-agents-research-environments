# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import asyncio
import typing as t
from dataclasses import dataclass

from browser_use import Browser
from browser_use.browser.views import BrowserStateSummary

from are.simulation.agents.agent_log import BaseAgentLog
from are.simulation.agents.default_agent.base_agent import (
    DEFAULT_STEP_2_MESSAGE,
    DEFAULT_STEP_2_ROLE,
    BaseAgent,
    ConditionalStep,
)
from are.simulation.agents.default_agent.tools.json_action_executor import (
    JsonActionExecutor,
)
from are.simulation.agents.llm.llm_engine import LLMEngine
from are.simulation.agents.multimodal import Attachment


@dataclass
class BrowserStateLog(BaseAgentLog):
    """Log entry for browser state (DOM + screenshot)."""

    dom_content: str
    screenshot: Attachment | None = None

    def get_content_for_llm(self) -> str | None:
        return f"Current Browser DOM:\n{self.dom_content}"

    def get_attachments_for_llm(self) -> list[Attachment]:
        return [self.screenshot] if self.screenshot else []

    def get_type(self) -> str:
        return "browser_state"


class BrowserAgent(BaseAgent):
    """
    Agent specialized for browser interaction.
    Automatically captures browser state (DOM + screenshot) before each LLM call.

    Args:
        llm_engine: The LLM engine for agent reasoning
        browser: Optional Browser instance to use. If None, agent creates its own.
        loop: Optional event loop to use. Will be injected from scenario before use.
    """

    def __init__(
        self,
        llm_engine: LLMEngine,
        browser: Browser | None = None,
        loop: asyncio.AbstractEventLoop | None = None,
        **kwargs,
    ):
        # Extend the Role Dictionary
        role_dict = kwargs.get("role_dict", DEFAULT_STEP_2_ROLE).copy()
        role_dict["browser_state"] = "user"  # Treat browser state as a user message

        # Extend the Message Dictionary
        message_dict = kwargs.get("message_dict", DEFAULT_STEP_2_MESSAGE).copy()
        message_dict["browser_state"] = "BROWSER STATE UPDATE:\n***\n{content}\n***\n"

        # Default action executor if not provided
        if "action_executor" not in kwargs:
            kwargs["action_executor"] = JsonActionExecutor(use_custom_logger=True)

        # Store loop (can be None initially, will be injected before use)
        self._loop = loop

        # Use provided browser or create new one
        self.browser = browser
        self._owns_browser = browser is None  # Track if we created the browser

        # Lazy initialization of browser during first run
        conditional_pre_steps = kwargs.pop("conditional_pre_steps", [])

        # Only add browser start step if we need to create our own browser
        # if self._owns_browser:
        #     conditional_pre_steps.insert(
        #         0,
        #         ConditionalStep(
        #             condition=lambda agent: agent.iterations
        #             == 0,  # Only the first time
        #             function=lambda _agent: self._start_browser(),
        #             name="start_browser_lazy",
        #         ),
        #     )

        # Always add the browser state capture pre-step
        conditional_pre_steps.insert(
            1 if self._owns_browser else 0,
            ConditionalStep(
                condition=lambda _agent: True,  # Run on every step
                function=lambda _agent: self.capture_browser_state_step(),
                name="capture_browser_state",
            ),
        )

        super().__init__(
            llm_engine=llm_engine,
            role_dict=role_dict,
            message_dict=message_dict,
            conditional_pre_steps=conditional_pre_steps,
            **kwargs,
        )

    def _start_browser(self):
        self.browser = Browser()
        self._fake_await(self.browser.start())

    def _fake_await(self, coro: t.Any) -> t.Any:
        """Run an async coroutine."""
        return self._loop.run_until_complete(coro)

    def capture_browser_state_step(self):
        """
        Captures the current browser state (DOM + screenshot) and adds it to agent logs.
        This function is called as a pre-step before each LLM call.
        """
        try:
            if self.browser:
                browser_state: BrowserStateSummary = self._fake_await(
                    self.browser.get_browser_state_summary(
                        include_screenshot=True, include_recent_events=True
                    )
                )
                timestamp = self.make_timestamp()
                screenshot = Attachment(
                    name=f"ss_{timestamp}",
                    base64_data=browser_state.screenshot,
                    mime="image/jpeg",
                )
                self.append_agent_log(
                    BrowserStateLog(
                        dom_content=browser_state.dom_state.llm_representation(),
                        screenshot=screenshot,
                        timestamp=timestamp,
                        agent_id=self.agent_id,
                    )
                )
        except Exception as e:
            self.logger.error(f"Failed to capture browser state: {e}")
