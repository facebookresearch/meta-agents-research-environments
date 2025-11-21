# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import asyncio

from browser_use import Browser

from are.simulation.apps.agent_user_interface import AgentUserInterface
from are.simulation.apps.gmail import GmailApp
from are.simulation.scenarios.scenario import Scenario, ScenarioValidationResult
from are.simulation.scenarios.utils.registry import register_scenario
from are.simulation.types import EventRegisterer


@register_scenario("scenario_gmail_browser")
class GmailBrowserScenario(Scenario):
    """
    Scenario that demonstrates shared browser between Gmail app and BrowserAgent.

    The browser instance is created once in the scenario and shared with:
    1. GmailApp - uses browser for all tool operations
    2. BrowserAgent - captures browser state before each LLM call

    This allows the agent to see and interact with the same browser state
    that the Gmail app manipulates.
    """

    start_time: float | None = 0
    duration: float | None = 120

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.browser: Browser | None = None
        self._loop = asyncio.new_event_loop()

    def initialize(self, **kwargs) -> None:
        """Initialize scenario and create browser instance."""

        # Create browser instance ONCE
        self.browser = Browser()
        self._loop.run_until_complete(self.browser.start())

        super().initialize(**kwargs)

    def init_and_populate_apps(self, **kwargs) -> None:
        """Initialize apps with shared browser."""
        _ = kwargs  # Acknowledge unused parameter
        # Agent User Interface
        aui = AgentUserInterface()

        # Gmail app with shared browser
        gmail = GmailApp(browser=self.browser, loop=self._loop)  # type: ignore[call-arg]

        # Register all apps
        self.apps = [aui, gmail]

    def build_events_flow(self) -> None:
        """Define scenario events."""
        aui = self.get_typed_app(AgentUserInterface)

        # Use capture_mode to create events
        with EventRegisterer.capture_mode():
            # Send initial task to agent
            event1 = aui.send_message_to_agent(
                content="What's the top email say? Open it and read?",
            ).depends_on(None, delay_seconds=2)

        self.events = [event1]

    def validate(self, env) -> ScenarioValidationResult:
        """
        Validate scenario completion.

        Args:
            env: The environment containing all apps and their final states

        Returns:
            ScenarioValidationResult with success status and feedback
        """
        gmail = env.get_app("GmailApp")
        current_state = gmail.get_state()

        return ScenarioValidationResult(
            success=len(current_state["emails"]) > 1,
        )

    def get_user_prompt(self) -> str:
        """Return the initial task prompt for the agent."""
        return "Navigate to Gmail and verify that you can see the inbox. Describe what you see."
