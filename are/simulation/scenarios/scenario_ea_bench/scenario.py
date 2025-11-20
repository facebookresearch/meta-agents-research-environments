import asyncio
import json
import logging
import typing as t
from dataclasses import dataclass, field

from browser_use import (
    Agent,
    Browser,
    BrowserSession,
    ChatBrowserUse,  # Browser control
    DomService,  # DOM extraction
    Tools,  # Action registry
)
from browser_use.agent.prompts import (  # Message formatting
    AgentMessagePrompt,
    SystemPrompt,
)
from browser_use.agent.views import AgentHistory, AgentState  # State tracking
from browser_use.browser.views import (
    BrowserStateSummary,  # Browser state data structure
)
from browser_use.llm.base import BaseChatModel  # LLM abstraction
from browser_use.llm.openai import ChatOpenAI
from web_simulators.simulation import Simulation

from are.simulation.apps.agent_user_interface import AgentUserInterface
from are.simulation.apps.gmail import GmailApp
from are.simulation.scenarios.scenario import Scenario
from are.simulation.scenarios.utils.registry import register_scenario
from are.simulation.scenarios.validation_result import ScenarioValidationResult
from are.simulation.tool_utils import OperationType, data_tool
from are.simulation.types import EventRegisterer, event_registered
from are.simulation.utils import get_state_dict, type_check

logger = logging.getLogger(__name__)


@register_scenario("gmail_test_scenario")
class GmailTestScenario(Scenario):
    """
    Test scenario built around the Gmail Simulation.
    This scenario shows:
    - Simulaton initialization
    - Browser control through the action space
    - Agent interaction with the simulation
    - Validation and verification
    """

    start_time: float | None = 0
    duration: float | None = 30

    def init_and_populate_apps(self, *args, **kwargs) -> None:
        gmail_app = GmailApp()
        agui = AgentUserInterface()

        self.apps = [gmail_app, agui]

    def build_events_flow(self) -> None:
        """Define the sequence of events that occur during the scenario"""

        agui = self.get_typed_app(AgentUserInterface)
        gmail_app = self.get_typed_app(GmailApp)

        with EventRegisterer.capture_mode():
            event1 = agui.send_message_to_agent(content="Say hi to me.").depends_on(
                None, delay_seconds=2
            )
            oracle1 = (
                agui.send_message_to_user(content="Hi!")
                .oracle()
                .depends_on(event1, delay_seconds=2)
            )
        self.events = [event1, oracle1]

    def validate(self, env) -> ScenarioValidationResult:
        """Validate the scenario"""
        return ScenarioValidationResult(success=True)


if __name__ == "__main__":
    from are.simulation.scenarios.utils.cli_utils import run_and_validate

    run_and_validate(GmailTestScenario())
