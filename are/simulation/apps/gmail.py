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
)
from browser_use.agent.prompts import (  # Message formatting
    AgentMessagePrompt,
    SystemPrompt,
)
from browser_use.agent.views import (  # State tracking
    ActionResult,
    AgentHistory,
    AgentState,
)
from browser_use.browser.views import (
    BrowserStateSummary,  # Browser state data structure
)
from browser_use.llm.base import BaseChatModel  # LLM abstraction
from browser_use.llm.openai import ChatOpenAI
from browser_use.tools.service import Tools
from browser_use.tools.views import SearchAction
from pydantic import Field
from web_simulators.simulation import Simulation

from are.simulation.agents.multimodal import Attachment
from are.simulation.apps.agent_user_interface import AgentUserInterface
from are.simulation.apps.app import App
from are.simulation.scenarios.scenario import Scenario
from are.simulation.scenarios.utils.registry import register_scenario
from are.simulation.scenarios.validation_result import ScenarioValidationResult
from are.simulation.tool_utils import OperationType, data_tool
from are.simulation.types import EventRegisterer, event_registered
from are.simulation.utils import get_state_dict, type_check

logger = logging.getLogger(__name__)


@dataclass(init=False)
class GmailApp(App):
    """
    VibrantLabs Browser running a simulation of Gmail.
    """

    _skip_deepcopy_fields = ["_loop", "_session", "_spwan", "browser"]
    _skip_pickle_fields = _skip_deepcopy_fields

    name: str | None = "GmailApp"
    _state_file: str = "data/gmail_state.json"

    def __init__(self, browser: Browser | None = None):
        """Initialize the Gmail App by starting and spawning the simulation immediately.

        Args:
            browser: Optional Browser instance to use. If None, creates its own browser.
        """
        self._connected = False

        # Use provided browser or create new one
        self.browser = browser if browser is not None else Browser()
        self._owns_browser = browser is None  # Track if we created the browser

        # Async event loop for the simulation
        self._loop = asyncio.new_event_loop()
        self.app_state = self._load_state_from_file(self._state_file)
        self.sim: Simulation = self.spawn()
        self.agent = Agent(task="", llm=ChatBrowserUse())

        # Only start browser if we own it
        if self._owns_browser:
            self._fake_await(self.browser.start())
        super().__init__(self.name)

    def _load_state_from_file(self, path: str) -> dict:
        """Load Gmail state from any JSON file"""
        with open(path) as f:
            return json.load(f)

    def _fake_await(self, coro: t.Any) -> t.Any:
        """Run an async coroutine."""
        return self._loop.run_until_complete(coro)

    def spawn(self):
        """Spawn the simulation"""
        sim: Simulation = self._fake_await(
            Simulation.spawn("gmail-clone", state=self.app_state, headless=True)
        )
        self._connected = True
        current_url = self._fake_await(sim.get_url())
        logger.info("Successfully spawned up the simulation")
        # Only start browser if we own it (external browsers are already started)
        if self._owns_browser:
            self._fake_await(self.browser.start())
        self._fake_await(self.browser.navigate_to(url=current_url))

        return sim

    def get_state(self) -> dict[str, t.Any]:
        """
        Return the app's current state for the Gmail Simulation.
        """
        return get_state_dict(self, ["app_state"])

    def load_state(self, state_dict: dict[str, t.Any]):
        """
        Restore app state.
        """
        # Load the serializable configuration
        for key, value in state_dict.items():
            setattr(self, key, value)

        self.sim: Simulation = self.spawn()

    def reset(self):
        """Reset app to initial state."""
        self.app_state = {}
        self.close()
        super().reset()

    def close(self):
        """Shutdown the Simulation"""
        self._fake_await(self.sim.aclose())
        # Only stop browser if we own it
        if self._owns_browser:
            self._fake_await(self.browser.stop())

    @type_check
    @data_tool()
    @event_registered(operation_type=OperationType.READ)
    def search(self, params: SearchAction):
        import urllib.parse

        encoded_query = urllib.parse.quote_plus(params.query)
        search_engines = {
            "duckduckgo": f"https://duckduckgo.com/?q={encoded_query}",
            "google": f"https://www.google.com/search?q={encoded_query}&udm=14",
            "bing": f"https://www.bing.com/search?q={encoded_query}",
        }
        if params.engine.lower() not in search_engines:
            return ActionResult(
                error=f"Unsupported search engine: {params.engine}. Options: duckduckgo, google, bing"
            )

        search_url = search_engines[params.engine.lower()]

        # Simple tab logic: use current tab by default
        use_new_tab = False

        # Dispatch navigation event
        try:
            from browser_use.browser.events import NavigateToUrlEvent

            event = self.browser.event_bus.dispatch(
                NavigateToUrlEvent(
                    url=search_url,
                    new_tab=use_new_tab,
                )
            )
            self._fake_await(event)
            self._fake_await(event.event_result(raise_if_any=True, raise_if_none=False))
            memory = f"Searched {params.engine.title()} for '{params.query}'"
            msg = f"🔍  {memory}"
            logger.info(msg)
            return ActionResult(extracted_content=memory, long_term_memory=memory)
        except Exception as e:
            logger.error(f"Failed to search {params.engine}: {e}")
            return ActionResult(
                error=f'Failed to search {params.engine} for "{params.query}": {str(e)}'
            )
