import asyncio
import json
import logging
import typing as t
from dataclasses import dataclass

from browser_use import (
    Browser,
)
from browser_use.agent.views import (  # State tracking
    ActionResult,
)
from browser_use.tools.views import SearchAction
from web_simulators.simulation import Simulation

from are.simulation.apps.app import App
from are.simulation.tool_utils import OperationType, app_tool, data_tool
from are.simulation.types import event_registered
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

    def __init__(
        self,
        browser: Browser | None = None,
        loop: asyncio.AbstractEventLoop | None = None,
    ):
        """Initialize the Gmail App by starting and spawning the simulation immediately.

        Args:
            browser: Optional Browser instance to use. If None, creates its own browser.
            loop: Optional event loop to use. If None, uses asyncio.get_event_loop().
        """
        self._connected = False

        # Use provided browser or create new one
        self.browser = browser if browser is not None else Browser()
        self._owns_browser = browser is None  # Track if we created the browser

        # Use provided loop or get the current event loop
        self._loop = loop if loop is not None else asyncio.get_event_loop()

        self.app_state = self._load_state_from_file(self._state_file)
        self.sim: Simulation = self.spawn()

        super().__init__(self.name)

    def _load_state_from_file(self, path: str) -> dict:
        """Load Gmail state from any JSON file"""
        with open(path) as f:
            return json.load(f)

    def _run_async(self, coro: t.Any) -> t.Any:
        """Run an async coroutine in the event loop."""
        return self._loop.run_until_complete(coro)

    def spawn(self):
        """Spawn the simulation"""
        sim: Simulation = self._run_async(
            Simulation.spawn("gmail-clone", state=self.app_state, headless=True)
        )
        self._connected = True
        current_url = self._run_async(sim.get_url())
        logger.info("Successfully spawned up the simulation")
        # Only start browser if we own it (external browsers are already started)
        logger.info(f"Gmail spawning new browser instance: {self._owns_browser}")

        if self._owns_browser:
            self._run_async(self.browser.start())
        self._run_async(self.browser.navigate_to(url=current_url))

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
        self._run_async(self.sim.aclose())
        # Only stop browser if we own it
        if self._owns_browser:
            self._run_async(self.browser.stop())

    @type_check
    @app_tool()
    @data_tool()
    @event_registered(operation_type=OperationType.READ)
    def search(self, params: SearchAction):
        """
        Perform a web search using the specified search engine.

        :param params: SearchAction containing query string and search engine
        :returns: ActionResult with search completion status and memory
        """
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
            self._run_async(event)
            self._run_async(event.event_result(raise_if_any=True, raise_if_none=False))
            memory = f"Searched {params.engine.title()} for '{params.query}'"
            msg = f"🔍  {memory}"
            logger.info(msg)
            return ActionResult(extracted_content=memory, long_term_memory=memory)
        except Exception as e:
            logger.error(f"Failed to search {params.engine}: {e}")
            return ActionResult(
                error=f'Failed to search {params.engine} for "{params.query}": {str(e)}'
            )
