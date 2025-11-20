import asyncio
import json
import logging
import typing as t
from dataclasses import dataclass

from browser_use import Browser
from web_simulators.simulation import Simulation

from are.simulation.apps.app import App
from are.simulation.tool_utils import OperationType, app_tool, data_tool, env_tool
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
    @env_tool()
    @app_tool()
    @data_tool()
    @event_registered(operation_type=OperationType.READ)
    def search(self, query: str, engine: str = "duckduckgo") -> str:
        """
        Perform a web search using the specified search engine.

        :param query: Search query string
        :param engine: Search engine to use (duckduckgo, google, or bing). Defaults to duckduckgo.
        :returns: Search completion status message
        """
        import urllib.parse

        encoded_query = urllib.parse.quote_plus(query)
        search_engines = {
            "duckduckgo": f"https://duckduckgo.com/?q={encoded_query}",
            "google": f"https://www.google.com/search?q={encoded_query}&udm=14",
            "bing": f"https://www.bing.com/search?q={encoded_query}",
        }
        if engine.lower() not in search_engines:
            return f"Unsupported search engine: {engine}. Options: duckduckgo, google, bing"

        search_url = search_engines[engine.lower()]

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
            memory = f"Searched {engine.title()} for '{query}'"
            logger.info(f"🔍  {memory}")
            return memory
        except Exception as e:
            logger.error(f"Failed to search {engine}: {e}")
            return f'Failed to search {engine} for "{query}": {str(e)}'

    @type_check
    @env_tool()
    @app_tool()
    @data_tool()
    @event_registered(operation_type=OperationType.WRITE)
    def navigate(self, url: str, new_tab: bool = False) -> str:
        """
        Navigate to a URL in the browser.

        :param url: URL to navigate to
        :param new_tab: Whether to open in a new tab. Defaults to False.
        :returns: Navigation status message
        """
        try:
            from browser_use.browser.events import NavigateToUrlEvent

            event = self.browser.event_bus.dispatch(
                NavigateToUrlEvent(url=url, new_tab=new_tab)
            )
            self._run_async(event)
            self._run_async(event.event_result(raise_if_any=True, raise_if_none=False))

            if new_tab:
                memory = f"Opened new tab with URL {url}"
            else:
                memory = f"Navigated to {url}"

            logger.info(f"🔗 {memory}")
            return memory
        except Exception as e:
            logger.error(f"❌ Navigation failed: {str(e)}")
            return f"Navigation failed: {str(e)}"

    @type_check
    @env_tool()
    @app_tool()
    @data_tool()
    @event_registered(operation_type=OperationType.WRITE)
    def go_back(self) -> str:
        """
        Navigate back in browser history.

        :returns: Navigation status message
        """
        try:
            from browser_use.browser.events import GoBackEvent

            event = self.browser.event_bus.dispatch(GoBackEvent())
            self._run_async(event)
            memory = "Navigated back"
            logger.info(f"🔙  {memory}")
            return memory
        except Exception as e:
            logger.error(f"Failed to go back: {type(e).__name__}: {e}")
            return f"Failed to go back: {str(e)}"

    @type_check
    @env_tool()
    @app_tool()
    @data_tool()
    @event_registered(operation_type=OperationType.READ)
    def wait(self, seconds: int = 3) -> str:
        """
        Wait for a specified number of seconds.

        :param seconds: Number of seconds to wait (max 30). Defaults to 3.
        :returns: Wait status message
        """
        # Cap wait time at maximum 30 seconds
        actual_seconds = min(max(seconds - 1, 0), 30)
        memory = f"Waited for {seconds} seconds"
        logger.info(f"🕒 waited for {seconds} second{'' if seconds == 1 else 's'}")
        self._run_async(asyncio.sleep(actual_seconds))
        return memory

    @type_check
    @env_tool()
    @app_tool()
    @data_tool()
    @event_registered(operation_type=OperationType.WRITE)
    def click(
        self,
        index: int | None = None,
        coordinate_x: int | None = None,
        coordinate_y: int | None = None,
    ) -> str:
        """
        Click an element by index or coordinates. Prefer index over coordinates when possible.

        :param index: Element index from browser state. Defaults to None.
        :param coordinate_x: Horizontal coordinate relative to viewport left edge. Defaults to None.
        :param coordinate_y: Vertical coordinate relative to viewport top edge. Defaults to None.
        :returns: Click status message
        """
        # Validate that either index or coordinates are provided
        if index is None and (coordinate_x is None or coordinate_y is None):
            return "Must provide either index or both coordinate_x and coordinate_y"

        # Try index-based clicking first if index is provided
        if index is not None:
            return self._click_by_index(index)
        else:
            return self._click_by_coordinate(coordinate_x, coordinate_y)

    def _click_by_index(self, index: int) -> str:
        """Helper to click by element index"""
        try:
            from browser_use.browser.events import ClickElementEvent

            # Get the element node
            async def get_and_click():
                node = await self.browser.get_element_by_index(index)
                if node is None:
                    return (
                        f"Element index {index} not available - page may have changed"
                    )

                event = self.browser.event_bus.dispatch(ClickElementEvent(node=node))
                await event
                await event.event_result(raise_if_any=True, raise_if_none=False)
                return f"Clicked element {index}"

            memory = self._run_async(get_and_click())
            logger.info(f"🖱️ {memory}")
            return memory
        except Exception as e:
            logger.error(f"Failed to click element {index}: {str(e)}")
            return f"Failed to click element {index}: {str(e)}"

    def _click_by_coordinate(self, x: int, y: int) -> str:
        """Helper to click by coordinate"""
        try:

            async def click_coord():
                page = await self.browser.get_current_page()
                if page is None:
                    return "No active page found"

                mouse = await page.mouse
                await mouse.click(x, y)
                return f"Clicked on coordinate {x}, {y}"

            memory = self._run_async(click_coord())
            logger.info(f"🖱️ {memory}")
            return memory
        except Exception as e:
            logger.error(f"Failed to click at coordinates ({x}, {y}): {str(e)}")
            return f"Failed to click at coordinates ({x}, {y}): {str(e)}"

    @type_check
    @env_tool()
    @app_tool()
    @data_tool()
    @event_registered(operation_type=OperationType.WRITE)
    def input(self, index: int, text: str, clear: bool = True) -> str:
        """
        Input text into an element.

        :param index: Element index from browser state
        :param text: Text to input
        :param clear: Whether to clear existing text first (True) or append (False). Defaults to True.
        :returns: Input status message
        """
        try:
            from browser_use.browser.events import TypeTextEvent

            async def type_text():
                node = await self.browser.get_element_by_index(index)
                if node is None:
                    return (
                        f"Element index {index} not available - page may have changed"
                    )

                event = self.browser.event_bus.dispatch(
                    TypeTextEvent(node=node, text=text, clear=clear)
                )
                await event
                await event.event_result(raise_if_any=True, raise_if_none=False)
                return f"Typed '{text}'"

            memory = self._run_async(type_text())
            logger.debug(memory)
            return memory
        except Exception as e:
            logger.error(
                f"Failed to type text into element {index}: {type(e).__name__}: {e}"
            )
            return f"Failed to type text into element {index}: {str(e)}"

    @type_check
    @env_tool()
    @app_tool()
    @data_tool()
    @event_registered(operation_type=OperationType.WRITE)
    def send_keys(self, keys: str) -> str:
        """
        Send keyboard keys or shortcuts to the browser.

        :param keys: Keys to send (e.g., 'Escape', 'Enter', 'Control+o')
        :returns: Send keys status message
        """
        try:
            from browser_use.browser.events import SendKeysEvent

            event = self.browser.event_bus.dispatch(SendKeysEvent(keys=keys))
            self._run_async(event)
            self._run_async(event.event_result(raise_if_any=True, raise_if_none=False))
            memory = f"Sent keys: {keys}"
            logger.info(f"⌨️  {memory}")
            return memory
        except Exception as e:
            logger.error(f"Failed to send keys: {type(e).__name__}: {e}")
            return f"Failed to send keys: {str(e)}"

    @type_check
    @env_tool()
    @app_tool()
    @data_tool()
    @event_registered(operation_type=OperationType.WRITE)
    def switch_tab(self, tab_id: str) -> str:
        """
        Switch to another open tab by tab_id.

        :param tab_id: 4-character tab ID from browser state tabs list
        :returns: Switch tab status message
        """
        try:
            from browser_use.browser.events import SwitchTabEvent

            async def do_switch():
                target_id = await self.browser.get_target_id_from_tab_id(tab_id)
                event = self.browser.event_bus.dispatch(
                    SwitchTabEvent(target_id=target_id)
                )
                await event
                new_target_id = await event.event_result(
                    raise_if_any=False, raise_if_none=False
                )

                if new_target_id:
                    return f"Switched to tab #{new_target_id[-4:]}"
                else:
                    return f"Switched to tab #{tab_id}"

            memory = self._run_async(do_switch())
            logger.info(f"🔄  {memory}")
            return memory
        except Exception as e:
            logger.warning(f"Tab switch may have failed: {e}")
            return f"Attempted to switch to tab #{tab_id}"

    @type_check
    @env_tool()
    @app_tool()
    @data_tool()
    @event_registered(operation_type=OperationType.WRITE)
    def close_tab(self, tab_id: str) -> str:
        """
        Close a tab by tab_id.

        :param tab_id: 4-character tab ID from browser state tabs list
        :returns: Close tab status message
        """
        try:
            from browser_use.browser.events import CloseTabEvent

            async def do_close():
                target_id = await self.browser.get_target_id_from_tab_id(tab_id)
                event = self.browser.event_bus.dispatch(
                    CloseTabEvent(target_id=target_id)
                )
                await event
                await event.event_result(raise_if_any=False, raise_if_none=False)
                return f"Closed tab #{tab_id}"

            memory = self._run_async(do_close())
            logger.info(f"🗑️  {memory}")
            return memory
        except Exception as e:
            logger.warning(f"Tab {tab_id} may already be closed: {e}")
            return f"Tab #{tab_id} closed (was already closed or invalid)"

    @type_check
    @env_tool()
    @app_tool()
    @data_tool()
    @event_registered(operation_type=OperationType.READ)
    def scroll(
        self, down: bool = True, pages: float = 1.0, index: int | None = None
    ) -> str:
        """
        Scroll the page or a specific element.

        :param down: True to scroll down, False to scroll up. Defaults to True.
        :param pages: Number of pages to scroll (0.5-10.0). Defaults to 1.0.
        :param index: Optional element index to scroll within specific container. Defaults to None.
        :returns: Scroll status message
        """
        try:
            from browser_use.browser.events import ScrollEvent

            async def do_scroll():
                node = None
                if index is not None and index != 0:
                    node = await self.browser.get_element_by_index(index)
                    if node is None:
                        return f"Element index {index} not found in browser state"

                direction = "down" if down else "up"

                # Get viewport height
                try:
                    cdp_session = await self.browser.get_or_create_cdp_session()
                    metrics = await cdp_session.cdp_client.send.Page.getLayoutMetrics(
                        session_id=cdp_session.session_id
                    )
                    css_viewport = metrics.get("cssVisualViewport", {})
                    css_layout_viewport = metrics.get("cssLayoutViewport", {})
                    viewport_height = int(
                        css_viewport.get("clientHeight")
                        or css_layout_viewport.get("clientHeight", 1000)
                    )
                except Exception:
                    viewport_height = 1000

                # Calculate scroll amount
                pixels = int(pages * viewport_height)
                if not down:
                    pixels = -pixels

                event = self.browser.event_bus.dispatch(
                    ScrollEvent(direction=direction, amount=abs(pixels), node=node)
                )
                await event
                await event.event_result(raise_if_any=True, raise_if_none=False)

                target = f"element {index}" if index is not None and index != 0 else ""
                if pages == 1.0:
                    return f"Scrolled {direction} {target} {viewport_height}px".replace(
                        "  ", " "
                    )
                else:
                    return f"Scrolled {direction} {target} {pages} pages".replace(
                        "  ", " "
                    )

            memory = self._run_async(do_scroll())
            logger.info(f"🔍 {memory}")
            return memory
        except Exception as e:
            logger.error(f"Failed to scroll: {type(e).__name__}: {e}")
            return "Failed to execute scroll action"

    @type_check
    @env_tool()
    @app_tool()
    @data_tool()
    @event_registered(operation_type=OperationType.READ)
    def find_text(self, text: str) -> str:
        """
        Scroll to text on the page.

        :param text: Text to find and scroll to
        :returns: Find text status message
        """
        try:
            from browser_use.browser.events import ScrollToTextEvent

            event = self.browser.event_bus.dispatch(ScrollToTextEvent(text=text))

            async def do_find():
                try:
                    await event.event_result(raise_if_any=True, raise_if_none=False)
                    return f"Scrolled to text: {text}"
                except Exception:
                    return f"Text '{text}' not found or not visible on page"

            memory = self._run_async(do_find())
            logger.info(f"🔍  {memory}")
            return memory
        except Exception as e:
            logger.error(f"Failed to find text: {type(e).__name__}: {e}")
            return f"Failed to find text '{text}'"

    @type_check
    @env_tool()
    @app_tool()
    @data_tool()
    @event_registered(operation_type=OperationType.READ)
    def screenshot(self) -> str:
        """
        Request a screenshot of the current viewport to be included in the next observation.

        :returns: Screenshot request status message
        """
        memory = "Requested screenshot for next observation"
        logger.info(f"📸 {memory}")
        return memory

    @type_check
    @env_tool()
    @app_tool()
    @data_tool()
    @event_registered(operation_type=OperationType.READ)
    def dropdown_options(self, index: int) -> str:
        """
        Get all options from a dropdown menu.

        :param index: Element index of the dropdown from browser state
        :returns: Dropdown options list
        """
        try:
            from browser_use.browser.events import GetDropdownOptionsEvent

            async def get_options():
                node = await self.browser.get_element_by_index(index)
                if node is None:
                    return (
                        f"Element index {index} not available - page may have changed"
                    )

                event = self.browser.event_bus.dispatch(
                    GetDropdownOptionsEvent(node=node)
                )
                dropdown_data = await event.event_result(
                    timeout=3.0, raise_if_none=True, raise_if_any=True
                )

                if not dropdown_data:
                    return "Failed to get dropdown options - no data returned"

                return dropdown_data.get("short_term_memory", "Got dropdown options")

            memory = self._run_async(get_options())
            logger.info(memory)
            return memory
        except Exception as e:
            logger.error(f"Failed to get dropdown options: {type(e).__name__}: {e}")
            return f"Failed to get dropdown options for element {index}"

    @type_check
    @env_tool()
    @app_tool()
    @data_tool()
    @event_registered(operation_type=OperationType.WRITE)
    def select_dropdown(self, index: int, text: str) -> str:
        """
        Select an option from a dropdown menu by text.

        :param index: Element index of the dropdown from browser state
        :param text: Exact text or value of the option to select
        :returns: Selection status message
        """
        try:
            from browser_use.browser.events import SelectDropdownOptionEvent

            async def do_select():
                node = await self.browser.get_element_by_index(index)
                if node is None:
                    return (
                        f"Element index {index} not available - page may have changed"
                    )

                event = self.browser.event_bus.dispatch(
                    SelectDropdownOptionEvent(node=node, text=text)
                )
                selection_data = await event.event_result()

                if not selection_data:
                    return "Failed to select dropdown option - no data returned"

                if selection_data.get("success") == "true":
                    return selection_data.get("message", f"Selected option: {text}")
                else:
                    if "short_term_memory" in selection_data:
                        return selection_data["short_term_memory"]
                    else:
                        return selection_data.get(
                            "error", f"Failed to select option: {text}"
                        )

            memory = self._run_async(do_select())
            logger.info(memory)
            return memory
        except Exception as e:
            logger.error(f"Failed to select dropdown: {type(e).__name__}: {e}")
            return f"Failed to select dropdown option '{text}' at index {index}"

    @type_check
    @env_tool()
    @app_tool()
    @data_tool()
    @event_registered(operation_type=OperationType.WRITE)
    def write_file(
        self,
        file_name: str,
        content: str,
        append: bool = False,
        trailing_newline: bool = True,
        leading_newline: bool = False,
    ) -> str:
        """
        Write content to a file in the local file system.

        :param file_name: Name of the file to write
        :param content: Content to write to the file
        :param append: Whether to append to existing file. Defaults to False.
        :param trailing_newline: Whether to add trailing newline. Defaults to True.
        :param leading_newline: Whether to add leading newline. Defaults to False.
        :returns: Write status message
        """
        # TODO: Implement file system integration
        # For now, return a placeholder
        logger.warning("write_file: FileSystem integration not yet implemented")
        return f"write_file not yet implemented for file: {file_name}"

    @type_check
    @env_tool()
    @app_tool()
    @data_tool()
    @event_registered(operation_type=OperationType.WRITE)
    def replace_file(self, file_name: str, old_str: str, new_str: str) -> str:
        """
        Replace specific text within a file.

        :param file_name: Name of the file
        :param old_str: Text to search for
        :param new_str: Text to replace with
        :returns: Replace status message
        """
        # TODO: Implement file system integration
        logger.warning("replace_file: FileSystem integration not yet implemented")
        return f"replace_file not yet implemented for file: {file_name}"

    @type_check
    @env_tool()
    @app_tool()
    @data_tool()
    @event_registered(operation_type=OperationType.READ)
    def read_file(self, file_name: str) -> str:
        """
        Read the complete content of a file.

        :param file_name: Name of the file to read
        :returns: File contents or error message
        """
        # TODO: Implement file system integration
        logger.warning("read_file: FileSystem integration not yet implemented")
        return f"read_file not yet implemented for file: {file_name}"

    @type_check
    @env_tool()
    @app_tool()
    @data_tool()
    @event_registered(operation_type=OperationType.WRITE)
    def evaluate(self, code: str) -> str:
        """
        Execute JavaScript code in the browser context.

        :param code: JavaScript code to execute
        :returns: Execution result or error message
        """
        try:

            async def run_js():
                cdp_session = await self.browser.get_or_create_cdp_session()

                result = await cdp_session.cdp_client.send.Runtime.evaluate(
                    params={
                        "expression": code,
                        "returnByValue": True,
                        "awaitPromise": True,
                    },
                    session_id=cdp_session.session_id,
                )

                # Check for JavaScript execution errors
                if result.get("exceptionDetails"):
                    exception = result["exceptionDetails"]
                    return f"JavaScript execution error: {exception.get('text', 'Unknown error')}"

                # Get the result data
                result_data = result.get("result", {})

                if result_data.get("wasThrown"):
                    return "JavaScript execution failed (wasThrown=true)"

                # Get the actual value
                value = result_data.get("value")

                # Handle different value types
                if value is None:
                    result_text = str(value) if "value" in result_data else "undefined"
                elif isinstance(value, (dict, list)):
                    try:
                        result_text = json.dumps(value, ensure_ascii=False)
                    except (TypeError, ValueError):
                        result_text = str(value)
                else:
                    result_text = str(value)

                # Apply length limit with truncation
                if len(result_text) > 20000:
                    result_text = (
                        result_text[:19950] + "\n... [Truncated after 20000 characters]"
                    )

                return result_text

            memory = self._run_async(run_js())
            logger.debug(
                f"JavaScript executed successfully, result length: {len(memory)}"
            )
            return memory
        except Exception as e:
            logger.error(f"Failed to execute JavaScript: {type(e).__name__}: {e}")
            return f"Failed to execute JavaScript: {str(e)}"
