# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


import asyncio
import logging
import random
from typing import Any, Callable, TypeVar

from mcp import ClientSession, StdioServerParameters
from mcp.client.sse import sse_client
from mcp.client.stdio import stdio_client
from pydantic.networks import AnyUrl

from are.simulation.apps.app import App, ToolType
from are.simulation.tool_utils import (
    AppTool,
    AppToolArg,
    OperationType,
    ToolAttributeName,
)
from are.simulation.types import event_registered
from are.simulation.utils import get_state_dict

logger = logging.getLogger(__name__)


class MCPApp(App):
    """
    A Meta Agents Research Environments app that connects to an MCP server and exposes its tools.

    This app allows Meta Agents Research Environments to interact with any MCP-compatible server, making
    the server's tools available as Meta Agents Research Environments app tools.
    """

    # Fields to skip during deep copying
    _skip_deepcopy_fields = [
        "_loop",
        "_session",
        "_server_context",
        "_session_context",
        "_dynamic_tools",
        "_tools",
        "_resources",
        "_prompts",
        "_connected",
        "add_event_callbacks",
    ]
    _skip_pickle_fields = _skip_deepcopy_fields

    def __init__(
        self,
        name: str | None = None,
        server_command: str | None = None,
        server_args: list[str] | None = None,
        server_env: dict[str, str] | None = None,
        server_url: str | None = None,
        sse_headers: dict[str, Any] | None = None,
        description_modifier: Callable[[str, str | None], str | None] | None = None,
        exclude_tools: list[str] | None = None,
        only_read_only: bool = False,
        timeout: float = 10.0,
    ):
        """Initialize the MCP app and connect to the server immediately.

        :param name: Optional name for the app. Defaults to "MCPApp".
        :param server_command: The command to run the MCP server for stdio connection.
        :param server_args: Arguments to pass to the server command.
        :param server_env: Environment variables to set for the server.
        :param server_url: URL for SSE server connection. If provided, stdio parameters are ignored.
        :param sse_headers: If using an SSE server, you can pass connection headers (httpx headers).
        :param description_modifier: A function that modifies the description returned by the mcp server.
                                    Signature is change_description(tool_name: str, server_description: str | None) -> str | None
        :param exclude_tools: List of tool names to exclude from the app.
        :param only_read_only: If True, only include tools marked as read-only.
        :param timeout: Timeout in seconds for async operations. Defaults to 10.0.
        """
        super().__init__(name=name or "MCPApp")

        # MCP server connection parameters
        self.server_command = server_command
        self.server_args = [] if server_args is None else server_args
        self.server_env = {} if server_env is None else server_env
        self.sse_headers = sse_headers
        self.server_url = server_url
        self.exclude_tools = [] if exclude_tools is None else exclude_tools
        self.only_read_only = only_read_only
        self.timeout = timeout  # Timeout for async operations in seconds

        # Validate that we have either stdio or sse parameters
        if self.server_url is None and self.server_command is None:
            raise ValueError("Either server_command or server_url must be provided")

        if self.server_url is not None and (
            server_args is not None or server_env is not None
        ):
            logger.warning(
                "server_args and server_env are not supported for SSE connection"
            )

        # MCP client session
        self._session = None
        self._tools = {}
        self._resources = {}
        self._prompts = {}

        # Flag to track if we're connected to the server
        self._connected = False

        # Dynamic tool methods that will be created based on MCP server tools
        self._dynamic_tools = {}

        self._loop = asyncio.new_event_loop()

        # Store context managers to keep them alive
        self._server_context = None
        self._session_context = None

        self._descrition_modifier = description_modifier

        # Connect to the server immediately
        self.connect()

    def get_state(self) -> dict[str, Any]:
        """Get the current state of the app for serialization.

        :return: A dictionary containing the app's state.
        :rtype: dict[str, Any]
        """
        # Only save serializable configuration, not connection objects
        return get_state_dict(
            self,
            [
                "server_command",
                "server_args",
                "server_env",
                "timeout",
            ],
        )

    def load_state(self, state_dict: dict[str, Any]):
        """Load the app's state from a dictionary.

        :param state_dict: A dictionary containing the app's state.
        :type state_dict: dict[str, Any]
        """
        # Load the serializable configuration
        for key, value in state_dict.items():
            setattr(self, key, value)

        # Reset connection-related attributes
        self._session = None
        self._tools = {}
        self._resources = {}
        self._prompts = {}
        self._connected = False
        self._dynamic_tools = {}

        # Reconnect to the server to recover tools, resources, and prompts
        self.connect()

    # Define a type variable for the return type of the awaitable
    T = TypeVar("T")

    def _fake_await(self, coro: Any) -> Any:
        """run an async coroutine"""
        return self._loop.run_until_complete(
            coro,
        )

    def connect(self):
        # Temporarily increase timeout for connection as it might take longer
        original_timeout = self.timeout
        self.timeout = 30.0  # Use a longer timeout for connection
        try:
            # Run this synchronously using our persistent event loop with a timeout
            result = self._fake_await(self._connect_to_server())
        finally:
            # Restore original timeout
            self.timeout = original_timeout
        if result:
            logger.info(
                f"Successfully connected to MCP server. Available tools: {', '.join(self._tools.keys())}"
            )
        else:
            logger.error("Failed to connect to MCP server during initialization.")
            raise Exception("Failed to connect to MCP server during initialization.")

    def close(self):
        """Close all connections and clean up resources synchronously.

        This method should be called when the app is no longer needed.
        """
        # Clean up existing connections
        if self._connected:
            try:
                # Run the disconnect method in the event loop with a timeout
                self._fake_await(self._disconnect_from_server())
            except Exception as e:
                logger.error(f"Error during disconnect: {e}")

            try:
                self._loop.close()
            except Exception as e:
                logger.error(f"Error closing event loop: {e}")

        # Reset all attributes
        self._session = None
        self._server_context = None
        self._session_context = None
        self._tools = {}
        self._resources = {}
        self._prompts = {}
        self._connected = False
        self._dynamic_tools = {}

    def reset(self):
        """Reset the app to its initial state."""
        # Don't call super().reset() as it tries to reinitialize with stored args
        # which may not preserve server_command/server_url properly

        # Reset base App attributes to their initial state
        self.is_state_modified = False
        self.add_event_callbacks = {}
        self._tool_registries = {
            ToolType.APP: None,
            ToolType.USER: None,
            ToolType.ENV: None,
            ToolType.DATA: None,
        }

        # Reset the random number generator to initial seed
        self.rng = random.Random(self.seed)

        # Close all connections
        self.close()

        # Create a new event loop
        self._loop = asyncio.new_event_loop()

        # Reconnect to the server
        self.connect()

    @event_registered(operation_type=OperationType.READ)
    def list_resources(self) -> str:
        """List all available resources from the MCP server.

        :return: A formatted string listing all available resources with their descriptions.
        :rtype: str
        """
        if not self._connected:
            return "Not connected to MCP server. Please connect first."

        if not self._resources:
            return "No resources available from the MCP server."

        result = "Available MCP resources:\n\n"
        for name, resource in self._resources.items():
            result += f"- {name}: {resource.description or 'No description'}\n"
            # Check for arguments in the old format
            if hasattr(resource, "arguments") and resource.arguments:  # type: ignore
                result += "  Arguments:\n"
                for arg in resource.arguments:  # type: ignore
                    required = (
                        "(required)"
                        if hasattr(arg, "required") and arg.required
                        else "(optional)"
                    )
                    description = (
                        arg.description
                        if hasattr(arg, "description")
                        else "No description"
                    )
                    result += f"    - {arg.name} {required}: {description}\n"
            result += "\n"

        return result

    @event_registered(operation_type=OperationType.READ)
    def list_prompts(self) -> str:
        """List all available prompts from the MCP server.

        :return: A formatted string listing all available prompts with their descriptions.
        :rtype: str
        """
        if not self._connected:
            return "Not connected to MCP server. Please connect first."

        if not self._prompts:
            return "No prompts available from the MCP server."

        result = "Available MCP prompts:\n\n"
        for name, prompt in self._prompts.items():
            result += f"- {name}: {prompt.description or 'No description'}\n"
            # Check for arguments in the old format
            if hasattr(prompt, "arguments") and prompt.arguments:
                result += "  Arguments:\n"
                for arg in prompt.arguments:
                    required = (
                        "(required)"
                        if hasattr(arg, "required") and arg.required
                        else "(optional)"
                    )
                    description = (
                        arg.description
                        if hasattr(arg, "description")
                        else "No description"
                    )
                    result += f"    - {arg.name} {required}: {description}\n"
            result += "\n"

        return result

    def _call_tool(self, tool_name: str, **kwargs) -> str:
        """Call a tool on the MCP server.

        :param tool_name: The name of the tool to call.
        :type tool_name: str
        :param kwargs: Arguments to pass to the tool.

        :return: The result of the tool call as a string.
        :rtype: str

        :raises ConnectionError: If the connection to the server times out.
        :raises Exception: Any other exception raised during the tool call.
        """
        if not self._connected:
            return "Not connected to MCP server. Please connect first."

        if tool_name not in self._tools:
            return f"Tool '{tool_name}' not found. Available tools: {', '.join(self._tools.keys())}"

        try:
            # Run the tool call with a timeout
            result = self._fake_await(self._call_mcp_tool(tool_name, kwargs))
            return str(result)
        except Exception as e:
            logger.error(f"Error calling MCP tool '{tool_name}'", exc_info=e)
            # Re-raise the exception
            raise e

    @event_registered(operation_type=OperationType.READ)
    def read_resource(self, resource_uri: str) -> str:
        """Read a resource from the MCP server.

        :param resource_uri: The URI of the resource to read.
        :type resource_uri: str

        :return: The content of the resource as a string.
        :rtype: str

        :raises ConnectionError: If the connection to the server times out.
        :raises Exception: Any other exception raised during the resource read.
        """
        if not self._connected:
            return "Not connected to MCP server. Please connect first."

        try:
            # Run the resource read with a timeout
            content, mime_type = self._fake_await(self._read_mcp_resource(resource_uri))
            return f"Resource content ({mime_type}):\n\n{content}"
        except Exception as e:
            logger.error(f"Error reading MCP resource '{resource_uri}': {e}")
            # Return the error as a string since this is a user-facing method
            return f"Error reading MCP resource '{resource_uri}': {str(e)}"

    async def _connect_to_server(self) -> bool:
        """Connect to the MCP server asynchronously.

        :return: True if the connection was successful, False otherwise.
        :rtype: bool
        """
        try:
            # Determine which client to use based on provided configuration
            if self.server_url:
                # Use SSE client with the provided URL
                self._server_context = sse_client(self.server_url, self.sse_headers)
            elif self.server_command:
                # Use stdio client with the provided command parameters
                server_params = StdioServerParameters(
                    command=self.server_command,
                    args=self.server_args,
                    env=self.server_env,
                )
                self._server_context = stdio_client(server_params)
            else:
                # This should never happen due to the validation in __init__
                raise ValueError("Either server_url or server_command must be provided")

            # Connect to the server using async context manager but store it
            # to keep it alive throughout the lifetime of this instance
            (read_stream, write_stream) = await self._server_context.__aenter__()

            # Create a client session and store the context
            self._session = ClientSession(read_stream, write_stream)
            await self._session.__aenter__()

            # Initialize the connection
            await self._session.initialize()

            # Discover server capabilities
            await self._discover_server_capabilities()

            self._connected = True
            return True
        except Exception as e:
            logger.error("Failed to connect to MCP server", exc_info=e)
            # Clean up if connection failed
            self.close()

        return False

    async def _disconnect_from_server(self) -> None:
        """Disconnect from the MCP server asynchronously."""
        # Properly exit the context managers
        if self._session:
            try:
                await self._session.__aexit__(None, None, None)
            except Exception as e:
                logger.error(f"Error closing session: {e}")
            self._session = None

        if self._server_context:
            try:
                await self._server_context.__aexit__(None, None, None)
            except Exception as e:
                logger.error(f"Error closing server context: {e}")
            self._server_context = None

        self._connected = False
        self._tools = {}
        self._resources = {}
        self._prompts = {}

        # Remove dynamic tool methods
        for name in list(self._dynamic_tools.keys()):
            if hasattr(self, name):
                delattr(self, name)
        self._dynamic_tools = {}

    async def _discover_server_capabilities(self) -> None:
        """Discover the capabilities of the connected MCP server."""
        if not self._session:
            return

        # Get available tools
        try:
            tools = await self._session.list_tools()
            tools = tools.tools
            # Filter out excluded tools and non-read-only tools if requested
            filtered_tools = []
            excluded_count = 0
            non_read_only_count = 0

            for tool in tools:
                # Skip excluded tools
                if tool.name in self.exclude_tools:
                    excluded_count += 1
                    continue

                # Skip non-read-only tools if only_read_only is True
                if self.only_read_only:
                    is_read_only = False
                    # Check if tool has annotations attribute
                    if hasattr(tool, "annotations"):
                        annotations = getattr(tool, "annotations")
                        if annotations is not None and hasattr(
                            annotations, "readOnlyHint"
                        ):
                            read_only_hint = getattr(annotations, "readOnlyHint")
                            if read_only_hint is not None:
                                is_read_only = read_only_hint

                    if not is_read_only:
                        non_read_only_count += 1
                        continue

                filtered_tools.append(tool)

            self._tools = {tool.name: tool for tool in filtered_tools}

            # Log filtering results
            if excluded_count > 0:
                logger.debug(f"Excluded {excluded_count} tools by name.")

            if non_read_only_count > 0:
                logger.debug(f"Excluded {non_read_only_count} non-read-only tools.")

            logger.debug(f"Discovered {len(self._tools)} MCP tools after filtering.")
        except Exception as e:
            logger.warning(f"Failed to list MCP tools: {e}", exc_info=e)
            self._tools = {}

        # Get available resources
        try:
            resources = await self._session.list_resources()
            resources = resources.resources if hasattr(resources, "resources") else []
            self._resources = {resource.name: resource for resource in resources}
            logger.debug(f"Discovered {len(resources)} MCP resources")
        except Exception as e:
            logger.warning(f"Failed to list MCP resources: {e}", exc_info=e)
            self._resources = {}

        # Get available prompts
        try:
            prompts = await self._session.list_prompts()
            prompts = prompts.prompts if hasattr(prompts, "prompts") else []
            self._prompts = {prompt.name: prompt for prompt in prompts}
            logger.debug(f"Discovered {len(prompts)} MCP prompts")
        except Exception as e:
            logger.warning(f"Failed to list MCP prompts: {e}", exc_info=e)
            self._prompts = {}

    async def _call_mcp_tool(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        """Call a tool on the MCP server asynchronously.

        :param tool_name: The name of the tool to call.
        :type tool_name: str
        :param arguments: Arguments to pass to the tool.
        :type arguments: dict[str, Any]

        :return: The result of the tool call.
        :rtype: Any

        :raises RuntimeError: If not connected to MCP server.
        """
        if not self._session:
            raise RuntimeError("Not connected to MCP server")

        result = await self._session.call_tool(tool_name, arguments=arguments)
        return result

    async def _read_mcp_resource(self, resource_uri: str) -> tuple[Any, Any]:
        """Read a resource from the MCP server asynchronously.

        :param resource_uri: The URI of the resource to read.
        :type resource_uri: str

        :return: A tuple containing the content and MIME type of the resource.
        :rtype: tuple[Any, Any]

        :raises RuntimeError: If not connected to MCP server.
        """
        if not self._session:
            raise RuntimeError("Not connected to MCP server")

        content, mime_type = await self._session.read_resource(AnyUrl(resource_uri))
        return content, mime_type

    def get_tools_with_attribute(
        self,
        attribute: ToolAttributeName,
        tool_type: ToolType,
    ) -> list[AppTool]:
        """Override the default implementation to directly use the MCP tools list.

        This eliminates the need for dynamic method creation and discovery.

        :param attribute: The attribute to look for in the tools as a ToolAttributeName enum value.
        :type attribute: ToolAttributeName
        :param tool_type: The type of tool being registered (e.g., "APP", "USER", "ENV").
        :type tool_type: ToolType

        :return: A list of AppTool objects created from the MCP tools.
        :rtype: list[AppTool]
        """
        # Get the string value from the enum
        attr_name = attribute.value

        if attr_name != ToolAttributeName.APP.value:
            return []

        # If not connected or no tools, return empty list
        if not self._connected or not self._tools:
            return []

        tools = []

        # Convert MCP tools to AppTools
        for name, mcp_tool in self._tools.items():
            # Create function description from tool description
            function_description = (
                self._descrition_modifier(name, mcp_tool.description)
                if self._descrition_modifier
                else mcp_tool.description
            )
            function_description = function_description or "No description"

            # Create args from inputSchema
            args = []
            return_type = "Any"
            return_description = "No description"
            if hasattr(mcp_tool, "inputSchema") and mcp_tool.inputSchema:
                properties = mcp_tool.inputSchema.get("properties", {})
                required = mcp_tool.inputSchema.get("required", [])

                for prop_name, prop_info in properties.items():
                    arg_type = prop_info.get("type", "Any")
                    description = prop_info.get("description", "No description")
                    has_default = prop_name not in required

                    args.append(
                        AppToolArg(
                            name=prop_name,
                            arg_type=arg_type,
                            description=description,
                            has_default=has_default,
                        )
                    )

            # Determine write_operation based on annotations
            write_operation = None
            annotations_info = []  # Initialize outside the if block

            if hasattr(mcp_tool, "annotations"):
                annotations = getattr(mcp_tool, "annotations")
                if annotations is not None:
                    # Check for readOnlyHint
                    if hasattr(annotations, "readOnlyHint"):
                        read_only_hint = getattr(annotations, "readOnlyHint")
                        if read_only_hint is not None:
                            write_operation = not read_only_hint
                            read_only = "Yes" if read_only_hint else "No"
                            annotations_info.append(
                                f"""Read-only: {read_only}
  If true, the tool does not modify its environment."""
                            )

                    # Check for title
                    if hasattr(annotations, "title"):
                        title = getattr(annotations, "title")
                        if title is not None:
                            annotations_info.append(
                                f"""Title: {title}
  A human-readable title for the tool."""
                            )

                    # Check for destructiveHint
                    if hasattr(annotations, "destructiveHint"):
                        destructive_hint = getattr(annotations, "destructiveHint")
                        if destructive_hint is not None:
                            destructive = "Yes" if destructive_hint else "No"
                            annotations_info.append(
                                f"""Destructive: {destructive}
  If true, the tool may perform destructive updates to its environment.
  If false, the tool performs only additive updates.
  (This property is meaningful only when Read-only is false)"""
                            )

                    # Check for idempotentHint
                    if hasattr(annotations, "idempotentHint"):
                        idempotent_hint = getattr(annotations, "idempotentHint")
                        if idempotent_hint is not None:
                            idempotent = "Yes" if idempotent_hint else "No"
                            annotations_info.append(
                                f"""Idempotent: {idempotent}
  If true, calling the tool repeatedly with the same arguments
  will have no additional effect on its environment.
  (This property is meaningful only when Read-only is false)"""
                            )

                    # Check for openWorldHint
                    if hasattr(annotations, "openWorldHint"):
                        open_world_hint = getattr(annotations, "openWorldHint")
                        if open_world_hint is not None:
                            open_world = "Yes" if open_world_hint else "No"
                            annotations_info.append(
                                f"""Open World: {open_world}
  If true, this tool may interact with an "open world" of external entities.
  If false, the tool's domain of interaction is closed.
  For example, the world of a web search tool is open, whereas that
  of a memory tool is not."""
                            )

            # Create a callable function that will call the MCP tool. A tool the
            # server marks readOnlyHint=False is registered as a WRITE, so its
            # calls reach validation like any other app's write actions.
            def create_callable(tool_name, operation_type):
                @event_registered(operation_type=operation_type)
                def call_mcp_tool(self_instance, **kwargs):
                    return self_instance._call_tool(tool_name, **kwargs)

                return call_mcp_tool

            callable_func = create_callable(
                name,
                OperationType.WRITE if write_operation else OperationType.READ,
            )

            # Append annotations to function description (moved outside the if block)
            if annotations_info:
                props_header = "\n\nTool Properties:\n"
                props_content = "\n".join(annotations_info)
                function_description += props_header + props_content

            # Create the AppTool
            app_tool = AppTool(
                class_name=self.__class__.__name__,
                app_name=self.name,
                name=f"{self.name}__{name}",
                function_description=function_description,
                args=args,
                class_instance=self,
                function=callable_func,
                failure_probability=self.failure_probability,
                return_type=return_type,
                return_description=return_description,
                write_operation=write_operation,
            )

            tools.append(app_tool)

        # Add list_prompts as AppTool
        tools.append(
            AppTool(
                class_name=self.__class__.__name__,
                app_name=self.name,
                name=f"{self.name}__list_prompts",
                # doc from https://modelcontextprotocol.io/docs/concepts/prompts
                function_description="""List all available prompts from the MCP server.
                Prompts enable {self.name} to define reusable prompt templates and workflows that the agent can easily surface to users and LLMs.
                They provide a powerful way to standardize and share common LLM interactions.
                Prompts in are predefined templates that can:
Accept dynamic arguments
Include context from resources
Chain multiple interactions
Guide specific workflows
Surface as UI elements (like slash commands)
                """,
                args=[],
                class_instance=self,
                function=self.list_prompts,
                failure_probability=self.failure_probability,
                return_type="str",
                return_description="A formatted string listing all available prompts with their descriptions.",
            )
        )

        # Add list_resources as AppTool
        tools.append(
            AppTool(
                class_name=self.__class__.__name__,
                app_name=self.name,
                name=f"{self.name}__list_resources",
                # doc from https://modelcontextprotocol.io/docs/concepts/resources
                function_description="""List all available resources that can be accessed from {self.name}.
                Resources represent any kind of data that an MCP server wants to make available to clients. This can include:

File contents
Database records
API responses
Live system data
Screenshots and images
Log files
And more
Each resource is identified by a unique URI and can contain either text or binary data. Resources can be accessed using the {self.name}__read_resource tool.

Resource types
Resources can contain two types of content:

Text resources
Text resources contain UTF-8 encoded text data. These are suitable for:

Source code
Configuration files
Log files
JSON/XML data
Plain text

Binary resources
Binary resources contain raw binary data encoded in base64. These are suitable for:

Images
PDFs
Audio files
Video files
Other non-text formats
""",
                args=[],
                class_instance=self,
                function=self.list_resources,
                failure_probability=self.failure_probability,
                return_type="str",
                return_description="A formatted string listing all available resources with their descriptions.",
            )
        )

        # Add read_resource as AppTool
        tools.append(
            AppTool(
                class_name=self.__class__.__name__,
                app_name=self.name,
                name=f"{self.name}__read_resource",
                function_description="Read a resource from the {self.name} app.",
                args=[
                    AppToolArg(
                        name="resource_uri",
                        arg_type="str",
                        description="The URI of the resource to read.",
                        has_default=False,
                    )
                ],
                class_instance=self,
                function=self.read_resource,
                failure_probability=self.failure_probability,
                return_type="str",
                return_description="The content of the resource as a string.",
            )
        )

        logger.debug(
            f"Created {len(tools)} AppTools directly from MCP tools and additional methods"
        )
        return tools

    def __del__(self):
        """Clean up resources when the instance is garbage collected."""
        try:
            # Use the close method to properly clean up resources
            self.close()
        except Exception:
            # We can't log here as the logger might be gone already
            pass
