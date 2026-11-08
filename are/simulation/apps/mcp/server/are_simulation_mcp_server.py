# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

#!/usr/bin/env python3
"""
Meta Agents Research Environments MCP Server - Expose multiple Meta Agents Research Environments apps as an MCP server.

This script creates an MCP server that wraps multiple Meta Agents Research Environments apps, exposing the apps'
tools as MCP tools. This allows any MCP client to interact with Meta Agents Research Environments apps.

Usage:
    python are_simulation_mcp_server.py --apps <app_class1> <app_class2> ... [--name <server_name>] [--transport sse|stdio]
    python are_simulation_mcp_server.py --scenario <scenario_id> [--name <server_name>] [--transport sse|stdio]

Example:
    python are_simulation_mcp_server.py --apps are.simulation.apps.calendar.CalendarApp are.simulation.apps.email.EmailApp --name "Meta Agents Research Environments MCP Server"
    python are_simulation_mcp_server.py --scenario scenario_mz_dinner --transport sse
"""

import importlib
import json
import logging
import sys
from typing import Any, Literal, Sequence, Type

import click
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.tools.base import Tool as MCPTool
from mcp.server.fastmcp.tools.tool_manager import ToolManager
from mcp.server.fastmcp.utilities.func_metadata import ArgModelBase, FuncMetadata
from mcp.types import ToolAnnotations
from pydantic import Field, ValidationInfo, create_model, field_validator
from pydantic_core import PydanticUndefined

from are.simulation.apps.app import App
from are.simulation.scenarios.scenario import Scenario
from are.simulation.tool_utils import AppTool
from are.simulation.utils import make_serializable

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class ARESimulationToolManager(ToolManager):
    """
    A custom ToolManager for Meta Agents Research Environments apps that creates MCP tools from Meta Agents Research Environments AppTool objects.
    The normal tool manager gets confused parsing the signature of the AppTool function and we already
    have all the information we need to create the tool, so we can just create the tool directly.
    """

    def __init__(
        self,
        apps: Sequence[App],
    ):
        """
        Initialize the ARESimulationToolManager.

        :param apps: A sequence of Meta Agents Research Environments apps to expose tools from
        """
        super().__init__(warn_on_duplicate_tools=False)
        self.apps = apps
        self._discover_and_register_tools()

    def _discover_and_register_tools(self):
        """
        Discover all tools in the Meta Agents Research Environments apps and register them as MCP tools.
        """
        # Track tool names to handle potential duplicates
        tool_names = set()

        # Skip if no apps are available
        if not self.apps:
            logger.warning("No apps available to expose tools from")
            return

        for app in self.apps:
            app_name = app.app_name()
            app_tools = app.get_tools()

            for app_tool in app_tools:
                tool_name = app_tool.name

                if tool_name in tool_names:
                    logger.warning(
                        f"Duplicate tool name '{tool_name}' found, skipping registration"
                    )
                    continue

                tool_names.add(tool_name)
                tool_func = app_tool.function
                assert tool_func is not None, (
                    f"{app_name}/{tool_name} doesn't have a callable function."
                )
                tool_doc = app_tool.function_description or "No description available."

                # MCP tool has no concept of write vs read operations, so let's put this info in the tool description
                # to help the agent
                operation_type = (
                    f"This is a write operation that might change the app state or have side effects, check the `app://{app_name}/state` resource to see the changed state."
                    if app_tool.write_operation
                    else "This is a read operation and should not have side effects."
                )
                # Add app name to the description for clarity
                tool_doc = f"[{app_name}] {tool_doc}. {operation_type}"

                # Create an MCP tool that calls the Meta Agents Research Environments app tool
                self._register_app_tool(app, tool_name, app_tool, tool_doc)

                logger.debug(
                    f"Exposed ARE_SIMULATION app tool '{tool_name}' from app '{app_name}' as MCP tool"
                )

    def _register_app_tool(
        self,
        app: App,
        tool_name: str,
        app_tool: AppTool,
        tool_doc: str,
    ) -> MCPTool:
        """
        Register a Meta Agents Research Environments app tool as an MCP tool.

        :param app: The Meta Agents Research Environments app instance
        :param tool_name: The name for the tool
        :param app_tool: The Meta Agents Research Environments AppTool object
        :param tool_doc: The tool's documentation

        :returns: The created Tool object
        """

        # Create a wrapper function that doesn't expose 'self'
        async def wrapper(**kwargs):
            try:
                # get the mcp context to send notifications
                context = kwargs.pop("_mcp_context", None)
                result = app_tool(**kwargs)
                try:
                    if app_tool.write_operation:
                        # send resource updated notification to notify of app state change (we might send too many of these, but it's cheaper than checking the state)
                        await context.session.send_resource_updated(
                            f"app://{app.app_name()}/state"
                        )
                except Exception as e:
                    logger.error(
                        f"Error sending resource updated notification for app {app.app_name()}: {e}"
                    )
                    # ignore this failure, still return the result
                return result
            except Exception as e:
                logger.error(
                    f"Error calling ARE_SIMULATION app tool '{tool_name}': {e}"
                )
                raise

        # Set the wrapper's name and docstring
        wrapper.__name__ = tool_name
        wrapper.__doc__ = tool_doc

        # Create a custom ArgModelBase for the function arguments
        field_definitions = {}
        args_with_none = set()

        for arg in app_tool.args:
            # Use the type_obj directly if available, otherwise fall back to parsing the string
            python_type = arg.type_obj if arg.type_obj is not None else Any

            # Create field with appropriate type and description
            field_info = {}
            if arg.description:
                field_info["description"] = arg.description
            if arg.has_default:
                field_info["default"] = arg.default
            if arg.name == "kwargs":
                # kwargs has an implicit default of empty dict
                field_info["default"] = {}

            if arg.arg_type.endswith("| None"):
                args_with_none.add(arg.name)

            field_definitions[arg.name] = (python_type, Field(**field_info))

        # Define a validator function to handle None values for fields with defaults
        # it looks like the MCP explorer (and maybe some other MCP clients) will set the optional fields to 'null'/None by default
        # but this confuses pydantic, that thinks a None value is being passed instead of having an omitted argument.
        # This validator will replace None values with the default value if it exists, otherwise it will just pass the value through.
        # It will also check if the field type allows None (is optional) before applying the default.
        @field_validator("*", mode="before")
        def handle_none_values(cls, v, info: ValidationInfo):
            if v is None and info.field_name in cls.model_fields:
                field = cls.model_fields[info.field_name]
                # if the arg allows None, we don't default, we just take the None (v)
                if info.field_name in args_with_none:
                    return v

                # else if the field has a default value, return that
                if field.default is not None and field.default is not PydanticUndefined:
                    logger.debug(
                        f"Replacing None value for {info.field_name} with default: {field.default}"
                    )
                    return field.default

            return v

        # Create the argument model with the validator
        arg_model = create_model(
            f"{tool_name}Arguments",
            **field_definitions,
            __base__=ArgModelBase,
            __validators__={"handle_none_values": handle_none_values},
        )

        # Advertise the JSON Schema of the model that validates the call, so
        # Python types become JSON Schema types (int -> integer, str | None ->
        # string or null) instead of being copied into "type" as-is.
        parameters = arg_model.model_json_schema(by_alias=True)

        # Create FuncMetadata with our custom arg_model
        custom_metadata = FuncMetadata(arg_model=arg_model)

        # Create and register the tool
        tool = MCPTool(
            fn=wrapper,
            name=tool_name,
            title=tool_name,
            description=tool_doc,
            parameters=parameters,
            fn_metadata=custom_metadata,
            is_async=True,
            context_kwarg="_mcp_context",
            annotations=ToolAnnotations(
                readOnlyHint=(
                    not app_tool.write_operation
                    if app_tool.write_operation is not None
                    else None
                ),
            ),
        )

        self._tools[tool_name] = tool
        return tool


class ARESimulationMCPServer:
    """
    A class that creates an MCP server wrapping multiple Meta Agents Research Environments apps.

    This class takes multiple Meta Agents Research Environments apps and exposes their tools as MCP tools,
    allowing any MCP client to interact with the apps.
    """

    def __init__(
        self,
        apps: Sequence[App] | None = None,
        server_name: str | None = None,
        scenario: Scenario | None = None,
    ):
        """
        Initialize the Meta Agents Research Environments MCP Server.

        :param apps: A sequence of Meta Agents Research Environments apps to wrap. If scenario is provided, this parameter is ignored.
        :param server_name: Optional name for the MCP server. Defaults to "Meta Agents Research Environments MCP Server".
        :param scenario: Optional scenario to load apps and their states from.
        """
        self.scenario = scenario

        # If a scenario is provided, use its apps
        if scenario:
            # Make sure the scenario is initialized
            if not scenario._initialized:
                scenario.initialize()

            self.apps = scenario.apps if scenario.apps is not None else []
            # Use scenario ID as server name if not provided
            self.server_name = (
                server_name or f"ARE_SIMULATION MCP Server - {scenario.scenario_id}"
            )
        else:
            # Use provided apps
            self.apps = list(apps) if apps is not None else []
            self.server_name = server_name or "ARE_SIMULATION MCP Server"

        # Create an MCP server
        self.mcp = FastMCP(self.server_name)

        # Create a custom tool manager for Meta Agents Research Environments apps and replace the default one
        self.tool_manager = ARESimulationToolManager(
            apps=self.apps,
        )
        self.mcp._tool_manager = self.tool_manager

        # Expose app info as a resource
        self._expose_app_info()

    def _expose_app_info(self):
        """
        Expose information about the Meta Agents Research Environments apps as MCP resources.
        """

        @self.mcp.resource("app://info")
        def get_app_info() -> str:
            """Get information about all Meta Agents Research Environments apps."""
            apps_info = []

            if not self.apps:
                return json.dumps(
                    {
                        "apps": [],
                        "app_count": 0,
                        "total_tool_count": 0,
                    },
                    indent=2,
                )

            for app in self.apps:
                app_name = app.app_name()
                app_tools = app.get_tools()

                apps_info.append(
                    {
                        "name": app_name,
                        "tools": [tool.name for tool in app_tools],
                        "tool_count": len(app_tools),
                    }
                )

            info = {
                "apps": apps_info,
                "app_count": len(self.apps),
                "total_tool_count": (
                    sum(len(app.get_tools()) for app in self.apps) if self.apps else 0
                ),
            }

            return json.dumps(make_serializable(info), indent=2)

        # Add app-specific resources
        for app in self.apps:
            app_name = app.app_name()

            # Create closures with app and app_name bound to the current values
            def create_info_closure(current_app, current_app_name):
                @self.mcp.resource(f"app://{current_app_name}/info")
                def get_app_specific_info() -> str:
                    f"""Get information about a {current_app_name} state and specific tools."""
                    return json.dumps(
                        make_serializable(
                            {
                                "name": current_app_name,
                                "tools": [
                                    tool.name for tool in current_app.get_tools()
                                ],
                            }
                        ),
                        indent=2,
                    )

                return get_app_specific_info

            def create_state_closure(current_app, current_app_name):
                @self.mcp.resource(f"app://{current_app_name}/state")
                def get_app_specific_state() -> str:
                    f"""Get the internal state of app {current_app_name}."""
                    app_state = current_app.get_state() or {}
                    return json.dumps(
                        make_serializable(app_state),
                        indent=2,
                    )

                return get_app_specific_state

            # Create and register the closures with the current app values
            create_info_closure(app, app_name)
            create_state_closure(app, app_name)

    def get_app_by_name(self, app_name: str) -> App | None:
        """
        Get an app by name.

        :param app_name: The name of the app.
        :type app_name: str

        :returns: The app instance or None if not found.
        :rtype: App or None
        """
        for app in self.apps:
            if app.app_name() == app_name:
                return app
        return None

    def run(self, transport: Literal["stdio", "sse"] = "stdio"):
        """
        Run the MCP server.
        """
        # Run the server
        self.mcp.run(transport)


def load_scenario_class(scenario_class_path: str) -> Type[Scenario]:
    """
    Load a Scenario class from its fully qualified path.

    :param scenario_class_path: The fully qualified path to the scenario class (e.g., "are.simulation.scenarios.calendar.CalendarScenario").
    :type scenario_class_path: str

    :returns: The scenario class.
    :rtype: Type[Scenario]

    :raises ImportError: If the scenario class cannot be imported.
    :raises TypeError: If the imported class is not a subclass of Scenario.
    """
    try:
        # Split the path into module path and class name
        s = scenario_class_path.rsplit(".", 1)
        assert len(s) == 2, f"Invalid scenario class path: {scenario_class_path}"
        module_path, class_name = s

        # Import the module
        module = importlib.import_module(module_path)

        # Get the class
        scenario_class = getattr(module, class_name)

        # Verify that it's a subclass of Scenario
        if not issubclass(scenario_class, Scenario):
            raise TypeError(f"{scenario_class_path} is not a subclass of Scenario")

        return scenario_class
    except (ImportError, AttributeError) as e:
        raise ImportError(f"Failed to import scenario class {scenario_class_path}: {e}")


def load_app_class(app_class_path: str) -> Type[App]:
    """
    Load a Meta Agents Research Environments app class from its fully qualified path.

    :param app_class_path: The fully qualified path to the app class (e.g., "are.simulation.apps.calendar.CalendarApp").
    :type app_class_path: str

    :returns: The app class.
    :rtype: Type[App]

    :raises ImportError: If the app class cannot be imported.
    :raises TypeError: If the imported class is not a subclass of App.
    """
    try:
        # Split the path into module path and class name
        module_path, class_name = app_class_path.rsplit(".", 1)

        # Import the module
        module = importlib.import_module(module_path)

        # Get the class
        app_class = getattr(module, class_name)

        # Verify that it's a subclass of App
        if not issubclass(app_class, App):
            raise TypeError(f"{app_class_path} is not a subclass of App")

        return app_class
    except (ImportError, AttributeError) as e:
        raise ImportError(f"Failed to import app class {app_class_path}: {e}")


@click.command(help="Create an MCP server that wraps multiple ARE_SIMULATION apps.")
@click.option(
    "--apps",
    multiple=True,
    help="Fully qualified paths to the ARE_SIMULATION app classes (e.g., 'are.simulation.apps.calendar.CalendarApp')",
)
@click.option("--name", help="Name for the MCP server")
@click.option(
    "--scenario",
    help="Fully qualified path to a scenario class to load apps and their states from",
)
@click.option(
    "--transport",
    type=click.Choice(["stdio", "sse"]),
    default="stdio",
    help="Transport to use for the MCP server (stdio or sse). Default is stdio.",
)
def main(apps, name, scenario, transport):
    """
    Main entry point for the script.
    """
    from are.simulation.scenarios.utils.constants import ALL_SCENARIOS

    try:
        scenario_obj = None
        app_list = []

        # Check if a scenario is provided
        if scenario:
            if apps:
                raise click.UsageError("Can't specify --scenario and --apps together")

            if scenario in ALL_SCENARIOS:
                scenario_obj = ALL_SCENARIOS[scenario]()
            else:
                # Load the scenario class
                scenario_class = load_scenario_class(scenario)

                # Create an instance of the scenario
                scenario_obj = scenario_class()

            # Initialize the scenario
            scenario_obj.initialize()

            logger.info(
                f"Loaded scenario '{scenario_obj.scenario_id}' from '{scenario}'"
            )
        elif apps:
            # Load each app class and create instances
            for app_class_path in apps:
                # Load the app class
                app_class = load_app_class(app_class_path)

                # Create an instance of the app
                app = app_class()
                app_list.append(app)

                logger.info(
                    f"Loaded ARE_SIMULATION app '{app.app_name()}' from '{app_class_path}'"
                )
        else:
            raise click.UsageError("Either --apps or --scenario must be provided")

        # Create the Meta Agents Research Environments MCP server with apps or scenario
        server = ARESimulationMCPServer(
            apps=app_list, server_name=name, scenario=scenario_obj
        )

        # Run the server
        if scenario_obj:
            logger.info(
                f"Starting MCP server for scenario '{scenario_obj.scenario_id}'"
            )
        else:
            app_names = ", ".join(f"'{app.app_name()}'" for app in app_list)
            logger.info(f"Starting MCP server for ARE_SIMULATION apps: {app_names}")

        server.run(transport)
    except Exception as e:
        logger.error("Error starting ARE_SIMULATION MCP server", exc_info=e)
        sys.exit(1)


if __name__ == "__main__":
    main()
