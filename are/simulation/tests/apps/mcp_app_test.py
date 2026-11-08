# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


import unittest

from are.simulation.apps.app import ToolType
from are.simulation.apps.mcp.mcp_app import MCPApp
from are.simulation.config import ARE_SIMULATION_ROOT
from are.simulation.tool_utils import ToolAttributeName
from are.simulation.types import OperationType
from are.simulation.utils import time_limit

EXTRA_DESCRIPTION = " (note: this tool does not work with negative numbers)."
SERVER_SCRIPT_PATH = str(
    ARE_SIMULATION_ROOT
    / "simulation"
    / "scenarios"
    / "scenario_mcp_demo"
    / "math_server.py"
)
TODO_SERVER_SCRIPT_PATH = str(
    ARE_SIMULATION_ROOT
    / "simulation"
    / "scenarios"
    / "scenario_mcp_demo"
    / "todo_server.py"
)


def desc_modifier(name, description):
    if name == "divide":
        return f"{description}{EXTRA_DESCRIPTION}"
    return description


class MCPAppTest(unittest.TestCase):
    def test_mcp_app_tools_and_description_modifier(self):
        with time_limit(20):
            """Test that MCPApp correctly lists tools and applies the description modifier."""
            # Create the MCPApp instance with the description modifier.
            mcp_app = MCPApp(
                name="TestApp",
                server_command="python",
                server_args=[SERVER_SCRIPT_PATH],
                description_modifier=desc_modifier,
            )

            try:
                # Get the tools from the app
                tools = mcp_app.get_tools_with_attribute(
                    ToolAttributeName.APP, ToolType.APP
                )

                # Create a dictionary of tools by name for easier lookup.
                tools_dict = {tool.name: tool for tool in tools}

                # Check that all expected tools are present
                # Note: We're now using math_server.py which only provides math tools
                expected_tools = {
                    "TestApp__add",
                    "TestApp__subtract",
                    "TestApp__multiply",
                    "TestApp__divide",
                    "TestApp__square_root",
                    "TestApp__list_prompts",
                    "TestApp__list_resources",
                    "TestApp__read_resource",
                }
                self.assertEqual(tools_dict.keys(), expected_tools)

                # Check that the description modifier was applied to the divide tool.
                divide_tool = tools_dict["TestApp__divide"]
                self.assertIsNotNone(divide_tool.function_description)
                self.assertIn(EXTRA_DESCRIPTION, divide_tool.function_description)  # type: ignore

                # Check that annotations are included in the description
                self.assertIsNotNone(divide_tool.function_description)
                description = divide_tool.function_description or ""
                self.assertIn("Tool Properties:", description)
                self.assertIn("Title: Division Tool", description)
                self.assertIn("Read-only: Yes", description)

                # Check that the description modifier was not applied to other tools.
                add_tool = tools_dict["TestApp__add"]
                self.assertIsNotNone(add_tool.function_description)
                description = add_tool.function_description or ""
                self.assertIn("Add two numbers.", description)
                # Check that annotations are included in the description
                self.assertIn("Tool Properties:", description)
                self.assertIn("Title: Addition Tool", description)
            finally:
                # Clean up resources.
                mcp_app.close()

    def test_mcp_app_without_description_modifier(self):
        """Test that MCPApp works correctly without a description modifier."""
        # Create the MCPApp instance without a description modifier.
        with time_limit(20):
            mcp_app = MCPApp(
                name="TestApp",
                server_command="python",
                server_args=[SERVER_SCRIPT_PATH],
            )

            try:
                # Get the tools from the app.
                tools = mcp_app.get_tools_with_attribute(
                    ToolAttributeName.APP, ToolType.APP
                )

                # Create a dictionary of tools by name for easier lookup.
                tools_dict = {tool.name: tool for tool in tools}

                # Check that the description was not modified for the divide tool.
                self.assertIn("TestApp__divide", tools_dict)
                divide_tool = tools_dict["TestApp__divide"]
                self.assertIsNotNone(divide_tool.function_description)
                description = divide_tool.function_description or ""
                self.assertIn("Divide a by b.", description)

                # Check that annotations are included in the description
                self.assertIn("Tool Properties:", description)
                self.assertIn("Title: Division Tool", description)
                self.assertIn("Read-only: Yes", description)
                self.assertIn("Idempotent: Yes", description)
                self.assertIn("Open World: No", description)
            finally:
                # Clean up resources
                mcp_app.close()

    def test_mcp_app_registers_write_tools_as_write_events(self):
        """A tool with readOnlyHint=False is a WRITE; read-only tools stay READ."""
        with time_limit(20):
            mcp_app = MCPApp(
                name="TodoApp",
                server_command="python",
                server_args=[TODO_SERVER_SCRIPT_PATH],
            )

            try:
                tools = mcp_app.get_tools_with_attribute(
                    ToolAttributeName.APP, ToolType.APP
                )
                tools_dict = {tool.name: tool for tool in tools}

                # todo_server.py marks add_todo readOnlyHint=False.
                add_todo = tools_dict["TodoApp__add_todo"]
                self.assertTrue(add_todo.write_operation)
                self.assertEqual(
                    add_todo.function.__operation_type__,  # type: ignore
                    OperationType.WRITE,
                )

                # ...and list_todos readOnlyHint=True.
                list_todos = tools_dict["TodoApp__list_todos"]
                self.assertFalse(list_todos.write_operation)
                self.assertEqual(
                    list_todos.function.__operation_type__,  # type: ignore
                    OperationType.READ,
                )
            finally:
                mcp_app.close()


if __name__ == "__main__":
    unittest.main()
