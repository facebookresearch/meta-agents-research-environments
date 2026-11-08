# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

#!/usr/bin/env python3
"""
Pytest tests for the Meta Agents Research Environments MCP Server.

These tests verify that:
1. The right app is initialized and its tools are exposed on the MCP server.
2. A scenario can be loaded.
3. The state exposed to the resources corresponds to the app, and write/reads work correctly.
"""

import json

import pytest
from mcp.types import TextContent

from are.simulation.apps.calendar import CalendarApp, CalendarEvent
from are.simulation.apps.contacts import Contact, ContactsApp
from are.simulation.apps.mcp.server.are_simulation_mcp_server import (
    ARESimulationMCPServer,
)
from are.simulation.scenarios.scenario import Scenario
from are.simulation.utils import make_serializable


class MockScenario(Scenario):
    """A simple test scenario for testing the Meta Agents Research Environments MCP Server."""

    scenario_id: str = "test_scenario"

    def __init__(self):
        self.apps = None

    def init_and_populate_apps(self, *args, **kwargs) -> None:
        """Initialize and populate the apps for the test scenario."""
        # Create instances of the apps we want to include
        calendar_app = CalendarApp()
        contacts_app = ContactsApp()

        # Add sample data to the calendar app
        calendar_app.add_calendar_event(
            title="Team Meeting",
            start_datetime="2023-07-15 09:00:00",
            end_datetime="2023-07-15 10:00:00",
            tag="Work",
            description="Weekly team sync",
            location="Conference Room A",
            attendees=["Alice", "Bob", "Charlie"],
        )

        calendar_app.add_calendar_event(
            title="Lunch with Client",
            start_datetime="2023-07-15 12:00:00",
            end_datetime="2023-07-15 13:30:00",
            tag="Work",
            description="Discuss project requirements",
            location="Cafe Downtown",
            attendees=["David", "Emma"],
        )

        # Add sample data to the contacts app
        contacts_app.add_contact(
            Contact(
                first_name="Alice",
                last_name="Smith",
                email="alice@example.com",
                phone="555-1234",
                address="123 Main St",
            )
        )

        contacts_app.add_contact(
            Contact(
                first_name="Bob",
                last_name="Johnson",
                email="bob@example.com",
                phone="555-5678",
                address="456 Oak Ave",
            )
        )

        # Set the apps for the scenario
        self.apps = [calendar_app, contacts_app]


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
def calendar_app():
    """
    Fixture that returns just the CalendarApp instance.
    """
    return CalendarApp()


@pytest.fixture
def contacts_app():
    return ContactsApp()


@pytest.mark.anyio
async def test_app_initialization_and_tools_exposure(calendar_app, contacts_app):
    """
    Test that the right apps are initialized and their tools are exposed on the MCP server.
    """
    # These are our expected apps that we'll use to verify server behavior
    expected_apps = [calendar_app, contacts_app]

    # Directly instantiate the ARESimulationMCPServer
    server = ARESimulationMCPServer(apps=expected_apps)
    mcp = server.mcp

    # Get tools using the FastMCP list_tools method (actual server response)
    actual_mcp_tools = await mcp.list_tools()
    actual_mcp_tool_names = [tool.name for tool in actual_mcp_tools]

    # Check that we have tools
    assert len(actual_mcp_tools) > 0, "No tools were exposed by the MCP server"

    # Get app info resource using read_resource (actual server response)
    actual_app_info_content = await mcp.read_resource("app://info")
    actual_app_info = json.loads(next(iter(actual_app_info_content)).content)

    # Build a map of app name to app info for easier lookup
    actual_app_info_map = {app["name"]: app for app in actual_app_info["apps"]}

    # Check that the app info contains all our expected apps
    assert len(actual_app_info["apps"]) == len(expected_apps), "App count mismatch"

    all_expected_tool_names = set()

    # Check each expected app's tools
    for expected_app in expected_apps:
        # Get the expected app's tools directly
        expected_app_tools = expected_app.get_tools()
        expected_app_tool_names = [tool.name for tool in expected_app_tools]

        # Check that the app info contains this expected app
        expected_app_name = expected_app.app_name()
        assert expected_app_name in actual_app_info_map, (
            f"{expected_app_name} not found in app info"
        )

        # Check that the tool count matches
        actual_app_info_item = actual_app_info_map[expected_app_name]
        assert actual_app_info_item["tool_count"] == len(expected_app_tools), (
            f"Tool count mismatch for {expected_app_name}"
        )

        for expected_app_tool_name in expected_app_tool_names:
            all_expected_tool_names.add(expected_app_tool_name)

        # Check that app state resource is available
        expected_app_state_resource_path = f"app://{expected_app.app_name()}/state"

        # Get app state using read_resource (actual server response)
        actual_app_state_response = await mcp.read_resource(
            expected_app_state_resource_path
        )
        actual_app_state_content = next(iter(actual_app_state_response)).content
        actual_app_state = json.loads(actual_app_state_content)

        # Verify the actual app state matches what we expect from the app
        expected_app_state = make_serializable(expected_app.get_state())
        assert actual_app_state == expected_app_state, (
            f"App state mismatch for {expected_app.app_name()}"
        )

    # Check that all expected app tools are exposed on the MCP server
    assert all_expected_tool_names == set(actual_mcp_tool_names)


@pytest.mark.anyio
async def test_tool_input_schemas_are_valid_json_schema(calendar_app, contacts_app):
    """Every advertised inputSchema is valid JSON Schema, with JSON types."""
    from jsonschema import Draft202012Validator

    server = ARESimulationMCPServer(apps=[calendar_app, contacts_app])
    tools = {tool.name: tool for tool in await server.mcp.list_tools()}

    for tool in tools.values():
        Draft202012Validator.check_schema(tool.inputSchema)

    properties = tools["CalendarApp__add_calendar_event"].inputSchema["properties"]
    # `title: str` stays a string; `start_datetime: str | None` allows null.
    assert properties["title"]["type"] == "string"
    assert {"type": "null"} in properties["start_datetime"]["anyOf"]

    properties = tools["CalendarApp__get_calendar_events_from_to"].inputSchema[
        "properties"
    ]
    # `offset: int` is an integer, not "int".
    assert properties["offset"]["type"] == "integer"


@pytest.mark.anyio
async def test_scenario_loading():
    """
    Test that a scenario can be loaded and that the data in the MCP server
    matches the expected scenario data.
    """
    # Create an instance of our test scenario
    expected_scenario = MockScenario()
    expected_scenario.initialize()
    apps = expected_scenario.apps
    assert apps is not None, "MockScenario should have apps for the test."

    # Directly instantiate the ARESimulationMCPServer with the scenario
    server = ARESimulationMCPServer(scenario=expected_scenario)
    mcp = server.mcp

    # Get tools using the FastMCP list_tools method (actual server response)
    actual_mcp_tools = await mcp.list_tools()

    # Check that we have tools
    assert len(actual_mcp_tools) > 0, "No tools were exposed by the MCP server"

    # Get app info resource using read_resource (actual server response)
    actual_app_info_content = await mcp.read_resource("app://info")
    actual_app_info = json.loads(next(iter(actual_app_info_content)).content)

    # Build a map of app name to app info for easier lookup
    actual_app_info_map = {app["name"]: app for app in actual_app_info["apps"]}

    # Check that the app info contains apps from the scenario
    assert len(actual_app_info["apps"]) > 0, "No apps found in app info"

    # Check that all apps from the scenario are in the app info
    for expected_app in apps:
        # Get the expected app's tools directly
        expected_app_tools = expected_app.get_tools()
        expected_app_name = expected_app.app_name()

        # Check that the app info contains this expected app
        assert expected_app_name in actual_app_info_map, (
            f"{expected_app_name} not found in app info"
        )

        # Check that the tool count matches
        actual_app_info_item = actual_app_info_map[expected_app_name]
        assert actual_app_info_item["tool_count"] == len(expected_app_tools), (
            f"Tool count mismatch for {expected_app_name}"
        )

        # Get the app state from the MCP server
        expected_app_state_resource_path = f"app://{expected_app.app_name()}/state"

        # Get app state using read_resource (actual server response)
        actual_app_state_response = await mcp.read_resource(
            expected_app_state_resource_path
        )
        actual_app_state_content = next(iter(actual_app_state_response)).content

        # Get the expected app state
        expected_app_state = make_serializable(expected_app.get_state())
        actual_app_state = json.loads(actual_app_state_content)

        # For other apps, compare the entire state
        assert actual_app_state == expected_app_state, (
            f"App state mismatch for {expected_app_name}"
        )


@pytest.mark.anyio
async def test_state_update_and_tool_call(calendar_app):
    """
    Test that the state exposed to the resources corresponds to the app,
    and verify write/reads work correctly for both apps.
    """
    # Directly instantiate the ARESimulationMCPServer with the calendar app
    server = ARESimulationMCPServer(apps=[calendar_app])
    mcp = server.mcp

    # Get the initial state using read_resource
    calendar_app_state_resource_path = f"app://{calendar_app.app_name()}/state"

    # Get app state using read_resource (actual server response)
    actual_initial_state_response = await mcp.read_resource(
        calendar_app_state_resource_path
    )
    actual_initial_state_content = next(iter(actual_initial_state_response)).content
    actual_initial_state = json.loads(actual_initial_state_content)

    # Check the initial state structure
    assert "events" in actual_initial_state, "Events not found in initial state"
    initial_event_count = len(actual_initial_state["events"])

    # Add a calendar event directly to the app
    result = await mcp.call_tool(
        "CalendarApp__add_calendar_event",
        {
            "title": "Test Event",
            "start_datetime": "2023-06-15 10:00:00",
            "end_datetime": "2023-06-15 11:00:00",
            "tag": "Work",
            "description": "Test event for pytest",
            "location": "Test Location",
            "attendees": ["Alice", "Bob"],
        },
    )
    result = next(iter(result))
    assert isinstance(result, TextContent)
    expected_event_id = result.text

    # Get the updated state using read_resource (actual server response)
    actual_updated_state_response = await mcp.read_resource(
        calendar_app_state_resource_path
    )
    actual_updated_state_content = next(iter(actual_updated_state_response)).content
    actual_updated_state = json.loads(actual_updated_state_content)
    actual_events = actual_updated_state["events"]

    # Check that the event count increased
    assert len(actual_events) == initial_event_count + 1, (
        "Event count did not increase after adding an event"
    )

    # Find the event in the actual state by title
    found_event = actual_events.get(expected_event_id)
    assert found_event is not None, "Added event not found in updated state"

    # get the event from the raw calendar app
    expected_event = calendar_app.get_calendar_event(expected_event_id)

    # should be the same
    assert CalendarEvent(**found_event) == expected_event, "Event mismatch"
