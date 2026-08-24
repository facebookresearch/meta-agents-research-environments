# Meta Agents Research Environments MCP Server

## Overview

The Meta Agents Research Environments MCP Server is a tool that allows you to expose any Meta Agents Research Environments app as an MCP (Model Context Protocol) server. This enables any MCP client to interact with Meta Agents Research Environments apps, providing a standardized way to access Meta Agents Research Environments functionality from external systems.

While the [MCPApp](mcp_app_README.md) allows Meta Agents Research Environments to connect to MCP servers, the Meta Agents Research Environments MCP Server does the opposite - it exposes Meta Agents Research Environments apps as MCP servers that can be accessed by any MCP client.

## Features

The Meta Agents Research Environments MCP Server provides the following features:

- Expose any Meta Agents Research Environments app as an MCP server
- Automatically discover and expose all tools from the Meta Agents Research Environments app
- Provide detailed information about the app, its tools and its state through MCP resources
- Handle parameter conversion between Meta Agents Research Environments and MCP
- Support for command-line usage with any Meta Agents Research Environments app
- notify the MCP client when the app state changes

## Installation

The Meta Agents Research Environments MCP Server is included in the Meta Agents Research Environments package. To use it, you need to:

1. Install the MCP Python SDK, this should come with the Meta Agents Research Environments package:
   ```bash
   pip install "mcp[cli]"
   ```

2. Make sure you have the Meta Agents Research Environments package installed:
   ```bash
   pip install -e .
   ```

## Usage

### Command-Line Usage

You can run the Meta Agents Research Environments MCP Server from the command line to expose any Meta Agents Research Environments app as an MCP server:

```bash
python are_simulation_mcp_server.py --app <app_class> [--name <server_name>] [--transport sse|stdio]
```

Where:
- `<app_class>` is the fully qualified path to the Meta Agents Research Environments app class (e.g., `are.simulation.apps.calendar.CalendarApp`)
- `<server_name>` is an optional name for the MCP server (defaults to the app's name + " MCP Server")
- `sse|stdio` choose what MCP transport to use

Example:
```bash
python are_simulation_mcp_server.py --app are.simulation.apps.calendar.CalendarApp --name "Calendar MCP Server"
```

### Explorer

You can use the MCP Explorer to interact with the Meta Agents Research Environments MCP Server. To do this, you need to use npx. For instance,
you can load the mz_dinner scenario like this:
```bash
npx @modelcontextprotocol/inspector are_simulation_mcp_server.py --scenario scenarion_mz_dinner
```

### Programmatic Usage

You can also use the Meta Agents Research Environments MCP Server programmatically in your Python code:

```python
from are.simulation.apps.calendar import CalendarApp
from are_simulation_mcp_server import Meta Agents Research EnvironmentsMCPServer

# Create an instance of the Meta Agents Research Environments app
app = CalendarApp()

# Create the Meta Agents Research Environments MCP server
server = Meta Agents Research EnvironmentsMCPServer(app, server_name="Calendar MCP Server")

# Run the server
server.run()
```

### Connecting to the Server

Once the Meta Agents Research Environments MCP Server is running, you can connect to it using any MCP client. Here's an example using the MCP Python SDK:

```python
import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def main():
    # Create server parameters for stdio connection
    server_params = StdioServerParameters(
        command="python",
        args=[
            "are_simulation_mcp_server.py",
            "--app",
            "are.simulation.apps.calendar.CalendarApp",
        ],
        env=None,
    )

    # Connect to the server
    read_stream, write_stream = await stdio_client(server_params)

    # Create a client session
    session = ClientSession(read_stream, write_stream)

    # Initialize the connection
    await session.initialize()

    # List available tools
    tools = await session.list_tools()
    for tool in tools:
        print(f"- {tool.name}: {tool.description}")

    # Call a tool
    result = await session.call_tool(
        "add_calendar_event",
        arguments={
            "title": "Team Meeting",
            "start_datetime": "2023-06-15 10:00:00",
            "end_datetime": "2023-06-15 11:00:00",
        },
    )
    print(f"Result: {result}")

    # Close the session
    await session.close()


if __name__ == "__main__":
    asyncio.run(main())
```

## Available Resources

The Meta Agents Research Environments MCP Server exposes the following MCP resources:

- `app://info`: Provides general information about the Meta Agents Research Environments app, including its name and available tools.
- `app://tools`: Provides detailed information about the tools exposed by the Meta Agents Research Environments app, including their parameters.
- `app://{app_name}/state`: Provides information about the current state of the Meta Agents Research Environments app.
- `app://{app_name}/info`: Provides general information about the Meta Agents Research Environments app, including its name and available tools.
