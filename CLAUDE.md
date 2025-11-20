# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Meta Agents Research Environments (ARE) is a research platform for evaluating AI agents in dynamic, realistic scenarios. The platform runs the Gaia2 benchmark with 800 scenarios across multiple domains, testing agents' abilities to adapt and reason through complex, multi-step tasks.

**Core Architecture Pattern**: The platform uses a simulation-based approach where:
- **Environments** manage the event loop, time management, and app orchestration
- **Apps** provide interactive applications (email, calendar, filesystem, messaging) with APIs
- **Events** drive dynamic evolution of scenarios over time
- **Scenarios** combine apps, events, and validation logic into complete tasks
- **Agents** interact using the ReAct framework (Reasoning + Acting)

## Development Commands

### Package Management
```bash
# Install dependencies (uses uv)
uv sync

# Install with GUI support
uv sync --extra gui

# Install dev dependencies
uv sync --extra dev
```

### Code Quality

```bash
# Linting
uvx ruff check .
uvx ruff check --fix .

# Formatting
uvx ruff format .
uvx ruff format --check .

# Import sorting
uvx ruff check --select I --fix .

# Type checking
uv run --extra dev pyright

# Run all pre-commit hooks
pre-commit run --all-files
```

### Testing

```bash
# Run all tests
uv run pytest

# Run specific test file
uv run pytest are/simulation/tests/path/to/test_file.py

# Run specific test function
uv run pytest are/simulation/tests/path/to/test_file.py::test_function_name

# Run with verbose output
uv run pytest -v

# Run with coverage
uv run pytest --cov=are.simulation
```

### Running Scenarios

```bash
# Run a single scenario
uv run are-run -s scenario_tutorial -a default

# Run with GUI
uv run are-gui -s scenario_tutorial

# Run Gaia2 benchmark validation set
uv run are-benchmark gaia2-run --hf meta-agents-research-environments/gaia2 --hf_split validation -l 5

# Run benchmark with custom dataset
uv run are-benchmark run -d /path/to/scenarios --agent default --limit 10
```

## Architecture & Key Concepts

### Event System

The platform's core is an event-driven architecture where events can be:
- **AGENT events**: Initiated by agent actions (tool calls)
- **ENV events**: Scheduled by scenario designers to simulate environment changes
- **ORACLE events**: Only processed in oracle mode for testing/validation
- **VALIDATION events**: Check scenario completion and correctness
- **CONDITION events**: Trigger other events based on state conditions

Events use a dependency chain system:
```python
event1 = messaging.add_message(...).depends_on(None, delay_seconds=5)
event2 = aui.send_message_to_agent(...).depends_on(event1, delay_seconds=1)
```

### Apps and Tools

Apps expose their functionality through the `@app_tool` decorator, which:
- Automatically generates tool descriptions for agents
- Handles argument validation and type checking
- Supports operation types (READ, WRITE, DELETE) for tracking
- Enables tool augmentation (e.g., simulated failures)

```python
@app_tool(name="send_email", description="Send an email")
def send_email(self, to: str, subject: str, body: str) -> str:
    # Implementation
```

### Scenario Development Pattern

Scenarios inherit from `Scenario` class and follow this structure:

1. **init_and_populate_apps()**: Initialize apps and populate with test data
2. **build_events_flow()**: Define event dependencies using capture_mode
3. **validation()**: Implement validation logic for scenario completion
4. **get_user_prompt()**: Return the initial task prompt for the agent

```python
@register_scenario("my_scenario")
class MyScenario(Scenario):
    start_time: float | None = 0
    duration: float | None = 60

    def init_and_populate_apps(self, *args, **kwargs) -> None:
        # Initialize apps
        self.apps = [agui, email, calendar, ...]

    def build_events_flow(self) -> None:
        with EventRegisterer.capture_mode():
            # Define event chains
            event1 = app.method(...).depends_on(None, delay_seconds=5)

    def validation(self) -> ScenarioValidationResult:
        # Check completion criteria
        return ScenarioValidationResult(passed=True, feedback="Success")
```

### Agent Architecture

The default agent (`are/simulation/agents/default_agent/`) uses a ReAct loop:
1. Receives task and environment state
2. Generates thought and action (via LLM)
3. Executes tool calls
4. Observes results
5. Repeats until task completion or max turns

**Key components**:
- `base_agent.py`: Core agent loop and state management
- `tools/action_executor.py`: Parses and executes agent actions
- `llm/`: LLM engine abstractions (LiteLLM, OpenAI)
- Agent logs track: SystemPrompt, Task, LLMOutput, Observation, Error, Stop

### Model Configuration

The platform uses LiteLLM for model provider abstraction. Supported providers are defined in `config.PROVIDERS`. Common patterns:

```bash
# Llama API
export LLAMA_API_KEY="your-key"
uv run are-benchmark run --model Llama-3.1-70B-Instruct --provider llama-api

# Local deployment
uv run are-benchmark run --model local-model --provider local --endpoint http://localhost:8000

# HuggingFace
export HF_TOKEN="your-token"
uv run are-benchmark run --model meta-llama/Meta-Llama-3.3-70B-Instruct --provider huggingface
```

### Validation System

Scenarios are validated using:
- **Hard validation**: Exact matches, type checks, structured comparisons
- **Soft validation**: LLM-based semantic evaluation (using judge models)
- **Tool checkers**: Validate specific tool call arguments

Default judge model: `meta-llama/Meta-Llama-3.3-70B-Instruct`

Configure judge in validation:
```python
from are.simulation.validation.configs import create_judge_engine, LLMEngineConfig

judge_engine = create_judge_engine(
    LLMEngineConfig(model_name="your-model", provider="your-provider")
)
```

### MCP (Model Context Protocol) Integration

The platform supports MCP servers through `MCPApp`:
- Apps can expose MCP servers for external tool integration
- Scenarios can connect to multiple MCP servers
- See `are/simulation/apps/mcp/` for implementation details

## File Structure

```
are/simulation/
├── agents/                    # Agent implementations
│   ├── default_agent/         # Default ReAct agent
│   └── llm/                   # LLM engine abstractions
├── apps/                      # Application implementations
│   ├── email_client.py        # Email app
│   ├── calendar.py            # Calendar app
│   ├── messaging.py           # Messaging app
│   ├── sandbox_file_system.py # Filesystem app
│   ├── contacts.py            # Contacts app
│   └── mcp/                   # MCP integration
├── scenarios/                 # Scenario implementations
│   └── scenario_tutorial/     # Example scenario
├── validation/                # Validation framework
│   ├── configs.py             # Judge configuration
│   └── utils/                 # Validation helpers
├── benchmark/                 # Benchmarking system
│   ├── cli.py                 # Benchmark CLI commands
│   ├── scenario_executor.py   # Scenario execution logic
│   └── gaia2_submission.py    # Gaia2 specific logic
├── gui/                       # Web-based GUI
│   ├── server/                # FastAPI backend (GraphQL)
│   └── client/                # React frontend
├── environment.py             # Core environment class
├── types.py                   # Core type definitions
├── tool_utils.py              # App tool decorator and utilities
├── main.py                    # CLI entry point (are-run)
└── notification_system.py     # Event notification system
```

## Important Patterns

### Type Annotations

Follow modern Python typing conventions (enforced by ruff):
- Use `dict` instead of `typing.Dict`
- Use `list` instead of `typing.List`
- Use `T | None` instead of `typing.Optional[T]`
- Use `tuple` instead of `typing.Tuple`

### EventRegisterer.capture_mode()

When building event flows, always use `EventRegisterer.capture_mode()`:
```python
with EventRegisterer.capture_mode():
    event = app.method(...).depends_on(prev_event, delay_seconds=5)
```
This captures app method calls as events without executing them immediately.

### Strawberry Optional

The project supports optional Strawberry (GraphQL) dependency:
- When installed: Used for GUI GraphQL API
- When not installed: No-op decorators maintain functionality
- Never assume Strawberry is available in core simulation code

### Sandbox Isolation

File system operations use `SandboxLocalFileSystem`:
- Creates isolated temporary directories per scenario
- Prevents contamination between test runs
- Automatically cleaned up after scenario completion

## CI/CD

The project uses GitHub Actions for:
- **Python checks**: Linting (ruff), formatting (ruff format), type checking (pyright)
- **TypeScript checks**: Type checking (tsc), formatting (prettier), tests
- **Lock file validation**: Ensures uv.lock is up to date
- **Documentation**: Builds and deploys Sphinx docs

All checks must pass before merging PRs.

## Environment Variables

Common environment variables:
```bash
# Model API keys
LLAMA_API_KEY=...
OPENAI_API_KEY=...
HF_TOKEN=...

# File system paths
FS_PATH=/tmp/are_simulation_datasets/fs_states
DEMO_FS_PATH=hf://datasets/meta-agents-research-environments/gaia2_filesystem/demo_filesystem

# Logging
LOG_LEVEL=INFO  # or DEBUG
```

## Common Pitfalls

1. **Event dependencies**: Events without proper dependencies may execute out of order
2. **App initialization**: Always call `init_and_populate_apps()` before `build_events_flow()`
3. **Tool names**: MCP tools must use `server__tool` format (double underscore)
4. **Validation timing**: Validation runs after scenario completion, not during execution
5. **Time management**: Respect event loop timing; don't block the event thread
6. **Oracle mode**: OracleEvents only execute in oracle mode (for testing ideal agent behavior)

## Documentation

Full documentation at: https://facebookresearch.github.io/meta-agents-research-environments/

Key sections:
- [Core Concepts](https://facebookresearch.github.io/meta-agents-research-environments/foundations/index.html)
- [Benchmarking Guide](https://facebookresearch.github.io/meta-agents-research-environments/user_guide/benchmarking.html)
- [Scenario Development](https://facebookresearch.github.io/meta-agents-research-environments/tutorials/scenario_development.html)
- [API Reference](https://facebookresearch.github.io/meta-agents-research-environments/api_reference/)
