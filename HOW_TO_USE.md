# Using Meta ARE for Vibrant Labs Simulatons

## Setup

### 1. Installation

```bash
uv sync --all-packages --all-extras
source ./venv/bin/activate
```

```bash
uv pip install -r requirements-gui.txt
uv pip install browser_use
```

### 2. Install Web Simulators

With the same `VIRTUALENV` as above: `cd` into the `<your_root_dir>/web_simulators/packages/web-simulators`

> Note: Please check out to `sid/downgrade-python`. [Link](https://github.com/explodinggradients/web_simulators/tree/sid/downgrade-python)

```bash
# keep the same VIRTUALENV.
# We're doing an editable install for now
cd web_simulators/packages/web-simulators/packages/web-simulators
uv pip install -e .
```

---

## Scenarios

As of now a basic scenario exists. Following are where the apps, agents and scenarios are implemented:

- Scenario: `are/simulation/scenarios/scenario_gmail_browser/scenario.py`
- Custom browser agenr using `browser_use`: `are/simulation/agents/custom_agents/browser_agent.py`
- "Gmail" App, Better name/tools possible: `are/simulation/apps/gmail.py`

```bash
are-run -s scenario_gmail_browser -m 'gpt-5-mini' -a browser -mp local --endpoint https://api.openai.com/v1 --export --output_dir ./traces
```

---

## TODOs

- [ ] Tracing has all the browser state but could be improved
- [ ] Implement and test validation + oracle
