# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
"""gaia2-replay: Replay scenario oracle actions through the gaia2-eventd pipeline.

Parses a Gaia2 scenario JSON directly (no Gaia2 framework imports needed),
extracts oracle AGENT events grouped by turn using the dependency DAG, and
replays them by calling CLI tools — simulating what a perfect agent would do.

The replay acts as a fake agent:
  1. gaia2-init sets up the state directory (per-app state + empty events.jsonl)
  2. For each oracle turn, replay calls CLI tools via subprocess
     (the CLIs write to events.jsonl and mutate app state)
  3. Replay writes the turn boundary (send_message_to_user) via the channel
  4. gaia2-eventd daemon (if --with-daemon) ticks the environment, fires ENV
     reactions as more CLI calls, and delivers notifications

No LLM needed — the oracle provides the "perfect" agent actions.

Usage:
    gaia2-replay --scenario scenario.json
    gaia2-replay --scenario scenario.json --state-dir /tmp/test --with-daemon
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import click

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Faketime synchronization
# ---------------------------------------------------------------------------
_FAKETIME_PATH = Path("/tmp/faketime.rc")


def _read_faketime(faketime_path: Path = _FAKETIME_PATH) -> float | None:
    """Read the current simulated timestamp from faketime.rc.

    Returns a Unix timestamp (float), or None if the file is missing/empty.
    """
    from datetime import datetime, timezone

    try:
        text = faketime_path.read_text().strip()
        if text:
            dt = datetime.strptime(text, "%Y-%m-%d %H:%M:%S").replace(
                tzinfo=timezone.utc
            )
            return dt.timestamp()
    except (FileNotFoundError, ValueError, OSError):
        pass
    return None


def _wait_for_faketime(
    target_time: float,
    faketime_path: Path = _FAKETIME_PATH,
    timeout: float = 360.0,
    poll_interval: float = 1.0,
) -> bool:
    """Poll faketime.rc until simulated time reaches *target_time* (Unix ts).

    Returns True if target was reached, False on timeout.
    """
    from datetime import datetime, timezone

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            text = faketime_path.read_text().strip()
            if text:
                dt = datetime.strptime(text, "%Y-%m-%d %H:%M:%S").replace(
                    tzinfo=timezone.utc
                )
                if dt.timestamp() >= target_time:
                    return True
        except (FileNotFoundError, ValueError, OSError):
            pass
        time.sleep(poll_interval)
    logger.warning(
        "Faketime did not reach target %.0f after %.0fs", target_time, timeout
    )
    return False


# ---------------------------------------------------------------------------
# JSON scenario parser — zero framework imports
# ---------------------------------------------------------------------------
def _parse_args(args_list: list[dict[str, Any]]) -> dict[str, Any]:
    """Convert Gaia2 JSON args list to a flat dict.

    Gaia2 format:  [{"name": "to", "value": "alice", "value_type": "str"}, ...]
    Output:       {"to": "alice", ...}
    """
    result = {}
    for arg in args_list:
        name = arg["name"]
        value = arg.get("value")
        vtype = arg.get("value_type", "str")

        # Parse value types that JSON doesn't preserve
        if value is not None:
            if vtype == "int":
                try:
                    value = int(value)
                except (ValueError, TypeError):
                    pass
            elif vtype == "float":
                try:
                    value = float(value)
                except (ValueError, TypeError):
                    pass
            elif vtype == "bool":
                value = str(value).lower() in ("true", "1")

        result[name] = value
    return result


def _to_oracle_entry(event: dict[str, Any]) -> dict[str, Any]:
    """Convert a scenario JSON event to an internal oracle entry.

    Keeps event_id so we can track return values for placeholder resolution.
    """
    action = event["action"]
    args = _parse_args(action.get("args", []))
    is_write = (action.get("operation_type") or "").upper() == "WRITE"

    return {
        "event_id": event["event_id"],
        "t": event.get("event_time") or 0.0,
        "app": action["app"],
        "fn": action["function"],
        "args": args,
        "w": is_write,
        "_event_relative_time": event.get("event_relative_time"),
    }


def load_scenario_events(
    scenario_path: str,
) -> tuple[str | None, list[list[dict[str, Any]]], list[dict[str, Any]]]:
    """Parse scenario JSON, return (task, agent_turns, env_events).

    Lightweight fallback parser — no Gaia2 framework imports needed.
    Resolves the dependency DAG to group AGENT events into turns,
    split at send_message_to_user boundaries.

    Prefer load_scenario_events_via_env() when the framework is available; it uses
    the real Environment oracle run for correct ordering.

    Returns:
        task: The initial user message, or None.
        agent_turns: List of turns. Each turn is a list of oracle entry dicts.
            The last event of each turn is send_message_to_user.
        env_events: List of ENV event dicts (for reference / logging).
    """
    with open(scenario_path) as f:
        data = json.load(f)

    events = data.get("events", [])

    # Extract task: find the USER send_message_to_agent with no dependencies
    # (the initial message). In multi-turn scenarios, events are in DAG order
    # and the initial message may not be first in the list.
    task = None
    fallback_task = None
    for ev in events:
        if ev.get("event_type") == "USER":
            action = ev.get("action", {})
            if action.get("function") == "send_message_to_agent":
                content = None
                for arg in action.get("args", []):
                    if arg["name"] == "content":
                        content = arg.get("value")
                        break
                if content:
                    if fallback_task is None:
                        fallback_task = content
                    if not ev.get("dependencies", []):
                        task = content
                        break
    if task is None:
        task = fallback_task

    # Build lookup by event_id
    by_id: dict[str, dict[str, Any]] = {}
    for ev in events:
        by_id[ev["event_id"]] = ev

    # Separate by type
    agent_events = [e for e in events if e.get("event_type") == "AGENT"]
    env_events = [e for e in events if e.get("event_type") == "ENV"]

    # Topological sort of AGENT events using dependencies
    sorted_agents = _topo_sort(agent_events, by_id)

    # Group into turns: split at send_message_to_user
    turns: list[list[dict[str, Any]]] = []
    current_turn: list[dict[str, Any]] = []

    for ev in sorted_agents:
        entry = _to_oracle_entry(ev)
        current_turn.append(entry)

        if (
            entry["app"] == "AgentUserInterface"
            and entry["fn"] == "send_message_to_user"
        ):
            turns.append(current_turn)
            current_turn = []

    # Trailing events (shouldn't happen in well-formed scenarios)
    if current_turn:
        turns.append(current_turn)

    env_entries = [_to_oracle_entry(e) for e in env_events]
    return task, turns, env_entries


def _topo_sort(
    agent_events: list[dict[str, Any]],
    all_events_by_id: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Topological sort of agent events respecting dependency order.

    Events without dependencies (or whose deps are non-AGENT) come first.
    When an agent event depends on a non-AGENT event (e.g. ENV), we trace
    through that event's dependency chain to find the ultimate AGENT ancestor
    and use it as the effective dependency.  This ensures correct turn ordering
    when AGENT → ENV → AGENT chains exist.

    Ties are broken by event_time.
    """
    id_set = {e["event_id"] for e in agent_events}

    def _resolve_agent_ancestors(
        dep_id: str, visited: set[str] | None = None
    ) -> list[str]:
        """Walk non-AGENT deps until we find AGENT event(s)."""
        if visited is None:
            visited = set()
        if dep_id in visited:
            return []
        visited.add(dep_id)
        if dep_id in id_set:
            return [dep_id]
        ev = all_events_by_id.get(dep_id)
        if ev is None:
            return []
        result = []
        for parent in ev.get("dependencies", []):
            result.extend(_resolve_agent_ancestors(parent, visited))
        return result

    # Build adjacency: for each agent event, which other agent events must come first
    in_degree: dict[str, int] = defaultdict(int)
    dependents: dict[str, list[str]] = defaultdict(list)

    for ev in agent_events:
        eid = ev["event_id"]
        if eid not in in_degree:
            in_degree[eid] = 0
        for dep_id in ev.get("dependencies", []):
            if dep_id in id_set:
                in_degree[eid] += 1
                dependents[dep_id].append(eid)
            else:
                # Trace through non-AGENT events to find AGENT ancestors
                for ancestor_id in _resolve_agent_ancestors(dep_id):
                    if ancestor_id != eid:  # avoid self-loops
                        in_degree[eid] += 1
                        dependents[ancestor_id].append(eid)

    # Kahn's algorithm
    queue = sorted(
        [eid for eid, deg in in_degree.items() if deg == 0],
        key=lambda eid: all_events_by_id[eid].get("event_time") or 0,
    )
    result = []

    while queue:
        eid = queue.pop(0)
        result.append(all_events_by_id[eid])
        for dep in sorted(
            dependents.get(eid, []),
            key=lambda d: all_events_by_id[d].get("event_time") or 0,
        ):
            in_degree[dep] -= 1
            if in_degree[dep] == 0:
                queue.append(dep)
        queue.sort(key=lambda d: all_events_by_id[d].get("event_time") or 0)

    return result


# ---------------------------------------------------------------------------
# ScenarioLoader-based loader (correct ordering via dependency DAG)
# ---------------------------------------------------------------------------
def load_scenario_events_via_env(
    scenario_path: str,
) -> tuple[str | None, list[list[dict[str, Any]]], list[dict[str, Any]]]:
    """Load scenario using ScenarioLoader + EventProcessor for correct turn assignment.

    Uses the same static DAG-based turn assignment that the daemon's judge uses
    (EventProcessor.build_event_id_to_turn_idx), ensuring the replay groups
    events into the same turns as the grader.

    Returns the same (task, agent_turns, env_events) format as
    load_scenario_events() for drop-in compatibility.
    """
    from gaia2_core.event_loop import EventProcessor
    from gaia2_core.loader import ScenarioLoader

    loader = ScenarioLoader(scenario_path)

    # Use EventProcessor for turn assignment — same as the daemon/judge.
    processor = EventProcessor(
        events=loader.events,
        start_time=loader.start_time,
        duration=loader.duration,
        time_increment=loader.time_increment,
        app_name_to_class=loader.app_name_to_class,
    )
    processor.build_turn_triggers()
    nb_turns = processor.nb_turns

    # Extract oracle data using the processor's turn assignment.
    (
        turn_to_oracle_events,
        turn_to_oracle_graph,
        tasks,
        _,
    ) = loader.extract_oracle_data(processor.event_id_to_turn_idx, nb_turns)

    # Build lookup for ENV dependency detection.
    all_events_by_id = {e.event_id: e for e in loader.events}

    # Precompute absolute target offsets (seconds from scenario start) for
    # each event by walking the dependency DAG. event_relative_time is
    # relative to the parent, so abs_offset = max(parent offsets) + rt.
    abs_offsets: dict[str, float] = {}

    def _get_abs_offset(eid: str) -> float:
        if eid in abs_offsets:
            return abs_offsets[eid]
        ev = all_events_by_id.get(eid)
        if not ev:
            return 0.0
        rt = ev.event_relative_time or 0.0
        deps = ev.dependency_ids
        parent_max = max((_get_abs_offset(d) for d in deps), default=0.0)
        abs_offsets[eid] = parent_max + rt
        return abs_offsets[eid]

    for ev in loader.events:
        _get_abs_offset(ev.event_id)

    # Convert OracleEvent objects to the dict format expected by the replay loop.
    # turn_to_oracle_events is a list[list[OracleEvent]], indexed by turn.
    # ENV prerequisite events are NOT inserted — the daemon fires them via
    # _advance_time() as simulated time progresses. Each entry carries its
    # ENV dependency IDs so the replay can wait for them before executing.
    turns: list[list[dict[str, Any]]] = []
    for tidx in range(nb_turns):
        oracle_evs = (
            turn_to_oracle_events[tidx] if tidx < len(turn_to_oracle_events) else []
        )
        oracle_graph = (
            turn_to_oracle_graph[tidx] if tidx < len(turn_to_oracle_graph) else {}
        )
        turn_entries: list[dict[str, Any]] = []
        for oracle_ev in oracle_evs:
            # Mirror the judge's parent selection:
            # - AGENT parents come from the per-turn oracle graph
            # - ENV parents come from the event's direct dependency_ids
            # - USER parents are handled as a start_time fallback at replay time
            env_deps: list[str] = []
            src_ev = all_events_by_id.get(oracle_ev.event_id)
            if src_ev:
                for dep_id in src_ev.dependency_ids:
                    dep = all_events_by_id.get(dep_id)
                    if dep and dep.event_type == "ENV":
                        env_deps.append(dep_id)

            dependency_ids = list(src_ev.dependency_ids) if src_ev else []
            entry = {
                "event_id": oracle_ev.event_id,
                "t": oracle_ev.event_time or 0.0,
                "app": oracle_ev.action.app_name,
                "fn": oracle_ev.action.function_name,
                "args": dict(oracle_ev.action.args),
                "w": oracle_ev.action.operation_type.upper() == "WRITE",
                "_agent_parents": list(oracle_graph.get(oracle_ev.event_id, [])),
                "_env_deps": env_deps,
                "_dependency_ids": dependency_ids,
                "_event_relative_time": oracle_ev.event_relative_time,
                "_target_offset": abs_offsets.get(oracle_ev.event_id, 0.0),
            }
            turn_entries.append(entry)
        turns.append(turn_entries)

    # Collect ENV events for logging (exclude those inserted as prereqs).
    env_entries: list[dict[str, Any]] = []
    for e in loader.events:
        if e.event_type == "ENV" and e.action:
            env_entries.append(
                {
                    "event_id": e.event_id,
                    "t": e.event_time or 0.0,
                    "app": e.action.app_name,
                    "fn": e.action.function_name,
                    "args": dict(e.action.args),
                    "w": e.action.operation_type.upper() == "WRITE",
                }
            )

    total_tool_calls = sum(
        len([e for e in t if e["fn"] != "send_message_to_user"]) for t in turns
    )
    logger.info(
        "EventProcessor-based loader: %d turns, %d tool calls, %d ENV events",
        len(turns),
        total_tool_calls,
        len(env_entries),
    )

    return loader.task, turns, env_entries


# ---------------------------------------------------------------------------
# Placeholder resolution  (mirrors gaia2 environment.resolve_arg_placeholders)
# ---------------------------------------------------------------------------

_PLACEHOLDER_RE = re.compile(r"^\{\{(.*?)\}\}$")


def _resolve_placeholders(
    args: dict[str, Any],
    return_values: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    """Resolve {{event_id}} and {{event_id.key}} placeholders in args.

    Returns (resolved_args, log_messages).
    """
    resolved = dict(args)
    logs: list[str] = []
    for arg_name, arg_value in list(resolved.items()):
        if not isinstance(arg_value, str):
            continue
        m = _PLACEHOLDER_RE.match(arg_value.strip())
        if not m:
            continue

        parts = m.group(1).split(".")
        ref_event_id = parts[0]

        if ref_event_id not in return_values:
            logs.append(f"    ! unresolved placeholder {arg_name}={arg_value}")
            continue

        value = return_values[ref_event_id]
        for key in parts[1:]:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                logs.append(
                    f"    ! failed to resolve key '{key}' in return value "
                    f"of {ref_event_id}"
                )
                value = arg_value  # keep original
                break

        resolved[arg_name] = value
        logs.append(f"    resolved {arg_name}: {arg_value} -> {repr(value)[:60]}")
    return resolved, logs


# ---------------------------------------------------------------------------
# CLI invocation
# ---------------------------------------------------------------------------

# Reuse the canonical app-to-CLI mapping (single source of truth).
from gaia2_cli.app_registry import APP_TO_CLI as _APP_TO_CLI


def _load_fn_to_command_map(state_dir: str) -> dict[str, dict[str, str]]:
    """Load schema from each CLI and build oracle_function → command mapping.

    Returns {cli_name: {oracle_function: command_name}} for each CLI.
    Calls ``<cli> schema`` subprocess for each entry point.
    """
    cli_names = sorted(set(_APP_TO_CLI.values()))
    fn_map: dict[str, dict[str, str]] = {}

    for cli_name in cli_names:
        try:
            result = subprocess.run(
                [cli_name, "schema"],
                capture_output=True,
                text=True,
                timeout=10,
                env={**os.environ, "GAIA2_STATE_DIR": state_dir},
            )
            if result.returncode != 0:
                logger.warning(f"schema failed for {cli_name}: {result.stderr[:80]}")
                continue
            schema = json.loads(result.stdout)
            mapping: dict[str, str] = {}
            for entry in schema:
                oracle_fn = entry.get("oracle_function", "")
                command = entry.get("command", "")
                if oracle_fn and command:
                    mapping[oracle_fn] = command
                # Also map the short function name as fallback
                short_fn = entry.get("function", "")
                if short_fn and command and short_fn not in mapping:
                    mapping[short_fn] = command
            fn_map[cli_name] = mapping
        except Exception as e:
            logger.warning(f"Failed to load schema for {cli_name}: {e}")

    return fn_map


def _build_cli_cmd(
    app_name: str,
    fn_name: str,
    args: dict[str, Any],
    fn_map: dict[str, dict[str, str]],
) -> list[str] | None:
    """Convert an oracle action to a CLI command list.

    Uses the schema-derived fn_map to translate Gaia2 function names
    (e.g. add_calendar_event) to CLI command names (e.g. add-event).
    """
    cli = _APP_TO_CLI.get(app_name)
    if not cli:
        return None

    # Look up the CLI command name from the schema mapping
    cli_fn_map = fn_map.get(cli, {})
    subcmd = cli_fn_map.get(fn_name)
    if subcmd is None:
        # Fallback: convert snake_case to kebab-case directly
        subcmd = fn_name.replace("_", "-")
        logger.debug(f"No schema mapping for {cli}.{fn_name}, falling back to {subcmd}")

    cmd = [cli, subcmd]

    for key, value in args.items():
        option = f"--{key.replace('_', '-')}"
        if isinstance(value, bool):
            if value:
                cmd.append(option)
        elif isinstance(value, (list, dict)):
            cmd.extend([option, json.dumps(value)])
        elif value is not None and value != "":
            cmd.extend([option, str(value)])

    return cmd


def _run_cli(
    cmd: list[str],
    state_dir: str,
    event_id: str = "",
) -> tuple[int, str, str]:
    """Execute a CLI command with GAIA2_STATE_DIR set.

    If *event_id* is provided it is passed via ``GAIA2_EVENT_ID`` so that the
    CLI's ``log_action()`` includes it in the events.jsonl entry.

    Returns (returncode, stdout, stderr).
    """
    env = {**os.environ, "GAIA2_STATE_DIR": state_dir}
    if event_id:
        env["GAIA2_EVENT_ID"] = event_id
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
            env=env,
        )
        return result.returncode, result.stdout.strip(), result.stderr.strip()
    except subprocess.TimeoutExpired:
        return -1, "", "timeout"
    except Exception as e:
        return -1, "", str(e)


def _get_last_event_sim_t(events_path: Path) -> float | None:
    """Read the sim_t (simulated timestamp) from the last event in events.jsonl."""
    if not events_path.exists():
        return None
    last_line = ""
    with open(events_path, "rb") as f:
        f.seek(0, 2)
        pos = f.tell()
        while pos > 0:
            pos -= 1
            f.seek(pos)
            ch = f.read(1)
            if ch == b"\n" and last_line:
                break
            last_line = ch.decode("utf-8", errors="replace") + last_line
    if last_line.strip():
        try:
            entry = json.loads(last_line.strip())
            sim_t_str = entry.get("sim_t")
            if sim_t_str:
                from datetime import datetime, timezone

                dt = datetime.strptime(sim_t_str, "%Y-%m-%d %H:%M:%S").replace(
                    tzinfo=timezone.utc
                )
                return dt.timestamp()
        except (json.JSONDecodeError, ValueError):
            pass
    return None


def _collect_event_sim_times(
    events_path: Path,
    sim_times: dict[str, float],
) -> None:
    """Scan events.jsonl and collect sim_t for all events (AGENT + ENV)."""
    from datetime import datetime, timezone

    for entry in read_all_lines(events_path):
        eid = entry.get("event_id", "")
        sim_t_str = entry.get("sim_t")
        if eid and sim_t_str and eid not in sim_times:
            try:
                dt = datetime.strptime(sim_t_str, "%Y-%m-%d %H:%M:%S").replace(
                    tzinfo=timezone.utc
                )
                sim_times[eid] = dt.timestamp()
            except ValueError:
                pass


def _compute_judge_parent_time(
    entry: dict[str, Any],
    fired_event_times: dict[str, float],
    scenario_start_time: float,
) -> float | None:
    """Mirror the judge's parent-time selection for one oracle event.

    The judge uses:
    - AGENT parents from the per-turn oracle graph
    - ENV parents from the event's direct dependency_ids
    - start_time when the event only depends on USER events

    Events with no dependencies at all skip time checking entirely.
    """
    max_parent_time = scenario_start_time
    has_parent = False

    for parent_id in entry.get("_agent_parents", []):
        parent_time = fired_event_times.get(parent_id)
        if parent_time is not None:
            has_parent = True
            max_parent_time = max(max_parent_time, parent_time)

    for dep_id in entry.get("_env_deps", []):
        env_time = fired_event_times.get(dep_id)
        if env_time is not None:
            has_parent = True
            max_parent_time = max(max_parent_time, env_time)

    if not has_parent and entry.get("_dependency_ids"):
        return scenario_start_time

    if not has_parent:
        return None

    return max_parent_time


def _get_last_event_ret(events_path: Path) -> Any:
    """Read the return value from the last event written to events.jsonl."""
    if not events_path.exists():
        return None
    # Read the last non-empty line
    last_line = ""
    with open(events_path, "rb") as f:
        f.seek(0, 2)  # end of file
        pos = f.tell()
        while pos > 0:
            pos -= 1
            f.seek(pos)
            ch = f.read(1)
            if ch == b"\n" and last_line:
                break
            last_line = ch.decode("utf-8", errors="replace") + last_line
    if last_line.strip():
        try:
            return json.loads(last_line.strip()).get("ret")
        except json.JSONDecodeError:
            pass
    return None


# ---------------------------------------------------------------------------
# File I/O helpers
# ---------------------------------------------------------------------------
def _collect_env_return_values(
    events_path: Path,
    return_values: dict[str, Any],
) -> None:
    """Scan events.jsonl for ENV event return values not yet in return_values."""
    for entry in read_all_lines(events_path):
        eid = entry.get("event_id", "")
        if eid and eid not in return_values:
            return_values[eid] = entry.get("ret")


def read_all_lines(path: Path) -> list[dict[str, Any]]:
    """Read all JSON lines from a file."""
    if not path.exists():
        return []
    result = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                result.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return result


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def replay_scenario(
    scenario: str,
    state_dir: str | None = None,
    turn_delay: float = 1.0,
    with_daemon: bool = False,
    verbose: bool = False,
    quiet: bool = False,
    time_speed: float | None = None,
    skip_init: bool = False,
    external_daemon: bool = False,
) -> bool:
    """Run a single scenario replay. Returns True on success (0 errors).

    When *quiet* is True, suppresses click.echo output (used in batch mode).
    """
    echo = (lambda *a, **kw: None) if quiet else click.echo

    # State dir
    if state_dir is None:
        state_dir = tempfile.mkdtemp(prefix="gaia2-replay-")
    state_path = Path(state_dir)
    state_path.mkdir(parents=True, exist_ok=True)

    events_path = state_path / "events.jsonl"
    notifications_path = state_path / "notifications.jsonl"

    # --- Set up file logging to {state_dir}/replay.log ---
    log_level = logging.DEBUG if verbose else logging.INFO
    log_path = state_path / "replay.log"
    replay_logger = logging.getLogger(f"gaia2_cli.replay.{Path(scenario).stem}")
    replay_logger.setLevel(log_level)
    replay_logger.propagate = False
    # File handler — always DEBUG for full trace
    fh = logging.FileHandler(log_path, mode="w")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s"))
    replay_logger.addHandler(fh)
    if not quiet:
        # Console handler — respects -v flag
        ch = logging.StreamHandler()
        ch.setLevel(log_level)
        ch.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        replay_logger.addHandler(ch)

    # --- Initialize state directory via gaia2-init ---
    echo(f"Scenario:  {scenario}")
    echo(f"State dir: {state_path}")
    echo(f"Log file:  {log_path}")
    echo()

    if skip_init:
        echo("Skipping gaia2-init (already initialized by entrypoint)")
        replay_logger.info("gaia2-init skipped (skip_init=True)")
    else:
        echo("Running gaia2-init...")
        replay_logger.info(
            "Running gaia2-init --scenario %s --state-dir %s", scenario, state_path
        )
        rc, out, err = _run_cli(
            ["gaia2-init", "--scenario", scenario, "--state-dir", str(state_path)],
            str(state_path),
        )
        if rc != 0:
            replay_logger.error("gaia2-init failed (rc=%d): %s", rc, err)
            echo(f"gaia2-init failed (rc={rc}): {err}", err=True)
            return False
        replay_logger.info("gaia2-init: %s", out)
        echo(f"  {out}")
    echo()

    # --- Load CLI schemas for function name mapping ---
    echo("Loading CLI schemas...")
    fn_map = _load_fn_to_command_map(str(state_path))
    total_mapped = sum(len(m) for m in fn_map.values())
    replay_logger.info(
        "Loaded schemas: %d CLIs, %d function mappings", len(fn_map), total_mapped
    )
    for cli_name, mapping in sorted(fn_map.items()):
        replay_logger.debug("  %s: %s", cli_name, ", ".join(sorted(mapping.keys())))
    echo(f"  {len(fn_map)} CLIs, {total_mapped} function mappings")
    echo()

    # --- Parse scenario for oracle events ---
    # Use DAG mode when a daemon (in-process or external) handles ENV events.
    daemon_active = with_daemon or external_daemon
    if daemon_active:
        # Use ScenarioLoader-based loader for correct DAG ordering
        echo("Loading scenario via ScenarioLoader (DAG mode)...")
        task, oracle_turns, env_events = load_scenario_events_via_env(scenario)
    else:
        task, oracle_turns, env_events = load_scenario_events(scenario)
    total_tool_calls = sum(
        len([e for e in t if e["fn"] != "send_message_to_user"]) for t in oracle_turns
    )

    echo(f"Task: {(task or '-')[:120]}{'...' if task and len(task) > 120 else ''}")
    echo(
        f"Oracle: {len(oracle_turns)} turns, {total_tool_calls} tool calls, "
        f"{len(env_events)} ENV events"
    )
    replay_logger.info(
        "Oracle parsed: %d turns, %d tool calls, %d ENV events",
        len(oracle_turns),
        total_tool_calls,
        len(env_events),
    )
    for i, turn in enumerate(oracle_turns):
        fns = [f"{e['app']}.{e['fn']}" for e in turn]
        replay_logger.debug("  Turn %d: %s", i, " → ".join(fns))
    if env_events:
        replay_logger.info("Expected ENV events:")
        for e in env_events:
            replay_logger.info(
                "  ENV: %s.%s(%s)", e["app"], e["fn"], ", ".join(e["args"].keys())
            )
    echo()

    # --- Optionally start daemon ---
    daemon_thread = None
    daemon = None
    if with_daemon:
        import threading

        from gaia2_cli.daemon.eventd import Gaia2EventDaemon

        daemon = Gaia2EventDaemon(
            scenario_path=scenario,
            state_dir=str(state_path),
            poll_interval=0.1,
            log_path=str(state_path / "eventd.log"),
            time_speed=time_speed,
        )
        echo("Starting daemon...")
        replay_logger.info("Starting gaia2-eventd daemon (poll=0.1s)")
        daemon.setup()
        replay_logger.info("Daemon setup complete")
        daemon_thread = threading.Thread(target=daemon.run, daemon=True)
        daemon_thread.start()
        echo("Daemon running")
        echo(f"Daemon log: {state_path / 'eventd.log'}")
        echo()

    # --- Create channel adapter for turn boundaries ---
    from gaia2_cli.daemon.channel import FileChannelAdapter

    channel = FileChannelAdapter(
        notifications_path=notifications_path,
        responses_path=state_path / "agent_responses.jsonl",
        events_jsonl_path=events_path,
    )

    # --- Read scenario start time from faketime.rc ---
    scenario_start_time: float | None = None
    if daemon_active and _FAKETIME_PATH.exists():
        scenario_start_time = _read_faketime()
        if scenario_start_time:
            replay_logger.info("Scenario start time: %.0f", scenario_start_time)

    # Track actual sim_t of fired events — mirrors the judge's logic for
    # computing max_parent_agent_time. Keys are oracle event_ids, values
    # are the sim_t (Unix timestamp) at which the event actually fired.
    fired_event_times: dict[str, float] = {}

    # --- Replay turns ---
    return_values: dict[str, Any] = {}
    total_ok = 0
    total_err = 0
    replay_start = time.monotonic()  # noqa: F841
    try:
        for turn_idx, turn_events in enumerate(oracle_turns):
            notifs_before = len(read_all_lines(notifications_path))

            tool_calls = [e for e in turn_events if e["fn"] != "send_message_to_user"]
            # Sort by precomputed target offset so sibling events
            # with different timing execute in the correct order.
            tool_calls.sort(key=lambda e: e.get("_target_offset", 0))
            boundary = [e for e in turn_events if e["fn"] == "send_message_to_user"]

            echo(f"--- Turn {turn_idx} ({len(tool_calls)} tool calls) ---")
            replay_logger.info(
                "=== TURN %d === (%d tool calls, boundary=%s)",
                turn_idx,
                len(tool_calls),
                bool(boundary),
            )

            for entry in tool_calls:
                env_deps = entry.get("_env_deps", [])

                eid = entry["event_id"]
                app = entry["app"]
                fn = entry["fn"]

                replay_logger.info("CALL %s.%s() [event_id=%s]", app, fn, eid)
                replay_logger.debug(
                    "  raw args: %s", json.dumps(entry["args"], default=str)[:200]
                )

                # Wait for ENV dependencies to fire before executing.
                # The daemon fires ENV events via _advance_time() as
                # simulated time progresses. We poll events.jsonl until
                # all prerequisite ENV events have appeared.
                if env_deps and daemon_active:
                    missing = [d for d in env_deps if d not in return_values]
                    if missing:
                        replay_logger.info(
                            "  Waiting for %d ENV dep(s): %s",
                            len(missing),
                            ", ".join(d[:30] for d in missing),
                        )
                        for _wait in range(360):
                            time.sleep(1.0)
                            _collect_env_return_values(events_path, return_values)
                            missing = [d for d in env_deps if d not in return_values]
                            if not missing:
                                replay_logger.info(
                                    "  ENV deps resolved after %ds", _wait + 1
                                )
                                break
                        if missing:
                            replay_logger.warning(
                                "  ENV deps still missing after 360s: %s",
                                ", ".join(d[:30] for d in missing),
                            )
                # Resolve {{event_id}} placeholders.
                # If any reference ENV events whose return values we don't have
                # yet, poll events.jsonl until the daemon fires them.
                resolved_args, res_logs = _resolve_placeholders(
                    entry["args"], return_values
                )
                has_unresolved = any("unresolved" in m for m in res_logs)

                if has_unresolved and with_daemon:
                    # Wait for daemon to fire ENV events and collect their
                    # return values from events.jsonl. Time scenarios may
                    # need up to 5+ minutes of simulated time to elapse.
                    for _wait in range(360):  # up to 6 min
                        time.sleep(1.0)
                        _collect_env_return_values(events_path, return_values)
                        resolved_args, res_logs = _resolve_placeholders(
                            entry["args"], return_values
                        )
                        has_unresolved = any("unresolved" in m for m in res_logs)
                        if not has_unresolved:
                            replay_logger.info(
                                "  ENV placeholders resolved after %ds", _wait + 1
                            )
                            break
                    if has_unresolved:
                        replay_logger.warning(
                            "  ENV placeholders still unresolved after 360s"
                        )

                for msg in res_logs:
                    replay_logger.info(msg)
                    echo(msg)

                # Schedule event at the correct simulated time — mirrors the
                # judge's parent-time logic:
                #   AGENT parents come from the per-turn oracle graph
                #   ENV parents come from direct dependency_ids
                #   USER-only deps fall back to start_time
                rt = entry.get("_event_relative_time") or 0
                if rt > 1.0 and daemon_active and scenario_start_time is not None:
                    # Collect AGENT/ENV event times from events.jsonl.
                    _collect_event_sim_times(events_path, fired_event_times)
                    parent_time = _compute_judge_parent_time(
                        entry,
                        fired_event_times,
                        scenario_start_time,
                    )
                    if parent_time is not None:
                        target = parent_time + rt
                        replay_logger.info(
                            "  Scheduling: rt=%.0fs, max_parent=%.0f, target=%.0f",
                            rt,
                            parent_time - scenario_start_time,
                            target - scenario_start_time,
                        )
                        _wait_for_faketime(target)

                # Build and run CLI command
                cmd = _build_cli_cmd(app, fn, resolved_args, fn_map)
                if cmd is None:
                    replay_logger.warning(
                        "SKIP %s.%s() — no CLI mapping for app %r", app, fn, app
                    )
                    echo(f"  {app}.{fn}() -> SKIP (no CLI mapping)")
                    continue

                replay_logger.info("  cmd: %s", " ".join(cmd))

                # Don't pass oracle event_id to CLI — events.jsonl entries
                # must NOT have OracleEvent-* IDs, otherwise the post-hoc
                # grader's collect_turn_agent_events() skips them.
                # The replay tracks return values internally via eid key.
                rc, stdout, stderr = _run_cli(cmd, str(state_path))

                if rc == 0:
                    total_ok += 1
                    # log_action wrote event_id + ret to events.jsonl;
                    # read ret and sim_t from the last entry for immediate use
                    ret = _get_last_event_ret(events_path)
                    sim_t = _get_last_event_sim_t(events_path)
                    return_values[eid] = ret
                    if sim_t is not None:
                        fired_event_times[eid] = sim_t
                    replay_logger.info(
                        "  OK ret=%r sim_t=%.0f", repr(ret)[:80], sim_t or 0
                    )
                    if stdout:
                        replay_logger.debug("  stdout: %s", stdout[:200])
                    echo(f"  {app}.{fn}() -> ok (ret={repr(ret)[:50]})")
                else:
                    total_err += 1
                    return_values[eid] = eid  # fallback for placeholders
                    replay_logger.error("  FAIL rc=%d", rc)
                    if stderr:
                        replay_logger.error("  stderr: %s", stderr[:300])
                    if stdout:
                        replay_logger.error("  stdout: %s", stdout[:300])
                    echo(f"  {app}.{fn}() -> FAIL (rc={rc})")
                    if stderr:
                        echo(f"    stderr: {stderr[:120]}")

            # Write turn boundary via channel adapter
            if boundary:
                rt = boundary[0].get("_event_relative_time") or 0
                b_env_deps = boundary[0].get("_env_deps", [])
                if b_env_deps and daemon_active:
                    missing = [d for d in b_env_deps if d not in return_values]
                    if missing:
                        replay_logger.info(
                            "  Waiting for %d ENV dep(s) before boundary: %s",
                            len(missing),
                            ", ".join(d[:30] for d in missing),
                        )
                        for _wait in range(360):
                            time.sleep(1.0)
                            _collect_env_return_values(events_path, return_values)
                            missing = [d for d in b_env_deps if d not in return_values]
                            if not missing:
                                replay_logger.info(
                                    "  Boundary ENV deps resolved after %ds",
                                    _wait + 1,
                                )
                                break
                        if missing:
                            replay_logger.warning(
                                "  Boundary ENV deps still missing after 360s: %s",
                                ", ".join(d[:30] for d in missing),
                            )
                if rt > 1.0 and daemon_active and scenario_start_time is not None:
                    _collect_event_sim_times(events_path, fired_event_times)
                    parent_time = _compute_judge_parent_time(
                        boundary[0],
                        fired_event_times,
                        scenario_start_time,
                    )
                    if parent_time is not None:
                        target = parent_time + rt
                        replay_logger.info(
                            "  Scheduling boundary: rt=%.0fs, target=%.0f",
                            rt,
                            target - scenario_start_time,
                        )
                        _wait_for_faketime(target)
                msg_content = boundary[0]["args"].get("content", "")
                channel.send_response(msg_content)
                replay_logger.info(
                    "TURN BOUNDARY: send_message_to_user (%d chars) → events.jsonl + agent_responses.jsonl",
                    len(msg_content),
                )
                replay_logger.debug("  message: %s", msg_content[:200])
                echo(f"  -> send_message_to_user ({len(msg_content)} chars)")

            if daemon_active:
                replay_logger.info(
                    "Waiting %.1fs for daemon to process turn %d...",
                    turn_delay,
                    turn_idx,
                )
                time.sleep(turn_delay)

                # Scan events.jsonl for new ENV return values written by
                # daemon CLI calls (log_action includes event_id via
                # GAIA2_EVENT_ID env var).
                all_events = read_all_lines(events_path)
                env_events_seen = 0
                for ev in all_events:
                    eid = ev.get("event_id", "")
                    if eid and eid not in return_values:
                        ret = ev.get("ret")
                        return_values[eid] = ret
                        if eid.startswith("Event-ENV-"):
                            env_events_seen += 1
                            replay_logger.info(
                                "  ENV ret: %s → %r", eid, repr(ret)[:80]
                            )
                            echo(f"    env_ret: {eid} → {repr(ret)[:50]}")

                new_notifs = read_all_lines(notifications_path)[notifs_before:]
                if new_notifs or env_events_seen:
                    replay_logger.info(
                        "Daemon after turn %d: %d notifications, %d ENV events",
                        turn_idx,
                        len(new_notifs),
                        env_events_seen,
                    )
                    echo(f"  Notifications: {len(new_notifs) + env_events_seen}")
                    for n in new_notifs:
                        ntype = n.get("type", "?")
                        if ntype == "user_message":
                            replay_logger.info(
                                "  NOTIFICATION user_message: %s",
                                str(n.get("content", ""))[:120],
                            )
                            echo(f"    user_message: {str(n.get('content', ''))[:80]}")
                        elif ntype == "env_action":
                            replay_logger.info(
                                "  NOTIFICATION env_action: %s.%s(%s) result=%s",
                                n.get("app"),
                                n.get("function"),
                                ", ".join(str(k) for k in (n.get("args") or {}).keys()),
                                repr(n.get("result", ""))[:60],
                            )
                            echo(
                                f"    env_action: {n.get('app')}.{n.get('function')}()"
                            )
                        else:
                            replay_logger.info(
                                "  NOTIFICATION %s: %s",
                                ntype,
                                json.dumps(n, default=str)[:120],
                            )
                            echo(f"    {ntype}")
                else:
                    replay_logger.info("No daemon activity after turn %d", turn_idx)
                    echo("  (no daemon activity)")

            echo()

        # If the last turn had no send_message_to_user boundary (e.g.,
        # execution scenarios where the oracle only does tool calls), write
        # a synthetic boundary so the external daemon detects the turn end
        # and runs the judge.
        if daemon_active and oracle_turns:
            last_turn = oracle_turns[-1]
            has_boundary = any(e["fn"] == "send_message_to_user" for e in last_turn)
            if not has_boundary:
                replay_logger.info(
                    "Writing synthetic turn boundary (last turn had no "
                    "send_message_to_user)"
                )
                channel.send_response("[oracle replay complete]")
                echo("  -> synthetic send_message_to_user (turn boundary)")
                time.sleep(turn_delay)

    finally:
        replay_logger.info("Shutting down...")
        if daemon:
            daemon.shutdown()
        if daemon_thread:
            daemon_thread.join(timeout=2.0)
        replay_logger.info("Shutdown complete")

    # --- Summary ---
    events_written = read_all_lines(events_path)
    all_notifs = read_all_lines(notifications_path)

    echo("=" * 50)
    echo("REPLAY COMPLETE")
    echo(f"  Turns:          {len(oracle_turns)}")
    echo(f"  CLI calls ok:   {total_ok}")
    echo(f"  CLI calls err:  {total_err}")
    echo(f"  Events written: {len(events_written)}")
    if daemon_active:
        echo(f"  Notifications:  {len(all_notifs)}")
    echo(f"  State dir:      {state_path}")
    echo()
    if events_path.exists():
        echo(f"  events.jsonl:        {events_path.stat().st_size:,} bytes")
    if notifications_path.exists():
        echo(f"  notifications.jsonl: {notifications_path.stat().st_size:,} bytes")
    echo(f"  replay.log:          {log_path}")
    if daemon_active:
        echo(f"  eventd.log:          {state_path / 'eventd.log'}")

    replay_logger.info("=" * 50)
    replay_logger.info("REPLAY COMPLETE")
    replay_logger.info("  Turns: %d", len(oracle_turns))
    replay_logger.info("  CLI calls ok: %d, err: %d", total_ok, total_err)
    replay_logger.info("  Events written: %d", len(events_written))
    if with_daemon:
        replay_logger.info("  Notifications: %d", len(all_notifs))
    replay_logger.info("  State dir: %s", state_path)

    return total_err == 0


def _run_all(
    scenario_dir: str,
    turn_delay: float,
    with_daemon: bool,
    verbose: bool,
) -> None:
    """Batch-replay all scenario JSON files in a directory."""
    import shutil

    scenario_path = Path(scenario_dir)
    scenarios = sorted(scenario_path.glob("*.json"))
    if not scenarios:
        click.echo(f"No .json files found in {scenario_dir}", err=True)
        raise SystemExit(1)

    click.echo(f"Found {len(scenarios)} scenarios in {scenario_dir}")
    click.echo(f"Mode: {'with daemon' if with_daemon else 'replay only'}")
    click.echo()

    passed = 0
    failed = 0
    failures: list[tuple[str, str]] = []
    t0 = time.time()

    for i, s in enumerate(scenarios):
        name = s.stem[:4]
        state_dir = tempfile.mkdtemp(prefix=f"gaia2-replay-{name}-")
        try:
            ok = replay_scenario(
                scenario=str(s),
                state_dir=state_dir,
                turn_delay=turn_delay,
                with_daemon=with_daemon,
                verbose=verbose,
                quiet=True,
            )
        except Exception as exc:
            ok = False
            failures.append((s.stem, f"EXCEPTION: {exc}"))
        else:
            if not ok:
                # Read last line of replay.log for context
                log_path = Path(state_dir) / "replay.log"
                detail = ""
                if log_path.exists():
                    lines = log_path.read_text().strip().split("\n")
                    # Find last ERROR line
                    for line in reversed(lines):
                        if "ERROR" in line:
                            detail = line.split("ERROR", 1)[-1].strip()[:120]
                            break
                failures.append((s.stem, detail or "unknown error"))

        if ok:
            passed += 1
        else:
            failed += 1
            click.echo(f"  FAIL: {s.stem}")

        # Clean up temp dirs for passing scenarios
        if ok:
            shutil.rmtree(state_dir, ignore_errors=True)

        # Progress every 20 scenarios
        if (i + 1) % 20 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (len(scenarios) - i - 1) / rate if rate > 0 else 0
            click.echo(
                f"  [{i + 1:3d}/{len(scenarios)}] {passed} pass, {failed} fail "
                f"({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)"
            )

    elapsed = time.time() - t0
    click.echo()
    click.echo("=" * 60)
    click.echo(
        f"Results: {passed}/{len(scenarios)} PASS, {failed} FAIL "
        f"({elapsed:.1f}s total, {elapsed / len(scenarios):.1f}s/scenario)"
    )

    if failures:
        click.echo(f"\nFailed scenarios ({len(failures)}):")
        for name, detail in failures:
            click.echo(f"  {name}: {detail}")

    raise SystemExit(0 if failed == 0 else 1)


@click.command()
@click.option(
    "--scenario",
    default=None,
    type=click.Path(exists=True),
    help="Path to scenario JSON file.",
)
@click.option(
    "--scenario-dir",
    default=None,
    type=click.Path(exists=True),
    help="Directory of scenario JSON files (batch mode).",
)
@click.option(
    "--state-dir",
    default=None,
    type=click.Path(),
    help="State directory (default: auto-created temp dir).",
)
@click.option(
    "--turn-delay",
    default=1.0,
    type=float,
    help="Seconds to wait between turns (for daemon processing).",
)
@click.option(
    "--with-daemon",
    is_flag=True,
    default=False,
    help="Also start gaia2-eventd daemon in background thread.",
)
@click.option("-v", "--verbose", is_flag=True, help="Verbose logging.")
def main(
    scenario: str | None,
    scenario_dir: str | None,
    state_dir: str | None,
    turn_delay: float,
    with_daemon: bool,
    verbose: bool,
) -> None:
    """Replay scenario oracle actions through the gaia2-eventd pipeline.

    Parses a Gaia2 scenario JSON, extracts oracle AGENT events grouped by
    turn, and replays them by calling CLI tools — simulating a perfect agent.

    \b
    Single scenario:
        gaia2-replay --scenario scenario.json
        gaia2-replay --scenario scenario.json --with-daemon

    Batch (all scenarios in a directory):
        gaia2-replay --scenario-dir path/to/validation/ --with-daemon
    """
    if scenario_dir:
        _run_all(scenario_dir, turn_delay, with_daemon, verbose)
    elif scenario:
        ok = replay_scenario(scenario, state_dir, turn_delay, with_daemon, verbose)
        raise SystemExit(0 if ok else 1)
    else:
        click.echo("Error: provide --scenario or --scenario-dir", err=True)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
