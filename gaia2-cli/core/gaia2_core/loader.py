# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
"""Lightweight scenario JSON parser.

Parses Gaia2 scenario JSON directly with no app registry or framework imports.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict, deque
from typing import Any

from gaia2_core.types import EventAction, OracleEvent, ScenarioEvent, UserDetails

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def _parse_args(args_list: list[dict[str, Any]]) -> dict[str, Any]:
    """Convert Gaia2 JSON args list to a flat dict.

    Gaia2 format:  [{"name": "to", "value": "alice", "value_type": "str"}, ...]
    Output:       {"to": "alice", ...}

    Mirrors ``Scenario._parse_parameter_value`` from
    ``gaia2/scenarios/scenario.py`` — handles Optional[T] wrappers and
    deserialises list/dict via ``json.loads``.
    """
    import re

    result = {}
    for arg in args_list:
        name = arg["name"]
        value = arg.get("value")
        vtype = arg.get("value_type")
        if not vtype:
            # value_type is null or missing — infer from the value itself
            if isinstance(value, str) and value.startswith(("[", "{")):
                vtype = "list" if value.startswith("[") else "dict"
            else:
                vtype = "str"

        if value is not None:
            # Unwrap Optional[T] → T, then list[X] → list, dict[X,Y] → dict
            m = re.match(r"Optional\[(.+)\]", vtype)
            base_type = m.group(1) if m else vtype
            # Strip type parameters: list[str] → list, dict[str,int] → dict
            base_type = re.sub(r"\[.*\]", "", base_type)

            # Skip placeholder references — resolved later
            if isinstance(value, str) and value.startswith("{{"):
                result[name] = value
                continue

            try:
                if base_type == "int":
                    value = int(value)
                elif base_type == "float":
                    value = float(value)
                elif base_type == "bool":
                    value = str(value).lower() in ("true", "1", "t", "y", "yes")
                elif base_type in ("list", "dict"):
                    value = json.loads(value) if isinstance(value, str) else value
            except (ValueError, TypeError, json.JSONDecodeError):
                pass  # keep original value on parse failure

        result[name] = value
    return result


# ---------------------------------------------------------------------------
# ScenarioLoader
# ---------------------------------------------------------------------------


class ScenarioLoader:
    """Parse a Gaia2 scenario JSON directly.

    No Gaia2 app registry, no framework imports.
    """

    def __init__(self, scenario_path: str) -> None:
        with open(scenario_path) as f:
            data = json.load(f)

        self._data = data
        self._events_raw = data.get("events", [])

        # Build app_name -> class_name mapping from the apps section
        self.app_name_to_class: dict[str, str] = {}
        for app in data.get("apps", []):
            name = app.get("name", "")
            cls = app.get("class_name", name)
            if name:
                self.app_name_to_class[name] = cls

        # Parse events
        self.events: list[ScenarioEvent] = []
        self._events_by_id: dict[str, ScenarioEvent] = {}
        self._parse_events()

        # Extract task
        self.task: str | None = self._extract_task()

        # Metadata
        metadata = data.get("metadata", {}).get("definition", {})
        self.start_time: float = float(
            metadata.get("start_time") or data.get("start_time") or 0.0
        )
        self.duration: float = float(
            metadata.get("duration") or data.get("duration") or 86400.0
        )
        self.time_increment: float = float(
            metadata.get("time_increment_in_seconds")
            or data.get("time_increment_in_seconds")
            or 1.0
        )
        self.nb_turns: int = 0  # set by build_event_id_to_turn_idx

        logger.info(
            "ScenarioLoader: %d events, %d apps, start=%.0f, duration=%.0f, time_increment=%.1f",
            len(self.events),
            len(self.app_name_to_class),
            self.start_time,
            self.duration,
            self.time_increment,
        )

    def _parse_events(self) -> None:
        """Parse raw JSON events into ScenarioEvent objects."""
        for ev in self._events_raw:
            action = None
            raw_action = ev.get("action")
            if raw_action:
                app_name = raw_action.get("app", "")
                class_name = raw_action.get("class_name", "")
                if not class_name:
                    class_name = self.app_name_to_class.get(app_name, app_name)
                args = _parse_args(raw_action.get("args", []))
                operation_type = (raw_action.get("operation_type") or "read").lower()
                action = EventAction(
                    app_name=app_name,
                    class_name=class_name,
                    function_name=raw_action.get("function", ""),
                    args=args,
                    operation_type=operation_type,
                )

            event = ScenarioEvent(
                event_id=ev["event_id"],
                event_type=ev.get("event_type", ""),
                event_time=ev.get("event_time"),
                event_relative_time=ev.get("event_relative_time"),
                action=action,
                dependency_ids=ev.get("dependencies", []),
            )
            self.events.append(event)
            self._events_by_id[event.event_id] = event

        # Build successor_ids (reverse of dependency_ids)
        for event in self.events:
            for dep_id in event.dependency_ids:
                dep = self._events_by_id.get(dep_id)
                if dep is not None:
                    dep.successor_ids.append(event.event_id)

    def _extract_task(self) -> str | None:
        """Extract the initial user task from the first USER send_message_to_agent event.

        In multi-turn scenarios, events are stored in DAG order (not
        chronological).  The initial user message is the one with no
        dependencies.  Falls back to the first USER event if none have
        empty deps.
        """
        fallback: str | None = None
        for ev in self.events:
            if ev.event_type == "USER" and ev.action:
                if ev.action.function_name == "send_message_to_agent":
                    if fallback is None:
                        fallback = ev.action.args.get("content")
                    if not ev.dependency_ids:
                        return ev.action.args.get("content")
        return fallback

    # ------------------------------------------------------------------
    # Oracle data extraction (for the judge)
    # ------------------------------------------------------------------

    def extract_oracle_data(
        self,
        event_id_to_turn_idx: dict[str, int],
        nb_turns: int,
    ) -> tuple[
        list[list[OracleEvent]],
        list[dict[str, list[str]]],
        list[str],
        UserDetails | None,
    ]:
        """Extract oracle data needed by the Judge.

        Uses the already-computed *event_id_to_turn_idx* mapping from
        ``EventProcessor.build_event_id_to_turn_idx()``.

        Returns:
            turn_to_oracle_events: per-turn lists of OracleEvent (topo-sorted)
            turn_to_oracle_graph: per-turn causal graphs {event_id: [parent_ids]}
            tasks: per-turn task strings
            user_details: UserDetails from contacts app initial state (or None)
        """
        # --- Filter AGENT oracle events by turn ---
        agent_events_by_turn: dict[int, list[ScenarioEvent]] = defaultdict(list)
        for ev in self.events:
            if ev.event_type != "AGENT":
                continue
            tidx = event_id_to_turn_idx.get(ev.event_id)
            if tidx is None:
                continue
            agent_events_by_turn[tidx].append(ev)

        turn_to_oracle_events: list[list[OracleEvent]] = []
        turn_to_oracle_graph: list[dict[str, list[str]]] = []

        for tidx in range(nb_turns):
            turn_events = agent_events_by_turn.get(tidx, [])
            target_ids = {ev.event_id for ev in turn_events}

            # Build causal graph (oracle→oracle deps only, within this turn)
            graph: dict[str, list[str]] = {eid: [] for eid in target_ids}
            for ev in turn_events:
                graph[ev.event_id] = [
                    dep for dep in ev.dependency_ids if dep in target_ids
                ]

            # Topological sort
            sorted_ids = _topological_sort(graph)

            # Convert to OracleEvent
            oracle_events: list[OracleEvent] = []
            for eid in sorted_ids:
                sev = self._events_by_id[eid]
                oracle_events.append(
                    OracleEvent(
                        event_id=sev.event_id,
                        event_type=sev.event_type,
                        action=sev.action,
                        args=sev.action.args if sev.action else {},
                        event_time=sev.event_time,
                        event_relative_time=sev.event_relative_time,
                        dependency_ids=list(sev.dependency_ids),
                    )
                )

            turn_to_oracle_events.append(oracle_events)
            turn_to_oracle_graph.append(graph)

        # --- Extract per-turn tasks from USER send_message_to_agent ---
        turn_idx_to_task: dict[int, str] = defaultdict(str)
        for ev in self.events:
            if (
                ev.event_type == "USER"
                and ev.action
                and ev.action.function_name == "send_message_to_agent"
            ):
                tidx = event_id_to_turn_idx.get(ev.event_id, 0)
                content = ev.action.args.get("content", "")
                if content:
                    turn_idx_to_task[tidx] += content + "\n"

        tasks = [turn_idx_to_task.get(i, "") for i in range(nb_turns)]

        # --- Extract UserDetails from contacts initial state ---
        user_details = self._extract_user_details()

        logger.info(
            "Oracle data: %d turns, %d total oracle events, user=%s",
            nb_turns,
            sum(len(t) for t in turn_to_oracle_events),
            (
                f"{user_details.first_name} {user_details.last_name}"
                if user_details
                else "None"
            ),
        )

        return turn_to_oracle_events, turn_to_oracle_graph, tasks, user_details

    def _extract_user_details(self) -> UserDetails | None:
        """Best-effort extraction of user details from scenario JSON.

        This is a fallback — the daemon dumps ``user_details.json`` from
        the container's runtime contacts state, which is the canonical
        source.  The JSON structure varies across dataset formats, so
        this method checks ``app_state`` (GAIA2) first.
        """
        apps = self._data.get("apps", [])
        for app in apps:
            cls = app.get("class_name", app.get("name", ""))
            if cls not in ("ContactsApp", "Contacts", "InternalContacts"):
                continue
            contacts = app.get("app_state", {}).get("contacts", {})
            for contact in contacts.values():
                if contact.get("is_user"):
                    return UserDetails(
                        first_name=contact.get("first_name", ""),
                        last_name=contact.get("last_name", ""),
                        address=contact.get("address", ""),
                    )
        return None


# ---------------------------------------------------------------------------
# Topological sort (Kahn's algorithm)
# ---------------------------------------------------------------------------


def _topological_sort(graph: dict[str, list[str]]) -> list[str]:
    """BFS-based topological sort. *graph* maps event_id → [parent_ids]."""
    in_degree: dict[str, int] = defaultdict(int)
    adj: dict[str, list[str]] = defaultdict(list)

    for node, parents in graph.items():
        for parent in parents:
            adj[parent].append(node)
            in_degree[node] += 1

    queue: deque[str] = deque(node for node in graph if in_degree[node] == 0)
    result: list[str] = []

    while queue:
        node = queue.popleft()
        result.append(node)
        for child in adj[node]:
            in_degree[child] -= 1
            if in_degree[child] == 0:
                queue.append(child)

    if len(result) != len(graph):
        raise ValueError("Oracle event graph contains a cycle")

    return result
