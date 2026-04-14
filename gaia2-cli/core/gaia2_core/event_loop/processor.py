# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
"""Core simulation engine for Gaia2 scenarios.

Executor-agnostic event processor that handles turn index assignment,
turn trigger insertion, event scheduling, ENV event interception, and
placeholder resolution.  Returns pending ENV actions as tuples —
callers decide how to execute them (CLI subprocess, in-process method
call, etc.).

Used by both the gaia2-cli daemon (eventd) and the GAIA2 framework.
"""

from __future__ import annotations

import heapq
import logging
from collections import defaultdict, deque
from typing import TYPE_CHECKING, Any

from gaia2_core.types import (
    CompletedEvent,
    ConditionCheck,
    ScenarioEvent,
    resolve_placeholders,
)

if TYPE_CHECKING:
    from gaia2_core.loader import ScenarioLoader

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_oracle_send_message_to_user(event: ScenarioEvent) -> bool:
    """Check if a scenario event is an ORACLE send_message_to_user."""
    return (
        event.event_type == "AGENT"
        and event.action is not None
        and event.action.app_name == "AgentUserInterface"
        and event.action.function_name == "send_message_to_user"
    )


def _is_completed_send_message_to_user(event: CompletedEvent) -> bool:
    """Check if a completed event is a send_message_to_user."""
    return (
        event.event_type == "AGENT"
        and event.action is not None
        and event.action.function_name == "send_message_to_user"
    )


# ---------------------------------------------------------------------------
# EventProcessor
# ---------------------------------------------------------------------------


class EventProcessor:
    """Core simulation engine for Gaia2 scenarios.

    Handles:
    - Turn index assignment (BFS over dependency DAG)
    - Turn trigger insertion (ConditionCheck events)
    - Event scheduling (time-based priority queue)
    - ENV event interception (returns pending actions for caller execution)
    - Placeholder resolution

    This class is **executor-agnostic**: ``tick()`` returns pending ENV
    actions as ``(app_class, function_name, resolved_args, event_id)``
    tuples.  The caller decides how to execute them — CLI subprocess
    (eventd), direct Python call (environment.py), etc.
    """

    def __init__(
        self,
        events: list[ScenarioEvent],
        start_time: float,
        time_increment: float,
        app_name_to_class: dict[str, str],
        duration: float | None = None,
        judge: Any | None = None,
    ) -> None:
        self._events = events
        self._events_by_id: dict[str, ScenarioEvent] = {e.event_id: e for e in events}
        self.start_time = start_time
        self.duration = duration
        self.time_increment = time_increment
        self.app_name_to_class = app_name_to_class

        # Runtime state
        self._current_time: float = start_time
        self._completed: dict[str, CompletedEvent] = {}

        # Priority queue: (event_time, counter, item)
        # item is either ScenarioEvent or ConditionCheck
        self._queue: list[tuple[float, int, ScenarioEvent | ConditionCheck]] = []
        self._counter: int = 0

        # Turn trigger data
        self._condition_checks: dict[str, ConditionCheck] = {}
        self._event_id_to_turn_idx: dict[str, int] = {}
        self.nb_turns: int = 0

        # Judge (optional)
        self._judge = judge
        self._stopped: bool = False
        self._last_judgment: Any | None = None  # last JudgmentResult from judge

        # IDs of shadow copies created by _apply_id_remapping.  Used to
        # exclude remapped duplicates when counting SMUs or collecting
        # turn events — without relying on event_id naming conventions.
        self._remapped_ids: set[str] = set()

    @classmethod
    def from_loader(
        cls,
        loader: ScenarioLoader,
        judge: Any | None = None,
    ) -> EventProcessor:
        """Create an EventProcessor from a ScenarioLoader instance."""
        return cls(
            events=loader.events,
            start_time=loader.start_time,
            time_increment=loader.time_increment,
            app_name_to_class=loader.app_name_to_class,
            judge=judge,
        )

    @property
    def stopped(self) -> bool:
        """True if the judge rejected a turn."""
        return self._stopped

    @property
    def event_id_to_turn_idx(self) -> dict[str, int]:
        return self._event_id_to_turn_idx

    @property
    def all_conditions_fired(self) -> bool:
        """True when all ConditionCheck events have fired (scenario complete)."""
        if not self._condition_checks:
            return True  # vacuous truth — no conditions means nothing to wait for
        return all(cid in self._completed for cid in self._condition_checks)

    @property
    def next_event_time(self) -> float | None:
        """Return the time of the next queued event, or None if queue is empty."""
        return self._queue[0][0] if self._queue else None

    # ------------------------------------------------------------------
    # Turn assignment
    # ------------------------------------------------------------------

    def build_event_id_to_turn_idx(self) -> None:
        """BFS to assign turn indices.

        The turn of an event is the number of send_message_to_user ORACLE
        events among its ancestors (matching the GAIA2 framework logic).
        """
        visited: set[str] = set()
        turn_idx: dict[str, int] = defaultdict(int)

        # Start from events with no dependencies
        queue: deque[str] = deque()
        for e in self._events:
            if not e.dependency_ids:
                queue.append(e.event_id)

        while queue:
            eid = queue.popleft()
            if eid in visited:
                continue
            event = self._events_by_id.get(eid)
            if event is None:
                continue

            visited.add(eid)
            turn_idx[eid] = 0

            if event.dependency_ids:
                # Max turn index among dependencies
                turn_idx[eid] = max(
                    turn_idx.get(dep_id, 0) for dep_id in event.dependency_ids
                )
                # If any dependency is send_message_to_user, increment
                for dep_id in event.dependency_ids:
                    dep = self._events_by_id.get(dep_id)
                    if dep and _is_oracle_send_message_to_user(dep):
                        turn_idx[eid] = max(
                            turn_idx[eid],
                            turn_idx.get(dep_id, 0) + 1,
                        )

            # Schedule successors
            for succ_id in event.successor_ids:
                if succ_id not in visited:
                    # Only add if all deps visited
                    succ = self._events_by_id.get(succ_id)
                    if succ and all(d in visited for d in succ.dependency_ids):
                        queue.append(succ_id)

        self._event_id_to_turn_idx = dict(turn_idx)
        self.nb_turns = (max(turn_idx.values()) + 1) if turn_idx else 1

        logger.info(
            "Turn assignment: %d events, %d turns",
            len(self._event_id_to_turn_idx),
            self.nb_turns,
        )

    # ------------------------------------------------------------------
    # Turn triggers
    # ------------------------------------------------------------------

    def build_turn_triggers(self) -> None:
        """Insert ConditionCheck events at turn boundaries.

        For each turn boundary (turn 1..nb_turns-1):
        1. Find the send_message_to_user ORACLE event that ends that turn
        2. Steal its successors (non-ORACLE only)
        3. Create a ConditionCheck that fires when the completed log has
           the right number of send_message_to_user events
        4. Chain conditions: condition_turn_2 depends on condition_turn_1
        """
        self.build_event_id_to_turn_idx()

        if self.nb_turns <= 1:
            logger.info("Single-turn scenario, no turn triggers needed")
            return

        # Find end-of-turn ORACLE events (send_message_to_user) by turn
        end_of_turn: dict[int, ScenarioEvent] = {}
        for event in self._events:
            if _is_oracle_send_message_to_user(event):
                tidx = self._event_id_to_turn_idx.get(event.event_id, 0)
                end_of_turn[tidx] = event

        prev_condition_id: str | None = None

        for turn_idx in range(1, self.nb_turns):
            eot_event = end_of_turn.get(turn_idx - 1)
            if eot_event is None:
                logger.warning("No end-of-turn event for turn %d", turn_idx - 1)
                continue

            # Steal successors from the end-of-turn event
            # Only take non-ORACLE successors
            successor_ids: list[str] = []
            for succ_id in eot_event.successor_ids:
                succ = self._events_by_id.get(succ_id)
                if succ and succ.event_type != "AGENT":
                    successor_ids.append(succ_id)
                    # Remove the end-of-turn event from successor's deps
                    succ.dependency_ids = [
                        d
                        for d in succ.dependency_ids
                        if not (
                            self._events_by_id.get(d) is not None
                            and _is_oracle_send_message_to_user(self._events_by_id[d])
                        )
                    ]

            condition_id = f"condition_turn_{turn_idx}"
            dep_ids = [prev_condition_id] if prev_condition_id else []

            condition = ConditionCheck(
                event_id=condition_id,
                turn_idx=turn_idx,
                event_time=self.start_time,
                schedule_every_secs=self.time_increment,
                successor_ids=successor_ids,
                dependency_ids=dep_ids,
            )
            self._condition_checks[condition_id] = condition

            # Chain: previous condition schedules this one as a successor
            if prev_condition_id and prev_condition_id in self._condition_checks:
                self._condition_checks[prev_condition_id].successor_ids.append(
                    condition_id
                )

            prev_condition_id = condition_id

            logger.info(
                "Turn trigger: %s (turn_idx=%d, successors=%d, deps=%s)",
                condition_id,
                turn_idx,
                len(successor_ids),
                dep_ids,
            )

    # ------------------------------------------------------------------
    # Event scheduling
    # ------------------------------------------------------------------

    def _push(self, item: ScenarioEvent | ConditionCheck, at_time: float) -> None:
        heapq.heappush(self._queue, (at_time, self._counter, item))
        self._counter += 1

    def schedule_initial_events(self) -> None:
        """Put USER/ENV events and turn-1 conditions into the queue.

        All USER and ENV events are scheduled at ``start_time + delay``
        unless they are condition successors (those fire via
        ``_schedule_successors`` when the condition passes). AGENT/ORACLE
        events are oracle data — never scheduled.
        """
        condition_successors: set[str] = set()
        for cond in self._condition_checks.values():
            condition_successors.update(cond.successor_ids)

        for event in self._events:
            if event.event_type == "AGENT":
                continue
            if event.event_id in condition_successors:
                continue
            delay = event.event_relative_time or 0.0
            t = self.start_time + delay
            self._push(event, t)
            logger.debug("Scheduled: %s at %.2f", event.event_id, t)

        # Schedule conditions with no dependencies (first turn trigger)
        for cond in self._condition_checks.values():
            if not cond.dependency_ids:
                self._push(cond, cond.event_time)
                logger.debug(
                    "Scheduled condition: %s at %.2f", cond.event_id, cond.event_time
                )

    # ------------------------------------------------------------------
    # Agent event registration
    # ------------------------------------------------------------------

    def add_agent_event(self, completed: CompletedEvent) -> None:
        """Register a completed agent action.

        The caller is responsible for invoking any judge hooks at turn
        boundaries. The processor only records the completed event so
        placeholder resolution and turn collection can see it.
        """
        self._completed[completed.event_id] = completed
        logger.debug(
            "Registered agent event: %s (%s)",
            completed.event_id,
            completed.action.function_name if completed.action else "?",
        )

    # ------------------------------------------------------------------
    # Tick — the core loop
    # ------------------------------------------------------------------

    def tick(self) -> list[tuple[str, str, dict[str, Any], str]]:
        """Evaluate conditions and drain events at _current_time.

        Does NOT advance time — call ``tick_to()`` for that.
        Conditions scheduled at or before _current_time are evaluated,
        and any triggered ENV/USER successors are processed.

        Returns a list of pending ENV actions:
            [(app_class_name, function_name, resolved_args, event_id), ...]
        """
        return self._drain_queue()

    def tick_to(self, target_time: float) -> list[tuple[str, str, dict[str, Any], str]]:
        """Advance _current_time to *target_time* and drain the queue.

        Used by the daemon's time advancement loop for Time scenarios.
        """
        self._current_time = target_time
        return self._drain_queue()

    def _drain_queue(self) -> list[tuple[str, str, dict[str, Any], str]]:
        """Drain the event queue up to _current_time.

        Returns a list of pending ENV actions.
        """
        pending_env_actions: list[tuple[str, str, dict[str, Any], str]] = []
        reschedule: list[tuple[float, int, ScenarioEvent | ConditionCheck]] = []

        while self._queue:
            if self._stopped:
                break
            t, cnt, item = self._queue[0]
            if t > self._current_time:
                break
            heapq.heappop(self._queue)

            if isinstance(item, ConditionCheck):
                if item.event_id in self._completed:
                    continue  # Already fired — skip duplicate
                self._process_condition(item, pending_env_actions, reschedule)
            elif isinstance(item, ScenarioEvent):
                self._process_scenario_event(item, pending_env_actions, reschedule)

        # Re-add rescheduled items
        for entry in reschedule:
            heapq.heappush(self._queue, entry)

        if pending_env_actions:
            logger.info(
                "Tick: %d ENV actions, queue=%d, completed=%d",
                len(pending_env_actions),
                len(self._queue),
                len(self._completed),
            )
        return pending_env_actions

    def _process_condition(
        self,
        condition: ConditionCheck,
        pending: list[tuple[str, str, dict[str, Any], str]],
        reschedule: list[tuple[float, int, ScenarioEvent | ConditionCheck]],
    ) -> None:
        """Evaluate a turn-boundary condition check."""
        condition.check_count += 1

        # Count send_message_to_user in completed events
        # Exclude remapped shadow copies created by _apply_id_remapping
        # — they duplicate real agent SMUs and would inflate the count,
        # causing premature turn evaluation.
        nb_smu = sum(
            1
            for ev in self._completed.values()
            if _is_completed_send_message_to_user(ev)
            and ev.event_id not in self._remapped_ids
        )

        if nb_smu >= condition.turn_idx:
            # Turn reached — collect agent events and optionally judge
            logger.info(
                "Condition %s PASSED (smu=%d, turn_idx=%d)",
                condition.event_id,
                nb_smu,
                condition.turn_idx,
            )

            # Judge the turn if judge is present.
            # condition.turn_idx = N means "N SMUs seen". The agent events
            # for this turn are everything before the Nth SMU, which is
            # turn index N-1 in the split. Oracle turns are 0-indexed.
            if self._judge is not None:
                judge_turn = condition.turn_idx - 1
                turn_events = self.collect_turn_agent_events(judge_turn)
                result = self._judge.judge_turn(judge_turn, turn_events)
                self._last_judgment = result
                if not result.success:
                    logger.error(
                        "JUDGE REJECTED turn %d: %s — stopping",
                        condition.turn_idx,
                        result.failure_reason,
                    )
                    self._stopped = True
                    return
                if result.agent_event_id_to_oracle_event_id:
                    self._apply_id_remapping(result.agent_event_id_to_oracle_event_id)

            # Mark condition as completed so dependent conditions can fire
            self._completed[condition.event_id] = CompletedEvent(
                event_id=condition.event_id,
                event_type="CONDITION",
                event_time=self._current_time,
                action=None,
            )
            self._schedule_successors(condition.successor_ids, pending)
        else:
            # Not yet — reschedule
            new_time = condition.event_time + (
                condition.check_count * condition.schedule_every_secs
            )
            reschedule.append((new_time, self._counter, condition))
            self._counter += 1
            logger.debug(
                "Condition %s rescheduled (smu=%d < turn_idx=%d, next=%.2f)",
                condition.event_id,
                nb_smu,
                condition.turn_idx,
                new_time,
            )

    def _schedule_successors(
        self,
        successor_ids: list[str],
        pending: list[tuple[str, str, dict[str, Any], str]],
    ) -> None:
        """Schedule successor events (ENV/USER) after a condition passes."""
        for succ_id in successor_ids:
            event = self._events_by_id.get(succ_id)
            if event is None:
                # Could be a condition check
                cond = self._condition_checks.get(succ_id)
                if cond:
                    self._push(cond, self._current_time)
                continue

            delay = event.event_relative_time or 0.0
            t = self._current_time + delay

            if event.event_type == "ENV":
                if delay > 0:
                    # Queue ENV event for its scheduled time
                    event.event_time = t
                    self._push(event, t)
                else:
                    # Zero delay: fire immediately
                    self._process_user_env_event(event, pending)
            elif event.event_type == "USER":
                # Process through _process_user_env_event so that
                # send_message_to_agent actions are added to pending
                # and routed to the agent by the daemon.
                # Successor scheduling is handled inside _process_user_env_event.
                self._process_user_env_event(event, pending)
            else:
                self._push(event, t)

    def _process_scenario_event(
        self,
        event: ScenarioEvent,
        pending: list[tuple[str, str, dict[str, Any], str]],
        reschedule: list | None = None,
    ) -> None:
        """Process a scheduled ScenarioEvent (ENV or USER)."""
        if event.event_type == "ENV":
            self._process_user_env_event(event, pending, reschedule=reschedule)
        elif event.event_type == "USER":
            t = event.event_time or self._current_time
            self._completed[event.event_id] = CompletedEvent(
                event_id=event.event_id,
                event_type="USER",
                event_time=t,
                action=event.action,
            )
            self._schedule_successors(event.successor_ids, pending)

    def _process_user_env_event(
        self,
        event: ScenarioEvent,
        pending: list[tuple[str, str, dict[str, Any], str]],
        reschedule: list | None = None,
    ) -> None:
        """Process a USER or ENV event: resolve args, add to pending, mark completed."""
        if event.event_id in self._completed:
            return  # Already fired — skip duplicate
        if event.action is None:
            logger.warning(
                "%s event %s has no action, skipping", event.event_type, event.event_id
            )
            return

        # Check if all dependency events are completed. If not, reschedule
        # so dependent events fire after their prerequisites within the same
        # tick (e.g., add_product must complete before add_item_to_product).
        if reschedule is not None and event.dependency_ids:
            missing = [d for d in event.dependency_ids if d not in self._completed]
            if missing:
                logger.debug(
                    "ENV event %s has %d unresolved deps, rescheduling",
                    event.event_id[:30],
                    len(missing),
                )
                reschedule.append((self._current_time, self._counter, event))
                self._counter += 1
                return

        # Resolve placeholders using completed events
        resolved_args = resolve_placeholders(event.action.args, self._completed)
        # Remove 'self' arg if present
        resolved_args.pop("self", None)

        # Use app_name (instance name) for routing. This is critical
        # for apps like MessagingAppV2 where the same class backs different
        # instances: "Messages" vs "Chats". Callers use app_name to
        # dispatch to the right executor.
        app_class = event.action.app_name
        if not app_class:
            app_class = event.action.class_name or ""

        pending.append(
            (
                app_class,
                event.action.function_name,
                resolved_args,
                event.event_id,
            )
        )

        # Mark as completed so successors and placeholders can resolve
        t = event.event_time or self._current_time
        self._completed[event.event_id] = CompletedEvent(
            event_id=event.event_id,
            event_type=event.event_type,
            event_time=t,
            action=event.action,
        )

        # Schedule successors
        self._schedule_successors(event.successor_ids, pending)

    # ------------------------------------------------------------------
    # Judge helpers
    # ------------------------------------------------------------------

    def judge_final_turn(self) -> Any:
        """Judge the final turn (nb_turns - 1). Returns None if no judge."""
        if self._judge is None:
            return None
        final_turn = self.nb_turns - 1
        turn_events = self.collect_turn_agent_events(final_turn)
        result = self._judge.judge_turn(final_turn, turn_events)
        if result.success:
            self._apply_id_remapping(result.agent_event_id_to_oracle_event_id)
        return result

    def collect_turn_agent_events(self, turn_idx: int) -> list[CompletedEvent]:
        """Collect agent events for the given turn.

        Groups agent events by turn boundary (send_message_to_user count).
        The judge filters to writes-only internally.
        """
        # Collect all agent events in order, assign to turns.
        # Exclude remapped shadow copies — they duplicate real agent
        # events and would create spurious turn boundaries.
        agent_events: list[CompletedEvent] = []
        for ev in sorted(self._completed.values(), key=lambda e: e.event_time):
            if ev.event_type not in ("AGENT", "ENV"):
                continue
            if ev.action is None:
                continue
            if ev.event_id in self._remapped_ids:
                continue
            agent_events.append(ev)

        # Split into turns by send_message_to_user
        turns: list[list[CompletedEvent]] = [[]]
        for ev in agent_events:
            turns[-1].append(ev)
            if (
                ev.action is not None
                and ev.action.function_name == "send_message_to_user"
            ):
                turns.append([])

        if turn_idx < len(turns):
            return turns[turn_idx]
        return []

    def _apply_id_remapping(self, mapping: dict[str, str]) -> None:
        """Add CompletedEvent copies under oracle IDs for placeholder resolution.

        For each agent_id → oracle_id, register a copy of the agent's
        CompletedEvent under the oracle_id so that ENV placeholder
        resolution (``{{oracle_id}}``) finds the agent's return value.
        """
        for agent_id, oracle_id in mapping.items():
            agent_ev = self._completed.get(agent_id)
            if agent_ev is None:
                continue
            # Register under oracle ID (don't overwrite if already present)
            if oracle_id not in self._completed:
                self._completed[oracle_id] = CompletedEvent(
                    event_id=oracle_id,
                    event_type=agent_ev.event_type,
                    event_time=agent_ev.event_time,
                    action=agent_ev.action,
                    return_value=agent_ev.return_value,
                )
                self._remapped_ids.add(oracle_id)
                logger.debug(
                    "ID remap: %s → %s (ret=%s)",
                    agent_id,
                    oracle_id,
                    (
                        str(agent_ev.return_value)[:60]
                        if agent_ev.return_value
                        else "None"
                    ),
                )
