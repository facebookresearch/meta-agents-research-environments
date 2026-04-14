# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
"""Judge orchestrator — ports GraphPerEventJudge from gaia2.validation.judge.

Called by EventProcessor at each turn boundary to validate agent actions
against oracle expectations.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any, Callable

from gaia2_core.types import (
    CompletedEvent,
    JudgmentResult,
    OracleEvent,
    UserDetails,
)

logger = logging.getLogger(__name__)


class Judge:
    """Per-event judge matching Gaia2 ``GraphPerEventJudge`` semantics.

    Validates that each turn's agent events match the oracle events
    using hard checkers (pure Python) and optional soft checkers (LLM).
    """

    def __init__(
        self,
        turn_to_oracle_events: list[list[OracleEvent]],
        turn_to_oracle_graph: list[dict[str, list[str]]],
        tasks: list[str],
        user_details: UserDetails | None = None,
        start_time: float = 0.0,
        engine: Callable | None = None,
        extra_send_message_to_user_allowed: int = 1,
        app_name_to_class: dict[str, str] | None = None,
        state_dir: str | Path | None = None,
        judge_mode: str = "eval",
        check_time_threshold_seconds: float = 1.0,
        pre_event_tolerance_seconds: float = 10.0,
        post_event_tolerance_seconds: float = 25.0,
        time_check_warn_only: bool = False,
    ) -> None:
        self.turn_to_oracle_events = turn_to_oracle_events
        self.turn_to_oracle_graph = turn_to_oracle_graph
        self.tasks = tasks
        self.user_details = user_details
        self.start_time = start_time
        self.extra_smu_allowed = extra_send_message_to_user_allowed
        self.app_name_to_class = app_name_to_class or {}
        self._judgments_path = (
            Path(state_dir) / "judgments.jsonl" if state_dir else None
        )
        self._check_time_threshold = check_time_threshold_seconds
        self._pre_event_tolerance = pre_event_tolerance_seconds
        self._post_event_tolerance = post_event_tolerance_seconds
        self._time_check_warn_only = time_check_warn_only

        # ENV return values — populated by the daemon after executing ENV CLIs.
        # Used by _resolve_oracle_placeholders for {{Event-ENV-xxx}} refs.
        self.env_return_values: dict[str, Any] = {}

        # Build checker registries
        from gaia2_core.judge.config import build_checker_registries

        self._arg_registry, self._soft_registry = build_checker_registries(
            mode=judge_mode
        )

        # Build LLM checkers (None if no engine)
        self._llm_checkers: dict | None = None
        if engine is not None:
            try:
                from gaia2_core.judge.checkers import build_llm_checkers

                self._llm_checkers = build_llm_checkers(engine)
            except Exception as e:
                logger.warning("Failed to build LLM checkers: %s", e)

        # Cumulative state across turns
        self._all_agent_events: list[CompletedEvent] = []
        self._agent_idx_to_oracle_id: dict[int, str] = {}
        self._oracle_id_to_agent_idx: dict[str, int] = {}
        self._agent_id_to_oracle_id: dict[str, str] = {}
        # ENV event times — used as parent references for time checking
        self._env_event_times: dict[str, float] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def judge_turn(
        self,
        turn_idx: int,
        agent_events: list[CompletedEvent],
    ) -> JudgmentResult:
        """Judge a single turn. Called by EventProcessor.

        Returns JudgmentResult with success flag and ID mapping.
        """
        # Record ENV event times before filtering — the time checker
        # needs them as parent references for AGENT events that depend
        # on ENV events (e.g. "reply 30s after receiving an email").
        for ev in agent_events:
            if ev.event_type == "ENV" and ev.event_id:
                self._env_event_times[ev.event_id] = ev.event_time

        # Filter to write-only AGENT events, matching the original Gaia2
        # framework's AgentEventFilter.  Reads are exploratory and not
        # part of the oracle contract.  ENV/USER events are excluded
        # since our oracle only contains AGENT events.
        agent_events = [
            ev
            for ev in agent_events
            if ev.action is not None
            and ev.action.operation_type == "write"
            and ev.event_type not in ("ENV", "USER")
        ]

        if turn_idx >= len(self.turn_to_oracle_events):
            logger.warning(
                "Turn %d beyond oracle data (%d turns), auto-pass",
                turn_idx,
                len(self.turn_to_oracle_events),
            )
            return JudgmentResult(success=True)

        oracle_events = self.turn_to_oracle_events[turn_idx]
        oracle_graph = self.turn_to_oracle_graph[turn_idx]

        if not oracle_events:
            logger.info("Turn %d: no oracle events, auto-pass", turn_idx)
            return JudgmentResult(success=True)

        # Normalize agent event tool names using app_name_to_class
        self._normalize_agent_tool_names(agent_events)

        # Record cumulative agent events (offset = previous count)
        base_idx = len(self._all_agent_events)
        self._all_agent_events.extend(agent_events)

        logger.info(
            "JUDGE turn %d: %d agent events, %d oracle events",
            turn_idx,
            len(agent_events),
            len(oracle_events),
        )

        # 1. Preliminary checks — tool call counts
        prelim = self._preliminary_checks(agent_events, oracle_events)
        if prelim is not True:
            reason = prelim  # it's the failure string
            logger.info(
                "JUDGE REJECT turn %d: preliminary checks failed: %s", turn_idx, reason
            )
            result = JudgmentResult(success=False, failure_reason=str(reason))
            self._log_judgment(turn_idx, result, agent_events, oracle_events)
            return result

        # 2. Match each oracle event (topo-sorted).
        # Don't short-circuit on first failure — match all events so the
        # partial ID mapping covers as many placeholders as possible.
        match_details: list[dict] = []
        first_failure: str | None = None
        for oracle_event in oracle_events:
            # Resolve placeholders in oracle args using matched agent returns
            oracle_event = self._resolve_oracle_placeholders(oracle_event)

            # Try to match
            matched, reason, judge_output = self._match_oracle_event(
                oracle_event,
                oracle_graph,
                agent_events,
                base_idx,
                turn_idx,
            )
            match_details.append(
                {
                    "oracle_id": oracle_event.event_id,
                    "oracle_tool": oracle_event.tool_name,
                    "matched": matched,
                    "reason": reason,
                    "judge_output": judge_output,
                }
            )
            if not matched:
                logger.info(
                    "JUDGE REJECT turn %d: %s — %s",
                    turn_idx,
                    oracle_event.tool_name,
                    reason,
                )
                if first_failure is None:
                    first_failure = reason
                # Continue matching remaining oracle events so that
                # partial ID remapping covers as many placeholders as
                # possible for ENV events in subsequent turns.

        mapping = dict(self._agent_id_to_oracle_id)

        if first_failure is not None:
            result = JudgmentResult(
                success=False,
                failure_reason=first_failure,
                agent_event_id_to_oracle_event_id=mapping,
            )
            self._log_judgment(
                turn_idx, result, agent_events, oracle_events, match_details
            )
            return result
        logger.info("JUDGE ACCEPT turn %d: all oracle events matched", turn_idx)
        result = JudgmentResult(
            success=True,
            agent_event_id_to_oracle_event_id=mapping,
        )
        self._log_judgment(turn_idx, result, agent_events, oracle_events, match_details)
        return result

    # ------------------------------------------------------------------
    # Judgment logging
    # ------------------------------------------------------------------

    def _log_judgment(
        self,
        turn_idx: int,
        result: JudgmentResult,
        agent_events: list[CompletedEvent],
        oracle_events: list[OracleEvent],
        match_details: list[dict] | None = None,
    ) -> None:
        """Append structured judgment record to judgments.jsonl."""
        if self._judgments_path is None:
            return
        record = {
            "turn": turn_idx,
            "success": result.success,
            "failure_reason": result.failure_reason or None,
            "agent_events": [
                {
                    "id": e.event_id,
                    "tool": e.tool_name,
                    "args": {
                        k: str(v)
                        if not isinstance(v, (str, int, float, bool, type(None)))
                        else v
                        for k, v in (e.get_args() or {}).items()
                    },
                    "args_summary": {
                        k: (str(v)[:80] if isinstance(v, str) else v)
                        for k, v in (e.get_args() or {}).items()
                    },
                }
                for e in agent_events
            ],
            "oracle_events": [
                {
                    "id": e.event_id,
                    "tool": e.tool_name,
                    "args": {
                        k: str(v)
                        if not isinstance(v, (str, int, float, bool, type(None)))
                        else v
                        for k, v in (e.get_args() or {}).items()
                    },
                    "args_summary": {
                        k: (str(v)[:80] if isinstance(v, str) else v)
                        for k, v in (e.get_args() or {}).items()
                    },
                }
                for e in oracle_events
            ],
            "match_details": match_details,
            "id_mapping": result.agent_event_id_to_oracle_event_id or None,
        }
        try:
            with open(self._judgments_path, "a") as f:
                f.write(json.dumps(record, default=str) + "\n")
            logger.debug("Judgment logged to %s", self._judgments_path)
        except Exception as e:
            logger.warning("Failed to write judgment: %s", e)

    # ------------------------------------------------------------------
    # Preliminary checks
    # ------------------------------------------------------------------

    def _preliminary_checks(
        self,
        agent_events: list[CompletedEvent],
        oracle_events: list[OracleEvent],
    ) -> bool | str:
        """Check tool call counts match. Returns True or failure string.

        send_message_to_user is allowed to appear more times than in
        the oracle (controlled by ``extra_smu_allowed``) because the
        adapter may inject turn-boundary AUI events.  But oracle SMU
        events still go through the full matching pipeline so their
        content gets validated by the user_message_checker.
        """
        aui_name = "AgentUserInterface__send_message_to_user"

        agent_counter = Counter(e.tool_name for e in agent_events)
        oracle_counter = Counter(e.tool_name for e in oracle_events)

        agent_aui = agent_counter.get(aui_name, 0)
        oracle_aui = oracle_counter.get(aui_name, 0)

        # Allow agent to have extra SMU events (adapter turn boundaries)
        if not (oracle_aui <= agent_aui <= oracle_aui + self.extra_smu_allowed):
            return (
                f"SMU count mismatch: agent={agent_aui}, oracle={oracle_aui}, "
                f"allowed_extra={self.extra_smu_allowed}"
            )

        # Compare all other tool counts (excluding SMU tolerance)
        agent_other = Counter({k: v for k, v in agent_counter.items() if k != aui_name})
        oracle_other = Counter(
            {k: v for k, v in oracle_counter.items() if k != aui_name}
        )

        if agent_other != oracle_other:
            return (
                f"Tool count mismatch: agent={dict(agent_other)}, "
                f"oracle={dict(oracle_other)}"
            )

        return True

    # ------------------------------------------------------------------
    # Time checking
    #
    # Time scenarios have oracle events with event_relative_time > 1.0,
    # meaning "the agent should perform this action N seconds after its
    # parent event."  The judge enforces this with a tolerance window:
    #   [oracle_relative - 10s, oracle_relative + 25s]
    #
    # How it works:
    #   1. Skip if event_relative_time is None or <= 1.0 (not timed).
    #   2. Find the parent event's timestamp to use as the reference:
    #      - AGENT parents: from the oracle graph (AGENT→AGENT deps)
    #      - ENV parents: from _env_event_times (recorded before filtering)
    #      - USER parents: use start_time (initial message = scenario start)
    #   3. Compute agent_relative = agent_event_time - max_parent_time.
    #   4. Compare agent_relative against event_relative_time ± tolerance.
    #
    # All event_time values are in the scenario epoch (2024) because
    # the daemon parses sim_t from events.jsonl when building
    # CompletedEvent objects.  ENV event times come from the same source.
    # ------------------------------------------------------------------

    def _check_event_time(
        self,
        agent_time: float,
        oracle_time: float,
    ) -> bool:
        """Check if agent event time is within tolerance of oracle time."""
        return (
            agent_time >= oracle_time - self._pre_event_tolerance
            and agent_time <= oracle_time + self._post_event_tolerance
        )

    def _check_time(
        self,
        agent_event: CompletedEvent,
        oracle_event: OracleEvent,
        max_parent_agent_time: float,
    ) -> bool:
        """Check if the agent acted within the oracle's expected time window.

        Uses ``event_relative_time`` (from the scenario JSON) as the
        oracle's expected delay from its parent event.  The agent's
        actual delay is computed as the wall-clock gap between the agent
        event and its matched parent agent event.

        Only enforces timing when ``event_relative_time > threshold``.
        """
        oracle_relative = oracle_event.event_relative_time
        if oracle_relative is None or oracle_relative <= self._check_time_threshold:
            return True
        agent_relative = agent_event.event_time - max_parent_agent_time
        return self._check_event_time(agent_relative, oracle_relative)

    # ------------------------------------------------------------------
    # Oracle event matching
    # ------------------------------------------------------------------

    def _match_oracle_event(
        self,
        oracle_event: OracleEvent,
        oracle_graph: dict[str, list[str]],
        agent_events: list[CompletedEvent],
        base_idx: int,
        turn_idx: int,
    ) -> tuple[bool, str, str]:
        """Try to match an oracle event with an unmatched agent event."""
        if oracle_event.event_type in ("ENV", "USER"):
            return self._match_env_oracle_event(oracle_event, agent_events, base_idx)
        return self._match_agent_oracle_event(
            oracle_event, oracle_graph, agent_events, base_idx, turn_idx
        )

    def _match_env_oracle_event(
        self,
        oracle_event: OracleEvent,
        agent_events: list[CompletedEvent],
        base_idx: int,
    ) -> tuple[bool, str, str]:
        """Match ENV/USER oracle event by event_id equality."""
        for i, agent_event in enumerate(agent_events):
            if agent_event.event_id == oracle_event.event_id:
                global_idx = base_idx + i
                self._record_match(
                    global_idx, oracle_event.event_id, agent_event.event_id
                )
                return True, "", ""
        return False, f"ENV event {oracle_event.event_id} not found in agent events", ""

    def _match_agent_oracle_event(
        self,
        oracle_event: OracleEvent,
        oracle_graph: dict[str, list[str]],
        agent_events: list[CompletedEvent],
        base_idx: int,
        turn_idx: int,
    ) -> tuple[bool, str, str]:
        """Match AGENT oracle event using tool judges + causality + time."""
        from gaia2_core.judge.checkers import mild_compare

        oracle_tool = oracle_event.tool_name
        failures: list[str] = []
        last_judge_output = ""

        for i, agent_event in enumerate(agent_events):
            global_idx = base_idx + i

            # Skip already-matched agent events
            if global_idx in self._agent_idx_to_oracle_id:
                continue

            # Compare using mild_compare (hard + soft)
            matched, rationale, judge_output = mild_compare(
                agent_event=agent_event,
                oracle_event=oracle_event,
                arg_checker_registry=self._arg_registry,
                soft_checker_registry=self._soft_registry,
                llm_checkers=self._llm_checkers,
                tolerance_list_str=self._get_tolerance_list(),
                tasks=self.tasks[: turn_idx + 1] if self.tasks else None,
                user_details=self.user_details,
                scenario_start_time=self.start_time,
            )
            if judge_output:
                last_judge_output = judge_output

            if matched is None:
                # Inconclusive (LLM failure)
                failures.append(f"{agent_event.event_id}: inconclusive ({rationale})")
                continue

            if not matched:
                if agent_event.tool_name == oracle_tool:
                    failures.append(f"{agent_event.event_id}: {rationale}")
                continue

            # Check causality — all parent oracle events must be matched
            # to agent events with lower global index
            causality_ok = True
            parent_ids = oracle_graph.get(oracle_event.event_id, [])
            for parent_id in parent_ids:
                parent_agent_idx = self._oracle_id_to_agent_idx.get(parent_id)
                if parent_agent_idx is None or parent_agent_idx >= global_idx:
                    causality_ok = False
                    failures.append(
                        f"{agent_event.event_id}: causality violation "
                        f"(parent {parent_id} not matched before idx {global_idx})"
                    )
                    break

            if not causality_ok:
                continue

            # Check event timing using event_relative_time (the oracle's
            # expected delay from its parent) vs agent's wall-clock delay.
            if oracle_event.event_relative_time is not None:
                max_parent_agent_time = self.start_time
                has_parent = False
                # Check AGENT parents (from oracle graph)
                for pid in parent_ids:
                    agent_idx = self._oracle_id_to_agent_idx.get(pid)
                    if agent_idx is not None:
                        has_parent = True
                        max_parent_agent_time = max(
                            max_parent_agent_time,
                            self._all_agent_events[agent_idx].event_time,
                        )
                # Check ENV parents (from original dependency_ids)
                for dep_id in oracle_event.dependency_ids:
                    env_time = self._env_event_times.get(dep_id)
                    if env_time is not None:
                        has_parent = True
                        max_parent_agent_time = max(max_parent_agent_time, env_time)
                # USER parents — use start_time as reference (the
                # initial message fires at scenario start).
                if not has_parent and oracle_event.dependency_ids:
                    has_parent = True

                if has_parent:
                    time_ok = self._check_time(
                        agent_event, oracle_event, max_parent_agent_time
                    )
                    if not time_ok:
                        if self._time_check_warn_only:
                            logger.warning(
                                "Time check failed for %s (warn-only)",
                                oracle_event.event_id,
                            )
                        else:
                            agent_rel = agent_event.event_time - max_parent_agent_time
                            failures.append(
                                f"{agent_event.event_id}: time check failed "
                                f"(agent_relative={agent_rel:.0f}s, "
                                f"oracle_relative="
                                f"{oracle_event.event_relative_time:.0f}s)"
                            )
                            continue

            # Match found
            self._record_match(global_idx, oracle_event.event_id, agent_event.event_id)
            return True, "", last_judge_output

        # No match
        failure_detail = "; ".join(failures) if failures else "no candidates"
        return (
            False,
            f"Cannot match {oracle_tool} ({oracle_event.event_id}): {failure_detail}",
            last_judge_output,
        )

    # ------------------------------------------------------------------
    # Placeholder resolution
    # ------------------------------------------------------------------

    def _resolve_oracle_placeholders(self, oracle_event: OracleEvent) -> OracleEvent:
        """Replace {{oracle_id}} refs in oracle args with matched agent return values."""
        if not oracle_event.args:
            return oracle_event

        resolved = {}
        for arg_name, arg_value in oracle_event.args.items():
            if not isinstance(arg_value, str) or "{{" not in arg_value:
                resolved[arg_name] = arg_value
                continue

            # Try to resolve {{event_id}} or {{event_id.key}}
            import re

            match = re.match(r"^\{\{(.*?)\}\}$", arg_value.strip())
            if not match:
                resolved[arg_name] = arg_value
                continue

            parts = match.group(1).split(".")
            ref_oracle_id = parts[0]

            # Find the return value — check agent matches first, then ENV
            agent_idx = self._oracle_id_to_agent_idx.get(ref_oracle_id)
            if agent_idx is not None:
                value = self._all_agent_events[agent_idx].return_value
            elif ref_oracle_id in self.env_return_values:
                value = self.env_return_values[ref_oracle_id]
            else:
                logger.warning(
                    "PLACEHOLDER unresolved: %s (env_keys=%s, agent_keys=%s)",
                    ref_oracle_id,
                    list(self.env_return_values.keys())[:5],
                    list(self._oracle_id_to_agent_idx.keys())[:5],
                )
                resolved[arg_name] = arg_value
                continue

            for key in parts[1:]:
                if isinstance(value, dict) and key in value:
                    value = value[key]
                else:
                    value = arg_value  # keep original
                    break

            resolved[arg_name] = value

        oracle_event.resolved_args = resolved
        return oracle_event

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _normalize_agent_tool_names(self, events: list[CompletedEvent]) -> None:
        """Ensure agent events use class_name (not app alias) in tool_name."""
        for ev in events:
            if ev.action is None:
                continue
            # If class_name is an alias, map to canonical
            cls = self.app_name_to_class.get(ev.action.app_name)
            if cls and cls != ev.action.class_name:
                ev.action.class_name = cls

    def _get_tolerance_list(self) -> list[str]:
        """User name as tolerance for attendee checks."""
        if self.user_details is None:
            return []
        name = f"{self.user_details.first_name} {self.user_details.last_name}".strip()
        return [name] if name else []

    def _record_match(
        self, agent_idx: int, oracle_id: str, agent_event_id: str
    ) -> None:
        self._agent_idx_to_oracle_id[agent_idx] = oracle_id
        self._oracle_id_to_agent_idx[oracle_id] = agent_idx
        self._agent_id_to_oracle_id[agent_event_id] = oracle_id
        logger.debug(
            "MATCH: agent[%d] %s → oracle %s",
            agent_idx,
            agent_event_id,
            oracle_id,
        )
