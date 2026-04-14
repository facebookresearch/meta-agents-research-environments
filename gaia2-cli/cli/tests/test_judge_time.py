# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
"""Tests for event time checking in the gaia2-core judge.

The judge uses ``event_relative_time`` (from the scenario JSON) as the
oracle's expected delay from its parent event.  The agent's actual delay
is the wall-clock gap between its event and its matched parent.
Only events with ``event_relative_time > threshold`` are time-checked.
"""

from __future__ import annotations

from gaia2_core.judge.judge import Judge
from gaia2_core.types import (
    CompletedEvent,
    EventAction,
    OracleEvent,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _agent_event(
    event_id: str = "agent-1",
    class_name: str = "Calendar",
    function_name: str = "add_calendar_event",
    args: dict | None = None,
    event_time: float = 100.0,
    operation_type: str = "write",
) -> CompletedEvent:
    return CompletedEvent(
        event_id=event_id,
        event_type="AGENT",
        event_time=event_time,
        action=EventAction(
            app_name=class_name,
            class_name=class_name,
            function_name=function_name,
            args=args or {},
            operation_type=operation_type,
        ),
    )


def _oracle_event(
    event_id: str = "oracle-1",
    class_name: str = "Calendar",
    function_name: str = "add_calendar_event",
    args: dict | None = None,
    event_relative_time: float | None = None,
) -> OracleEvent:
    action = EventAction(
        app_name=class_name,
        class_name=class_name,
        function_name=function_name,
        args=args or {},
        operation_type="write",
    )
    return OracleEvent(
        event_id=event_id,
        event_type="AGENT",
        action=action,
        args=args or {},
        event_relative_time=event_relative_time,
    )


def _make_judge(**kwargs) -> Judge:
    defaults = {
        "turn_to_oracle_events": [[]],
        "turn_to_oracle_graph": [{}],
        "tasks": ["test"],
        "start_time": 0.0,
        "check_time_threshold_seconds": 1.0,
        "pre_event_tolerance_seconds": 10.0,
        "post_event_tolerance_seconds": 25.0,
        "time_check_warn_only": False,
    }
    defaults.update(kwargs)
    return Judge(**defaults)


# ---------------------------------------------------------------------------
# TestCheckEventTime
# ---------------------------------------------------------------------------


class TestCheckEventTime:
    """Test the core tolerance comparison."""

    def test_within_tolerance(self) -> None:
        judge = _make_judge()
        assert judge._check_event_time(60.0, 60.0) is True

    def test_too_early(self) -> None:
        judge = _make_judge(pre_event_tolerance_seconds=10.0)
        assert judge._check_event_time(45.0, 60.0) is False

    def test_too_late(self) -> None:
        judge = _make_judge(post_event_tolerance_seconds=25.0)
        assert judge._check_event_time(90.0, 60.0) is False

    def test_at_pre_boundary(self) -> None:
        judge = _make_judge(pre_event_tolerance_seconds=10.0)
        assert judge._check_event_time(50.0, 60.0) is True

    def test_at_post_boundary(self) -> None:
        judge = _make_judge(post_event_tolerance_seconds=25.0)
        assert judge._check_event_time(85.0, 60.0) is True


# ---------------------------------------------------------------------------
# TestCheckTime
# ---------------------------------------------------------------------------


class TestCheckTime:
    """Test _check_time using event_relative_time."""

    def test_no_relative_time_skipped(self) -> None:
        judge = _make_judge()
        agent = _agent_event(event_time=500.0)
        oracle = _oracle_event(event_relative_time=None)
        assert judge._check_time(agent, oracle, 0.0) is True

    def test_small_delay_skipped(self) -> None:
        judge = _make_judge(check_time_threshold_seconds=1.0)
        agent = _agent_event(event_time=500.0)
        oracle = _oracle_event(event_relative_time=1.0)
        assert judge._check_time(agent, oracle, 0.0) is True

    def test_within_tolerance(self) -> None:
        """Agent 65s after parent, oracle expects 60s → pass."""
        judge = _make_judge()
        agent = _agent_event(event_time=165.0)
        oracle = _oracle_event(event_relative_time=60.0)
        # agent_relative = 165 - 100 = 65, within [50, 85]
        assert judge._check_time(agent, oracle, 100.0) is True

    def test_too_late(self) -> None:
        judge = _make_judge()
        agent = _agent_event(event_time=200.0)
        oracle = _oracle_event(event_relative_time=60.0)
        # agent_relative = 200 - 100 = 100, > 85
        assert judge._check_time(agent, oracle, 100.0) is False

    def test_too_early(self) -> None:
        judge = _make_judge()
        agent = _agent_event(event_time=145.0)
        oracle = _oracle_event(event_relative_time=60.0)
        # agent_relative = 145 - 100 = 45, < 50
        assert judge._check_time(agent, oracle, 100.0) is False


# ---------------------------------------------------------------------------
# TestMatchWithTimeCheck
# ---------------------------------------------------------------------------


class TestMatchWithTimeCheck:
    """Test time checking integrated with event matching."""

    def _run_judge(
        self,
        oracle_events: list[OracleEvent],
        agent_events: list[CompletedEvent],
        oracle_graph: dict[str, list[str]] | None = None,
        **judge_kwargs,
    ) -> tuple[bool, str]:
        if oracle_graph is None:
            oracle_graph = {oe.event_id: [] for oe in oracle_events}
        judge = _make_judge(
            turn_to_oracle_events=[oracle_events],
            turn_to_oracle_graph=[oracle_graph],
            **judge_kwargs,
        )
        result = judge.judge_turn(0, agent_events)
        return result.success, result.failure_reason

    def test_rejects_wrong_time(self) -> None:
        """Agent does right action but too late → fail."""
        oracle1 = _oracle_event(
            event_id="o1",
            class_name="EmailClientV2",
            function_name="send_email",
            event_relative_time=1.0,
        )
        oracle2 = _oracle_event(
            event_id="o2",
            event_relative_time=60.0,
        )
        oracle_graph = {"o1": [], "o2": ["o1"]}

        agent1 = _agent_event(
            event_id="a1",
            class_name="EmailClientV2",
            function_name="send_email",
            event_time=5.0,
        )
        agent2 = _agent_event(event_id="a2", event_time=205.0)  # 200s after a1
        success, reason = self._run_judge(
            [oracle1, oracle2], [agent1, agent2], oracle_graph=oracle_graph
        )
        assert success is False
        assert "time check failed" in reason

    def test_passes_with_correct_time(self) -> None:
        """Agent does right action at right time → pass."""
        oracle1 = _oracle_event(
            event_id="o1",
            class_name="EmailClientV2",
            function_name="send_email",
            event_relative_time=1.0,
        )
        oracle2 = _oracle_event(
            event_id="o2",
            event_relative_time=60.0,
        )
        oracle_graph = {"o1": [], "o2": ["o1"]}

        agent1 = _agent_event(
            event_id="a1",
            class_name="EmailClientV2",
            function_name="send_email",
            event_time=5.0,
        )
        agent2 = _agent_event(event_id="a2", event_time=70.0)  # 65s after a1
        success, _ = self._run_judge(
            [oracle1, oracle2], [agent1, agent2], oracle_graph=oracle_graph
        )
        assert success is True

    def test_time_warn_only(self) -> None:
        oracle1 = _oracle_event(
            event_id="o1",
            class_name="EmailClientV2",
            function_name="send_email",
            event_relative_time=1.0,
        )
        oracle2 = _oracle_event(event_id="o2", event_relative_time=60.0)
        oracle_graph = {"o1": [], "o2": ["o1"]}

        agent1 = _agent_event(
            event_id="a1",
            class_name="EmailClientV2",
            function_name="send_email",
            event_time=5.0,
        )
        agent2 = _agent_event(event_id="a2", event_time=500.0)
        success, _ = self._run_judge(
            [oracle1, oracle2],
            [agent1, agent2],
            oracle_graph=oracle_graph,
            time_check_warn_only=True,
        )
        assert success is True

    def test_no_relative_time_unaffected(self) -> None:
        """Events without event_relative_time match normally."""
        oracle = _oracle_event(event_id="o1", event_relative_time=None)
        agent = _agent_event(event_id="a1", event_time=999.0)
        success, _ = self._run_judge([oracle], [agent])
        assert success is True

    def test_small_relative_time_not_checked(self) -> None:
        """rt ≤ threshold passes regardless of agent time."""
        oracle = _oracle_event(event_id="o1", event_relative_time=1.0)
        agent = _agent_event(event_id="a1", event_time=500.0)
        success, _ = self._run_judge([oracle], [agent])
        assert success is True

    def test_env_parent_time_checked(self) -> None:
        """Events with ENV parents are time-checked using ENV timestamps.

        The judge records ENV event times from events.jsonl.  AGENT
        events whose dependencies are ENV events use those timestamps
        as the parent reference for time checking.
        """
        # Oracle: AGENT event depends on ENV event, rt=30
        oracle = _oracle_event(
            event_id="o1",
            event_relative_time=30.0,
        )
        oracle.dependency_ids = ["Event-ENV-abc"]
        oracle_graph = {"o1": []}  # ENV deps filtered from graph

        judge = _make_judge(
            turn_to_oracle_events=[[oracle]],
            turn_to_oracle_graph=[oracle_graph],
        )
        # Simulate ENV event arriving at t=100
        env_event = _agent_event(
            event_id="Event-ENV-abc",
            class_name="EmailClientV2",
            function_name="create_and_add_email",
            event_time=100.0,
            operation_type="write",
        )
        env_event = CompletedEvent(
            event_id="Event-ENV-abc",
            event_type="ENV",
            event_time=100.0,
            action=env_event.action,
        )
        # Agent acts at t=135 → 35s after ENV → within [20, 55] → pass
        agent = _agent_event(event_id="a1", event_time=135.0)
        result = judge.judge_turn(0, [env_event, agent])
        assert result.success is True

    def test_env_parent_too_late_fails(self) -> None:
        """Agent acts too late after ENV notification → fail."""
        oracle = _oracle_event(
            event_id="o1",
            event_relative_time=30.0,
        )
        oracle.dependency_ids = ["Event-ENV-abc"]
        oracle_graph = {"o1": []}

        judge = _make_judge(
            turn_to_oracle_events=[[oracle]],
            turn_to_oracle_graph=[oracle_graph],
        )
        env_event = CompletedEvent(
            event_id="Event-ENV-abc",
            event_type="ENV",
            event_time=100.0,
            action=None,
        )
        # Agent acts at t=200 → 100s after ENV → outside [20, 55] → fail
        agent = _agent_event(event_id="a1", event_time=200.0)
        result = judge.judge_turn(0, [env_event, agent])
        assert result.success is False
        assert "time check failed" in result.failure_reason

    def test_user_parent_uses_start_time(self) -> None:
        """Events with USER parents use start_time as reference.

        Root timed events (e.g. 'send a message every 30s') depend on
        the USER event.  Their parent time is start_time.
        """
        oracle = _oracle_event(
            event_id="o1",
            event_relative_time=60.0,
        )
        oracle.dependency_ids = ["Event-USER-abc"]
        oracle_graph = {"o1": []}

        judge = _make_judge(
            turn_to_oracle_events=[[oracle]],
            turn_to_oracle_graph=[oracle_graph],
            start_time=1000.0,
        )
        # Agent acts at t=1070 → 70s after start → within [50, 85] → pass
        agent = _agent_event(event_id="a1", event_time=1070.0)
        result = judge.judge_turn(0, [agent])
        assert result.success is True

    def test_user_parent_too_late_fails(self) -> None:
        """Agent acts too late from start_time → fail."""
        oracle = _oracle_event(
            event_id="o1",
            event_relative_time=60.0,
        )
        oracle.dependency_ids = ["Event-USER-abc"]
        oracle_graph = {"o1": []}

        judge = _make_judge(
            turn_to_oracle_events=[[oracle]],
            turn_to_oracle_graph=[oracle_graph],
            start_time=1000.0,
        )
        # Agent acts at t=1200 → 200s after start → outside [50, 85] → fail
        agent = _agent_event(event_id="a1", event_time=1200.0)
        result = judge.judge_turn(0, [agent])
        assert result.success is False

    def test_mixed_agent_env_parent(self) -> None:
        """Event with both AGENT and ENV parents uses AGENT parent time.

        When both exist, AGENT parents are found first (from oracle
        graph), so the AGENT parent time is used.
        """
        oracle1 = _oracle_event(
            event_id="o1",
            class_name="CabApp",
            function_name="order_ride",
            event_relative_time=1.0,
        )
        oracle2 = _oracle_event(
            event_id="o2",
            event_relative_time=30.0,
        )
        oracle2.dependency_ids = ["Event-ENV-abc", "o1"]
        oracle_graph = {"o1": [], "o2": ["o1"]}  # only AGENT dep in graph

        judge = _make_judge(
            turn_to_oracle_events=[[oracle1, oracle2]],
            turn_to_oracle_graph=[oracle_graph],
        )
        agent1 = _agent_event(
            event_id="a1",
            class_name="CabApp",
            function_name="order_ride",
            event_time=5.0,
        )
        # Agent acts at t=40 → 35s after AGENT parent → within [20, 55] → pass
        agent2 = _agent_event(event_id="a2", event_time=40.0)
        result = judge.judge_turn(0, [agent1, agent2])
        assert result.success is True

    def test_no_parent_no_deps_skips(self) -> None:
        """Event with no dependencies at all skips time check."""
        oracle = _oracle_event(
            event_id="o1",
            event_relative_time=60.0,
        )
        # No dependency_ids
        oracle_graph = {"o1": []}

        judge = _make_judge(
            turn_to_oracle_events=[[oracle]],
            turn_to_oracle_graph=[oracle_graph],
        )
        agent = _agent_event(event_id="a1", event_time=999.0)
        result = judge.judge_turn(0, [agent])
        assert result.success is True
