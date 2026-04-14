# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
"""Unit tests for time advancement support (event_relative_time scheduling).

Tests the functionality in loader, processor, and daemon that enables
ENV events to fire at their scheduled simulated times instead of all at once.
"""

from __future__ import annotations

import time

import pytest
from gaia2_cli.scenario import EventAction, EventProcessor, ScenarioEvent

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_event(
    event_id: str,
    event_type: str = "ENV",
    event_relative_time: float | None = None,
    dependency_ids: list[str] | None = None,
    app_name: str = "RentAFlat",
    function_name: str = "update_apartment",
) -> ScenarioEvent:
    """Create a minimal ScenarioEvent for testing."""
    return ScenarioEvent(
        event_id=event_id,
        event_type=event_type,
        event_time=None,
        event_relative_time=event_relative_time,
        action=EventAction(
            app_name=app_name,
            class_name=app_name,
            function_name=function_name,
            args={"id": event_id},
            operation_type="write",
        ),
        dependency_ids=dependency_ids or [],
    )


def _make_processor(
    events: list[ScenarioEvent],
    start_time: float = 1000.0,
    time_increment: float = 1.0,
) -> EventProcessor:
    """Create a processor with the given events."""
    return EventProcessor(
        events=events,
        start_time=start_time,
        time_increment=time_increment,
        app_name_to_class={},
    )


# ---------------------------------------------------------------------------
# EventProcessor: next_event_time
# ---------------------------------------------------------------------------


class TestNextEventTime:
    def test_empty_queue(self):
        proc = _make_processor([])
        assert proc.next_event_time is None

    def test_returns_earliest_time(self):
        proc = _make_processor([])
        ev1 = _make_event("e1")
        ev2 = _make_event("e2")
        proc._push(ev1, 1050.0)
        proc._push(ev2, 1030.0)
        assert proc.next_event_time == 1030.0


# ---------------------------------------------------------------------------
# EventProcessor: tick and tick_to
# ---------------------------------------------------------------------------


class TestTick:
    def test_tick_does_not_advance_time(self):
        proc = _make_processor([], start_time=1000.0)
        proc.tick()
        assert proc._current_time == 1000.0

    def test_tick_drains_events_at_current_time(self):
        proc = _make_processor([], start_time=1000.0)
        ev = _make_event("e1")
        proc._push(ev, 1000.0)

        actions = proc.tick()
        assert len(actions) == 1
        assert actions[0][3] == "e1"

    def test_tick_does_not_drain_future_events(self):
        proc = _make_processor([], start_time=1000.0)
        ev = _make_event("e1")
        proc._push(ev, 1050.0)

        actions = proc.tick()
        assert actions == []
        assert proc.next_event_time == 1050.0


class TestTickTo:
    def test_advances_current_time(self):
        proc = _make_processor([], start_time=1000.0)
        proc.tick_to(1050.0)
        assert proc._current_time == 1050.0

    def test_drains_events_up_to_target(self):
        proc = _make_processor([], start_time=1000.0)

        ev1 = _make_event("e1")
        ev2 = _make_event("e2")
        ev3 = _make_event("e3")

        proc._push(ev1, 1010.0)
        proc._push(ev2, 1030.0)
        proc._push(ev3, 1060.0)

        # Tick to 1035 — should drain ev1 and ev2 but not ev3
        actions = proc.tick_to(1035.0)
        assert len(actions) == 2
        event_ids = [a[3] for a in actions]
        assert "e1" in event_ids
        assert "e2" in event_ids

        # ev3 still in queue
        assert proc.next_event_time == 1060.0

    def test_returns_empty_when_no_due_events(self):
        proc = _make_processor([], start_time=1000.0)

        ev = _make_event("e1")
        proc._push(ev, 1050.0)

        actions = proc.tick_to(1020.0)
        assert actions == []
        assert proc.next_event_time == 1050.0


# ---------------------------------------------------------------------------
# EventProcessor: _schedule_successors
# ---------------------------------------------------------------------------


class TestScheduleSuccessors:
    """Verify ENV events with delay > 0 are queued, zero-delay fire immediately."""

    def test_env_with_delay_queued(self):
        env_event = _make_event("env1", event_relative_time=30.0)
        proc = _make_processor([env_event], start_time=1000.0)

        pending: list = []
        proc._schedule_successors(["env1"], pending)

        # Should NOT be in pending (not fired immediately)
        assert len(pending) == 0
        # Should be in queue at _current_time + delay = 1030.0
        assert proc.next_event_time == 1030.0

    def test_env_with_zero_delay_fires_immediately(self):
        env_event = _make_event("env1", event_relative_time=0.0)
        proc = _make_processor([env_event], start_time=1000.0)

        pending: list = []
        proc._schedule_successors(["env1"], pending)

        # Should fire immediately (zero delay)
        assert len(pending) == 1
        assert pending[0][3] == "env1"

    def test_env_with_no_relative_time_fires_immediately(self):
        """event_relative_time=None treated as zero delay."""
        env_event = _make_event("env1", event_relative_time=None)
        proc = _make_processor([env_event], start_time=1000.0)

        pending: list = []
        proc._schedule_successors(["env1"], pending)

        assert len(pending) == 1
        assert pending[0][3] == "env1"

    def test_env_delay_relative_to_current_time_not_start(self):
        """event_relative_time is relative to _current_time (parent fire time), not start_time.

        Matches the Gaia2 framework's semantics in environment.py:
            suc.event_time = event.event_time + suc.event_relative_time
        """
        env_event = _make_event("env1", event_relative_time=30.0)
        proc = _make_processor([env_event], start_time=1000.0)

        # Simulate time having advanced (e.g., condition fired at turn 2)
        proc._current_time = 1500.0

        pending: list = []
        proc._schedule_successors(["env1"], pending)

        # Should be queued at _current_time + delay = 1530.0, NOT start_time + delay = 1030.0
        assert len(pending) == 0
        assert proc.next_event_time == 1530.0


# ---------------------------------------------------------------------------
# Integration: full scenario flow
# ---------------------------------------------------------------------------


class TestScenarioFlow:
    """End-to-end test of the processor flow."""

    def test_env_events_fire_at_correct_times(self):
        """ENV events fire at +10, +20, +30 seconds via tick_to."""
        user_event = _make_event(
            "user1",
            event_type="USER",
            app_name="AgentUserInterface",
            function_name="send_message_to_agent",
        )
        env1 = _make_event("env1", event_relative_time=10.0, dependency_ids=["user1"])
        env2 = _make_event("env2", event_relative_time=20.0, dependency_ids=["user1"])
        env3 = _make_event("env3", event_relative_time=30.0, dependency_ids=["user1"])

        user_event.successor_ids = ["env1", "env2", "env3"]

        proc = _make_processor(
            [user_event, env1, env2, env3],
            start_time=1000.0,
        )

        # Schedule the USER event at start_time
        proc._push(user_event, 1000.0)

        # tick at start_time: USER event fires, ENV events get queued
        actions = proc.tick()
        assert len(actions) == 0  # USER events auto-complete, no ENV actions yet

        # Verify ENV events are queued at correct times
        assert proc.next_event_time == 1010.0

        # Tick to 1010: env1 fires
        actions = proc.tick_to(1010.0)
        assert len(actions) == 1
        assert actions[0][3] == "env1"

        # Tick to 1020: env2 fires
        actions = proc.tick_to(1020.0)
        assert len(actions) == 1
        assert actions[0][3] == "env2"

        # Tick to 1030: env3 fires
        actions = proc.tick_to(1030.0)
        assert len(actions) == 1
        assert actions[0][3] == "env3"

        # Queue empty
        assert proc.next_event_time is None

    def test_all_env_events_fire_when_ticking_past_all(self):
        """tick_to past all event times drains them all at once."""
        user_event = _make_event(
            "user1",
            event_type="USER",
            app_name="AgentUserInterface",
            function_name="send_message_to_agent",
        )
        env1 = _make_event("env1", event_relative_time=10.0, dependency_ids=["user1"])
        env2 = _make_event("env2", event_relative_time=20.0, dependency_ids=["user1"])

        user_event.successor_ids = ["env1", "env2"]

        proc = _make_processor(
            [user_event, env1, env2],
            start_time=1000.0,
        )
        proc._push(user_event, 1000.0)

        # tick: USER fires, ENV events queued
        proc.tick()

        # tick_to far future: all ENV events drain at once
        actions = proc.tick_to(2000.0)
        assert len(actions) == 2
        event_ids = {a[3] for a in actions}
        assert event_ids == {"env1", "env2"}


# ---------------------------------------------------------------------------
# Replay: faketime synchronization
# ---------------------------------------------------------------------------


class TestFaketimeSync:
    """Test that the oracle replay waits for simulated time correctly."""

    def test_wait_for_faketime_returns_immediately_when_past(self, tmp_path):
        """_wait_for_faketime returns True instantly if faketime is already past target."""
        from gaia2_cli.daemon.replay import _wait_for_faketime

        faketime_file = tmp_path / "faketime.rc"
        faketime_file.write_text("2024-10-15 07:02:00\n")

        # Target is 07:01:00 — already past
        from datetime import datetime, timezone

        target = datetime(2024, 10, 15, 7, 1, 0, tzinfo=timezone.utc).timestamp()
        result = _wait_for_faketime(target, faketime_path=faketime_file, timeout=2.0)
        assert result is True

    def test_wait_for_faketime_waits_until_target(self, tmp_path):
        """_wait_for_faketime blocks until faketime.rc reaches the target."""
        import threading
        from datetime import datetime, timezone

        from gaia2_cli.daemon.replay import _wait_for_faketime

        faketime_file = tmp_path / "faketime.rc"
        faketime_file.write_text("2024-10-15 07:00:00\n")

        target = datetime(2024, 10, 15, 7, 0, 30, tzinfo=timezone.utc).timestamp()

        # Advance faketime in a background thread after 1s
        def advance():
            time.sleep(1.0)
            faketime_file.write_text("2024-10-15 07:00:30\n")

        t = threading.Thread(target=advance, daemon=True)
        t.start()

        result = _wait_for_faketime(
            target, faketime_path=faketime_file, timeout=5.0, poll_interval=0.2
        )
        assert result is True

    def test_read_faketime(self, tmp_path):
        """_read_faketime returns the current timestamp from faketime.rc."""
        from datetime import datetime, timezone

        from gaia2_cli.daemon.replay import _read_faketime

        faketime_file = tmp_path / "faketime.rc"
        faketime_file.write_text("2024-10-15 07:01:30\n")

        expected = datetime(2024, 10, 15, 7, 1, 30, tzinfo=timezone.utc).timestamp()
        result = _read_faketime(faketime_file)
        assert result == pytest.approx(expected, abs=1.0)

    def test_read_faketime_returns_none_when_missing(self, tmp_path):
        """_read_faketime returns None if file doesn't exist."""
        from gaia2_cli.daemon.replay import _read_faketime

        result = _read_faketime(tmp_path / "nonexistent.rc")
        assert result is None

    def test_relative_wait_between_events(self, tmp_path):
        """Oracle must wait relative to previous event, not scenario start.

        Scenario: two AGENT events in the same turn.
        - Event A: offset=52s from start, rt=2s
        - Event B: offset=82s from start, rt=30s (relative to A)

        If faketime is already at +117s (from previous turn), an absolute
        wait for start+82 returns instantly. The correct behavior is to
        wait for prev_event_time + 30s.

        This test demonstrates the bug: with absolute offsets, event B
        fires immediately after A (agent_relative=0s instead of 30s).
        """
        from datetime import datetime, timezone

        from gaia2_cli.daemon.replay import _read_faketime, _wait_for_faketime

        faketime_file = tmp_path / "faketime.rc"
        start_time = datetime(2024, 10, 15, 7, 0, 0, tzinfo=timezone.utc).timestamp()

        # Simulate faketime at +117s (daemon advanced during previous turns)
        faketime_file.write_text("2024-10-15 07:01:57\n")

        # BUG: absolute wait for start+82 = 07:01:22 — already past!
        target_absolute = start_time + 82
        result = _wait_for_faketime(
            target_absolute, faketime_path=faketime_file, timeout=2.0
        )
        assert result is True  # returns instantly — this is the bug

        # FIX: relative wait from previous event time
        prev_event_time = _read_faketime(faketime_file)  # 07:01:57
        assert prev_event_time is not None
        target_relative = prev_event_time + 30  # wait 30s from prev event
        # 07:01:57 + 30 = 07:02:27 — faketime is only at 07:01:57, so must wait
        result = _wait_for_faketime(
            target_relative, faketime_path=faketime_file, timeout=2.0
        )
        assert result is False  # times out because faketime doesn't advance — correct!


class TestReplayJudgeParentTime:
    """Replay scheduling should use the same parent-time rule as the judge."""

    def test_uses_agent_parent_from_oracle_graph(self):
        from gaia2_cli.daemon.replay import _compute_judge_parent_time

        entry = {
            "_agent_parents": ["oracle-parent"],
            "_env_deps": [],
            "_dependency_ids": ["oracle-parent"],
        }
        parent_time = _compute_judge_parent_time(
            entry,
            {"oracle-parent": 145.0},
            scenario_start_time=100.0,
        )
        assert parent_time == 145.0

    def test_uses_env_parent_from_direct_dependencies(self):
        from gaia2_cli.daemon.replay import _compute_judge_parent_time

        entry = {
            "_agent_parents": [],
            "_env_deps": ["Event-ENV-1"],
            "_dependency_ids": ["Event-ENV-1"],
        }
        parent_time = _compute_judge_parent_time(
            entry,
            {"Event-ENV-1": 160.0},
            scenario_start_time=100.0,
        )
        assert parent_time == 160.0

    def test_user_only_dependencies_fall_back_to_start_time(self):
        from gaia2_cli.daemon.replay import _compute_judge_parent_time

        entry = {
            "_agent_parents": [],
            "_env_deps": [],
            "_dependency_ids": ["Event-USER-1"],
        }
        parent_time = _compute_judge_parent_time(
            entry,
            {},
            scenario_start_time=100.0,
        )
        assert parent_time == 100.0

    def test_cross_turn_agent_dependency_is_ignored_like_judge(self):
        from gaia2_cli.daemon.replay import _compute_judge_parent_time

        entry = {
            # Judge only includes intra-turn AGENT parents here.
            "_agent_parents": [],
            "_env_deps": [],
            "_dependency_ids": ["OracleEvent-AGENT-other-turn"],
        }
        parent_time = _compute_judge_parent_time(
            entry,
            {"OracleEvent-AGENT-other-turn": 240.0},
            scenario_start_time=100.0,
        )
        assert parent_time == 100.0

    def test_uses_latest_of_agent_and_env_parents(self):
        from gaia2_cli.daemon.replay import _compute_judge_parent_time

        entry = {
            "_agent_parents": ["oracle-parent"],
            "_env_deps": ["Event-ENV-1"],
            "_dependency_ids": ["oracle-parent", "Event-ENV-1"],
        }
        parent_time = _compute_judge_parent_time(
            entry,
            {
                "oracle-parent": 140.0,
                "Event-ENV-1": 155.0,
            },
            scenario_start_time=100.0,
        )
        assert parent_time == 155.0

    def test_no_dependencies_skips_time_check(self):
        from gaia2_cli.daemon.replay import _compute_judge_parent_time

        entry = {
            "_agent_parents": [],
            "_env_deps": [],
            "_dependency_ids": [],
        }
        parent_time = _compute_judge_parent_time(
            entry,
            {},
            scenario_start_time=100.0,
        )
        assert parent_time is None
