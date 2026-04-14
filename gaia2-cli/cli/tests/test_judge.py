# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
"""Tests for the gaia2-cli judge system.

Covers hard checkers, preliminary checks, single/multi-turn matching,
placeholder resolution, causality enforcement, and graceful degradation.
"""

from __future__ import annotations

from gaia2_core.types import (
    CompletedEvent,
    EventAction,
    OracleEvent,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _agent_event(
    event_id: str,
    class_name: str,
    function_name: str,
    args: dict | None = None,
    return_value=None,
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
        return_value=return_value,
    )


def _oracle_event(
    event_id: str,
    class_name: str,
    function_name: str,
    args: dict | None = None,
    event_type: str = "AGENT",
    event_time: float | None = 100.0,
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
        event_type=event_type,
        action=action,
        args=args or {},
        event_time=event_time,
    )


# ===========================================================================
# Hard checker tests
# ===========================================================================


class TestHardCheckers:
    def test_eq_checker(self):
        from gaia2_core.judge.checkers import eq_checker

        assert eq_checker("abc", "abc") is True
        assert eq_checker("abc", "def") is False
        assert eq_checker(42, 42) is True
        assert eq_checker(None, None) is True
        assert eq_checker(None, "x") is False

    def test_unordered_list_checker(self):
        from gaia2_core.judge.checkers import unordered_list_checker

        assert unordered_list_checker(["a", "b"], ["b", "a"]) is True
        assert unordered_list_checker(["a"], ["b"]) is False
        assert unordered_list_checker(None, None) is True
        assert unordered_list_checker(None, []) is True
        assert unordered_list_checker([], None) is True

    def test_path_checker(self):
        from gaia2_core.judge.checkers import path_checker

        assert path_checker("/tmp/file.txt", "/tmp/file.txt") is True
        assert path_checker("/tmp/../tmp/file.txt", "/tmp/file.txt") is True
        assert path_checker("tmp/file.txt", "/tmp/file.txt") is True
        assert path_checker(None, None) is True
        assert path_checker(None, "/tmp") is False

    def test_unordered_path_list_checker(self):
        from gaia2_core.judge.checkers import unordered_path_list_checker

        assert (
            unordered_path_list_checker(
                ["/tmp/a.txt", "/tmp/b.txt"],
                ["/tmp/b.txt", "/tmp/a.txt"],
            )
            is True
        )
        assert unordered_path_list_checker(None, []) is True

    def test_datetime_checker(self):
        from gaia2_core.judge.checkers import datetime_checker

        assert datetime_checker("2024-01-15 10:30:00", "2024-01-15 10:30:00") is True
        assert datetime_checker("2024-01-15 10:30:00", "2024-01-15 11:30:00") is False
        assert datetime_checker(None, None) is True

    def test_eq_str_strip_checker(self):
        from gaia2_core.judge.checkers import eq_str_strip_checker

        assert eq_str_strip_checker("  hello  ", "hello") is True
        assert eq_str_strip_checker("hello", "world") is False
        assert eq_str_strip_checker("", "") is True

    def test_phone_number_checker(self):
        from gaia2_core.judge.checkers import phone_number_checker

        assert phone_number_checker("+1-555-123-4567", "15551234567") is True
        assert phone_number_checker("(555) 123-4567", "5551234567") is True
        assert phone_number_checker("5551234567", "5551234568") is False
        assert phone_number_checker(None, None) is True
        assert phone_number_checker(None, "123") is False
        # None and "" are both "no phone number"
        assert phone_number_checker(None, "") is True
        assert phone_number_checker("", None) is True
        assert phone_number_checker("", "") is True

    def test_list_attendees_checker(self):
        from gaia2_core.judge.checkers import list_attendees_checker

        # Basic unordered comparison
        assert list_attendees_checker(["alice", "bob"], ["bob", "alice"]) is True
        # Tolerance: oracle contains only user name → auto-pass
        assert (
            list_attendees_checker(
                ["alice"],
                ["John Doe"],
                tolerance_list_str=["john doe"],
            )
            is True
        )
        # Empty oracle → pass
        assert list_attendees_checker(["alice"], []) is True
        assert list_attendees_checker(["alice"], None) is True

    def test_contain_any_checker(self):
        from gaia2_core.judge.checkers import contain_any_checker

        assert contain_any_checker("hello world", ["hello", "xyz"]) is True
        assert contain_any_checker("hello world", ["xyz", "abc"]) is False

    def test_contain_all_checker(self):
        from gaia2_core.judge.checkers import contain_all_checker

        assert contain_all_checker("hello world foo", ["hello", "world"]) is True
        assert contain_all_checker("hello world", ["hello", "xyz"]) is False


# ===========================================================================
# hard_compare tests
# ===========================================================================


class TestHardCompare:
    def test_all_pass(self):
        from gaia2_core.judge.checkers import hard_compare
        from gaia2_core.judge.config import CheckerType

        passed, rationale = hard_compare(
            agent_args={"email_id": "e123", "content": "hello"},
            oracle_args={"email_id": "e123", "content": "hello"},
            arg_to_checker_type={
                "email_id": CheckerType.eq_checker,
                "content": CheckerType.llm_checker,  # skipped by hard_compare
            },
        )
        assert passed is True
        assert rationale == ""

    def test_eq_fail(self):
        from gaia2_core.judge.checkers import hard_compare
        from gaia2_core.judge.config import CheckerType

        passed, rationale = hard_compare(
            agent_args={"email_id": "e123"},
            oracle_args={"email_id": "e999"},
            arg_to_checker_type={"email_id": CheckerType.eq_checker},
        )
        assert passed is False
        assert "email_id" in rationale


# ===========================================================================
# Preliminary checks tests
# ===========================================================================


class TestPreliminaryChecks:
    def _make_judge(self, oracle_events, oracle_graph=None, extra_smu=1):
        from gaia2_core.judge.judge import Judge

        return Judge(
            turn_to_oracle_events=[oracle_events],
            turn_to_oracle_graph=[oracle_graph or {}],
            tasks=["Do something"],
            extra_send_message_to_user_allowed=extra_smu,
        )

    def test_matching_counts(self):
        judge = self._make_judge(
            [
                _oracle_event("o1", "ContactsApp", "add_new_contact"),
                _oracle_event("o2", "AgentUserInterface", "send_message_to_user"),
            ]
        )
        agent_events = [
            _agent_event("a1", "ContactsApp", "add_new_contact"),
            _agent_event("a2", "AgentUserInterface", "send_message_to_user"),
        ]
        result = judge._preliminary_checks(agent_events, judge.turn_to_oracle_events[0])
        assert result is True

    def test_extra_smu_allowed(self):
        judge = self._make_judge(
            [
                _oracle_event("o1", "AgentUserInterface", "send_message_to_user"),
            ],
            extra_smu=1,
        )
        agent_events = [
            _agent_event("a1", "AgentUserInterface", "send_message_to_user"),
            _agent_event("a2", "AgentUserInterface", "send_message_to_user"),
        ]
        result = judge._preliminary_checks(agent_events, judge.turn_to_oracle_events[0])
        assert result is True

    def test_too_many_smu(self):
        judge = self._make_judge(
            [
                _oracle_event("o1", "AgentUserInterface", "send_message_to_user"),
            ],
            extra_smu=0,
        )
        agent_events = [
            _agent_event("a1", "AgentUserInterface", "send_message_to_user"),
            _agent_event("a2", "AgentUserInterface", "send_message_to_user"),
        ]
        result = judge._preliminary_checks(agent_events, judge.turn_to_oracle_events[0])
        assert isinstance(result, str)
        assert "SMU" in result

    def test_tool_count_mismatch(self):
        judge = self._make_judge(
            [
                _oracle_event("o1", "ContactsApp", "add_new_contact"),
            ]
        )
        agent_events = [
            _agent_event("a1", "ContactsApp", "add_new_contact"),
            _agent_event("a2", "ContactsApp", "add_new_contact"),
        ]
        result = judge._preliminary_checks(agent_events, judge.turn_to_oracle_events[0])
        assert isinstance(result, str)
        assert "mismatch" in result.lower()


# ===========================================================================
# Single-turn judge tests
# ===========================================================================


class TestSingleTurnJudge:
    def test_simple_match(self):
        """Agent events with matching hard args should pass."""
        from gaia2_core.judge.judge import Judge

        oracle_events = [
            _oracle_event(
                "o1",
                "ContactsApp",
                "add_new_contact",
                {
                    "first_name": "Alice",
                    "last_name": "Smith",
                    "email": "alice@example.com",
                    "phone": "5551234567",
                },
            ),
            _oracle_event(
                "o2",
                "AgentUserInterface",
                "send_message_to_user",
                {
                    "content": "Done!",
                },
            ),
        ]
        oracle_graph = {"o1": [], "o2": []}

        judge = Judge(
            turn_to_oracle_events=[oracle_events],
            turn_to_oracle_graph=[oracle_graph],
            tasks=["Add Alice Smith as a contact"],
        )

        agent_events = [
            _agent_event(
                "a1",
                "ContactsApp",
                "add_new_contact",
                {
                    "first_name": "Alice",
                    "last_name": "Smith",
                    "email": "alice@example.com",
                    "phone": "(555) 123-4567",
                },
            ),
            _agent_event(
                "a2",
                "AgentUserInterface",
                "send_message_to_user",
                {
                    "content": "Done!",
                },
            ),
        ]

        result = judge.judge_turn(0, agent_events)
        assert result.success is True

    def test_hard_check_failure(self):
        """Mismatched hard args should reject."""
        from gaia2_core.judge.judge import Judge

        oracle_events = [
            _oracle_event(
                "o1",
                "ContactsApp",
                "delete_contact",
                {
                    "contact_id": "c123",
                },
            ),
        ]
        oracle_graph = {"o1": []}

        judge = Judge(
            turn_to_oracle_events=[oracle_events],
            turn_to_oracle_graph=[oracle_graph],
            tasks=["Delete contact"],
        )

        agent_events = [
            _agent_event(
                "a1",
                "ContactsApp",
                "delete_contact",
                {
                    "contact_id": "c999",  # wrong ID
                },
            ),
        ]

        result = judge.judge_turn(0, agent_events)
        assert result.success is False
        assert "contact_id" in result.failure_reason


# ===========================================================================
# Multi-turn judge tests
# ===========================================================================


class TestMultiTurnJudge:
    def test_id_remapping_across_turns(self):
        """Agent return values should be accessible via oracle IDs after remapping."""
        from gaia2_core.judge.judge import Judge

        # Turn 0: agent creates a contact, oracle expects event o1
        oracle_events_t0 = [
            _oracle_event(
                "o1",
                "ContactsApp",
                "add_new_contact",
                {
                    "first_name": "Bob",
                    "last_name": "Jones",
                    "email": "bob@test.com",
                    "phone": "1234567890",
                },
            ),
            _oracle_event(
                "o2",
                "AgentUserInterface",
                "send_message_to_user",
                {
                    "content": "Added",
                },
            ),
        ]
        # Turn 1: oracle event references o1's return value
        oracle_events_t1 = [
            _oracle_event(
                "o3",
                "ContactsApp",
                "edit_contact",
                {
                    "contact_id": "{{o1}}",
                    "updates": "new email",
                },
            ),
            _oracle_event(
                "o4",
                "AgentUserInterface",
                "send_message_to_user",
                {
                    "content": "Updated",
                },
            ),
        ]

        judge = Judge(
            turn_to_oracle_events=[oracle_events_t0, oracle_events_t1],
            turn_to_oracle_graph=[
                {"o1": [], "o2": []},
                {"o3": [], "o4": []},
            ],
            tasks=["Add Bob", "Update Bob"],
        )

        # Turn 0
        agent_events_t0 = [
            _agent_event(
                "a1",
                "ContactsApp",
                "add_new_contact",
                {
                    "first_name": "Bob",
                    "last_name": "Jones",
                    "email": "bob@test.com",
                    "phone": "1234567890",
                },
                return_value="contact_abc",
            ),
            _agent_event(
                "a2",
                "AgentUserInterface",
                "send_message_to_user",
                {
                    "content": "Added",
                },
            ),
        ]
        r0 = judge.judge_turn(0, agent_events_t0)
        assert r0.success is True

        # Turn 1 — oracle arg {{o1}} should resolve to "contact_abc"
        agent_events_t1 = [
            _agent_event(
                "a3",
                "ContactsApp",
                "edit_contact",
                {
                    "contact_id": "contact_abc",
                    "updates": "new email",
                },
            ),
            _agent_event(
                "a4",
                "AgentUserInterface",
                "send_message_to_user",
                {
                    "content": "Updated",
                },
            ),
        ]
        r1 = judge.judge_turn(1, agent_events_t1)
        assert r1.success is True


# ===========================================================================
# Placeholder resolution tests
# ===========================================================================


class TestPlaceholderResolution:
    def test_simple_placeholder(self):
        from gaia2_core.judge.judge import Judge

        judge = Judge(
            turn_to_oracle_events=[[]],
            turn_to_oracle_graph=[{}],
            tasks=[],
        )
        # Simulate a matched agent event
        agent_ev = _agent_event("a1", "X", "y", return_value="ret_value")
        judge._all_agent_events.append(agent_ev)
        judge._oracle_id_to_agent_idx["oracle_1"] = 0

        oracle_ev = _oracle_event("o2", "X", "z", {"ref": "{{oracle_1}}"})
        resolved = judge._resolve_oracle_placeholders(oracle_ev)
        assert resolved.resolved_args["ref"] == "ret_value"

    def test_nested_placeholder(self):
        from gaia2_core.judge.judge import Judge

        judge = Judge(
            turn_to_oracle_events=[[]],
            turn_to_oracle_graph=[{}],
            tasks=[],
        )
        agent_ev = _agent_event("a1", "X", "y", return_value={"id": "nested_val"})
        judge._all_agent_events.append(agent_ev)
        judge._oracle_id_to_agent_idx["oracle_1"] = 0

        oracle_ev = _oracle_event("o2", "X", "z", {"ref": "{{oracle_1.id}}"})
        resolved = judge._resolve_oracle_placeholders(oracle_ev)
        assert resolved.resolved_args["ref"] == "nested_val"

    def test_unresolved_placeholder(self):
        from gaia2_core.judge.judge import Judge

        judge = Judge(
            turn_to_oracle_events=[[]],
            turn_to_oracle_graph=[{}],
            tasks=[],
        )
        oracle_ev = _oracle_event("o1", "X", "z", {"ref": "{{unknown_id}}"})
        resolved = judge._resolve_oracle_placeholders(oracle_ev)
        assert resolved.resolved_args["ref"] == "{{unknown_id}}"


# ===========================================================================
# Causality tests
# ===========================================================================


class TestCausality:
    def test_causality_enforced(self):
        """Parent oracle event must be matched to an earlier agent event."""
        from gaia2_core.judge.judge import Judge

        # o2 depends on o1 — o1 must match before o2
        oracle_events = [
            _oracle_event(
                "o1",
                "ContactsApp",
                "add_new_contact",
                {
                    "first_name": "A",
                    "last_name": "B",
                    "email": "a@b.com",
                    "phone": "123",
                },
            ),
            _oracle_event(
                "o2",
                "ContactsApp",
                "add_new_contact",
                {
                    "first_name": "C",
                    "last_name": "D",
                    "email": "c@d.com",
                    "phone": "456",
                },
            ),
        ]
        oracle_graph = {"o1": [], "o2": ["o1"]}

        judge = Judge(
            turn_to_oracle_events=[oracle_events],
            turn_to_oracle_graph=[oracle_graph],
            tasks=["Add two contacts"],
        )

        # Agent events in wrong order — a1 matches o2, a2 matches o1
        # But o2 depends on o1, so a1 (idx 0) can't match o2 because
        # o1 hasn't been matched to a lower-index agent yet
        agent_events = [
            _agent_event(
                "a1",
                "ContactsApp",
                "add_new_contact",
                {
                    "first_name": "C",
                    "last_name": "D",
                    "email": "c@d.com",
                    "phone": "456",
                },
            ),
            _agent_event(
                "a2",
                "ContactsApp",
                "add_new_contact",
                {
                    "first_name": "A",
                    "last_name": "B",
                    "email": "a@b.com",
                    "phone": "123",
                },
            ),
        ]

        result = judge.judge_turn(0, agent_events)
        # o1 matches a2 (idx 1), then o2 tries to match a1 (idx 0) but
        # parent o1 is at idx 1 which is NOT < 0, so causality fails
        assert result.success is False
        assert "causality" in result.failure_reason.lower()


# ===========================================================================
# Graceful degradation tests
# ===========================================================================


class TestGracefulDegradation:
    def test_no_llm_hard_only(self):
        """Without LLM engine, soft checks should pass by default."""
        from gaia2_core.judge.judge import Judge

        # CalendarApp has llm_checker args (title, description, location)
        oracle_events = [
            _oracle_event(
                "o1",
                "CalendarApp",
                "add_calendar_event",
                {
                    "title": "Meeting",
                    "description": "Team sync",
                    "start_datetime": "2024-01-15 10:00:00",
                    "end_datetime": "2024-01-15 11:00:00",
                    "attendees": ["alice"],
                },
            ),
        ]
        oracle_graph = {"o1": []}

        judge = Judge(
            turn_to_oracle_events=[oracle_events],
            turn_to_oracle_graph=[oracle_graph],
            tasks=["Schedule meeting"],
            engine=None,  # no LLM
        )

        agent_events = [
            _agent_event(
                "a1",
                "CalendarApp",
                "add_calendar_event",
                {
                    "title": "Different Title",  # LLM would check, but no LLM
                    "description": "Different desc",
                    "start_datetime": "2024-01-15 10:00:00",
                    "end_datetime": "2024-01-15 11:00:00",
                    "attendees": ["alice"],
                },
            ),
        ]

        result = judge.judge_turn(0, agent_events)
        # Hard checks pass (datetime, attendees match), soft skipped
        assert result.success is True

    def test_empty_oracle_events_auto_pass(self):
        from gaia2_core.judge.judge import Judge

        judge = Judge(
            turn_to_oracle_events=[[]],
            turn_to_oracle_graph=[{}],
            tasks=["Task"],
        )
        result = judge.judge_turn(0, [])
        assert result.success is True

    def test_turn_beyond_oracle_auto_pass(self):
        from gaia2_core.judge.judge import Judge

        judge = Judge(
            turn_to_oracle_events=[[]],
            turn_to_oracle_graph=[{}],
            tasks=["Task"],
        )
        result = judge.judge_turn(5, [])
        assert result.success is True


# ===========================================================================
# Judge config tests
# ===========================================================================


class TestJudgeConfig:
    def test_checker_type_is_hard(self):
        from gaia2_core.judge.config import CheckerType

        assert CheckerType.eq_checker.is_hard()
        assert CheckerType.datetime_checker.is_hard()
        assert not CheckerType.llm_checker.is_hard()
        assert not CheckerType.contain_any_checker.is_hard()

    def test_checker_type_is_scripted(self):
        from gaia2_core.judge.config import CheckerType

        assert CheckerType.contain_any_checker.is_scripted()
        assert CheckerType.contain_all_checker.is_scripted()
        assert not CheckerType.eq_checker.is_scripted()

    def test_soft_checker_need_subtask(self):
        from gaia2_core.judge.config import SoftCheckerType

        assert SoftCheckerType.content_checker.need_subtask
        assert SoftCheckerType.user_message_checker.need_subtask
        assert not SoftCheckerType.placeholder_checker.need_subtask
        assert not SoftCheckerType.tone_checker.need_subtask

    def test_build_checker_registries(self):
        from gaia2_core.judge.config import build_checker_registries

        arg_reg, soft_reg = build_checker_registries()

        # Canonical entries should still be present.
        assert "CalendarApp__add_calendar_event" in arg_reg

        # Public dataset aliases should be expanded.
        assert "Calendar__add_calendar_event" in arg_reg
        assert "Emails__send_email" in arg_reg
        assert "Files__mkdir" in arg_reg
        assert "Calendar__add_calendar_event" in soft_reg

        # Legacy internal aliases should be gone from the exported registry.
        assert "CalendarSqliteApp__add_calendar_event" not in arg_reg
        assert "ContactsTreeApp__add_new_contact" not in arg_reg
        assert "VirtualFileSystem__mkdir" not in arg_reg


# ===========================================================================
# tool_name property tests
# ===========================================================================


class TestToolName:
    def test_completed_event_tool_name(self):
        ev = _agent_event("a1", "ContactsApp", "add_new_contact")
        assert ev.tool_name == "ContactsApp__add_new_contact"

    def test_oracle_event_tool_name(self):
        ev = _oracle_event("o1", "CalendarApp", "delete_calendar_event")
        assert ev.tool_name == "CalendarApp__delete_calendar_event"

    def test_no_action_tool_name(self):
        ev = CompletedEvent(
            event_id="x",
            event_type="AGENT",
            event_time=0.0,
            action=None,
        )
        assert ev.tool_name == "NoApp__NoFunction"


# ===========================================================================
# placeholder_checker tests
# ===========================================================================


class TestPlaceholderChecker:
    def test_clean_args_pass(self):
        from gaia2_core.judge.checkers import placeholder_checker

        assert placeholder_checker({"content": "Hello, how are you?"}) is True

    def test_placeholder_detected(self):
        from gaia2_core.judge.checkers import placeholder_checker

        assert placeholder_checker({"content": "Hello [Your Name]"}) is False
        assert placeholder_checker({"content": "Hi [User's Name]"}) is False
        assert placeholder_checker({"content": "Best,\nYour Name"}) is False
