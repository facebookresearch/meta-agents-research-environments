# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
"""Unit tests for gaia2_cli/eventd.py and gaia2_cli/channel.py.

These tests exercise the pure-logic pieces of the gaia2-eventd daemon without
requiring the Gaia2 conda environment (no Gaia2 framework imports).
"""

import json
import time
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from gaia2_cli.daemon.channel import FileChannelAdapter
from gaia2_cli.daemon.eventd import (
    _APP_TO_CLI,
    Gaia2EventDaemon,
    _action_to_event_dict,
    _is_send_message_to_user,
)

# ---------------------------------------------------------------------------
# _action_to_event_dict
# ---------------------------------------------------------------------------


class TestActionToEventDict:
    """Tests for the CLI → CompletedEvent dict bridge function."""

    def test_basic_conversion(self):
        """Verify required fields are populated from a CLI action entry."""
        action = {
            "t": 1700000000.0,
            "app": "Contacts",
            "fn": "add_new_contact",
            "args": {"first_name": "Alice", "last_name": "Smith"},
            "w": True,
            "ret": "contact-123",
        }
        result = _action_to_event_dict(action)

        assert result["class_name"] == "CompletedEvent"
        assert result["event_type"] == "AGENT"
        assert result["event_time"] == 1700000000.0
        assert result["event_id"].startswith("AGENT-Contacts.add_new_contact-")

        # Action sub-dict
        act = result["action"]
        assert act["class_name"] == "Contacts"
        assert act["app_name"] == "Contacts"
        assert act["function_name"] == "add_new_contact"
        assert act["args"] == {"first_name": "Alice", "last_name": "Smith"}
        assert act["operation_type"] == "write"

        # Metadata
        assert result["metadata"]["return_value"] == "contact-123"
        assert result["metadata"]["completed"] is True
        assert result["metadata"]["exception"] is None

    def test_read_operation_type(self):
        """Non-write actions get operation_type='read'."""
        action = {
            "t": 1.0,
            "app": "Calendar",
            "fn": "get_events",
            "args": {},
            "w": False,
            "ret": [],
        }
        result = _action_to_event_dict(action)
        assert result["action"]["operation_type"] == "read"

    def test_missing_w_defaults_to_read(self):
        """If 'w' key is missing, operation_type defaults to 'read'."""
        action = {"t": 1.0, "app": "Calendar", "fn": "get_events", "args": {}}
        result = _action_to_event_dict(action)
        assert result["action"]["operation_type"] == "read"

    def test_successors_and_dependencies_empty(self):
        """Output always has empty successors and dependencies."""
        action = {"t": 1.0, "app": "X", "fn": "y", "args": {}, "w": False}
        result = _action_to_event_dict(action)
        assert result["successors"] == []
        assert result["dependencies"] == []


# ---------------------------------------------------------------------------
# _is_send_message_to_user
# ---------------------------------------------------------------------------


class TestIsSendMessageToUser:
    """Tests for turn boundary detection."""

    def test_true_for_send_message(self):
        entry = {"app": "AgentUserInterface", "fn": "send_message_to_user"}
        assert _is_send_message_to_user(entry) is True

    def test_false_for_other_app(self):
        entry = {"app": "Contacts", "fn": "send_message_to_user"}
        assert _is_send_message_to_user(entry) is False

    def test_false_for_other_fn(self):
        entry = {"app": "AgentUserInterface", "fn": "get_last_message"}
        assert _is_send_message_to_user(entry) is False

    def test_false_for_empty_dict(self):
        assert _is_send_message_to_user({}) is False

    def test_false_for_missing_keys(self):
        assert _is_send_message_to_user({"app": "AgentUserInterface"}) is False
        assert _is_send_message_to_user({"fn": "send_message_to_user"}) is False


# ---------------------------------------------------------------------------
# _APP_TO_CLI mapping coverage
# ---------------------------------------------------------------------------


class TestAppToCliMapping:
    """Verify the app class → CLI entry point mapping is complete."""

    def test_all_known_apps_mapped(self):
        """All gaia2-cli apps from pyproject.toml should have a mapping."""
        expected_clis = {
            "calendar",
            "contacts",
            "emails",
            "messages",
            "chats",
            "rent-a-flat",
            "city",
            "cabs",
            "shopping",
            "cloud-drive",
        }
        actual_clis = set(_APP_TO_CLI.values())
        assert expected_clis == actual_clis

    def test_mapping_values_are_strings(self):
        for k, v in _APP_TO_CLI.items():
            assert isinstance(k, str)
            assert isinstance(v, str)


# ---------------------------------------------------------------------------
# Gaia2EventDaemon._build_cli_cmd (tested via a partial instance)
# ---------------------------------------------------------------------------


@pytest.fixture
def daemon(tmp_path):
    """Create a daemon instance for testing pure methods (no setup() call)."""
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    (state_dir / "events.jsonl").touch()

    # Create a dummy scenario file so Path validation doesn't complain
    scenario = tmp_path / "scenario.json"
    scenario.write_text("{}")

    d = Gaia2EventDaemon(
        scenario_path=str(scenario),
        state_dir=str(state_dir),
    )

    return d


# ---------------------------------------------------------------------------
# Gaia2EventDaemon._read_new_events
# ---------------------------------------------------------------------------


class TestReadNewEvents:
    """Tests for the incremental events.jsonl reader."""

    def test_reads_new_lines(self, daemon):
        events_path = daemon.events_jsonl
        # Write two events
        with open(events_path, "a") as f:
            f.write(json.dumps({"t": 1.0, "app": "X", "fn": "a", "args": {}}) + "\n")
            f.write(json.dumps({"t": 2.0, "app": "Y", "fn": "b", "args": {}}) + "\n")

        entries = daemon._read_new_events()
        assert len(entries) == 2
        assert entries[0]["app"] == "X"
        assert entries[1]["app"] == "Y"

    def test_incremental_reads(self, daemon):
        events_path = daemon.events_jsonl

        # First batch
        with open(events_path, "a") as f:
            f.write(json.dumps({"t": 1.0, "app": "A", "fn": "x", "args": {}}) + "\n")
        assert len(daemon._read_new_events()) == 1

        # Second batch — should only return new lines
        with open(events_path, "a") as f:
            f.write(json.dumps({"t": 2.0, "app": "B", "fn": "y", "args": {}}) + "\n")
            f.write(json.dumps({"t": 3.0, "app": "C", "fn": "z", "args": {}}) + "\n")

        entries = daemon._read_new_events()
        assert len(entries) == 2
        assert entries[0]["app"] == "B"

    def test_empty_file_returns_empty(self, daemon):
        assert daemon._read_new_events() == []

    def test_malformed_lines_skipped(self, daemon):
        events_path = daemon.events_jsonl
        with open(events_path, "a") as f:
            f.write("not json\n")
            f.write(json.dumps({"t": 1.0, "app": "A", "fn": "x", "args": {}}) + "\n")

        entries = daemon._read_new_events()
        assert len(entries) == 1
        assert entries[0]["app"] == "A"

    def test_missing_file_returns_empty(self, tmp_path):
        scenario = tmp_path / "scenario.json"
        scenario.write_text("{}")
        d = Gaia2EventDaemon(
            scenario_path=str(scenario),
            state_dir=str(tmp_path),
            events_jsonl=str(tmp_path / "nonexistent.jsonl"),
        )
        assert d._read_new_events() == []


# ---------------------------------------------------------------------------
# Gaia2EventDaemon._add_events_to_processor
# ---------------------------------------------------------------------------


class TestAddEventsToProcessor:
    def test_env_entry_preserves_event_id_and_sim_t(self, daemon):
        class RecordingProcessor:
            def __init__(self):
                self.stopped = False
                self.events = []

            def add_agent_event(self, completed):
                self.events.append(completed)

        daemon._processor = RecordingProcessor()

        daemon._add_events_to_processor(
            [
                {
                    "t": 1775590291.2427995,
                    "app": "Shopping",
                    "fn": "add_item_to_product",
                    "args": {"product_id": "abc"},
                    "w": True,
                    "ret": "item-1",
                    "event_id": "Event-ENV-3427b6a3-05d9-4a81-ba5a-9843e6a3995e",
                    "sim_t": "2024-10-15 07:01:00",
                }
            ]
        )

        assert len(daemon._processor.events) == 1
        completed = daemon._processor.events[0]
        assert completed.event_id == "Event-ENV-3427b6a3-05d9-4a81-ba5a-9843e6a3995e"
        assert completed.event_type == "ENV"
        assert completed.event_time == pytest.approx(1728975660.0)
        assert completed.action is not None
        assert completed.action.operation_type == "write"


# ---------------------------------------------------------------------------
# Gaia2EventDaemon._run_file
# ---------------------------------------------------------------------------


class TestRunFilePolling:
    def test_env_only_poll_batch_still_reaches_processor(self, daemon, monkeypatch):
        env_entry = {
            "t": 1775590291.2427995,
            "app": "Shopping",
            "fn": "add_item_to_product",
            "args": {"product_id": "abc"},
            "w": True,
            "ret": "item-1",
            "event_id": "Event-ENV-3427b6a3-05d9-4a81-ba5a-9843e6a3995e",
            "sim_t": "2024-10-15 07:01:00",
        }
        batches = iter([[env_entry], []])
        added_batches = []

        daemon._task = None
        daemon._running = True
        daemon._processor = SimpleNamespace(
            stopped=False,
            _queue=[],
            nb_turns=1,
            all_conditions_fired=False,
        )

        def fake_read_new_events():
            batch = next(batches)
            if not batch:
                daemon._running = False
            return batch

        monkeypatch.setattr(daemon, "_advance_time", lambda: False)
        monkeypatch.setattr(daemon, "_read_new_events", fake_read_new_events)
        monkeypatch.setattr(daemon, "_write_status", lambda *_args, **_kwargs: None)
        monkeypatch.setattr(
            daemon,
            "_add_events_to_processor",
            lambda entries: added_batches.append(entries),
        )
        monkeypatch.setattr(daemon, "_check_scenario_complete", lambda: False)
        monkeypatch.setattr("gaia2_cli.daemon.eventd.time.sleep", lambda *_args: None)

        daemon._run_file()

        assert added_batches == [[env_entry]]


class TestCreateJudge:
    def test_passes_judge_api_key_to_engine(self, daemon, monkeypatch):
        import gaia2_cli.judge as judge_module

        daemon.judge_model = "judge-model"
        daemon.judge_provider = "openai"
        daemon.judge_base_url = "https://example.invalid/v1"
        daemon.judge_api_key = "judge-key"

        captured: dict[str, dict] = {}

        def fake_engine(**kwargs):
            captured["engine"] = kwargs
            return "engine"

        class FakeJudge:
            def __init__(self, **kwargs):
                captured["judge"] = kwargs

        monkeypatch.setattr(judge_module, "create_litellm_engine", fake_engine)
        monkeypatch.setattr(judge_module, "Judge", FakeJudge)

        loader = SimpleNamespace(
            extract_oracle_data=lambda **_: ([[{"event": 1}]], [{}], ["task"], None),
            start_time=0.0,
            app_name_to_class={},
        )
        processor = SimpleNamespace(event_id_to_turn_idx={}, nb_turns=1)

        judge = daemon._create_judge(loader, processor)

        assert isinstance(judge, FakeJudge)
        assert captured["engine"]["api_key"] == "judge-key"
        assert captured["engine"]["provider"] == "openai"


# ---------------------------------------------------------------------------
# Gaia2EventDaemon._format_notification
# ---------------------------------------------------------------------------


class TestFormatNotification:
    """Tests for notification formatting logic."""

    def test_notifiable_fn_returns_string(self, daemon):
        result = daemon._format_notification(
            "EmailClientV2",
            "send_email_to_user_only",
            {"sender": "alice@example.com"},
        )
        assert result is not None
        assert "alice@example.com" in result

    def test_non_notifiable_fn_returns_none(self, daemon):
        result = daemon._format_notification("Contacts", "add_new_contact", {})
        assert result is None

    @pytest.mark.parametrize(
        "app_name,fn_name",
        [
            ("EmailClientApp", "send_email_to_user_only"),
            ("EmailClientApp", "reply_to_email_from_user"),
            ("MessagingAppV2", "create_and_add_message"),
            ("CalendarApp", "add_calendar_event_by_attendee"),
            ("ShoppingApp", "cancel_order"),
            ("CabApp", "cancel_ride"),
        ],
    )
    def test_all_notifiable_functions(self, daemon, app_name, fn_name):
        result = daemon._format_notification(app_name, fn_name, {"arg": "val"})
        assert result is not None
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# FileChannelAdapter
# ---------------------------------------------------------------------------


class TestFileChannelAdapter:
    """Tests for the file-based channel adapter."""

    @pytest.fixture
    def channel(self, tmp_path):
        """Create a FileChannelAdapter with temp paths."""
        return FileChannelAdapter(
            notifications_path=tmp_path / "notifications.jsonl",
            responses_path=tmp_path / "agent_responses.jsonl",
            events_jsonl_path=tmp_path / "events.jsonl",
        )

    def test_send_response_writes_to_responses(self, channel):
        channel.send_response("Hello, user!")

        lines = channel.responses_path.read_text().strip().splitlines()
        assert len(lines) == 1
        data = json.loads(lines[0])
        assert data["content"] == "Hello, user!"
        assert "timestamp" in data

    def test_send_response_writes_event(self, channel):
        channel.send_response("Hello!")

        lines = channel.events_jsonl_path.read_text().strip().splitlines()
        assert len(lines) == 1
        event = json.loads(lines[0])
        assert event["app"] == "AgentUserInterface"
        assert event["fn"] == "send_message_to_user"
        assert event["args"]["content"] == "Hello!"
        assert event["w"] is True

    def test_send_response_includes_sim_t(self, channel, tmp_path):
        """send_message_to_user should include sim_t from faketime.rc."""
        faketime_path = tmp_path / "faketime.rc"
        faketime_path.write_text("2024-10-15 07:05:30\n")

        with patch("gaia2_cli.daemon.channel.open", create=False) as mock_open:  # noqa: F841
            # We need a real open for events.jsonl and responses, but fake
            # one for /tmp/faketime.rc. Easier: just write a real faketime.rc
            # at the expected path.
            pass

        # Use monkeypatch to make the channel read our faketime.rc
        import builtins

        real_open = builtins.open

        def patched_open(path, *args, **kwargs):
            if str(path) == "/tmp/faketime.rc":
                return real_open(str(faketime_path), *args, **kwargs)
            return real_open(path, *args, **kwargs)

        with patch("builtins.open", side_effect=patched_open):
            channel.send_response("Done!")

        lines = channel.events_jsonl_path.read_text().strip().splitlines()
        event = json.loads(lines[0])
        assert event["sim_t"] == "2024-10-15 07:05:30"

    def test_send_response_no_faketime_no_sim_t(self, channel):
        """Without faketime.rc, event should not have sim_t."""
        channel.send_response("Done!")

        lines = channel.events_jsonl_path.read_text().strip().splitlines()
        event = json.loads(lines[0])
        assert "sim_t" not in event

    def test_read_notifications_empty(self, channel):
        assert channel.read_notifications() == []

    def test_read_notifications_incremental(self, channel):
        # Write two notifications
        notif_path = channel.notifications_path
        with open(notif_path, "a") as f:
            f.write(json.dumps({"type": "user_message", "content": "Hi"}) + "\n")
            f.write(json.dumps({"type": "env_action", "app": "X"}) + "\n")

        msgs = channel.read_notifications()
        assert len(msgs) == 2
        assert msgs[0]["type"] == "user_message"
        assert msgs[1]["type"] == "env_action"

        # Second read should return empty (no new data)
        assert channel.read_notifications() == []

        # Append more
        with open(notif_path, "a") as f:
            f.write(json.dumps({"type": "user_message", "content": "More"}) + "\n")
        msgs = channel.read_notifications()
        assert len(msgs) == 1
        assert msgs[0]["content"] == "More"

    def test_send_response_is_turn_boundary(self, channel):
        """send_response writes a send_message_to_user event (turn boundary)."""
        channel.send_response("Done.")

        event_line = channel.events_jsonl_path.read_text().strip()
        event = json.loads(event_line)
        assert _is_send_message_to_user(event)

    def test_multiple_responses(self, channel):
        channel.send_response("First")
        channel.send_response("Second")

        resp_lines = channel.responses_path.read_text().strip().splitlines()
        assert len(resp_lines) == 2
        assert json.loads(resp_lines[0])["content"] == "First"
        assert json.loads(resp_lines[1])["content"] == "Second"

        event_lines = channel.events_jsonl_path.read_text().strip().splitlines()
        assert len(event_lines) == 2

    def test_wait_for_notification_returns_immediately(self, channel):
        """If notifications are already available, returns immediately."""
        with open(channel.notifications_path, "a") as f:
            f.write(json.dumps({"type": "user_message", "content": "Yo"}) + "\n")

        start = time.time()
        msgs = channel.wait_for_notification(poll_interval=0.1, timeout=5.0)
        elapsed = time.time() - start

        assert len(msgs) == 1
        assert elapsed < 1.0  # Should be near-instant

    def test_wait_for_notification_timeout(self, channel):
        """Returns empty list on timeout when no notifications arrive."""
        start = time.time()
        msgs = channel.wait_for_notification(poll_interval=0.05, timeout=0.2)
        elapsed = time.time() - start

        assert msgs == []
        assert elapsed >= 0.15  # Should have waited ~0.2s


# ---------------------------------------------------------------------------
# Gaia2EventDaemon._write_notification
# ---------------------------------------------------------------------------


class TestWriteNotification:
    """Tests for the notification writer."""

    def test_appends_json_line(self, daemon):
        daemon._write_notification({"type": "user_message", "content": "Hi"})
        daemon._write_notification({"type": "env_action", "app": "X"})

        lines = daemon.notifications_path.read_text().strip().splitlines()
        assert len(lines) == 2
        assert json.loads(lines[0])["type"] == "user_message"
        assert json.loads(lines[1])["type"] == "env_action"


# ---------------------------------------------------------------------------
# Faketime advancement
# ---------------------------------------------------------------------------


class TestFaketimeAdvancement:
    """Tests for _advance_time() faketime.rc updates."""

    def _make_daemon_with_processor(self, tmp_path, start_time=1728975600.0):
        """Create a daemon with a minimal processor (no scenario import)."""
        from gaia2_core.event_loop.processor import EventProcessor

        state_dir = tmp_path / "state"
        state_dir.mkdir()
        (state_dir / "events.jsonl").touch()

        faketime_path = tmp_path / "faketime.rc"
        scenario = tmp_path / "scenario.json"
        scenario.write_text("{}")

        d = Gaia2EventDaemon(
            scenario_path=str(scenario),
            state_dir=str(state_dir),
            faketime_path=str(faketime_path),
            poll_interval=0.5,
        )

        # Create a processor with no events (empty queue)
        d._processor = EventProcessor(
            events=[],
            start_time=start_time,
            time_increment=1.0,
            app_name_to_class={},
        )
        d._sim_start = start_time

        return d, faketime_path

    def test_faketime_advances_without_env_events(self, tmp_path):
        """Faketime.rc should update even when no ENV events fire."""
        d, faketime_path = self._make_daemon_with_processor(tmp_path)

        # Queue is empty — previously _advance_time returned False immediately
        d._advance_time()

        assert faketime_path.exists()
        content = faketime_path.read_text().strip()
        assert content == "2024-10-15 07:00:00"

    def test_faketime_advances_continuously(self, tmp_path):
        """Each _advance_time() call should advance the clock."""
        d, faketime_path = self._make_daemon_with_processor(tmp_path)

        d._advance_time()
        t1 = faketime_path.read_text().strip()

        d._advance_time()
        t2 = faketime_path.read_text().strip()

        d._advance_time()
        t3 = faketime_path.read_text().strip()

        # poll_interval=0.5, time_increment=1.0 → 0.5s advance per tick
        assert t1 == "2024-10-15 07:00:00"
        assert t2 == "2024-10-15 07:00:01"  # +0.5s rounds
        assert t3 == "2024-10-15 07:00:01"  # +1.0s total

    def test_faketime_matches_processor_time(self, tmp_path):
        """Faketime.rc content should match processor._current_time."""
        from datetime import datetime, timezone

        d, faketime_path = self._make_daemon_with_processor(tmp_path)

        # Advance 10 times (5s of simulated time)
        for _ in range(10):
            d._advance_time()

        content = faketime_path.read_text().strip()
        expected_time = d._processor._current_time
        expected_dt = datetime.fromtimestamp(expected_time, tz=timezone.utc)
        expected_str = expected_dt.strftime("%Y-%m-%d %H:%M:%S")

        assert content == expected_str

    def test_no_faketime_path_no_error(self, tmp_path):
        """When faketime_path is None, _advance_time should not crash."""
        state_dir = tmp_path / "state"
        state_dir.mkdir()
        (state_dir / "events.jsonl").touch()
        scenario = tmp_path / "scenario.json"
        scenario.write_text("{}")

        d = Gaia2EventDaemon(
            scenario_path=str(scenario),
            state_dir=str(state_dir),
            faketime_path=None,
        )

        from gaia2_core.event_loop.processor import EventProcessor

        d._processor = EventProcessor(
            events=[],
            start_time=1728975600.0,
            time_increment=1.0,
            app_name_to_class={},
        )
        d._sim_start = 1728975600.0

        # Should not raise
        d._advance_time()
