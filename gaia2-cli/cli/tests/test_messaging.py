# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
"""Unit tests for the Messages and Chats CLIs."""

import base64
import json

from conftest import assert_event, parse_output, read_events, seed_state
from gaia2_cli.apps.chats import cli as chats_cli
from gaia2_cli.apps.messages import cli as messages_cli

# ---------------------------------------------------------------------------
# State templates
# ---------------------------------------------------------------------------

MESSAGES_STATE = {
    "current_user_id": "user1",
    "name_to_id": {
        "Alice Smith": "alice1",
        "Bob Jones": "bob1",
        "Carol White": "carol1",
    },
    "id_to_name": {
        "alice1": "Alice Smith",
        "bob1": "Bob Jones",
        "carol1": "Carol White",
        "user1": "Current User",
    },
    "conversations": {},
    "conversation_view_limit": 5,
    "messages_view_limit": 10,
}

CHATS_STATE = {
    "current_user_id": "+1234567890",
    "name_to_id": {"Alice Smith": "+1111111111"},
    "id_to_name": {"+1111111111": "Alice Smith", "+1234567890": "Current User"},
    "conversations": {},
    "conversation_view_limit": 5,
    "messages_view_limit": 10,
}


def _messages_state(**overrides):
    """Return a copy of the base Messages state with optional overrides."""
    state = json.loads(json.dumps(MESSAGES_STATE))
    state.update(overrides)
    return state


def _chats_state(**overrides):
    """Return a copy of the base Chats state with optional overrides."""
    state = json.loads(json.dumps(CHATS_STATE))
    state.update(overrides)
    return state


def _make_conversation(
    conv_id, title, participants, messages=None, last_updated=1000.0
):
    """Build a conversation dict for seeding state."""
    return {
        "conversation_id": conv_id,
        "title": title,
        "participant_ids": participants,
        "last_updated": last_updated,
        "messages": messages or [],
    }


def _make_message(sender_id, message_id, timestamp, content, **extra):
    """Build a message dict for seeding state."""
    msg = {
        "sender_id": sender_id,
        "message_id": message_id,
        "timestamp": timestamp,
        "content": content,
    }
    msg.update(extra)
    return msg


# ---------------------------------------------------------------------------
# Messages tests (messages_cli)
# ---------------------------------------------------------------------------


class TestMessagesCli:
    """Tests for the Messages CLI (messages binary, NAME mode)."""

    # -- User lookup --------------------------------------------------------

    def test_get_user_id_exact_match(self, cli_env):
        state_dir, runner = cli_env
        seed_state(state_dir, "Messages", _messages_state())

        result = runner.invoke(
            messages_cli, ["get-user-id", "--user-name", "Alice Smith"]
        )
        assert result.exit_code == 0
        assert parse_output(result) == "alice1"

    def test_lookup_user_id_fuzzy_match(self, cli_env):
        state_dir, runner = cli_env
        seed_state(state_dir, "Messages", _messages_state())

        result = runner.invoke(
            messages_cli, ["lookup-user-id", "--user-name", "Alice Smit"]
        )
        assert result.exit_code == 0
        data = parse_output(result)
        assert "Alice Smith" in data
        assert data["Alice Smith"] == "alice1"

    # -- Send message -------------------------------------------------------

    def test_send_message_creates_conversation(self, cli_env, fixed_uuid, fixed_time):
        state_dir, runner = cli_env
        seed_state(state_dir, "Messages", _messages_state())

        result = runner.invoke(
            messages_cli,
            ["send-message", "--user-id", "alice1", "--content", "Hello Alice"],
        )
        assert result.exit_code == 0
        conv_id = parse_output(result)
        # First uuid creates the message, second creates the conversation
        assert conv_id == fixed_uuid(1)

    def test_send_message_existing_conversation(self, cli_env, fixed_uuid, fixed_time):
        state_dir, runner = cli_env
        conv = _make_conversation(
            "existing_conv",
            "Alice Smith",
            ["user1", "alice1"],
            messages=[_make_message("alice1", "msg0", 1000.0, "Hi there")],
        )
        seed_state(
            state_dir,
            "Messages",
            _messages_state(conversations={"existing_conv": conv}),
        )

        result = runner.invoke(
            messages_cli,
            ["send-message", "--user-id", "alice1", "--content", "Reply!"],
        )
        assert result.exit_code == 0
        assert parse_output(result) == "existing_conv"

        updated = json.loads((state_dir / "messages.json").read_text())
        assert len(updated["conversations"]["existing_conv"]["messages"]) == 2

    def test_send_message_with_attachment(
        self,
        cli_env,
        fixed_uuid,
        fixed_time,
    ):
        state_dir, runner = cli_env
        seed_state(state_dir, "Messages", _messages_state())

        # Place file in cloud-drive sandbox
        fs_dir = state_dir / "filesystem" / "Pictures"
        fs_dir.mkdir(parents=True)
        (fs_dir / "photo.png").write_bytes(b"fake-png-data")

        result = runner.invoke(
            messages_cli,
            [
                "send-message",
                "--user-id",
                "alice1",
                "--content",
                "See attached",
                "--attachment-path",
                "/Pictures/photo.png",
            ],
        )
        assert result.exit_code == 0

        updated = json.loads((state_dir / "messages.json").read_text())
        conv = list(updated["conversations"].values())[0]
        msg = conv["messages"][0]
        assert msg["attachment_name"] == "photo.png"
        assert base64.b64decode(msg["attachment"]) == b"fake-png-data"

    def test_send_message_invalid_user(self, cli_env):
        state_dir, runner = cli_env
        seed_state(state_dir, "Messages", _messages_state())

        result = runner.invoke(
            messages_cli,
            ["send-message", "--user-id", "nonexistent_id", "--content", "Hello"],
        )
        assert result.exit_code == 1

    # -- Group conversations ------------------------------------------------

    def test_create_group_conversation(self, cli_env, fixed_uuid, fixed_time):
        state_dir, runner = cli_env
        seed_state(state_dir, "Messages", _messages_state())

        user_ids = json.dumps(["alice1", "bob1"])
        result = runner.invoke(
            messages_cli,
            ["create-group-conversation", "--user-ids", user_ids],
        )
        assert result.exit_code == 0
        conv_id = parse_output(result)
        assert conv_id == fixed_uuid(0)

        updated = json.loads((state_dir / "messages.json").read_text())
        conv = updated["conversations"][conv_id]
        assert set(conv["participant_ids"]) == {"user1", "alice1", "bob1"}

    def test_create_group_conversation_too_few(self, cli_env):
        state_dir, runner = cli_env
        seed_state(state_dir, "Messages", _messages_state())

        result = runner.invoke(
            messages_cli,
            ["create-group-conversation", "--user-ids", json.dumps(["alice1"])],
        )
        assert result.exit_code == 1

    def test_create_group_conversation_rejects_self_in_ids(self, cli_env):
        """Passing current_user_id as one of the user-ids should fail since
        it doesn't count as an 'other' participant."""
        state_dir, runner = cli_env
        seed_state(state_dir, "Messages", _messages_state())

        # user1 is the current user — 2 IDs but only 1 is "other"
        result = runner.invoke(
            messages_cli,
            [
                "create-group-conversation",
                "--user-ids",
                json.dumps(["alice1", "user1"]),
            ],
        )
        assert result.exit_code == 1
        assert "two other participants" in result.output

    def test_send_message_to_group(self, cli_env, fixed_uuid, fixed_time):
        state_dir, runner = cli_env
        conv = _make_conversation("grp1", "Team Chat", ["user1", "alice1", "bob1"])
        seed_state(state_dir, "Messages", _messages_state(conversations={"grp1": conv}))

        result = runner.invoke(
            messages_cli,
            [
                "send-message-to-group-conversation",
                "--conversation-id",
                "grp1",
                "--content",
                "Hello team",
            ],
        )
        assert result.exit_code == 0
        assert parse_output(result) == "grp1"

    def test_send_message_to_group_not_a_group(self, cli_env, fixed_time):
        state_dir, runner = cli_env
        conv = _make_conversation("dm1", "Alice Smith", ["user1", "alice1"])
        seed_state(state_dir, "Messages", _messages_state(conversations={"dm1": conv}))

        result = runner.invoke(
            messages_cli,
            [
                "send-message-to-group-conversation",
                "--conversation-id",
                "dm1",
                "--content",
                "hi",
            ],
        )
        assert result.exit_code == 1

    # -- Conversation listing and reading -----------------------------------

    def test_get_existing_conversation_ids(self, cli_env):
        state_dir, runner = cli_env
        convs = {
            "conv1": _make_conversation("conv1", "Alice Smith", ["user1", "alice1"]),
            "conv2": _make_conversation(
                "conv2", "Bob Jones", ["user1", "bob1"], last_updated=900.0
            ),
        }
        seed_state(state_dir, "Messages", _messages_state(conversations=convs))

        result = runner.invoke(
            messages_cli,
            ["get-existing-conversation-ids", "--user-ids", json.dumps(["alice1"])],
        )
        assert result.exit_code == 0
        assert parse_output(result) == ["conv1"]

    def test_list_recent_ordering_and_limit(self, cli_env):
        state_dir, runner = cli_env
        convs = {
            "conv_old": _make_conversation(
                "conv_old", "Old Chat", ["user1", "bob1"], last_updated=500.0
            ),
            "conv_new": _make_conversation(
                "conv_new", "New Chat", ["user1", "alice1"], last_updated=2000.0
            ),
            "conv_mid": _make_conversation(
                "conv_mid", "Mid Chat", ["user1", "carol1"], last_updated=1000.0
            ),
        }
        seed_state(state_dir, "Messages", _messages_state(conversations=convs))

        result = runner.invoke(
            messages_cli, ["list-recent-conversations", "--limit", "2"]
        )
        assert result.exit_code == 0
        data = parse_output(result)
        assert len(data) == 2
        assert data[0]["conversation_id"] == "conv_new"
        assert data[1]["conversation_id"] == "conv_mid"

    def test_list_recent_exceeds_view_limit(self, cli_env):
        state_dir, runner = cli_env
        convs = {
            "conv1": _make_conversation("conv1", "Chat", ["user1", "alice1"]),
        }
        seed_state(state_dir, "Messages", _messages_state(conversations=convs))

        result = runner.invoke(
            messages_cli, ["list-recent-conversations", "--limit", "10"]
        )
        assert result.exit_code == 1

    def test_read_conversation_pagination(self, cli_env):
        state_dir, runner = cli_env
        msgs = [
            _make_message("alice1", "m1", 1000.0, "First message"),
            _make_message("user1", "m2", 2000.0, "Second message"),
            _make_message("alice1", "m3", 3000.0, "Third message"),
        ]
        conv = _make_conversation(
            "conv1",
            "Alice Smith",
            ["user1", "alice1"],
            messages=msgs,
            last_updated=3000.0,
        )
        seed_state(
            state_dir, "Messages", _messages_state(conversations={"conv1": conv})
        )

        # offset=1, limit=1 — most recent first: m3, m2, m1 → skip m3, get m2
        result = runner.invoke(
            messages_cli,
            [
                "read-conversation",
                "--conversation-id",
                "conv1",
                "--offset",
                "1",
                "--limit",
                "1",
            ],
        )
        assert result.exit_code == 0
        data = parse_output(result)
        assert len(data["messages"]) == 1
        assert data["messages"][0]["message_id"] == "m2"
        assert data["metadata"]["conversation_length"] == 3
        assert data["metadata"]["message_range"] == [1, 2]

    def test_read_conversation_date_filter(self, cli_env):
        state_dir, runner = cli_env
        msgs = [
            _make_message("alice1", "m1", 1000.0, "First"),
            _make_message("user1", "m2", 2000.0, "Second"),
            _make_message("alice1", "m3", 3000.0, "Third"),
        ]
        conv = _make_conversation(
            "conv1",
            "Alice Smith",
            ["user1", "alice1"],
            messages=msgs,
            last_updated=3000.0,
        )
        seed_state(
            state_dir, "Messages", _messages_state(conversations={"conv1": conv})
        )

        result = runner.invoke(
            messages_cli,
            [
                "read-conversation",
                "--conversation-id",
                "conv1",
                "--min-date",
                "1970-01-01 00:25:00",
                "--max-date",
                "1970-01-01 00:40:00",
            ],
        )
        assert result.exit_code == 0
        data = parse_output(result)
        assert len(data["messages"]) == 1
        assert data["messages"][0]["message_id"] == "m2"

    # -- Search -------------------------------------------------------------

    def test_search_case_insensitive(self, cli_env):
        state_dir, runner = cli_env
        convs = {
            "conv1": _make_conversation(
                "conv1",
                "Chat One",
                ["user1", "alice1"],
                messages=[_make_message("alice1", "m1", 1000.0, "Hello WORLD")],
            ),
            "conv2": _make_conversation(
                "conv2",
                "Chat Two",
                ["user1", "bob1"],
                messages=[_make_message("bob1", "m2", 900.0, "No match here")],
                last_updated=900.0,
            ),
        }
        seed_state(state_dir, "Messages", _messages_state(conversations=convs))

        result = runner.invoke(messages_cli, ["search", "--query", "hello world"])
        assert result.exit_code == 0
        data = parse_output(result)
        assert "conv1" in data
        assert "conv2" not in data

    def test_regex_search(self, cli_env):
        state_dir, runner = cli_env
        convs = {
            "conv1": _make_conversation(
                "conv1",
                "Chat",
                ["user1", "alice1"],
                messages=[
                    _make_message("alice1", "m1", 1000.0, "Meeting at 3pm tomorrow")
                ],
            ),
            "conv2": _make_conversation(
                "conv2",
                "Other",
                ["user1", "bob1"],
                messages=[_make_message("bob1", "m2", 900.0, "Just chatting")],
                last_updated=900.0,
            ),
        }
        seed_state(state_dir, "Messages", _messages_state(conversations=convs))

        result = runner.invoke(
            messages_cli, ["regex-search", "--query", r"meeting.*\d+pm"]
        )
        assert result.exit_code == 0
        data = parse_output(result)
        assert "conv1" in data
        assert "conv2" not in data

    # -- Participant management ---------------------------------------------

    def test_add_participant_title_update(self, cli_env, fixed_time):
        state_dir, runner = cli_env
        conv = _make_conversation("conv1", "Alice Smith", ["user1", "alice1"])
        seed_state(
            state_dir, "Messages", _messages_state(conversations={"conv1": conv})
        )

        result = runner.invoke(
            messages_cli,
            [
                "add-participant-to-conversation",
                "--conversation-id",
                "conv1",
                "--user-id",
                "bob1",
            ],
        )
        assert result.exit_code == 0

        updated = json.loads((state_dir / "messages.json").read_text())
        conv = updated["conversations"]["conv1"]
        assert "bob1" in conv["participant_ids"]
        assert conv["title"] == "Alice Smith, Bob Jones"

    def test_remove_participant_title_update(self, cli_env, fixed_time):
        state_dir, runner = cli_env
        conv = _make_conversation(
            "conv1", "Alice Smith, Bob Jones", ["user1", "alice1", "bob1"]
        )
        seed_state(
            state_dir, "Messages", _messages_state(conversations={"conv1": conv})
        )

        result = runner.invoke(
            messages_cli,
            [
                "remove-participant-from-conversation",
                "--conversation-id",
                "conv1",
                "--user-id",
                "bob1",
            ],
        )
        assert result.exit_code == 0

        updated = json.loads((state_dir / "messages.json").read_text())
        conv = updated["conversations"]["conv1"]
        assert "bob1" not in conv["participant_ids"]
        assert conv["title"] == "Alice Smith"

    def test_change_conversation_title(self, cli_env, fixed_time):
        state_dir, runner = cli_env
        conv = _make_conversation("conv1", "Old Title", ["user1", "alice1"])
        seed_state(
            state_dir, "Messages", _messages_state(conversations={"conv1": conv})
        )

        result = runner.invoke(
            messages_cli,
            [
                "change-conversation-title",
                "--conversation-id",
                "conv1",
                "--title",
                "New Title",
            ],
        )
        assert result.exit_code == 0

        updated = json.loads((state_dir / "messages.json").read_text())
        assert updated["conversations"]["conv1"]["title"] == "New Title"

    # -- Attachments --------------------------------------------------------

    def test_download_attachment(self, cli_env):
        state_dir, runner = cli_env
        original_data = b"binary-file-content-here"
        encoded = base64.b64encode(original_data).decode("utf-8")
        conv = _make_conversation(
            "conv1",
            "Alice Smith",
            ["user1", "alice1"],
            messages=[
                _make_message(
                    "alice1",
                    "msg1",
                    1000.0,
                    "Here is the file",
                    attachment=encoded,
                    attachment_name="report.pdf",
                )
            ],
        )
        seed_state(
            state_dir, "Messages", _messages_state(conversations={"conv1": conv})
        )

        # Ensure the sandbox filesystem root exists
        (state_dir / "filesystem").mkdir(parents=True, exist_ok=True)

        result = runner.invoke(
            messages_cli,
            [
                "download-attachment",
                "--conversation-id",
                "conv1",
                "--message-id",
                "msg1",
                "--download-path",
                "/Downloads",
            ],
        )
        assert result.exit_code == 0
        output_path = parse_output(result)
        assert output_path.endswith("report.pdf")
        with open(output_path, "rb") as f:
            assert f.read() == original_data

    # -- Event format -------------------------------------------------------

    def test_event_format(self, cli_env, fixed_uuid, fixed_time):
        state_dir, runner = cli_env
        seed_state(state_dir, "Messages", _messages_state())

        result = runner.invoke(
            messages_cli,
            ["send-message", "--user-id", "alice1", "--content", "Test event"],
        )
        assert result.exit_code == 0

        events = read_events(state_dir)
        assert len(events) == 1
        ev = events[0]
        assert_event(ev, app="MessagingAppV2", fn="send_message", write=True)
        assert ev["args"]["user_id"] == "alice1"
        assert ev["args"]["content"] == "Test event"
        assert ev["args"]["attachment_path"] is None
        assert ev["t"] == 1522479600.0


# ---------------------------------------------------------------------------
# Chats tests (chats_cli)
# ---------------------------------------------------------------------------


class TestChatsCli:
    """Tests for the Chats CLI (chats binary, PHONE_NUMBER mode)."""

    def test_send_message(self, cli_env, fixed_uuid, fixed_time):
        state_dir, runner = cli_env
        seed_state(state_dir, "Chats", _chats_state())

        result = runner.invoke(
            chats_cli,
            ["send-message", "--user-id", "+1111111111", "--content", "Hey"],
        )
        assert result.exit_code == 0

        events = read_events(state_dir)
        assert len(events) == 1
        assert events[0]["app"] == "MessagingAppV2"

    def test_send_message_with_attachment(self, cli_env, fixed_uuid, fixed_time):
        state_dir, runner = cli_env
        seed_state(state_dir, "Chats", _chats_state())

        fs_dir = state_dir / "filesystem" / "Documents"
        fs_dir.mkdir(parents=True)
        (fs_dir / "file.pdf").write_bytes(b"chat-pdf-data")

        result = runner.invoke(
            chats_cli,
            [
                "send-message",
                "--user-id",
                "+1111111111",
                "--content",
                "See attached",
                "--attachment-path",
                "/Documents/file.pdf",
            ],
        )
        assert result.exit_code == 0

        updated = json.loads((state_dir / "chats.json").read_text())
        conv = list(updated["conversations"].values())[0]
        msg = conv["messages"][0]
        assert msg["attachment_name"] == "file.pdf"
        assert base64.b64decode(msg["attachment"]) == b"chat-pdf-data"

    def test_download_attachment(self, cli_env):
        state_dir, runner = cli_env
        original_data = b"chat-binary-content"
        encoded = base64.b64encode(original_data).decode("utf-8")
        conv = _make_conversation(
            "conv1",
            "Alice Smith",
            ["+1234567890", "+1111111111"],
            messages=[
                _make_message(
                    "+1111111111",
                    "msg1",
                    1000.0,
                    "Here is the file",
                    attachment=encoded,
                    attachment_name="doc.pdf",
                )
            ],
        )
        seed_state(state_dir, "Chats", _chats_state(conversations={"conv1": conv}))

        (state_dir / "filesystem").mkdir(parents=True, exist_ok=True)

        result = runner.invoke(
            chats_cli,
            [
                "download-attachment",
                "--conversation-id",
                "conv1",
                "--message-id",
                "msg1",
                "--download-path",
                "/Downloads",
            ],
        )
        assert result.exit_code == 0
        output_path = parse_output(result)
        assert output_path.endswith("doc.pdf")
        with open(output_path, "rb") as f:
            assert f.read() == original_data

    def test_phone_number_mode_rejects_non_digit(self, cli_env):
        state_dir, runner = cli_env
        seed_state(state_dir, "Chats", _chats_state())

        result = runner.invoke(
            chats_cli,
            ["send-message", "--user-id", "not_a_phone", "--content", "Hello"],
        )
        assert result.exit_code == 1

    def test_event_format(self, cli_env, fixed_uuid, fixed_time):
        state_dir, runner = cli_env
        seed_state(state_dir, "Chats", _chats_state())

        result = runner.invoke(
            chats_cli,
            ["send-message", "--user-id", "+1111111111", "--content", "Chat msg"],
        )
        assert result.exit_code == 0

        events = read_events(state_dir)
        assert len(events) == 1
        ev = events[0]
        assert_event(ev, app="MessagingAppV2", fn="send_message", write=True)
        assert ev["args"]["user_id"] == "+1111111111"
        assert ev["args"]["content"] == "Chat msg"
