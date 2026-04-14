# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
"""Unit tests for gaia2_cli.email_app (EmailClientV2 CLI)."""

import base64
import json

import pytest
from conftest import assert_event, parse_output, read_events, seed_state
from gaia2_cli.apps.email import DEFAULT_USER_EMAIL, cli

APP = "EmailClientV2"


# ---------------------------------------------------------------------------
# Autouse fixture: ensure the shared module-level _APP_NAME is always set
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def set_app():
    """Set the shared app name before every test."""
    from gaia2_cli.shared import set_app as _set_app

    _set_app(APP)


# ---------------------------------------------------------------------------
# State helpers
# ---------------------------------------------------------------------------


def _empty_state(user_email: str = "user@test.com") -> dict:
    """Return a minimal empty email state."""
    return {
        "user_email": user_email,
        "view_limit": 5,
        "folders": {
            "INBOX": {"folder_name": "INBOX", "emails": []},
            "SENT": {"folder_name": "SENT", "emails": []},
            "DRAFT": {"folder_name": "DRAFT", "emails": []},
            "TRASH": {"folder_name": "TRASH", "emails": []},
        },
    }


def _make_email(
    email_id: str = "e1",
    sender: str = "alice@test.com",
    recipients: list[str] | None = None,
    subject: str = "Hello",
    content: str = "Hi there",
    parent_id: str | None = None,
    cc: list[str] | None = None,
    attachments: dict[str, str] | None = None,
    timestamp: float = 1000.0,
    is_read: bool = False,
) -> dict:
    return {
        "email_id": email_id,
        "sender": sender,
        "recipients": recipients or ["user@test.com"],
        "subject": subject,
        "content": content,
        "parent_id": parent_id,
        "cc": cc or [],
        "attachments": attachments or {},
        "timestamp": timestamp,
        "is_read": is_read,
    }


# ===================================================================
# 1. list-emails: empty INBOX -> total_emails=0
# ===================================================================


def test_list_emails_empty_inbox(cli_env):
    state_dir, runner = cli_env
    seed_state(state_dir, APP, _empty_state())

    result = runner.invoke(cli, ["list-emails"])
    assert result.exit_code == 0, result.stderr
    data = parse_output(result)
    assert data["total_emails"] == 0
    assert data["total_returned_emails"] == 0
    assert data["emails"] == []


# ===================================================================
# 2. list-emails: pagination and sorting by timestamp desc
# ===================================================================


def test_list_emails_pagination_and_sorting(cli_env):
    state_dir, runner = cli_env
    state = _empty_state()
    # Insert 3 emails with ascending timestamps; list should return newest first
    state["folders"]["INBOX"]["emails"] = [
        _make_email(email_id="e1", subject="First", timestamp=100.0),
        _make_email(email_id="e2", subject="Second", timestamp=200.0),
        _make_email(email_id="e3", subject="Third", timestamp=300.0),
    ]
    seed_state(state_dir, APP, state)

    # Request offset=0, limit=2 -> should get e3 (300) and e2 (200)
    result = runner.invoke(cli, ["list-emails", "--offset", "0", "--limit", "2"])
    assert result.exit_code == 0, result.stderr
    data = parse_output(result)
    assert data["total_emails"] == 3
    assert data["total_returned_emails"] == 2
    assert data["emails_range"] == [0, 2]
    subjects = [e["subject"] for e in data["emails"]]
    assert subjects == ["Third", "Second"]

    # Request offset=2, limit=2 -> should get e1 (100)
    result = runner.invoke(cli, ["list-emails", "--offset", "2", "--limit", "2"])
    assert result.exit_code == 0, result.stderr
    data = parse_output(result)
    assert data["total_returned_emails"] == 1
    assert data["emails"][0]["subject"] == "First"


# ===================================================================
# 3. list-emails: offset out of range -> exit 1
# ===================================================================


def test_list_emails_offset_out_of_range(cli_env):
    state_dir, runner = cli_env
    state = _empty_state()
    state["folders"]["INBOX"]["emails"] = [
        _make_email(email_id="e1", timestamp=100.0),
    ]
    seed_state(state_dir, APP, state)

    result = runner.invoke(cli, ["list-emails", "--offset", "5"])
    assert result.exit_code == 1


# ===================================================================
# 4. get-email-by-id: marks is_read=True in state
# ===================================================================


def test_get_email_by_id_marks_read(cli_env):
    state_dir, runner = cli_env
    state = _empty_state()
    state["folders"]["INBOX"]["emails"] = [
        _make_email(email_id="e1", is_read=False),
    ]
    seed_state(state_dir, APP, state)

    result = runner.invoke(cli, ["get-email-by-id", "--email-id", "e1"])
    assert result.exit_code == 0, result.stderr
    data = parse_output(result)
    assert data["email_id"] == "e1"
    assert data["is_read"] is True

    # Verify persisted state has is_read=True
    persisted = json.loads((state_dir / "email_client_v2.json").read_text())
    inbox_emails = persisted["folders"]["INBOX"]["emails"]
    assert inbox_emails[0]["is_read"] is True


# ===================================================================
# 5. get-email-by-id: not found -> exit 1
# ===================================================================


def test_get_email_by_id_not_found(cli_env):
    state_dir, runner = cli_env
    seed_state(state_dir, APP, _empty_state())

    result = runner.invoke(cli, ["get-email-by-id", "--email-id", "nonexistent"])
    assert result.exit_code == 1


# ===================================================================
# 6. get-email-by-index: happy path (0-based, newest first)
# ===================================================================


def test_get_email_by_index_happy(cli_env):
    state_dir, runner = cli_env
    state = _empty_state()
    state["folders"]["INBOX"]["emails"] = [
        _make_email(email_id="e_old", subject="Old", timestamp=100.0),
        _make_email(email_id="e_new", subject="New", timestamp=200.0),
    ]
    seed_state(state_dir, APP, state)

    # Index 0 should be the newest email (timestamp 200)
    result = runner.invoke(cli, ["get-email-by-index", "--idx", "0"])
    assert result.exit_code == 0, result.stderr
    data = parse_output(result)
    assert data["email_id"] == "e_new"
    assert data["subject"] == "New"
    assert data["is_read"] is True


# ===================================================================
# 7. send-email: happy path, check email in SENT folder state
# ===================================================================


def test_send_email_happy(cli_env, fixed_uuid, fixed_time):
    state_dir, runner = cli_env
    seed_state(state_dir, APP, _empty_state())

    result = runner.invoke(
        cli,
        [
            "send-email",
            "--recipients",
            '["bob@test.com"]',
            "--subject",
            "Test Subject",
            "--content",
            "Test Body",
            "--cc",
            '["carol@test.com"]',
        ],
    )
    assert result.exit_code == 0, result.stderr
    data = parse_output(result)
    assert data["status"] == "ok"
    expected_id = fixed_uuid(0)
    assert data["email_id"] == expected_id

    # Verify the email is in SENT folder on disk
    persisted = json.loads((state_dir / "email_client_v2.json").read_text())
    sent_emails = persisted["folders"]["SENT"]["emails"]
    assert len(sent_emails) == 1
    sent = sent_emails[0]
    assert sent["sender"] == "user@test.com"
    assert sent["recipients"] == ["bob@test.com"]
    assert sent["subject"] == "Test Subject"
    assert sent["content"] == "Test Body"
    assert sent["cc"] == ["carol@test.com"]


def test_send_email_uses_neutral_default_user_email(cli_env, fixed_uuid, fixed_time):
    state_dir, runner = cli_env

    result = runner.invoke(
        cli,
        [
            "send-email",
            "--recipients",
            '["bob@test.com"]',
            "--subject",
            "Default Sender",
            "--content",
            "Body",
        ],
    )
    assert result.exit_code == 0, result.stderr

    persisted = json.loads((state_dir / "email_client_v2.json").read_text())
    sent = persisted["folders"]["SENT"]["emails"][0]
    assert sent["sender"] == DEFAULT_USER_EMAIL


# ===================================================================
# 8. send-email: with attachment (create a real temp file, pass path)
# ===================================================================


def test_send_email_with_attachment(cli_env, fixed_uuid, fixed_time):
    """Attachment paths are cloud-drive paths resolved via the FS sandbox."""
    state_dir, runner = cli_env
    seed_state(state_dir, APP, _empty_state())

    # Place file in the cloud-drive sandbox
    fs_root = state_dir / "filesystem" / "Documents"
    fs_root.mkdir(parents=True)
    attach_file = fs_root / "report.txt"
    attach_file.write_bytes(b"attachment content here")

    result = runner.invoke(
        cli,
        [
            "send-email",
            "--recipients",
            '["bob@test.com"]',
            "--subject",
            "With Attachment",
            "--content",
            "See attached",
            "--attachment-paths",
            '["/Documents/report.txt"]',
        ],
    )
    assert result.exit_code == 0, result.output
    data = parse_output(result)
    assert data["status"] == "ok"

    # Verify the attachment is base64-encoded in state
    persisted = json.loads((state_dir / "email_client_v2.json").read_text())
    sent = persisted["folders"]["SENT"]["emails"][0]
    assert "report.txt" in sent["attachments"]
    decoded = base64.b64decode(sent["attachments"]["report.txt"])
    assert decoded == b"attachment content here"


def test_send_email_with_attachment_relative_path(cli_env, fixed_uuid, fixed_time):
    """Relative cloud-drive paths (no leading /) also resolve correctly."""
    state_dir, runner = cli_env
    seed_state(state_dir, APP, _empty_state())

    fs_root = state_dir / "filesystem" / "Pictures" / "Personal"
    fs_root.mkdir(parents=True)
    pic = fs_root / "photo.jpg"
    pic.write_bytes(b"JPEG-data")

    result = runner.invoke(
        cli,
        [
            "send-email",
            "--recipients",
            '["bob@test.com"]',
            "--subject",
            "Photo",
            "--content",
            "See attached",
            "--attachment-paths",
            '["Pictures/Personal/photo.jpg"]',
        ],
    )
    assert result.exit_code == 0, result.output
    data = parse_output(result)
    assert data["status"] == "ok"

    persisted = json.loads((state_dir / "email_client_v2.json").read_text())
    sent = persisted["folders"]["SENT"]["emails"][0]
    assert "photo.jpg" in sent["attachments"]
    decoded = base64.b64decode(sent["attachments"]["photo.jpg"])
    assert decoded == b"JPEG-data"


# ===================================================================
# 9. send-email event format: app=EmailClientV2, fn=send_email, w=True
# ===================================================================


def test_send_email_event_format(cli_env, fixed_uuid, fixed_time):
    state_dir, runner = cli_env
    seed_state(state_dir, APP, _empty_state())

    result = runner.invoke(
        cli,
        [
            "send-email",
            "--recipients",
            '["bob@test.com"]',
            "--subject",
            "Event Test",
            "--content",
            "Body",
        ],
    )
    assert result.exit_code == 0, result.stderr

    events = read_events(state_dir)
    assert len(events) == 1
    ev = events[0]
    assert_event(ev, app=APP, fn="send_email", write=True)
    assert ev["args"]["recipients"] == ["bob@test.com"]
    assert ev["args"]["subject"] == "Event Test"
    assert ev["ret"] == fixed_uuid(0)


# ===================================================================
# 10. forward-email: FWD: prefix, parent_id set
# ===================================================================


def test_forward_email(cli_env, fixed_uuid, fixed_time):
    state_dir, runner = cli_env
    state = _empty_state()
    state["folders"]["INBOX"]["emails"] = [
        _make_email(
            email_id="orig1",
            subject="Important",
            content="Original body",
            timestamp=1000.0,
        ),
    ]
    seed_state(state_dir, APP, state)

    result = runner.invoke(
        cli,
        [
            "forward-email",
            "--email-id",
            "orig1",
            "--recipients",
            '["dave@test.com"]',
        ],
    )
    assert result.exit_code == 0, result.stderr
    data = parse_output(result)
    assert data["status"] == "ok"
    assert data["email_id"]

    # Check forwarded email in SENT
    persisted = json.loads((state_dir / "email_client_v2.json").read_text())
    sent_emails = persisted["folders"]["SENT"]["emails"]
    assert len(sent_emails) == 1
    fwd = sent_emails[0]
    assert fwd["subject"] == "FWD: Important"
    assert fwd["content"] == "> Original body"
    assert fwd["parent_id"] == "orig1"
    assert fwd["recipients"] == ["dave@test.com"]
    assert fwd["sender"] == "user@test.com"


# ===================================================================
# 11. reply-to-email: Re: prefix, parent chain resolution
# ===================================================================


def test_reply_to_email(cli_env, fixed_uuid, fixed_time):
    state_dir, runner = cli_env
    state = _empty_state()
    state["folders"]["INBOX"]["emails"] = [
        _make_email(
            email_id="msg1",
            sender="alice@test.com",
            subject="Question",
            content="How are you?",
            timestamp=1000.0,
        ),
    ]
    seed_state(state_dir, APP, state)

    result = runner.invoke(
        cli,
        [
            "reply-to-email",
            "--email-id",
            "msg1",
            "--content",
            "I am fine!",
        ],
    )
    assert result.exit_code == 0, result.stderr
    data = parse_output(result)
    assert data["status"] == "ok"

    persisted = json.loads((state_dir / "email_client_v2.json").read_text())
    sent_emails = persisted["folders"]["SENT"]["emails"]
    assert len(sent_emails) == 1
    reply = sent_emails[0]
    assert reply["subject"] == "Re: Question"
    assert reply["content"] == "I am fine!"
    assert reply["parent_id"] == "msg1"
    assert reply["recipients"] == ["alice@test.com"]


def test_reply_to_email_with_attachment(cli_env, fixed_uuid, fixed_time):
    """reply-to-email with --attachment-paths resolves cloud-drive paths."""
    state_dir, runner = cli_env
    state = _empty_state()
    state["folders"]["INBOX"]["emails"] = [
        _make_email(email_id="msg1", sender="alice@test.com", subject="Question"),
    ]
    seed_state(state_dir, APP, state)

    fs_dir = state_dir / "filesystem" / "Documents"
    fs_dir.mkdir(parents=True)
    (fs_dir / "notes.txt").write_bytes(b"reply-attachment-data")

    result = runner.invoke(
        cli,
        [
            "reply-to-email",
            "--email-id",
            "msg1",
            "--content",
            "Here are my notes",
            "--attachment-paths",
            '["/Documents/notes.txt"]',
        ],
    )
    assert result.exit_code == 0, result.output
    data = parse_output(result)
    assert data["status"] == "ok"

    persisted = json.loads((state_dir / "email_client_v2.json").read_text())
    sent = persisted["folders"]["SENT"]["emails"][0]
    assert "notes.txt" in sent["attachments"]
    decoded = base64.b64decode(sent["attachments"]["notes.txt"])
    assert decoded == b"reply-attachment-data"


def test_reply_to_email_from_user_with_attachment(cli_env, fixed_uuid, fixed_time):
    """ENV command reply-to-email-from-user resolves cloud-drive attachment paths."""
    state_dir, runner = cli_env
    state = _empty_state()
    # The ENV command replies to a SENT email (user sent it, NPC replies)
    state["folders"]["SENT"]["emails"] = [
        _make_email(
            email_id="sent1",
            sender="user@test.com",
            recipients=["alice@test.com"],
            subject="Hello",
        ),
    ]
    seed_state(state_dir, APP, state)

    fs_dir = state_dir / "filesystem" / "Pictures"
    fs_dir.mkdir(parents=True)
    (fs_dir / "photo.jpg").write_bytes(b"env-attachment-data")

    result = runner.invoke(
        cli,
        [
            "reply-to-email-from-user",
            "--email-id",
            "sent1",
            "--sender",
            "alice@test.com",
            "--content",
            "Here is the photo",
            "--attachment-paths",
            '["/Pictures/photo.jpg"]',
        ],
    )
    assert result.exit_code == 0, result.output

    persisted = json.loads((state_dir / "email_client_v2.json").read_text())
    inbox = persisted["folders"]["INBOX"]["emails"]
    assert len(inbox) == 1
    reply = inbox[0]
    assert "photo.jpg" in reply["attachments"]
    decoded = base64.b64decode(reply["attachments"]["photo.jpg"])
    assert decoded == b"env-attachment-data"


def test_reply_to_email_parent_chain_self_sender(cli_env, fixed_uuid, fixed_time):
    """When the original sender is the user, walk the parent chain to find
    the real external recipient."""
    state_dir, runner = cli_env
    state = _empty_state()

    # Build a chain:
    #   grandparent (from external alice) -> parent (from user, reply) -> child (from user)
    # The reply-to-email on the child should walk up and find alice.
    state["folders"]["INBOX"]["emails"] = [
        _make_email(
            email_id="gp1",
            sender="alice@test.com",
            subject="Thread",
            content="Start",
            timestamp=100.0,
        ),
    ]
    state["folders"]["SENT"]["emails"] = [
        _make_email(
            email_id="p1",
            sender="user@test.com",
            subject="Re: Thread",
            content="My reply",
            parent_id="gp1",
            timestamp=200.0,
        ),
    ]
    seed_state(state_dir, APP, state)

    # Reply to the sent email (sender == user) -> should resolve to alice
    result = runner.invoke(
        cli,
        [
            "reply-to-email",
            "--email-id",
            "p1",
            "--content",
            "Follow up",
            "--folder-name",
            "SENT",
        ],
    )
    assert result.exit_code == 0, result.stderr

    persisted = json.loads((state_dir / "email_client_v2.json").read_text())
    sent_emails = persisted["folders"]["SENT"]["emails"]
    # Should have 2 sent emails now (the original p1 + the new reply)
    new_replies = [e for e in sent_emails if e["email_id"] != "p1"]
    assert len(new_replies) == 1
    assert new_replies[0]["recipients"] == ["alice@test.com"]


# ===================================================================
# 12. download-attachments: base64 decode to filesystem
# ===================================================================


def test_download_attachments(cli_env):
    state_dir, runner = cli_env
    file_content = b"hello world binary"
    b64_content = base64.b64encode(file_content).decode()

    state = _empty_state()
    state["folders"]["INBOX"]["emails"] = [
        _make_email(
            email_id="att1",
            attachments={"data.bin": b64_content},
            timestamp=1000.0,
        ),
    ]
    seed_state(state_dir, APP, state)

    # Ensure the sandbox filesystem root exists
    (state_dir / "filesystem").mkdir(parents=True, exist_ok=True)

    result = runner.invoke(
        cli,
        [
            "download-attachments",
            "--email-id",
            "att1",
            "--path-to-save",
            "/Downloads",
        ],
    )
    assert result.exit_code == 0, result.output
    data = parse_output(result)
    assert data["status"] == "ok"
    assert len(data["downloaded"]) == 1

    # Verify the file was written correctly
    downloaded_path = data["downloaded"][0]
    with open(downloaded_path, "rb") as f:
        assert f.read() == file_content


# ===================================================================
# 13. move-email: from INBOX to TRASH
# ===================================================================


def test_move_email_inbox_to_trash(cli_env):
    state_dir, runner = cli_env
    state = _empty_state()
    state["folders"]["INBOX"]["emails"] = [
        _make_email(email_id="mv1", subject="Move Me", timestamp=1000.0),
    ]
    seed_state(state_dir, APP, state)

    result = runner.invoke(
        cli,
        [
            "move-email",
            "--email-id",
            "mv1",
            "--source-folder-name",
            "INBOX",
            "--dest-folder-name",
            "TRASH",
        ],
    )
    assert result.exit_code == 0, result.stderr
    data = parse_output(result)
    assert data["status"] == "ok"
    assert data["email_id"] == "mv1"

    # Verify the email moved in persisted state
    persisted = json.loads((state_dir / "email_client_v2.json").read_text())
    inbox_ids = [e["email_id"] for e in persisted["folders"]["INBOX"]["emails"]]
    trash_ids = [e["email_id"] for e in persisted["folders"]["TRASH"]["emails"]]
    assert "mv1" not in inbox_ids
    assert "mv1" in trash_ids


# ===================================================================
# 14. delete-email: moves to TRASH
# ===================================================================


def test_delete_email(cli_env):
    state_dir, runner = cli_env
    state = _empty_state()
    state["folders"]["INBOX"]["emails"] = [
        _make_email(email_id="del1", subject="Delete Me", timestamp=1000.0),
    ]
    seed_state(state_dir, APP, state)

    result = runner.invoke(
        cli,
        ["delete-email", "--email-id", "del1"],
    )
    assert result.exit_code == 0, result.stderr
    data = parse_output(result)
    assert data["status"] == "ok"
    assert data["email_id"] == "del1"

    # Verify moved to TRASH
    persisted = json.loads((state_dir / "email_client_v2.json").read_text())
    inbox_ids = [e["email_id"] for e in persisted["folders"]["INBOX"]["emails"]]
    trash_ids = [e["email_id"] for e in persisted["folders"]["TRASH"]["emails"]]
    assert "del1" not in inbox_ids
    assert "del1" in trash_ids

    # Verify event
    events = read_events(state_dir)
    assert len(events) == 1
    assert_event(events[0], app=APP, fn="delete_email", write=True)


# ===================================================================
# 15. search-emails: by subject substring
# ===================================================================


def test_search_emails_by_subject(cli_env):
    state_dir, runner = cli_env
    state = _empty_state()
    state["folders"]["INBOX"]["emails"] = [
        _make_email(email_id="s1", subject="Weekly Report", timestamp=100.0),
        _make_email(email_id="s2", subject="Meeting Notes", timestamp=200.0),
        _make_email(email_id="s3", subject="Weekly Summary", timestamp=300.0),
    ]
    seed_state(state_dir, APP, state)

    result = runner.invoke(cli, ["search-emails", "--query", "Weekly"])
    assert result.exit_code == 0, result.stderr
    data = parse_output(result)
    assert len(data) == 2
    subjects = [e["subject"] for e in data]
    # Results sorted by timestamp desc: s3 (300) then s1 (100)
    assert subjects == ["Weekly Summary", "Weekly Report"]


# ===================================================================
# 16. _load_state: handles folder objects missing "emails" key
# ===================================================================


def test_load_state_folder_without_emails_key(cli_env):
    """Folder dicts with no 'emails' key (e.g. {"folder_name": "INBOX"})
    must be treated as empty rather than crashing _sorted_emails()."""
    state_dir, runner = cli_env
    state = {
        "user_email": "user@test.com",
        "view_limit": 5,
        "folders": {
            "INBOX": {"folder_name": "INBOX"},  # no "emails" key
            "SENT": {"folder_name": "SENT", "emails": []},
            "DRAFT": {"folder_name": "DRAFT", "emails": []},
            "TRASH": {"folder_name": "TRASH", "emails": []},
        },
    }
    seed_state(state_dir, APP, state)

    result = runner.invoke(cli, ["list-emails"])
    assert result.exit_code == 0, result.output
    data = parse_output(result)
    assert data["total_emails"] == 0
    assert data["emails"] == []


def test_load_state_folder_is_not_a_dict(cli_env):
    """Non-dict folder values (e.g. a bare string) must not crash."""
    state_dir, runner = cli_env
    state = {
        "user_email": "user@test.com",
        "view_limit": 5,
        "folders": {
            "INBOX": "corrupted",  # not a dict at all
            "SENT": {"folder_name": "SENT", "emails": []},
            "DRAFT": {"folder_name": "DRAFT", "emails": []},
            "TRASH": {"folder_name": "TRASH", "emails": []},
        },
    }
    seed_state(state_dir, APP, state)

    result = runner.invoke(cli, ["list-emails"])
    assert result.exit_code == 0, result.output
    data = parse_output(result)
    assert data["total_emails"] == 0
    assert data["emails"] == []


# ===================================================================
# Email address validation
# ===================================================================


def test_send_email_rejects_name_as_recipient(cli_env, fixed_uuid, fixed_time):
    """send-email must reject non-email strings like names."""
    state_dir, runner = cli_env
    seed_state(state_dir, APP, _empty_state())

    result = runner.invoke(
        cli,
        [
            "send-email",
            "--recipients",
            '["Linnea Astrom"]',
            "--subject",
            "Hi",
            "--content",
            "Hello",
        ],
    )
    assert result.exit_code != 0
    assert "not a valid email address" in (result.output + (result.stderr or ""))


def test_send_email_rejects_name_in_cc(cli_env, fixed_uuid, fixed_time):
    """send-email must reject non-email strings in CC."""
    state_dir, runner = cli_env
    seed_state(state_dir, APP, _empty_state())

    result = runner.invoke(
        cli,
        [
            "send-email",
            "--recipients",
            '["valid@test.com"]',
            "--subject",
            "Hi",
            "--content",
            "Hello",
            "--cc",
            '["John Smith"]',
        ],
    )
    assert result.exit_code != 0
    assert "not a valid email address" in (result.output + (result.stderr or ""))


def test_forward_email_rejects_name_as_recipient(cli_env, fixed_uuid, fixed_time):
    """forward-email must reject non-email strings."""
    state_dir, runner = cli_env
    state = _empty_state()
    state["folders"]["INBOX"]["emails"] = [
        _make_email(email_id="msg1", sender="alice@test.com"),
    ]
    seed_state(state_dir, APP, state)

    result = runner.invoke(
        cli,
        [
            "forward-email",
            "--email-id",
            "msg1",
            "--recipients",
            '["Bob Jones"]',
        ],
    )
    assert result.exit_code != 0
    assert "not a valid email address" in (result.output + (result.stderr or ""))


def test_send_email_accepts_valid_email(cli_env, fixed_uuid, fixed_time):
    """send-email accepts properly formatted email addresses."""
    state_dir, runner = cli_env
    seed_state(state_dir, APP, _empty_state())

    result = runner.invoke(
        cli,
        [
            "send-email",
            "--recipients",
            '["linnea.astrom@gaia2mail.com"]',
            "--subject",
            "Hi",
            "--content",
            "Hello",
        ],
    )
    assert result.exit_code == 0, result.output
