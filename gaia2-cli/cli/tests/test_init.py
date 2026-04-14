# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
"""Unit tests for gaia2-init (gaia2_cli/init_cmd.py)."""

import base64
import json
from pathlib import Path

from conftest import make_cli_runner
from gaia2_cli.init_cmd import main

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_scenario(tmp_path: Path, data) -> Path:
    """Write scenario data to a JSON file and return the path."""
    scenario_path = tmp_path / "scenario.json"
    with open(scenario_path, "w") as f:
        json.dump(data, f)
    return scenario_path


def _read_state(state_dir: Path, filename: str) -> dict:
    """Read and parse a state JSON file from the state directory."""
    with open(state_dir / filename) as f:
        return json.load(f)


def _patch_attachments_dir(monkeypatch, attachments_dir: Path) -> None:
    """Redirect Path("/workspace/attachments/...") to a temp directory.

    Patches ``Path`` inside ``gaia2_cli.init_cmd`` so that any path starting
    with ``/workspace/attachments`` is rewritten to *attachments_dir*.
    """
    _OrigPath = Path

    def _patched_path(*args, **kwargs):
        p = _OrigPath(*args, **kwargs)
        s = str(p)
        if s.startswith("/workspace/attachments"):
            return _OrigPath(
                s.replace("/workspace/attachments", str(attachments_dir), 1)
            )
        return p

    monkeypatch.setattr("gaia2_cli.init_cmd.Path", _patched_path)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestBasicSplit:
    """Test 1: Scenario with 2 apps produces 2 state files + events.jsonl."""

    def test_two_apps_split(self, tmp_path):
        scenario = {
            "apps": [
                {
                    "name": "Calendar",
                    "app_state": {"events": {"ev1": {"title": "Meeting"}}},
                },
                {
                    "name": "Contacts",
                    "app_state": {"contacts": {"c1": {"first_name": "Alice"}}},
                },
            ]
        }
        scenario_path = _write_scenario(tmp_path, scenario)
        state_dir = tmp_path / "state"

        runner = make_cli_runner()
        result = runner.invoke(
            main,
            ["--scenario", str(scenario_path), "--state-dir", str(state_dir)],
        )

        assert result.exit_code == 0, f"CLI failed: {result.output}"

        # Two state files created
        assert (state_dir / "calendar.json").exists()
        assert (state_dir / "contacts.json").exists()

        # events.jsonl created (empty)
        assert (state_dir / "events.jsonl").exists()
        assert (state_dir / "events.jsonl").read_text() == ""

        # Verify state contents
        cal_state = _read_state(state_dir, "calendar.json")
        assert cal_state == {"events": {"ev1": {"title": "Meeting"}}}

        con_state = _read_state(state_dir, "contacts.json")
        assert con_state == {"contacts": {"c1": {"first_name": "Alice"}}}

        # Output mentions both apps
        assert "2 app(s)" in result.output
        assert "Calendar" in result.output
        assert "Contacts" in result.output


class TestAliasMailToEmailClientV2:
    """Test 2: 'Mail' alias resolves to 'EmailClientV2' (email_client_v2.json)."""

    def test_mail_alias(self, tmp_path):
        scenario = {
            "apps": [
                {
                    "name": "Mail",
                    "app_state": {"user_email": "test@test.com", "folders": {}},
                },
            ]
        }
        scenario_path = _write_scenario(tmp_path, scenario)
        state_dir = tmp_path / "state"

        runner = make_cli_runner()
        result = runner.invoke(
            main,
            ["--scenario", str(scenario_path), "--state-dir", str(state_dir)],
        )

        assert result.exit_code == 0, f"CLI failed: {result.output}"

        # State file uses the canonical name's normalized form
        assert (state_dir / "email_client_v2.json").exists()
        assert not (state_dir / "mail.json").exists()

        state = _read_state(state_dir, "email_client_v2.json")
        assert state["user_email"] == "test@test.com"


class TestMessagesApp:
    """Test 3: 'Messages' resolves to messages.json."""

    def test_messages_app(self, tmp_path):
        scenario = {
            "apps": [
                {"name": "Messages", "app_state": {"conversations": {}}},
            ]
        }
        scenario_path = _write_scenario(tmp_path, scenario)
        state_dir = tmp_path / "state"

        runner = make_cli_runner()
        result = runner.invoke(
            main,
            ["--scenario", str(scenario_path), "--state-dir", str(state_dir)],
        )

        assert result.exit_code == 0, f"CLI failed: {result.output}"

        assert (state_dir / "messages.json").exists()

        state = _read_state(state_dir, "messages.json")
        assert state == {"conversations": {}}


class TestChatsApp:
    """Test 4: 'Chats' resolves to chats.json."""

    def test_chats_app(self, tmp_path):
        scenario = {
            "apps": [
                {"name": "Chats", "app_state": {"conversations": {}}},
            ]
        }
        scenario_path = _write_scenario(tmp_path, scenario)
        state_dir = tmp_path / "state"

        runner = make_cli_runner()
        result = runner.invoke(
            main,
            ["--scenario", str(scenario_path), "--state-dir", str(state_dir)],
        )

        assert result.exit_code == 0, f"CLI failed: {result.output}"

        assert (state_dir / "chats.json").exists()

        state = _read_state(state_dir, "chats.json")
        assert state == {"conversations": {}}


class TestPublicScenarioNames:
    """Public scenario app names still resolve to the expected state files."""

    def test_public_names_resolve_to_state_files(self, tmp_path):
        scenario = {
            "apps": [
                {
                    "name": "Emails",
                    "class_name": "EmailClientV2",
                    "app_state": {"user_email": "test@test.com", "folders": {}},
                },
                {
                    "name": "Messages",
                    "class_name": "MessagingAppV2",
                    "app_state": {"conversations": {}},
                },
                {
                    "name": "Chats",
                    "class_name": "MessagingAppV2",
                    "app_state": {"conversations": {}},
                },
                {
                    "name": "RentAFlat",
                    "class_name": "RentAFlat",
                    "app_state": {"apartments": []},
                },
                {
                    "name": "Cabs",
                    "class_name": "CabApp",
                    "app_state": {"rides": []},
                },
                {
                    "name": "City",
                    "class_name": "CityApp",
                    "app_state": {"cities": []},
                },
            ]
        }
        scenario_path = _write_scenario(tmp_path, scenario)
        state_dir = tmp_path / "state"

        runner = make_cli_runner()
        result = runner.invoke(
            main,
            ["--scenario", str(scenario_path), "--state-dir", str(state_dir)],
        )

        assert result.exit_code == 0, f"CLI failed: {result.output}"
        assert (state_dir / "email_client_v2.json").exists()
        assert (state_dir / "messages.json").exists()
        assert (state_dir / "chats.json").exists()
        assert (state_dir / "rent_a_flat.json").exists()
        assert (state_dir / "cab_app.json").exists()
        assert (state_dir / "city_app.json").exists()


class TestBareListFormat:
    """Test 5: Scenario JSON is a bare list (not a dict with 'apps' key)."""

    def test_bare_list(self, tmp_path):
        scenario = [
            {"name": "Calendar", "app_state": {"events": {}}},
            {"name": "Shopping", "app_state": {"products": []}},
        ]
        scenario_path = _write_scenario(tmp_path, scenario)
        state_dir = tmp_path / "state"

        runner = make_cli_runner()
        result = runner.invoke(
            main,
            ["--scenario", str(scenario_path), "--state-dir", str(state_dir)],
        )

        assert result.exit_code == 0, f"CLI failed: {result.output}"

        assert (state_dir / "calendar.json").exists()
        assert (state_dir / "shopping.json").exists()
        assert "2 app(s)" in result.output


class TestNoAppsWarning:
    """Test 6: Empty apps list produces a warning on stderr."""

    def test_no_apps_warning(self, tmp_path):
        scenario = {"apps": []}
        scenario_path = _write_scenario(tmp_path, scenario)
        state_dir = tmp_path / "state"

        runner = make_cli_runner()
        result = runner.invoke(
            main,
            ["--scenario", str(scenario_path), "--state-dir", str(state_dir)],
        )

        assert result.exit_code == 0
        # stderr warning about missing apps (Click 8.3+ captures stderr separately)
        assert "Warning: No apps found in scenario JSON" in (
            getattr(result, "stderr", "") or result.output
        )

        # events.jsonl still created even with no apps
        assert (state_dir / "events.jsonl").exists()
        assert "0 app(s)" in result.output

    def test_no_apps_key_warning(self, tmp_path):
        """Dict scenario without an 'apps' key also warns."""
        scenario = {"something_else": "value"}
        scenario_path = _write_scenario(tmp_path, scenario)
        state_dir = tmp_path / "state"

        runner = make_cli_runner()
        result = runner.invoke(
            main,
            ["--scenario", str(scenario_path), "--state-dir", str(state_dir)],
        )

        assert result.exit_code == 0
        assert "Warning: No apps found in scenario JSON" in (
            getattr(result, "stderr", "") or result.output
        )


class TestDecodeEmailAttachments:
    """Test 7: Email app state with base64 attachments decoded to filesystem."""

    def test_email_attachments_decoded(self, tmp_path, monkeypatch):
        file_content = b"Hello, this is a test attachment."
        b64_content = base64.b64encode(file_content).decode("ascii")

        scenario = {
            "apps": [
                {
                    "name": "EmailClientV2",
                    "app_state": {
                        "user_email": "user@example.com",
                        "folders": {
                            "INBOX": {
                                "folder_name": "INBOX",
                                "emails": [
                                    {
                                        "email_id": "email_001",
                                        "subject": "Test",
                                        "attachments": {
                                            "report.pdf": b64_content,
                                        },
                                    }
                                ],
                            }
                        },
                    },
                }
            ]
        }
        scenario_path = _write_scenario(tmp_path, scenario)
        state_dir = tmp_path / "state"
        attachments_dir = tmp_path / "attachments"

        _patch_attachments_dir(monkeypatch, attachments_dir)

        runner = make_cli_runner()
        result = runner.invoke(
            main,
            ["--scenario", str(scenario_path), "--state-dir", str(state_dir)],
        )

        assert result.exit_code == 0, f"CLI failed: {result.output}"

        # State file written
        assert (state_dir / "email_client_v2.json").exists()

        # Attachment decoded to filesystem
        decoded_path = attachments_dir / "email_001" / "report.pdf"
        assert decoded_path.exists(), f"Expected decoded file at {decoded_path}"
        assert decoded_path.read_bytes() == file_content

    def test_mail_alias_with_attachments(self, tmp_path, monkeypatch):
        """Legacy 'Mail' name also triggers attachment decoding."""
        file_content = b"PDF data here"
        b64_content = base64.b64encode(file_content).decode("ascii")

        scenario = {
            "apps": [
                {
                    "name": "Mail",
                    "app_state": {
                        "user_email": "u@example.com",
                        "folders": {
                            "INBOX": {
                                "folder_name": "INBOX",
                                "emails": [
                                    {
                                        "email_id": "em_42",
                                        "subject": "Invoice",
                                        "attachments": {"invoice.pdf": b64_content},
                                    }
                                ],
                            }
                        },
                    },
                }
            ]
        }
        scenario_path = _write_scenario(tmp_path, scenario)
        state_dir = tmp_path / "state"
        attachments_dir = tmp_path / "attachments"

        _patch_attachments_dir(monkeypatch, attachments_dir)

        runner = make_cli_runner()
        result = runner.invoke(
            main,
            ["--scenario", str(scenario_path), "--state-dir", str(state_dir)],
        )

        assert result.exit_code == 0, f"CLI failed: {result.output}"

        # Alias resolved: state file is email_client_v2.json
        assert (state_dir / "email_client_v2.json").exists()

        # Attachment decoded
        decoded_path = attachments_dir / "em_42" / "invoice.pdf"
        assert decoded_path.exists()
        assert decoded_path.read_bytes() == file_content


class TestDecodeMessagingAttachments:
    """Test 8: Messaging app state with base64 attachments in conversations."""

    def test_chats_attachments_decoded(self, tmp_path, monkeypatch):
        file_content = b"\x89PNG\r\n\x1a\nfake image data"
        b64_content = base64.b64encode(file_content).decode("ascii")

        scenario = {
            "apps": [
                {
                    "name": "Chats",
                    "app_state": {
                        "conversations": {
                            "conv_001": {
                                "participants": ["Alice", "Bob"],
                                "messages": [
                                    {
                                        "message_id": "msg_101",
                                        "sender": "Alice",
                                        "body": "See attached photo",
                                        "attachments": {
                                            "photo.png": b64_content,
                                        },
                                    }
                                ],
                            }
                        }
                    },
                }
            ]
        }
        scenario_path = _write_scenario(tmp_path, scenario)
        state_dir = tmp_path / "state"
        attachments_dir = tmp_path / "attachments"

        _patch_attachments_dir(monkeypatch, attachments_dir)

        runner = make_cli_runner()
        result = runner.invoke(
            main,
            ["--scenario", str(scenario_path), "--state-dir", str(state_dir)],
        )

        assert result.exit_code == 0, f"CLI failed: {result.output}"

        # State file written with correct normalized name
        assert (state_dir / "chats.json").exists()

        # Attachment decoded from conversation message
        decoded_path = attachments_dir / "msg_101" / "photo.png"
        assert decoded_path.exists(), f"Expected decoded file at {decoded_path}"
        assert decoded_path.read_bytes() == file_content

    def test_messages_attachments_decoded(self, tmp_path, monkeypatch):
        """'Messages' app state triggers attachment decoding."""
        file_content = b"document bytes"
        b64_content = base64.b64encode(file_content).decode("ascii")

        scenario = {
            "apps": [
                {
                    "name": "Messages",
                    "app_state": {
                        "conversations": {
                            "conv_abc": {
                                "participants": ["Carol"],
                                "messages": [
                                    {
                                        "message_id": "msg_xyz",
                                        "sender": "Carol",
                                        "body": "Here is the doc",
                                        "attachments": {"doc.txt": b64_content},
                                    }
                                ],
                            }
                        }
                    },
                }
            ]
        }
        scenario_path = _write_scenario(tmp_path, scenario)
        state_dir = tmp_path / "state"
        attachments_dir = tmp_path / "attachments"

        _patch_attachments_dir(monkeypatch, attachments_dir)

        runner = make_cli_runner()
        result = runner.invoke(
            main,
            ["--scenario", str(scenario_path), "--state-dir", str(state_dir)],
        )

        assert result.exit_code == 0, f"CLI failed: {result.output}"

        assert (state_dir / "messages.json").exists()

        decoded_path = attachments_dir / "msg_xyz" / "doc.txt"
        assert decoded_path.exists()
        assert decoded_path.read_bytes() == file_content
