# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
"""
Standalone CLI for the Gaia2 EmailClientV2 app.

Binary name: ``emails``
App class name for events: ``EmailClientV2``

State format (matches original EmailClientApp.get_state()):
{
  "user_email": "...",
  "view_limit": 5,
  "folders": {
    "INBOX": {"folder_name": "INBOX", "emails": [list of email dicts]},
    "SENT":  {"folder_name": "SENT",  "emails": [...]},
    "DRAFT": {"folder_name": "DRAFT", "emails": [...]},
    "TRASH": {"folder_name": "TRASH", "emails": [...]},
  }
}

Each email dict has fields: sender, recipients, subject, content, email_id,
parent_id, cc, attachments (dict[str, str] mapping filename -> base64), timestamp, is_read.
"""

import base64
import json
import os
import time
import uuid

import click

from gaia2_cli.shared import (
    build_schema,
    cli_error,
    json_output,
    load_app_state,
    log_action,
    resolve_sandbox_path,
    save_app_state,
    set_app,
    validate_email_list,
)

APP_NAME = "EmailClientV2"
DEFAULT_USER_EMAIL = "user@example.com"

set_app(APP_NAME)
VALID_FOLDERS = {"INBOX", "SENT", "DRAFT", "TRASH"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_json_list(value: str | None) -> list[str]:
    """Parse a JSON-encoded list of strings from a CLI option."""
    if value is None:
        return []
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        raise click.BadParameter(f"Invalid JSON list: {value}")
    if not isinstance(parsed, list):
        raise click.BadParameter(f"Expected a JSON list, got: {type(parsed).__name__}")
    return [str(item) for item in parsed]


def _default_state() -> dict:
    """Return a default state skeleton."""
    return {
        "user_email": DEFAULT_USER_EMAIL,
        "view_limit": 5,
        "folders": {
            fname: {"folder_name": fname, "emails": []} for fname in VALID_FOLDERS
        },
    }


def _load_state() -> dict:
    """Load email state, returning a default skeleton if missing.

    Internally converts the list-based email storage to dict-based
    (keyed by email_id) for fast lookups.  The on-disk format stores
    emails as lists to match the original EmailClientApp.get_state().
    """
    state = load_app_state(APP_NAME)
    if not state:
        state = _default_state()

    # Convert list-based folders -> dict-based (email_id -> email) for internal use
    folders = state.get("folders", {})
    for fname in list(folders.keys()):
        folder_obj = folders[fname]
        emails_list = (
            folder_obj.get("emails", []) if isinstance(folder_obj, dict) else []
        )
        folders[fname] = {
            e["email_id"]: e
            for e in emails_list
            if isinstance(e, dict) and e.get("email_id")
        }

    return state


def _save_state(state: dict) -> None:
    """Save state, converting internal dict-based folders back to list format."""
    save_copy = dict(state)
    save_copy["folders"] = {}

    for fname, emails_dict in state.get("folders", {}).items():
        if isinstance(emails_dict, dict):
            emails_list = sorted(
                emails_dict.values(),
                key=lambda e: e.get("timestamp", 0),
                reverse=True,
            )
        else:
            emails_list = emails_dict

        save_copy["folders"][fname] = {
            "folder_name": fname,
            "emails": emails_list,
        }

    save_app_state(APP_NAME, save_copy)


def _get_folder(state: dict, folder_name: str) -> dict:
    """Return the email dict {email_id: email} for a folder."""
    folder_name = folder_name.upper()
    folders = state.get("folders", {})
    if folder_name not in folders:
        raise ValueError(f"Folder {folder_name} not found")
    return folders[folder_name]


def _find_email(
    state: dict, email_id: str, folder_name: str | None = None
) -> tuple[dict, str]:
    """Find an email by ID in a specific folder (or all folders).

    Returns (email_dict, folder_name).
    """
    if folder_name:
        folder = _get_folder(state, folder_name)
        if email_id in folder:
            return folder[email_id], folder_name.upper()
        raise ValueError(f"Email with id {email_id} does not exist in {folder_name}")
    for fname, folder in state.get("folders", {}).items():
        if isinstance(folder, dict) and email_id in folder:
            return folder[email_id], fname
    raise ValueError(f"Email with id {email_id} does not exist")


def _sorted_emails(folder: dict) -> list[dict]:
    """Return emails in a folder sorted by timestamp descending."""
    return sorted(folder.values(), key=lambda e: e.get("timestamp", 0), reverse=True)


# ---------------------------------------------------------------------------
# CLI group
# ---------------------------------------------------------------------------


@click.group(context_settings={"terminal_width": 10000, "max_content_width": 10000})
def cli():
    """EmailClientV2 — manage emails, folders, and attachments."""
    pass


# ---------------------------------------------------------------------------
# 1. list-emails (READ)
# ---------------------------------------------------------------------------


@cli.command("list-emails")
@click.option(
    "--folder-name",
    "--folder",
    default="INBOX",
    help="Folder name: INBOX, SENT, DRAFT, TRASH.",
)
@click.option("--offset", default=0, type=int, help="Offset of first email to return.")
@click.option(
    "--limit", default=10, type=int, help="Maximum number of emails to return."
)
def list_emails(folder_name: str, offset: int, limit: int):
    """List emails in a folder with pagination."""
    try:
        state = _load_state()
        folder_data = _get_folder(state, folder_name)
        emails = _sorted_emails(folder_data)

        total = len(emails)
        if offset < 0:
            raise ValueError("Offset must be non-negative")
        if offset > total:
            raise ValueError("Offset must be less than the number of emails")

        end = min(offset + limit, total)
        page = emails[offset:end]

        result = {
            "emails": page,
            "emails_range": [offset, end],
            "total_returned_emails": len(page),
            "total_emails": total,
        }

        log_action(
            "list_emails",
            {"folder_name": folder_name.upper(), "offset": offset, "limit": limit},
            ret=result,
        )
        json_output(result)
    except Exception as e:
        cli_error(str(e))


# ---------------------------------------------------------------------------
# 2. get-email-by-id (READ)
# ---------------------------------------------------------------------------


@cli.command("get-email-by-id")
@click.option("--email-id", required=True, help="The email ID.")
@click.option(
    "--folder-name",
    "--folder",
    default="INBOX",
    help="Folder name (INBOX, SENT, DRAFT, TRASH).",
)
def get_email_by_id(email_id: str, folder_name: str):
    """Get an email by its ID, marking it as read."""
    try:
        state = _load_state()
        folder_data = _get_folder(state, folder_name)
        if email_id not in folder_data:
            raise ValueError(f"Email with id {email_id} does not exist")

        email = folder_data[email_id]
        email["is_read"] = True
        _save_state(state)

        log_action(
            "get_email_by_id",
            {"email_id": email_id, "folder_name": folder_name.upper()},
            ret=email,
        )
        json_output(email)
    except Exception as e:
        cli_error(str(e))


# ---------------------------------------------------------------------------
# 3. get-email-by-index (READ)
# ---------------------------------------------------------------------------


@cli.command("get-email-by-index")
@click.option(
    "--idx",
    required=True,
    type=int,
    help="Index in the folder (0-based, newest first).",
)
@click.option(
    "--folder-name",
    "--folder",
    default="INBOX",
    help="Folder name (INBOX, SENT, DRAFT, TRASH).",
)
def get_email_by_index(idx: int, folder_name: str):
    """Get an email by its index in a folder, marking it as read."""
    try:
        state = _load_state()
        folder_data = _get_folder(state, folder_name)
        emails = _sorted_emails(folder_data)

        if idx < 0 or idx >= len(emails):
            raise ValueError(f"Email with index {idx} does not exist")

        email = emails[idx]
        email["is_read"] = True
        _save_state(state)

        log_action(
            "get_email_by_index",
            {"idx": idx, "folder_name": folder_name.upper()},
            ret=email,
        )
        json_output(email)
    except Exception as e:
        cli_error(str(e))


# ---------------------------------------------------------------------------
# 4. send-email (WRITE)
# ---------------------------------------------------------------------------


@cli.command("send-email")
@click.option(
    "--recipients",
    required=True,
    help="JSON list of recipient email addresses, e.g. '[\"a@b.com\"]'.",
)
@click.option("--subject", default="", help="Email subject.")
@click.option("--content", default="", help="Email body.")
@click.option("--cc", default=None, help="JSON list of CC addresses (optional).")
@click.option(
    "--attachment-paths",
    default=None,
    help="JSON list of cloud-drive paths to attach, e.g. '[\"/Documents/report.pdf\"]'.",
)
def send_email(
    recipients: str,
    subject: str,
    content: str,
    cc: str | None,
    attachment_paths: str | None,
):
    """Send an email to the specified recipients."""
    parsed_recipients = _parse_json_list(recipients)
    parsed_cc = _parse_json_list(cc)
    parsed_attachments = _parse_json_list(attachment_paths)

    try:
        validate_email_list(parsed_recipients, "recipients")
        validate_email_list(parsed_cc, "cc")

        state = _load_state()
        user_email = state.get("user_email", DEFAULT_USER_EMAIL)

        # Build attachments dict: {filename: base64_encoded_content}
        attachments: dict[str, str] = {}
        for path in parsed_attachments:
            resolved = resolve_sandbox_path(path)
            if not os.path.exists(resolved):
                raise ValueError(f"File does not exist: {path}")
            with open(resolved, "rb") as f:
                encoded = base64.b64encode(f.read()).decode("utf-8")
            attachments[os.path.basename(resolved)] = encoded

        email_id = uuid.uuid4().hex
        email_obj = {
            "sender": user_email,
            "recipients": parsed_recipients,
            "subject": subject,
            "content": content,
            "email_id": email_id,
            "parent_id": None,
            "cc": parsed_cc,
            "attachments": attachments,
            "timestamp": time.time(),
            "is_read": True,
        }

        sent_folder = _get_folder(state, "SENT")
        sent_folder[email_id] = email_obj
        _save_state(state)

        log_action(
            "send_email",
            {
                "recipients": parsed_recipients,
                "subject": subject,
                "content": content,
                "cc": parsed_cc,
                "attachment_paths": parsed_attachments,
            },
            ret=email_id,
            write=True,
        )
        json_output({"status": "ok", "email_id": email_id})
    except Exception as e:
        cli_error(str(e))


# ---------------------------------------------------------------------------
# 5. forward-email (WRITE)
# ---------------------------------------------------------------------------


@cli.command("forward-email")
@click.option("--email-id", required=True, help="ID of the email to forward.")
@click.option(
    "--recipients", required=True, help="JSON list of recipient email addresses."
)
@click.option(
    "--folder-name",
    "--folder",
    default="INBOX",
    help="Folder containing the email (INBOX, SENT, DRAFT, TRASH).",
)
def forward_email(email_id: str, recipients: str, folder_name: str):
    """Forward an email to new recipients."""
    parsed_recipients = _parse_json_list(recipients)

    try:
        validate_email_list(parsed_recipients, "recipients")

        state = _load_state()
        user_email = state.get("user_email", DEFAULT_USER_EMAIL)

        folder_data = _get_folder(state, folder_name)
        if email_id not in folder_data:
            raise ValueError(f"Email with id {email_id} does not exist")
        original = folder_data[email_id]

        new_id = uuid.uuid4().hex
        forwarded = {
            "sender": user_email,
            "recipients": parsed_recipients,
            "subject": "FWD: " + original.get("subject", ""),
            "content": "> " + original.get("content", ""),
            "email_id": new_id,
            "parent_id": email_id,
            "cc": [],
            "attachments": original.get("attachments", {}),
            "timestamp": time.time(),
            "is_read": True,
        }

        sent_folder = _get_folder(state, "SENT")
        sent_folder[new_id] = forwarded
        _save_state(state)

        log_action(
            "forward_email",
            {
                "email_id": email_id,
                "recipients": parsed_recipients,
                "folder_name": folder_name.upper(),
            },
            ret=new_id,
            write=True,
        )
        json_output({"status": "ok", "email_id": new_id})
    except Exception as e:
        cli_error(str(e))


# ---------------------------------------------------------------------------
# 6. download-attachments (WRITE)
# ---------------------------------------------------------------------------


@cli.command("download-attachments")
@click.option("--email-id", required=True, help="ID of the email.")
@click.option(
    "--folder-name",
    "--folder",
    default="INBOX",
    help="Folder containing the email (INBOX, SENT, DRAFT, TRASH).",
)
@click.option(
    "--path-to-save",
    "--download-path",
    default="Downloads/",
    help="Cloud-drive directory to save attachments to.",
)
def download_attachments(email_id: str, folder_name: str, path_to_save: str):
    """Download attachments from an email to the local filesystem."""
    try:
        state = _load_state()
        email, _ = _find_email(state, email_id, folder_name)
        attachments = email.get("attachments", {})
        filenames: list[str] = []

        if attachments and isinstance(attachments, dict):
            resolved_dir = resolve_sandbox_path(path_to_save)
            os.makedirs(resolved_dir, exist_ok=True)
            for filename, b64_content in attachments.items():
                full_path = os.path.join(resolved_dir, filename)
                with open(full_path, "wb") as f:
                    f.write(base64.b64decode(b64_content))
                filenames.append(full_path)

        log_action(
            "download_attachments",
            {
                "email_id": email_id,
                "folder_name": folder_name.upper(),
                "path_to_save": path_to_save,
            },
            ret=filenames,
            write=True,
        )
        json_output({"status": "ok", "downloaded": filenames})
    except Exception as e:
        cli_error(str(e))


# ---------------------------------------------------------------------------
# 7. reply-to-email (WRITE)
# ---------------------------------------------------------------------------


@cli.command("reply-to-email")
@click.option("--email-id", required=True, help="ID of the email to reply to.")
@click.option("--content", default="", help="Reply body.")
@click.option(
    "--folder-name",
    "--folder",
    default="INBOX",
    help="Folder containing the original email (INBOX, SENT, DRAFT, TRASH).",
)
@click.option(
    "--attachment-paths",
    default=None,
    help="JSON list of cloud-drive paths to attach, e.g. '[\"/Documents/report.pdf\"]'.",
)
def reply_to_email(
    email_id: str, content: str, folder_name: str, attachment_paths: str | None
):
    """Reply to an email."""
    parsed_attachments = _parse_json_list(attachment_paths)

    try:
        state = _load_state()
        user_email = state.get("user_email", DEFAULT_USER_EMAIL)

        folder_data = _get_folder(state, folder_name)
        if email_id not in folder_data:
            raise ValueError(f"Email with id {email_id} does not exist")
        original = folder_data[email_id]

        # Determine recipient: walk parent chain if sender is self
        recipient = original.get("sender", "")
        parent = original
        while parent.get("sender") == user_email and parent.get("parent_id"):
            parent, _ = _find_email(state, parent["parent_id"])
            recipient = parent.get("sender", recipient)

        # Build attachments dict: {filename: base64_encoded_content}
        attachments: dict[str, str] = {}
        for path in parsed_attachments:
            resolved = resolve_sandbox_path(path)
            if not os.path.exists(resolved):
                raise ValueError(f"File does not exist: {path}")
            with open(resolved, "rb") as f:
                encoded = base64.b64encode(f.read()).decode("utf-8")
            attachments[os.path.basename(resolved)] = encoded

        new_id = uuid.uuid4().hex
        reply = {
            "sender": user_email,
            "recipients": [recipient],
            "subject": "Re: " + original.get("subject", ""),
            "content": content,
            "email_id": new_id,
            "parent_id": email_id,
            "cc": [],
            "attachments": attachments,
            "timestamp": time.time(),
            "is_read": True,
        }

        sent_folder = _get_folder(state, "SENT")
        sent_folder[new_id] = reply
        _save_state(state)

        log_action(
            "reply_to_email",
            {
                "email_id": email_id,
                "folder_name": folder_name.upper(),
                "content": content,
                "attachment_paths": parsed_attachments,
            },
            ret=new_id,
            write=True,
        )
        json_output({"status": "ok", "email_id": new_id})
    except Exception as e:
        cli_error(str(e))


# ---------------------------------------------------------------------------
# 8. move-email (WRITE)
# ---------------------------------------------------------------------------


@cli.command("move-email")
@click.option("--email-id", required=True, help="ID of the email to move.")
@click.option(
    "--source-folder-name",
    "--source-folder",
    required=True,
    help="Source folder name (INBOX, SENT, DRAFT, TRASH).",
)
@click.option(
    "--dest-folder-name",
    "--dest-folder",
    required=True,
    help="Destination folder name (INBOX, SENT, DRAFT, TRASH).",
)
def move_email(email_id: str, source_folder_name: str, dest_folder_name: str):
    """Move an email from one folder to another."""
    try:
        state = _load_state()
        src = _get_folder(state, source_folder_name)
        dst = _get_folder(state, dest_folder_name)

        if email_id not in src:
            raise ValueError(
                f"Email with id {email_id} does not exist in {source_folder_name}"
            )

        email = src.pop(email_id)
        dst[email_id] = email
        _save_state(state)

        log_action(
            "move_email",
            {
                "email_id": email_id,
                "source_folder_name": source_folder_name.upper(),
                "dest_folder_name": dest_folder_name.upper(),
            },
            ret=email_id,
            write=True,
        )
        json_output({"status": "ok", "email_id": email_id})
    except Exception as e:
        cli_error(str(e))


# ---------------------------------------------------------------------------
# 9. delete-email (WRITE)
# ---------------------------------------------------------------------------


@cli.command("delete-email")
@click.option("--email-id", required=True, help="ID of the email to delete.")
@click.option(
    "--folder-name",
    "--folder",
    default="INBOX",
    help="Folder containing the email (INBOX, SENT, DRAFT, TRASH).",
)
def delete_email(email_id: str, folder_name: str):
    """Delete an email (moves it to TRASH)."""
    try:
        state = _load_state()
        src = _get_folder(state, folder_name)
        trash = _get_folder(state, "TRASH")

        if email_id not in src:
            raise ValueError(
                f"Email with id {email_id} does not exist in {folder_name}"
            )

        email = src.pop(email_id)
        trash[email_id] = email
        _save_state(state)

        log_action(
            "delete_email",
            {"email_id": email_id, "folder_name": folder_name.upper()},
            ret=email_id,
            write=True,
        )
        json_output({"status": "ok", "email_id": email_id})
    except Exception as e:
        cli_error(str(e))


# ---------------------------------------------------------------------------
# 10. search-emails (READ)
# ---------------------------------------------------------------------------


@cli.command("search-emails")
@click.option("--query", required=True, help="Search query string.")
@click.option(
    "--folder-name",
    "--folder",
    default="INBOX",
    help="Folder to search in (INBOX, SENT, DRAFT, TRASH).",
)
def search_emails(query: str, folder_name: str):
    """Search emails by query string across sender, recipients, subject, and content."""
    try:
        state = _load_state()
        folder_data = _get_folder(state, folder_name)
        q = query.lower()
        results: list[dict] = []

        for email in folder_data.values():
            if not isinstance(email, dict):
                continue
            sender = email.get("sender", "").lower()
            recipients = [r.lower() for r in email.get("recipients", [])]
            subject = email.get("subject", "").lower()
            content = email.get("content", "").lower()
            if (
                q in sender
                or any(q in r for r in recipients)
                or q in subject
                or q in content
            ):
                results.append(email)

        results.sort(key=lambda e: e.get("timestamp", 0), reverse=True)

        log_action(
            "search_emails",
            {"query": query, "folder_name": folder_name.upper()},
            ret=results,
        )
        json_output(results)
    except Exception as e:
        cli_error(str(e))


# ===========================================================================
# Hidden ENV-tool commands (used by gaia2-eventd, not visible to agents)
# ===========================================================================


@cli.command("create-and-add-email", hidden=True)
@click.option("--sender", required=True, help="Sender email address.")
@click.option("--recipients", default=None, help="JSON list of recipient addresses.")
@click.option("--subject", default="", help="Email subject.")
@click.option("--content", default="", help="Email body.")
@click.option("--folder-name", default="INBOX", help="Target folder.")
def create_and_add_email(
    sender: str,
    recipients: str | None,
    subject: str,
    content: str,
    folder_name: str,
):
    """[ENV] Create and add an email to a folder (simulates incoming mail)."""
    parsed_recipients = _parse_json_list(recipients) if recipients else []

    try:
        state = _load_state()
        email_id = uuid.uuid4().hex
        email_obj = {
            "sender": sender,
            "recipients": parsed_recipients,
            "subject": subject,
            "content": content,
            "email_id": email_id,
            "parent_id": None,
            "cc": [],
            "attachments": {},
            "timestamp": time.time(),
            "is_read": False,
        }

        folder = _get_folder(state, folder_name)
        folder[email_id] = email_obj
        _save_state(state)

        log_action(
            "create_and_add_email",
            {
                "sender": sender,
                "recipients": parsed_recipients,
                "subject": subject,
                "content": content,
                "folder_name": folder_name.upper(),
            },
            ret=email_id,
            write=True,
        )
        json_output({"status": "ok", "email_id": email_id})
    except Exception as e:
        cli_error(str(e))


@cli.command("send-email-to-user-only", hidden=True)
@click.option("--sender", required=True, help="Sender email address.")
@click.option("--subject", default="", help="Email subject.")
@click.option("--content", default="", help="Email body.")
def send_email_to_user_only(sender: str, subject: str, content: str):
    """[ENV] Send an email to the user's INBOX (simulates incoming mail)."""
    try:
        state = _load_state()
        user_email = state.get("user_email", DEFAULT_USER_EMAIL)

        email_id = uuid.uuid4().hex
        email_obj = {
            "sender": sender,
            "recipients": [user_email],
            "subject": subject,
            "content": content,
            "email_id": email_id,
            "parent_id": None,
            "cc": [],
            "attachments": {},
            "timestamp": time.time(),
            "is_read": False,
        }

        inbox = _get_folder(state, "INBOX")
        inbox[email_id] = email_obj
        _save_state(state)

        log_action(
            "send_email_to_user_only",
            {"sender": sender, "subject": subject, "content": content},
            ret=email_id,
            write=True,
        )
        json_output({"status": "ok", "email_id": email_id})
    except Exception as e:
        cli_error(str(e))


@cli.command("reply-to-email-from-user", hidden=True)
@click.option(
    "--sender",
    required=True,
    help="Sender email address (the external person replying).",
)
@click.option(
    "--email-id",
    required=True,
    help="ID of the email being replied to (in SENT folder).",
)
@click.option("--content", default="", help="Reply body.")
@click.option(
    "--attachment-paths", default=None, help="JSON list of attachment file paths."
)
def reply_to_email_from_user(
    sender: str, email_id: str, content: str, attachment_paths: str | None
):
    """[ENV] Reply to a user's sent email (simulates NPC reply to user's INBOX)."""
    parsed_attachments = _parse_json_list(attachment_paths)

    try:
        state = _load_state()
        user_email = state.get("user_email", DEFAULT_USER_EMAIL)

        # Find the original email in SENT folder
        sent = _get_folder(state, "SENT")
        if email_id not in sent:
            raise ValueError(f"Email with id {email_id} does not exist in SENT")
        original = sent[email_id]

        # Build attachments dict
        attachments: dict[str, str] = {}
        for path in parsed_attachments:
            resolved = resolve_sandbox_path(path)
            if not os.path.exists(resolved):
                raise ValueError(f"File does not exist: {path}")
            with open(resolved, "rb") as f:
                encoded = base64.b64encode(f.read()).decode("utf-8")
            attachments[os.path.basename(resolved)] = encoded

        new_id = uuid.uuid4().hex
        reply = {
            "sender": sender,
            "recipients": [user_email],
            "subject": "Re: " + original.get("subject", ""),
            "content": content,
            "email_id": new_id,
            "parent_id": email_id,
            "cc": [],
            "attachments": attachments,
            "timestamp": time.time(),
            "is_read": False,
        }

        inbox = _get_folder(state, "INBOX")
        inbox[new_id] = reply
        _save_state(state)

        log_action(
            "reply_to_email_from_user",
            {
                "sender": sender,
                "email_id": email_id,
                "content": content,
                "attachment_paths": parsed_attachments,
            },
            ret=new_id,
            write=True,
        )
        json_output({"status": "ok", "email_id": new_id})
    except Exception as e:
        cli_error(str(e))


# ---------------------------------------------------------------------------
# schema
# ---------------------------------------------------------------------------


@cli.command("schema")
def schema():
    """Print machine-readable schema for all commands."""
    json_output(build_schema(cli))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    cli()


if __name__ == "__main__":
    main()
