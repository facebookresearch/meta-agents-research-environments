# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
"""
gaia2-init: Split a scenario JSON into per-app state files.

Reads a scenario JSON file, iterates over the app_state blocks, and writes one
JSON file per app into the state directory. Also creates an empty events.jsonl
for later use by CLI binaries.

For email/messaging apps whose state contains base64-encoded attachment data
inside messages, decodes the data and writes the files to the real filesystem
so that CLI tools can reference them by path.

Usage:
    gaia2-init --scenario /workspace/scenario.json --state-dir /workspace/state
"""

import base64
import json
import os
import shutil
from pathlib import Path

import click

# Map scenario app names to the canonical APP_NAME used by each CLI module
# (determines the state file name). Sourced from app_registry.py.
from gaia2_cli.app_registry import APP_NAME_TO_STATE as _APP_NAME_TO_STATE_RAW
from gaia2_cli.shared import (
    make_serializable,
    state_file_for_app,
)

_APP_NAME_ALIASES = {
    name: state
    for name, state in _APP_NAME_TO_STATE_RAW.items()
    if name != state  # only include actual aliases, not identity mappings
}

# Canonical app names that use filesystem initialization instead of JSON state.
_FILESYSTEM_APPS = {"Files"}

# Apps whose message/email state may contain base64-encoded attachment data.
_ATTACHMENT_APP_CLASSES = {
    "EmailClientApp",
    "EmailClientV2",
    "Mail",
    "Messages",
    "Chats",
}


def _decode_attachments_in_state(app_name: str, app_state: dict) -> None:  # noqa: C901
    """Walk the app state looking for base64-encoded attachment data.

    Email apps store attachments as ``{filename: base64_string}`` dicts inside
    email objects. We decode them and write the binary files so the container
    filesystem has them available for download_attachments calls.

    This mutates *app_state* in place — attachment values are left as-is in the
    JSON (they're still needed by the CLI for download), but the binary files
    are written to disk.
    """
    if app_name not in _ATTACHMENT_APP_CLASSES:
        return

    # Email apps: state may have folders at the top level (legacy format) or
    # nested under app_state["folders"] (EmailClientV2 format):
    #   {"user_email": "...", "folders": {"INBOX": {"folder_name": "INBOX", "emails": [...]}, ...}}
    # We check both locations.
    _FOLDER_NAMES = (
        "inbox",
        "sent",
        "draft",
        "trash",
        "INBOX",
        "SENT",
        "DRAFT",
        "TRASH",
    )

    def _decode_email_attachments(folder: dict) -> None:
        """Decode base64 attachments from emails in a folder dict."""
        # Handle list-based email storage (EmailClientV2 format: {"emails": [...]})
        emails_list = folder.get("emails", [])
        if isinstance(emails_list, list):
            for email_obj in emails_list:
                if not isinstance(email_obj, dict):
                    continue
                attachments = email_obj.get("attachments")
                if not attachments or not isinstance(attachments, dict):
                    continue
                email_id = email_obj.get("email_id", "unknown")
                for filename, b64_data in attachments.items():
                    if not isinstance(b64_data, str):
                        continue
                    try:
                        file_bytes = base64.b64decode(b64_data)
                        out_dir = Path("/workspace/attachments") / str(email_id)
                        out_dir.mkdir(parents=True, exist_ok=True)
                        out_path = out_dir / filename
                        with open(out_path, "wb") as f:
                            f.write(file_bytes)
                    except Exception as e:
                        click.echo(
                            f"Warning: Failed to decode attachment {filename} "
                            f"for email {email_id}: {e}",
                            err=True,
                        )

        # Handle dict-based email storage (legacy format: {"<id>": {attachments: {...}}})
        for email_id, email_obj in folder.items():
            if email_id in ("folder_name", "emails"):
                continue
            if not isinstance(email_obj, dict):
                continue
            attachments = email_obj.get("attachments")
            if not attachments or not isinstance(attachments, dict):
                continue
            for filename, b64_data in attachments.items():
                if not isinstance(b64_data, str):
                    continue
                try:
                    file_bytes = base64.b64decode(b64_data)
                    out_dir = Path("/workspace/attachments") / email_id
                    out_dir.mkdir(parents=True, exist_ok=True)
                    out_path = out_dir / filename
                    with open(out_path, "wb") as f:
                        f.write(file_bytes)
                except Exception as e:
                    click.echo(
                        f"Warning: Failed to decode attachment {filename} "
                        f"for email {email_id}: {e}",
                        err=True,
                    )

    # Check top-level folder keys (legacy format)
    for folder_name in _FOLDER_NAMES:
        folder = app_state.get(folder_name, {})
        if isinstance(folder, dict):
            _decode_email_attachments(folder)

    # Check nested folders structure (EmailClientV2 format)
    nested_folders = app_state.get("folders", {})
    if isinstance(nested_folders, dict):
        for _folder_name, folder in nested_folders.items():
            if isinstance(folder, dict):
                _decode_email_attachments(folder)

    # Messaging apps: state has conversations -> messages -> attachments
    conversations = app_state.get("conversations", {})
    if not isinstance(conversations, dict):
        return
    for conv_id, conv in conversations.items():
        if not isinstance(conv, dict):
            continue
        messages = conv.get("messages", [])
        if not isinstance(messages, list):
            continue
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            attachments = msg.get("attachments")
            if not attachments or not isinstance(attachments, dict):
                continue
            msg_id = msg.get("message_id", conv_id)
            for filename, b64_data in attachments.items():
                if not isinstance(b64_data, str):
                    continue
                try:
                    file_bytes = base64.b64decode(b64_data)
                    out_dir = Path("/workspace/attachments") / str(msg_id)
                    out_dir.mkdir(parents=True, exist_ok=True)
                    out_path = out_dir / filename
                    with open(out_path, "wb") as f:
                        f.write(file_bytes)
                except Exception as e:
                    click.echo(
                        f"Warning: Failed to decode attachment {filename} "
                        f"for message {msg_id}: {e}",
                        err=True,
                    )


def _init_filesystem(
    app_state: dict, state_dir: Path, fs_backing_dir: str | None
) -> None:
    """Create the filesystem directory tree from scenario state.

    Walks the recursive tree in ``app_state["files"]`` and creates the
    corresponding directories and files under ``<state_dir>/filesystem/``.

    For file nodes with a ``real_path`` field, the actual content is copied
    from ``<fs_backing_dir>/<real_path>``.  If the backing file is missing
    or *fs_backing_dir* is ``None``, an empty placeholder is created instead.
    """
    fs_root = state_dir / "filesystem"
    fs_root.mkdir(parents=True, exist_ok=True)

    files_tree = app_state.get("files", {})

    def _walk_children(node: dict, parent: Path) -> None:
        """Walk children of a directory node, creating files and dirs."""
        for child in node.get("children", []):
            name = child.get("name", "")
            if not name:
                continue
            child_type = child.get("type", "file")
            current = parent / name

            if child_type == "directory":
                current.mkdir(parents=True, exist_ok=True)
                _walk_children(child, current)
            else:
                # File node — try to copy from backing store
                current.parent.mkdir(parents=True, exist_ok=True)
                real_path = child.get("real_path")
                copied = False
                if real_path and fs_backing_dir:
                    src = os.path.join(fs_backing_dir, real_path)
                    if os.path.isfile(src):
                        shutil.copy2(src, current)
                        copied = True
                if not copied:
                    current.touch()

    if files_tree:
        # The root node maps to fs_root regardless of its name
        # (Gaia2 uses a temp dir name like "tmppom6e92_" which we ignore).
        _walk_children(files_tree, fs_root)


@click.command(context_settings={"terminal_width": 10000, "max_content_width": 10000})
@click.option(
    "--scenario",
    required=True,
    type=click.Path(exists=True),
    help="Path to the scenario JSON file",
)
@click.option(
    "--state-dir",
    required=True,
    type=click.Path(),
    help="Directory to write per-app state files into",
)
@click.option(
    "--fs-backing-dir",
    default=None,
    type=click.Path(),
    help="Directory containing demo_filesystem backing files (default: $GAIA2_FS_BACKING_DIR).",
)
def main(scenario: str, state_dir: str, fs_backing_dir: str | None):
    """Split a scenario JSON into per-app state files."""
    scenario_path = Path(scenario)
    state_dir_path = Path(state_dir)
    state_dir_path.mkdir(parents=True, exist_ok=True)

    # Resolve backing dir from CLI flag or env var
    if fs_backing_dir is None:
        fs_backing_dir = os.environ.get("GAIA2_FS_BACKING_DIR")

    # Read scenario JSON
    with open(scenario_path) as f:
        scenario_data = json.load(f)

    # The scenario JSON can be either:
    # 1. A dict with an "apps" key: {"apps": [{...}, ...]}
    # 2. A bare list of app entries: [{...}, ...]
    # Each entry has: name, class_name (optional), app_state (dict)
    if isinstance(scenario_data, list):
        apps_list = scenario_data
    else:
        apps_list = scenario_data.get("apps", [])
    if not apps_list:
        click.echo("Warning: No apps found in scenario JSON", err=True)

    written = []
    app_class_map: dict[str, str] = {}  # canonical APP_NAME → scenario class_name
    for app_entry in apps_list:
        # Support both dict and object-style entries
        if isinstance(app_entry, dict):
            app_name = app_entry.get("name", "")
            class_name = app_entry.get("class_name", app_name)
            app_state = app_entry.get("app_state", {})
        else:
            app_name = getattr(app_entry, "name", "")
            class_name = getattr(app_entry, "class_name", app_name)
            app_state = getattr(app_entry, "app_state", {})

        if not app_name:
            click.echo("Warning: Skipping app entry with no name", err=True)
            continue

        # Resolve legacy app name to canonical name before normalizing
        canonical_name = _APP_NAME_ALIASES.get(app_name, app_name)

        # Record the mapping from CLI APP_NAME to scenario class_name.
        # log_action() uses this to write the correct "app" field in events.jsonl.
        if canonical_name != class_name:
            app_class_map[canonical_name] = class_name

        # Filesystem apps get a real directory tree instead of a JSON state file
        if canonical_name in _FILESYSTEM_APPS and isinstance(app_state, dict):
            _init_filesystem(app_state, state_dir_path, fs_backing_dir)
            # Also write the JSON state file (for reference / grading)
            state_file = state_file_for_app(canonical_name, state_dir_path)
            with open(state_file, "w") as f:
                json.dump(make_serializable(app_state or {}), f, indent=2)
                f.write("\n")
            written.append(app_name)
            continue

        # Decode base64 attachments in email/messaging app state
        # (uses canonical name so compatibility names like "Mail" are recognized)
        if isinstance(app_state, dict):
            _decode_attachments_in_state(canonical_name, app_state)

        state_file = state_file_for_app(canonical_name, state_dir_path)
        with open(state_file, "w") as f:
            json.dump(make_serializable(app_state or {}), f, indent=2)
            f.write("\n")
        written.append(app_name)

    # Write app_class_map.json so log_action() can resolve the correct
    # class_name for events.jsonl (e.g. "Chats" → "MessagingAppV2").
    if app_class_map:
        map_file = state_dir_path / "app_class_map.json"
        with open(map_file, "w") as f:
            json.dump(app_class_map, f, indent=2)
            f.write("\n")

    # Create empty events.jsonl
    events_file = state_dir_path / "events.jsonl"
    events_file.touch()

    click.echo(f"Initialized {len(written)} app(s): {', '.join(written)}")
    click.echo(f"State directory: {state_dir_path}")


if __name__ == "__main__":
    main()
