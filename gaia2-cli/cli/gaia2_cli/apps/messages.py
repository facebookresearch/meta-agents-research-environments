# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
"""Standalone CLI for the Messages messaging app (NAME mode)."""

import base64
import json
import os
import re
import time
import uuid
from datetime import datetime, timezone
from difflib import SequenceMatcher
from functools import partial
from typing import Any

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
    set_log_class,
)

APP_NAME = "Messages"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_state() -> dict[str, Any]:
    state = load_app_state(APP_NAME)
    if not state:
        cli_error(f"State file for {APP_NAME} not found")
    return state


def _get_default_title(state: dict[str, Any], user_ids: list[str]) -> str:
    """Build a default title from participant names, excluding the current user."""
    current_user_id = state["current_user_id"]
    id_to_name = state["id_to_name"]

    ids_without_current = set(user_ids)
    ids_without_current.discard(current_user_id)

    names: list[str] = []
    for uid in ids_without_current:
        if uid in id_to_name:
            names.append(id_to_name[uid])

    return ", ".join(sorted(names))


def _get_messages_in_date_range(
    messages: list[dict[str, Any]],
    min_date: str | None,
    max_date: str | None,
) -> list[dict[str, Any]]:
    start_ts = -float("inf")
    end_ts = float("inf")
    if min_date is not None:
        start_ts = (
            datetime.strptime(min_date, "%Y-%m-%d %H:%M:%S")
            .replace(tzinfo=timezone.utc)
            .timestamp()
        )
    if max_date is not None:
        end_ts = (
            datetime.strptime(max_date, "%Y-%m-%d %H:%M:%S")
            .replace(tzinfo=timezone.utc)
            .timestamp()
        )
    return [m for m in messages if start_ts <= int(m["timestamp"]) <= end_ts]


def _validate_user_ids(state: dict[str, Any], user_ids: list[str]) -> None:
    mode = state.get("mode", "NAME")
    for uid in user_ids:
        if mode == "PHONE_NUMBER":
            if all(not c.isdigit() for c in uid):
                raise ValueError(
                    f"User {uid} is not a valid phone number, "
                    "you need to provide the phone number of an existing user"
                )
        else:
            if uid not in state["id_to_name"]:
                raise ValueError(
                    f"User {uid} does not exist in Ids, "
                    "you need to provide the Id of an existing user"
                )


def _get_or_create_default_conversation(
    state: dict[str, Any],
    user_ids: list[str],
) -> str:
    """Find or create a default 1:1 conversation for the given user_ids."""
    current_user_id = state["current_user_id"]
    conversations = state["conversations"]

    target_set = set(user_ids)
    target_set.add(current_user_id)

    matching_conv_ids: list[str] = []
    for conv_id, conv in conversations.items():
        if set(conv["participant_ids"]) == target_set:
            matching_conv_ids.append(conv_id)

    default_title = _get_default_title(state, user_ids)

    for conv_id in matching_conv_ids:
        if conversations[conv_id]["title"] == default_title:
            return conv_id

    _validate_user_ids(state, user_ids)

    participants = list(target_set)
    conv_id = uuid.uuid4().hex
    conversations[conv_id] = {
        "conversation_id": conv_id,
        "title": default_title,
        "participant_ids": list(set(participants)),
        "last_updated": time.time(),
        "messages": [],
    }
    return conv_id


def _create_message(
    state: dict[str, Any],
    content: str,
    attachment_path: str | None,
) -> dict[str, Any]:
    current_user_id = state["current_user_id"]
    msg: dict[str, Any] = {
        "sender_id": current_user_id,
        "message_id": uuid.uuid4().hex,
        "timestamp": time.time(),
        "content": content,
    }
    if attachment_path is not None:
        resolved = resolve_sandbox_path(attachment_path)
        if not os.path.exists(resolved):
            raise ValueError(f"File {attachment_path} does not exist")
        with open(resolved, "rb") as f:
            file_content = base64.b64encode(f.read()).decode("utf-8")
        msg["attachment"] = file_content
        msg["attachment_name"] = os.path.basename(resolved)
    return msg


def _apply_conversation_limits(
    conversations: list[dict[str, Any]],
    messages_view_limit: int,
    offset_msgs: int,
    limit_msgs: int,
) -> list[dict[str, Any]]:
    if limit_msgs > messages_view_limit:
        raise ValueError(
            f"Limit must be smaller than the view limit of {messages_view_limit} "
            "- Please use a smaller limit and use offset to navigate"
        )
    if offset_msgs < 0:
        raise ValueError("Offset must be non-negative")

    result = []
    for conv in conversations:
        sorted_msgs = sorted(
            conv["messages"],
            key=lambda m: m["timestamp"],
            reverse=True,
        )
        end = min(len(sorted_msgs), offset_msgs + limit_msgs)
        trimmed = dict(conv)
        trimmed["messages"] = sorted_msgs[offset_msgs:end]
        result.append(trimmed)
    return result


# ---------------------------------------------------------------------------
# Search helpers
# ---------------------------------------------------------------------------


def _conversation_matches(
    conv: dict[str, Any],
    query: str,
    min_date: str | None,
    max_date: str | None,
) -> bool:
    if any(query in p.lower() for p in conv["participant_ids"]):
        return True
    if conv.get("title") and query in conv["title"].lower():
        return True
    messages = _get_messages_in_date_range(conv["messages"], min_date, max_date)
    return any(query in m.get("content", "").lower() for m in messages)


def _regex_conversation_matches(
    conv: dict[str, Any],
    get_match: Any,
    min_date: str | None,
    max_date: str | None,
) -> bool:
    if any(get_match(p.lower()) for p in conv["participant_ids"]):
        return True
    if conv.get("title") and get_match(conv["title"].lower()):
        return True
    messages = _get_messages_in_date_range(conv["messages"], min_date, max_date)
    return any(get_match(m.get("content", "")) for m in messages)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@click.group(context_settings={"terminal_width": 10000, "max_content_width": 10000})
def cli() -> None:
    set_app(APP_NAME)
    set_log_class("MessagingAppV2")


@cli.command()
def schema() -> None:
    """Print machine-readable schema of all commands."""
    json_output(build_schema(cli))


@cli.command("get-user-id")
@click.option("--user-name", required=True, help="Exact user name to look up")
def get_user_id(user_name: str) -> None:
    """Get user id for a given user name (exact match)."""
    state = _load_state()
    result = state["name_to_id"].get(user_name, None)
    log_action("get_user_id", {"user_name": user_name}, result)
    json_output(result)


@cli.command("get-user-name-from-id")
@click.option("--user-id", required=True, help="Exact user id to look up")
def get_user_name_from_id(user_id: str) -> None:
    """Get user name for a given user id (exact match)."""
    state = _load_state()
    result = state["id_to_name"].get(user_id, None)
    log_action("get_user_name_from_id", {"user_id": user_id}, result)
    json_output(result)


@cli.command("lookup-user-id")
@click.option("--user-name", required=True, help="User name to fuzzy-search")
def lookup_user_id(user_name: str) -> None:
    """Fuzzy lookup of user id by name. Returns dict of matching name->id pairs."""
    state = _load_state()
    name_to_id = state["name_to_id"]

    if user_name in name_to_id:
        result = {user_name: name_to_id[user_name]}
    else:
        query = user_name.lower().strip()
        result = {}
        for name, uid in name_to_id.items():
            ratio = SequenceMatcher(None, query, name.lower()).ratio()
            if ratio > 0.65:
                result[name] = uid

    log_action("lookup_user_id", {"user_name": user_name}, result)
    json_output(result)


@cli.command("send-message")
@click.option("--user-id", required=True, help="Recipient user id")
@click.option("--content", default="", help="Message content")
@click.option(
    "--attachment-path",
    default=None,
    help="Cloud-drive path to attach, e.g. /Documents/report.pdf",
)
def send_message(user_id: str, content: str, attachment_path: str | None) -> None:
    """Send a message to a user. Creates conversation if none exists."""
    state = _load_state()
    try:
        msg = _create_message(state, content, attachment_path)
        conv_id = _get_or_create_default_conversation(state, [user_id])
        state["conversations"][conv_id]["messages"].append(msg)
        state["conversations"][conv_id]["last_updated"] = max(
            msg["timestamp"],
            state["conversations"][conv_id]["last_updated"],
        )
        save_app_state(APP_NAME, state)
        result = conv_id
    except Exception as e:
        cli_error(str(e))
        return

    log_action(
        "send_message",
        {
            "user_id": user_id,
            "content": content,
            "attachment_path": attachment_path,
        },
        result,
        write=True,
    )
    json_output(result)


@cli.command("send-message-to-group-conversation")
@click.option("--conversation-id", required=True, help="Group conversation id")
@click.option("--content", default="", help="Message content")
@click.option(
    "--attachment-path",
    default=None,
    help="Cloud-drive path to attach, e.g. /Documents/report.pdf",
)
def send_message_to_group_conversation(
    conversation_id: str,
    content: str,
    attachment_path: str | None,
) -> None:
    """Send a message to an existing group conversation (3+ participants)."""
    state = _load_state()
    try:
        if conversation_id not in state["conversations"]:
            raise ValueError(f"Conversation {conversation_id} does not exist")
        if len(state["conversations"][conversation_id]["participant_ids"]) < 3:
            raise ValueError(
                f"Conversation {conversation_id} is not a group conversation"
            )
        msg = _create_message(state, content, attachment_path)
        state["conversations"][conversation_id]["messages"].append(msg)
        state["conversations"][conversation_id]["last_updated"] = max(
            msg["timestamp"],
            state["conversations"][conversation_id]["last_updated"],
        )
        save_app_state(APP_NAME, state)
        result = conversation_id
    except Exception as e:
        cli_error(str(e))
        return

    log_action(
        "send_message_to_group_conversation",
        {
            "conversation_id": conversation_id,
            "content": content,
            "attachment_path": attachment_path,
        },
        result,
        write=True,
    )
    json_output(result)


@cli.command("create-group-conversation")
@click.option(
    "--user-ids",
    required=True,
    help='JSON list of other participant user ids (not yourself), e.g. \'["id1","id2"]\'',
)
@click.option("--title", default=None, help="Optional conversation title")
def create_group_conversation(user_ids: str, title: str | None) -> None:
    """Create a new group conversation. Requires at least 2 other participants (you are added automatically)."""
    state = _load_state()
    try:
        parsed_ids: list[str] = json.loads(user_ids)
        if len(parsed_ids) < 2:
            raise Exception("Must have at least two other participants")
        _validate_user_ids(state, parsed_ids)

        current_user_id = state["current_user_id"]
        participant_set = set(parsed_ids)
        participant_set.add(current_user_id)
        if len(participant_set) < 3:
            raise Exception(
                "Must have at least two other participants "
                "(do not include your own user id)"
            )
        conv_title = _get_default_title(state, parsed_ids) if title is None else title

        conv_id = uuid.uuid4().hex
        state["conversations"][conv_id] = {
            "conversation_id": conv_id,
            "title": conv_title,
            "participant_ids": list(participant_set),
            "last_updated": time.time(),
            "messages": [],
        }
        save_app_state(APP_NAME, state)
        result = conv_id
    except Exception as e:
        cli_error(str(e))
        return

    log_action(
        "create_group_conversation",
        {"user_ids": parsed_ids, "title": title},
        result,
        write=True,
    )
    json_output(result)


@cli.command("get-existing-conversation-ids")
@click.option("--user-ids", required=True, help="JSON list of participant user ids")
def get_existing_conversation_ids(user_ids: str) -> None:
    """Get conversation ids matching the exact set of participants."""
    state = _load_state()
    parsed_ids: list[str] = json.loads(user_ids)
    current_user_id = state["current_user_id"]
    target_set = set(parsed_ids)
    target_set.add(current_user_id)

    result: list[str] = []
    for conv_id, conv in state["conversations"].items():
        if set(conv["participant_ids"]) == target_set:
            result.append(conv_id)

    log_action("get_existing_conversation_ids", {"user_ids": parsed_ids}, result)
    json_output(result)


@cli.command("list-recent-conversations")
@click.option("--offset", default=0, type=int, help="Starting index")
@click.option("--limit", default=5, type=int, help="Number of conversations")
@click.option(
    "--offset-recent-messages-per-conversation",
    default=0,
    type=int,
    help="Message offset per conversation",
)
@click.option(
    "--limit-recent-messages-per-conversation",
    default=10,
    type=int,
    help="Message limit per conversation",
)
def list_recent_conversations(
    offset: int,
    limit: int,
    offset_recent_messages_per_conversation: int,
    limit_recent_messages_per_conversation: int,
) -> None:
    """List conversations ordered by most recent update."""
    state = _load_state()
    try:
        conv_view_limit = state.get("conversation_view_limit", 5)
        msg_view_limit = state.get("messages_view_limit", 10)

        if limit > conv_view_limit:
            raise ValueError(
                f"Limit must be smaller than the view limit of {conv_view_limit} "
                "- Please use a smaller limit and use offset to navigate"
            )
        if offset < 0:
            raise ValueError("Offset must be non-negative")
        if offset > len(state["conversations"]):
            raise ValueError("Offset is larger than the number of conversations")

        sorted_convs = sorted(
            state["conversations"].values(),
            key=lambda c: c["last_updated"],
            reverse=True,
        )
        end = min(len(sorted_convs), offset + limit)
        page = sorted_convs[offset:end]

        result = _apply_conversation_limits(
            page,
            msg_view_limit,
            offset_recent_messages_per_conversation,
            limit_recent_messages_per_conversation,
        )
    except Exception as e:
        cli_error(str(e))
        return

    log_action(
        "list_recent_conversations",
        {
            "offset": offset,
            "limit": limit,
            "offset_recent_messages_per_conversation": offset_recent_messages_per_conversation,
            "limit_recent_messages_per_conversation": limit_recent_messages_per_conversation,
        },
        result,
    )
    json_output(result)


@cli.command("list-conversations-by-participant")
@click.option("--user-id", required=True, help="Participant user id")
@click.option("--offset", default=0, type=int, help="Starting index")
@click.option("--limit", default=5, type=int, help="Number of conversations")
@click.option(
    "--offset-recent-messages-per-conversation",
    default=0,
    type=int,
    help="Message offset per conversation",
)
@click.option(
    "--limit-recent-messages-per-conversation",
    default=5,
    type=int,
    help="Message limit per conversation",
)
def list_conversations_by_participant(
    user_id: str,
    offset: int,
    limit: int,
    offset_recent_messages_per_conversation: int,
    limit_recent_messages_per_conversation: int,
) -> None:
    """List conversations that include the given participant."""
    state = _load_state()
    try:
        conv_view_limit = state.get("conversation_view_limit", 5)
        msg_view_limit = state.get("messages_view_limit", 10)

        if limit > conv_view_limit:
            raise ValueError(
                f"Limit must be smaller than the view limit of {conv_view_limit} "
                "- Please use a smaller limit and use offset to navigate"
            )
        if offset < 0:
            raise ValueError("Offset must be non-negative")
        if offset > len(state["conversations"]):
            raise ValueError("Offset is larger than the number of conversations")

        matching = [
            conv
            for conv in state["conversations"].values()
            if any(user_id in pid for pid in conv["participant_ids"])
        ]

        end = min(len(matching), offset + limit)
        page = matching[offset:end]

        result = _apply_conversation_limits(
            page,
            msg_view_limit,
            offset_recent_messages_per_conversation,
            limit_recent_messages_per_conversation,
        )
    except Exception as e:
        cli_error(str(e))
        return

    log_action(
        "list_conversations_by_participant",
        {
            "user_id": user_id,
            "offset": offset,
            "limit": limit,
            "offset_recent_messages_per_conversation": offset_recent_messages_per_conversation,
            "limit_recent_messages_per_conversation": limit_recent_messages_per_conversation,
        },
        result,
    )
    json_output(result)


@cli.command("download-attachment")
@click.option("--conversation-id", required=True, help="Conversation id")
@click.option("--message-id", required=True, help="Message id")
@click.option(
    "--download-path",
    default="Downloads/",
    help="Cloud-drive directory to save attachment to",
)
def download_attachment(
    conversation_id: str,
    message_id: str,
    download_path: str,
) -> None:
    """Download an attachment from a message."""
    state = _load_state()
    try:
        if conversation_id not in state["conversations"]:
            raise Exception(f"Conversation with id {conversation_id} not found")
        conv = state["conversations"][conversation_id]
        message = next(
            (m for m in conv["messages"] if m["message_id"] == message_id),
            None,
        )
        if message is None or "attachment" not in message or not message["attachment"]:
            raise Exception("No attachment found in the specified message")

        file_data = base64.b64decode(message["attachment"])
        resolved_dir = resolve_sandbox_path(download_path)
        full_path = os.path.join(resolved_dir, message["attachment_name"])
        os.makedirs(
            os.path.dirname(full_path) if os.path.dirname(full_path) else resolved_dir,
            exist_ok=True,
        )
        with open(full_path, "wb") as f:
            f.write(file_data)

        result = full_path
    except Exception as e:
        cli_error(str(e))
        return

    log_action(
        "download_attachment",
        {
            "conversation_id": conversation_id,
            "message_id": message_id,
            "download_path": download_path,
        },
        result,
        write=True,
    )
    json_output(result)


@cli.command("read-conversation")
@click.option("--conversation-id", required=True, help="Conversation id")
@click.option("--offset", default=0, type=int, help="Message offset")
@click.option("--limit", default=10, type=int, help="Number of messages")
@click.option("--min-date", default=None, help="Min date (YYYY-MM-DD HH:MM:SS)")
@click.option("--max-date", default=None, help="Max date (YYYY-MM-DD HH:MM:SS)")
def read_conversation(
    conversation_id: str,
    offset: int,
    limit: int,
    min_date: str | None,
    max_date: str | None,
) -> None:
    """Read messages from a conversation (most recent first)."""
    state = _load_state()
    try:
        if conversation_id not in state["conversations"]:
            raise ValueError(f"Conversation with id {conversation_id} not found")
        if offset < 0:
            raise ValueError("Offset must be positive")

        conv = state["conversations"][conversation_id]
        messages = _get_messages_in_date_range(conv["messages"], min_date, max_date)
        if offset > len(messages):
            raise ValueError("Offset is larger than the number of messages")

        messages.sort(key=lambda m: m["timestamp"], reverse=True)
        conversation_length = len(messages)
        end = min(len(messages), offset + limit)
        page = messages[offset:end]

        result = {
            "messages": page,
            "metadata": {
                "message_range": [offset, end],
                "conversation_length": conversation_length,
                "conversation_id": conversation_id,
                "conversation_title": conv["title"],
            },
        }
    except Exception as e:
        cli_error(str(e))
        return

    log_action(
        "read_conversation",
        {
            "conversation_id": conversation_id,
            "offset": offset,
            "limit": limit,
            "min_date": min_date,
            "max_date": max_date,
        },
        result,
    )
    json_output(result)


@cli.command("add-participant-to-conversation")
@click.option("--conversation-id", required=True, help="Conversation id")
@click.option("--user-id", required=True, help="User id to add")
def add_participant_to_conversation(
    conversation_id: str,
    user_id: str,
) -> None:
    """Add a user to a conversation. Updates title if it was the default."""
    state = _load_state()
    try:
        if conversation_id not in state["conversations"]:
            raise Exception(f"Conversation with id {conversation_id} not found")
        conv = state["conversations"][conversation_id]
        if user_id in conv["participant_ids"]:
            raise Exception(f"Participant {user_id} already in conversation")

        old_default = _get_default_title(state, conv["participant_ids"])
        update_title = conv["title"] == old_default

        conv["participant_ids"].append(user_id)

        if update_title:
            conv["title"] = _get_default_title(state, conv["participant_ids"])

        save_app_state(APP_NAME, state)
        result = conversation_id
    except Exception as e:
        cli_error(str(e))
        return

    log_action(
        "add_participant_to_conversation",
        {"conversation_id": conversation_id, "user_id": user_id},
        result,
        write=True,
    )
    json_output(result)


@cli.command("remove-participant-from-conversation")
@click.option("--conversation-id", required=True, help="Conversation id")
@click.option("--user-id", required=True, help="User id to remove")
def remove_participant_from_conversation(
    conversation_id: str,
    user_id: str,
) -> None:
    """Remove a user from a conversation. Updates title if it was the default."""
    state = _load_state()
    try:
        if conversation_id not in state["conversations"]:
            raise Exception(f"Conversation with id {conversation_id} not found")
        conv = state["conversations"][conversation_id]
        if user_id not in conv["participant_ids"]:
            raise Exception(f"Participant {user_id} not in conversation")

        old_default = _get_default_title(state, conv["participant_ids"])
        update_title = conv["title"] == old_default

        conv["participant_ids"].remove(user_id)

        if update_title:
            conv["title"] = _get_default_title(state, conv["participant_ids"])

        save_app_state(APP_NAME, state)
        result = conversation_id
    except Exception as e:
        cli_error(str(e))
        return

    log_action(
        "remove_participant_from_conversation",
        {"conversation_id": conversation_id, "user_id": user_id},
        result,
        write=True,
    )
    json_output(result)


@cli.command("change-conversation-title")
@click.option("--conversation-id", required=True, help="Conversation id")
@click.option("--title", required=True, help="New title")
def change_conversation_title(conversation_id: str, title: str) -> None:
    """Change the title of a conversation."""
    state = _load_state()
    try:
        if conversation_id not in state["conversations"]:
            raise Exception(f"Conversation with id {conversation_id} not found")
        state["conversations"][conversation_id]["title"] = title
        save_app_state(APP_NAME, state)
        result = conversation_id
    except Exception as e:
        cli_error(str(e))
        return

    log_action(
        "change_conversation_title",
        {"conversation_id": conversation_id, "title": title},
        result,
        write=True,
    )
    json_output(result)


@cli.command("search")
@click.option("--query", required=True, help="Case-insensitive search query")
@click.option("--min-date", default=None, help="Min date (YYYY-MM-DD HH:MM:SS)")
@click.option("--max-date", default=None, help="Max date (YYYY-MM-DD HH:MM:SS)")
def search_cmd(query: str, min_date: str | None, max_date: str | None) -> None:
    """Search conversations by participants, title, and message content."""
    state = _load_state()
    q = query.lower()
    results: list[str] = []
    for conv in state["conversations"].values():
        if _conversation_matches(conv, q, min_date, max_date):
            results.append(conv["conversation_id"])

    log_action(
        "search",
        {"query": query, "min_date": min_date, "max_date": max_date},
        results,
    )
    json_output(results)


@cli.command("regex-search")
@click.option("--query", required=True, help="Regex search query")
@click.option("--min-date", default=None, help="Min date (YYYY-MM-DD HH:MM:SS)")
@click.option("--max-date", default=None, help="Max date (YYYY-MM-DD HH:MM:SS)")
def regex_search_cmd(
    query: str,
    min_date: str | None,
    max_date: str | None,
) -> None:
    """Search conversations using a regex pattern (case-insensitive)."""
    state = _load_state()
    compiled = re.compile(query, re.IGNORECASE)
    get_match = partial(re.search, compiled)
    results: list[str] = []
    for conv in state["conversations"].values():
        if _regex_conversation_matches(conv, get_match, min_date, max_date):
            results.append(conv["conversation_id"])

    log_action(
        "regex_search",
        {"query": query, "min_date": min_date, "max_date": max_date},
        results,
    )
    json_output(results)


# ===========================================================================
# Hidden ENV-tool commands (used by gaia2-eventd, not visible to agents)
# ===========================================================================


@cli.command("create-and-add-message", hidden=True)
@click.option("--conversation-id", required=True, help="Conversation id")
@click.option("--sender-id", required=True, help="Sender user id (the NPC sending)")
@click.option("--content", default="", help="Message content")
def create_and_add_message(conversation_id: str, sender_id: str, content: str) -> None:
    """[ENV] Add a message from an external sender to a conversation."""
    state = _load_state()
    try:
        if conversation_id not in state["conversations"]:
            raise Exception(f"Conversation with id {conversation_id} not found")
        conv = state["conversations"][conversation_id]
        if sender_id not in conv["participant_ids"]:
            raise Exception(f"Sender {sender_id} not in conversation")

        msg: dict[str, Any] = {
            "sender_id": sender_id,
            "message_id": uuid.uuid4().hex,
            "timestamp": time.time(),
            "content": content,
        }
        conv["messages"].append(msg)
        conv["last_updated"] = max(msg["timestamp"], conv["last_updated"])
        save_app_state(APP_NAME, state)
        result = conversation_id
    except Exception as e:
        cli_error(str(e))
        return

    log_action(
        "create_and_add_message",
        {
            "conversation_id": conversation_id,
            "sender_id": sender_id,
            "content": content,
        },
        result,
        write=True,
    )
    json_output(result)


if __name__ == "__main__":
    cli()
