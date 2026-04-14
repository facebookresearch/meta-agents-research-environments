# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
"""Static registries for the judge.

Zero external dependencies.
"""

from __future__ import annotations

from enum import Enum

# ---------------------------------------------------------------------------
# Checker type enums
# ---------------------------------------------------------------------------


class CheckerType(str, Enum):
    """Hard-checker types for per-argument comparison."""

    eq_checker = "eq_checker"
    unordered_list_checker = "unordered_list_checker"
    list_attendees_checker = "list_attendees_checker"
    datetime_checker = "datetime_checker"
    phone_number_checker = "phone_number_checker"
    eq_str_strip_checker = "eq_str_strip_checker"
    llm_checker = "llm_checker"
    path_checker = "path_checker"
    unordered_path_list_checker = "unordered_path_list_checker"
    contain_any_checker = "contain_any_checker"
    contain_all_checker = "contain_all_checker"

    _HARD = {
        "eq_checker",
        "unordered_list_checker",
        "datetime_checker",
        "list_attendees_checker",
        "phone_number_checker",
        "eq_str_strip_checker",
        "path_checker",
        "unordered_path_list_checker",
    }
    _SCRIPTED = {"contain_any_checker", "contain_all_checker"}

    def is_hard(self) -> bool:
        return self.value in self._HARD  # type: ignore[attr-defined]

    def is_scripted(self) -> bool:
        return self.value in self._SCRIPTED  # type: ignore[attr-defined]


class SoftCheckerType(str, Enum):
    """Soft (LLM-based) checker types."""

    content_checker = "content_checker"
    sanity_checker = "sanity_checker"
    signature_checker = "signature_checker"
    placeholder_checker = "placeholder_checker"
    cab_checker = "cab_checker"
    event_checker = "event_checker"
    message_checker = "message_checker"
    email_checker = "email_checker"
    user_message_checker = "user_message_checker"
    tone_checker = "tone_checker"

    @property
    def need_subtask(self) -> bool:
        return self in {
            SoftCheckerType.content_checker,
            SoftCheckerType.user_message_checker,
            SoftCheckerType.event_checker,
            SoftCheckerType.sanity_checker,
        }


# ---------------------------------------------------------------------------
# Per-tool argument → checker type
# ---------------------------------------------------------------------------

PER_TOOL_ARG_TO_CHECKER_TYPE: dict[str, dict[str, CheckerType]] = {
    "ApartmentListingApp__save_apartment": {
        "apartment_id": CheckerType.eq_checker,
    },
    "ApartmentListingApp__remove_saved_apartment": {
        "apartment_id": CheckerType.eq_checker,
    },
    "CalendarApp__add_calendar_event": {
        "attendees": CheckerType.list_attendees_checker,
        "end_datetime": CheckerType.datetime_checker,
        "start_datetime": CheckerType.datetime_checker,
        "title": CheckerType.llm_checker,
        "description": CheckerType.llm_checker,
        "location": CheckerType.llm_checker,
    },
    "CalendarApp__delete_calendar_event": {
        "event_id": CheckerType.eq_checker,
    },
    "ShoppingApp__checkout": {
        "discount_code": CheckerType.eq_str_strip_checker,
    },
    "ShoppingApp__add_to_cart": {
        "item_id": CheckerType.eq_checker,
        "quantity": CheckerType.eq_checker,
    },
    "ShoppingApp__remove_from_cart": {
        "item_id": CheckerType.eq_checker,
        "quantity": CheckerType.eq_checker,
    },
    "ShoppingApp__cancel_order": {
        "order_id": CheckerType.eq_checker,
    },
    "ContactsApp__delete_contact": {
        "contact_id": CheckerType.eq_checker,
    },
    "ContactsApp__add_new_contact": {
        "first_name": CheckerType.eq_str_strip_checker,
        "last_name": CheckerType.eq_str_strip_checker,
        "email": CheckerType.eq_str_strip_checker,
        "phone": CheckerType.phone_number_checker,
    },
    "ContactsApp__edit_contact": {
        "contact_id": CheckerType.eq_checker,
        "updates": CheckerType.llm_checker,
    },
    "EmailClientApp__reply_to_email": {
        "email_id": CheckerType.eq_checker,
        "content": CheckerType.llm_checker,
        "attachment_paths": CheckerType.unordered_path_list_checker,
    },
    "EmailClientApp__delete_email": {
        "email_id": CheckerType.eq_checker,
    },
    "EmailClientApp__send_email": {
        "recipients": CheckerType.unordered_list_checker,
        "cc": CheckerType.unordered_list_checker,
        "attachment_paths": CheckerType.unordered_path_list_checker,
        "subject": CheckerType.llm_checker,
        "content": CheckerType.llm_checker,
    },
    "EmailClientApp__move_email": {
        "email_id": CheckerType.eq_checker,
        "source_folder_name": CheckerType.path_checker,
        "dest_folder_name": CheckerType.path_checker,
    },
    "EmailClientApp__forward_email": {
        "email_id": CheckerType.eq_checker,
        "recipients": CheckerType.unordered_list_checker,
        "folder_name": CheckerType.eq_checker,
    },
    "EmailClientApp__download_attachments": {
        "email_id": CheckerType.eq_checker,
        "folder_name": CheckerType.eq_checker,
        "path_to_save": CheckerType.path_checker,
    },
    "CabApp__order_ride": {
        "start_location": CheckerType.llm_checker,
        "end_location": CheckerType.llm_checker,
        "service_type": CheckerType.eq_checker,
        "ride_time": CheckerType.datetime_checker,
    },
    "CabApp__user_cancel_ride": {},
    "MessagingApp__create_conversation": {
        "participants": CheckerType.unordered_list_checker,
    },
    "MessagingApp__add_participant_to_conversation": {
        "conversation_id": CheckerType.eq_checker,
        "participant": CheckerType.eq_checker,
    },
    "MessagingApp__remove_participant_from_conversation": {
        "conversation_id": CheckerType.eq_checker,
        "participant": CheckerType.eq_checker,
    },
    "MessagingApp__send_message": {
        "conversation_id": CheckerType.eq_checker,
        "content": CheckerType.llm_checker,
    },
    "MessagingAppV2__send_message": {
        "user_id": CheckerType.eq_checker,
        "content": CheckerType.llm_checker,
        "attachment_path": CheckerType.path_checker,
    },
    "MessagingAppV2__send_message_to_group_conversation": {
        "conversation_id": CheckerType.eq_checker,
        "content": CheckerType.llm_checker,
        "attachment_path": CheckerType.path_checker,
    },
    "MessagingAppV2__create_group_conversation": {
        "user_ids": CheckerType.unordered_list_checker,
        "title": CheckerType.llm_checker,
    },
    "MessagingAppV2__add_participant_to_conversation": {
        "conversation_id": CheckerType.eq_checker,
        "user_id": CheckerType.eq_checker,
    },
    "MessagingAppV2__remove_participant_from_conversation": {
        "conversation_id": CheckerType.eq_checker,
        "user_id": CheckerType.eq_checker,
    },
    "MessagingAppV2__change_conversation_title": {
        "conversation_id": CheckerType.eq_checker,
        "title": CheckerType.llm_checker,
    },
    "SandboxLocalFileSystem__mv": {
        "path1": CheckerType.path_checker,
        "path2": CheckerType.path_checker,
    },
    "SandboxLocalFileSystem__open": {
        "path": CheckerType.path_checker,
        "mode": CheckerType.eq_checker,
    },
    "SandboxLocalFileSystem__mkdir": {
        "path": CheckerType.path_checker,
    },
    "SandboxLocalFileSystem__rm": {
        "path": CheckerType.path_checker,
        "recursive": CheckerType.eq_checker,
    },
    "SandboxLocalFileSystem__rmdir": {
        "path": CheckerType.path_checker,
    },
    "AgentUserInterface__send_message_to_user": {
        "content": CheckerType.llm_checker,
    },
}


# ---------------------------------------------------------------------------
# APP_ALIAS — public dataset app aliases
# ---------------------------------------------------------------------------

APP_ALIAS: dict[str, list[str]] = {
    "EmailClientApp": ["EmailClientV2", "Emails"],
    "ApartmentListingApp": ["RentAFlat"],
    "ContactsApp": ["Contacts"],
    "MessagingAppV2": ["Chats", "Messages"],
    "CalendarApp": ["Calendar"],
    "ShoppingApp": ["Shopping"],
    "CabApp": ["Cabs"],
    "SandboxLocalFileSystem": ["Files"],
}

# Tool-level aliases (messaging v1 → v2 mappings)
TOOL_ALIAS: dict[str, list[str]] = {
    "MessagingApp__send_message": [
        "MessagingAppV2__send_message",
        "MessagingAppV2__send_message_to_group_conversation",
    ],
    "MessagingApp__create_conversation": [
        "MessagingAppV2__create_group_conversation",
        "MessagingAppV2__change_conversation_title",
    ],
}


# ---------------------------------------------------------------------------
# Per-tool → soft checker types
# ---------------------------------------------------------------------------

# RL soft checkers — strict, includes tone_checker and sanity_checker.
# Used for RL training reward signals.
PER_TOOL_TO_SOFT_CHECKER_TYPES_RL: dict[str, list[SoftCheckerType]] = {
    "CalendarApp__add_calendar_event": [SoftCheckerType.event_checker],
    "EmailClientApp__send_email": [
        SoftCheckerType.placeholder_checker,
        SoftCheckerType.signature_checker,
        SoftCheckerType.email_checker,
    ],
    "EmailClientApp__reply_to_email": [
        SoftCheckerType.placeholder_checker,
        SoftCheckerType.signature_checker,
        SoftCheckerType.email_checker,
    ],
    "MessagingApp__send_message": [
        SoftCheckerType.placeholder_checker,
        SoftCheckerType.message_checker,
    ],
    "CabApp__order_ride": [SoftCheckerType.cab_checker],
    "AgentUserInterface__send_message_to_user": [
        SoftCheckerType.user_message_checker,
    ],
    "MessagingApp__create_conversation": [
        SoftCheckerType.content_checker,
    ],
}

# Eval soft checkers — lenient, no tone_checker or sanity_checker.
# Matches gaia2.validation.constants.PER_TOOL_TO_SOFT_CHECKER_TYPES_EVAL.
PER_TOOL_TO_SOFT_CHECKER_TYPES_EVAL: dict[str, list[SoftCheckerType]] = {
    "CalendarApp__add_calendar_event": [SoftCheckerType.event_checker],
    "EmailClientApp__send_email": [
        SoftCheckerType.placeholder_checker,
        SoftCheckerType.signature_checker,
        SoftCheckerType.email_checker,
    ],
    "EmailClientApp__reply_to_email": [
        SoftCheckerType.placeholder_checker,
        SoftCheckerType.signature_checker,
        SoftCheckerType.email_checker,
    ],
    "MessagingApp__send_message": [
        SoftCheckerType.placeholder_checker,
        SoftCheckerType.message_checker,
    ],
    "CabApp__order_ride": [SoftCheckerType.cab_checker],
    "AgentUserInterface__send_message_to_user": [
        SoftCheckerType.user_message_checker,
    ],
    "MessagingApp__create_conversation": [
        SoftCheckerType.content_checker,
    ],
}

# Default alias for backwards compat
PER_TOOL_TO_SOFT_CHECKER_TYPES = PER_TOOL_TO_SOFT_CHECKER_TYPES_RL


# ---------------------------------------------------------------------------
# Registry expansion
# ---------------------------------------------------------------------------


def _expand_app_aliases(
    registry: dict[str, object],
    app_alias: dict[str, list[str]],
) -> dict[str, object]:
    """Expand a registry with app aliases."""
    expanded = dict(registry)
    for canonical, aliases in app_alias.items():
        for key, val in list(expanded.items()):
            if key.split("__")[0] == canonical:
                for alias in aliases:
                    alias_key = f"{alias}__{key.split('__', 1)[1]}"
                    expanded[alias_key] = val
    return expanded


def _expand_tool_aliases(
    registry: dict[str, object],
    tool_alias: dict[str, list[str]],
) -> dict[str, object]:
    """Expand a registry with tool-level aliases."""
    expanded = dict(registry)
    for original, aliases in tool_alias.items():
        if original in expanded:
            for alias in aliases:
                if alias not in expanded:
                    expanded[alias] = expanded[original]
    return expanded


def build_checker_registries(
    mode: str = "eval",
) -> tuple[dict[str, dict[str, CheckerType]], dict[str, list[SoftCheckerType]]]:
    """Build expanded arg-checker and soft-checker registries.

    Args:
        mode: ``"eval"`` for lenient soft checkers (no tone/sanity),
              ``"rl"`` for strict RL training checkers.

    Returns:
        (arg_checker_registry, soft_checker_registry)
    """
    # Arg checkers: tool aliases first, then app aliases
    arg_reg = _expand_tool_aliases(PER_TOOL_ARG_TO_CHECKER_TYPE, TOOL_ALIAS)
    arg_reg = _expand_app_aliases(arg_reg, APP_ALIAS)

    # Soft checkers: pick RL or eval variant
    soft_base = (
        PER_TOOL_TO_SOFT_CHECKER_TYPES_RL
        if mode == "rl"
        else PER_TOOL_TO_SOFT_CHECKER_TYPES_EVAL
    )
    soft_reg = _expand_tool_aliases(soft_base, TOOL_ALIAS)
    soft_reg = _expand_app_aliases(soft_reg, APP_ALIAS)

    return arg_reg, soft_reg  # type: ignore[return-value]
