# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
"""Core data structures for the Gaia2 scenario engine.

Pydantic BaseModel types shared between gaia2-cli (container judge)
and the Gaia2 framework (training/eval judge), plus placeholder
resolution helpers.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------


class EventAction(BaseModel):
    """Describes a single app method call."""

    app_name: str  # scenario instance name ("Chats", "Calendar")
    class_name: str  # Python class name ("MessagingAppV2", "CalendarApp")
    function_name: str  # method name
    args: dict[str, Any]  # parsed flat dict, placeholders unresolved
    operation_type: str  # "read" or "write"


class ScenarioEvent(BaseModel):
    """A single event in the scenario DAG (before processing)."""

    event_id: str
    event_type: str  # AGENT, ENV, USER, ORACLE
    event_time: float | None
    event_relative_time: float | None
    action: EventAction | None
    dependency_ids: list[str]
    successor_ids: list[str] = Field(default_factory=list)  # rebuilt


class CompletedEvent(BaseModel):
    """An event that has been executed (agent action or ENV action)."""

    event_id: str
    event_type: str
    event_time: float
    action: EventAction | None
    return_value: Any = None

    @property
    def tool_name(self) -> str:
        """``ClassName__function_name`` — matches Gaia2 CompletedEvent.tool_name."""
        if self.action is None:
            return "NoApp__NoFunction"
        return f"{self.action.class_name}__{self.action.function_name}"

    def get_args(self) -> dict[str, Any]:
        """Return action args (resolved if available, else raw)."""
        if self.action is None:
            return {}
        return dict(self.action.args)


class OracleEvent(BaseModel):
    """A single oracle event extracted from the scenario for judging."""

    model_config = {"validate_assignment": False}

    event_id: str
    event_type: str  # AGENT, ENV, USER
    action: EventAction | None
    args: dict[str, Any] = Field(default_factory=dict)
    resolved_args: dict[str, Any] | None = None
    return_value: Any = None
    event_time: float | None = None
    event_relative_time: float | None = None
    dependency_ids: list[str] = Field(default_factory=list)

    @property
    def tool_name(self) -> str:
        if self.action is None:
            return "NoApp__NoFunction"
        return f"{self.action.class_name}__{self.action.function_name}"

    def get_args(self) -> dict[str, Any]:
        """Return resolved_args if present, else args."""
        if self.resolved_args is not None:
            return dict(self.resolved_args)
        return dict(self.args)


class UserDetails(BaseModel):
    """User contact details used by the judge (tolerances, signatures)."""

    first_name: str = ""
    last_name: str = ""
    address: str = ""


class JudgmentResult(BaseModel):
    """Result of judging a single turn."""

    success: bool
    agent_event_id_to_oracle_event_id: dict[str, str] = Field(default_factory=dict)
    failure_reason: str = ""


class ConditionCheck(BaseModel):
    """A turn-boundary condition check inserted by build_turn_triggers().

    Replaces Gaia2 ConditionCheckEvent.  Fires when the number of
    send_message_to_user events in the completed log equals ``turn_idx``.
    """

    event_id: str
    turn_idx: int
    event_time: float
    schedule_every_secs: float
    successor_ids: list[str] = Field(default_factory=list)
    dependency_ids: list[str] = Field(default_factory=list)
    check_count: int = 0


# ---------------------------------------------------------------------------
# Placeholder resolution (mirrors gaia2 environment.resolve_arg_placeholders)
# ---------------------------------------------------------------------------

_PLACEHOLDER_RE = re.compile(r"^\{\{(.*?)\}\}$")


def resolve_placeholders(
    args: dict[str, Any],
    completed: dict[str, CompletedEvent],
) -> dict[str, Any]:
    """Resolve ``{{event_id}}`` and ``{{event_id.key}}`` placeholders.

    Uses *completed* events' return_value for resolution.
    """
    resolved = dict(args)
    for arg_name, arg_value in list(resolved.items()):
        if not isinstance(arg_value, str):
            continue
        m = _PLACEHOLDER_RE.match(arg_value.strip())
        if not m:
            continue

        parts = m.group(1).split(".")
        ref_event_id = parts[0]

        completed_event = completed.get(ref_event_id)
        if completed_event is None:
            logger.debug("Unresolved placeholder %s=%s", arg_name, arg_value)
            continue

        value = completed_event.return_value
        for key in parts[1:]:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                logger.debug(
                    "Failed to resolve key '%s' in return value of %s",
                    key,
                    ref_event_id,
                )
                value = arg_value  # keep original
                break

        resolved[arg_name] = value
    return resolved
