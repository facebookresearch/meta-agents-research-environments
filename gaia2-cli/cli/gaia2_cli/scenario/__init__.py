# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
"""Scenario engine — data structures, JSON parser, event processor."""

from gaia2_core.event_loop import EventProcessor
from gaia2_core.loader import ScenarioLoader
from gaia2_core.types import (
    CompletedEvent,
    ConditionCheck,
    EventAction,
    JudgmentResult,
    OracleEvent,
    ScenarioEvent,
    UserDetails,
    resolve_placeholders,
)

__all__ = [
    "CompletedEvent",
    "ConditionCheck",
    "EventAction",
    "JudgmentResult",
    "OracleEvent",
    "ScenarioEvent",
    "UserDetails",
    "resolve_placeholders",
    "ScenarioLoader",
    "EventProcessor",
]
