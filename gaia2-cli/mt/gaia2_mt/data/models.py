# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Data models for the GAIA2 translation pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import datasets


if TYPE_CHECKING:
    from gaia2_mt.lid import LidReport


@dataclass
class OracleEventArg:
    """One translatable arg from an oracle event."""

    scenario_idx: int
    event_idx: int
    app: str
    function: str
    arg_name: str
    arg_value: str


@dataclass
class ScenarioTranslation:
    """Per-scenario tracking through the translation pipeline.

    Created by ``process_split()`` after the prompt-translation step.
    If ``failed`` is True after the review/post-edit step, downstream
    steps (oracle-arg translation, app-state translation) are skipped
    and the scenario's original data is preserved unchanged in the final
    dataset.  A scenario initially marked as failed may be recovered by
    the review/post-edit agent, in which case ``failed`` is set back to
    ``False`` and ``failure_reason`` is updated to indicate the recovery.
    """

    scenario_idx: int
    original_prompt: str | None
    translated_prompt: str | None = None
    prompt_review: dict | None = None
    failed: bool = False
    failure_reason: str | None = None


@dataclass
class SplitResult:
    """Holds the translated dataset and review metadata for a single split."""

    dataset: datasets.Dataset
    original_prompts: list[str | None] = field(default_factory=list)
    translated_prompts: list[str] = field(default_factory=list)
    expected_responses: list[str] = field(default_factory=list)
    translation_reviews: list[dict | None] = field(default_factory=list)
    oracle_arg_translations: dict = field(default_factory=dict)
    app_state_translations: dict = field(default_factory=dict)
    translation_plan: list[OracleEventArg] = field(default_factory=list)
    scenario_translations: list[ScenarioTranslation] = field(default_factory=list)
    lid_report: LidReport | None = None
