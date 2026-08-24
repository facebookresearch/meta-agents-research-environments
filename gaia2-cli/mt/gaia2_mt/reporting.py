# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Results & auditability: build summary DataFrames from pipeline outputs."""

from __future__ import annotations

from typing import TYPE_CHECKING

import datasets
import pandas as pd

from gaia2_mt.data import OracleEventArg


if TYPE_CHECKING:
    from gaia2_mt.lid import LidReport


def build_results_dataframe(
    ds: datasets.Dataset,
    original_prompts: list[str | None],
    translated_prompts: list[str],
    expected_responses: list[str],
    translation_reviews: list[dict | None],
    translation_plan: list[OracleEventArg] | None = None,
    oracle_arg_map: dict[tuple, str] | None = None,
    app_state_map: dict[tuple, str] | None = None,
    lid_report: LidReport | None = None,
) -> pd.DataFrame:
    """Build a unified summary DataFrame with all pipeline results."""
    plan_by_scenario: dict[int, list[int]] = {}
    if translation_plan:
        for plan_idx, arg in enumerate(translation_plan):
            plan_by_scenario.setdefault(arg.scenario_idx, []).append(plan_idx)

    app_state_by_scenario: dict[int, int] = {}
    if app_state_map:
        for key in app_state_map:
            scenario_idx = key[0]
            app_state_by_scenario[scenario_idx] = (
                app_state_by_scenario.get(scenario_idx, 0) + 1
            )

    # Distinguish "the review pass did not run" from "it ran and its output
    # failed to parse". Under a translator-only run every entry is None, which
    # must not be reported as a parse error.
    reviews_ran = any(r is not None for r in translation_reviews)

    rows = []
    for i in range(len(ds)):
        row = {
            "scenario_id": ds[i]["scenario_id"],
            "original_prompt": (original_prompts[i] or "")[:200],
            "translated_prompt": (translated_prompts[i] or "")[:200],
            "expected_response": expected_responses[i][:200],
        }

        tr = translation_reviews[i] if i < len(translation_reviews) else None
        if tr:
            row["translation_reasoning"] = (tr.get("reasoning") or "")[:300]
            row["translation_quality"] = tr.get("quality")
            row["preserves_meaning"] = tr.get("preserves_meaning")
            row["is_fluent"] = tr.get("is_fluent")
            row["translation_issues"] = "; ".join(tr.get("issues", []))
            row["translation_suggestion"] = (tr.get("suggestion") or "")[:200]
        else:
            row["translation_quality"] = "PARSE_ERROR" if reviews_ran else None

        scenario_plan_indices = plan_by_scenario.get(i, [])
        row["n_oracle_args_translated"] = len(scenario_plan_indices)
        row["n_app_state_fields_translated"] = app_state_by_scenario.get(i, 0)

        if scenario_plan_indices and translation_plan:
            arg_details = []
            for pi in scenario_plan_indices:
                arg = translation_plan[pi]
                detail = f"{arg.app}.{arg.function}.{arg.arg_name}"
                arg_details.append(detail)
            row["oracle_arg_details"] = "; ".join(arg_details)
        else:
            row["oracle_arg_details"] = None

        # LID columns (per-prompt)
        if lid_report and i < len(lid_report.prompt_results):
            lr = lid_report.prompt_results[i]
            if lr is not None:
                row["lid_prompt_lang"] = lr.detected_lang
                row["lid_prompt_confidence"] = lr.confidence
                row["lid_prompt_correct"] = lr.is_correct
            else:
                row["lid_prompt_lang"] = None
                row["lid_prompt_confidence"] = None
                row["lid_prompt_correct"] = None

        rows.append(row)

    return pd.DataFrame(rows)
