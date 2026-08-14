# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Review and post-editing steps for the GAIA2 translation pipeline."""

from __future__ import annotations

import json
import logging
from collections import Counter

from gaia2_mt.data import AppStateField, OracleEventArg, get_language_display_name
from gaia2_mt.llm import OpenAICompatInferencer, parse_json_response
from gaia2_mt.prompts import (
    ORACLE_ARG_REVIEW_PROMPT,
    ORACLE_ARG_REVIEW_SYSTEM_PROMPT,
    POST_EDIT_PROMPT,
    POST_EDIT_SYSTEM_PROMPT,
    TRANSLATION_REVIEW_PROMPT,
    TRANSLATION_REVIEW_SYSTEM_PROMPT,
)


logger = logging.getLogger(__name__)


def review_and_post_edit(
    originals: list[str],
    translations: list[str],
    src_lang: str,
    tgt_lang: str,
    review_model: str,
    desc: str = "items",
    failure_flags: list[bool] | None = None,
) -> tuple[list[str], list[dict | None]]:
    """Review and post-edit a list of translations.

    Returns a tuple of (final_translations, reviews).

    Generic utility used for prompts, oracle args, and responses alike.
    Pipeline: review quality → post-edit poor quality → return final list.

    When *failure_flags* is provided, items where ``failure_flags[i]`` is
    ``True`` are treated as **failed translations**: the review agent is
    informed that the upstream translation step failed and the text shown
    is the untranslated original, and these items are always routed to
    post-editing regardless of the review quality score.
    """
    if not translations:
        return translations, []

    if failure_flags is None:
        failure_flags = [False] * len(translations)

    src_display = get_language_display_name(src_lang)
    tgt_display = get_language_display_name(tgt_lang)

    n_failed = sum(failure_flags)
    if n_failed:
        logger.info(
            f"Reviewing {len(translations)} {desc} translations "
            f"({n_failed} flagged as failed upstream — will attempt recovery)"
        )
    else:
        logger.info(f"Reviewing {len(translations)} {desc} translations")

    reviewer = OpenAICompatInferencer(
        system_prompt=TRANSLATION_REVIEW_SYSTEM_PROMPT,
        prompt_template=TRANSLATION_REVIEW_PROMPT,
        model_name=review_model,
    )

    _FAILURE_NOTICE = (
        "[TRANSLATION FAILED] The upstream translation agent failed to produce "
        "a translation for this text (e.g. content was filtered by the provider). "
        "The text shown below is the UNTRANSLATED ORIGINAL. Please treat this as "
        "a missing translation and provide a full translation in your suggestion."
    )

    review_inputs = []
    for i, (orig, trans) in enumerate(zip(originals, translations)):
        translated_text = trans
        if failure_flags[i]:
            translated_text = f"{_FAILURE_NOTICE}\n\n{trans}"
        review_inputs.append(
            {
                "src_lang": src_display,
                "tgt_lang": tgt_display,
                "src_text": orig,
                "translated_text": translated_text,
            }
        )

    raw_reviews = reviewer.infer_batch(review_inputs)
    reviews = [parse_json_response(r) for r in raw_reviews]

    parsed = sum(1 for r in reviews if r is not None)
    quality_dist = Counter(r.get("quality") for r in reviews if r)
    logger.info(
        f"  {desc} review: parsed {parsed}/{len(reviews)}, quality={dict(quality_dist)}"
    )

    # Always route failed items to post-editing, even if the review parsed
    # as "good" (reviewer may not have understood the failure notice).
    post_edit_indices = []
    for i, review in enumerate(reviews):
        if failure_flags[i]:
            post_edit_indices.append(i)
        elif review and review.get("quality") != "good" and review.get("issues"):
            post_edit_indices.append(i)

    if not post_edit_indices:
        logger.info(f"  No {desc} translations need post-editing")
        return translations, reviews

    n_recovery = sum(1 for i in post_edit_indices if failure_flags[i])
    n_quality = len(post_edit_indices) - n_recovery
    logger.info(
        f"  Post-editing {len(post_edit_indices)} {desc} translations "
        f"({n_recovery} recovery attempts, {n_quality} quality improvements)"
    )

    editor = OpenAICompatInferencer(
        system_prompt=POST_EDIT_SYSTEM_PROMPT,
        prompt_template=POST_EDIT_PROMPT,
        model_name=review_model,
    )

    _FAILURE_ISSUES = (
        "- The upstream translation agent FAILED to translate this text entirely. "
        "The text shown as 'Current translation' is actually the untranslated "
        "original. Please provide a complete, natural translation from scratch."
    )

    edit_inputs = []
    for i in post_edit_indices:
        if failure_flags[i]:
            issues_str = _FAILURE_ISSUES
        else:
            tr = reviews[i]
            issues_str = "\n".join(f"- {issue}" for issue in tr.get("issues", []))
        edit_inputs.append(
            {
                "src_lang": src_display,
                "tgt_lang": tgt_display,
                "src_text": originals[i],
                "translated_text": translations[i],
                "issues": issues_str,
            }
        )

    edited = editor.infer_batch(edit_inputs)

    final = list(translations)
    for j, i in enumerate(post_edit_indices):
        if edited[j] is not None:
            final[i] = edited[j]
            if failure_flags[i]:
                logger.info(
                    f"  Recovery succeeded for {desc} index {i}: "
                    f"post-editor produced a translation"
                )
        else:
            if failure_flags[i]:
                logger.warning(
                    f"  Recovery FAILED for {desc} index {i}: "
                    f"post-editor also returned None, scenario remains failed"
                )
            else:
                logger.warning(
                    f"Post-edit LLM call returned None for index {i}, "
                    f"keeping original translation"
                )

    return final, reviews


def review_and_post_edit_oracle_args(
    translation_plan: list[OracleEventArg],
    oracle_arg_map: dict[tuple, str],
    original_prompts: list[str | None],
    src_lang: str,
    tgt_lang: str,
    review_model: str,
) -> tuple[dict[tuple, str], list[dict | None]]:
    """Review and post-edit oracle arg translations grouped by scenario.

    Instead of reviewing each arg individually (losing all context), groups
    all args per scenario — mirroring translate_oracle_args — so the reviewer
    sees the full picture: user prompt, tool call context for every arg, and
    all sibling args together.  This lets the reviewer catch cross-arg
    inconsistencies and assess faithfulness against the original task.

    Returns (updated_oracle_arg_map, flat_reviews_in_plan_order).
    """
    if not translation_plan:
        return oracle_arg_map, []

    src_display = get_language_display_name(src_lang)
    tgt_display = get_language_display_name(tgt_lang)

    args_by_scenario: dict[int, list[OracleEventArg]] = {}
    for arg in translation_plan:
        args_by_scenario.setdefault(arg.scenario_idx, []).append(arg)

    n_scenarios = len(args_by_scenario)
    logger.info(
        f"Reviewing {len(translation_plan)} oracle arg translations "
        f"across {n_scenarios} scenarios ({n_scenarios} LLM calls)"
    )

    reviewer = OpenAICompatInferencer(
        system_prompt=ORACLE_ARG_REVIEW_SYSTEM_PROMPT,
        prompt_template=ORACLE_ARG_REVIEW_PROMPT,
        model_name=review_model,
    )

    scenario_indices = sorted(args_by_scenario.keys())
    inputs = []
    for scenario_idx in scenario_indices:
        args = args_by_scenario[scenario_idx]
        args_dict = {}
        for j, arg in enumerate(args):
            key = (arg.scenario_idx, arg.event_idx, arg.arg_name)
            args_dict[str(j)] = {
                "context": f"{arg.app}.{arg.function}({arg.arg_name}=...)",
                "original": arg.arg_value,
                "translation": oracle_arg_map.get(key, ""),
            }

        args_formatted = json.dumps(args_dict, ensure_ascii=False, indent=2)
        inputs.append(
            {
                "src_lang": src_display,
                "tgt_lang": tgt_display,
                "user_prompt": original_prompts[scenario_idx] or "",
                "args_to_review": args_formatted,
            }
        )

    raw_responses = reviewer.infer_batch(inputs)

    all_reviews: list[dict | None] = []
    post_edit_items: list[tuple[OracleEventArg, dict]] = []

    for i, scenario_idx in enumerate(scenario_indices):
        parsed = parse_json_response(raw_responses[i])
        args = args_by_scenario[scenario_idx]

        for j, arg in enumerate(args):
            review = parsed.get(str(j)) if parsed is not None else None
            all_reviews.append(review)
            if review and review.get("quality") != "good" and review.get("issues"):
                post_edit_items.append((arg, review))

    quality_dist = Counter(r.get("quality") for r in all_reviews if r is not None)
    parsed_count = sum(1 for r in all_reviews if r is not None)
    logger.info(
        f"  oracle arg review: parsed {parsed_count}/{len(all_reviews)}, "
        f"quality={dict(quality_dist)}"
    )

    if not post_edit_items:
        logger.info("  No oracle arg translations need post-editing")
        return oracle_arg_map, all_reviews

    logger.info(f"  Post-editing {len(post_edit_items)} oracle arg translations")

    editor = OpenAICompatInferencer(
        system_prompt=POST_EDIT_SYSTEM_PROMPT,
        prompt_template=POST_EDIT_PROMPT,
        model_name=review_model,
    )

    edit_inputs = []
    for arg, review in post_edit_items:
        key = (arg.scenario_idx, arg.event_idx, arg.arg_name)
        issues_str = "\n".join(f"- {issue}" for issue in review.get("issues", []))
        edit_inputs.append(
            {
                "src_lang": src_display,
                "tgt_lang": tgt_display,
                "src_text": arg.arg_value,
                "translated_text": oracle_arg_map.get(key, ""),
                "issues": issues_str,
            }
        )

    edited = editor.infer_batch(edit_inputs)

    updated_map = dict(oracle_arg_map)
    for idx, (arg, _) in enumerate(post_edit_items):
        key = (arg.scenario_idx, arg.event_idx, arg.arg_name)
        if edited[idx] is not None:
            updated_map[key] = edited[idx]
        else:
            logger.warning(
                f"Post-edit LLM call returned None for arg {key}, keeping original translation"
            )

    return updated_map, all_reviews


APP_STATE_REVIEW_BATCH_SIZES: dict[str, int] = {
    "Emails": 10,
    "EmailClientV2": 10,
    "Shopping": 25,
    "RentAFlat": 25,
}
APP_STATE_REVIEW_DEFAULT_BATCH_SIZE = 50


def _build_app_state_review_batches(
    fields: list[AppStateField],
) -> list[list[AppStateField]]:
    """Split fields into per-app chunks using app-specific batch sizes."""
    by_app: dict[str, list[AppStateField]] = {}
    for f in fields:
        by_app.setdefault(f.app_name, []).append(f)

    batches: list[list[AppStateField]] = []
    for app_name, app_fields in by_app.items():
        batch_size = APP_STATE_REVIEW_BATCH_SIZES.get(
            app_name, APP_STATE_REVIEW_DEFAULT_BATCH_SIZE
        )
        for start in range(0, len(app_fields), batch_size):
            batches.append(app_fields[start : start + batch_size])
    return batches


def review_and_post_edit_app_state(
    universe_fields: dict[str, list[AppStateField]],
    universe_translations: dict[str, dict[tuple, str]],
    src_lang: str,
    tgt_lang: str,
    review_model: str,
) -> dict[str, dict[tuple, str]]:
    """Review and post-edit app state translations for each unique universe.

    Mirrors :func:`translate_app_state`: operates on deduplicated universes
    and sub-batches large field lists using per-app batch sizes.

    Returns ``{universe_hash: {(app_idx, *field_path): final_value}}``.
    """
    if not universe_fields:
        return universe_translations

    src_display = get_language_display_name(src_lang)
    tgt_display = get_language_display_name(tgt_lang)

    total_fields = sum(len(fs) for fs in universe_fields.values())
    logger.info(
        f"Reviewing {total_fields} app state translations "
        f"across {len(universe_fields)} unique universes"
    )

    reviewer = OpenAICompatInferencer(
        system_prompt=ORACLE_ARG_REVIEW_SYSTEM_PROMPT,
        prompt_template=ORACLE_ARG_REVIEW_PROMPT,
        model_name=review_model,
    )

    batch_meta: list[tuple[str, list[AppStateField]]] = []
    inputs: list[dict] = []

    for u_hash, fields in universe_fields.items():
        trans = universe_translations.get(u_hash, {})
        batches = _build_app_state_review_batches(fields)
        for batch in batches:
            args_dict = {}
            for j, f in enumerate(batch):
                key = (f.app_idx, *f.field_path)
                path_str = ".".join(str(p) for p in f.field_path)
                args_dict[str(j)] = {
                    "context": f"{f.app_name}.app_state.{path_str}",
                    "original": f.field_value,
                    "translation": trans.get(key, ""),
                }

            args_formatted = json.dumps(args_dict, ensure_ascii=False, indent=2)
            inputs.append(
                {
                    "src_lang": src_display,
                    "tgt_lang": tgt_display,
                    "user_prompt": "",
                    "args_to_review": args_formatted,
                }
            )
            batch_meta.append((u_hash, batch))

    logger.info(
        f"  Sub-batched into {len(inputs)} review LLM calls "
        f"(per-app sizes, default={APP_STATE_REVIEW_DEFAULT_BATCH_SIZE})"
    )

    raw_responses = reviewer.infer_batch(inputs)

    post_edit_items: list[tuple[str, AppStateField, dict]] = []

    for i, (u_hash, batch) in enumerate(batch_meta):
        parsed = parse_json_response(raw_responses[i])

        for j, f in enumerate(batch):
            review = parsed.get(str(j)) if parsed is not None else None
            if review and review.get("quality") != "good" and review.get("issues"):
                post_edit_items.append((u_hash, f, review))

    if not post_edit_items:
        logger.info("  No app state translations need post-editing")
        return universe_translations

    logger.info(f"  Post-editing {len(post_edit_items)} app state translations")

    editor = OpenAICompatInferencer(
        system_prompt=POST_EDIT_SYSTEM_PROMPT,
        prompt_template=POST_EDIT_PROMPT,
        model_name=review_model,
    )

    edit_inputs = []
    for u_hash, f, review in post_edit_items:
        key = (f.app_idx, *f.field_path)
        trans = universe_translations.get(u_hash, {})
        issues_str = "\n".join(f"- {issue}" for issue in review.get("issues", []))
        edit_inputs.append(
            {
                "src_lang": src_display,
                "tgt_lang": tgt_display,
                "src_text": f.field_value,
                "translated_text": trans.get(key, ""),
                "issues": issues_str,
            }
        )

    edited = editor.infer_batch(edit_inputs)

    updated: dict[str, dict[tuple, str]] = {
        h: dict(t) for h, t in universe_translations.items()
    }
    for idx, (u_hash, f, _) in enumerate(post_edit_items):
        key = (f.app_idx, *f.field_path)
        if edited[idx] is not None:
            updated[u_hash][key] = edited[idx]
        else:
            logger.warning(
                f"Post-edit LLM call returned None for app state field {key}, "
                f"keeping original translation"
            )

    return updated
