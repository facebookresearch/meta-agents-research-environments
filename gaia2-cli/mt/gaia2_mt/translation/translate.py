# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Extraction and translation steps for the GAIA2 translation pipeline."""

from __future__ import annotations

import json
import logging
from collections import Counter

import datasets
from tqdm.auto import tqdm

from gaia2_mt.data import (
    SKIP_FUNCTIONS,
    TRANSLATABLE_ARG_NAMES,
    AppStateField,
    OracleEventArg,
    extract_expected_response,
    extract_initial_prompt,
    extract_oracle_events,
    get_language_display_name,
    get_special_instructions,
)
from gaia2_mt.llm import OpenAICompatInferencer, parse_json_response
from gaia2_mt.prompts import (
    ORACLE_ARG_TRANSLATION_PROMPT,
    ORACLE_ARG_TRANSLATION_SYSTEM_PROMPT,
    TRANSLATION_PROMPT,
    TRANSLATION_SYSTEM_PROMPT,
)


logger = logging.getLogger(__name__)


def format_glossary_section(glossary: dict[str, str] | None) -> str:
    """Format glossary as a prompt section, or return empty string if no glossary."""
    if not glossary:
        return ""
    lines = [f'- "{orig}" → "{trans}"' for orig, trans in glossary.items()]
    return (
        "\n### Terminology (use these exact translations for the following terms):\n"
        + "\n".join(lines)
        + "\n"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Extraction
# ─────────────────────────────────────────────────────────────────────────────


def extract_prompts(ds: datasets.Dataset) -> list[str | None]:
    """Extract initial user prompts."""
    prompts = []
    for item in tqdm(ds, desc="Extracting prompts"):
        prompts.append(extract_initial_prompt(item["data"]))
    logger.info(
        f"Extracted {sum(p is not None for p in prompts)}/{len(prompts)} prompts"
    )
    return prompts


def extract_responses(ds: datasets.Dataset) -> list[str]:
    """Extract expected responses."""
    responses = []
    for item in ds:
        responses.append(extract_expected_response(item["data"]) or "")
    logger.info(
        f"Extracted {sum(1 for r in responses if r)}/{len(responses)} expected responses"
    )
    return responses


def build_heuristic_translation_plan(
    ds: datasets.Dataset,
) -> list[OracleEventArg]:
    """Build translation plan using heuristic allowlist instead of LLM classification.

    Identifies translatable oracle event args by checking arg names against
    TRANSLATABLE_ARG_NAMES. Skips send_message_to_agent events (user prompts
    are handled by their own dedicated pipeline). All other events — including
    send_message_to_user oracle responses — are included.
    """
    plan: list[OracleEventArg] = []

    for i in range(len(ds)):
        oracle_events = extract_oracle_events(ds[i]["data"])

        for ev in oracle_events:
            if ev["function"] in SKIP_FUNCTIONS:
                continue

            for arg_name, arg_value in ev["args"].items():
                if arg_name not in TRANSLATABLE_ARG_NAMES:
                    continue
                if not isinstance(arg_value, str) or not arg_value.strip():
                    continue

                plan.append(
                    OracleEventArg(
                        scenario_idx=i,
                        event_idx=ev["event_idx"],
                        app=ev["app"],
                        function=ev["function"],
                        arg_name=arg_name,
                        arg_value=arg_value,
                    )
                )

    n_scenarios = len({a.scenario_idx for a in plan})
    logger.info(
        f"Heuristic plan: {len(plan)} args to translate across {n_scenarios} scenarios"
    )
    arg_name_dist = Counter(a.arg_name for a in plan)
    logger.info(f"  By arg name: {dict(arg_name_dist)}")

    return plan


# ─────────────────────────────────────────────────────────────────────────────
# Translation
# ─────────────────────────────────────────────────────────────────────────────


def translate_prompts(
    prompts: list[str | None],
    src_lang: str,
    tgt_lang: str,
    model_name: str,
    glossary: dict[str, str] | None = None,
    glossary_sections: list[str] | None = None,
) -> list[str | None]:
    """Translate prompts using LLM-based translation.

    Returns a list with one entry per prompt.  An entry is ``None`` when the
    LLM call was rejected (e.g. by a content filter) and the prompt could not
    be translated.

    When ``glossary_sections`` is provided, it OVERRIDES the batch-shared
    ``glossary`` for that input — used by the term-table contract to inject
    per-scenario term tables. ``special`` language instructions are still
    prepended to each per-input section.
    """
    src_display = get_language_display_name(src_lang)
    tgt_display = get_language_display_name(tgt_lang)
    special = get_special_instructions(tgt_lang)

    translator = OpenAICompatInferencer(
        system_prompt=TRANSLATION_SYSTEM_PROMPT,
        prompt_template=TRANSLATION_PROMPT,
        model_name=model_name,
    )

    shared_glossary_section = format_glossary_section(glossary)
    if special:
        shared_glossary_section = special + shared_glossary_section

    if glossary_sections is not None and len(glossary_sections) != len(prompts):
        raise ValueError(
            f"glossary_sections length ({len(glossary_sections)}) does not match "
            f"prompts length ({len(prompts)})"
        )

    def _section_for(i: int) -> str:
        if glossary_sections is None:
            return shared_glossary_section
        per_input = glossary_sections[i] or ""
        return (special + per_input) if special else per_input

    inputs = [
        {
            "src_lang": src_display,
            "tgt_lang": tgt_display,
            "src_text": p if p is not None else "",
            "glossary_section": _section_for(i),
        }
        for i, p in enumerate(prompts)
    ]

    translated = translator.infer_batch(inputs)

    n_failed = sum(1 for t in translated if t is None)
    if n_failed:
        logger.warning(
            f"Prompt translation: {n_failed}/{len(translated)} returned None "
            f"(content-filtered)"
        )
    logger.info(f"Translated {len(translated) - n_failed}/{len(translated)} prompts")
    return translated


def translate_oracle_args(
    translation_plan: list[OracleEventArg],
    original_prompts: list[str | None],
    src_lang: str,
    tgt_lang: str,
    model_name: str,
    glossary: dict[str, str] | None = None,
    glossary_sections_by_scenario: dict[int, str] | None = None,
) -> dict[tuple, str]:
    """Translate oracle args grouped per scenario (one LLM call per scenario).

    Instead of one call per arg, all translatable args for a scenario are sent
    together so the LLM can translate them coherently in context.

    Returns mapping (scenario_idx, event_idx, arg_name) → translated_value.

    When ``glossary_sections_by_scenario`` is provided, each scenario's
    glossary_section is replaced with the per-scenario entry — used by the
    term-table contract to inject per-scenario term tables. ``special``
    instructions are still prepended.
    """
    if not translation_plan:
        logger.info("No oracle args to translate")
        return {}

    src_display = get_language_display_name(src_lang)
    tgt_display = get_language_display_name(tgt_lang)
    special = get_special_instructions(tgt_lang)

    args_by_scenario: dict[int, list[OracleEventArg]] = {}
    for arg in translation_plan:
        args_by_scenario.setdefault(arg.scenario_idx, []).append(arg)

    n_scenarios = len(args_by_scenario)
    logger.info(
        f"Translating {len(translation_plan)} oracle event args "
        f"across {n_scenarios} scenarios ({n_scenarios} LLM calls)"
    )

    translator = OpenAICompatInferencer(
        system_prompt=ORACLE_ARG_TRANSLATION_SYSTEM_PROMPT,
        prompt_template=ORACLE_ARG_TRANSLATION_PROMPT,
        model_name=model_name,
    )

    glossary_section = format_glossary_section(glossary)
    if special:
        glossary_section = special + glossary_section

    def _section_for_scenario(idx: int) -> str:
        if glossary_sections_by_scenario is None:
            return glossary_section
        per = glossary_sections_by_scenario.get(idx, "") or ""
        return (special + per) if special else per

    scenario_indices = sorted(args_by_scenario.keys())
    inputs = []
    for scenario_idx in scenario_indices:
        args = args_by_scenario[scenario_idx]
        args_dict = {}
        for j, arg in enumerate(args):
            args_dict[str(j)] = {
                "context": f"{arg.app}.{arg.function}({arg.arg_name}=...)",
                "text": arg.arg_value,
            }

        args_formatted = json.dumps(args_dict, ensure_ascii=False, indent=2)
        inputs.append(
            {
                "src_lang": src_display,
                "tgt_lang": tgt_display,
                "user_prompt": original_prompts[scenario_idx] or "",
                "glossary_section": _section_for_scenario(scenario_idx),
                "args_to_translate": args_formatted,
            }
        )

    raw_responses = translator.infer_batch(inputs)

    result = {}
    for i, scenario_idx in enumerate(scenario_indices):
        parsed = parse_json_response(raw_responses[i])
        args = args_by_scenario[scenario_idx]

        if parsed is None:
            logger.warning(
                f"Failed to parse translation response for scenario {scenario_idx}, "
                f"keeping original values for {len(args)} args"
            )
            for arg in args:
                key = (arg.scenario_idx, arg.event_idx, arg.arg_name)
                result[key] = arg.arg_value
            continue

        for j, arg in enumerate(args):
            translated_value = parsed.get(str(j))
            if translated_value is None:
                logger.warning(
                    f"Missing key '{j}' in translation response for "
                    f"scenario {scenario_idx}, keeping original"
                )
                translated_value = arg.arg_value
            key = (arg.scenario_idx, arg.event_idx, arg.arg_name)
            result[key] = translated_value

    logger.info(f"Translated {len(result)} oracle event args")
    return result


APP_STATE_BATCH_SIZES: dict[str, int] = {
    "Emails": 5,
    "EmailClientV2": 5,
    "Shopping": 10,
    "RentAFlat": 10,
}
APP_STATE_DEFAULT_BATCH_SIZE = 20


def _build_app_state_batches(
    fields: list[AppStateField],
) -> list[list[AppStateField]]:
    """Split fields into per-app chunks using app-specific batch sizes."""
    by_app: dict[str, list[AppStateField]] = {}
    for f in fields:
        by_app.setdefault(f.app_name, []).append(f)

    batches: list[list[AppStateField]] = []
    for app_name, app_fields in by_app.items():
        batch_size = APP_STATE_BATCH_SIZES.get(app_name, APP_STATE_DEFAULT_BATCH_SIZE)
        for start in range(0, len(app_fields), batch_size):
            batches.append(app_fields[start : start + batch_size])
    return batches


def translate_app_state(
    universe_fields: dict[str, list[AppStateField]],
    src_lang: str,
    tgt_lang: str,
    model_name: str,
) -> dict[str, dict[tuple, str]]:
    """Translate app state fields for each unique universe with sub-batching.

    Because GAIA2 scenarios share app states through "universes", callers
    group fields by universe hash and this function translates each universe
    only once.  Each universe may have thousands of fields, so they are
    sub-batched per app using app-specific batch sizes (smaller for apps
    with long text like Emails).

    Args:
        universe_fields: ``{universe_hash: fields}`` where fields come from
            :func:`extract_translatable_app_fields` (``scenario_idx`` is
            ignored — only ``app_idx``, ``app_name``, ``field_path``,
            ``field_value`` matter).

    Returns:
        ``{universe_hash: {(app_idx, *field_path): translated_value}}``.
    """
    if not universe_fields:
        logger.info("No app state fields to translate")
        return {}

    src_display = get_language_display_name(src_lang)
    tgt_display = get_language_display_name(tgt_lang)
    special = get_special_instructions(tgt_lang)

    total_fields = sum(len(fs) for fs in universe_fields.values())
    logger.info(
        f"Translating {total_fields} app state fields "
        f"across {len(universe_fields)} unique universes"
    )

    translator = OpenAICompatInferencer(
        system_prompt=ORACLE_ARG_TRANSLATION_SYSTEM_PROMPT,
        prompt_template=ORACLE_ARG_TRANSLATION_PROMPT,
        model_name=model_name,
    )

    batch_meta: list[tuple[str, list[AppStateField]]] = []
    inputs: list[dict] = []

    for u_hash, fields in universe_fields.items():
        batches = _build_app_state_batches(fields)
        for batch in batches:
            args_dict = {}
            for j, f in enumerate(batch):
                path_str = ".".join(str(p) for p in f.field_path)
                args_dict[str(j)] = {
                    "context": f"{f.app_name}.app_state.{path_str}",
                    "text": f.field_value,
                }

            args_formatted = json.dumps(args_dict, ensure_ascii=False, indent=2)
            inputs.append(
                {
                    "src_lang": src_display,
                    "tgt_lang": tgt_display,
                    "user_prompt": "",
                    "glossary_section": special,
                    "args_to_translate": args_formatted,
                }
            )
            batch_meta.append((u_hash, batch))

    app_sizes = ", ".join(
        f"{app}={size}" for app, size in sorted(APP_STATE_BATCH_SIZES.items())
    )
    logger.info(
        f"  Sub-batched into {len(inputs)} LLM calls "
        f"(per-app sizes: {app_sizes}, default={APP_STATE_DEFAULT_BATCH_SIZE})"
    )

    raw_responses = translator.infer_batch(inputs)

    result: dict[str, dict[tuple, str]] = {h: {} for h in universe_fields}
    n_parsed = 0
    n_failed = 0

    for i, (u_hash, batch) in enumerate(batch_meta):
        parsed = parse_json_response(raw_responses[i])

        if parsed is None:
            n_failed += len(batch)
            app_name = batch[0].app_name if batch else "?"
            logger.warning(
                f"Failed to parse app state batch for universe {u_hash[:12]} "
                f"({app_name}, {len(batch)} fields), keeping originals"
            )
            for f in batch:
                key = (f.app_idx, *f.field_path)
                result[u_hash][key] = f.field_value
            continue

        for j, f in enumerate(batch):
            translated_value = parsed.get(str(j))
            if translated_value is None:
                translated_value = f.field_value
            key = (f.app_idx, *f.field_path)
            result[u_hash][key] = translated_value
            n_parsed += 1

    logger.info(
        f"Translated {n_parsed} app state fields, "
        f"{n_failed} kept original (parse failures)"
    )
    return result
