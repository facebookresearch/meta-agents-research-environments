# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Pipeline orchestration: load_dataset, build_final_dataset, and process_split."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import datasets

from gaia2_mt.checkpoint import (
    CheckpointManager,
    deserialize_nested_tuple_key_dict,
    deserialize_tuple_key_dict,
    serialize_nested_tuple_key_dict,
    serialize_tuple_key_dict,
)
from gaia2_mt.data import (
    ScenarioTranslation,
    SplitResult,
    apply_app_state_translations,
    compute_universe_hash,
    extract_translatable_app_fields,
)
from gaia2_mt.translation.contract import (
    TermTable,
    apply_term_table,
    format_term_table_section,
)
from gaia2_mt.translation.contract.wiring import (
    build_scenario_term_tables,
    deserialize_term_tables,
    serialize_term_tables,
)
from gaia2_mt.translation.review import (
    review_and_post_edit,
    review_and_post_edit_app_state,
    review_and_post_edit_oracle_args,
)
from gaia2_mt.translation.translate import (
    build_heuristic_translation_plan,
    extract_prompts,
    extract_responses,
    translate_app_state,
    translate_oracle_args,
    translate_prompts,
)


logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset loading and assembly
# ─────────────────────────────────────────────────────────────────────────────


def load_dataset_from_directory(
    data_dir: str,
    subset: str,
) -> datasets.Dataset:
    """Load a GAIA2 subset from a local directory of JSON scenario files.

    Reads all ``.json`` files from ``{data_dir}/{subset}/`` and returns a
    :class:`datasets.Dataset` with a single ``"data"`` column where each row
    is the JSON string content of one scenario file.  This matches the format
    returned by ``datasets.load_dataset("meta-agents-research-environments/gaia2",
    subset)["train"]``.

    Handles nested directory structures produced by ``manifold getr``
    (e.g. ``3103_gaia2_verified/3103_gaia2_verified/subset/``).
    """
    base = Path(data_dir)
    subset_dir = base / subset

    # Handle nested directory (manifold getr often creates dir/dir/subset/)
    if not subset_dir.is_dir():
        nested = base / base.name / subset
        if nested.is_dir():
            subset_dir = nested
        else:
            raise FileNotFoundError(
                f"Subset directory not found: tried {base / subset} and {nested}"
            )

    json_files = sorted(subset_dir.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"No .json files found in {subset_dir}")

    rows = []
    for json_file in json_files:
        rows.append({"data": json_file.read_text(encoding="utf-8")})

    ds = datasets.Dataset.from_dict({"data": [r["data"] for r in rows]})
    logger.info(f"Loaded {len(ds)} scenarios from local directory: {subset_dir}")
    return ds


def load_dataset(
    dataset_id: str,
    subset: str,
    local_data_dir: str | None = None,
) -> datasets.DatasetDict:
    """Load the GAIA2 dataset as a DatasetDict, preserving split structure.

    When *local_data_dir* is provided, scenarios are loaded from a local
    directory instead of HuggingFace and wrapped in a ``DatasetDict`` with
    a single ``"train"`` split (matching the HuggingFace layout).
    """
    if local_data_dir:
        ds = load_dataset_from_directory(local_data_dir, subset)
        ds_dict = datasets.DatasetDict({"train": ds})
    else:
        ds_dict = datasets.load_dataset(dataset_id, subset)

    for split_name, split_ds in ds_dict.items():
        logger.info(
            f"Loaded split '{split_name}': {len(split_ds)} rows from '{subset}' subset"
        )
    return ds_dict


def _apply_prompt_translation(data: dict, prompt_value: str) -> None:
    """Replace the user prompt in a scenario's events (in-place)."""
    for event in data["events"]:
        if "send_message_to_agent" in event["action"]["action_id"]:
            for arg in event["action"]["args"]:
                if arg["name"] == "content":
                    arg["value"] = prompt_value
            break


def _apply_oracle_arg_translations(
    data: dict,
    scenario_args: list[tuple[int, str, str]],
) -> None:
    """Apply translated oracle args to a scenario's events (in-place).

    Handles ALL translatable oracle args including ``send_message_to_user``
    expected-response events, which are part of the oracle args translation
    plan (not a separate response pipeline).
    """
    for event_idx, arg_name, new_value in scenario_args:
        ev = data["events"][event_idx]
        for arg in ev["action"]["args"]:
            if arg["name"] == arg_name:
                arg["value"] = new_value
                break


def build_final_dataset(
    ds: datasets.Dataset,
    translated_prompts: list[str | None],
    oracle_arg_map: dict[tuple, str] | None = None,
    app_state_map: dict[tuple, str] | None = None,
    post_edited_prompt_map: dict[int, str] | None = None,
    failed_indices: set[int] | None = None,
    scenario_term_tables: list[TermTable | None] | None = None,
) -> datasets.Dataset:
    """Build the final dataset with all mutations applied in a single pass.

    For each scenario, parses the JSON once, applies all mutations, and
    serializes once:
      1. Replace the user prompt (translated_prompts — always applied)
      2. Replace oracle event args (from heuristic plan — includes tool-call
         args AND send_message_to_user response contents)
      3. Replace app state fields (calendar events, emails, messages, etc.)
      4. When ``scenario_term_tables`` is provided, apply the term-table
         validator sweep to the final prompt + oracle arg strings (substitutes
         any leaked source-language span with its canonical target rendering).

    Scenarios in *failed_indices* are left untouched (original JSON preserved).
    """
    oracle_arg_map = oracle_arg_map or {}
    app_state_map = app_state_map or {}
    post_edited_prompt_map = post_edited_prompt_map or {}
    failed_indices = failed_indices or set()
    scenario_term_tables = scenario_term_tables or []

    args_by_scenario: dict[int, list[tuple[int, str, str]]] = {}
    for (scenario_idx, event_idx, arg_name), value in oracle_arg_map.items():
        args_by_scenario.setdefault(scenario_idx, []).append(
            (event_idx, arg_name, value)
        )

    app_state_by_scenario: dict[int, dict[tuple, str]] = {}
    for key, value in app_state_map.items():
        scenario_idx = key[0]
        inner_key = key[1:]
        app_state_by_scenario.setdefault(scenario_idx, {})[inner_key] = value

    final_data = []
    total_validator_subs = 0
    for i in range(len(ds)):
        data = json.loads(ds[i]["data"])

        if i in failed_indices:
            final_data.append(json.dumps(data, ensure_ascii=False))
            continue

        prompt_value = post_edited_prompt_map.get(i, translated_prompts[i])

        # Term-table validator sweep: substitute any leaked source span with
        # its canonical target rendering BEFORE serializing back to JSON.
        tt = scenario_term_tables[i] if i < len(scenario_term_tables) else None
        if tt is not None and prompt_value:
            prompt_value, n = apply_term_table(prompt_value, tt)
            total_validator_subs += n

        _apply_prompt_translation(data, prompt_value)

        if i in args_by_scenario:
            # Apply term-table sweep to each translated arg value as well.
            if tt is not None:
                rewritten = []
                for event_idx, arg_name, new_value in args_by_scenario[i]:
                    new_value2, n = apply_term_table(new_value, tt)
                    total_validator_subs += n
                    rewritten.append((event_idx, arg_name, new_value2))
                _apply_oracle_arg_translations(data, rewritten)
            else:
                _apply_oracle_arg_translations(data, args_by_scenario[i])

        if i in app_state_by_scenario:
            apply_app_state_translations(data, app_state_by_scenario[i])

        final_data.append(json.dumps(data, ensure_ascii=False))

    n_post_edited = sum(1 for i in range(len(ds)) if i in post_edited_prompt_map)
    n_skipped = len(failed_indices)
    logger.info(
        f"Final dataset: {len(final_data)} rows, "
        f"{n_skipped} skipped (failed), "
        f"{n_post_edited} post-edited prompts, "
        f"{len(oracle_arg_map)} translated oracle args, "
        f"{len(app_state_map)} translated app state fields, "
        f"{total_validator_subs} validator-sweep substitutions"
    )

    return ds.map(
        lambda example, idx: {"data": final_data[idx]},
        with_indices=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Universe helpers
# ─────────────────────────────────────────────────────────────────────────────


def _build_scenario_to_hash(
    split_ds: datasets.Dataset,
    failed_indices: set[int],
) -> dict[int, str]:
    """Map each non-failed scenario to its universe hash."""
    scenario_to_hash: dict[int, str] = {}
    for i in range(len(split_ds)):
        if i in failed_indices:
            continue
        scenario_to_hash[i] = compute_universe_hash(split_ds[i]["data"])
    return scenario_to_hash


def _build_universe_groups(
    split_ds: datasets.Dataset,
    failed_indices: set[int],
) -> tuple[dict[int, str], dict[str, list]]:
    """Group scenarios by universe hash and extract fields once per universe.

    Returns:
        scenario_to_hash: ``{scenario_idx: universe_hash}``
        universe_fields:  ``{universe_hash: list[AppStateField]}``
            (fields have ``scenario_idx=0`` — the index is irrelevant at
            universe level, only ``app_idx`` and ``field_path`` matter).
    """
    scenario_to_hash = _build_scenario_to_hash(split_ds, failed_indices)
    seen_hashes: set[str] = set()
    universe_fields = {}

    for i, u_hash in scenario_to_hash.items():
        if u_hash not in seen_hashes:
            seen_hashes.add(u_hash)
            fields = extract_translatable_app_fields(
                split_ds[i]["data"], scenario_idx=0
            )
            universe_fields[u_hash] = fields

    n_scenarios = len(scenario_to_hash)
    n_universes = len(universe_fields)
    total_fields = sum(len(fs) for fs in universe_fields.values())
    logger.info(
        f"Universe dedup: {n_scenarios} scenarios → {n_universes} unique universes, "
        f"{total_fields} fields to translate"
    )
    return scenario_to_hash, universe_fields


def _fan_out_universe_translations(
    scenario_to_hash: dict[int, str],
    universe_translations: dict[str, dict[tuple, str]],
) -> dict[tuple, str]:
    """Expand universe-level translations to per-scenario app_state_map.

    Returns ``{(scenario_idx, app_idx, *field_path): translated_value}``.
    """
    app_state_map: dict[tuple, str] = {}
    for scenario_idx, u_hash in scenario_to_hash.items():
        for inner_key, value in universe_translations.get(u_hash, {}).items():
            app_state_map[(scenario_idx, *inner_key)] = value
    return app_state_map


# ─────────────────────────────────────────────────────────────────────────────
# Cross-split universe deduplication
# ─────────────────────────────────────────────────────────────────────────────


def collect_all_universe_fields(
    splits: dict[str, datasets.Dataset],
) -> dict[str, list]:
    """Collect unique universe fields across ALL splits in a subset.

    Iterates every scenario in every split, hashes its ``apps`` JSON, and
    extracts translatable fields for each unique universe.  This lets the
    caller translate each universe only once regardless of how many splits
    share it.

    Returns ``{universe_hash: list[AppStateField]}``.
    """
    seen_hashes: set[str] = set()
    universe_fields: dict[str, list] = {}

    for _split_name, split_ds in splits.items():
        for i in range(len(split_ds)):
            u_hash = compute_universe_hash(split_ds[i]["data"])
            if u_hash not in seen_hashes:
                seen_hashes.add(u_hash)
                fields = extract_translatable_app_fields(
                    split_ds[i]["data"], scenario_idx=0
                )
                universe_fields[u_hash] = fields

    total_fields = sum(len(fs) for fs in universe_fields.values())
    n_splits = len(splits)
    logger.info(
        f"Cross-split universe collection: {len(universe_fields)} unique universes "
        f"across {n_splits} splits, {total_fields} fields to translate"
    )
    return universe_fields


def translate_and_review_universes(
    universe_fields: dict[str, list],
    src_lang: str,
    tgt_lang: str,
    translation_model: str,
    review_model: str,
    skip_validation: bool,
    checkpoint_mgr: CheckpointManager | None = None,
) -> dict[str, dict[tuple, str]]:
    """Translate and review app state for all unique universes (global step).

    Wraps :func:`translate_app_state` and :func:`review_and_post_edit_app_state`
    with optional checkpointing.

    Returns ``{universe_hash: {(app_idx, *field_path): translated_value}}``.
    """
    if not universe_fields:
        return {}

    # ── Translate ────────────────────────────────────────────────────────
    if checkpoint_mgr and checkpoint_mgr.exists("universe_translations"):
        raw = checkpoint_mgr.load("universe_translations")
        universe_translations = deserialize_nested_tuple_key_dict(raw)
    else:
        universe_translations = translate_app_state(
            universe_fields,
            src_lang,
            tgt_lang,
            translation_model,
        )
        if checkpoint_mgr:
            checkpoint_mgr.save(
                "universe_translations",
                serialize_nested_tuple_key_dict(universe_translations),
            )

    # ── Review + post-edit ───────────────────────────────────────────────
    if not skip_validation:
        if checkpoint_mgr and checkpoint_mgr.exists("universe_reviews"):
            raw = checkpoint_mgr.load("universe_reviews")
            universe_translations = deserialize_nested_tuple_key_dict(raw)
        else:
            universe_translations = review_and_post_edit_app_state(
                universe_fields,
                universe_translations,
                src_lang,
                tgt_lang,
                review_model,
            )
            if checkpoint_mgr:
                checkpoint_mgr.save(
                    "universe_reviews",
                    serialize_nested_tuple_key_dict(universe_translations),
                )

    return universe_translations


# ─────────────────────────────────────────────────────────────────────────────
# Per-split processing
# ─────────────────────────────────────────────────────────────────────────────


def load_precomputed_universe_translations(
    path: str | Path,
) -> dict[str, dict[tuple, str]]:
    """Load pre-translated universe translations from a JSON checkpoint file.

    Accepts the same format saved by CheckpointManager for universe_translations
    or universe_reviews (nested tuple-key dicts).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Pre-computed universe file not found: {path}")
    raw = json.loads(path.read_text())
    result = deserialize_nested_tuple_key_dict(raw)
    logger.info(
        f"Loaded pre-computed universe translations from {path}: "
        f"{len(result)} universes, "
        f"{sum(len(v) for v in result.values())} fields"
    )
    return result


def process_split(
    split_ds: datasets.Dataset,
    src_lang: str,
    tgt_lang: str,
    translation_model: str,
    review_model: str,
    skip_validation: bool,
    universe_translations: dict[str, dict[tuple, str]] | None = None,
    checkpoint_mgr: CheckpointManager | None = None,
    split_name: str | None = None,
    glossary: dict[str, str] | None = None,
    term_table_extractor_model: str | None = None,
) -> SplitResult:
    """Run the full translate → review → post-edit pipeline on a split.

    Uses ``translation_model`` (e.g. Claude) for all translation steps and
    ``review_model`` (e.g. GPT) for all review and post-editing steps.

    If a prompt translation fails (e.g. content-filtered by the provider),
    the scenario still flows to the review/post-edit stage, where the
    reviewing and post-editing agents are informed of the failure and
    attempt to produce a translation (recovery).  If recovery succeeds,
    the scenario proceeds normally through oracle-arg and app-state
    translation.  If recovery also fails, the scenario's original JSON
    data is preserved unchanged in the final dataset.

    App state translation uses **universe dedup**: scenarios sharing the same
    simulated environment (identical ``apps`` JSON) are translated only once,
    then fanned out.  When ``universe_translations`` is provided (pre-computed
    across all splits), the per-split translation/review steps are skipped
    entirely and only the lightweight fan-out is performed.

    Pipeline:
     1. extract_prompts()
     2. extract_responses()                   (for reporting only)
     3. build_heuristic_translation_plan()    (no LLM — instant)
     4. translate_prompts()                   → translation_model
     5. detect failed scenarios
     6. review + post-edit prompts            → review_model
     7. build universe groups + extract       ← universe dedup
     8. translate_app_state()                 → translation_model (sub-batched)
     9. review + post-edit app state          → review_model (sub-batched)
    10. fan out universe translations to per-scenario map
    11. translate_oracle_args()               → translation_model
    12. review + post-edit oracle args        → review_model
    13. build_final_dataset()                 — single-pass mutation
    """
    original_prompts = extract_prompts(split_ds)
    expected_responses = extract_responses(split_ds)

    translation_plan = build_heuristic_translation_plan(split_ds)

    # ── Term-table contract (optional) ───────────────────────────────────
    # When the contract is enabled (term_table_extractor_model set AND universe
    # translations available), build per-scenario term tables NOW so they can
    # be injected into stage-2 (prompts) and stage-3 (oracle args). Falls back
    # to legacy glossary path when disabled. See contract/__init__.py for
    # design.
    contract_enabled = (
        term_table_extractor_model is not None and universe_translations is not None
    )
    scenario_term_tables: list[TermTable | None] = [None] * len(original_prompts)
    if contract_enabled:
        if checkpoint_mgr and checkpoint_mgr.exists("scenario_term_tables", split_name):
            raw = checkpoint_mgr.load("scenario_term_tables", split_name)
            scenario_term_tables = deserialize_term_tables(raw)
            logger.info(
                f"Loaded scenario_term_tables from checkpoint "
                f"({sum(1 for tt in scenario_term_tables if tt)} non-empty)"
            )
        else:
            # Build scenario_to_hash early (same logic as later in the file,
            # but failed_indices is not yet populated — universe translation
            # already ran, so all scenarios have hashes).
            early_scenario_to_hash = _build_scenario_to_hash(split_ds, set())
            scenario_term_tables = build_scenario_term_tables(
                split_ds=split_ds,
                src_lang=src_lang,
                tgt_lang=tgt_lang,
                universe_translations=universe_translations,
                scenario_to_hash=early_scenario_to_hash,
                failed_indices=set(),
                extractor_model=term_table_extractor_model,
            )
            if checkpoint_mgr:
                checkpoint_mgr.save(
                    "scenario_term_tables",
                    serialize_term_tables(scenario_term_tables),
                    split_name,
                )

    # Pre-render per-scenario glossary sections (used by stage 2 + 3 when the
    # contract is on). Empty string for scenarios with no term table.
    prompt_glossary_sections: list[str] | None = None
    oracle_glossary_sections_by_scenario: dict[int, str] | None = None
    if contract_enabled:
        prompt_glossary_sections = [
            format_term_table_section(tt) if tt is not None else ""
            for tt in scenario_term_tables
        ]
        oracle_glossary_sections_by_scenario = {
            i: format_term_table_section(scenario_term_tables[i])
            for i in range(len(scenario_term_tables))
            if scenario_term_tables[i] is not None
        }

    # ── 4. Translate prompts ─────────────────────────────────────────────
    if checkpoint_mgr and checkpoint_mgr.exists("translated_prompts", split_name):
        translated_prompts = checkpoint_mgr.load("translated_prompts", split_name)
        logger.info(
            f"Loaded translated_prompts from checkpoint ({len(translated_prompts)} items)"
        )
    else:
        translated_prompts = translate_prompts(
            original_prompts,
            src_lang,
            tgt_lang,
            translation_model,
            glossary=glossary,
            glossary_sections=prompt_glossary_sections,
        )
        if checkpoint_mgr:
            checkpoint_mgr.save("translated_prompts", translated_prompts, split_name)

    # ── 5. Build per-scenario tracking and detect failures ───────────────
    scenario_translations: list[ScenarioTranslation] = []
    for i, (orig, trans) in enumerate(zip(original_prompts, translated_prompts)):
        st = ScenarioTranslation(scenario_idx=i, original_prompt=orig)
        if trans is None:
            st.failed = True
            st.failure_reason = "prompt_translation_filtered"
            logger.warning(
                f"Scenario {i}: prompt translation returned None, "
                f"will attempt recovery via review/post-edit"
            )
        else:
            st.translated_prompt = trans
        scenario_translations.append(st)

    failed_indices = {st.scenario_idx for st in scenario_translations if st.failed}

    if failed_indices:
        logger.info(
            f"Pipeline: {len(failed_indices)}/{len(scenario_translations)} "
            f"scenarios failed prompt translation, will attempt recovery"
        )

    # ── 6. Review + post-edit prompts (all scenarios, including failed) ──
    translation_reviews: list[dict | None] = [None] * len(original_prompts)

    if checkpoint_mgr and checkpoint_mgr.exists("prompt_reviews", split_name):
        saved = checkpoint_mgr.load("prompt_reviews", split_name)
        translated_prompts = saved["translated_prompts"]
        translation_reviews = saved["translation_reviews"]
        failed_indices = set(saved.get("failed_indices", []))
        for i in range(len(original_prompts)):
            if i not in failed_indices:
                scenario_translations[i].translated_prompt = translated_prompts[i]
                scenario_translations[i].prompt_review = translation_reviews[i]
                scenario_translations[i].failed = False
        for i in failed_indices:
            scenario_translations[i].failed = True
        logger.info(
            f"Loaded prompt_reviews from checkpoint "
            f"({len(failed_indices)} still failed after recovery)"
        )
    elif not skip_validation:
        all_originals = [
            original_prompts[i] or "" for i in range(len(original_prompts))
        ]
        all_translations = [
            translated_prompts[i]
            if translated_prompts[i] is not None
            else (original_prompts[i] or "")
            for i in range(len(original_prompts))
        ]
        failure_flags = [i in failed_indices for i in range(len(original_prompts))]

        edited_all, reviews_all = review_and_post_edit(
            all_originals,
            all_translations,
            src_lang,
            tgt_lang,
            review_model,
            desc="prompt",
            failure_flags=failure_flags,
        )

        for i in range(len(original_prompts)):
            translated_prompts[i] = edited_all[i]
            translation_reviews[i] = reviews_all[i]
            scenario_translations[i].translated_prompt = edited_all[i]
            scenario_translations[i].prompt_review = reviews_all[i]

        # Check for recovered scenarios: previously failed but post-editor
        # produced a translation different from the original prompt.
        recovered = set()
        for i in list(failed_indices):
            if edited_all[i] is not None and edited_all[i] != (
                original_prompts[i] or ""
            ):
                recovered.add(i)
                scenario_translations[i].failed = False
                scenario_translations[
                    i
                ].failure_reason = "prompt_translation_filtered_then_recovered"
                logger.info(
                    f"Scenario {i}: RECOVERED by review/post-edit agent "
                    f"(was: prompt_translation_filtered)"
                )

        if recovered:
            failed_indices -= recovered
            logger.info(
                f"Pipeline: recovered {len(recovered)} scenarios via review/post-edit, "
                f"{len(failed_indices)} still failed"
            )

        if checkpoint_mgr:
            checkpoint_mgr.save(
                "prompt_reviews",
                {
                    "translated_prompts": translated_prompts,
                    "translation_reviews": translation_reviews,
                    "failed_indices": list(failed_indices),
                },
                split_name,
            )

    # ── 7–10. Translate + review app state (universe dedup) ──────────────
    app_state_map: dict[tuple, str] = {}

    if universe_translations is not None:
        # Pre-computed across all splits — just map scenarios to hashes and fan out
        scenario_to_hash = _build_scenario_to_hash(split_ds, failed_indices)
        app_state_map = _fan_out_universe_translations(
            scenario_to_hash, universe_translations
        )
        logger.info(
            f"Using pre-computed universe translations: "
            f"fanned out to {len(app_state_map)} app state fields"
        )
    else:
        # Fallback: per-split universe dedup (backward compatible)
        scenario_to_hash, per_split_fields = _build_universe_groups(
            split_ds, failed_indices
        )

        if per_split_fields:
            per_split_translations = translate_app_state(
                per_split_fields,
                src_lang,
                tgt_lang,
                translation_model,
            )

            if not skip_validation:
                per_split_translations = review_and_post_edit_app_state(
                    per_split_fields,
                    per_split_translations,
                    src_lang,
                    tgt_lang,
                    review_model,
                )

            app_state_map = _fan_out_universe_translations(
                scenario_to_hash, per_split_translations
            )

    # ── 11–12. Translate + review oracle args (skip failed scenarios) ────
    oracle_arg_map: dict[tuple, str] = {}
    ok_plan = [
        arg for arg in translation_plan if arg.scenario_idx not in failed_indices
    ]

    if checkpoint_mgr and checkpoint_mgr.exists("oracle_args_reviewed", split_name):
        raw = checkpoint_mgr.load("oracle_args_reviewed", split_name)
        oracle_arg_map = deserialize_tuple_key_dict(raw)
        logger.info(
            f"Loaded oracle_args_reviewed from checkpoint ({len(oracle_arg_map)} args)"
        )
    elif checkpoint_mgr and checkpoint_mgr.exists("oracle_args_translated", split_name):
        raw = checkpoint_mgr.load("oracle_args_translated", split_name)
        oracle_arg_map = deserialize_tuple_key_dict(raw)
        logger.info(
            f"Loaded oracle_args_translated from checkpoint "
            f"({len(oracle_arg_map)} args)"
        )

        if not skip_validation:
            oracle_arg_map, _ = review_and_post_edit_oracle_args(
                ok_plan,
                oracle_arg_map,
                original_prompts,
                src_lang,
                tgt_lang,
                review_model,
            )
            if checkpoint_mgr:
                checkpoint_mgr.save(
                    "oracle_args_reviewed",
                    serialize_tuple_key_dict(oracle_arg_map),
                    split_name,
                )
    elif ok_plan:
        oracle_arg_map = translate_oracle_args(
            ok_plan,
            original_prompts,
            src_lang,
            tgt_lang,
            translation_model,
            glossary=glossary,
            glossary_sections_by_scenario=oracle_glossary_sections_by_scenario,
        )
        if checkpoint_mgr:
            checkpoint_mgr.save(
                "oracle_args_translated",
                serialize_tuple_key_dict(oracle_arg_map),
                split_name,
            )

        if not skip_validation:
            oracle_arg_map, _ = review_and_post_edit_oracle_args(
                ok_plan,
                oracle_arg_map,
                original_prompts,
                src_lang,
                tgt_lang,
                review_model,
            )
            if checkpoint_mgr:
                checkpoint_mgr.save(
                    "oracle_args_reviewed",
                    serialize_tuple_key_dict(oracle_arg_map),
                    split_name,
                )

    # ── 13. Assemble final dataset ───────────────────────────────────────
    final_ds = build_final_dataset(
        split_ds,
        translated_prompts,
        oracle_arg_map=oracle_arg_map,
        app_state_map=app_state_map,
        failed_indices=failed_indices,
        scenario_term_tables=scenario_term_tables if contract_enabled else None,
    )

    return SplitResult(
        dataset=final_ds,
        original_prompts=original_prompts,
        translated_prompts=[t if t is not None else "" for t in translated_prompts],
        expected_responses=expected_responses,
        translation_reviews=translation_reviews,
        oracle_arg_translations=oracle_arg_map,
        app_state_translations=app_state_map,
        translation_plan=translation_plan,
        scenario_translations=scenario_translations,
    )
