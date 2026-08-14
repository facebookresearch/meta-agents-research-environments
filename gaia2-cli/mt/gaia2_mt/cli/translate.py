#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""
GAIA2: Translate User Prompts + Oracle Event Args + LLM Validation + Post-editing

CLI entry point for the GAIA2 translation pipeline.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import datasets

from gaia2_mt.checkpoint import CheckpointManager
from gaia2_mt.data import (
    ALL_SUBSETS,
    LANG_CODE_TO_NAME,
    SplitResult,
    extract_expected_response,
    extract_glossary,
    extract_initial_prompt,
)
from gaia2_mt.llm import (
    DEFAULT_REVIEW_MODEL,
    DEFAULT_TRANSLATION_MODEL,
    resolve_endpoint,
)
from gaia2_mt.reporting import build_results_dataframe
from gaia2_mt.translation import (
    collect_all_universe_fields,
    load_dataset,
    load_precomputed_universe_translations,
    process_split,
    translate_and_review_universes,
)


logger = logging.getLogger(__name__)


def main(
    output_dir: str,
    dataset_id: str = "meta-agents-research-environments/gaia2",
    subset: str = "search",
    translation_model: str = DEFAULT_TRANSLATION_MODEL,
    review_model: str = DEFAULT_REVIEW_MODEL,
    src_lang: str = "eng_Latn",
    tgt_lang: str = "spa_Latn",
    results_csv: str | None = None,
    review: bool = False,
    limit: int | None = None,
    checkpoint_dir: str | None = None,
    local_data_dir: str | None = None,
    precomputed_universes: str | None = None,
    skip_app_state: bool = False,
    lid_check: bool = False,
    lid_threshold: float = 0.3,
    term_table: bool = True,
    term_table_extractor_model: str | None = None,
):
    """Translate GAIA2 dataset with heuristic oracle arg identification, validation and post-editing.

    Supports all four GAIA2 subsets (search, execution, ambiguity, adaptability).
    Uses a heuristic allowlist (TRANSLATABLE_ARG_NAMES) to identify translatable
    oracle event args, replacing the previous LLM classification pipeline.

    Models are served by a local OpenAI-compatible server (vLLM); point
    ``GAIA2_MT_LLM_BASE_URL`` at it, or ``GAIA2_MT_PER_MODEL_ENDPOINTS`` for an
    asymmetric translator + reviewer deployment. Model names must match the
    server's ``--served-model-name``.

    Args:
        output_dir: Base directory to save the translated dataset.
        dataset_id: HuggingFace dataset ID.
        subset: Dataset subset to translate. Use "all" to process all four subsets
            sequentially, saving to output_dir/{subset}/.
        translation_model: Model for translation steps (prompts, oracle args,
            responses). Defaults to the published translator.
        review_model: Model used for review and post-editing, when ``review``
            is enabled.
        src_lang: Source language code.
        tgt_lang: Target language code.
        results_csv: Path to save the audit results CSV (optional).
        review: Add a second review + post-edit model pass after translation.
            Off by default: the published pipeline is a single translator-only
            pass, because the reviewer rewrote under 9% of fields with almost
            entirely stylistic edits and no measurable quality gain, while
            roughly doubling per-scenario latency. Retained as a reproducible
            negative result — see ``translation/review.py``.
            Note this also enables the only recovery path for prompts the
            translator's content filter rejects; with review off, those
            scenarios ship with their prompt untranslated.
        limit: If set, truncate each split to at most this many scenarios.
            Useful for quick smoke tests (e.g. --limit 3).
        checkpoint_dir: Directory for saving/loading intermediate pipeline
            results.  When provided, completed steps are skipped on re-runs
            and splits with existing output parquets are skipped entirely.
        local_data_dir: Path to a local directory containing subset folders
            with JSON scenario files.  When provided, scenarios are loaded
            from disk instead of HuggingFace.  Handles nested structures
            produced by ``manifold getr``.
        skip_app_state: Skip universe/app-state translation entirely.
            Only translates prompts and oracle args.  Useful for fast
            smoke tests to verify the pipeline works for a new language.
        lid_check: Run GlotLID language identification over the translated
            content as an advisory signal. Off by default, as it was for the
            released dataset: it gates nothing, its verdicts only reach the
            ``results_csv``, and it needs a downloadable fastText model.
        lid_threshold: Minimum GlotLID confidence for a text to be
            considered correctly identified (default: 0.3).
        term_table: Build a per-scenario term table and enforce it across
            surfaces. On by default — this is stage (b) of the published
            pipeline, and what keeps the translated environment consistent
            enough for the verifier to admit the intended solution. Turning it
            off falls back to the legacy batch-shared glossary path.
        term_table_extractor_model: Model for the term table's per-scenario
            extraction pass. Defaults to ``translation_model``. Term tables are
            injected into stages 2 + 3 via the ``{glossary_section}`` slot, and
            a final sweep substitutes any leaked source span in the assembled
            dataset. See ``gaia2_mt.translation.contract`` for full design.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logging.getLogger("httpx").setLevel(logging.ERROR)
    logging.getLogger("openai").setLevel(logging.ERROR)

    for envvar in [
        "HF_HUB_OFFLINE",
        "TRANSFORMERS_OFFLINE",
        "HF_DATASETS_OFFLINE",
        "HF_HUB_ENABLE_HF_TRANSFER",
    ]:
        os.environ.pop(envvar, None)

    subsets_to_process = ALL_SUBSETS if subset == "all" else [subset]

    if limit is not None:
        logger.info(f"Smoke-test mode: limiting each split to {limit} scenarios")

    tgt_lang_name = LANG_CODE_TO_NAME.get(tgt_lang, tgt_lang)
    logger.info(f"Target language: {tgt_lang_name} ({tgt_lang})")
    logger.info(
        f"Translation model: {translation_model} "
        f"({resolve_endpoint(translation_model)})"
    )
    # The published pipeline is translator-only; `review` opts back into the
    # second model pass. Internally the stages still take `skip_validation`.
    skip_validation = not review
    if review:
        logger.info(
            f"Review model:      {review_model} ({resolve_endpoint(review_model)})"
        )
    else:
        logger.info("Review pass:       DISABLED (translator-only, as published)")

    # Stage (b) of the pipeline: on unless explicitly disabled. The extractor
    # defaults to the translator so a single served model covers every stage.
    term_table_extractor_model = (
        (term_table_extractor_model or translation_model) if term_table else None
    )
    if term_table_extractor_model:
        logger.info(
            f"TermTable contract: ENABLED (extractor={term_table_extractor_model})"
        )
    else:
        logger.warning(
            "TermTable contract: DISABLED — falling back to the legacy shared "
            "glossary. Cross-surface consistency is not enforced."
        )

    if checkpoint_dir:
        logger.info(f"Checkpointing enabled: {checkpoint_dir}")

    # ── Pass 1: load all subsets, detect pending splits, collect universes ─
    # We aggregate universe fields across ALL subsets before translating,
    # because different subsets share many of the same app-state universes.

    SubsetInfo = tuple[
        Path,  # save_path
        dict[str, datasets.Dataset],  # pending_splits
        dict[str, datasets.Dataset],  # original_splits
        CheckpointManager | None,  # per-subset checkpoint_mgr
    ]
    subset_infos: dict[str, SubsetInfo] = {}
    global_universe_fields: dict[str, list] = {}

    for current_subset in subsets_to_process:
        logger.info(f"{'#' * 60}")
        logger.info(f"# Loading subset: {current_subset}")
        logger.info(f"{'#' * 60}")

        ds_dict = load_dataset(
            dataset_id, current_subset, local_data_dir=local_data_dir
        )

        subset_output_dir = str(Path(output_dir) / current_subset)

        save_path = Path(subset_output_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        pending_splits: dict[str, datasets.Dataset] = {}
        original_splits: dict[str, datasets.Dataset] = {}

        for split_name, split_ds in ds_dict.items():
            if limit is not None:
                split_ds = split_ds.select(range(min(limit, len(split_ds))))
            original_splits[split_name] = split_ds

            done_marker = save_path / f".{split_name}.done"
            if done_marker.exists():
                logger.info(
                    f"Skipping split '{split_name}': output already exists "
                    f"at {save_path}"
                )
            else:
                pending_splits[split_name] = split_ds

        if not pending_splits:
            logger.info(
                f"All splits already completed for subset '{current_subset}', skipping"
            )
            continue

        checkpoint_mgr = (
            CheckpointManager(Path(checkpoint_dir), current_subset)
            if checkpoint_dir
            else None
        )

        subset_infos[current_subset] = (
            save_path,
            pending_splits,
            original_splits,
            checkpoint_mgr,
        )

        # Collect universe fields from this subset's pending splits and merge
        # into the global dict (keyed by universe hash, so duplicates are free).
        subset_fields = collect_all_universe_fields(pending_splits)
        for u_hash, fields in subset_fields.items():
            if u_hash not in global_universe_fields:
                global_universe_fields[u_hash] = fields

    if not subset_infos:
        logger.info("Nothing to do — all subsets already completed.")
        return

    total_fields = sum(len(fs) for fs in global_universe_fields.values())
    logger.info(
        f"Global universe collection: {len(global_universe_fields)} unique universes "
        f"across {len(subset_infos)} subsets, {total_fields} fields to translate"
    )

    # ── Translate universes once (global, cross-subset) ───────────────────
    if skip_app_state:
        logger.info("Skipping app-state/universe translation (--skip_app_state)")
        universe_translations: dict[str, dict[tuple, str]] = {}
        glossary: dict[str, str] = {}
    else:
        global_checkpoint_mgr = (
            CheckpointManager(Path(checkpoint_dir), "_global")
            if checkpoint_dir
            else None
        )

        if precomputed_universes:
            universe_translations = load_precomputed_universe_translations(
                precomputed_universes
            )
        else:
            universe_translations = translate_and_review_universes(
                global_universe_fields,
                src_lang=src_lang,
                tgt_lang=tgt_lang,
                translation_model=translation_model,
                review_model=review_model,
                skip_validation=skip_validation,
                checkpoint_mgr=global_checkpoint_mgr,
            )

        glossary = extract_glossary(global_universe_fields, universe_translations)
        logger.info(f"Extracted terminology glossary: {len(glossary)} entries")

    # ── Pass 2: per-subset, per-split processing ──────────────────────────
    all_results_frames = []

    for current_subset, (
        save_path,
        pending_splits,
        original_splits,
        checkpoint_mgr,
    ) in subset_infos.items():
        logger.info(f"{'#' * 60}")
        logger.info(f"# Processing subset: {current_subset}")
        logger.info(f"{'#' * 60}")

        split_results: dict[str, SplitResult] = {}

        for split_name, split_ds in pending_splits.items():
            logger.info(f"{'=' * 60}")
            logger.info(
                f"Processing split: {split_name} ({len(split_ds)} rows) "
                f"[subset={current_subset}]"
            )
            logger.info(f"{'=' * 60}")

            split_results[split_name] = process_split(
                split_ds=split_ds,
                src_lang=src_lang,
                tgt_lang=tgt_lang,
                translation_model=translation_model,
                review_model=review_model,
                skip_validation=skip_validation,
                universe_translations=universe_translations,
                checkpoint_mgr=checkpoint_mgr,
                split_name=split_name,
                glossary=glossary or None,
                term_table_extractor_model=term_table_extractor_model,
            )

            # ── LID validation (informational) ───────────────────────
            if lid_check:
                from gaia2_mt.lid import validate_translations_lid

                lid_report = validate_translations_lid(
                    split_results[split_name],
                    tgt_lang,
                    threshold=lid_threshold,
                )
                split_results[split_name].lid_report = lid_report

        # ── Save pending splits as individual JSON scenario files ──────
        for split_name, result in split_results.items():
            count = 0
            for idx in range(len(result.dataset)):
                scenario_json = json.loads(result.dataset[idx]["data"])
                json_file = save_path / f"scenario_{idx:04d}.json"
                json_file.write_text(
                    json.dumps(scenario_json, indent=2, ensure_ascii=False) + "\n"
                )
                count += 1
            done_marker = save_path / f".{split_name}.done"
            done_marker.write_text(f"{count}\n")
            logger.info(f"Saved {split_name} ({count} scenarios) to: {save_path}")

        # ── Build results CSV (pending splits only) ──────────────────────
        # Not gated on the review pass: the CSV also carries the LID verdicts
        # and per-arg audit trail, which are the whole point under a
        # translator-only run.
        if results_csv:
            for split_name, result in split_results.items():
                df = build_results_dataframe(
                    original_splits[split_name],
                    result.original_prompts,
                    result.translated_prompts,
                    result.expected_responses,
                    result.translation_reviews,
                    translation_plan=result.translation_plan or None,
                    oracle_arg_map=result.oracle_arg_translations or None,
                    app_state_map=result.app_state_translations or None,
                    lid_report=result.lid_report,
                )
                df.insert(0, "subset", current_subset)
                df.insert(1, "split", split_name)
                all_results_frames.append(df)

        # ── Verify all splits ─────────────────────────────────────────
        all_split_results = dict(split_results)
        json_files = sorted(save_path.glob("scenario_*.json"))

        for split_name in original_splits:
            original_input = original_splits[split_name]

            if split_name in all_split_results:
                translated = all_split_results[split_name].dataset
            else:
                translated = original_input

            assert len(json_files) == len(translated), (
                f"JSON file count mismatch in '{split_name}': "
                f"{len(json_files)} files vs {len(translated)} rows"
            )

            changed_prompts = sum(
                1
                for i in range(len(translated))
                if extract_initial_prompt(original_input[i]["data"])
                != extract_initial_prompt(translated[i]["data"])
            )
            changed_responses = sum(
                1
                for i in range(len(translated))
                if extract_expected_response(original_input[i]["data"])
                != extract_expected_response(translated[i]["data"])
            )

            if split_name in all_split_results:
                changed_args = len(
                    all_split_results[split_name].oracle_arg_translations
                )
                changed_app_state = len(
                    all_split_results[split_name].app_state_translations
                )
            else:
                changed_args = 0
                changed_app_state = 0

            logger.info(
                f"[{current_subset}/{split_name}] "
                f"Translated prompts: {changed_prompts}/{len(translated)}, "
                f"translated responses: {changed_responses}/{len(translated)}, "
                f"translated intermediate args: {changed_args}, "
                f"translated app state fields: {changed_app_state}"
            )

    if results_csv and all_results_frames:
        import pandas as pd

        results_df = pd.concat(all_results_frames, ignore_index=True)
        os.makedirs(os.path.dirname(results_csv) or ".", exist_ok=True)
        results_df.to_csv(results_csv, index=False)
        logger.info(f"Saved validation results to: {results_csv}")


if __name__ == "__main__":
    import fire

    fire.Fire(main)
