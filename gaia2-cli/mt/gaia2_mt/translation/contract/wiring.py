# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Per-scenario term-table assembly — bridges the universe pipeline to the
TermTable contract.

Given the universe-translator's outputs and per-scenario context (user prompt,
oracle args, app_state fields), produces one ``TermTable`` per scenario for
injection into stage-2 (prompt translation) and stage-3 (oracle arg
translation) via the existing ``{glossary_section}`` slot.

Cost: one Pass-A LLM call per scenario (~6400 calls for a full run); negligible
vs. the universe pass's tens of thousands of calls. Falls back gracefully to
glossary-only term tables when the LLM client is None or Pass A fails.
"""

from __future__ import annotations

import logging

import datasets

from gaia2_mt.data import (
    extract_initial_prompt,
    extract_oracle_events,
    extract_translatable_app_fields,
    get_language_display_name,
)

from .llm_adapter import OpenAICompatLLMClient
from .term_table import TermTable, build_term_table


logger = logging.getLogger(__name__)


def _per_scenario_universe_fields(
    split_ds: datasets.Dataset,
    scenario_to_hash: dict[int, str],
    universe_translations: dict[str, dict[tuple, str]],
) -> dict[int, list[tuple[tuple, str, str]]]:
    """Build ``{scenario_idx: [(path, src, tgt), ...]}`` from the universe map.

    Path tuples are ``(app_idx, *field_path)`` — the same key shape the
    universe translator uses, so a zip on equal keys is well-defined.
    """
    out: dict[int, list[tuple[tuple, str, str]]] = {}
    for scenario_idx, u_hash in scenario_to_hash.items():
        tgt_map = universe_translations.get(u_hash, {})
        if not tgt_map:
            out[scenario_idx] = []
            continue
        fields = extract_translatable_app_fields(
            split_ds[scenario_idx]["data"], scenario_idx=0
        )
        triples: list[tuple[tuple, str, str]] = []
        for f in fields:
            key = (f.app_idx, *f.field_path)
            tgt = tgt_map.get(key)
            if tgt is None:
                # Universe map covers the universe via the FIRST scenario seen;
                # missing keys here mean translator dropped that field. Keep
                # the src-only row so passthrough/glossary detection still
                # works (target == source preserves identity).
                triples.append((key, f.field_value, f.field_value))
            else:
                triples.append((key, f.field_value, tgt))
        out[scenario_idx] = triples
    return out


def _per_scenario_oracle_args(
    split_ds: datasets.Dataset,
) -> dict[int, list[tuple[str, str]]]:
    """Build ``{scenario_idx: [(arg_name, arg_value), ...]}`` from oracle events.

    Pulls all string-valued args across all non-send_message_to_agent events.
    Used by the term-extractor to render the oracle-replies surface in its
    prompt.
    """
    out: dict[int, list[tuple[str, str]]] = {}
    for i in range(len(split_ds)):
        events = extract_oracle_events(split_ds[i]["data"])
        pairs: list[tuple[str, str]] = []
        for ev in events:
            if ev["function"] == "send_message_to_agent":
                continue
            for k, v in ev["args"].items():
                if isinstance(v, str) and v.strip():
                    pairs.append((k, v))
        out[i] = pairs
    return out


def build_scenario_term_tables(
    split_ds: datasets.Dataset,
    src_lang: str,
    tgt_lang: str,
    universe_translations: dict[str, dict[tuple, str]],
    scenario_to_hash: dict[int, str],
    failed_indices: set[int] | None = None,
    extractor_model: str | None = None,
) -> list[TermTable | None]:
    """Assemble per-scenario term tables.

    Returns a list of length ``len(split_ds)``; entry ``i`` is None when the
    scenario is in ``failed_indices`` or has no usable input.

    When ``extractor_model`` is None, Pass A is skipped — term tables fall
    back to glossary + passthrough only (still useful, ~30% F2 coverage from
    end-to-end tests).
    """
    failed_indices = failed_indices or set()
    tgt_display = get_language_display_name(tgt_lang)

    # Build LLM client lazily; one instance shared across all scenarios.
    llm_client = None
    if extractor_model:
        try:
            llm_client = OpenAICompatLLMClient(model_name=extractor_model)
        except Exception as e:  # pragma: no cover - construction-time only
            logger.warning(
                f"Failed to construct term extractor LLM client "
                f"({extractor_model}): {e}; falling back to glossary-only"
            )
            llm_client = None

    per_scn_fields = _per_scenario_universe_fields(
        split_ds, scenario_to_hash, universe_translations
    )
    per_scn_oracle = _per_scenario_oracle_args(split_ds)

    # Pass A: extract terms for every non-failed scenario in one batch call,
    # using the adapter's batched interface. Skip when no LLM configured.
    extracted_by_idx: dict[int, list] = {}
    if llm_client is not None:
        # Build (system, prompt) tuples paired with scenario indices.
        from .term_extractor import (
            TERM_EXTRACTION_PROMPT,
            TERM_EXTRACTION_SYSTEM_PROMPT,
            _render_oracle_replies,
            _sample_app_state,
            parse_extraction_response,
        )

        indices: list[int] = []
        pairs: list[tuple[str, str]] = []
        for i in range(len(split_ds)):
            if i in failed_indices:
                continue
            user_prompt = extract_initial_prompt(split_ds[i]["data"])
            if not user_prompt:
                continue
            triples = per_scn_fields.get(i, [])
            app_fields_for_extractor = [(p, s) for p, s, _ in triples]
            oracle_args = per_scn_oracle.get(i, [])
            prompt = TERM_EXTRACTION_PROMPT.format(
                tgt_lang=tgt_display,
                user_prompt=user_prompt,
                oracle_replies=_render_oracle_replies(oracle_args),
                app_state_samples=_sample_app_state(app_fields_for_extractor),
            )
            indices.append(i)
            pairs.append((TERM_EXTRACTION_SYSTEM_PROMPT, prompt))

        logger.info(
            f"Term extractor: running Pass A on {len(pairs)} scenarios "
            f"(model={extractor_model})"
        )
        responses = llm_client.infer_batch(pairs) if pairs else []
        for i, raw in zip(indices, responses):
            try:
                extracted_by_idx[i] = parse_extraction_response(raw)
            except Exception as e:
                logger.warning(f"Pass A parse failed for scenario {i}: {e}")
                extracted_by_idx[i] = []

    # Build per-scenario term tables.
    out: list[TermTable | None] = []
    for i in range(len(split_ds)):
        if i in failed_indices:
            out.append(None)
            continue
        user_prompt = extract_initial_prompt(split_ds[i]["data"])
        triples = per_scn_fields.get(i, [])
        oracle_args = per_scn_oracle.get(i, [])
        extracted = extracted_by_idx.get(i, [])
        tt = build_term_table(
            universe_app_fields=triples,
            prompts=[user_prompt] if user_prompt else [],
            oracle_args=oracle_args,
            extracted_terms=extracted if extracted else None,
        )
        out.append(tt)

    n_built = sum(1 for tt in out if tt is not None)
    total_entries = sum(len(tt) for tt in out if tt is not None)
    logger.info(
        f"Built {n_built} per-scenario term tables, "
        f"{total_entries} total entries ({total_entries / max(1, n_built):.1f} avg)"
    )
    return out


def serialize_term_tables(
    term_tables: list[TermTable | None],
) -> list[dict | None]:
    """JSON-safe representation for checkpointing."""
    out: list[dict | None] = []
    for tt in term_tables:
        if tt is None:
            out.append(None)
        else:
            out.append({"entries": tt.entries, "provenance": tt.provenance})
    return out


def deserialize_term_tables(
    raw: list[dict | None],
) -> list[TermTable | None]:
    """Inverse of ``serialize_term_tables``."""
    out: list[TermTable | None] = []
    for r in raw:
        if r is None:
            out.append(None)
        else:
            out.append(
                TermTable(
                    entries=dict(r.get("entries", {})),
                    provenance=dict(r.get("provenance", {})),
                )
            )
    return out
