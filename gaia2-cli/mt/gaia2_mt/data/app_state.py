# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""App state extraction and mutation for the GAIA2 translation pipeline.

Provides data models and utilities to identify translatable text fields
within each app's initial state (calendar events, emails, messages, etc.)
and to apply translations back into the scenario JSON.

GAIA2 scenarios share app states through "universes" — a handful of distinct
simulated environments.  :func:`compute_universe_hash` lets callers deduplicate
translation work so each universe is translated only once.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass

import xxhash


logger: logging.Logger = logging.getLogger(__name__)

APP_TRANSLATABLE_FIELDS: dict[str, set[str]] = {
    "Calendar": {"title", "description", "tag"},
    "Emails": {"subject", "content"},
    "EmailClientV2": {"subject", "content"},
    "Messages": {"content", "title"},
    "Chats": {"content", "title"},
    "MessagingAppV2": {"content", "title"},
    "WhatsAppV2": {"content", "title"},
    "MessengerV2": {"content", "title"},
    "Contacts": {"job", "description"},
    "InternalContacts": {"job", "description"},
    "Shopping": {"name", "description"},
    "RentAFlat": {
        "name",
        "property_type",
        "furnished_status",
        "floor_level",
        "pet_policy",
        "lease_term",
    },
}

SKIP_APPS: set[str] = {
    "AgentUserInterface",
    "Files",
    "City",
    "Cabs",
}


@dataclass
class AppStateField:
    """One translatable field from an app's initial state."""

    scenario_idx: int
    app_idx: int
    app_name: str
    field_path: tuple
    field_value: str


def compute_universe_hash(data_json: str) -> str:
    """Hash the ``apps`` portion of a scenario to identify its universe.

    Scenarios sharing the same universe have identical app states; translating
    once per unique hash avoids redundant LLM calls.
    """
    data = json.loads(data_json)
    apps_json = json.dumps(data.get("apps", []), sort_keys=True)
    return xxhash.xxh64(apps_json.encode()).hexdigest()


def _walk_app_state(
    obj: object,
    translatable_fields: set[str],
    path: tuple,
    results: list[tuple[tuple, str]],
) -> None:
    """Recursively walk a nested structure, collecting translatable string fields."""
    if isinstance(obj, dict):
        for key, value in obj.items():
            child_path = (*path, key)
            if key in translatable_fields and isinstance(value, str) and value.strip():
                results.append((child_path, value))
            else:
                _walk_app_state(value, translatable_fields, child_path, results)
    elif isinstance(obj, list):
        for idx, item in enumerate(obj):
            _walk_app_state(item, translatable_fields, (*path, idx), results)


def extract_translatable_app_fields(
    data_json: str,
    scenario_idx: int,
) -> list[AppStateField]:
    """Extract all translatable text fields from every app's initial state.

    Walks each app's ``app_state`` recursively; when a key matches the app's
    translatable field set and the value is a non-empty string, emits an
    ``AppStateField``.
    """
    data = json.loads(data_json)
    fields: list[AppStateField] = []

    for app_idx, app in enumerate(data.get("apps", [])):
        app_name = app.get("name", "")
        if app_name in SKIP_APPS:
            continue

        translatable = APP_TRANSLATABLE_FIELDS.get(app_name)
        if not translatable:
            continue

        app_state = app.get("app_state", {})
        collected: list[tuple[tuple, str]] = []
        _walk_app_state(app_state, translatable, (), collected)

        for field_path, field_value in collected:
            fields.append(
                AppStateField(
                    scenario_idx=scenario_idx,
                    app_idx=app_idx,
                    app_name=app_name,
                    field_path=field_path,
                    field_value=field_value,
                )
            )

    return fields


GLOSSARY_FIELDS: set[str] = {"title", "subject", "tag", "name", "job", "property_type"}

# When set, cap the glossary returned by :func:`extract_glossary` to N entries
# by shortest source string. The per-scenario translation prompt inlines the
# whole glossary, which without a cap inflates to 8K–33K tokens on large
# universes (1057 entries observed in Omnilingual-GAIA2 v2). Shortest-source picks
# the actual terminology (city names, single-word jobs, contact names) over
# free-form description paragraphs that are not re-referenced cross-scenario.
# Unset / "0" disables capping (default — behaviour unchanged for existing
# callers).
GLOSSARY_MAX_ENTRIES_ENV = "GAIA2_MT_GLOSSARY_MAX_ENTRIES"


def _resolve_glossary_cap() -> int:
    # Default cap = 200, taking the shortest source strings. An earlier default
    # of 0 (unlimited) overran a 131072-token context window once the global
    # glossary reached ~7500 entries, so the cap is on by default.
    # OMNILINGUAL-GAIA2_GLOSSARY_MAX_ENTRIES overrides it; 0 restores unlimited.
    raw = os.environ.get(GLOSSARY_MAX_ENTRIES_ENV, "").strip()
    if not raw:
        return 200
    try:
        cap = int(raw)
    except ValueError as e:
        raise RuntimeError(
            f"{GLOSSARY_MAX_ENTRIES_ENV} must be a non-negative integer, got {raw!r}"
        ) from e
    if cap < 0:
        raise RuntimeError(
            f"{GLOSSARY_MAX_ENTRIES_ENV} must be a non-negative integer, got {cap}"
        )
    return cap


def extract_glossary(
    universe_fields: dict[str, list[AppStateField]],
    universe_translations: dict[str, dict[tuple, str]],
) -> dict[str, str]:
    """Build a glossary of original→translated short reference strings.

    Extracts only short reference fields (titles, subjects, tags, names)
    from universe translations to anchor terminology across all layers.
    Deduplicates entries — if the same English string appears multiple times,
    only one mapping is kept.

    When ``GAIA2_MT_GLOSSARY_MAX_ENTRIES`` is set to a positive integer N and
    the full glossary exceeds N entries, the result is truncated to the N
    shortest source strings (see :data:`GLOSSARY_MAX_ENTRIES_ENV`).

    Returns ``{"Catching up with friends": "Rattraper le temps avec des amis", ...}``.
    """
    glossary: dict[str, str] = {}
    for u_hash, fields in universe_fields.items():
        translations = universe_translations.get(u_hash, {})
        for f in fields:
            if f.field_path[-1] not in GLOSSARY_FIELDS:
                continue
            key = (f.app_idx, *f.field_path)
            translated = translations.get(key)
            if translated and translated != f.field_value:
                glossary[f.field_value] = translated

    cap = _resolve_glossary_cap()
    if cap and len(glossary) > cap:
        sorted_items = sorted(glossary.items(), key=lambda kv: len(kv[0]))
        capped = dict(sorted_items[:cap])
        logger.info(
            "extract_glossary: capped %d -> %d entries (by shortest source, "
            "GAIA2_MT_GLOSSARY_MAX_ENTRIES=%d)",
            len(glossary),
            cap,
            cap,
        )
        return capped
    return glossary


def apply_app_state_translations(
    data: dict,
    translations: dict[tuple, str],
) -> None:
    """Apply translated values back into a scenario's app states (in-place).

    *translations* maps ``(app_idx, *field_path) → translated_value``.
    Navigates into ``data["apps"][app_idx]["app_state"]`` and replaces the
    leaf value at *field_path*.
    """
    for key, translated_value in translations.items():
        app_idx = key[0]
        field_path = key[1:]

        obj = data["apps"][app_idx]["app_state"]
        for step in field_path[:-1]:
            obj = obj[step]
        obj[field_path[-1]] = translated_value
