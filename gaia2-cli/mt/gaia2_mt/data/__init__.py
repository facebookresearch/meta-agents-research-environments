# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Data layer: models, constants, and GAIA2 JSON parsing."""

from __future__ import annotations

from gaia2_mt.data.app_state import (
    APP_TRANSLATABLE_FIELDS,
    GLOSSARY_FIELDS,
    AppStateField,
    apply_app_state_translations,
    compute_universe_hash,
    extract_glossary,
    extract_translatable_app_fields,
)
from gaia2_mt.data.constants import (
    ALL_SUBSETS,
    CODE_SWITCHED_LANGUAGES,
    DIALECT_LANGUAGES,
    LANG_CODE_TO_NAME,
    LID_SKIP_LANGUAGES,
    ROMANIZED_LANGUAGES,
    SKIP_FUNCTIONS,
    TRANSLATABLE_ARG_NAMES,
    get_language_display_name,
    get_special_instructions,
)
from gaia2_mt.data.models import OracleEventArg, ScenarioTranslation, SplitResult
from gaia2_mt.data.parse import (
    extract_all_expected_responses,
    extract_completed_events,
    extract_expected_response,
    extract_initial_prompt,
    extract_oracle_events,
    replace_completed_event_arg,
    replace_event_arg,
)


__all__ = [
    "ALL_SUBSETS",
    "APP_TRANSLATABLE_FIELDS",
    "AppStateField",
    "CODE_SWITCHED_LANGUAGES",
    "DIALECT_LANGUAGES",
    "GLOSSARY_FIELDS",
    "LANG_CODE_TO_NAME",
    "LID_SKIP_LANGUAGES",
    "OracleEventArg",
    "ROMANIZED_LANGUAGES",
    "ScenarioTranslation",
    "SKIP_FUNCTIONS",
    "SplitResult",
    "TRANSLATABLE_ARG_NAMES",
    "apply_app_state_translations",
    "compute_universe_hash",
    "extract_all_expected_responses",
    "extract_completed_events",
    "extract_expected_response",
    "extract_glossary",
    "extract_initial_prompt",
    "extract_oracle_events",
    "extract_translatable_app_fields",
    "get_language_display_name",
    "get_special_instructions",
    "replace_completed_event_arg",
    "replace_event_arg",
]
