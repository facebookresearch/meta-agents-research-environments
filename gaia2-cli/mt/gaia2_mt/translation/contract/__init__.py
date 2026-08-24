# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""TermTable contract — cross-stage translation coordination.

The three translation stages (universe app_state, prompt, oracle args)
historically operated independently, with no shared policy on which spans must
be translated identically. This caused two failure classes in agent rollouts:

  F1 (oracle harness): proper nouns translated in prompt but oracle eq_checker
      still expects the English literal.
  F2 (translator-side): quoted phrase in user prompt left in English while the
      app_state body the agent is supposed to search through has been translated.

The TermTable is a per-universe map ``source_span → canonical_target_span``
DERIVED from the universe-translator's already-produced output. It is injected
into the existing ``{glossary_section}`` slot of ``TRANSLATION_PROMPT`` /
``ORACLE_ARG_TRANSLATION_PROMPT`` to constrain stages 2 and 3, then a
validator final sweep substitutes any leaked English spans in the assembled
dataset.

End-to-end empirical result: an early prototype of this design measured an 86%
reduction in F2 leaks across 25 scenarios in 3 languages.
"""

from .llm_adapter import OpenAICompatLLMClient
from .term_extractor import (
    TERM_EXTRACTION_PROMPT,
    TERM_EXTRACTION_SYSTEM_PROMPT,
    ExtractedTerm,
    LLMClient,
    extract_terms,
    parse_extraction_response,
)
from .term_table import (
    T_EXTRACTED,
    T_GLOSSARY,
    T_PASSTHROUGH,
    T_PINNED,
    T_QUOTED,
    TermTable,
    align_substring,
    apply_term_table,
    build_term_table,
    detect_passthrough_spans,
    detect_quoted_spans,
    format_term_table_section,
)


__all__ = [
    "TERM_EXTRACTION_PROMPT",
    "TERM_EXTRACTION_SYSTEM_PROMPT",
    "T_EXTRACTED",
    "T_GLOSSARY",
    "T_PASSTHROUGH",
    "T_PINNED",
    "T_QUOTED",
    "ExtractedTerm",
    "LLMClient",
    "OpenAICompatLLMClient",
    "TermTable",
    "align_substring",
    "apply_term_table",
    "build_term_table",
    "detect_passthrough_spans",
    "detect_quoted_spans",
    "extract_terms",
    "format_term_table_section",
    "parse_extraction_response",
]
