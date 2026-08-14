# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""LLM term extractor — Pass A of the term-table contract.

A small LLM pass that runs ONCE per scenario, before the universe / prompt /
oracle stages execute. It identifies source-language phrases that
appear in multiple surfaces of the scenario and would need consistent
target-language rendering across stages.

The LLM-extracted terms are merged into the TermTable with provenance
``T_EXTRACTED``. Pass B (post-universe) overrides each extracted term's target
with the universe-translator's actual rendering when the term's referent is
present in app_state — this is what closes the F2 loop:

   1. extractor flags ``"schedule a call"`` as cross-surface.
   2. universe pass renders the email body that *contains* ``"schedule a call"``.
   3. Pass B extracts ``"programar una llamada"`` from that rendering (either by
      exact substring or via a second small LLM call) and overwrites the hint.
   4. stages 2+3 receive the correct target via the existing ``{glossary_section}``
      slot, and the validator catches any leak.

Costs:

- Pass A: one LLM call per scenario, ~1K-2K input tokens, ~500 output tokens.
  Negligible vs the universe pass's tens of thousands of calls.
- Pass B (LLM fallback): one extra small LLM call per unresolved term — bounded
  by the number of extracted terms whose source is a substring of some
  app_state field but not an exact field value. Empirically ~5-15 per scenario.

This module does NOT import vLLM, HuggingFace, or any heavy dependency. It
defines a thin protocol-based interface so unit tests can swap in a stub LLM.
Production code wires it to ``gaia2_mt.llm.OpenAICompatInferencer``.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Protocol


logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Prompt — designed to be model-agnostic; tested on Gemma-4-31B-it.
# ─────────────────────────────────────────────────────────────────────────────


TERM_EXTRACTION_SYSTEM_PROMPT = """\
You are an expert translation coordinator. Your job is to identify
source-language terms in a scenario that MUST be translated identically
wherever they appear, so that a downstream agent operating on the translated
scenario can search, reference, and equality-check correctly.

Return strict JSON only. No prose, no markdown fences.
"""


TERM_EXTRACTION_PROMPT = """\
You will be given an English scenario consisting of:
- USER_PROMPT: the initial instruction the user gives to an agent.
- ORACLE_REPLIES: expected agent replies (oracle ground truth).
- APP_STATE_SAMPLES: representative text values from the scenario's apps
  (calendar titles, email bodies, contact descriptions, etc.).

Identify terms (entities, named phrases, quoted strings, app-state values
referenced in the prompt) that appear in MULTIPLE surfaces and therefore must
be translated identically across all surfaces in target language {tgt_lang}.

Examples of what to flag:
- A quoted phrase in USER_PROMPT or ORACLE_REPLIES (anything inside single or
  double quotes — e.g. 'schedule a call', "Project Update Meeting",
  "Fintech discussion"). Flag every quoted phrase, even short ones.
- A title/subject/name from APP_STATE referenced literally in USER_PROMPT.
- An entity name (person, place, project) that appears in any 2 of the 3
  surfaces.

Do NOT flag:
- Generic vocabulary ("the email", "next week") that doesn't anchor anything.
- Brand names that are the same in both languages (e.g. "iPhone").

For each flagged term, return JSON with these fields:
- "source": the exact English source string (verbatim from the scenario).
- "surfaces": list of surfaces where it appears, subset of
  ["user_prompt", "oracle_reply", "app_state"].
- "recommended_translation": your best initial target-language translation
  (will be overridden by the actual app_state translation if available).

Return strict JSON in this shape:

{{
  "terms": [
    {{"source": "...", "surfaces": ["user_prompt", "app_state"],
      "recommended_translation": "..."}},
    ...
  ]
}}

If there are no cross-surface terms, return {{"terms": []}}.

──────────────────────────────────────────────────────────────────────────────
SCENARIO

USER_PROMPT:
{user_prompt}

ORACLE_REPLIES:
{oracle_replies}

APP_STATE_SAMPLES:
{app_state_samples}
──────────────────────────────────────────────────────────────────────────────
"""


# ─────────────────────────────────────────────────────────────────────────────
# Data types
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ExtractedTerm:
    source: str
    surfaces: tuple[str, ...]
    recommended_translation: str
    appears_as_substring: bool = False


class LLMClient(Protocol):
    """Minimal protocol — anything that can produce a string from a system +
    user prompt pair works. Production: ``OpenAICompatInferencer``. Tests: stub."""

    def infer_one(self, system: str, prompt: str) -> str | None: ...


# ─────────────────────────────────────────────────────────────────────────────
# Scenario surface assembly
# ─────────────────────────────────────────────────────────────────────────────


def _sample_app_state(
    app_fields: list[tuple[tuple, str]], max_chars: int = 4000
) -> str:
    """Render a representative sample of app_state strings, bounded by
    ``max_chars`` so the prompt stays within budget on big universes."""
    if not app_fields:
        return "(no app_state strings)"
    lines: list[str] = []
    total = 0
    for path, value in app_fields:
        path_str = ".".join(str(p) for p in path)
        line = f"- [{path_str}] {value}"
        if total + len(line) > max_chars:
            lines.append(f"  …(+{len(app_fields) - len(lines)} more)")
            break
        lines.append(line)
        total += len(line)
    return "\n".join(lines)


def _render_oracle_replies(oracle_args: list[tuple[str, str]]) -> str:
    """Show only argname=value pairs whose name is content-like."""
    content_args = [
        v for k, v in oracle_args if k in ("content", "message", "text", "body")
    ]
    if not content_args:
        return "(no oracle replies)"
    return "\n".join(f"- {v}" for v in content_args)


# ─────────────────────────────────────────────────────────────────────────────
# Response parsing
# ─────────────────────────────────────────────────────────────────────────────


_JSON_BLOCK_RE = re.compile(r"\{[\s\S]*\}", re.MULTILINE)


def parse_extraction_response(raw: str | None) -> list[ExtractedTerm]:
    """Parse the LLM response into ExtractedTerms. Robust to slight format drift
    (markdown fences, trailing prose) — strips to the outermost JSON block.

    Returns an empty list on any parse failure (caller logs + degrades to
    glossary-only term table)."""
    if not raw:
        return []
    text = raw.strip()
    # Strip leading/trailing markdown fences if present.
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    # If there's still trailing prose, grab the outermost {...}.
    m = _JSON_BLOCK_RE.search(text)
    if not m:
        return []
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError as e:
        logger.warning("term-extractor: JSON parse failed (%s); raw=%r", e, raw[:200])
        return []
    terms_raw = data.get("terms")
    if not isinstance(terms_raw, list):
        return []
    out: list[ExtractedTerm] = []
    for t in terms_raw:
        if not isinstance(t, dict):
            continue
        src = t.get("source")
        tgt = t.get("recommended_translation")
        surfaces = t.get("surfaces") or []
        if not isinstance(src, str) or not src.strip():
            continue
        if not isinstance(tgt, str) or not tgt.strip():
            continue
        if not isinstance(surfaces, list):
            surfaces = []
        out.append(
            ExtractedTerm(
                source=src.strip(),
                surfaces=tuple(s for s in surfaces if isinstance(s, str)),
                recommended_translation=tgt.strip(),
                appears_as_substring=bool(t.get("appears_as_substring", False)),
            )
        )
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────


def extract_terms(
    *,
    user_prompt: str | None,
    oracle_args: list[tuple[str, str]],
    app_fields: list[tuple[tuple, str]],
    tgt_lang: str,
    llm: LLMClient,
) -> list[ExtractedTerm]:
    """Pass A: ask the LLM to identify cross-surface canonical terms.

    Returns ``[]`` on parse failure or empty input (degrades gracefully — the
    rest of the term-table pipeline still works on T-glossary alone)."""
    if not user_prompt and not oracle_args:
        return []
    prompt = TERM_EXTRACTION_PROMPT.format(
        tgt_lang=tgt_lang,
        user_prompt=user_prompt or "(none)",
        oracle_replies=_render_oracle_replies(oracle_args),
        app_state_samples=_sample_app_state(app_fields),
    )
    raw = llm.infer_one(TERM_EXTRACTION_SYSTEM_PROMPT, prompt)
    return parse_extraction_response(raw)
