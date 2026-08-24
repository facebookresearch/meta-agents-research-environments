# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""TermTable: per-universe shared term lexicon for cross-stage coordination.

Three span classes:

- **T-glossary** — frequent short reference fields (existing ``GLOSSARY_FIELDS``).
  Canonical target = universe-translator's output for that field.
- **T-quoted** — quoted ASCII fragments in prompts/oracle replies that
  substring-match an app_state field. Canonical target = aligned substring of
  the universe-translator's output for that field.
- **T-pinned** — oracle eq_checker args that appear verbatim in a prompt.
  Canonical target = either the universe-translator's translation (if the
  string is also an app_state value) or the prompt-translator's translation
  reused for the oracle side.

The table is **derived** from already-executed translation stages; no extra
LLM call. The one non-trivial piece is T-quoted alignment — pure string ops
(proportional projection + word-boundary expansion), graceful fallback when
alignment fails.

This module has zero external dependencies — fully unit-testable without
vLLM, HuggingFace, or the upstream gaia2_mt package.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field


# Span-class provenance labels. Ordered by priority — when the same source
# string is registered by multiple classes, the FIRST registration wins on
# canonical target, but provenance may be re-labelled (e.g. T_GLOSSARY ->
# T_PINNED, indicating cross-stage role).
T_GLOSSARY = "t_glossary"  # short reference field (title, subject, name, ...)
T_EXTRACTED = "t_extracted"  # LLM term-extractor pass A (hint, may be overridden)
T_QUOTED = "t_quoted"  # quoted ASCII fragment substring-matched into app_state
T_PINNED = "t_pinned"  # oracle eq_checker arg also appearing in prompt
T_PASSTHROUGH = "t_passthrough"  # filename / identifier — must NOT be translated

# Quoted-fragment detector. ASCII-only quoted fragments of either ≥3 words
# OR ≥15 chars are flagged. Quoted = enclosed in single or double quotes
# (straight or curly). Apostrophe-handling: opening quote must be preceded by
# whitespace/punctuation/start-of-string (i.e. NOT a word character), and the
# closing quote must be followed by whitespace/punctuation/end (NOT a word
# character). This eliminates the v2 bug where "next week's Tuesday" matched
# the possessive apostrophe as a closing quote.
_QUOTED_RE = re.compile(
    r"(?:(?<=^)|(?<=[\s\(\[]))"  # left boundary
    r"['\"‘“]"  # opening quote
    r"([A-Za-z][A-Za-z0-9 '\-,\.&\?\!]{4,200}?)"  # content (lazy)
    r"['\"’”]"  # closing quote
    r"(?=$|[\s\.\,\;\:\?\!\)\]])"  # right boundary
)

# Env vars (parity with existing GAIA2_MT_GLOSSARY_MAX_ENTRIES style).
TERM_TABLE_MAX_ENTRIES_ENV = "GAIA2_MT_TERM_TABLE_MAX_ENTRIES"
TERM_TABLE_ENABLE_T_QUOTED_ENV = "GAIA2_MT_TERM_TABLE_ENABLE_T_QUOTED"


def _resolve_max_entries(default: int = 200) -> int:
    raw = os.environ.get(TERM_TABLE_MAX_ENTRIES_ENV, "").strip()
    if not raw:
        return default
    try:
        cap = int(raw)
    except ValueError as e:
        raise RuntimeError(
            f"{TERM_TABLE_MAX_ENTRIES_ENV} must be a non-negative integer, got {raw!r}"
        ) from e
    if cap < 0:
        raise RuntimeError(
            f"{TERM_TABLE_MAX_ENTRIES_ENV} must be a non-negative integer, got {cap}"
        )
    return cap


def _t_quoted_enabled() -> bool:
    """Default OFF in v3 (real-data validation showed projection-aligner
    unreliable). Opt-in via env for ablation."""
    raw = os.environ.get(TERM_TABLE_ENABLE_T_QUOTED_ENV, "false").strip().lower()
    return raw in ("1", "true", "yes", "on")


@dataclass
class TermTable:
    """A per-universe map ``source_span -> canonical_target_span``.

    ``provenance`` records which span class each entry came from (for
    debugging / leak attribution).
    """

    entries: dict[str, str] = field(default_factory=dict)
    provenance: dict[str, str] = field(default_factory=dict)

    def add(self, source: str, target: str, kind: str) -> None:
        """Add an entry. Idempotent on (source, target) — first-write-wins on
        the source key (caller orders classes by desired priority)."""
        if not source or not target or source == target:
            return
        if source in self.entries:
            return
        self.entries[source] = target
        self.provenance[source] = kind

    def __len__(self) -> int:
        return len(self.entries)

    def cap(self, max_entries: int) -> "TermTable":
        """Return a copy capped to ``max_entries`` total entries.

        Cap policy (post 2026-07-07 fix — see qwen MT prompt-translation
        blowup: partial-payload 1737 entries emitted, extrapolated ~7,500
        per prompt, ~247k tokens vs qwen fp8 max_model_len=131072).

        Prior policy exempted T_PASSTHROUGH from the cap on the theory that
        passthroughs are identity entries (src == tgt) filtered out by
        ``format_term_table_section``. That's only partially true: any
        T_PASSTHROUGH entry whose extractor produced a non-identity target
        (e.g. quoted-fragment sub-translations) DID emit into the section,
        and — more importantly — the extractor promoted every translatable
        universe field to T_GLOSSARY, thousands of which are non-identity.
        Net effect: the exemption dropped only ~2% of entries, and the
        section blew past qwen's context. Post-fix: cap ALL entries
        together with the same priority ordering, T_PASSTHROUGH now at
        top priority so they survive the cap when substantive. Matches
        the paper's Omnilingual-GAIA2 appendix (~200-entry cap by shortest source).
        """
        if max_entries <= 0:
            return self
        if len(self.entries) <= max_entries:
            return self
        _PRIORITY = {
            T_PASSTHROUGH: 0,
            T_EXTRACTED: 0,
            T_PINNED: 0,
            T_GLOSSARY: 1,
            T_QUOTED: 2,
        }
        ranked = sorted(
            self.entries.keys(),
            key=lambda k: (_PRIORITY.get(self.provenance[k], 3), len(k)),
        )[:max_entries]
        kept = {k: self.entries[k] for k in ranked}
        kept_prov = {k: self.provenance[k] for k in ranked}
        return TermTable(entries=kept, provenance=kept_prov)


# ─────────────────────────────────────────────────────────────────────────────
# T-quoted: detect quoted ASCII fragments in a string
# ─────────────────────────────────────────────────────────────────────────────


def detect_quoted_spans(text: str) -> list[str]:
    """Return all quoted ASCII fragments in ``text`` that meet the
    ≥3-words-OR-≥15-chars threshold. Returns deduped, longest-first."""
    if not text:
        return []
    out: list[str] = []
    seen: set[str] = set()
    for m in _QUOTED_RE.finditer(text):
        span = m.group(1).strip()
        if not span:
            continue
        words = span.split()
        if len(words) < 3 and len(span) < 15:
            continue
        # ASCII-only (Latin alphabet + common punct). Mixed scripts =
        # already-translated, skip.
        if any(ord(c) > 127 for c in span):
            continue
        if span in seen:
            continue
        seen.add(span)
        out.append(span)
    out.sort(key=len, reverse=True)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# T-quoted: align a source substring to its target counterpart by projection
# ─────────────────────────────────────────────────────────────────────────────


def align_substring(
    source_full: str,
    target_full: str,
    source_span: str,
    window_ratio: float = 0.6,
) -> str | None:
    """Extract the target-language rendering of ``source_span`` from
    ``target_full``, given that ``source_span`` is a substring of
    ``source_full`` and ``target_full`` is the target-language translation of
    ``source_full``.

    Uses proportional character projection + word-boundary expansion. Returns
    ``None`` when alignment confidence is too low (caller should fall back to
    leaving the source span in the term table without a canonical target).
    """
    if not source_full or not target_full or not source_span:
        return None
    src_idx = source_full.find(source_span)
    if src_idx < 0:
        return None
    src_len = len(source_span)
    src_total = len(source_full)
    tgt_total = len(target_full)
    if src_total == 0 or tgt_total == 0:
        return None

    # Proportional projection of the source span midpoint onto the target.
    src_mid = src_idx + src_len / 2
    tgt_mid = int(src_mid * tgt_total / src_total)
    # Project length too (target may be shorter or longer than source).
    proj_len = int(src_len * tgt_total / src_total)
    # Search window: ±window_ratio of projected length around projected midpoint.
    half_window = int(proj_len * (0.5 + window_ratio))
    lo = max(0, tgt_mid - half_window)
    hi = min(tgt_total, tgt_mid + half_window)

    window = target_full[lo:hi]
    # Snap to word boundaries inside the window.
    # Strategy: find a contiguous span of target text that's roughly proj_len
    # characters long, ending and starting on whitespace/punct.
    candidates = _word_boundary_candidates(window, proj_len)
    if not candidates:
        return None
    # Pick the candidate whose length is closest to proj_len.
    best = min(candidates, key=lambda c: abs(len(c) - proj_len))
    return best.strip()


def _word_boundary_candidates(text: str, target_len: int) -> list[str]:
    """Generate substring candidates of roughly ``target_len`` chars from
    ``text``, aligned to word boundaries."""
    if not text:
        return []
    # Tolerance: ±50% of target length, with a floor of 8 chars.
    min_len = max(8, int(target_len * 0.5))
    max_len = max(min_len + 1, int(target_len * 1.8))
    # Word-boundary positions.
    boundaries = [0]
    for m in re.finditer(r"\s+", text):
        boundaries.append(m.end())
    boundaries.append(len(text))
    candidates: list[str] = []
    for i, start in enumerate(boundaries):
        for end in boundaries[i + 1 :]:
            slen = end - start
            if slen < min_len:
                continue
            if slen > max_len:
                break
            cand = text[start:end].strip(" ,.;:!?\"'")
            if cand:
                candidates.append(cand)
    return candidates


# ─────────────────────────────────────────────────────────────────────────────
# Passthrough detector (T_PASSTHROUGH): identify spans that look like
# filenames / identifiers / programmatic strings and MUST NOT be translated.
# ─────────────────────────────────────────────────────────────────────────────


# A literal extension suffix (e.g. ".csv", ".docx", ".tar.gz" handled by shape match)
_EXTENSION_RE = re.compile(r"\.[A-Za-z0-9]{2,5}$")

# snake_case, kebab-case, or mixed identifier (``my_file``, ``quarterly-report``,
# ``meeting-notes-2024``). Requires a leading ASCII letter and at least one
# separator that is followed by more characters.
#
# The naive spelling ``^[A-Za-z][\w]*(?:[_\-][\w]+)+$`` is a ReDoS hazard:
# ``\w`` already contains ``_``, so ``[\w]*`` and ``[_\-][\w]+`` overlap and
# every ``_`` can either extend the previous run or open a new group. That's
# 2**(number of underscores) distinct parses, all of which the engine explores
# before rejecting (e.g. ``"A-" + "0_" * 30 + "!"`` takes minutes).
#
# The form below is unambiguous, so each branch is linear in the input:
#   * ``\w*(?:-\w+)+``  — kebab-ish: ``-`` is NOT in ``\w``, so the ``\w`` runs
#     and the ``-`` delimiters cannot compete for the same character.
#   * ``[^\W_]*_\w+``   — underscore-only: ``[^\W_]`` is ``\w`` minus ``_``, so
#     the prefix cannot swallow the first ``_``.
# The branches are mutually exclusive (one requires a ``-``, the other forbids
# it), and together they accept exactly the same language as the old pattern
# (verified by exhaustive comparison over short strings + fuzzing).
_IDENTIFIER_SHAPE_RE = re.compile(r"^[A-Za-z](?:\w*(?:-\w+)+|[^\W_]*_\w+)$")

# Context: "file/folder/document/attachment NAMED/CALLED/TITLED/LABELED X"
# Captures both quoted and unquoted X (up to 80 chars). Greedy until a hard
# terminator (comma, period, semicolon, conjunction word, closing bracket,
# end-of-line). Lazy + space-terminator was too narrow ('canadian startups'
# would clip at the space).
_FILENAME_CONTEXT_RE = re.compile(
    r"\b(?:file|folder|directory|attachment|document|sheet|spreadsheet)s?\s+"
    r"(?:named|called|titled|labeled)\s+"
    r"(?:"
    r"['\"]([A-Za-z][^'\"]{1,80}?)['\"]"  # quoted
    r"|([A-Za-z][\w\s\-\.]{0,79}?)(?=[,.;:!?\)\]]|\s+(?:and|or|then|please|with|to|in|for|so|but)\b|$)"  # unquoted, end at hard terminator or conjunction
    r")",
    re.IGNORECASE,
)


def detect_passthrough_spans(
    prompts: list[str],
    app_fields: list[tuple[tuple, str]] | list[tuple[tuple, str, str]],
) -> set[str]:
    """Return source spans that should be passthrough (no translation).

    Patterns:
      1. Extension suffix (``.csv``, ``.docx``, ...)
      2. snake_case / kebab-case identifier (``my_file``, ``quarterly-report``)
      3. Filename context in prompts (``file named X``) AND ``X`` appears
         verbatim in app_state (tightener — avoids suppressing generic prose).
      4. Person names from Contacts app_state (``first_name``, ``last_name``,
         ``full_name`` leaf fields). Latin-script names that downstream agents
         identify by entity-matching must NOT be translated/transliterated.
         Examples (from judge-validation failures): ``Alessia Ramseyer``,
         ``Smith``, ``Søren Kjær``.

    Returns a set of source strings to register as ``T_PASSTHROUGH``.
    """
    out: set[str] = set()

    # Normalize app_fields to (path, source) for uniform iteration; tolerate
    # the (path, src, tgt) triples used by build_term_table.
    app_values: set[str] = set()
    paths_and_values: list[tuple[tuple, str]] = []
    for row in app_fields:
        if len(row) == 2:
            path, v = row
        else:
            path, v, _ = row
        if isinstance(v, str) and v.strip():
            app_values.add(v.strip())
            paths_and_values.append((path, v.strip()))

    # Patterns 1 + 2: shape match on app_state values.
    for v in app_values:
        if _EXTENSION_RE.search(v) or _IDENTIFIER_SHAPE_RE.match(v):
            out.add(v)

    # Pattern 3: context-introduced spans in prompts that ALSO appear in
    # app_state (verbatim).
    for p in prompts:
        if not p:
            continue
        for m in _FILENAME_CONTEXT_RE.finditer(p):
            cand = (m.group(1) or m.group(2) or "").strip().rstrip(",.;:!?")
            if not cand:
                continue
            if cand in app_values or any(cand in v for v in app_values):
                out.add(cand)

    # Pattern 4: person names from Contacts app_state. Leaf field name
    # signals identity-type values. Also collect "first last" combos by
    # joining first_name + last_name within the same contact.
    name_leaves = {"first_name", "last_name", "full_name", "given_name", "family_name"}
    by_contact: dict[tuple, dict[str, str]] = {}
    for path, v in paths_and_values:
        if not path:
            continue
        leaf = str(path[-1])
        if leaf in name_leaves:
            out.add(v)
            # Group by contact (path up to one level above the leaf).
            contact_key = tuple(path[:-1])
            by_contact.setdefault(contact_key, {})[leaf] = v
    # Combine first + last (and given + family) per contact.
    for contact_fields in by_contact.values():
        first = contact_fields.get("first_name") or contact_fields.get("given_name")
        last = contact_fields.get("last_name") or contact_fields.get("family_name")
        if first and last:
            out.add(f"{first} {last}")
            # Possessive form ("Alice's") commonly appears in prompts.
            out.add(f"{first}'s")

    return out


# ─────────────────────────────────────────────────────────────────────────────
# Builder: assemble TermTable from already-computed stage outputs
# ─────────────────────────────────────────────────────────────────────────────


# Field names whose leaf is considered glossary-worthy (mirrors upstream
# ``data/app_state.py::GLOSSARY_FIELDS``).
GLOSSARY_LEAF_FIELDS: set[str] = {
    "title",
    "subject",
    "tag",
    "name",
    "job",
    "property_type",
}


def build_term_table(
    *,
    universe_app_fields: list[tuple[tuple, str, str]],
    prompts: list[str],
    oracle_args: list[tuple[str, str]],
    extracted_terms: list | None = None,
    max_entries: int | None = None,
) -> TermTable:
    """Build a TermTable from the universe pass's outputs + per-scenario texts.

    Args:
        universe_app_fields: list of ``((app_idx, *field_path), source_value,
            target_value)`` for every translated app_state field in the
            universe.
        prompts: list of source-language user prompts in the scenarios that
            belong to this universe.
        oracle_args: list of ``(arg_name, source_value)`` for every oracle arg
            in the scenarios.
        extracted_terms: optional list of ``term_extractor.ExtractedTerm`` from
            the LLM Pass A. If supplied, the builder runs Pass B (post-universe
            override): each extracted term whose source string is an exact
            value of some app_state field gets its target overridden with the
            universe-translator's actual rendering for that field. Extracted
            terms not resolved this way keep the LLM's
            ``recommended_translation`` as the canonical target.
        max_entries: cap; defaults to env-resolved ``GAIA2_MT_TERM_TABLE_MAX_ENTRIES``.

    Returns:
        A populated TermTable. Priority (first-write-wins on source key):
        T_GLOSSARY → T_EXTRACTED → T_QUOTED → T_PINNED.
    """
    if max_entries is None:
        max_entries = _resolve_max_entries()

    tt = TermTable()

    # ── T-passthrough: filenames / identifiers (highest priority — registered
    #     first so first-write-wins suppresses any later translation attempt).
    #     ``.add()`` short-circuits src == tgt, so a passthrough entry creates
    #     no row but reserves the source key. We bypass that by using
    #     ``_register_passthrough`` which DOES insert the identity row, so
    #     subsequent calls to ``.add(src, anything_else, ...)`` see the source
    #     as already-present and skip.
    passthrough = detect_passthrough_spans(prompts, universe_app_fields)
    for src in passthrough:
        # Manual insert: bypass .add()'s identity skip so the entry blocks
        # future overrides. apply_term_table will then skip on src == tgt
        # (identity substitution is a no-op).
        if src not in tt.entries:
            tt.entries[src] = src
            tt.provenance[src] = T_PASSTHROUGH

    # ── T-glossary: short reference fields, source -> target.
    for key, src_val, tgt_val in universe_app_fields:
        leaf = str(key[-1]) if key else ""
        if leaf not in GLOSSARY_LEAF_FIELDS:
            continue
        tt.add(src_val, tgt_val, T_GLOSSARY)

    # ── T-extracted: LLM-flagged cross-surface terms (Pass A).
    #     Pass B: resolve the canonical target from the universe-translator's
    #     actual output. Two cases:
    #       (a) Exact app_state value -> use that field's target directly.
    #       (b) Substring of a longer app_state field -> we know the LLM said
    #           "this English fragment is a substring of host_field"; the LLM
    #           ALSO produced a recommended_translation hint. We trust the
    #           hint for substring cases (the host field's target is too long
    #           to safely extract from without an aligner). The hint comes
    #           from the SAME model that translated the host field, so it's
    #           more likely than not to be consistent.
    #     This is the F2 fix: the LLM gives us WHICH spans need enforcement;
    #     the hint provides the target rendering when no exact match exists.
    if extracted_terms:
        app_state_by_source: dict[str, str] = {}
        for _, src, tgt in universe_app_fields:
            if src and tgt:
                app_state_by_source.setdefault(src, tgt)
        for term in extracted_terms:
            src = term.source
            override = app_state_by_source.get(src)
            target = override if override else term.recommended_translation
            tt.add(src, target, T_EXTRACTED)

    # ── T-quoted: DEPRECATED in v3 (real-data validation showed proportional
    #     projection produces too many wrong alignments, e.g. 'doctoral
    #     candidate' -> 'bien. Como candidato'). Kept under an opt-in env flag
    #     for ablation studies; default is OFF. The T-extracted path (LLM
    #     Pass A) handles the substring-of-app-state case instead.
    if _t_quoted_enabled():
        src_to_target: list[tuple[str, str]] = [
            (s, t) for _, s, t in universe_app_fields if s and t
        ]
        for prompt in prompts:
            if not prompt:
                continue
            for span in detect_quoted_spans(prompt):
                # Find an app_state source that contains the span (longest
                # containing source preferred — most context for alignment).
                hosts = [(s, t) for s, t in src_to_target if span in s]
                if not hosts:
                    continue
                hosts.sort(key=lambda st: len(st[0]), reverse=True)
                src_full, tgt_full = hosts[0]
                aligned = align_substring(src_full, tgt_full, span)
                if aligned:
                    tt.add(span, aligned, T_QUOTED)

    # ── T-pinned: oracle arg values shared with some prompt. We only attach
    #     an entry when there's a canonical target to enforce — usually that
    #     means T-glossary already covered it. The downstream first-write-wins
    #     in .add() means a glossary entry already exists; we re-label the
    #     provenance to T_PINNED so the validator can attribute leaks.
    #     Spans without a canonical target are deferred (F1 oracle-runner issue,
    #     out of scope per design.md §8).
    prompt_blob = "\n".join(p for p in prompts if p)
    for _, arg_val in oracle_args:
        if not arg_val or arg_val not in prompt_blob:
            continue
        if arg_val in tt.entries:
            # Re-label provenance so leak attribution sees this as cross-stage.
            tt.provenance[arg_val] = T_PINNED

    return tt.cap(max_entries)


# ─────────────────────────────────────────────────────────────────────────────
# Formatter: render TermTable into the existing {glossary_section} prompt slot
# ─────────────────────────────────────────────────────────────────────────────


def format_term_table_section(term_table: TermTable) -> str:
    """Render the term table as a prompt section.

    Mirrors ``gaia2_mt.translation.translate.format_glossary_section`` so the
    output drops directly into the existing ``{glossary_section}`` slot of
    ``TRANSLATION_PROMPT`` / ``ORACLE_ARG_TRANSLATION_PROMPT``.
    """
    if not term_table.entries:
        return ""
    lines = [
        f'- "{src}" → "{tgt}"'
        for src, tgt in term_table.entries.items()
        if src != tgt  # T-pinned placeholders aren't useful as prompt hints
    ]
    if not lines:
        return ""
    return (
        "\n### Terminology (use these exact translations for the following terms):\n"
        + "\n".join(lines)
        + "\n"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Validator: substitute leaks in post-stage outputs
# ─────────────────────────────────────────────────────────────────────────────


def apply_term_table(text: str, term_table: TermTable) -> tuple[str, int]:
    """Substitute any TermTable source span that appears verbatim in ``text``
    with its canonical target. Returns ``(rewritten_text, leak_count)``.

    Order: longest source first, so that ``"schedule a call"`` is substituted
    before ``"a call"`` would be (defensive — substrings shouldn't both be in
    the table but we cap the risk anyway).
    """
    if not text or not term_table.entries:
        return text, 0
    leaks = 0
    out = text
    sorted_entries = sorted(
        term_table.entries.items(), key=lambda kv: len(kv[0]), reverse=True
    )
    for src, tgt in sorted_entries:
        if src == tgt:  # T-pinned placeholder, nothing to substitute
            continue
        # Word-boundary substitute to avoid partial-word replacements.
        pattern = re.compile(rf"(?<![A-Za-z0-9_]){re.escape(src)}(?![A-Za-z0-9_])")
        new_out, n = pattern.subn(tgt, out)
        if n:
            leaks += n
            out = new_out
    return out, leaks
