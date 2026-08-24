# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Unit tests for ``gaia2_mt.translation.contract.term_table``.

Pure-Python tests — no LLM calls, no network. Run from ``gaia2-cli/mt`` with:

    pytest gaia2_mt/translation/contract/tests/test_term_table.py
"""

from __future__ import annotations

import os
import unittest

from ..term_table import (
    T_EXTRACTED,
    T_GLOSSARY,
    T_PASSTHROUGH,
    T_PINNED,
    T_QUOTED,
    TERM_TABLE_ENABLE_T_QUOTED_ENV,
    TermTable,
    align_substring,
    apply_term_table,
    build_term_table,
    detect_passthrough_spans,
    detect_quoted_spans,
    format_term_table_section,
)


class TestDetectQuotedSpans(unittest.TestCase):
    def test_finds_simple_quoted_fragment(self) -> None:
        text = "find emails that mention 'schedule a call' please"
        self.assertEqual(detect_quoted_spans(text), ["schedule a call"])

    def test_threshold_3_words_or_15_chars(self) -> None:
        # 2 words, 11 chars -> rejected
        self.assertEqual(detect_quoted_spans("look for 'tiny bit'"), [])
        # 3 words -> accepted
        self.assertEqual(
            detect_quoted_spans("look for 'one two three'"), ["one two three"]
        )
        # 2 words but 17 chars -> accepted
        self.assertEqual(
            detect_quoted_spans("look for 'aaaaaaaa bbbbbbbb'"),
            ["aaaaaaaa bbbbbbbb"],
        )

    def test_ascii_only(self) -> None:
        # The detector is purely syntactic: any quoted ASCII fragment passes,
        # even if the words happen to be Spanish. Semantic filtering is the
        # downstream substring-match-against-app_state's job — false positives
        # at this stage are harmless because they won't match anything.
        text = "busca correos que mencionen 'programar una llamada'"
        out = detect_quoted_spans(text)
        self.assertEqual(out, ["programar una llamada"])

    def test_skips_non_ascii_chars(self) -> None:
        # Quoted fragments containing target-language diacritics ARE skipped
        # (the high-bit char makes it non-ASCII).
        text = "busca correos que mencionen 'programación urgente'"  # ó
        self.assertEqual(detect_quoted_spans(text), [])

    def test_handles_double_quotes(self) -> None:
        text = 'find "schedule a call" mentions'
        self.assertEqual(detect_quoted_spans(text), ["schedule a call"])

    def test_dedup_and_longest_first(self) -> None:
        text = (
            "find 'short fragment one' and 'a longer fragment than the other'"
            " and 'short fragment one' again"
        )
        out = detect_quoted_spans(text)
        self.assertEqual(out[0], "a longer fragment than the other")
        self.assertIn("short fragment one", out)
        self.assertEqual(len(out), 2)

    def test_empty_and_none_safe(self) -> None:
        self.assertEqual(detect_quoted_spans(""), [])

    def test_canonical_f2_case_jvohtk(self) -> None:
        prompt = (
            "Encuentra los correos que mencionen 'schedule a call' y responde "
            "a sus remitentes"
        )
        self.assertEqual(detect_quoted_spans(prompt), ["schedule a call"])

    def test_canonical_f2_case_3gny21(self) -> None:
        prompt = (
            "Reagenda 'Project Update Meeting' y 'Snake Oil Game' para el "
            "viernes que viene"
        )
        spans = detect_quoted_spans(prompt)
        self.assertIn("Project Update Meeting", spans)
        self.assertIn("Snake Oil Game", spans)

    def test_possessive_apostrophe_does_not_swallow_quoted_phrase(self) -> None:
        # v2 BUG: "next week's Tuesday ... 'schedule a call'" matched
        # week's...mention as a fake quoted span, hiding the intended
        # 'schedule a call'. v3 fix: closing quote needs a right boundary.
        prompt = (
            "On next week's Tuesday, reply to the four latest emails "
            "that mention 'schedule a call' with my available slots."
        )
        spans = detect_quoted_spans(prompt)
        self.assertIn("schedule a call", spans)
        # Make sure we DON'T include the bogus 'week's Tuesday...mention' span.
        for s in spans:
            self.assertNotIn("Tuesday", s, f"bogus span leaked: {s!r}")

    def test_possessive_apostrophe_in_middle_of_quoted_phrase_ok(self) -> None:
        # A real apostrophe inside the quoted content is fine.
        prompt = "send the email titled 'don't forget the meeting' please"
        spans = detect_quoted_spans(prompt)
        self.assertIn("don't forget the meeting", spans)


class TestAlignSubstring(unittest.TestCase):
    def test_clean_alignment_spanish(self) -> None:
        src = "let's schedule a call about the launch tomorrow morning"
        tgt = "programemos una llamada sobre el lanzamiento mañana por la mañana"
        out = align_substring(src, tgt, "schedule a call")
        self.assertIsNotNone(out)
        # Must contain the key target words.
        self.assertTrue(
            any(w in out.lower() for w in ("programemos", "llamada", "programar")),
            f"got {out!r}",
        )

    def test_returns_none_when_source_span_not_in_source(self) -> None:
        self.assertIsNone(
            align_substring("source full text", "target full text", "not present")
        )

    def test_empty_inputs(self) -> None:
        self.assertIsNone(align_substring("", "x", "y"))
        self.assertIsNone(align_substring("x", "", "x"))
        self.assertIsNone(align_substring("x", "y", ""))


class TestTermTableAddCap(unittest.TestCase):
    def test_add_skips_identity_and_empty(self) -> None:
        tt = TermTable()
        tt.add("", "x", T_GLOSSARY)
        tt.add("x", "", T_GLOSSARY)
        tt.add("same", "same", T_GLOSSARY)
        self.assertEqual(len(tt), 0)

    def test_first_write_wins_on_source(self) -> None:
        tt = TermTable()
        tt.add("Home", "Casa", T_GLOSSARY)
        tt.add("Home", "Hogar", T_PINNED)  # ignored
        self.assertEqual(tt.entries["Home"], "Casa")
        self.assertEqual(tt.provenance["Home"], T_GLOSSARY)

    def test_cap_keeps_shortest_within_priority(self) -> None:
        # All same class -> shortest source wins (as before).
        tt = TermTable()
        tt.add("short", "s", T_GLOSSARY)
        tt.add("medium term", "mt", T_GLOSSARY)
        tt.add("a much longer term here", "x", T_GLOSSARY)
        capped = tt.cap(2)
        self.assertEqual(set(capped.entries.keys()), {"short", "medium term"})

    def test_cap_prioritises_extracted_over_glossary(self) -> None:
        # T_EXTRACTED must survive even when many T_GLOSSARY entries compete.

        tt = TermTable()
        for i in range(10):
            tt.add(f"glossary{i}", f"gl{i}", T_GLOSSARY)
        tt.add("long-extracted-term", "le", T_EXTRACTED)
        capped = tt.cap(5)
        self.assertIn("long-extracted-term", capped.entries)
        self.assertEqual(len(capped), 5)

    def test_cap_noop_when_under(self) -> None:
        tt = TermTable()
        tt.add("a", "b", T_GLOSSARY)
        self.assertIs(tt.cap(10), tt)


class TestBuildTermTable(unittest.TestCase):
    def test_glossary_entries(self) -> None:
        fields = [
            (
                (0, "events", 0, "title"),
                "Project Update Meeting",
                "Reunión de Actualización",
            ),
            ((1, "items", 0, "subject"), "Welcome", "Bienvenida"),
            # leaf is "content" → not in GLOSSARY_LEAF_FIELDS, must be skipped
            (
                (1, "items", 0, "content"),
                "long body text here",
                "texto largo del cuerpo",
            ),
        ]
        tt = build_term_table(universe_app_fields=fields, prompts=[], oracle_args=[])
        self.assertEqual(
            tt.entries["Project Update Meeting"], "Reunión de Actualización"
        )
        self.assertEqual(tt.entries["Welcome"], "Bienvenida")
        self.assertNotIn("long body text here", tt.entries)

    def test_t_quoted_alignment_end_to_end_opt_in(self) -> None:
        # Projection-based T-quoted is opt-in via env in v3. Without the env
        # flag, the path is dormant — F2 cases are handled by the LLM
        # extractor (see test_term_extractor.py).
        prev = os.environ.get(TERM_TABLE_ENABLE_T_QUOTED_ENV)
        os.environ[TERM_TABLE_ENABLE_T_QUOTED_ENV] = "true"
        try:
            fields = [
                (
                    (0, "items", 0, "content"),
                    "Hi team, let's schedule a call about the launch.",
                    "Hola equipo, programemos una llamada sobre el lanzamiento.",
                ),
            ]
            prompts = ["Find emails that mention 'schedule a call' and reply"]
            tt = build_term_table(
                universe_app_fields=fields, prompts=prompts, oracle_args=[]
            )
            self.assertIn("schedule a call", tt.entries)
            target = tt.entries["schedule a call"]
            self.assertNotEqual(target, "schedule a call")
            self.assertEqual(tt.provenance["schedule a call"], T_QUOTED)
        finally:
            if prev is None:
                del os.environ[TERM_TABLE_ENABLE_T_QUOTED_ENV]
            else:
                os.environ[TERM_TABLE_ENABLE_T_QUOTED_ENV] = prev

    def test_t_quoted_disabled_via_env(self) -> None:
        prev = os.environ.get(TERM_TABLE_ENABLE_T_QUOTED_ENV)
        os.environ[TERM_TABLE_ENABLE_T_QUOTED_ENV] = "false"
        try:
            fields = [
                (
                    (0, "items", 0, "content"),
                    "say schedule a call now",
                    "di programar una llamada ahora",
                ),
            ]
            tt = build_term_table(
                universe_app_fields=fields,
                prompts=["mention 'schedule a call' please"],
                oracle_args=[],
            )
            self.assertNotIn("schedule a call", tt.entries)
        finally:
            if prev is None:
                del os.environ[TERM_TABLE_ENABLE_T_QUOTED_ENV]
            else:
                os.environ[TERM_TABLE_ENABLE_T_QUOTED_ENV] = prev

    def test_t_pinned_re_labels_glossary_entry(self) -> None:
        # When an oracle arg value is also an app_state value AND appears in a
        # prompt, the existing glossary entry is re-labelled T_PINNED so the
        # validator can attribute leaks correctly.
        fields = [((0, "events", 0, "name"), "Home", "Casa")]
        prompts = ["go to Home please"]
        oracle_args = [("location", "Home")]
        tt = build_term_table(
            universe_app_fields=fields, prompts=prompts, oracle_args=oracle_args
        )
        self.assertEqual(tt.entries["Home"], "Casa")  # target still from T-glossary
        self.assertEqual(tt.provenance["Home"], T_PINNED)  # provenance escalated

    def test_t_pinned_no_canonical_target_skipped(self) -> None:
        # When the oracle arg has no app_state match, there's nothing to
        # enforce — the entry is not added. F1 oracle-runner coordination
        # is out of scope per design.md §8.
        prompts = ["from Home to Frejgatan via Stockholm"]
        oracle_args = [("start_location", "Home")]
        tt = build_term_table(
            universe_app_fields=[], prompts=prompts, oracle_args=oracle_args
        )
        self.assertNotIn("Home", tt.entries)

    def test_priority_glossary_target_preserved_pinned_relabels(self) -> None:
        # Same source string in both glossary (app_state) and oracle args.
        # T-glossary's canonical target wins, but provenance is escalated to
        # T_PINNED so the validator knows this is cross-stage-coordinated.
        fields = [((0, "events", 0, "title"), "Home", "Casa")]
        prompts = ["go to Home please"]
        oracle_args = [("location", "Home")]
        tt = build_term_table(
            universe_app_fields=fields, prompts=prompts, oracle_args=oracle_args
        )
        self.assertEqual(tt.entries["Home"], "Casa")
        self.assertEqual(tt.provenance["Home"], T_PINNED)


class TestFormatTermTableSection(unittest.TestCase):
    def test_empty(self) -> None:
        self.assertEqual(format_term_table_section(TermTable()), "")

    def test_renders_lines_compatible_with_glossary_section(self) -> None:
        tt = TermTable()
        tt.add("Home", "Casa", T_GLOSSARY)
        tt.add("Project Meeting", "Reunión del Proyecto", T_QUOTED)
        out = format_term_table_section(tt)
        self.assertIn("### Terminology", out)
        self.assertIn('"Home" → "Casa"', out)
        self.assertIn('"Project Meeting" → "Reunión del Proyecto"', out)

    def test_skips_pinned_placeholders(self) -> None:
        # T-pinned entries with source == target are useless as prompt hints.
        tt = TermTable()
        tt.add("Home", "Home", T_PINNED)  # placeholder
        # The .add filter already drops src==tgt, so this is implicitly tested
        # — but explicitly: format must not emit identity lines.
        self.assertEqual(format_term_table_section(tt), "")


class TestApplyTermTable(unittest.TestCase):
    def test_substitutes_leaked_source_spans(self) -> None:
        tt = TermTable()
        tt.add("schedule a call", "programar una llamada", T_QUOTED)
        text = "envía correos que mencionen 'schedule a call'"
        out, leaks = apply_term_table(text, tt)
        self.assertEqual(leaks, 1)
        self.assertIn("programar una llamada", out)
        self.assertNotIn("schedule a call", out)

    def test_zero_leaks_when_target_already_present(self) -> None:
        tt = TermTable()
        tt.add("schedule a call", "programar una llamada", T_QUOTED)
        text = "envía correos que mencionen 'programar una llamada'"
        out, leaks = apply_term_table(text, tt)
        self.assertEqual(leaks, 0)
        self.assertEqual(out, text)

    def test_word_boundary_safety(self) -> None:
        # Source "call" should not match "called" or "recall".
        tt = TermTable()
        tt.add("call", "llamada", T_QUOTED)
        text = "I was called and had to recall everything"
        out, leaks = apply_term_table(text, tt)
        self.assertEqual(leaks, 0, f"got leaks={leaks}, out={out!r}")

    def test_longest_first_substitution(self) -> None:
        # If both "schedule a call" and "a call" were in the table (defensive),
        # the longer one must win.
        tt = TermTable()
        tt.entries["a call"] = "una llamada"
        tt.provenance["a call"] = T_QUOTED
        tt.entries["schedule a call"] = "programar una llamada"
        tt.provenance["schedule a call"] = T_QUOTED
        text = "mention 'schedule a call' here"
        out, _ = apply_term_table(text, tt)
        self.assertIn("programar una llamada", out)
        self.assertNotIn("una una llamada", out)

    def test_empty_inputs(self) -> None:
        self.assertEqual(apply_term_table("", TermTable()), ("", 0))
        self.assertEqual(apply_term_table("hi", TermTable()), ("hi", 0))


class TestF2CanonicalEndToEnd(unittest.TestCase):
    """End-to-end test recreating the canonical 29_jvohtk failure mode using
    the LLM-extracted path (v3 default — projection aligner deprecated)."""

    def test_jvohtk_quoted_phrase_substituted_via_extracted_term(self) -> None:
        # Simulate Pass A: LLM extracted 'schedule a call' as cross-surface
        # with a recommended translation.
        from ..term_extractor import ExtractedTerm

        extracted = [
            ExtractedTerm(
                source="schedule a call",
                surfaces=("user_prompt", "app_state"),
                recommended_translation="programar una llamada",
                appears_as_substring=True,
            )
        ]
        # Stage 1 output: email body translated.
        fields = [
            (
                (0, "emails", 0, "content"),
                "Hi team, can we schedule a call to discuss the launch?",
                "Hola equipo, ¿podemos programar una llamada para discutir el lanzamiento?",
            ),
        ]
        source_prompt = (
            "Find any emails in my inbox that mention 'schedule a call' and "
            "reply to the sender confirming I am available tomorrow at 10am"
        )
        tt = build_term_table(
            universe_app_fields=fields,
            prompts=[source_prompt],
            oracle_args=[],
            extracted_terms=extracted,
        )
        self.assertIn("schedule a call", tt.entries)
        self.assertEqual(tt.entries["schedule a call"], "programar una llamada")

        # Bad translator output: kept English quote untranslated (F2 mode).
        bad_translated_prompt = (
            "Encuentra correos en mi bandeja que mencionen 'schedule a call' "
            "y responde al remitente confirmando que estoy disponible mañana a las 10am"
        )
        rewritten, leaks = apply_term_table(bad_translated_prompt, tt)
        self.assertEqual(
            leaks, 1, f"expected 1 leak caught, got {leaks}, rewritten={rewritten!r}"
        )
        self.assertNotIn("schedule a call", rewritten)


class TestDetectPassthroughSpans(unittest.TestCase):
    """T_PASSTHROUGH: filenames / identifiers must not be substituted by the
    validator. See judge-validation finding (scenario_universe_23_1hu54e):
    'canadian startups' is a filename, not a translatable phrase."""

    def test_extension_match(self) -> None:
        fields = [((0, "files", 0, "name"), "report.csv")]
        out = detect_passthrough_spans([], fields)
        self.assertIn("report.csv", out)

    def test_snake_case_identifier(self) -> None:
        fields = [((0, "files", 0, "name"), "names_and_achievements")]
        out = detect_passthrough_spans([], fields)
        self.assertIn("names_and_achievements", out)

    def test_kebab_case_identifier(self) -> None:
        fields = [((0, "files", 0, "name"), "quarterly-report")]
        out = detect_passthrough_spans([], fields)
        self.assertIn("quarterly-report", out)

    def test_plain_title_is_not_passthrough(self) -> None:
        # "Project Update Meeting" is a glossary-style title, NOT a filename.
        fields = [((0, "events", 0, "title"), "Project Update Meeting")]
        out = detect_passthrough_spans([], fields)
        self.assertNotIn("Project Update Meeting", out)

    def test_filename_context_in_prompt_with_app_state_match(self) -> None:
        # Canonical real-data failure: file referenced by translatable-looking
        # phrase, but the same phrase also appears in app_state (proving it's
        # a real filename, not generic prose).
        prompts = ["Search the file named canadian startups, read its contents"]
        fields = [
            ((0, "files", 0, "name"), "canadian startups"),
            ((1, "files", 0, "content"), "Some unrelated text"),
        ]
        out = detect_passthrough_spans(prompts, fields)
        self.assertIn("canadian startups", out)

    def test_filename_context_without_app_state_match_skipped(self) -> None:
        # Prompt says 'file named foo' but 'foo' never appears in app_state -->
        # tightener rejects: probably generic prose, not a real filename.
        prompts = ["Send me the file named XYZ later please"]
        fields = [((0, "files", 0, "name"), "report.csv")]  # different file
        out = detect_passthrough_spans(prompts, fields)
        self.assertNotIn("XYZ", out)

    def test_quoted_filename_in_context(self) -> None:
        prompts = ["Open the document named 'meeting-notes-2024'"]
        fields = [((0, "files", 0, "name"), "meeting-notes-2024")]
        out = detect_passthrough_spans(prompts, fields)
        self.assertIn("meeting-notes-2024", out)

    def test_person_name_first_only(self) -> None:
        fields = [
            ((0, "contacts", "c1", "first_name"), "Alessia"),
            ((0, "contacts", "c1", "last_name"), "Ramseyer"),
        ]
        out = detect_passthrough_spans([], fields)
        self.assertIn("Alessia", out)
        self.assertIn("Ramseyer", out)

    def test_person_name_combined(self) -> None:
        # When a contact has both first_name AND last_name, the combined
        # "First Last" form is added too — that's how prompts reference people.
        fields = [
            ((0, "contacts", "c1", "first_name"), "Alessia"),
            ((0, "contacts", "c1", "last_name"), "Ramseyer"),
        ]
        out = detect_passthrough_spans([], fields)
        self.assertIn("Alessia Ramseyer", out)
        self.assertIn("Alessia's", out)  # possessive form

    def test_person_name_alternate_leaves(self) -> None:
        # given_name / family_name alternatives also covered.
        fields = [
            ((0, "contacts", "c1", "given_name"), "Søren"),
            ((0, "contacts", "c1", "family_name"), "Kjær"),
        ]
        out = detect_passthrough_spans([], fields)
        self.assertIn("Søren", out)
        self.assertIn("Søren Kjær", out)

    def test_full_name_leaf(self) -> None:
        fields = [((0, "contacts", "c1", "full_name"), "Alessia Ramseyer")]
        out = detect_passthrough_spans([], fields)
        self.assertIn("Alessia Ramseyer", out)

    def test_non_name_leaf_not_passthrough(self) -> None:
        # "name" leaf (not first_name / last_name / full_name) is used for
        # non-person things (file names, event names) and is NOT auto-passthrough.
        fields = [((0, "events", 0, "name"), "Quarterly Review")]
        out = detect_passthrough_spans([], fields)
        self.assertNotIn("Quarterly Review", out)


class TestBuildTermTableWithPassthrough(unittest.TestCase):
    """Passthrough must suppress translation attempts from other classes."""

    def test_passthrough_blocks_glossary_translation(self) -> None:
        fields = [
            ((0, "files", 0, "name"), "canadian startups", "startups canadienses"),
        ]
        prompts = ["Read the file named canadian startups"]
        tt = build_term_table(
            universe_app_fields=fields,
            prompts=prompts,
            oracle_args=[],
        )
        self.assertEqual(tt.entries["canadian startups"], "canadian startups")
        self.assertEqual(tt.provenance["canadian startups"], T_PASSTHROUGH)

    def test_passthrough_blocks_extracted_translation(self) -> None:
        from ..term_extractor import ExtractedTerm

        extracted = [
            ExtractedTerm(
                source="meeting_notes",
                surfaces=("user_prompt",),
                recommended_translation="notas_reunion",
            )
        ]
        fields = [((0, "files", 0, "name"), "meeting_notes", "meeting_notes")]
        prompts = ["Open meeting_notes please"]
        tt = build_term_table(
            universe_app_fields=fields,
            prompts=prompts,
            oracle_args=[],
            extracted_terms=extracted,
        )
        self.assertEqual(tt.entries["meeting_notes"], "meeting_notes")
        self.assertEqual(tt.provenance["meeting_notes"], T_PASSTHROUGH)

    def test_validator_skips_passthrough_identity(self) -> None:
        fields = [
            ((0, "files", 0, "name"), "canadian startups", "startups canadienses"),
        ]
        prompts = ["Read the file named canadian startups"]
        tt = build_term_table(
            universe_app_fields=fields, prompts=prompts, oracle_args=[]
        )
        text = "Lee el archivo llamado canadian startups y dime"
        out, leaks = apply_term_table(text, tt)
        self.assertEqual(leaks, 0)
        self.assertEqual(out, text)


class TestBuildTermTableWithPersonNamePassthrough(unittest.TestCase):
    def test_person_name_blocks_glossary_translation(self) -> None:
        # Person name is also flagged as glossary leaf 'first_name' — but
        # passthrough wins.
        fields = [
            ((0, "contacts", "c1", "first_name"), "Alessia", "अलेसिया"),
            ((0, "contacts", "c1", "last_name"), "Ramseyer", "रैमसेयर"),
        ]
        tt = build_term_table(universe_app_fields=fields, prompts=[], oracle_args=[])
        # Both names preserved as source — no transliteration in target prompt.
        self.assertEqual(tt.entries["Alessia"], "Alessia")
        self.assertEqual(tt.entries["Ramseyer"], "Ramseyer")
        self.assertEqual(tt.entries["Alessia Ramseyer"], "Alessia Ramseyer")
        self.assertEqual(tt.provenance["Alessia"], T_PASSTHROUGH)


if __name__ == "__main__":
    unittest.main()
