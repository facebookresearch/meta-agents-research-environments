# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Unit tests for translation_contract.term_extractor.

Stub LLM client — no network, no vLLM dependency. Run with:

    python -m unittest tools.translation_contract.tests.test_term_extractor
"""

from __future__ import annotations

import unittest

from ..term_extractor import (
    ExtractedTerm,
    extract_terms,
    parse_extraction_response,
)
from ..term_table import T_EXTRACTED, T_GLOSSARY, build_term_table


class _StubLLM:
    """Captures the prompts sent and returns a canned response."""

    def __init__(self, response: str | None) -> None:
        self.response = response
        self.last_system: str | None = None
        self.last_prompt: str | None = None

    def infer_one(self, system: str, prompt: str) -> str | None:
        self.last_system = system
        self.last_prompt = prompt
        return self.response


class TestParseExtractionResponse(unittest.TestCase):
    def test_clean_json(self) -> None:
        raw = (
            '{"terms": [{"source": "schedule a call", '
            '"surfaces": ["user_prompt", "app_state"], '
            '"recommended_translation": "programar una llamada"}]}'
        )
        out = parse_extraction_response(raw)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].source, "schedule a call")
        self.assertEqual(out[0].recommended_translation, "programar una llamada")
        self.assertEqual(out[0].surfaces, ("user_prompt", "app_state"))

    def test_strips_markdown_fences(self) -> None:
        raw = (
            "```json\n"
            '{"terms": [{"source": "Home", "surfaces": ["user_prompt"], '
            '"recommended_translation": "Casa"}]}\n'
            "```"
        )
        out = parse_extraction_response(raw)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].source, "Home")

    def test_handles_trailing_prose(self) -> None:
        raw = (
            'Here are the terms: {"terms": [{"source": "X", "surfaces": '
            '["app_state"], "recommended_translation": "Y"}]}\n\n'
            "I hope this helps!"
        )
        out = parse_extraction_response(raw)
        self.assertEqual(len(out), 1)

    def test_empty_terms_list(self) -> None:
        out = parse_extraction_response('{"terms": []}')
        self.assertEqual(out, [])

    def test_garbage_returns_empty(self) -> None:
        self.assertEqual(parse_extraction_response(None), [])
        self.assertEqual(parse_extraction_response(""), [])
        self.assertEqual(parse_extraction_response("not json at all"), [])
        # Valid JSON but wrong shape.
        self.assertEqual(parse_extraction_response('{"foo": "bar"}'), [])

    def test_skips_invalid_entries(self) -> None:
        raw = (
            '{"terms": ['
            '{"source": "ok", "surfaces": ["x"], "recommended_translation": "y"},'
            '{"source": "", "surfaces": [], "recommended_translation": "y"},'  # empty src
            '{"source": "no_tgt", "surfaces": ["x"], "recommended_translation": ""},'  # empty tgt
            '{"not_a_term": true}'  # missing fields
            "]}"
        )
        out = parse_extraction_response(raw)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].source, "ok")


class TestExtractTerms(unittest.TestCase):
    def test_passes_prompt_and_returns_parsed(self) -> None:
        stub = _StubLLM(
            response=(
                '{"terms": [{"source": "Project Update Meeting", '
                '"surfaces": ["user_prompt", "app_state"], '
                '"recommended_translation": "Reunión de Actualización del Proyecto"}]}'
            )
        )
        out = extract_terms(
            user_prompt="Reschedule the 'Project Update Meeting' to tomorrow",
            oracle_args=[("content", "Done. Rescheduled it.")],
            app_fields=[((0, "events", 0, "title"), "Project Update Meeting")],
            tgt_lang="Spanish",
            llm=stub,
        )
        self.assertEqual(len(out), 1)
        self.assertIn("Project Update Meeting", stub.last_prompt)
        self.assertIn("Spanish", stub.last_prompt)
        self.assertIn("Done. Rescheduled it.", stub.last_prompt)

    def test_no_prompt_no_oracle_returns_empty(self) -> None:
        stub = _StubLLM(response="should not be called")
        out = extract_terms(
            user_prompt=None,
            oracle_args=[],
            app_fields=[],
            tgt_lang="Spanish",
            llm=stub,
        )
        self.assertEqual(out, [])
        # LLM was never called.
        self.assertIsNone(stub.last_prompt)

    def test_llm_returns_none_returns_empty(self) -> None:
        stub = _StubLLM(response=None)
        out = extract_terms(
            user_prompt="anything",
            oracle_args=[],
            app_fields=[],
            tgt_lang="Spanish",
            llm=stub,
        )
        self.assertEqual(out, [])

    def test_app_state_bounded_in_prompt(self) -> None:
        # Many app fields — the prompt must truncate, not include all.
        many = [
            ((0, "items", i, "content"), f"long field content number {i} " * 20)
            for i in range(500)
        ]
        stub = _StubLLM(response='{"terms": []}')
        extract_terms(
            user_prompt="x",
            oracle_args=[],
            app_fields=many,
            tgt_lang="Spanish",
            llm=stub,
        )
        # Prompt must mention truncation and be bounded.
        self.assertIsNotNone(stub.last_prompt)
        # Bound: APP_STATE_SAMPLES section is 4000 chars; total prompt with the
        # rest of the template should be under ~6KB.
        self.assertLess(len(stub.last_prompt), 8000)


class TestBuildTermTableWithExtractedTerms(unittest.TestCase):
    """Pass B: LLM-extracted terms get overridden with the universe-translator's
    actual rendering when their source is an app_state value."""

    def test_pass_b_overrides_with_universe_rendering(self) -> None:
        # LLM extracted a term with hint "Reunión del Proyecto" but the
        # universe pass actually translated the matching app_state title as
        # "Reunión de Actualización del Proyecto". Pass B must override.
        extracted = [
            ExtractedTerm(
                source="Project Update Meeting",
                surfaces=("user_prompt", "app_state"),
                recommended_translation="Reunión del Proyecto",  # LLM's hint
            )
        ]
        universe = [
            (
                (0, "events", 0, "title"),
                "Project Update Meeting",
                "Reunión de Actualización del Proyecto",  # universe pass output
            )
        ]
        tt = build_term_table(
            universe_app_fields=universe,
            prompts=[],
            oracle_args=[],
            extracted_terms=extracted,
        )
        # T-glossary wrote first → its target wins; T-extracted is no-op.
        self.assertEqual(
            tt.entries["Project Update Meeting"],
            "Reunión de Actualización del Proyecto",
        )
        self.assertEqual(tt.provenance["Project Update Meeting"], T_GLOSSARY)

    def test_extracted_only_when_no_glossary_match(self) -> None:
        # LLM flags a span that's NOT a glossary leaf (e.g. quoted phrase
        # inside an email body). T-glossary doesn't fire; T-extracted does.
        extracted = [
            ExtractedTerm(
                source="schedule a call",
                surfaces=("user_prompt", "app_state"),
                recommended_translation="programar una llamada",
            )
        ]
        # Email body in app_state — leaf is "content", NOT a glossary leaf.
        universe = [
            (
                (0, "items", 0, "content"),
                "Hi, can we schedule a call about the launch?",
                "Hola, ¿podemos programar una llamada sobre el lanzamiento?",
            )
        ]
        tt = build_term_table(
            universe_app_fields=universe,
            prompts=[],
            oracle_args=[],
            extracted_terms=extracted,
        )
        self.assertIn("schedule a call", tt.entries)
        self.assertEqual(tt.entries["schedule a call"], "programar una llamada")
        self.assertEqual(tt.provenance["schedule a call"], T_EXTRACTED)

    def test_pass_b_override_when_extracted_source_is_app_state_value(self) -> None:
        # The LLM hint is wrong; the universe pass renders the exact source
        # differently. Pass B must use the universe's rendering.
        extracted = [
            ExtractedTerm(
                source="Banking industry discussion",
                surfaces=("user_prompt", "app_state"),
                recommended_translation="Discusión de la industria bancaria",
            )
        ]
        universe = [
            # Not a glossary leaf — comes via Pass B's exact-source-match path.
            (
                (0, "chats", 0, "topic"),
                "Banking industry discussion",
                "Charla sobre el sector bancario",  # universe pass said this
            )
        ]
        tt = build_term_table(
            universe_app_fields=universe,
            prompts=[],
            oracle_args=[],
            extracted_terms=extracted,
        )
        # Pass B override: universe rendering wins over LLM hint.
        self.assertEqual(
            tt.entries["Banking industry discussion"],
            "Charla sobre el sector bancario",
        )
        self.assertEqual(tt.provenance["Banking industry discussion"], T_EXTRACTED)


if __name__ == "__main__":
    unittest.main()
