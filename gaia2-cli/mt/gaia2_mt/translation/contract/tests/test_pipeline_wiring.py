# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Integration test for the term-table contract wiring in build_final_dataset.

Asserts that:
  - With ``scenario_term_tables=None``, build_final_dataset is byte-identical
    to the legacy (no-contract) path.
  - With a TermTable that contains a known leak pattern, the validator sweep
    substitutes the leak in the final prompt + oracle arg strings.

This is an offline test — no LLM, no vLLM. It pins the wiring contract only;
end-to-end translation behaviour needs a live endpoint and is not covered here.
"""

from __future__ import annotations

import json
import unittest

import datasets

from gaia2_mt.translation.contract import T_EXTRACTED, TermTable
from gaia2_mt.translation.pipeline import build_final_dataset


def _make_one_scenario_ds(prompt: str, oracle_arg_value: str) -> datasets.Dataset:
    """Build a 1-row dataset containing a minimal GAIA2-style scenario."""
    data = {
        "events": [
            {
                "action": {
                    "action_id": "send_message_to_agent",
                    "function": "send_message_to_agent",
                    "args": [{"name": "content", "value": prompt}],
                }
            },
            {
                "action": {
                    "action_id": "Calendar.search",
                    "function": "Calendar.search",
                    "args": [{"name": "query", "value": oracle_arg_value}],
                }
            },
        ],
        "apps": [],
    }
    return datasets.Dataset.from_dict({"data": [json.dumps(data)]})


class BuildFinalDatasetContractTest(unittest.TestCase):
    def test_no_contract_path_preserves_legacy_output(self):
        """Without scenario_term_tables, output is unchanged from legacy path."""
        ds = _make_one_scenario_ds(
            "Find emails about 'schedule a call'", "schedule a call"
        )
        translated_prompts = ["Buscar correos sobre 'schedule a call'"]
        oracle_arg_map = {(0, 1, "query"): "schedule a call"}

        # Legacy path: scenario_term_tables=None
        result_legacy = build_final_dataset(
            ds,
            translated_prompts,
            oracle_arg_map=oracle_arg_map,
        )
        # New path with no term tables threaded
        result_new = build_final_dataset(
            ds,
            translated_prompts,
            oracle_arg_map=oracle_arg_map,
            scenario_term_tables=None,
        )
        # Both should produce identical JSON.
        self.assertEqual(result_legacy[0]["data"], result_new[0]["data"])

    def test_validator_sweep_substitutes_leaked_span(self):
        """A leaked English span in the translated prompt should be rewritten
        when the term table contains a canonical target for it."""
        ds = _make_one_scenario_ds(
            "Find emails about 'schedule a call'", "schedule a call"
        )
        # The translator left the English fragment in the Spanish prompt — the
        # canonical F2 regression pattern.
        translated_prompts = ["Buscar correos sobre 'schedule a call'"]
        oracle_arg_map = {(0, 1, "query"): "schedule a call"}

        tt = TermTable()
        tt.add("schedule a call", "programar una llamada", T_EXTRACTED)

        result = build_final_dataset(
            ds,
            translated_prompts,
            oracle_arg_map=oracle_arg_map,
            scenario_term_tables=[tt],
        )
        rewritten = json.loads(result[0]["data"])
        # User prompt was rewritten
        prompt_value = rewritten["events"][0]["action"]["args"][0]["value"]
        self.assertEqual(prompt_value, "Buscar correos sobre 'programar una llamada'")
        # Oracle arg was rewritten
        oracle_value = rewritten["events"][1]["action"]["args"][0]["value"]
        self.assertEqual(oracle_value, "programar una llamada")

    def test_validator_sweep_no_op_on_already_translated_text(self):
        """If the translator already produced the canonical target, the sweep
        is a no-op (no double-substitution)."""
        ds = _make_one_scenario_ds(
            "Find emails about 'schedule a call'", "schedule a call"
        )
        translated_prompts = ["Buscar correos sobre 'programar una llamada'"]
        oracle_arg_map = {(0, 1, "query"): "programar una llamada"}

        tt = TermTable()
        tt.add("schedule a call", "programar una llamada", T_EXTRACTED)

        result = build_final_dataset(
            ds,
            translated_prompts,
            oracle_arg_map=oracle_arg_map,
            scenario_term_tables=[tt],
        )
        rewritten = json.loads(result[0]["data"])
        prompt_value = rewritten["events"][0]["action"]["args"][0]["value"]
        self.assertEqual(prompt_value, "Buscar correos sobre 'programar una llamada'")

    def test_validator_skips_failed_scenarios(self):
        """Scenarios marked as failed are passed through untouched, ignoring
        any term table that might be associated with them."""
        ds = _make_one_scenario_ds(
            "Find emails about 'schedule a call'", "schedule a call"
        )
        translated_prompts = [None]  # translation failed

        tt = TermTable()
        tt.add("schedule a call", "programar una llamada", T_EXTRACTED)

        result = build_final_dataset(
            ds,
            translated_prompts,
            failed_indices={0},
            scenario_term_tables=[tt],
        )
        original = json.loads(ds[0]["data"])
        rewritten = json.loads(result[0]["data"])
        self.assertEqual(original, rewritten)


if __name__ == "__main__":
    unittest.main()
