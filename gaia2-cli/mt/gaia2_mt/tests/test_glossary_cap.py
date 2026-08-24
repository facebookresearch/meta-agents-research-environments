# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Tests for GAIA2_MT_GLOSSARY_MAX_ENTRIES cap in extract_glossary."""

from __future__ import annotations

import os
import unittest

from gaia2_mt.data.app_state import (
    GLOSSARY_MAX_ENTRIES_ENV,
    AppStateField,
    extract_glossary,
)


def _build_inputs(n: int) -> tuple[dict, dict]:
    """Build n entries with strictly increasing source-string lengths.

    Each field has a unique field_path (so it produces a distinct glossary
    entry) and a source value whose length is monotone in i.
    """
    universe_fields: dict[str, list[AppStateField]] = {"u0": []}
    translations: dict[tuple, str] = {}
    for i in range(n):
        src = f"k{i:04d}_" + "a" * (i + 1)  # length grows with i
        field = AppStateField(
            scenario_idx=0,
            app_idx=0,
            app_name="Calendar",
            field_path=("items", i, "title"),
            field_value=src,
        )
        universe_fields["u0"].append(field)
        translations[(0, "items", i, "title")] = f"T_{i}"
    return universe_fields, {"u0": translations}


class TestGlossaryCap(unittest.TestCase):
    def setUp(self) -> None:
        self._snapshot = os.environ.pop(GLOSSARY_MAX_ENTRIES_ENV, None)

    def tearDown(self) -> None:
        os.environ.pop(GLOSSARY_MAX_ENTRIES_ENV, None)
        if self._snapshot is not None:
            os.environ[GLOSSARY_MAX_ENTRIES_ENV] = self._snapshot

    def test_unset_returns_full_glossary(self) -> None:
        fields, trans = _build_inputs(20)
        g = extract_glossary(fields, trans)
        self.assertEqual(len(g), 20)

    def test_cap_zero_returns_full_glossary(self) -> None:
        os.environ[GLOSSARY_MAX_ENTRIES_ENV] = "0"
        fields, trans = _build_inputs(20)
        g = extract_glossary(fields, trans)
        self.assertEqual(len(g), 20)

    def test_cap_larger_than_glossary_is_noop(self) -> None:
        os.environ[GLOSSARY_MAX_ENTRIES_ENV] = "100"
        fields, trans = _build_inputs(5)
        g = extract_glossary(fields, trans)
        self.assertEqual(len(g), 5)

    def test_cap_keeps_shortest_sources(self) -> None:
        os.environ[GLOSSARY_MAX_ENTRIES_ENV] = "3"
        fields, trans = _build_inputs(20)
        g = extract_glossary(fields, trans)
        self.assertEqual(len(g), 3)
        all_src_lengths = sorted(len(f.field_value) for f in fields["u0"])
        # Every kept source must be no longer than the 3rd-shortest input.
        self.assertLessEqual(max(len(k) for k in g), all_src_lengths[2])

    def test_negative_cap_raises(self) -> None:
        os.environ[GLOSSARY_MAX_ENTRIES_ENV] = "-1"
        fields, trans = _build_inputs(5)
        with self.assertRaises(RuntimeError):
            extract_glossary(fields, trans)

    def test_non_integer_cap_raises(self) -> None:
        os.environ[GLOSSARY_MAX_ENTRIES_ENV] = "not-a-number"
        fields, trans = _build_inputs(5)
        with self.assertRaises(RuntimeError):
            extract_glossary(fields, trans)


if __name__ == "__main__":
    unittest.main()
