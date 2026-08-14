# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Tests for reasoning-model robustness: parse_json_response + Qwen3 detection."""

from __future__ import annotations

import unittest

from gaia2_mt.llm.client import _extra_body_for_model
from gaia2_mt.llm.utils import parse_json_response


class TestParseJsonResponse(unittest.TestCase):
    def test_plain_json(self) -> None:
        self.assertEqual(parse_json_response('{"v": 1}'), {"v": 1})

    def test_markdown_fenced_json(self) -> None:
        self.assertEqual(
            parse_json_response('```json\n{"v": 1}\n```'),
            {"v": 1},
        )

    def test_strip_closed_think_block(self) -> None:
        text = '<think>reasoning here</think>\n{"v": 1}'
        self.assertEqual(parse_json_response(text), {"v": 1})

    def test_strip_multiple_think_blocks(self) -> None:
        text = '<think>a</think>\n<think>b</think>\n{"v": 2}'
        self.assertEqual(parse_json_response(text), {"v": 2})

    def test_unterminated_think_recovers_inner_json(self) -> None:
        # Opening <think> with no </think> is still parseable when the
        # first-brace recovery path finds valid JSON in the tail.
        text = '<think>still thinking, here is the answer {"v": 1}'
        self.assertEqual(parse_json_response(text), {"v": 1})

    def test_unterminated_think_after_closing_block(self) -> None:
        text = '<think>step 1</think><think>step 2{"v": 7}'
        # No second </think> closes the second block; falls through to the
        # whole tail which fails strict JSON parse, then triggers first-{
        # recovery starting at index 0 of "<think>step 2{...}".
        self.assertEqual(parse_json_response(text), {"v": 7})

    def test_first_brace_recovery(self) -> None:
        text = 'preamble noise\n{"v": 3}\ntrailing chatter'
        self.assertEqual(parse_json_response(text), {"v": 3})

    def test_none_input_returns_none(self) -> None:
        self.assertIsNone(parse_json_response(None))

    def test_empty_string_returns_none(self) -> None:
        self.assertIsNone(parse_json_response(""))

    def test_unparseable_returns_none(self) -> None:
        self.assertIsNone(parse_json_response("definitely not json {{{{"))


class TestExtraBodyForModel(unittest.TestCase):
    def test_non_qwen3_returns_none(self) -> None:
        self.assertIsNone(_extra_body_for_model("gpt-5-4-genai-responses"))
        self.assertIsNone(_extra_body_for_model("claude-4-6-opus-tbd"))
        self.assertIsNone(_extra_body_for_model("google/gemma-4-31B-it"))

    def test_qwen3_disables_thinking(self) -> None:
        body = _extra_body_for_model("Qwen/Qwen3-32B")
        self.assertEqual(body, {"chat_template_kwargs": {"enable_thinking": False}})

    def test_qwen3_lowercase_prefix(self) -> None:
        body = _extra_body_for_model("qwen/qwen3-7b-instruct")
        self.assertIsNotNone(body)
        self.assertFalse(body["chat_template_kwargs"]["enable_thinking"])


if __name__ == "__main__":
    unittest.main()
