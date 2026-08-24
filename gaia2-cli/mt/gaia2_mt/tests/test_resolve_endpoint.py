# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Tests for resolve_endpoint, including GAIA2_MT_PER_MODEL_ENDPOINTS."""

from __future__ import annotations

import os
import unittest

from gaia2_mt.llm.config import (
    DEFAULT_ENDPOINT,
    LLM_BASE_URL_ENV,
    PER_MODEL_ENDPOINTS_ENV,
    resolve_endpoint,
)


class TestResolveEndpoint(unittest.TestCase):
    def setUp(self) -> None:
        self._snapshots = {
            k: os.environ.pop(k, None)
            for k in (LLM_BASE_URL_ENV, PER_MODEL_ENDPOINTS_ENV)
        }

    def tearDown(self) -> None:
        for k, v in self._snapshots.items():
            os.environ.pop(k, None)
            if v is not None:
                os.environ[k] = v

    def test_falls_back_to_local_default(self) -> None:
        self.assertEqual(resolve_endpoint("google/gemma-4-31B-it"), DEFAULT_ENDPOINT)

    def test_routing_is_independent_of_model_name(self) -> None:
        for name in ("google/gemma-4-31B-it", "openai/gpt-oss-120b", "anything"):
            self.assertEqual(resolve_endpoint(name), DEFAULT_ENDPOINT)

    def test_base_url_override(self) -> None:
        os.environ[LLM_BASE_URL_ENV] = "http://127.0.0.1:8000/v1/"
        self.assertEqual(
            resolve_endpoint("google/gemma-4-31B-it"), "http://127.0.0.1:8000/v1"
        )

    def test_per_model_takes_precedence_over_base_url(self) -> None:
        os.environ[LLM_BASE_URL_ENV] = "http://127.0.0.1:8000/v1"
        os.environ[PER_MODEL_ENDPOINTS_ENV] = (
            '{"google/gemma-4-31B-it": "http://127.0.0.1:8011/v1",'
            ' "openai/gpt-oss-120b": "http://127.0.0.1:8012/v1"}'
        )
        self.assertEqual(
            resolve_endpoint("google/gemma-4-31B-it"), "http://127.0.0.1:8011/v1"
        )
        self.assertEqual(
            resolve_endpoint("openai/gpt-oss-120b"), "http://127.0.0.1:8012/v1"
        )

    def test_per_model_falls_through_for_missing_keys(self) -> None:
        os.environ[LLM_BASE_URL_ENV] = "http://127.0.0.1:8000/v1"
        os.environ[PER_MODEL_ENDPOINTS_ENV] = (
            '{"google/gemma-4-31B-it": "http://127.0.0.1:8011/v1"}'
        )
        # Missing model falls through to GAIA2_MT_LLM_BASE_URL.
        self.assertEqual(
            resolve_endpoint("openai/gpt-oss-120b"), "http://127.0.0.1:8000/v1"
        )

    def test_per_model_invalid_json_raises(self) -> None:
        os.environ[PER_MODEL_ENDPOINTS_ENV] = "{not valid json"
        with self.assertRaises(RuntimeError):
            resolve_endpoint("any")

    def test_per_model_non_string_value_raises(self) -> None:
        os.environ[PER_MODEL_ENDPOINTS_ENV] = '{"model": 1234}'
        with self.assertRaises(RuntimeError):
            resolve_endpoint("model")

    def test_per_model_empty_string_is_treated_as_unset(self) -> None:
        os.environ[PER_MODEL_ENDPOINTS_ENV] = ""
        self.assertEqual(resolve_endpoint("google/gemma-4-31B-it"), DEFAULT_ENDPOINT)


if __name__ == "__main__":
    unittest.main()
