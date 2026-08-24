# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""LLM abstraction layer: client, config, and utilities."""

from __future__ import annotations

from gaia2_mt.llm.client import OpenAICompatInferencer
from gaia2_mt.llm.config import (
    DEFAULT_ENDPOINT,
    DEFAULT_REVIEW_MODEL,
    DEFAULT_TRANSLATION_MODEL,
    resolve_endpoint,
)
from gaia2_mt.llm.utils import parse_json_response


__all__ = [
    "DEFAULT_ENDPOINT",
    "DEFAULT_REVIEW_MODEL",
    "DEFAULT_TRANSLATION_MODEL",
    "OpenAICompatInferencer",
    "parse_json_response",
    "resolve_endpoint",
]
