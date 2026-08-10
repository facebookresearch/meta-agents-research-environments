# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""LLM abstraction layer: client, config, and utilities."""

from __future__ import annotations

from omnigaia.llm.client import LlamaApiInferencer
from omnigaia.llm.config import (
    DEFAULT_ENDPOINT,
    DEFAULT_REVIEW_MODEL,
    DEFAULT_TRANSLATION_MODEL,
    resolve_endpoint,
)
from omnigaia.llm.utils import parse_json_response


__all__ = [
    "DEFAULT_ENDPOINT",
    "DEFAULT_REVIEW_MODEL",
    "DEFAULT_TRANSLATION_MODEL",
    "LlamaApiInferencer",
    "parse_json_response",
    "resolve_endpoint",
]
