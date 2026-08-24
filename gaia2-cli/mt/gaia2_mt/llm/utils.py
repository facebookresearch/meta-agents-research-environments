# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""JSON parsing utilities for LLM responses."""

from __future__ import annotations

import json
import re


# Closed ``<think>…</think>`` blocks emitted by reasoning-tuned models
# (Qwen3, DeepSeek-R1, …) before their actual JSON payload. We strip these
# pre-parse so the upstream JSON parser is unaffected by reasoning traces.
_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL | re.IGNORECASE)


def parse_json_response(text: str | None) -> dict | None:
    """Try to parse a JSON response, handling markdown code blocks.

    Robust to two reasoning-model artefacts:

    * Closed ``<think>…</think>`` traces — stripped pre-parse.
    * Unterminated ``<think>`` (model ran out of budget mid-trace) — we keep
      only the substring after the final ``</think>`` and parse that.

    If the strict parse fails after cleanup, falls back to scanning for the
    first ``{`` and trying ``[{ … }]``-style enclosing slices.
    """
    if text is None:
        return None
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

    cleaned = _THINK_BLOCK_RE.sub("", text)
    if "<think>" in cleaned:
        # Unterminated trace: take everything after the LAST </think>.
        cleaned = cleaned.split("</think>")[-1]

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    start = cleaned.find("{")
    if start < 0:
        return None
    tail = cleaned[start:]
    try:
        return json.loads(tail)
    except json.JSONDecodeError:
        pass
    end = tail.rfind("}")
    if end <= 0:
        return None
    try:
        return json.loads(tail[: end + 1])
    except json.JSONDecodeError:
        return None
