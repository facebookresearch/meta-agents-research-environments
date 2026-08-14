# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""LLM endpoint routing and model defaults."""

from __future__ import annotations

import json
import logging
import os


logger: logging.Logger = logging.getLogger(__name__)


# Every model is served by a local OpenAI-compatible server (vLLM). There is no
# hosted fallback: point GAIA2_MT_LLM_BASE_URL (or the per-model map below) at
# your own deployment.
DEFAULT_ENDPOINT = "http://localhost:8000/v1"

# The published translator: selected via BOUQuET as the top-ranking
# open-license system across every target language, and the model that built the
# released dataset. Names must match the server's `--served-model-name`.
DEFAULT_TRANSLATION_MODEL = "google/gemma-4-31B-it"

# Only used when the optional review pass is enabled (`--review`). This is the
# cross-family reviewer whose ten-language sweep produced the negative result.
DEFAULT_REVIEW_MODEL = "openai/gpt-oss-120b"

# Base URL of the OpenAI-compatible server serving every model. Pair with
# ``GAIA2_MT_LLM_API_KEY`` for the matching key (vLLM accepts any non-empty
# value). Unset falls back to :data:`DEFAULT_ENDPOINT`.
LLM_BASE_URL_ENV = "GAIA2_MT_LLM_BASE_URL"

# JSON map of ``{model_name: base_url}`` for asymmetric multi-endpoint setups,
# e.g. ``{"google/gemma-4-31B-it": "http://127.0.0.1:8011/v1",
# "openai/gpt-oss-120b": "http://127.0.0.1:8012/v1"}``. A model present in the
# map takes precedence over ``GAIA2_MT_LLM_BASE_URL``; missing models fall
# through to the resolution order below. Unset = behaviour unchanged.
PER_MODEL_ENDPOINTS_ENV = "GAIA2_MT_PER_MODEL_ENDPOINTS"


def _resolve_per_model_endpoints() -> dict[str, str]:
    raw = os.environ.get(PER_MODEL_ENDPOINTS_ENV, "").strip()
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as e:
        raise RuntimeError(
            f"{PER_MODEL_ENDPOINTS_ENV} is not valid JSON: {e}"
        ) from None
    if not isinstance(parsed, dict) or not all(
        isinstance(k, str) and isinstance(v, str) for k, v in parsed.items()
    ):
        raise RuntimeError(
            f"{PER_MODEL_ENDPOINTS_ENV} must be a JSON object of "
            "{model_name: base_url} string pairs."
        )
    return parsed


def resolve_endpoint(model_name: str) -> str:
    """Resolve the API base URL for a given model name.

    Resolution order:

    1. ``GAIA2_MT_PER_MODEL_ENDPOINTS`` — exact model_name match in the JSON
       map (supports asymmetric translator+reviewer setups against separate
       vLLM servers).
    2. ``GAIA2_MT_LLM_BASE_URL`` — single endpoint serving every model.
    3. :data:`DEFAULT_ENDPOINT` — a local vLLM server on the default port.
    """
    per_model = _resolve_per_model_endpoints()
    if model_name in per_model:
        return per_model[model_name].rstrip("/")
    override = os.environ.get(LLM_BASE_URL_ENV)
    if override:
        return override.rstrip("/")
    return DEFAULT_ENDPOINT
