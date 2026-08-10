# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""LLM endpoint routing and model defaults."""

from __future__ import annotations

import json
import logging
import os


logger: logging.Logger = logging.getLogger(__name__)


LLAMA_API_BASE = "https://api.llama.com"

ENDPOINT_BY_MODEL_PREFIX: dict[str, str] = {
    "gpt-": f"{LLAMA_API_BASE}/experimental/passthrough/openai/v1",
    "claude-": f"{LLAMA_API_BASE}/compat/v1",
    "gemini-": f"{LLAMA_API_BASE}/compat/v1",
}

DEFAULT_ENDPOINT = f"{LLAMA_API_BASE}/experimental/passthrough/openai/v1"

DEFAULT_TRANSLATION_MODEL = "claude-4-6-opus-tbd"
DEFAULT_REVIEW_MODEL = "gpt-5-4-genai-responses"

# Override base URL for self-hosted / OpenAI-compatible endpoints, e.g. a local
# vLLM server. When ``OMNIGAIA_LLM_BASE_URL`` is set, every model routes to it
# regardless of name prefix. Set this unless you have credentials for the
# hosted default above. Pair with ``OMNIGAIA_LLM_API_KEY`` for the matching key
# (vLLM accepts any non-empty value).
LLM_BASE_URL_ENV = "OMNIGAIA_LLM_BASE_URL"

# JSON map of ``{model_name: base_url}`` for asymmetric multi-endpoint setups,
# e.g. ``{"google/gemma-4-31B-it": "http://127.0.0.1:8011/v1",
# "openai/gpt-oss-120b": "http://127.0.0.1:8012/v1"}``. A model present in the
# map takes precedence over ``OMNIGAIA_LLM_BASE_URL`` and the prefix table;
# missing models fall through to the existing resolution order. Unset =
# behaviour unchanged.
PER_MODEL_ENDPOINTS_ENV = "OMNIGAIA_PER_MODEL_ENDPOINTS"


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

    1. ``OMNIGAIA_PER_MODEL_ENDPOINTS`` — exact model_name match in the JSON
       map (supports asymmetric translator+reviewer setups against separate
       vLLM servers).
    2. ``OMNIGAIA_LLM_BASE_URL`` — single-endpoint override for self-hosted /
       OpenAI-compatible deployments.
    3. :data:`ENDPOINT_BY_MODEL_PREFIX` — Llama-API routing by model-name prefix.
    4. :data:`DEFAULT_ENDPOINT`.
    """
    per_model = _resolve_per_model_endpoints()
    if model_name in per_model:
        return per_model[model_name].rstrip("/")
    override = os.environ.get(LLM_BASE_URL_ENV)
    if override:
        return override.rstrip("/")
    for prefix, endpoint in ENDPOINT_BY_MODEL_PREFIX.items():
        if model_name.startswith(prefix):
            return endpoint
    return DEFAULT_ENDPOINT
