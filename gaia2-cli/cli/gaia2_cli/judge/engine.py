# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
"""Judge LLM engine construction.

Provider-specific request setup lives in ``gaia2-cli`` rather than
``gaia2-core`` so the shared core stays transport-agnostic.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

logger = logging.getLogger(__name__)

_DEFAULT_MAX_RETRIES = 12  # litellm handles exponential backoff internally
_OPENAI_COMPAT_PROVIDERS = frozenset(
    {"openai", "openai-codex", "openai-compat", "openai-completions"}
)


class RateLimitError(Exception):
    """Raised when the LLM API returns 429 after exhausting litellm retries."""


def _is_rate_limit_error(exc: Exception) -> bool:
    """Check if an exception indicates rate limiting (429)."""
    s = str(exc).lower()
    return "429" in s or "ratelimit" in s or "throttl" in s


def create_litellm_engine(
    model: str,
    provider: str | None = None,
    base_url: str | None = None,
    max_retries: int = _DEFAULT_MAX_RETRIES,
    validate: bool = True,
    api_key: str | None = None,
) -> Callable | None:
    """Create an LLM engine using litellm."""
    try:
        import litellm  # noqa: F811
    except ImportError as exc:
        raise RuntimeError(
            "litellm is required for the LLM judge but is not installed. "
            "Install with: pip install 'gaia2-cli[judge]'"
        ) from exc

    # Suppress litellm's verbose INFO logging (cost calculation, wrapper calls)
    logging.getLogger("LiteLLM").setLevel(logging.WARNING)
    logging.getLogger("litellm").setLevel(logging.WARNING)

    # For arbitrary model IDs behind OpenAI-compatible endpoints, prefix the
    # model so LiteLLM uses the OpenAI transport instead of provider inference.
    effective_api_base = base_url
    effective_model = model
    if (
        effective_api_base
        and (provider or "").lower() in _OPENAI_COMPAT_PROVIDERS
        and not model.startswith("openai/")
    ):
        effective_model = f"openai/{model}"

    def engine(messages: list[dict], **kwargs: Any) -> tuple[str | None, dict]:
        """Call the LLM via litellm."""
        try:
            response = litellm.completion(
                model=effective_model,
                messages=messages,
                api_base=effective_api_base,
                api_key=api_key,
                max_retries=max_retries,
                temperature=0,
            )
            content = response.choices[0].message.content
            return content, {"model": model}
        except Exception as exc:
            if _is_rate_limit_error(exc):
                raise RateLimitError(str(exc)) from exc
            logger.error("LLM call failed: %s", exc)
            return None, {"error": str(exc)}

    if validate:
        try:
            probe_text, probe_info = engine([{"role": "user", "content": "Say OK"}])
        except RateLimitError:
            raise
        if probe_text is None:
            raise RuntimeError(
                f"Judge LLM validation failed "
                f"(model={model}, provider={provider}, "
                f"base_url={effective_api_base}): {probe_info}"
            )
        logger.info(
            "Judge LLM validated: model=%s, provider=%s, base_url=%s",
            model,
            provider,
            effective_api_base,
        )

    return engine


__all__ = ["RateLimitError", "create_litellm_engine"]
