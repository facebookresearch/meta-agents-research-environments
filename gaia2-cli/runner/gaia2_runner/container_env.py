# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Container environment profiles for different container runtimes.

Maps container image names to the env-var naming conventions each
runtime expects (provider key, API key, model key, base URL keys, etc.).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Mapping


@dataclass(frozen=True)
class ContainerProfile:
    """Env-var naming profile for a container runtime.

    Agent containers expect their provider, API key, model, and base URL under
    slightly different env-var names. This dataclass captures those
    differences so callers can build ``-e KEY=VALUE`` pairs generically.
    """

    provider_key: str
    """Env var name for the LLM provider (e.g. ``PROVIDER``)."""

    api_key_key: str
    """Env var name for the API key (e.g. ``API_KEY``)."""

    model_key: str
    """Env var name for the model identifier (e.g. ``MODEL``)."""

    default_provider: str
    """Provider value used when the caller doesn't specify one."""

    extra_flags: dict[str, str] = field(default_factory=dict)
    """Additional env vars always set for this profile (e.g. ``OPENCLAW_FORCE_RECONFIG``)."""

    base_url_keys: list[str] = field(default_factory=list)
    """Env var names to set when a custom ``--base-url`` is provided."""

    requires_agent_llm: bool = True
    """Whether this image expects agent-side provider/model configuration."""


@dataclass(frozen=True)
class ResolvedApiKey:
    """Resolved API key value plus how it was sourced."""

    value: str
    source: str | None = None
    from_env: bool = False


_DEFAULT = ContainerProfile(
    provider_key="PROVIDER",
    api_key_key="API_KEY",
    model_key="MODEL",
    default_provider="anthropic",
    extra_flags={},
    base_url_keys=["BASE_URL"],
)

_OPENCLAW = ContainerProfile(
    provider_key="PROVIDER",
    api_key_key="API_KEY",
    model_key="MODEL",
    default_provider="anthropic",
    extra_flags={"OPENCLAW_FORCE_RECONFIG": "1"},
    base_url_keys=["BASE_URL", "OPENROUTER_BASE_URL"],
)

_HERMES = ContainerProfile(
    provider_key="PROVIDER",
    api_key_key="API_KEY",
    model_key="MODEL",
    default_provider="anthropic",
    extra_flags={},
    base_url_keys=["BASE_URL"],
)

_ORACLE = ContainerProfile(
    provider_key="PROVIDER",
    api_key_key="API_KEY",
    model_key="MODEL",
    default_provider="",
    extra_flags={},
    base_url_keys=[],
    requires_agent_llm=False,
)

_PROVIDER_API_KEY_ENV_KEYS: dict[str, tuple[str, ...]] = {
    "anthropic": ("ANTHROPIC_API_KEY",),
    "google": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    "openai": ("OPENAI_API_KEY",),
    "openai-codex": ("OPENAI_API_KEY",),
    "openai-compat": (
        "OPENAI_COMPAT_API_KEY",
        "OPENAI_COMPLETIONS_API_KEY",
        "OPENAI_API_KEY",
    ),
    "openai-completions": (
        "OPENAI_COMPAT_API_KEY",
        "OPENAI_COMPLETIONS_API_KEY",
        "OPENAI_API_KEY",
    ),
    "openrouter": ("OPENROUTER_API_KEY",),
}

_PROVIDER_API_KEY_EXPORT_KEYS: dict[str, tuple[str, ...]] = {
    "anthropic": ("ANTHROPIC_API_KEY",),
    "google": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    "openai": ("OPENAI_API_KEY",),
    "openai-codex": ("OPENAI_API_KEY",),
    "openai-compat": ("OPENAI_COMPAT_API_KEY", "OPENAI_COMPLETIONS_API_KEY"),
    "openai-completions": ("OPENAI_COMPAT_API_KEY", "OPENAI_COMPLETIONS_API_KEY"),
    "openrouter": ("OPENROUTER_API_KEY",),
}


def provider_api_key_export_keys(provider: str | None) -> tuple[str, ...]:
    """Return secondary env vars that should mirror the resolved API key."""
    if not provider:
        return ()
    return _PROVIDER_API_KEY_EXPORT_KEYS.get(provider.lower(), ())


def resolve_api_key_details(
    provider: str | None,
    explicit_api_key: str | None = None,
    env: Mapping[str, str] | None = None,
) -> ResolvedApiKey:
    """Resolve an API key from explicit input, provider-specific env, or API_KEY."""
    if explicit_api_key:
        return ResolvedApiKey(explicit_api_key.strip(), source="explicit")

    values = env or {}
    provider_keys = _PROVIDER_API_KEY_ENV_KEYS.get((provider or "").lower(), ())
    for key in provider_keys:
        value = (values.get(key) or os.environ.get(key, "")).strip()
        if value:
            return ResolvedApiKey(value, source=key, from_env=True)

    generic_value = (values.get("API_KEY") or os.environ.get("API_KEY", "")).strip()
    if generic_value:
        return ResolvedApiKey(generic_value, source="API_KEY", from_env=True)

    return ResolvedApiKey("")


def detect_profile(image: str) -> ContainerProfile:
    """Return the env-var profile for the given container image name."""
    image_lower = image.lower()
    if "gaia2-oc" in image_lower or "openclaw" in image_lower:
        return _OPENCLAW
    if "gaia2-hermes" in image_lower or "hermes" in image_lower:
        return _HERMES
    if "gaia2-oracle" in image_lower or "oracle" in image_lower:
        return _ORACLE
    return _DEFAULT
