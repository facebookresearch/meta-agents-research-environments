# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from __future__ import annotations

from gaia2_runner.container_env import (
    detect_profile,
    provider_api_key_export_keys,
    resolve_api_key_details,
)


def test_openclaw_profile_defaults_to_anthropic() -> None:
    profile = detect_profile("localhost/gaia2-oc:latest")

    assert profile.provider_key == "PROVIDER"
    assert profile.api_key_key == "API_KEY"
    assert profile.model_key == "MODEL"
    assert profile.default_provider == "anthropic"
    assert profile.base_url_keys == ["BASE_URL", "OPENROUTER_BASE_URL"]
    assert profile.extra_flags == {"OPENCLAW_FORCE_RECONFIG": "1"}


def test_hermes_profile_has_no_openclaw_flags() -> None:
    profile = detect_profile("localhost/gaia2-hermes:latest")

    assert profile.provider_key == "PROVIDER"
    assert profile.api_key_key == "API_KEY"
    assert profile.model_key == "MODEL"
    assert profile.default_provider == "anthropic"
    assert profile.base_url_keys == ["BASE_URL"]
    assert profile.extra_flags == {}


def test_oracle_profile_does_not_require_agent_llm() -> None:
    profile = detect_profile("localhost/gaia2-oracle:latest")

    assert profile.requires_agent_llm is False
    assert profile.default_provider == ""
    assert profile.base_url_keys == []


def test_resolve_api_key_prefers_provider_specific_env(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    monkeypatch.setenv("API_KEY", "generic-key")

    resolved = resolve_api_key_details("openai", None, {})
    assert resolved.value == "openai-key"
    assert resolved.source == "OPENAI_API_KEY"
    assert resolved.from_env is True


def test_resolve_api_key_falls_back_to_generic_env(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("API_KEY", "generic-key")

    resolved = resolve_api_key_details("openai", None, {})
    assert resolved.value == "generic-key"
    assert resolved.source == "API_KEY"
    assert resolved.from_env is True


def test_provider_api_key_export_keys_for_openai() -> None:
    assert provider_api_key_export_keys("openai") == ("OPENAI_API_KEY",)


def test_resolve_api_key_prefers_openai_compat_env(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_COMPAT_API_KEY", "compat-key")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")

    resolved = resolve_api_key_details("openai-compat", None, {})
    assert resolved.value == "compat-key"
    assert resolved.source == "OPENAI_COMPAT_API_KEY"
    assert resolved.from_env is True


def test_provider_api_key_export_keys_for_openai_compat() -> None:
    assert provider_api_key_export_keys("openai-compat") == (
        "OPENAI_COMPAT_API_KEY",
        "OPENAI_COMPLETIONS_API_KEY",
    )
