# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import openclaw_config

FIXED_NOW = "2026-04-08T21:00:00.000Z"
SCRIPT = Path(__file__).resolve().with_name("openclaw_config.py")


def test_openai_without_base_url_uses_builtin_provider() -> None:
    outputs = openclaw_config.generate_setup(
        {
            "PROVIDER": "openai",
            "API_KEY": "openai-key",
            "MODEL": "gpt-5.4",
        },
        now=FIXED_NOW,
    )

    assert outputs.provider_setup.provider == "openai"
    assert outputs.provider_setup.provider_config is None
    assert outputs.provider_setup.shell_exports["OPENAI_API_KEY"] == "openai-key"
    assert outputs.openclaw_config["models"] == {}
    assert outputs.auth_profiles == {}
    assert (
        outputs.openclaw_config["agents"]["defaults"]["model"]["primary"]
        == "openai/gpt-5.4"
    )


def test_openai_with_base_url_uses_custom_responses_provider() -> None:
    outputs = openclaw_config.generate_setup(
        {
            "PROVIDER": "openai",
            "API_KEY": "openai-key",
            "MODEL": "custom-openai-model",
            "BASE_URL": "https://proxy.example/v1/",
        },
        now=FIXED_NOW,
    )

    provider_cfg = outputs.openclaw_config["models"]["providers"]["openai"]
    assert provider_cfg == {
        "baseUrl": "https://proxy.example/v1",
        "api": "openai-responses",
        "models": [
            {
                "id": "custom-openai-model",
                "name": "custom-openai-model",
                "reasoning": True,
                "input": ["text", "image"],
                "contextWindow": 272000,
                "maxTokens": 128000,
            }
        ],
    }
    assert outputs.auth_profiles == {"openai": {"apiKey": "openai-key"}}


def test_google_without_base_url_uses_builtin_provider() -> None:
    outputs = openclaw_config.generate_setup(
        {
            "PROVIDER": "google",
            "GEMINI_API_KEY": "google-key",
        },
        now=FIXED_NOW,
    )

    assert outputs.provider_setup.provider == "google"
    assert outputs.provider_setup.model == "gemini-3.1-pro-preview"
    assert outputs.provider_setup.provider_config is None
    assert outputs.provider_setup.shell_exports["GEMINI_API_KEY"] == "google-key"
    assert outputs.provider_setup.shell_exports["GOOGLE_API_KEY"] == "google-key"
    assert outputs.openclaw_config["models"] == {}
    assert outputs.auth_profiles == {}


def test_google_with_base_url_uses_custom_native_provider() -> None:
    outputs = openclaw_config.generate_setup(
        {
            "PROVIDER": "google",
            "API_KEY": "custom-api-key",
            "MODEL": "custom-google-model",
            "BASE_URL": "https://provider.example/google/v1/",
        },
        now=FIXED_NOW,
    )

    provider_cfg = outputs.openclaw_config["models"]["providers"]["google"]
    assert provider_cfg == {
        "baseUrl": "https://provider.example/google/v1",
        "api": "google-generative-ai",
        "authHeader": True,
        "models": [
            {
                "id": "custom-google-model",
                "name": "custom-google-model",
                "reasoning": True,
                "input": ["text", "image"],
                "contextWindow": 1048576,
                "maxTokens": 65536,
            }
        ],
    }
    assert outputs.auth_profiles == {"google": {"apiKey": "custom-api-key"}}


def test_anthropic_without_base_url_uses_builtin_provider() -> None:
    outputs = openclaw_config.generate_setup(
        {
            "PROVIDER": "anthropic",
            "ANTHROPIC_API_KEY": "anthropic-key",
        },
        now=FIXED_NOW,
    )

    assert outputs.provider_setup.provider == "anthropic"
    assert outputs.provider_setup.model == "claude-sonnet-4-6"
    assert outputs.provider_setup.provider_config is None
    assert outputs.provider_setup.shell_exports["ANTHROPIC_API_KEY"] == "anthropic-key"
    assert outputs.openclaw_config["models"] == {}
    assert outputs.auth_profiles == {}


def test_anthropic_with_base_url_uses_custom_messages_provider() -> None:
    outputs = openclaw_config.generate_setup(
        {
            "PROVIDER": "anthropic",
            "API_KEY": "anthropic-key",
            "MODEL": "custom-anthropic-model",
            "BASE_URL": "https://proxy.example/anthropic/",
        },
        now=FIXED_NOW,
    )

    provider_cfg = outputs.openclaw_config["models"]["providers"]["anthropic"]
    assert provider_cfg == {
        "baseUrl": "https://proxy.example/anthropic",
        "api": "anthropic-messages",
        "models": [
            {
                "id": "custom-anthropic-model",
                "name": "custom-anthropic-model",
                "reasoning": True,
                "input": ["text", "image"],
                "contextWindow": 200000,
                "maxTokens": 64000,
            }
        ],
    }
    assert outputs.auth_profiles == {"anthropic": {"apiKey": "anthropic-key"}}


def test_default_provider_is_anthropic_builtin() -> None:
    outputs = openclaw_config.generate_setup(
        {
            "ANTHROPIC_API_KEY": "anthropic-key",
        },
        now=FIXED_NOW,
    )

    assert outputs.provider_setup.provider == "anthropic"
    assert outputs.provider_setup.model == "claude-sonnet-4-6"
    assert outputs.provider_setup.shell_exports["ANTHROPIC_API_KEY"] == "anthropic-key"
    assert outputs.provider_setup.provider_config is None
    assert outputs.openclaw_config["models"] == {}
    assert outputs.auth_profiles == {}


def test_openrouter_uses_custom_chat_completions_defaults() -> None:
    outputs = openclaw_config.generate_setup(
        {
            "PROVIDER": "openrouter",
            "OPENROUTER_API_KEY": "openrouter-key",
        },
        now=FIXED_NOW,
    )

    provider_cfg = outputs.openclaw_config["models"]["providers"]["openrouter"]
    model_cfg = provider_cfg["models"][0]
    assert outputs.provider_setup.model == "moonshotai/kimi-k2.5"
    assert provider_cfg["baseUrl"] == openclaw_config.DEFAULT_OPENROUTER_BASE_URL
    assert model_cfg["reasoning"] is True
    assert model_cfg["contextWindow"] == 262144
    assert model_cfg["maxTokens"] == 65535


def test_openai_compat_requires_base_url_and_model() -> None:
    with pytest.raises(openclaw_config.SetupError, match="MODEL is required"):
        openclaw_config.generate_setup(
            {
                "PROVIDER": "openai-compat",
                "BASE_URL": "https://compat.example/v1",
            },
            now=FIXED_NOW,
        )

    with pytest.raises(openclaw_config.SetupError, match="BASE_URL is required"):
        openclaw_config.generate_setup(
            {
                "PROVIDER": "openai-compat",
                "MODEL": "compat-model",
            },
            now=FIXED_NOW,
        )


def test_openai_compat_uses_custom_chat_completions_provider() -> None:
    outputs = openclaw_config.generate_setup(
        {
            "PROVIDER": "openai-compat",
            "API_KEY": "compat-key",
            "MODEL": "compat-model",
            "BASE_URL": "https://compat.example/v1/",
        },
        now=FIXED_NOW,
    )

    provider_cfg = outputs.openclaw_config["models"]["providers"]["openai-compat"]
    assert outputs.provider_setup.provider == "openai-compat"
    assert outputs.openclaw_config["agents"]["defaults"]["model"]["primary"] == (
        "openai-compat/compat-model"
    )
    assert provider_cfg == {
        "baseUrl": "https://compat.example/v1",
        "apiKey": "compat-key",
        "api": "openai-completions",
        "authHeader": True,
        "models": [
            {
                "id": "compat-model",
                "name": "compat-model",
                "api": "openai-completions",
                "reasoning": True,
                "input": ["text", "image"],
                "contextWindow": 262144,
                "maxTokens": 65535,
                "cost": {
                    "input": 0,
                    "output": 0,
                    "cacheRead": 0,
                    "cacheWrite": 0,
                },
            }
        ],
    }
    assert outputs.auth_profiles == {"openai-compat": {"apiKey": "compat-key"}}


def test_openai_completions_alias_matches_openai_compat() -> None:
    outputs = openclaw_config.generate_setup(
        {
            "PROVIDER": "openai-completions",
            "API_KEY": "compat-key",
            "MODEL": "compat-model",
            "BASE_URL": "https://compat.example/v1/",
        },
        now=FIXED_NOW,
    )

    assert outputs.provider_setup.provider == "openai-completions"
    assert (
        outputs.openclaw_config["models"]["providers"]["openai-completions"]["api"]
        == "openai-completions"
    )


def test_unknown_provider_raises_setup_error() -> None:
    with pytest.raises(openclaw_config.SetupError):
        openclaw_config.generate_setup({"PROVIDER": "bogus"}, now=FIXED_NOW)


def test_write_runtime_files_creates_expected_outputs(tmp_path: Path) -> None:
    home = tmp_path / "home"
    outputs = openclaw_config.generate_setup(
        {
            "PROVIDER": "openai",
            "API_KEY": "openai-key",
            "MODEL": "custom-openai-model",
            "BASE_URL": "https://proxy.example/v1",
        },
        now=FIXED_NOW,
    )

    openclaw_config.write_runtime_files(home, outputs)

    openclaw_json = json.loads((home / ".openclaw" / "openclaw.json").read_text())
    auth_profiles = json.loads(
        (
            home / ".openclaw" / "agents" / "main" / "agent" / "auth-profiles.json"
        ).read_text()
    )

    assert openclaw_json == outputs.openclaw_config
    assert auth_profiles == outputs.auth_profiles
    assert (home / "HEARTBEAT.md").read_text() == openclaw_config.HEARTBEAT_CONTENT
    for name in ("SOUL.md", "TOOLS.md", "IDENTITY.md", "USER.md"):
        assert (home / name).read_text() == ""


def test_cli_writes_files_and_emits_shell_exports(tmp_path: Path) -> None:
    home = tmp_path / "home"
    env = os.environ.copy()
    env.update(
        {
            "PROVIDER": "anthropic",
            "API_KEY": "anthropic-key",
            "BASE_URL": "https://proxy.example/anthropic",
            "HOME": str(home),
        }
    )

    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--home", str(home)],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    assert "export PROVIDER=anthropic" in proc.stdout
    assert "export ANTHROPIC_API_KEY=anthropic-key" in proc.stdout

    openclaw_json = json.loads((home / ".openclaw" / "openclaw.json").read_text())
    assert (
        openclaw_json["models"]["providers"]["anthropic"]["api"] == "anthropic-messages"
    )
