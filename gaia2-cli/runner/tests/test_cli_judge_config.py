# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Tests for mandatory judge configuration in the runner CLI."""

from __future__ import annotations

import json

import click
import pytest
from gaia2_runner.cli import (
    _resolve_agent_config,
    _resolve_judge_config,
    _save_run_config,
)


class TestResolveJudgeConfig:
    def test_prefers_cli_args_over_env(self, monkeypatch) -> None:
        monkeypatch.setenv("GAIA2_JUDGE_MODEL", "env-model")
        monkeypatch.setenv("GAIA2_JUDGE_PROVIDER", "env-provider")
        monkeypatch.setenv("GAIA2_JUDGE_BASE_URL", "https://env.example/v1")

        result = _resolve_judge_config(
            "cli-model",
            "cli-provider",
            "https://cli.example/v1",
            "cli-key",
        )

        assert result == (
            "cli-model",
            "cli-provider",
            "https://cli.example/v1",
            "cli-key",
        )

    def test_uses_env_fallback_when_flags_missing(self, monkeypatch, capsys) -> None:
        monkeypatch.setenv("GAIA2_JUDGE_MODEL", "env-model")
        monkeypatch.setenv("GAIA2_JUDGE_PROVIDER", "env-provider")
        monkeypatch.setenv("GAIA2_JUDGE_BASE_URL", "https://env.example/v1")
        monkeypatch.setenv("GAIA2_JUDGE_API_KEY", "env-key")

        result = _resolve_judge_config(None, None, None, None)
        err = capsys.readouterr().err

        assert result == (
            "env-model",
            "env-provider",
            "https://env.example/v1",
            "env-key",
        )
        assert "GAIA2_JUDGE_API_KEY" in err

    def test_raises_when_model_and_provider_missing(self, monkeypatch) -> None:
        monkeypatch.delenv("GAIA2_JUDGE_MODEL", raising=False)
        monkeypatch.delenv("GAIA2_JUDGE_PROVIDER", raising=False)
        monkeypatch.delenv("GAIA2_JUDGE_BASE_URL", raising=False)
        monkeypatch.delenv("GAIA2_JUDGE_API_KEY", raising=False)

        with pytest.raises(click.UsageError) as exc:
            _resolve_judge_config(None, None, None, None)

        assert (
            "In-container judge configuration is required for benchmark runs."
            in str(exc.value)
        )
        assert "--judge-model / GAIA2_JUDGE_MODEL" in str(exc.value)
        assert "--judge-provider / GAIA2_JUDGE_PROVIDER" in str(exc.value)


class TestResolveAgentConfig:
    def test_requires_provider_and_model_for_agent_images(self) -> None:
        with pytest.raises(click.UsageError) as exc:
            _resolve_agent_config("localhost/gaia2-oc:latest", None, None)

        assert "--provider" in str(exc.value)
        assert "--model" in str(exc.value)

    def test_allows_oracle_without_agent_llm_config(self) -> None:
        assert _resolve_agent_config("localhost/gaia2-oracle:latest", None, None) == (
            None,
            None,
        )


class TestSaveRunConfig:
    def test_persists_judge_settings(self, tmp_path) -> None:
        _save_run_config(
            output_dir=str(tmp_path),
            command="run",
            image="test-image:latest",
            runtime="podman",
            provider="openai",
            model="gpt-4.1-mini",
            base_url="https://example.invalid/v1",
            judge_model="judge-model",
            judge_provider="judge-provider",
            judge_base_url="https://judge.example/v1",
            timeout=600,
            health_timeout=120,
            concurrency=1,
            limit=None,
        )

        config = json.loads((tmp_path / "run_config.json").read_text())
        assert config["judge_model"] == "judge-model"
        assert config["judge_provider"] == "judge-provider"
        assert config["judge_base_url"] == "https://judge.example/v1"

    def test_persists_hf_dataset_cache_dir(self, tmp_path) -> None:
        cache_dir = tmp_path / "hf_cache"
        cache_dir.mkdir()

        _save_run_config(
            output_dir=str(tmp_path),
            command="run-dataset",
            dataset="meta-agents-research-environments/gaia2-cli",
            dataset_cache_dir=str(cache_dir),
            splits=["search"],
            image="test-image:latest",
            runtime="podman",
            provider="openai",
            model="gpt-4.1-mini",
            base_url=None,
            judge_model="judge-model",
            judge_provider="judge-provider",
            judge_base_url=None,
            timeout=600,
            health_timeout=120,
            concurrency=1,
            limit=None,
        )

        config = json.loads((tmp_path / "run_config.json").read_text())
        assert config["dataset"] == "meta-agents-research-environments/gaia2-cli"
        assert config["dataset_cache_dir"] == str(cache_dir.resolve())
        assert config["splits"] == ["search"]

    def test_preserves_existing_num_scenarios_for_retry(self, tmp_path) -> None:
        _save_run_config(
            output_dir=str(tmp_path),
            command="run-dataset",
            image="test-image:latest",
            runtime="podman",
            provider="openai",
            model="gpt-4.1-mini",
            base_url=None,
            judge_model="judge-model",
            judge_provider="judge-provider",
            judge_base_url=None,
            timeout=600,
            health_timeout=120,
            concurrency=40,
            limit=None,
            num_scenarios=160,
        )

        _save_run_config(
            output_dir=str(tmp_path),
            command="run-dataset",
            image="test-image:latest",
            runtime="podman",
            provider="openai",
            model="gpt-4.1-mini",
            base_url=None,
            judge_model="judge-model",
            judge_provider="judge-provider",
            judge_base_url=None,
            timeout=600,
            health_timeout=120,
            concurrency=40,
            limit=None,
            num_scenarios=25,
        )

        config = json.loads((tmp_path / "run_config.json").read_text())
        assert config["num_scenarios"] == 160
        assert config["retry_num_scenarios"] == 25
