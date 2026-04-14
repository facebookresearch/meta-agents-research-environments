# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from __future__ import annotations

import json
import os
from pathlib import Path

from click.testing import CliRunner
from gaia2_runner import cli as runner_cli
from gaia2_runner.cli import main
from gaia2_runner.envfile import load_env_file


def _write_scenario(path: Path, scenario_id: str = "scenario_1") -> None:
    path.write_text(
        json.dumps(
            {
                "metadata": {"definition": {"scenario_id": scenario_id}},
                "events": [],
            }
        )
        + "\n"
    )


def _write_config(path: Path) -> None:
    path.write_text("""
[target]
scenario = "scenario.json"

[agent]
image = "localhost/gaia2-hermes:latest"
provider = "anthropic"
model = "agent-model"
api_key_env = "TEST_AGENT_KEY"

[judge]
provider = "openai"
model = "judge-model"
api_key_env = "TEST_JUDGE_KEY"

[run]
output_dir = "out"
""")


def test_load_env_file_sets_missing_values_and_preserves_existing(
    tmp_path: Path, monkeypatch
) -> None:
    env_path = tmp_path / "test.env"
    env_path.write_text("""
# comment
FIRST=value-one
export SECOND="two words"
THIRD=three # trailing comment
EMPTY=
EXISTING=from-file
EMPTY_EXISTING=from-file
""")

    monkeypatch.setenv("EXISTING", "from-env")
    monkeypatch.setenv("EMPTY_EXISTING", "")

    loaded = load_env_file(env_path)

    assert loaded == ["FIRST", "SECOND", "THIRD", "EMPTY", "EMPTY_EXISTING"]
    assert "EXISTING" not in loaded
    assert os.environ["FIRST"] == "value-one"
    assert os.environ["SECOND"] == "two words"
    assert os.environ["THIRD"] == "three"
    assert os.environ["EMPTY"] == ""
    assert os.environ["EXISTING"] == "from-env"
    assert os.environ["EMPTY_EXISTING"] == "from-file"


def test_run_config_autoloads_dotenv_from_cwd(tmp_path: Path, monkeypatch) -> None:
    _write_scenario(tmp_path / "scenario.json")
    _write_config(tmp_path / "eval.toml")
    (tmp_path / ".env").write_text("""
TEST_AGENT_KEY=agent-secret
TEST_JUDGE_KEY=judge-secret
""")

    monkeypatch.delenv("TEST_AGENT_KEY", raising=False)
    monkeypatch.delenv("TEST_JUDGE_KEY", raising=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(runner_cli, "_default_env_file", lambda: tmp_path / ".env")

    runner = CliRunner()
    result = runner.invoke(
        main,
        ["run-config", "--config", str(tmp_path / "eval.toml"), "--dry-run"],
    )

    assert result.exit_code == 0


def test_run_config_autoloads_repo_root_dotenv_from_subdirectory(
    tmp_path: Path, monkeypatch
) -> None:
    _write_scenario(tmp_path / "scenario.json")
    _write_config(tmp_path / "eval.toml")
    (tmp_path / ".env").write_text("""
TEST_AGENT_KEY=agent-secret
TEST_JUDGE_KEY=judge-secret
""")
    nested_cwd = tmp_path / "runner"
    nested_cwd.mkdir()

    monkeypatch.delenv("TEST_AGENT_KEY", raising=False)
    monkeypatch.delenv("TEST_JUDGE_KEY", raising=False)
    monkeypatch.chdir(nested_cwd)
    monkeypatch.setattr(runner_cli, "_default_env_file", lambda: tmp_path / ".env")

    runner = CliRunner()
    result = runner.invoke(
        main,
        ["run-config", "--config", str(tmp_path / "eval.toml"), "--dry-run"],
    )

    assert result.exit_code == 0


def test_run_config_supports_explicit_env_file(tmp_path: Path, monkeypatch) -> None:
    _write_scenario(tmp_path / "scenario.json")
    _write_config(tmp_path / "eval.toml")
    env_path = tmp_path / "runner.env"
    env_path.write_text("""
TEST_AGENT_KEY=agent-secret
TEST_JUDGE_KEY=judge-secret
""")

    monkeypatch.delenv("TEST_AGENT_KEY", raising=False)
    monkeypatch.delenv("TEST_JUDGE_KEY", raising=False)
    monkeypatch.chdir(tmp_path)

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "--env-file",
            str(env_path),
            "run-config",
            "--config",
            str(tmp_path / "eval.toml"),
            "--dry-run",
        ],
    )

    assert result.exit_code == 0
