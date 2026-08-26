# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from __future__ import annotations

import json
from pathlib import Path

import click
import pytest
from click.testing import CliRunner
from gaia2_runner import cli as runner_cli
from gaia2_runner.cli import main
from gaia2_runner.config import (
    CANONICAL_SPLITS,
    MULTILINGUAL_SPLITS,
    load_runner_toml_config,
)


def _write_scenario(path: Path, scenario_id: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "metadata": {"definition": {"scenario_id": scenario_id}},
                "events": [],
            }
        )
        + "\n"
    )


def test_load_runner_toml_config_resolves_relative_paths_and_env_keys(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_scenario(tmp_path / "scenario.json", "scenario_1")
    monkeypatch.setenv("TEST_AGENT_KEY", "agent-secret")
    monkeypatch.setenv("TEST_JUDGE_KEY", "judge-secret")

    config_path = tmp_path / "eval.toml"
    config_path.write_text("""
[target]
scenario = "scenario.json"

[agent]
image = "localhost/gaia2-hermes:latest"
provider = "anthropic"
model = "agent-model"
api_key_env = "TEST_AGENT_KEY"
thinking = "high"

[judge]
provider = "judge-provider"
model = "judge-model"
api_key_env = "TEST_JUDGE_KEY"

[run]
output_dir = "out"
log_level = "DEBUG"
""")

    config = load_runner_toml_config(str(config_path))

    assert config.target.scenario == str((tmp_path / "scenario.json").resolve())
    assert config.agent.api_key == "agent-secret"
    assert config.judge.api_key == "judge-secret"
    assert config.run.output_dir == str((tmp_path / "out").resolve())
    assert config.target.is_single_scenario is True


def test_load_runner_toml_config_reads_judge_prompt_version_and_extra_body(
    tmp_path: Path,
) -> None:
    _write_scenario(tmp_path / "scenario.json", "scenario_1")

    config_path = tmp_path / "eval.toml"
    config_path.write_text("""
[target]
scenario = "scenario.json"

[agent]
image = "localhost/gaia2-hermes:latest"
provider = "anthropic"
model = "agent-model"

[judge]
provider = "judge-provider"
model = "judge-model"
prompt_version = "omnilingual-gaia2"
extra_body = { chat_template_kwargs = { enable_thinking = true } }

[run]
output_dir = "out"
idle_timeout = 900
""")

    config = load_runner_toml_config(str(config_path))

    assert config.judge.prompt_version == "omnilingual-gaia2"
    assert config.judge.extra_body == {
        "chat_template_kwargs": {"enable_thinking": True}
    }
    assert config.run.idle_timeout == 900.0


def test_load_runner_toml_config_defaults_judge_prompt_version_to_none(
    tmp_path: Path,
) -> None:
    _write_scenario(tmp_path / "scenario.json", "scenario_1")

    config_path = tmp_path / "eval.toml"
    config_path.write_text("""
[target]
scenario = "scenario.json"

[agent]
image = "localhost/gaia2-hermes:latest"
provider = "anthropic"
model = "agent-model"

[judge]
provider = "judge-provider"
model = "judge-model"

[run]
output_dir = "out"
""")

    config = load_runner_toml_config(str(config_path))

    assert config.judge.prompt_version is None
    assert config.judge.extra_body is None
    assert config.run.idle_timeout is None


def test_load_runner_toml_config_rejects_non_positive_idle_timeout(
    tmp_path: Path,
) -> None:
    _write_scenario(tmp_path / "scenario.json", "scenario_1")

    config_path = tmp_path / "eval.toml"
    config_path.write_text("""
[target]
scenario = "scenario.json"

[agent]
image = "localhost/gaia2-hermes:latest"
provider = "anthropic"
model = "agent-model"

[judge]
provider = "judge-provider"
model = "judge-model"

[run]
output_dir = "out"
idle_timeout = 0
""")

    with pytest.raises(click.UsageError, match=r"\[run\].idle_timeout must be > 0"):
        load_runner_toml_config(str(config_path))


def test_load_runner_toml_config_supports_all_splits_and_auto_output_jsonl(
    tmp_path: Path,
) -> None:
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()

    config_path = tmp_path / "eval.toml"
    config_path.write_text("""
[target]
dataset_root = "dataset"
splits = "all"

[agent]
image = "localhost/gaia2-hermes:latest"
provider = "anthropic"
model = "agent-model"

[judge]
provider = "judge-provider"
model = "judge-model"

[run]
output_dir = "out"
""")

    config = load_runner_toml_config(str(config_path))

    assert config.target.splits == CANONICAL_SPLITS
    assert config.run.output == str((tmp_path / "out" / "results.jsonl").resolve())


def test_load_runner_toml_config_expands_env_vars_in_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    monkeypatch.setenv("GAIA2_DATASET_DIR", str(dataset_root))
    monkeypatch.setenv("GAIA2_OUTPUT_ROOT", str(tmp_path / "outputs"))

    config_path = tmp_path / "eval.toml"
    config_path.write_text("""
[target]
dataset_root = "${GAIA2_DATASET_DIR}"
splits = "all"

[agent]
image = "localhost/gaia2-hermes:latest"
provider = "anthropic"
model = "agent-model"

[judge]
provider = "judge-provider"
model = "judge-model"

[run]
output_dir = "${GAIA2_OUTPUT_ROOT}/bench"
""")

    config = load_runner_toml_config(str(config_path))

    assert config.target.dataset_root == str(dataset_root.resolve())
    assert config.run.output_dir == str((tmp_path / "outputs" / "bench").resolve())
    assert config.run.output == str(
        (tmp_path / "outputs" / "bench" / "results.jsonl").resolve()
    )


def test_load_runner_toml_config_rejects_unset_env_var_in_path(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "eval.toml"
    config_path.write_text("""
[target]
dataset_root = "${MISSING_GAIA2_DATASET}"
splits = "all"

[agent]
image = "localhost/gaia2-hermes:latest"
provider = "anthropic"
model = "agent-model"

[judge]
provider = "judge-provider"
model = "judge-model"
""")

    with pytest.raises(Exception, match="references unset env var"):
        load_runner_toml_config(str(config_path))


def test_load_runner_toml_config_rejects_unknown_split(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()

    config_path = tmp_path / "eval.toml"
    config_path.write_text("""
[target]
dataset_root = "dataset"
splits = ["unknown"]

[agent]
image = "localhost/gaia2-hermes:latest"
provider = "anthropic"
model = "agent-model"

[judge]
provider = "judge-provider"
model = "judge-model"
""")

    with pytest.raises(Exception, match="Unknown split"):
        load_runner_toml_config(str(config_path))


def test_load_runner_toml_config_rejects_unknown_target_field(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()

    config_path = tmp_path / "eval.toml"
    config_path.write_text("""
[target]
dataset_root = "dataset"
splits = "all"
extra_flag = true

[agent]
image = "localhost/gaia2-hermes:latest"
provider = "anthropic"
model = "agent-model"

[judge]
provider = "judge-provider"
model = "judge-model"
""")

    with pytest.raises(Exception, match=r"\[target\] has unknown field"):
        load_runner_toml_config(str(config_path))


def test_load_runner_toml_config_rejects_dataset_only_fields_for_scenario(
    tmp_path: Path,
) -> None:
    _write_scenario(tmp_path / "scenario.json", "scenario_1")

    config_path = tmp_path / "eval.toml"
    config_path.write_text("""
[target]
scenario = "scenario.json"
recursive = false

[agent]
image = "localhost/gaia2-oracle:latest"

[judge]
provider = "judge-provider"
model = "judge-model"
""")

    with pytest.raises(Exception, match=r"\[target\]\.recursive"):
        load_runner_toml_config(str(config_path))


@pytest.mark.parametrize(
    ("config_text", "message"),
    [
        (
            """
[target]
scenario = "scenario.json"

[agent]
image = "localhost/gaia2-hermes:latest"
provider = "anthropic"
model = "agent-model"
thinking = "max"

[judge]
provider = "judge-provider"
model = "judge-model"
""",
            r"\[agent\]\.thinking",
        ),
        (
            """
[target]
scenario = "scenario.json"

[agent]
image = "localhost/gaia2-hermes:latest"
provider = "anthropic"
model = "agent-model"

[judge]
provider = "judge-provider"
model = "judge-model"

[run]
notification_mode = "toast"
""",
            r"\[run\]\.notification_mode",
        ),
    ],
)
def test_load_runner_toml_config_validates_choice_fields(
    tmp_path: Path,
    config_text: str,
    message: str,
) -> None:
    _write_scenario(tmp_path / "scenario.json", "scenario_1")

    config_path = tmp_path / "eval.toml"
    config_path.write_text(config_text)

    with pytest.raises(Exception, match=message):
        load_runner_toml_config(str(config_path))


def test_run_config_dry_run_resolves_selected_splits(tmp_path: Path) -> None:
    _write_scenario(tmp_path / "dataset" / "execution" / "s1.json", "s1")
    _write_scenario(tmp_path / "dataset" / "search" / "s2.json", "s2")
    _write_scenario(tmp_path / "dataset" / "time" / "s3.json", "s3")

    config_path = tmp_path / "eval.toml"
    config_path.write_text("""
[target]
dataset_root = "dataset"
splits = ["execution", "search"]

[agent]
image = "localhost/gaia2-oracle:latest"

[judge]
provider = "judge-provider"
model = "judge-model"

[run]
output_dir = "out"
""")

    runner = CliRunner()
    result = runner.invoke(
        main, ["run-config", "--config", str(config_path), "--dry-run"]
    )

    assert result.exit_code == 0
    assert "Mode: dataset" in result.output
    assert "Splits: execution, search" in result.output
    assert "Resolved scenarios: 2" in result.output


def test_run_config_dry_run_shows_hf_dataset_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from gaia2_runner import hf_dataset

    _write_scenario(tmp_path / "hf_cache" / "search" / "s1.json", "s1")
    monkeypatch.setattr(
        hf_dataset,
        "download_hf_dataset",
        lambda dataset_id, splits=None, language=None: str(tmp_path / "hf_cache"),
    )

    config_path = tmp_path / "eval.toml"
    config_path.write_text("""
[target]
dataset = "meta-agents-research-environments/gaia2-cli"
splits = ["search"]

[agent]
image = "localhost/gaia2-oracle:latest"

[judge]
provider = "judge-provider"
model = "judge-model"

[run]
output_dir = "out"
""")

    runner = CliRunner()
    result = runner.invoke(
        main, ["run-config", "--config", str(config_path), "--dry-run"]
    )

    assert result.exit_code == 0
    assert "Mode: dataset" in result.output
    assert "Dataset: meta-agents-research-environments/gaia2-cli" in result.output
    assert "Dataset root: None" not in result.output
    assert "Splits: search" in result.output


def test_run_config_hf_dataset_respects_selected_splits_with_cached_extra_splits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from gaia2_runner import hf_dataset

    _write_scenario(tmp_path / "hf_cache" / "search" / "s1.json", "s1")
    _write_scenario(tmp_path / "hf_cache" / "time" / "s2.json", "s2")
    monkeypatch.setattr(
        hf_dataset,
        "download_hf_dataset",
        lambda dataset_id, splits=None, language=None: str(tmp_path / "hf_cache"),
    )

    config_path = tmp_path / "eval.toml"
    config_path.write_text("""
[target]
dataset = "meta-agents-research-environments/gaia2-cli"
splits = ["time"]

[agent]
image = "localhost/gaia2-oracle:latest"

[judge]
provider = "judge-provider"
model = "judge-model"

[run]
output_dir = "out"
""")

    runner = CliRunner()
    result = runner.invoke(
        main, ["run-config", "--config", str(config_path), "--dry-run"]
    )

    assert result.exit_code == 0
    assert "Splits: time" in result.output
    assert "Resolved scenarios: 1" in result.output


def test_run_config_dry_run_shows_retry_override(tmp_path: Path) -> None:
    _write_scenario(tmp_path / "dataset" / "execution" / "s1.json", "s1")

    config_path = tmp_path / "eval.toml"
    config_path.write_text("""
[target]
dataset_root = "dataset"
splits = ["execution"]

[agent]
image = "localhost/gaia2-oracle:latest"

[judge]
provider = "judge-provider"
model = "judge-model"

[run]
output_dir = "out"
""")

    runner = CliRunner()
    result = runner.invoke(
        main, ["run-config", "--config", str(config_path), "--retry", "--dry-run"]
    )

    assert result.exit_code == 0
    assert "Retry: on" in result.output


def test_run_dataset_hf_metadata_includes_cache_dir_and_splits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from gaia2_runner import hf_dataset

    cache_dir = tmp_path / "hf_cache"
    _write_scenario(cache_dir / "execution" / "s1.json", "s1")

    captured: dict[str, object] = {}

    monkeypatch.setattr(
        hf_dataset,
        "download_hf_dataset",
        lambda dataset_id, splits=None, language=None: str(cache_dir),
    )
    monkeypatch.setattr(
        runner_cli,
        "_build_execution_config",
        lambda **kwargs: (object(), None, None, "judge-model", "judge-provider", None),
    )

    def fake_execute_dataset_selection(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(
        runner_cli, "_execute_dataset_selection", fake_execute_dataset_selection
    )

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "run-dataset",
            "--dataset",
            "meta-agents-research-environments/gaia2-cli",
            "--splits",
            "execution",
            "--image",
            "localhost/gaia2-oracle:latest",
            "--judge-provider",
            "judge-provider",
            "--judge-model",
            "judge-model",
            "--output-dir",
            str(tmp_path / "out"),
        ],
    )

    assert result.exit_code == 0
    run_config_base = captured["run_config_base"]
    assert run_config_base["dataset"] == "meta-agents-research-environments/gaia2-cli"
    assert run_config_base["dataset_cache_dir"] == str(cache_dir)
    assert run_config_base["splits"] == ["execution"]
    assert captured["dataset_root"] == cache_dir
    assert len(captured["scenario_paths"]) == 1


def test_run_config_retry_flag_is_forwarded_to_dataset_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_scenario(tmp_path / "dataset" / "execution" / "s1.json", "s1")

    config_path = tmp_path / "eval.toml"
    config_path.write_text("""
[target]
dataset_root = "dataset"
splits = ["execution"]

[agent]
image = "localhost/gaia2-oracle:latest"

[judge]
provider = "judge-provider"
model = "judge-model"

[run]
output_dir = "out"
""")

    captured: dict[str, object] = {}

    monkeypatch.setattr(
        runner_cli,
        "_build_execution_config",
        lambda **kwargs: (object(), None, None, "judge-model", "judge-provider", None),
    )

    def fake_execute_dataset_selection(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(
        runner_cli, "_execute_dataset_selection", fake_execute_dataset_selection
    )

    runner = CliRunner()
    result = runner.invoke(
        main, ["run-config", "--config", str(config_path), "--retry"]
    )

    assert result.exit_code == 0
    assert captured["retry"] is True


def test_run_config_retry_flag_rejected_for_scenario_target(tmp_path: Path) -> None:
    _write_scenario(tmp_path / "scenario.json", "scenario_1")

    config_path = tmp_path / "eval.toml"
    config_path.write_text("""
[target]
scenario = "scenario.json"

[agent]
image = "localhost/gaia2-oracle:latest"

[judge]
provider = "judge-provider"
model = "judge-model"
""")

    runner = CliRunner()
    result = runner.invoke(
        main, ["run-config", "--config", str(config_path), "--retry"]
    )

    assert result.exit_code != 0
    assert "--retry is only supported for dataset targets" in result.output


def test_load_runner_toml_config_reads_target_language(tmp_path: Path) -> None:
    config_path = tmp_path / "eval.toml"
    config_path.write_text("""
[target]
dataset = "facebook/omnilingual-gaia2"
language = "spa_Latn"
splits = ["search"]

[agent]
image = "localhost/gaia2-oracle:latest"

[judge]
provider = "judge-provider"
model = "judge-model"
""")

    config = load_runner_toml_config(str(config_path))

    assert config.target.language == "spa_Latn"
    assert config.target.splits == ("search",)


def test_load_runner_toml_config_language_defaults_to_multilingual_splits(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "eval.toml"
    config_path.write_text("""
[target]
dataset = "facebook/omnilingual-gaia2"
language = "spa_Latn"

[agent]
image = "localhost/gaia2-oracle:latest"

[judge]
provider = "judge-provider"
model = "judge-model"
""")

    config = load_runner_toml_config(str(config_path))

    assert config.target.splits == MULTILINGUAL_SPLITS
    assert "time" not in config.target.splits


def test_load_runner_toml_config_language_all_splits_excludes_time(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "eval.toml"
    config_path.write_text("""
[target]
dataset = "facebook/omnilingual-gaia2"
language = "spa_Latn"
splits = "all"

[agent]
image = "localhost/gaia2-oracle:latest"

[judge]
provider = "judge-provider"
model = "judge-model"
""")

    config = load_runner_toml_config(str(config_path))

    assert config.target.splits == MULTILINGUAL_SPLITS
    assert CANONICAL_SPLITS != MULTILINGUAL_SPLITS


def test_load_runner_toml_config_rejects_time_split_with_language(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "eval.toml"
    config_path.write_text("""
[target]
dataset = "facebook/omnilingual-gaia2"
language = "spa_Latn"
splits = ["search", "time"]

[agent]
image = "localhost/gaia2-oracle:latest"

[judge]
provider = "judge-provider"
model = "judge-model"
""")

    with pytest.raises(click.UsageError, match="not available for"):
        load_runner_toml_config(str(config_path))


def test_load_runner_toml_config_rejects_language_for_scenario_target(
    tmp_path: Path,
) -> None:
    _write_scenario(tmp_path / "scenario.json", "scenario_1")

    config_path = tmp_path / "eval.toml"
    config_path.write_text("""
[target]
scenario = "scenario.json"
language = "spa_Latn"

[agent]
image = "localhost/gaia2-oracle:latest"

[judge]
provider = "judge-provider"
model = "judge-model"
""")

    with pytest.raises(click.UsageError, match=r"\[target\]\.language"):
        load_runner_toml_config(str(config_path))


def test_load_runner_toml_config_rejects_language_for_dataset_root_target(
    tmp_path: Path,
) -> None:
    _write_scenario(tmp_path / "dataset" / "search" / "s1.json", "s1")

    config_path = tmp_path / "eval.toml"
    config_path.write_text("""
[target]
dataset_root = "dataset"
language = "spa_Latn"

[agent]
image = "localhost/gaia2-oracle:latest"

[judge]
provider = "judge-provider"
model = "judge-model"
""")

    with pytest.raises(click.UsageError, match="only applies to"):
        load_runner_toml_config(str(config_path))


def test_load_runner_toml_config_rejects_malformed_language(tmp_path: Path) -> None:
    config_path = tmp_path / "eval.toml"
    config_path.write_text("""
[target]
dataset = "facebook/omnilingual-gaia2"
language = "spa"

[agent]
image = "localhost/gaia2-oracle:latest"

[judge]
provider = "judge-provider"
model = "judge-model"
""")

    with pytest.raises(click.UsageError, match="language_Script"):
        load_runner_toml_config(str(config_path))


def test_run_config_dry_run_shows_language_and_forwards_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from gaia2_runner import hf_dataset

    _write_scenario(tmp_path / "hf_cache" / "search" / "s1.json", "s1")
    captured: dict[str, object] = {}

    def fake_download(dataset_id, splits=None, language=None):
        captured["dataset_id"] = dataset_id
        captured["splits"] = splits
        captured["language"] = language
        return str(tmp_path / "hf_cache")

    monkeypatch.setattr(hf_dataset, "download_hf_dataset", fake_download)

    config_path = tmp_path / "eval.toml"
    config_path.write_text("""
[target]
dataset = "facebook/omnilingual-gaia2"
language = "spa_Latn"
splits = ["search"]

[agent]
image = "localhost/gaia2-oracle:latest"

[judge]
provider = "judge-provider"
model = "judge-model"

[run]
output_dir = "out"
""")

    runner = CliRunner()
    result = runner.invoke(
        main, ["run-config", "--config", str(config_path), "--dry-run"]
    )

    assert result.exit_code == 0
    assert "Dataset: facebook/omnilingual-gaia2" in result.output
    assert "Language: spa_Latn" in result.output
    assert captured["language"] == "spa_Latn"


def test_run_dataset_language_flag_is_forwarded_and_recorded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from gaia2_runner import hf_dataset

    cache_dir = tmp_path / "hf_cache"
    _write_scenario(cache_dir / "search" / "s1.json", "s1")

    captured: dict[str, object] = {}
    download_kwargs: dict[str, object] = {}

    def fake_download(dataset_id, splits=None, language=None):
        download_kwargs["splits"] = splits
        download_kwargs["language"] = language
        return str(cache_dir)

    monkeypatch.setattr(hf_dataset, "download_hf_dataset", fake_download)
    monkeypatch.setattr(
        runner_cli,
        "_build_execution_config",
        lambda **kwargs: (object(), None, None, "judge-model", "judge-provider", None),
    )
    monkeypatch.setattr(
        runner_cli,
        "_execute_dataset_selection",
        lambda **kwargs: captured.update(kwargs),
    )

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "run-dataset",
            "--dataset",
            "facebook/omnilingual-gaia2",
            "--language",
            "spa_Latn",
            "--splits",
            "search",
            "--image",
            "localhost/gaia2-oracle:latest",
            "--judge-provider",
            "judge-provider",
            "--judge-model",
            "judge-model",
            "--output-dir",
            str(tmp_path / "out"),
        ],
    )

    assert result.exit_code == 0
    assert download_kwargs["language"] == "spa_Latn"
    run_config_base = captured["run_config_base"]
    assert run_config_base["language"] == "spa_Latn"
    assert run_config_base["splits"] == ["search"]


def test_run_dataset_language_metadata_defaults_exclude_time(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from gaia2_runner import hf_dataset

    cache_dir = tmp_path / "hf_cache"
    _write_scenario(cache_dir / "search" / "s1.json", "s1")

    captured: dict[str, object] = {}

    monkeypatch.setattr(
        hf_dataset,
        "download_hf_dataset",
        lambda dataset_id, splits=None, language=None: str(cache_dir),
    )
    monkeypatch.setattr(
        runner_cli,
        "_build_execution_config",
        lambda **kwargs: (object(), None, None, "judge-model", "judge-provider", None),
    )
    monkeypatch.setattr(
        runner_cli,
        "_execute_dataset_selection",
        lambda **kwargs: captured.update(kwargs),
    )

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "run-dataset",
            "--dataset",
            "facebook/omnilingual-gaia2",
            "--language",
            "spa_Latn",
            "--image",
            "localhost/gaia2-oracle:latest",
            "--judge-provider",
            "judge-provider",
            "--judge-model",
            "judge-model",
            "--output-dir",
            str(tmp_path / "out"),
        ],
    )

    assert result.exit_code == 0
    run_config_base = captured["run_config_base"]
    assert run_config_base["splits"] == list(MULTILINGUAL_SPLITS)
    assert "time" not in run_config_base["splits"]


def test_run_dataset_rejects_language_for_local_dataset_path(tmp_path: Path) -> None:
    _write_scenario(tmp_path / "dataset" / "search" / "s1.json", "s1")

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "run-dataset",
            "--dataset",
            str(tmp_path / "dataset"),
            "--language",
            "spa_Latn",
            "--image",
            "localhost/gaia2-oracle:latest",
            "--judge-provider",
            "judge-provider",
            "--judge-model",
            "judge-model",
        ],
    )

    assert result.exit_code != 0
    assert "--language is only supported for HuggingFace dataset IDs" in result.output
