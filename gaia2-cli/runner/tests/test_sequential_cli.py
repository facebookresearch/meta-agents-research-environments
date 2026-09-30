# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Tests for sequential runs through the runner CLI."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

import click
import pytest
from click.testing import CliRunner
from gaia2_runner import cli as runner_cli
from gaia2_runner.cli import _build_container_env, main
from gaia2_runner.runner import CarriedState
from gaia2_runner.sequential import (
    Chain,
    checkpoint_path,
    load_sequences,
    resolve_chains,
)


def _write_scenario(path: Path, scenario_id: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {"metadata": {"definition": {"scenario_id": scenario_id}}, "events": []}
        )
    )
    return path


def _write_sequences(path: Path, *chains: tuple[str, list[str]]) -> Path:
    path.write_text(
        "".join(
            json.dumps(
                {
                    "universe_id": universe_id,
                    "cap": "execution",
                    "scenarios": [{"scenario_id": sid} for sid in scenario_ids],
                }
            )
            + "\n"
            for universe_id, scenario_ids in chains
        )
    )
    return path


def _chains(tmp_path: Path, *chains: tuple[str, list[str]]) -> tuple[list[Chain], Path]:
    dataset_root = tmp_path / "dataset"
    paths = {
        sid: _write_scenario(dataset_root / "execution" / f"{sid}.json", sid)
        for _, scenario_ids in chains
        for sid in scenario_ids
    }
    sequences = load_sequences(_write_sequences(tmp_path / "seq.jsonl", *chains))
    return resolve_chains(sequences, paths), dataset_root


class FakeExecutionConfig:
    """Runs scenarios without containers, the way ContainerRunner reports them."""

    image = "localhost/gaia2-oc:latest"

    def __init__(self, outcomes: dict[str, bool | None] | None = None) -> None:
        self.outcomes = outcomes or {}
        self.calls: list[tuple[str, str | None]] = []
        self._lock = threading.Lock()

    def create_runner(self, adapter_port: int) -> object:
        return object()

    def run_with_runner(
        self,
        runner: object,
        scenario_path: Path,
        *,
        output_dir: str,
        gateway_port: int,
        carry_in: CarriedState | None,
        carry_out: CarriedState,
        result_metadata: dict[str, Any],
    ) -> dict[str, Any]:
        scenario_id = scenario_path.stem
        carried_home = (
            (carry_in.home_dir / "marker").read_text()
            if carry_in and carry_in.home_dir
            else None
        )
        with self._lock:
            self.calls.append((scenario_id, carried_home))
        for directory in (carry_out.home_dir, carry_out.app_state_dir):
            assert directory is not None
            directory.mkdir()
            (directory / "marker").write_text(scenario_id)
        result = {
            "scenario_id": scenario_id,
            "success": self.outcomes.get(scenario_id, True),
            "scenario_file": str(scenario_path),
            **result_metadata,
        }
        artifact_dir = Path(output_dir) / scenario_id
        artifact_dir.mkdir(parents=True, exist_ok=True)
        (artifact_dir / "result.json").write_text(json.dumps(result))
        return result


@pytest.fixture(autouse=True)
def _quiet_reporting(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runner_cli, "_save_dataset_run_config", lambda *a, **k: None)
    monkeypatch.setattr(
        runner_cli, "_generate_trace_viewer_if_possible", lambda *a, **k: None
    )
    monkeypatch.setattr(runner_cli, "generate_runs_index", lambda *a, **k: None)


def _run(
    chains: list[Chain],
    dataset_root: Path,
    output_dir: Path,
    execution_config: FakeExecutionConfig,
    *,
    pass_at: int = 1,
    retry: bool = False,
) -> None:
    runner_cli._execute_sequential_selection(
        chains=chains,
        dataset_root=dataset_root,
        execution_config=execution_config,  # type: ignore[arg-type]
        concurrency=2,
        output_dir=str(output_dir),
        output_file=str(output_dir / "results.jsonl"),
        pass_at=pass_at,
        retry=retry,
        run_config_base={},
    )


def _results(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines()]


class TestContextWindow:
    def test_is_passed_to_openclaw(self) -> None:
        env = _build_container_env(
            "localhost/gaia2-oc:latest", None, "low", context_window=64000
        )

        assert env["OPENCLAW_CONTEXT_WINDOW"] == "64000"

    def test_is_rejected_for_other_runtimes(self) -> None:
        with pytest.raises(click.UsageError, match="only supported for OpenClaw"):
            _build_container_env(
                "localhost/gaia2-hermes:latest", None, "low", context_window=64000
            )


class TestExecuteSequentialSelection:
    def test_runs_each_chain_in_order_with_carried_state(self, tmp_path: Path) -> None:
        chains, dataset_root = _chains(tmp_path, ("21", ["s2", "s1"]), ("22", ["s3"]))
        output_dir = tmp_path / "out"
        execution_config = FakeExecutionConfig()

        _run(chains, dataset_root, output_dir, execution_config)

        first_chain_calls = [c for c in execution_config.calls if c[0] in ("s1", "s2")]
        assert first_chain_calls == [("s2", None), ("s1", "s2")]
        results = {r["scenario_id"]: r for r in _results(output_dir / "results.jsonl")}
        assert {
            sid: (r["universe_id"], r["universe_position"])
            for sid, r in results.items()
        } == {
            "s2": ("21", 0),
            "s1": ("21", 1),
            "s3": ("22", 0),
        }
        assert (output_dir / "execution" / "s1" / "result.json").is_file()
        for chain in chains:
            for position in range(len(chain.scenario_ids)):
                assert checkpoint_path(output_dir, chain, position).is_file()

    def test_retry_resumes_each_chain_at_its_first_error(self, tmp_path: Path) -> None:
        chains, dataset_root = _chains(
            tmp_path, ("21", ["s1", "s2", "s3"]), ("22", ["s4"])
        )
        output_dir = tmp_path / "out"
        _run(chains, dataset_root, output_dir, FakeExecutionConfig({"s2": None}))
        retried = FakeExecutionConfig()

        _run(chains, dataset_root, output_dir, retried, retry=True)

        assert retried.calls == [("s2", "s1"), ("s3", "s2")]
        results = _results(output_dir / "results.jsonl")
        assert sorted(r["scenario_id"] for r in results) == ["s1", "s2", "s3", "s4"]
        assert all(r["success"] is True for r in results)

    def test_retry_of_a_complete_run_runs_nothing(self, tmp_path: Path) -> None:
        chains, dataset_root = _chains(tmp_path, ("21", ["s1", "s2"]))
        output_dir = tmp_path / "out"
        _run(chains, dataset_root, output_dir, FakeExecutionConfig({"s2": False}))
        retried = FakeExecutionConfig()

        _run(chains, dataset_root, output_dir, retried, retry=True)

        assert retried.calls == []

    def test_pass_at_replays_every_chain_in_its_own_run_dir(
        self, tmp_path: Path
    ) -> None:
        chains, dataset_root = _chains(tmp_path, ("21", ["s1", "s2"]))
        output_dir = tmp_path / "out"
        execution_config = FakeExecutionConfig()

        _run(chains, dataset_root, output_dir, execution_config, pass_at=2)

        assert sorted(execution_config.calls) == [
            ("s1", None),
            ("s1", None),
            ("s2", "s1"),
            ("s2", "s1"),
        ]
        for run in ("run_1", "run_2"):
            run_dir = output_dir / run
            assert len(_results(run_dir / "results.jsonl")) == 2
            assert checkpoint_path(run_dir, chains[0], 1).is_file()

    def test_requires_an_output_dir(self, tmp_path: Path) -> None:
        chains, dataset_root = _chains(tmp_path, ("21", ["s1"]))

        with pytest.raises(click.UsageError, match="require --output-dir"):
            runner_cli._execute_sequential_selection(
                chains=chains,
                dataset_root=dataset_root,
                execution_config=FakeExecutionConfig(),  # type: ignore[arg-type]
                concurrency=1,
                output_dir=None,
                output_file=None,
                pass_at=1,
                retry=False,
                run_config_base={},
            )


class TestCommands:
    @pytest.fixture
    def captured(self, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
        captured: dict[str, Any] = {}
        monkeypatch.setattr(
            runner_cli,
            "_build_execution_config",
            lambda **kwargs: (
                object(),
                None,
                None,
                "judge-model",
                "judge-provider",
                None,
            ),
        )
        monkeypatch.setattr(
            runner_cli,
            "_execute_sequential_selection",
            lambda **kwargs: captured.update(kwargs),
        )
        return captured

    def test_run_dataset_chains_the_dataset(
        self, tmp_path: Path, captured: dict[str, Any]
    ) -> None:
        for sid in ("s1", "s2", "s3"):
            _write_scenario(tmp_path / "dataset" / "execution" / f"{sid}.json", sid)
        sequences = _write_sequences(tmp_path / "seq.jsonl", ("21", ["s3", "s1", "s2"]))

        result = CliRunner().invoke(
            main,
            [
                "run-dataset",
                "--dataset",
                str(tmp_path / "dataset"),
                "--image",
                "localhost/gaia2-oc:latest",
                "--sequences",
                str(sequences),
                "--limit",
                "2",
                "--output-dir",
                str(tmp_path / "out"),
            ],
        )

        assert result.exit_code == 0, result.output
        [chain] = captured["chains"]
        assert chain.scenario_ids == ("s3", "s1")
        assert captured["run_config_base"]["sequences"] == str(sequences)

    @pytest.mark.parametrize(
        ("extra_args", "message"),
        [
            (
                ["--image", "localhost/gaia2-hermes:latest"],
                "requires an OpenClaw image",
            ),
            (
                ["--image", "localhost/gaia2-oc:latest", "--subset", "SUBSET"],
                "cannot be combined with --subset",
            ),
        ],
    )
    def test_run_dataset_rejects_unsupported_setups(
        self,
        tmp_path: Path,
        captured: dict[str, Any],
        extra_args: list[str],
        message: str,
    ) -> None:
        _write_scenario(tmp_path / "dataset" / "execution" / "s1.json", "s1")
        sequences = _write_sequences(tmp_path / "seq.jsonl", ("21", ["s1"]))
        subset = tmp_path / "subset.json"
        subset.write_text(json.dumps({"splits": {}}))
        args = [str(subset) if arg == "SUBSET" else arg for arg in extra_args]

        result = CliRunner().invoke(
            main,
            [
                "run-dataset",
                "--dataset",
                str(tmp_path / "dataset"),
                "--sequences",
                str(sequences),
                *args,
            ],
        )

        assert result.exit_code != 0
        assert message in result.output
        assert captured == {}

    def test_run_config_forwards_resolved_chains(
        self, tmp_path: Path, captured: dict[str, Any]
    ) -> None:
        for sid in ("s1", "s2"):
            _write_scenario(tmp_path / "dataset" / "execution" / f"{sid}.json", sid)
        _write_sequences(tmp_path / "seq.jsonl", ("21", ["s2", "s1"]))
        config_path = tmp_path / "eval.toml"
        config_path.write_text("""
[target]
dataset_root = "dataset"
splits = ["execution"]
sequences = "seq.jsonl"

[agent]
image = "localhost/gaia2-oc:latest"
provider = "anthropic"
model = "claude-sonnet-4-6"

[judge]
provider = "judge-provider"
model = "judge-model"

[run]
output_dir = "out"
pass_at = 3
""")

        result = CliRunner().invoke(main, ["run-config", "--config", str(config_path)])

        assert result.exit_code == 0, result.output
        [chain] = captured["chains"]
        assert chain.scenario_ids == ("s2", "s1")
        assert captured["pass_at"] == 3
        assert captured["output_dir"] == str((tmp_path / "out").resolve())
