# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Tests for sequence files, chain execution and checkpoints."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import click
import pytest
from gaia2_runner.runner import CarriedState
from gaia2_runner.sequential import (
    Chain,
    ChainTask,
    checkpoint_path,
    load_sequences,
    resolve_chains,
    restore_checkpoint,
    resume_position,
    run_chain,
    save_checkpoint,
)

SEQUENCES_DIR = Path(__file__).resolve().parents[1] / "sequences"


def _entry(universe_id: str, *scenario_ids: str, cap: str | None = "execution") -> dict:
    entry: dict[str, Any] = {
        "universe_id": universe_id,
        "scenarios": [{"scenario_id": sid} for sid in scenario_ids],
    }
    if cap:
        entry["cap"] = cap
    return entry


def _write_sequences(path: Path, *entries: dict) -> Path:
    path.write_text("".join(json.dumps(entry) + "\n" for entry in entries))
    return path


def _chain(tmp_path: Path, *scenario_ids: str, universe_id: str = "21") -> Chain:
    sequences = load_sequences(
        _write_sequences(tmp_path / "seq.jsonl", _entry(universe_id, *scenario_ids))
    )
    paths = {sid: tmp_path / f"{sid}.json" for sid in scenario_ids}
    [chain] = resolve_chains(sequences, paths)
    return chain


def _marker(directory: Path | None) -> str | None:
    if directory is None:
        return None
    return (directory / "marker").read_text()


class FakeScenarios:
    """Stands in for a container run: leaves carried state tagged with its name."""

    def __init__(self, outcomes: dict[str, bool | None] | None = None) -> None:
        self.outcomes = outcomes or {}
        self.calls: list[dict[str, Any]] = []

    def __call__(
        self,
        scenario_path: Path,
        carry_in: CarriedState | None,
        carry_out: CarriedState,
        result_metadata: dict[str, Any],
    ) -> dict[str, Any]:
        name = scenario_path.stem
        self.calls.append(
            {
                "scenario": name,
                "home": _marker(carry_in.home_dir) if carry_in else None,
                "app_state": _marker(carry_in.app_state_dir) if carry_in else None,
                "metadata": result_metadata,
            }
        )
        for directory in (carry_out.home_dir, carry_out.app_state_dir):
            assert directory is not None and not directory.exists()
            directory.mkdir()
            (directory / "marker").write_text(name)
        return {"scenario_id": name, "success": self.outcomes.get(name, True)}


class TestLoadSequences:
    def test_parses_chains_in_file_order(self, tmp_path: Path) -> None:
        path = _write_sequences(
            tmp_path / "seq.jsonl",
            _entry("22", "s_b", "s_a"),
            _entry("7", "s_c", cap=None),
        )

        sequences = load_sequences(path)

        assert [s.key for s in sequences] == ["execution_universe_22", "universe_7"]
        assert sequences[0].scenario_ids == ("s_b", "s_a")
        assert sequences[1].universe_id == "7"

    @pytest.mark.parametrize(
        ("split", "chains", "scenarios"),
        [("search", 10, 160), ("execution", 10, 95)],
    )
    def test_committed_sequence_files(
        self, split: str, chains: int, scenarios: int
    ) -> None:
        sequences = load_sequences(SEQUENCES_DIR / f"{split}.jsonl")

        assert len(sequences) == chains
        assert sum(len(s.scenario_ids) for s in sequences) == scenarios
        assert {s.cap for s in sequences} == {split}

    @pytest.mark.parametrize(
        "line",
        [
            "{not json",
            json.dumps(["not", "an", "object"]),
            json.dumps({"scenarios": [{"scenario_id": "s1"}]}),
            json.dumps({"universe_id": "21", "scenarios": []}),
            json.dumps({"universe_id": "21", "scenarios": [{"id": "s1"}]}),
        ],
    )
    def test_rejects_malformed_lines(self, tmp_path: Path, line: str) -> None:
        path = tmp_path / "seq.jsonl"
        path.write_text(line + "\n")

        with pytest.raises(click.UsageError, match="seq.jsonl:1"):
            load_sequences(path)

    @pytest.mark.parametrize(
        ("entries", "message"),
        [
            ((_entry("21", "s1"), _entry("21", "s2")), "duplicate chain"),
            ((_entry("21", "s1"), _entry("22", "s1")), "more than once: s1"),
            ((_entry("21", "s1", "s1"),), "more than once: s1"),
        ],
    )
    def test_rejects_repeated_chains_and_scenarios(
        self, tmp_path: Path, entries: tuple[dict, ...], message: str
    ) -> None:
        path = _write_sequences(tmp_path / "seq.jsonl", *entries)

        with pytest.raises(click.UsageError, match=message):
            load_sequences(path)

    def test_rejects_empty_file(self, tmp_path: Path) -> None:
        path = tmp_path / "seq.jsonl"
        path.write_text("\n")

        with pytest.raises(click.UsageError, match="no chains"):
            load_sequences(path)


class TestResolveChains:
    def test_maps_scenario_ids_to_files_in_chain_order(self, tmp_path: Path) -> None:
        sequences = load_sequences(
            _write_sequences(tmp_path / "seq.jsonl", _entry("21", "s2", "s1"))
        )
        paths = {"s1": tmp_path / "a.json", "s2": tmp_path / "b.json"}

        [chain] = resolve_chains(sequences, paths)

        assert chain.key == "execution_universe_21"
        assert chain.scenario_ids == ("s2", "s1")
        assert chain.scenario_paths == (paths["s2"], paths["s1"])

    def test_limit_keeps_the_start_of_every_chain(self, tmp_path: Path) -> None:
        sequences = load_sequences(
            _write_sequences(
                tmp_path / "seq.jsonl",
                _entry("21", "s1", "s2", "not_downloaded"),
                _entry("22", "s3", "s4"),
            )
        )
        paths = {sid: tmp_path / f"{sid}.json" for sid in ("s1", "s2", "s3", "s4")}

        chains = resolve_chains(sequences, paths, limit=2)

        assert [chain.scenario_ids for chain in chains] == [("s1", "s2"), ("s3", "s4")]

    def test_reports_scenarios_missing_from_the_dataset(self, tmp_path: Path) -> None:
        sequences = load_sequences(
            _write_sequences(tmp_path / "seq.jsonl", _entry("21", "s1", "gone"))
        )

        with pytest.raises(click.UsageError, match="1 scenario.*: gone"):
            resolve_chains(sequences, {"s1": tmp_path / "s1.json"})


class TestCheckpoints:
    def test_round_trip_keeps_files_and_absolute_symlinks(self, tmp_path: Path) -> None:
        home = tmp_path / "home"
        (home / ".openclaw" / "agents" / "main" / "sessions").mkdir(parents=True)
        (home / ".openclaw" / "agents" / "main" / "sessions" / "s.jsonl").write_text(
            "{}"
        )
        (home / "bin").mkdir()
        (home / "bin" / "calendar").symlink_to("/usr/local/bin/gaia2-exec")
        app_state = tmp_path / "app_state"
        (app_state / "filesystem").mkdir(parents=True)
        (app_state / "Calendar.json").write_text('{"events": []}')
        (app_state / "filesystem" / "notes.txt").write_text("hi")
        archive = tmp_path / "ckpt" / "0000_s1.tar.gz"

        save_checkpoint(archive, CarriedState(home_dir=home, app_state_dir=app_state))
        restored = restore_checkpoint(archive, tmp_path / "restored")

        assert restored.home_dir is not None and restored.app_state_dir is not None
        session = restored.home_dir / ".openclaw/agents/main/sessions/s.jsonl"
        assert session.read_text() == "{}"
        assert (restored.home_dir / "bin" / "calendar").readlink() == Path(
            "/usr/local/bin/gaia2-exec"
        )
        assert (restored.app_state_dir / "Calendar.json").read_text() == (
            '{"events": []}'
        )
        assert (restored.app_state_dir / "filesystem" / "notes.txt").read_text() == "hi"

    def test_home_only_checkpoint_restores_no_app_state(self, tmp_path: Path) -> None:
        home = tmp_path / "home"
        home.mkdir()
        archive = tmp_path / "0000_s1.tar.gz"

        save_checkpoint(archive, CarriedState(home_dir=home))
        restored = restore_checkpoint(archive, tmp_path / "restored")

        assert restored.home_dir is not None
        assert restored.app_state_dir is None


class TestResumePosition:
    def test_resumes_at_first_scenario_without_verdict(self, tmp_path: Path) -> None:
        chain = _chain(tmp_path, "s1", "s2", "s3")
        run_dir = tmp_path / "out"
        save_checkpoint(checkpoint_path(run_dir, chain, 0), CarriedState())

        position = resume_position(chain, run_dir, has_verdict=lambda p: p < 1)

        assert position == 1

    def test_falls_back_to_latest_checkpoint(self, tmp_path: Path) -> None:
        chain = _chain(tmp_path, "s1", "s2", "s3")
        run_dir = tmp_path / "out"
        save_checkpoint(checkpoint_path(run_dir, chain, 0), CarriedState())

        position = resume_position(chain, run_dir, has_verdict=lambda p: p < 2)

        assert position == 1

    def test_restarts_without_checkpoints(self, tmp_path: Path) -> None:
        chain = _chain(tmp_path, "s1", "s2")

        position = resume_position(chain, tmp_path, has_verdict=lambda p: p < 1)

        assert position == 0

    def test_complete_chain_has_nothing_to_resume(self, tmp_path: Path) -> None:
        chain = _chain(tmp_path, "s1", "s2")

        position = resume_position(chain, tmp_path, has_verdict=lambda p: True)

        assert position == 2


class TestRunChain:
    def test_carries_home_and_app_state_to_the_next_scenario(
        self, tmp_path: Path
    ) -> None:
        chain = _chain(tmp_path, "s1", "s2", "s3")
        run_dir = tmp_path / "out"
        scenarios = FakeScenarios()
        results: list[dict[str, Any]] = []

        run_chain(
            ChainTask(run_dir=run_dir, chain=chain),
            run_scenario=scenarios,
            on_result=results.append,
        )

        assert [
            (c["scenario"], c["home"], c["app_state"]) for c in scenarios.calls
        ] == [
            ("s1", None, None),
            ("s2", "s1", "s1"),
            ("s3", "s2", "s2"),
        ]
        assert [c["metadata"] for c in scenarios.calls] == [
            {"universe_id": "21", "universe_position": position}
            for position in range(3)
        ]
        assert [r["scenario_id"] for r in results] == ["s1", "s2", "s3"]
        for position in range(3):
            assert checkpoint_path(run_dir, chain, position).is_file()

    def test_errored_scenario_carries_home_but_not_app_state(
        self, tmp_path: Path
    ) -> None:
        chain = _chain(tmp_path, "s1", "s2", "s3")
        run_dir = tmp_path / "out"
        scenarios = FakeScenarios({"s2": None})

        run_chain(
            ChainTask(run_dir=run_dir, chain=chain),
            run_scenario=scenarios,
            on_result=lambda result: None,
        )

        third = scenarios.calls[2]
        assert (third["home"], third["app_state"]) == ("s2", "s1")
        assert not checkpoint_path(run_dir, chain, 1).exists()
        assert checkpoint_path(run_dir, chain, 2).is_file()

    def test_resumes_from_the_previous_checkpoint(self, tmp_path: Path) -> None:
        chain = _chain(tmp_path, "s1", "s2", "s3")
        run_dir = tmp_path / "out"
        run_chain(
            ChainTask(run_dir=run_dir, chain=chain),
            run_scenario=FakeScenarios(),
            on_result=lambda result: None,
        )
        resumed = FakeScenarios()

        run_chain(
            ChainTask(run_dir=run_dir, chain=chain, start=2),
            run_scenario=resumed,
            on_result=lambda result: None,
        )

        assert [(c["scenario"], c["home"], c["app_state"]) for c in resumed.calls] == [
            ("s3", "s2", "s2")
        ]
