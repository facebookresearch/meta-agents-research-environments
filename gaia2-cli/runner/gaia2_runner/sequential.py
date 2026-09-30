# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Sequential evaluation: scenarios of a universe chained with carried state.

A sequence file lists, one JSON object per line, the scenarios of a universe
in the order they are chained::

    {"universe_id": "21", "cap": "execution", "scenarios": [{"scenario_id": "..."}]}

Every scenario still runs in a fresh container, but the agent's home
directory (memory and conversation session) and the app state it leaves
behind are carried into the next container of its chain. After each judged
scenario the carried state is checkpointed, so ``--retry`` can resume a chain
at its first scenario without a verdict.
"""

from __future__ import annotations

import json
import shutil
import tarfile
import tempfile
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import click

from gaia2_runner.runner import CarriedState

SEQUENTIAL_STATE_DIRNAME = "sequential_state"
_HOME_ARCNAME = "home"
_APP_STATE_ARCNAME = "app_state"

ScenarioRunFn = Callable[
    [Path, CarriedState | None, CarriedState, dict[str, Any]], dict[str, Any]
]
"""Runs one scenario: ``(scenario_path, carry_in, carry_out, result_metadata)``."""


@dataclass(frozen=True, slots=True)
class Sequence:
    """One chain as listed in a sequence file."""

    universe_id: str
    cap: str | None
    scenario_ids: tuple[str, ...]

    @property
    def key(self) -> str:
        """Name of the chain, unique within a sequence file."""
        universe = f"universe_{self.universe_id}"
        return f"{self.cap}_{universe}" if self.cap else universe


@dataclass(frozen=True, slots=True)
class Chain:
    """A sequence resolved to scenario files."""

    universe_id: str
    key: str
    scenario_ids: tuple[str, ...]
    scenario_paths: tuple[Path, ...]


@dataclass(frozen=True, slots=True)
class ChainTask:
    """A chain to run in one pass output directory, from position *start*."""

    run_dir: Path
    chain: Chain
    start: int = 0

    @property
    def remaining(self) -> int:
        return len(self.chain.scenario_ids) - self.start


def load_sequences(path: str | Path) -> list[Sequence]:
    """Parse a sequence file into chains of scenario IDs."""
    sequences: list[Sequence] = []
    seen_keys: set[str] = set()
    seen_ids: set[str] = set()
    for line_number, line in enumerate(Path(path).read_text().splitlines(), start=1):
        if not line.strip():
            continue
        where = f"{path}:{line_number}"
        try:
            entry = json.loads(line)
        except json.JSONDecodeError as exc:
            raise click.UsageError(f"{where}: invalid JSON ({exc})") from exc
        if not isinstance(entry, dict):
            raise click.UsageError(f"{where}: expected a JSON object")

        universe_id = entry.get("universe_id")
        scenarios = entry.get("scenarios")
        if (
            universe_id in (None, "")
            or not isinstance(scenarios, list)
            or not scenarios
        ):
            raise click.UsageError(
                f"{where}: expected a 'universe_id' and a non-empty 'scenarios' list"
            )
        scenario_ids = tuple(
            item.get("scenario_id") if isinstance(item, dict) else None
            for item in scenarios
        )
        if not all(isinstance(sid, str) and sid for sid in scenario_ids):
            raise click.UsageError(
                f"{where}: every entry in 'scenarios' needs a 'scenario_id'"
            )

        sequence = Sequence(
            universe_id=str(universe_id),
            cap=entry.get("cap") or None,
            scenario_ids=scenario_ids,
        )
        if sequence.key in seen_keys:
            raise click.UsageError(f"{where}: duplicate chain {sequence.key}")
        counts = Counter(scenario_ids)
        repeated = sorted(sid for sid in counts if counts[sid] > 1 or sid in seen_ids)
        if repeated:
            raise click.UsageError(
                f"{where}: scenarios listed more than once: {', '.join(repeated)}"
            )
        seen_keys.add(sequence.key)
        seen_ids.update(scenario_ids)
        sequences.append(sequence)

    if not sequences:
        raise click.UsageError(f"Sequence file lists no chains: {path}")
    return sequences


def resolve_chains(
    sequences: list[Sequence],
    scenarios_by_id: Mapping[str, Path],
    *,
    limit: int | None = None,
) -> list[Chain]:
    """Map sequences to scenario files, keeping each chain's first *limit*."""
    selected = [sequence.scenario_ids[:limit] for sequence in sequences]
    missing = [sid for ids in selected for sid in ids if sid not in scenarios_by_id]
    if missing:
        preview = ", ".join(missing[:5])
        if len(missing) > 5:
            preview += f" (+{len(missing) - 5} more)"
        raise click.UsageError(
            f"{len(missing)} scenario(s) from the sequence file are not in the "
            f"selected dataset: {preview}"
        )
    return [
        Chain(
            universe_id=sequence.universe_id,
            key=sequence.key,
            scenario_ids=ids,
            scenario_paths=tuple(scenarios_by_id[sid] for sid in ids),
        )
        for sequence, ids in zip(sequences, selected)
    ]


def checkpoint_path(run_dir: Path, chain: Chain, position: int) -> Path:
    """Where the state carried out of *position* of *chain* is archived."""
    scenario_id = chain.scenario_ids[position]
    return (
        run_dir
        / SEQUENTIAL_STATE_DIRNAME
        / chain.key
        / f"{position:04d}_{scenario_id}.tar.gz"
    )


def save_checkpoint(path: Path, state: CarriedState) -> None:
    """Archive carried state; the archive only appears once complete."""
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    with tarfile.open(partial, "w:gz") as archive:
        if state.home_dir is not None:
            archive.add(state.home_dir, arcname=_HOME_ARCNAME)
        if state.app_state_dir is not None:
            archive.add(state.app_state_dir, arcname=_APP_STATE_ARCNAME)
    partial.replace(path)


def restore_checkpoint(path: Path, work_dir: Path) -> CarriedState:
    """Unpack a checkpoint into *work_dir* and return the carried state."""
    work_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, "r:gz") as archive:
        # The agent home holds absolute symlinks (bin/ → the setuid tool
        # wrapper), which the stricter "data" filter would refuse.
        archive.extractall(work_dir, filter="tar")
    home_dir = work_dir / _HOME_ARCNAME
    app_state_dir = work_dir / _APP_STATE_ARCNAME
    return CarriedState(
        home_dir=home_dir if home_dir.is_dir() else None,
        app_state_dir=app_state_dir if app_state_dir.is_dir() else None,
    )


def resume_position(
    chain: Chain,
    run_dir: Path,
    *,
    has_verdict: Callable[[int], bool],
) -> int:
    """Return where ``--retry`` resumes *chain*; ``len(chain)`` means done.

    A chain resumes at its first scenario without a verdict, from the
    checkpoint of the scenario before it. If that checkpoint is missing it
    falls back to the latest earlier one, or restarts the chain.
    """
    length = len(chain.scenario_ids)
    position = next((p for p in range(length) if not has_verdict(p)), length)
    if position == length:
        return length
    while position > 0 and not checkpoint_path(run_dir, chain, position - 1).exists():
        position -= 1
    return position


def run_chain(
    task: ChainTask,
    *,
    run_scenario: ScenarioRunFn,
    on_result: Callable[[dict[str, Any]], None],
) -> None:
    """Run a chain's scenarios in order, carrying state from one to the next.

    The agent home carries over whenever a scenario produced one. The app
    state only advances on a verdict, so an errored scenario leaves it as it
    was. Every judged scenario is checkpointed.
    """
    chain = task.chain
    with tempfile.TemporaryDirectory(prefix="gaia2-chain-") as tmp:
        work_dir = Path(tmp)
        carried = CarriedState()
        if task.start > 0:
            carried = restore_checkpoint(
                checkpoint_path(task.run_dir, chain, task.start - 1),
                work_dir / "restored",
            )

        for position in range(task.start, len(chain.scenario_ids)):
            carry_out = CarriedState(
                home_dir=work_dir / f"{position:04d}_home",
                app_state_dir=work_dir / f"{position:04d}_app_state",
            )
            has_carried = (
                carried.home_dir is not None or carried.app_state_dir is not None
            )
            result = run_scenario(
                chain.scenario_paths[position],
                carried if has_carried else None,
                carry_out,
                {"universe_id": chain.universe_id, "universe_position": position},
            )

            judged = result.get("success") is not None
            app_state_dir = carried.app_state_dir
            if judged:
                app_state_dir = _captured(carry_out.app_state_dir) or app_state_dir
            next_carried = CarriedState(
                home_dir=_captured(carry_out.home_dir) or carried.home_dir,
                app_state_dir=app_state_dir,
            )
            _discard_unused((carried, carry_out), keep=next_carried)
            carried = next_carried

            if judged:
                save_checkpoint(checkpoint_path(task.run_dir, chain, position), carried)
            on_result(result)


def _captured(path: Path | None) -> Path | None:
    return path if path is not None and path.is_dir() else None


def _discard_unused(states: tuple[CarriedState, ...], *, keep: CarriedState) -> None:
    kept = {keep.home_dir, keep.app_state_dir}
    for state in states:
        for path in (state.home_dir, state.app_state_dir):
            if path is not None and path not in kept:
                shutil.rmtree(path, ignore_errors=True)
