# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""CLI entry point for running Gaia2 scenarios in containers.

Usage:
    gaia2-runner run \\
        --scenario /path/to/scenario.json \\
        --image gaia2-oc:latest

    gaia2-runner run-dataset \\
        --dataset /path/to/scenarios/ \\
        --image gaia2-oc:latest \\
        --limit 5
"""

from __future__ import annotations

import json
import logging
import math
import os
import shutil
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import click
from tqdm import tqdm

from gaia2_runner.envfile import load_env_file
from gaia2_runner.launcher import (
    ContainerLauncher,
    LocalLauncher,
    _allocate_free_port,
)
from gaia2_runner.runner import ContainerRunner
from gaia2_runner.trace_viewer import (
    generate_all as generate_trace_viewer,
)
from gaia2_runner.trace_viewer import (
    generate_runs_index,
)

if TYPE_CHECKING:
    from gaia2_runner.config import RunnerTomlConfig

logger: logging.Logger = logging.getLogger(__name__)

SUMMARY_SEPARATOR = "=" * 60
SPLIT_DISPLAY_ORDER = ("adaptability", "ambiguity", "execution", "search", "time")
SPLIT_DISPLAY_ORDER_INDEX = {
    split_name: index for index, split_name in enumerate(SPLIT_DISPLAY_ORDER)
}
DATASET_PARTITION_DIR_NAMES = frozenset({"train", "validation", "test"})


def _default_env_file() -> Path:
    """Return the repo-root ``.env`` path for gaia2-cli."""
    return Path(__file__).resolve().parents[2] / ".env"


def _maybe_load_runner_env(env_file: Path | None) -> None:
    """Load dotenv-style env vars for runner secrets and network settings."""
    env_path = env_file or _default_env_file()
    explicit = env_file is not None

    if not env_path.exists():
        return

    loaded_keys = load_env_file(env_path, override=False)
    if explicit:
        if loaded_keys:
            click.secho(
                f"Loaded env vars from {env_path}",
                fg="cyan",
                err=True,
            )
        else:
            click.secho(
                f"Read env file {env_path} (existing environment kept precedence)",
                fg="cyan",
                err=True,
            )
    elif loaded_keys:
        click.secho(
            f"Loaded env vars from {env_path}",
            fg="cyan",
            err=True,
        )


class _TqdmLoggingHandler(logging.Handler):
    """Route log messages through ``tqdm.write`` so they don't break the bar."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            tqdm.write(self.format(record))
        except Exception:
            self.handleError(record)


def _stats_postfix(stats: "RunStats") -> dict[str, str]:
    """Build an ordered postfix dict for tqdm from current run stats."""
    rate = f"{stats.pass_rate_percent:.0f}%" if stats.total else "—"
    return {
        "P": str(stats.passed),
        "F": str(stats.failed),
        "E": str(stats.errors),
        "rate": rate,
    }


JsonDict = dict[str, Any]


@dataclass(frozen=True, slots=True)
class ScenarioExecutionConfig:
    """Everything needed to run a scenario inside a container."""

    image: str
    runtime: str
    timeout: int
    health_timeout: int
    container_env: dict[str, str]
    provider: str | None
    model: str | None
    api_key: str | None
    judge_model: str
    judge_provider: str
    judge_base_url: str | None
    extra_volumes: tuple[str, ...] = ()
    launcher_type: str = "podman"

    def create_runner(self, adapter_port: int) -> ContainerRunner:
        """Create a runner with a local LocalLauncher.

        For VMVM, callers must supply a launcher from a :class:`None`
        and use :meth:`run_with_runner` directly.
        """
        launcher: ContainerLauncher = LocalLauncher(runtime=self.runtime)
        return ContainerRunner(
            launcher=launcher,
            image=self.image,
            adapter_port=adapter_port,
        )

    def run_with_runner(
        self,
        runner: ContainerRunner,
        scenario_path: Path,
        *,
        output_dir: str | None = None,
        gateway_port: int | None = None,
    ) -> JsonDict:
        return runner.run_scenario(
            scenario_json_path=str(scenario_path),
            response_timeout=self.timeout,
            health_timeout=self.health_timeout,
            container_env=self.container_env,
            provider=self.provider,
            model=self.model,
            api_key=self.api_key,
            extra_volumes=self.extra_volumes,
            output_dir=output_dir,
            gateway_port=gateway_port,
        )

    def run_scenario(
        self,
        scenario_path: Path,
        *,
        adapter_port: int,
        output_dir: str | None = None,
        gateway_port: int | None = None,
    ) -> JsonDict:
        runner = self.create_runner(adapter_port)
        return self.run_with_runner(
            runner,
            scenario_path,
            output_dir=output_dir,
            gateway_port=gateway_port,
        )


@dataclass(slots=True)
class RunStats:
    """Accumulate results and summary counts for a run."""

    results: list[JsonDict] = field(default_factory=list)
    passed: int = 0
    failed: int = 0
    errors: int = 0

    def record(self, result: JsonDict) -> None:
        self.results.append(result)

        success = result.get("success")
        if success is True:
            self.passed += 1
        elif success is False:
            self.failed += 1
        else:
            self.errors += 1

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def pass_rate_percent(self) -> float:
        return (self.passed / self.total * 100) if self.total > 0 else 0.0

    @property
    def success_fraction(self) -> float:
        return (self.passed / self.total) if self.total > 0 else 0.0

    def to_summary(self, *, run_number: int | None = None) -> JsonDict:
        summary: JsonDict = {
            "passed": self.passed,
            "failed": self.failed,
            "errors": self.errors,
            "total": self.total,
            "pass_rate": self.pass_rate_percent,
        }
        if run_number is not None:
            summary["run_number"] = run_number
        return summary


def _split_sort_key(split_name: str) -> tuple[int, str]:
    return (
        SPLIT_DISPLAY_ORDER_INDEX.get(split_name, len(SPLIT_DISPLAY_ORDER)),
        split_name,
    )


def _infer_result_split(
    result: JsonDict,
    *,
    dataset_root: Path | None = None,
) -> str | None:
    scenario_file = result.get("scenario_file")
    if not isinstance(scenario_file, str) or not scenario_file.strip():
        return None

    scenario_path = Path(scenario_file).resolve()
    if dataset_root is not None:
        resolved_root = dataset_root.resolve()
        try:
            relative_path = scenario_path.relative_to(resolved_root)
        except ValueError:
            relative_path = None
        if relative_path is not None:
            if len(relative_path.parts) >= 2:
                return relative_path.parts[0]
            root_name = resolved_root.name
            if root_name in SPLIT_DISPLAY_ORDER_INDEX:
                return root_name
            if (
                root_name in DATASET_PARTITION_DIR_NAMES
                and resolved_root.parent.name in SPLIT_DISPLAY_ORDER_INDEX
            ):
                return resolved_root.parent.name
            return None

    parent_name = scenario_path.parent.name
    if not parent_name:
        return None
    if parent_name in DATASET_PARTITION_DIR_NAMES and scenario_path.parent.parent.name:
        return scenario_path.parent.parent.name
    return parent_name


def _summarize_results_by_split(
    results: list[JsonDict],
    *,
    dataset_root: Path | None = None,
) -> dict[str, RunStats]:
    split_stats: dict[str, RunStats] = {}
    for result in results:
        split_name = _infer_result_split(result, dataset_root=dataset_root)
        if not split_name:
            continue
        split_stats.setdefault(split_name, RunStats()).record(result)

    return {
        split_name: split_stats[split_name]
        for split_name in sorted(split_stats, key=_split_sort_key)
    }


def _split_summary_dict(split_stats: dict[str, RunStats]) -> JsonDict:
    return {split_name: stats.to_summary() for split_name, stats in split_stats.items()}


@dataclass(frozen=True, slots=True)
class RetrySelection:
    """Subset of scenarios that should be re-run in --retry mode."""

    scenarios: list[Path]
    passed: int
    failed: int

    @property
    def rerun_count(self) -> int:
        return len(self.scenarios)


@dataclass(frozen=True, slots=True)
class MultiRunRetrySelection:
    """Specific (scenario, run_number) pairs to re-run in pass@N --retry mode."""

    tasks: list[tuple[Path, int]]
    passed: int
    failed: int

    @property
    def rerun_count(self) -> int:
        return len(self.tasks)


def setup_logging(level: str = "INFO") -> None:
    handler = _TqdmLoggingHandler()
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        handlers=[handler],
    )


def _save_run_config(
    output_dir: str,
    command: str,
    *,
    config_path: str | None = None,
    dataset: str | None = None,
    dataset_cache_dir: str | None = None,
    scenario: str | None = None,
    subset: str | None = None,
    splits: list[str] | None = None,
    image: str,
    runtime: str,
    provider: str | None,
    model: str | None,
    base_url: str | None,
    judge_model: str,
    judge_provider: str,
    judge_base_url: str | None,
    timeout: int,
    health_timeout: int,
    concurrency: int,
    limit: int | None,
    num_scenarios: int | None = None,
    run_number: int | None = None,
    total_runs: int | None = None,
) -> None:
    """Write run_config.json to the output directory root."""
    try:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        run_config_path = out / "run_config.json"
        previous_config: JsonDict = {}
        if run_config_path.exists():
            try:
                previous_config = json.loads(run_config_path.read_text())
            except Exception:
                previous_config = {}

        config: JsonDict = {
            "command": command,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "user": os.environ.get("USER", ""),
            "image": image,
            "runtime": runtime,
            "provider": provider,
            "model": model,
            "base_url": base_url,
            "judge_model": judge_model,
            "judge_provider": judge_provider,
            "judge_base_url": judge_base_url,
            "timeout": timeout,
            "health_timeout": health_timeout,
            "concurrency": concurrency,
            "limit": limit,
        }

        if config_path:
            config["config_path"] = str(Path(config_path).resolve())
        if dataset:
            from gaia2_runner.hf_dataset import is_hf_dataset

            if is_hf_dataset(dataset):
                config["dataset"] = dataset
            else:
                ds = Path(dataset).resolve()
                config["dataset"] = str(ds)
                config["dataset_split"] = ds.parent.name
        if dataset_cache_dir:
            config["dataset_cache_dir"] = str(Path(dataset_cache_dir).resolve())
        if scenario:
            config["scenario"] = str(Path(scenario).resolve())
        if subset:
            config["subset"] = str(Path(subset).resolve())
        if splits:
            config["splits"] = list(splits)
        if num_scenarios is not None:
            previous_total = previous_config.get("num_scenarios")
            if isinstance(previous_total, int) and previous_total > num_scenarios:
                config["num_scenarios"] = previous_total
                config["retry_num_scenarios"] = num_scenarios
            else:
                config["num_scenarios"] = num_scenarios
        if run_number is not None:
            config["run_number"] = run_number
        if total_runs is not None:
            config["total_runs"] = total_runs

        run_config_path.write_text(json.dumps(config, indent=2, default=str) + "\n")
    except Exception as exc:
        logger.warning("Failed to save run config: %s", exc)


def _resolve_judge_config(
    judge_model: str | None,
    judge_provider: str | None,
    judge_base_url: str | None,
    judge_api_key: str | None,
) -> tuple[str, str, str | None, str | None]:
    """Resolve effective judge config from CLI args and environment.

    The benchmark always requires in-container judging. We therefore fail fast
    instead of silently running without LLM-backed soft checks.
    """
    resolved_model = (judge_model or os.environ.get("GAIA2_JUDGE_MODEL", "")).strip()
    resolved_provider = (
        judge_provider or os.environ.get("GAIA2_JUDGE_PROVIDER", "")
    ).strip()
    resolved_base_url = (
        judge_base_url or os.environ.get("GAIA2_JUDGE_BASE_URL", "")
    ).strip()
    resolved_api_key = (judge_api_key or "").strip()
    if not resolved_api_key:
        env_judge_api_key = os.environ.get("GAIA2_JUDGE_API_KEY", "").strip()
        if env_judge_api_key:
            click.secho(
                "Judge API key not passed via --judge-api-key; "
                "pulling from GAIA2_JUDGE_API_KEY",
                fg="red",
                err=True,
            )
            resolved_api_key = env_judge_api_key

    missing: list[str] = []
    if not resolved_model:
        missing.append("--judge-model / GAIA2_JUDGE_MODEL")
    if not resolved_provider:
        missing.append("--judge-provider / GAIA2_JUDGE_PROVIDER")
    if missing:
        raise click.UsageError(
            "In-container judge configuration is required for benchmark runs. "
            f"Missing: {', '.join(missing)}"
        )

    return (
        resolved_model,
        resolved_provider,
        resolved_base_url or None,
        resolved_api_key or None,
    )


def _resolve_agent_config(
    image: str,
    provider: str | None,
    model: str | None,
) -> tuple[str | None, str | None]:
    """Resolve and validate agent-side provider/model settings."""
    from .container_env import detect_profile

    profile = detect_profile(image)
    resolved_provider = (provider or "").strip() or None
    resolved_model = (model or "").strip() or None

    if not profile.requires_agent_llm:
        return None, None

    missing: list[str] = []
    if not resolved_provider:
        missing.append("--provider")
    if not resolved_model:
        missing.append("--model")
    if missing:
        raise click.UsageError(
            "Agent model configuration is required for this container image. "
            f"Missing: {', '.join(missing)}"
        )

    return resolved_provider, resolved_model


def _build_container_env(
    image: str,
    base_url: str | None,
    thinking: str,
    notification_mode: str = "message",
    time_speed: float | None = None,
    judge_model: str | None = None,
    judge_provider: str | None = None,
    judge_base_url: str | None = None,
    judge_api_key: str | None = None,
) -> dict[str, str]:
    container_env = {"THINKING": thinking}
    if notification_mode != "message":
        container_env["GAIA2_NOTIFICATION_MODE"] = notification_mode
    if time_speed is not None:
        container_env["GAIA2_TIME_SPEED"] = str(time_speed)
    if judge_model:
        container_env["GAIA2_JUDGE_MODEL"] = judge_model
    if judge_provider:
        container_env["GAIA2_JUDGE_PROVIDER"] = judge_provider
    if judge_base_url:
        container_env["GAIA2_JUDGE_BASE_URL"] = judge_base_url
    if judge_api_key:
        container_env["GAIA2_JUDGE_API_KEY"] = judge_api_key
    if base_url:
        from .container_env import detect_profile

        for key in detect_profile(image).base_url_keys:
            container_env[key] = base_url
    return container_env


def _load_subset_ids(subset_path: str) -> set[str]:
    """Load scenario IDs from a subset manifest JSON.

    The manifest has ``splits.<split_name>`` lists of scenario IDs.
    Returns the union of all scenario IDs across all splits.
    """
    with open(subset_path) as f:
        manifest = json.load(f)
    ids: set[str] = set()
    for split_ids in manifest.get("splits", {}).values():
        ids.update(split_ids)
    return ids


def _load_dataset_scenarios(
    dataset: str,
    limit: int | None,
    *,
    recursive: bool = True,
    subset: str | None = None,
) -> tuple[list[Path], Path | None, str | None]:
    """Expand a dataset path into concrete scenario files.

    Returns ``(scenario_paths, dataset_root, temporary_directory)``.

    *dataset_root* is the base directory when *dataset* is a directory (used
    by :func:`_effective_output_dir` to mirror the input structure in the
    output).  It is ``None`` for JSONL and single-file inputs.

    The temporary directory is only used when expanding a JSONL file into
    per-scenario JSON files and should be cleaned up by the caller.

    If *subset* is provided, only scenarios whose ID appears in the subset
    manifest are included.
    """
    dataset_path = Path(dataset)
    scenario_paths: list[Path]
    dataset_root: Path | None = None
    tmpdir: str | None = None

    if dataset_path.is_dir():
        dataset_root = dataset_path
        if recursive:
            candidate_paths = sorted(dataset_path.rglob("*.json"))
        else:
            candidate_paths = sorted(dataset_path.glob("*.json"))
        scenario_paths = [
            path
            for path in candidate_paths
            if _is_dataset_scenario_file(path, dataset_root)
        ]
    elif dataset_path.suffix == ".jsonl":
        scenario_paths = []
        tmpdir = tempfile.mkdtemp(prefix="gaia2-scenarios-")
        with open(dataset_path) as fh:
            for index, line in enumerate(fh):
                line = line.strip()
                if not line:
                    continue
                tmp_path = Path(tmpdir) / f"scenario_{index}.json"
                tmp_path.write_text(line)
                scenario_paths.append(tmp_path)
    else:
        scenario_paths = [dataset_path]

    if subset:
        subset_ids = _load_subset_ids(subset)
        before = len(scenario_paths)
        scenario_paths = [
            p for p in scenario_paths if _read_scenario_id(p) in subset_ids
        ]
        logger.info(
            "Subset filter: %d/%d scenarios match %s",
            len(scenario_paths),
            before,
            subset,
        )

    if limit is not None:
        scenario_paths = scenario_paths[:limit]

    return scenario_paths, dataset_root, tmpdir


def _effective_output_dir(
    output_dir: str | None,
    scenario_path: Path,
    dataset_root: Path | None,
) -> str | None:
    """Compute per-scenario output dir that mirrors the input directory structure.

    Given ``dataset_root=/data/dataset`` and
    ``scenario_path=/data/dataset/search/validation/s1.json``, returns
    ``output_dir/search/validation`` so the runner creates
    ``output_dir/search/validation/<scenario_id>/``.

    When *dataset_root* is ``None`` (JSONL or single-file input) or the
    scenario lives directly in the dataset root, returns *output_dir*
    unchanged.
    """
    if not output_dir or not dataset_root:
        return output_dir
    relative_subdir = scenario_path.parent.relative_to(dataset_root)
    if relative_subdir == Path("."):
        return output_dir
    return str(Path(output_dir) / relative_subdir)


def _is_dataset_scenario_file(path: Path, dataset_root: Path) -> bool:
    """Return whether a JSON file under a dataset root looks like a scenario."""
    try:
        relative_parts = path.relative_to(dataset_root).parts
    except ValueError:
        relative_parts = path.parts

    if any(part.startswith(".") for part in relative_parts):
        return False

    try:
        data = json.loads(path.read_text())
    except Exception:
        return False

    if not isinstance(data, dict):
        return False

    scenario_id = data.get("metadata", {}).get("definition", {}).get("scenario_id")
    events = data.get("events")
    return (
        isinstance(scenario_id, str) and bool(scenario_id) and isinstance(events, list)
    )


def _read_scenario_id(scenario_path: Path) -> str:
    """Resolve the scenario ID using the same logic as ContainerRunner."""
    try:
        data = json.loads(scenario_path.read_text())
    except Exception:
        return scenario_path.stem

    return (
        data.get("metadata", {})
        .get("definition", {})
        .get("scenario_id", scenario_path.stem)
    )


def _select_retry_scenarios(
    scenario_paths: list[Path],
    output_dir: str,
    dataset_root: Path | None = None,
) -> RetrySelection:
    """Keep only missing/errored scenarios for ``--retry`` mode."""
    rerun: list[Path] = []
    passed = 0
    failed = 0

    for scenario_path in scenario_paths:
        scenario_id = _read_scenario_id(scenario_path)
        effective_out = _effective_output_dir(output_dir, scenario_path, dataset_root)
        result_file = Path(effective_out or output_dir) / scenario_id / "result.json"
        if result_file.exists():
            try:
                previous_result = json.loads(result_file.read_text())
            except Exception:
                previous_result = None
            if previous_result is not None:
                success = previous_result.get("success")
                if success is True:
                    passed += 1
                    continue
                if success is False:
                    failed += 1
                    continue

        rerun.append(scenario_path)

    return RetrySelection(scenarios=rerun, passed=passed, failed=failed)


def _select_retry_tasks_multirun(
    scenario_paths: list[Path],
    output_dir: str,
    pass_at: int,
    dataset_root: Path | None = None,
) -> MultiRunRetrySelection:
    """Keep only missing/errored (scenario, run) pairs for pass@N --retry mode.

    Unlike ``_select_retry_scenarios`` which returns whole scenarios, this
    returns specific ``(scenario_path, run_number)`` pairs so that successful
    runs are preserved and only the errored/missing ones are retried.
    """
    out = Path(output_dir)
    rerun: list[tuple[Path, int]] = []
    passed = 0
    failed = 0

    for scenario_path in scenario_paths:
        scenario_id = _read_scenario_id(scenario_path)
        for run_number in range(1, pass_at + 1):
            run_out = str(out / f"run_{run_number}")
            effective_out = _effective_output_dir(run_out, scenario_path, dataset_root)
            result_file = Path(effective_out or run_out) / scenario_id / "result.json"
            if result_file.exists():
                try:
                    previous_result = json.loads(result_file.read_text())
                except Exception:
                    previous_result = None
                if previous_result is not None:
                    success = previous_result.get("success")
                    if success is True:
                        passed += 1
                        continue
                    if success is False:
                        failed += 1
                        continue

            rerun.append((scenario_path, run_number))

    return MultiRunRetrySelection(tasks=rerun, passed=passed, failed=failed)


def _result_status_label(result: JsonDict) -> str:
    return "PASS" if result.get("success") is True else "FAIL/ERROR"


def _result_exit_code(result: JsonDict) -> int:
    success = result.get("success")
    if success is True:
        return 0
    if success is False:
        return 1
    return 2


def _write_results_jsonl(
    output_file: str | Path,
    results: list[JsonDict],
    *,
    log_label: str = "Results",
) -> None:
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as fh:
        for result in results:
            fh.write(json.dumps(result, default=str) + "\n")
    logger.info("%s written to %s", log_label, output_path)


def _resolve_results_output_path(
    output_dir: str | None,
    output_file: str | None,
) -> Path | None:
    """Return where merged results should be written, if anywhere."""
    if output_file:
        return Path(output_file)

    if not output_dir:
        return None

    candidate = Path(output_dir) / "results.jsonl"
    if candidate.exists():
        return candidate
    return None


def _generate_trace_viewer_if_possible(
    output_dir: str | None,
    *,
    failure_message: str = "Failed to generate trace viewer",
) -> None:
    if not output_dir or not Path(output_dir).is_dir():
        return

    try:
        generate_trace_viewer(output_dir)
    except Exception as exc:
        logger.warning("%s: %s", failure_message, exc)


def _print_stats_summary(
    title: str,
    stats: RunStats,
    *,
    output_dir: str | None = None,
    split_stats: dict[str, RunStats] | None = None,
) -> None:
    print(f"\n{SUMMARY_SEPARATOR}")
    print(
        f"{title}: {stats.passed}/{stats.total} passed, "
        f"{stats.failed}/{stats.total} failed, {stats.errors}/{stats.total} errors"
    )
    if stats.total > 0:
        print(f"Success rate: {stats.success_fraction:.1%}")
    if split_stats:
        print("Per split:")
        for split_name, split_summary in split_stats.items():
            print(
                f"  {split_name}: {split_summary.passed}/{split_summary.total} passed, "
                f"{split_summary.failed}/{split_summary.total} failed, "
                f"{split_summary.errors}/{split_summary.total} errors "
                f"({split_summary.pass_rate_percent:.1f}%)"
            )
    print(SUMMARY_SEPARATOR)
    if output_dir:
        print(f"Trace viewer: {Path(output_dir).resolve()}/index.html")


def _load_run_stats_from_disk(output_dir: str) -> RunStats:
    """Rebuild RunStats from persisted per-scenario result files."""
    stats = RunStats()
    root = Path(output_dir)
    if not root.is_dir():
        return stats

    for result_file in sorted(root.rglob("result.json")):
        try:
            result = json.loads(result_file.read_text())
        except Exception:
            logger.warning("Skipping unreadable result file: %s", result_file)
            continue
        stats.record(result)
    return stats


def _print_aggregate_summary(
    per_run_stats: list[tuple[int, RunStats]],
    output_dir: str,
) -> None:
    """Print aggregate pass@N summary across completed runs."""
    completed_runs = [
        (run_number, stats) for run_number, stats in per_run_stats if stats.total > 0
    ]
    if not completed_runs:
        return

    total_weight = sum(stats.total for _, stats in completed_runs)
    mean_rate = (
        sum(stats.pass_rate_percent * stats.total for _, stats in completed_runs)
        / total_weight
        if total_weight > 0
        else 0.0
    )
    variance = (
        sum(
            stats.total * (stats.pass_rate_percent - mean_rate) ** 2
            for _, stats in completed_runs
        )
        / total_weight
        if total_weight > 0
        else 0.0
    )
    stddev = math.sqrt(variance)

    scenario_results: dict[str, list[bool | None]] = {}
    for _, stats in completed_runs:
        for result in stats.results:
            scenario_id = str(result.get("scenario_id", "unknown"))
            scenario_results.setdefault(scenario_id, []).append(result.get("success"))

    pass_at_rate = 0.0
    pass_at_se = 0.0
    if scenario_results:
        n_scenarios = len(scenario_results)
        n_passed_any = sum(
            1
            for outcomes in scenario_results.values()
            if any(v is True for v in outcomes)
        )
        pass_at_rate = n_passed_any / n_scenarios * 100
        p = pass_at_rate / 100
        pass_at_se = math.sqrt(p * (1 - p) / n_scenarios) * 100

    print(f"\n{SUMMARY_SEPARATOR}")
    print(f"Aggregate pass@{len(per_run_stats)} summary")
    for run_number, stats in per_run_stats:
        print(
            f"run_{run_number}: {stats.passed}/{stats.total} passed, "
            f"{stats.failed} failed, {stats.errors} errors "
            f"({stats.pass_rate_percent:.1f}%)"
        )
    print(f"avg@{len(per_run_stats)}: {mean_rate:.1f}% ± {stddev:.1f}%")
    if scenario_results:
        print(f"pass@{len(per_run_stats)}: {pass_at_rate:.1f}% ± {pass_at_se:.1f}%")
    print(SUMMARY_SEPARATOR)
    print(f"Runs landing page: {Path(output_dir).resolve()}/index.html")


def _save_dataset_run_config(
    output_dir: str | None,
    run_config_base: JsonDict,
    *,
    num_scenarios: int,
    run_number: int | None = None,
    total_runs: int | None = None,
) -> None:
    if not output_dir:
        return

    _save_run_config(
        output_dir=output_dir,
        **run_config_base,
        num_scenarios=num_scenarios,
        run_number=run_number,
        total_runs=total_runs,
    )


def _allocate_ports() -> tuple[int, int]:
    """Allocate unique ephemeral adapter and gateway ports."""
    return _allocate_free_port(), _allocate_free_port()


def _run_scenarios_sequential(
    *,
    scenario_paths: list[Path],
    execution_config: ScenarioExecutionConfig,
    adapter_port: int,
    output_dir: str | None,
    dataset_root: Path | None = None,
) -> RunStats:
    stats = RunStats()
    launcher = LocalLauncher(runtime=execution_config.runtime)

    runner = ContainerRunner(
        launcher=launcher,
        image=execution_config.image,
        adapter_port=adapter_port,
    )
    pbar = tqdm(scenario_paths, desc="Scenarios", unit="sc", dynamic_ncols=True)
    for scenario_path in pbar:
        logger.info("=== Scenario: %s ===", scenario_path.name)
        effective_out = _effective_output_dir(output_dir, scenario_path, dataset_root)
        result = execution_config.run_with_runner(
            runner,
            scenario_path,
            output_dir=effective_out,
        )
        stats.record(result)
        pbar.set_postfix(_stats_postfix(stats))

    return stats


def _run_scenarios_concurrent(
    *,
    scenario_paths: list[Path],
    execution_config: ScenarioExecutionConfig,
    output_dir: str | None,
    concurrency: int,
    dataset_root: Path | None = None,
) -> RunStats:
    stats = RunStats()

    def _run_one(scenario_path: Path, adapter_port: int, gateway_port: int) -> JsonDict:
        effective_out = _effective_output_dir(output_dir, scenario_path, dataset_root)
        return execution_config.run_scenario(
            scenario_path,
            adapter_port=adapter_port,
            output_dir=effective_out,
            gateway_port=gateway_port,
        )

    logger.info(
        "Running with concurrency=%d, launcher=%s",
        concurrency,
        execution_config.launcher_type,
    )

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {}
        for scenario_path in scenario_paths:
            adapter_port, gateway_port = _allocate_ports()
            logger.info(
                "Submitting %s (adapter=%d, gateway=%d)",
                scenario_path.name,
                adapter_port,
                gateway_port,
            )
            future = pool.submit(_run_one, scenario_path, adapter_port, gateway_port)
            futures[future] = scenario_path

        pbar = tqdm(
            total=len(scenario_paths), desc="Scenarios", unit="sc", dynamic_ncols=True
        )
        for future in as_completed(futures):
            scenario_path = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                logger.error("Scenario %s crashed: %s", scenario_path.name, exc)
                result = {
                    "scenario_id": scenario_path.stem,
                    "success": None,
                    "error": str(exc),
                    "scenario_file": str(scenario_path.resolve()),
                }
            stats.record(result)
            pbar.update(1)
            pbar.set_postfix(_stats_postfix(stats))

            logger.info(
                "Completed %s: %s",
                scenario_path.name,
                _result_status_label(result),
            )
        pbar.close()

    return stats


def _run_dataset_once(
    *,
    scenario_paths: list[Path],
    execution_config: ScenarioExecutionConfig,
    adapter_port: int,
    concurrency: int,
    output_dir: str | None,
    output_file: str | None,
    run_config_base: JsonDict,
    run_number: int | None = None,
    total_runs: int | None = None,
    dataset_root: Path | None = None,
    retry: bool = False,
) -> JsonDict:
    """Run a full dataset once and return summary stats."""
    _save_dataset_run_config(
        output_dir,
        run_config_base,
        num_scenarios=len(scenario_paths),
        run_number=run_number,
        total_runs=total_runs,
    )

    if concurrency <= 1:
        stats = _run_scenarios_sequential(
            scenario_paths=scenario_paths,
            execution_config=execution_config,
            adapter_port=adapter_port,
            output_dir=output_dir,
            dataset_root=dataset_root,
        )
    else:
        stats = _run_scenarios_concurrent(
            scenario_paths=scenario_paths,
            execution_config=execution_config,
            output_dir=output_dir,
            concurrency=concurrency,
            dataset_root=dataset_root,
        )

    final_stats = stats
    if retry and output_dir:
        final_stats = _load_run_stats_from_disk(output_dir)

    results_output_path = _resolve_results_output_path(output_dir, output_file)
    if results_output_path is not None:
        log_label = "Merged results" if final_stats is not stats else "Results"
        _write_results_jsonl(
            results_output_path, final_stats.results, log_label=log_label
        )

    _generate_trace_viewer_if_possible(output_dir)

    run_label = f"Results (run {run_number}/{total_runs})" if run_number else "Results"
    split_stats = _summarize_results_by_split(
        final_stats.results,
        dataset_root=dataset_root,
    )
    _print_stats_summary(
        run_label,
        final_stats,
        output_dir=output_dir,
        split_stats=split_stats or None,
    )

    summary = final_stats.to_summary(run_number=run_number)
    if split_stats:
        summary["by_split"] = _split_summary_dict(split_stats)
    if retry and final_stats.total != stats.total:
        summary["retry_batch"] = stats.to_summary()
        retry_split_stats = _summarize_results_by_split(
            stats.results,
            dataset_root=dataset_root,
        )
        if retry_split_stats:
            summary["retry_batch"]["by_split"] = _split_summary_dict(retry_split_stats)
    return summary


def _run_interleaved_passes(
    *,
    scenario_paths: list[Path],
    execution_config: ScenarioExecutionConfig,
    concurrency: int,
    output_dir: str,
    output_requested: bool,
    pass_at: int,
    run_config_base: JsonDict,
    retry_tasks: list[tuple[Path, int]] | None = None,
    dataset_root: Path | None = None,
) -> None:
    """Run pass@N by interleaving all runs through a shared worker pool.

    If *retry_tasks* is provided, only those specific ``(scenario, run_number)``
    pairs are executed instead of the full cross-product.
    """
    run_output_dirs: dict[int, str] = {}
    stats_by_run: dict[int, RunStats] = {}

    for run_number in range(1, pass_at + 1):
        run_output_dir = str(Path(output_dir) / f"run_{run_number}")
        run_output_dirs[run_number] = run_output_dir
        stats_by_run[run_number] = RunStats()
        _save_dataset_run_config(
            run_output_dir,
            run_config_base,
            num_scenarios=len(scenario_paths),
            run_number=run_number,
            total_runs=pass_at,
        )

    if retry_tasks is not None:
        all_tasks = retry_tasks
    else:
        all_tasks = [
            (scenario_path, run_number)
            for run_number in range(1, pass_at + 1)
            for scenario_path in scenario_paths
        ]
    total_tasks = len(all_tasks)
    effective_concurrency = max(concurrency, 1)

    if retry_tasks is not None:
        logger.info(
            "Interleaved pass@%d: submitting %d retry tasks into pool of %d workers",
            pass_at,
            total_tasks,
            effective_concurrency,
        )
    else:
        logger.info(
            "Interleaved pass@%d: submitting %d tasks (%d scenarios × %d runs) "
            "into pool of %d workers",
            pass_at,
            total_tasks,
            len(scenario_paths),
            pass_at,
            effective_concurrency,
        )

    def _run_one(
        scenario_path: Path,
        run_number: int,
        adapter_port: int,
        gateway_port: int,
    ) -> tuple[int, JsonDict]:
        run_out = run_output_dirs[run_number]
        effective_out = _effective_output_dir(run_out, scenario_path, dataset_root)
        result = execution_config.run_scenario(
            scenario_path,
            adapter_port=adapter_port,
            output_dir=effective_out,
            gateway_port=gateway_port,
        )
        return run_number, result

    with ThreadPoolExecutor(max_workers=effective_concurrency) as pool:
        futures = {}
        for scenario_path, run_number in all_tasks:
            adapter_port, gateway_port = _allocate_ports()
            logger.info(
                "Submitting run_%d/%s (adapter=%d, gateway=%d)",
                run_number,
                scenario_path.name,
                adapter_port,
                gateway_port,
            )
            future = pool.submit(
                _run_one,
                scenario_path,
                run_number,
                adapter_port,
                gateway_port,
            )
            futures[future] = (scenario_path, run_number)

        # Aggregate stats across all runs for the progress bar
        aggregate = RunStats()
        pbar = tqdm(
            total=total_tasks,
            desc=f"pass@{pass_at}",
            unit="task",
            dynamic_ncols=True,
        )
        for future in as_completed(futures):
            scenario_path, run_number = futures[future]
            try:
                _, result = future.result()
            except Exception as exc:
                logger.error(
                    "run_%d/%s crashed: %s", run_number, scenario_path.name, exc
                )
                result = {
                    "scenario_id": scenario_path.stem,
                    "success": None,
                    "error": str(exc),
                    "scenario_file": str(scenario_path.resolve()),
                }
            stats_by_run[run_number].record(result)
            aggregate.record(result)
            pbar.update(1)
            pbar.set_postfix(_stats_postfix(aggregate))

            logger.info(
                "Completed run_%d/%s: %s",
                run_number,
                scenario_path.name,
                _result_status_label(result),
            )
        pbar.close()

    # Build per-run stats for summary. In retry mode, rebuild from disk so
    # the summary reflects ALL results (including preserved previous runs),
    # not just what was run in this session.
    per_run_stats: list[tuple[int, RunStats]] = []
    for run_number in range(1, pass_at + 1):
        run_output_dir = run_output_dirs[run_number]

        if retry_tasks is not None:
            run_stats = _load_run_stats_from_disk(run_output_dir)
        else:
            run_stats = stats_by_run[run_number]

        per_run_stats.append((run_number, run_stats))

        if output_requested:
            _write_results_jsonl(
                Path(run_output_dir) / "results.jsonl",
                run_stats.results,
                log_label=f"Run {run_number} results",
            )

        _generate_trace_viewer_if_possible(
            run_output_dir,
            failure_message=f"Failed to generate trace viewer for run {run_number}",
        )
        split_stats = _summarize_results_by_split(
            run_stats.results,
            dataset_root=dataset_root,
        )
        _print_stats_summary(
            f"Results (run {run_number}/{pass_at})",
            run_stats,
            output_dir=run_output_dir,
            split_stats=split_stats or None,
        )

    try:
        generate_runs_index(output_dir)
    except Exception as exc:
        logger.warning("Failed to generate runs index: %s", exc)

    _print_aggregate_summary(per_run_stats, output_dir)


def _build_execution_config(
    *,
    image: str,
    runtime: str,
    timeout: int,
    health_timeout: int,
    provider: str | None,
    model: str | None,
    api_key: str | None,
    base_url: str | None,
    thinking: str,
    judge_model: str | None,
    judge_provider: str | None,
    judge_base_url: str | None,
    judge_api_key: str | None,
    volumes: tuple[str, ...],
    notification_mode: str,
    time_speed: float | None,
) -> tuple[
    ScenarioExecutionConfig,
    str | None,
    str | None,
    str,
    str,
    str | None,
]:
    """Resolve CLI config into a concrete ScenarioExecutionConfig."""
    resolved_provider, resolved_model = _resolve_agent_config(image, provider, model)
    (
        resolved_judge_model,
        resolved_judge_provider,
        resolved_judge_base_url,
        resolved_judge_api_key,
    ) = _resolve_judge_config(
        judge_model, judge_provider, judge_base_url, judge_api_key
    )

    execution_config = ScenarioExecutionConfig(
        image=image,
        runtime=runtime,
        timeout=timeout,
        health_timeout=health_timeout,
        container_env=_build_container_env(
            image,
            base_url,
            thinking,
            notification_mode,
            time_speed,
            judge_model=resolved_judge_model,
            judge_provider=resolved_judge_provider,
            judge_base_url=resolved_judge_base_url,
            judge_api_key=resolved_judge_api_key,
        ),
        provider=resolved_provider,
        model=resolved_model,
        api_key=api_key,
        judge_model=resolved_judge_model,
        judge_provider=resolved_judge_provider,
        judge_base_url=resolved_judge_base_url,
        extra_volumes=volumes,
        launcher_type=runtime,
    )

    return (
        execution_config,
        resolved_provider,
        resolved_model,
        resolved_judge_model,
        resolved_judge_provider,
        resolved_judge_base_url,
    )


def _execute_single_scenario_run(
    *,
    scenario: str,
    execution_config: ScenarioExecutionConfig,
    adapter_port: int,
    output_dir: str | None,
    run_config_base: JsonDict,
) -> int:
    """Run one scenario and return the process exit code."""
    if output_dir:
        _save_run_config(output_dir=output_dir, **run_config_base)

    result = execution_config.run_scenario(
        Path(scenario),
        adapter_port=adapter_port,
        output_dir=output_dir,
    )

    _generate_trace_viewer_if_possible(output_dir)

    print(json.dumps(result, indent=2, default=str))
    return _result_exit_code(result)


def _execute_dataset_selection(
    *,
    dataset_label: str,
    scenario_paths: list[Path],
    dataset_root: Path | None,
    execution_config: ScenarioExecutionConfig,
    adapter_port: int,
    concurrency: int,
    output_dir: str | None,
    output_file: str | None,
    pass_at: int,
    retry: bool,
    run_config_base: JsonDict,
) -> None:
    """Run a resolved dataset selection."""
    total_scenarios = len(scenario_paths)

    if pass_at < 1:
        raise click.BadParameter("--pass-at must be >= 1", param_hint="--pass-at")
    if pass_at > 1 and not output_dir:
        raise click.UsageError("--pass-at > 1 requires --output-dir")

    multirun_retry_tasks: list[tuple[Path, int]] | None = None

    if retry:
        if not output_dir or not Path(output_dir).is_dir():
            raise click.UsageError(
                "--retry requires --output-dir pointing to an existing run"
            )

        if pass_at > 1:
            mr_selection = _select_retry_tasks_multirun(
                scenario_paths,
                output_dir,
                pass_at,
                dataset_root=dataset_root,
            )
            multirun_retry_tasks = mr_selection.tasks
            total_scenarios = mr_selection.rerun_count
            logger.info(
                "Multi-run retry: %d tasks to re-run (%d passed, %d failed)",
                total_scenarios,
                mr_selection.passed,
                mr_selection.failed,
            )
        else:
            retry_sel = _select_retry_scenarios(
                scenario_paths, output_dir, dataset_root=dataset_root
            )
            scenario_paths = retry_sel.scenarios
            total_scenarios = len(scenario_paths)

    if total_scenarios == 0:
        logger.info("No scenarios to run (all passed or empty dataset)")
        return

    logger.info(
        "Running %d scenarios from %s (concurrency=%d, pass_at=%s)",
        total_scenarios,
        dataset_label,
        concurrency,
        pass_at,
    )

    if pass_at > 1:
        _run_interleaved_passes(
            scenario_paths=scenario_paths,
            execution_config=execution_config,
            concurrency=concurrency,
            output_dir=output_dir or str(Path(dataset_label).parent / "output"),
            output_requested=output_file is not None,
            pass_at=pass_at,
            run_config_base=run_config_base,
            retry_tasks=multirun_retry_tasks,
            dataset_root=dataset_root,
        )
    else:
        result_summary = _run_dataset_once(
            scenario_paths=scenario_paths,
            execution_config=execution_config,
            adapter_port=adapter_port,
            concurrency=concurrency,
            output_dir=output_dir,
            output_file=output_file,
            run_config_base=run_config_base,
            dataset_root=dataset_root,
            retry=retry,
        )
        logger.info("Summary: %s", json.dumps(result_summary, indent=2))

    _generate_trace_viewer_if_possible(output_dir)


def _load_run_config_dataset_scenarios(
    config: RunnerTomlConfig,
) -> tuple[list[Path], Path | None, str | None, Path | None]:
    """Resolve a run-config dataset target into concrete scenario paths."""
    if config.target.is_hf_dataset:
        from gaia2_runner.hf_dataset import download_hf_dataset

        split_list = list(config.target.splits) if config.target.splits else None
        cache_dir = download_hf_dataset(config.target.dataset, splits=split_list)
        scenario_paths, resolved_root, _ = _load_dataset_scenarios(
            cache_dir,
            config.target.limit,
            recursive=True,
            subset=config.target.subset_manifest,
        )
        return scenario_paths, Path(cache_dir), None, Path(cache_dir)

    if not config.target.dataset_root:
        raise click.UsageError("Dataset config is missing [target].dataset_root")

    dataset_root = Path(config.target.dataset_root)

    if not config.target.splits:
        scenario_paths, resolved_root, tmpdir = _load_dataset_scenarios(
            config.target.dataset_root,
            config.target.limit,
            recursive=config.target.recursive,
            subset=config.target.subset_manifest,
        )
        return scenario_paths, resolved_root, tmpdir, None

    missing_splits = [
        split for split in config.target.splits if not (dataset_root / split).is_dir()
    ]
    if missing_splits:
        raise click.UsageError(
            "Missing split directories under dataset root: "
            + ", ".join(str(dataset_root / split) for split in missing_splits)
        )

    scenario_paths: list[Path] = []
    for split in config.target.splits:
        split_dir = dataset_root / split
        split_paths, _, _ = _load_dataset_scenarios(
            str(split_dir),
            None,
            recursive=config.target.recursive,
            subset=config.target.subset_manifest,
        )
        scenario_paths.extend(split_paths)

    scenario_paths = sorted(scenario_paths)
    if config.target.limit is not None:
        scenario_paths = scenario_paths[: config.target.limit]

    return scenario_paths, dataset_root, None, None


def _normalize_hf_cli_splits(splits: str | None) -> list[str] | None:
    """Parse ``run-dataset --splits`` into canonical split names."""
    if splits is None:
        return None

    normalized = [part.strip().lower() for part in splits.split(",") if part.strip()]
    if not normalized:
        return None

    from gaia2_runner.config import CANONICAL_SPLITS

    if normalized == ["all"]:
        return list(CANONICAL_SPLITS)
    if "all" in normalized:
        raise click.UsageError("--splits cannot combine 'all' with named splits")

    unknown = sorted(set(normalized) - set(CANONICAL_SPLITS))
    if unknown:
        raise click.UsageError("Unknown split(s) for --splits: " + ", ".join(unknown))
    return list(dict.fromkeys(normalized))


def _resolved_dataset_splits_for_metadata(
    dataset: str | None,
    splits: list[str] | tuple[str, ...] | None,
) -> list[str] | None:
    """Return the concrete split list represented by a dataset selection."""
    if splits:
        return list(splits)
    if not dataset:
        return None

    from gaia2_runner.hf_dataset import is_hf_dataset

    if not is_hf_dataset(dataset):
        return None

    from gaia2_runner.config import CANONICAL_SPLITS

    return list(CANONICAL_SPLITS)


def _print_run_config_summary(
    config: RunnerTomlConfig,
    *,
    scenario_count: int | None = None,
    effective_retry: bool | None = None,
) -> None:
    """Print a concise summary of a resolved run-config file."""
    if effective_retry is None:
        effective_retry = config.run.retry

    click.echo(f"Config: {config.config_path}")
    click.echo(f"Image: {config.agent.image}")

    if config.target.is_single_scenario:
        click.echo("Mode: scenario")
        click.echo(f"Scenario: {config.target.scenario}")
    else:
        click.echo("Mode: dataset")
        if config.target.dataset:
            click.echo(f"Dataset: {config.target.dataset}")
        else:
            click.echo(f"Dataset root: {config.target.dataset_root}")
        if config.target.splits:
            click.echo(f"Splits: {', '.join(config.target.splits)}")
        elif config.target.dataset:
            click.echo("Splits: all benchmark splits")
        else:
            click.echo("Splits: all files under dataset_root")
        if config.target.subset_manifest:
            click.echo(f"Subset manifest: {config.target.subset_manifest}")
        if scenario_count is not None:
            click.echo(f"Resolved scenarios: {scenario_count}")

    click.echo(
        "Agent: "
        + (
            f"{config.agent.provider}/{config.agent.model}"
            if config.agent.provider and config.agent.model
            else "oracle"
        )
    )
    click.echo(f"Judge: {config.judge.provider}/{config.judge.model}")
    click.echo(f"Concurrency: {config.run.concurrency}")
    click.echo(f"Pass@: {config.run.pass_at}")
    if effective_retry:
        click.echo("Retry: on")
    if config.run.output_dir:
        click.echo(f"Output dir: {config.run.output_dir}")
    if config.run.output:
        click.echo(f"Output file: {config.run.output}")


@click.group()
@click.option(
    "--env-file",
    default=None,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Optional dotenv-style env file for API keys and host network settings. "
    "If omitted, the runner auto-loads gaia2-cli/.env.",
)
def main(env_file: Path | None) -> None:
    """Gaia2 Runner — run scenarios in containers and grade results."""
    _maybe_load_runner_env(env_file)


@main.command()
@click.option(
    "--scenario",
    "-s",
    required=True,
    type=click.Path(exists=True),
    help="Path to scenario JSON file",
)
@click.option("--image", "-i", required=True, help="Container image name")
@click.option(
    "--runtime",
    default="podman",
    help="Container runtime command (e.g. 'podman', 'docker')",
)
@click.option(
    "--adapter-port",
    default=8090,
    type=int,
    help="Port for the gaia2-adapter inside the container",
)
@click.option(
    "--timeout", default=600, type=int, help="Max seconds to wait for agent response"
)
@click.option(
    "--health-timeout",
    default=120,
    type=int,
    help="Max seconds to wait for adapter health",
)
@click.option(
    "--provider",
    default=None,
    help="LLM provider (e.g. anthropic, openai, google)",
)
@click.option("--model", default=None, help="Model ID override for the provider")
@click.option("--api-key", default=None, help="API key for the provider")
@click.option(
    "--base-url",
    default=None,
    help="Base URL override for the LLM API",
)
@click.option(
    "--thinking",
    default="low",
    type=click.Choice(["off", "low", "medium", "high"]),
    help="Reasoning/thinking strength (default: low)",
)
@click.option(
    "--judge-model",
    default=None,
    help="Model for the in-container judge. Required unless "
    "GAIA2_JUDGE_MODEL is already set.",
)
@click.option(
    "--judge-provider",
    default=None,
    help="Provider for the in-container judge. Required unless "
    "GAIA2_JUDGE_PROVIDER is already set.",
)
@click.option(
    "--judge-base-url",
    default=None,
    help="Optional API base URL for the in-container judge. "
    "Passed as GAIA2_JUDGE_BASE_URL env var.",
)
@click.option(
    "--judge-api-key",
    default=None,
    help="Optional API key for the in-container judge. "
    "Passed as GAIA2_JUDGE_API_KEY env var.",
)
@click.option(
    "--volume",
    "volumes",
    multiple=True,
    help="Extra bind mount for the local launcher, e.g. /host:/container[:opts]",
)
@click.option(
    "--output-dir",
    default=None,
    type=click.Path(),
    help="Directory to save run artifacts (events, response, result, openclaw log)",
)
@click.option("--log-level", default="INFO", help="Logging level")
@click.option(
    "--notification-mode",
    type=click.Choice(["message", "native"]),
    default="message",
    help="ENV notification delivery: 'message' (bundled user msg) or 'native' (OpenClaw system event)",
)
@click.option(
    "--time-speed",
    default=None,
    type=float,
    help="Time speed multiplier for fast-forward mode (e.g. 5 = 5x faster). "
    "Speeds up ENV event delays in time scenarios.",
)
def run(
    scenario: str,
    image: str,
    runtime: str,
    adapter_port: int,
    timeout: int,
    health_timeout: int,
    provider: str | None,
    model: str | None,
    api_key: str | None,
    base_url: str | None,
    thinking: str,
    judge_model: str | None,
    judge_provider: str | None,
    judge_base_url: str | None,
    judge_api_key: str | None,
    volumes: tuple[str, ...],
    output_dir: str | None,
    log_level: str,
    notification_mode: str,
    time_speed: float | None,
) -> None:
    """Run a single scenario in a container and grade the result."""
    (
        execution_config,
        resolved_provider,
        resolved_model,
        resolved_judge_model,
        resolved_judge_provider,
        resolved_judge_base_url,
    ) = _build_execution_config(
        image=image,
        runtime=runtime,
        timeout=timeout,
        health_timeout=health_timeout,
        provider=provider,
        model=model,
        api_key=api_key,
        base_url=base_url,
        thinking=thinking,
        judge_model=judge_model,
        judge_provider=judge_provider,
        judge_base_url=judge_base_url,
        judge_api_key=judge_api_key,
        volumes=volumes,
        notification_mode=notification_mode,
        time_speed=time_speed,
    )
    setup_logging(log_level)

    run_config_base: JsonDict = {
        "command": "run",
        "dataset": None,
        "scenario": scenario,
        "image": image,
        "runtime": runtime,
        "provider": resolved_provider,
        "model": resolved_model,
        "base_url": base_url,
        "judge_model": resolved_judge_model,
        "judge_provider": resolved_judge_provider,
        "judge_base_url": resolved_judge_base_url,
        "timeout": timeout,
        "health_timeout": health_timeout,
        "concurrency": 1,
        "limit": None,
    }

    exit_code = _execute_single_scenario_run(
        scenario=scenario,
        execution_config=execution_config,
        adapter_port=adapter_port,
        output_dir=output_dir,
        run_config_base=run_config_base,
    )
    sys.exit(exit_code)


@main.command()
@click.option(
    "--output-dir",
    required=True,
    type=click.Path(exists=True),
    help="Directory containing run artifacts (with per-scenario subdirectories)",
)
@click.option("--log-level", default="INFO")
def view(output_dir: str, log_level: str) -> None:
    """Generate trace viewer HTML from existing run artifacts.

    Scans output-dir for scenario subdirectories and generates:
    - Per-scenario trace.html pages
    - Top-level index.html with summary and scenario table

    Use this to regenerate the viewer without re-running scenarios.
    """
    setup_logging(log_level)
    generate_trace_viewer(output_dir)
    index_path = Path(output_dir).resolve() / "index.html"
    if index_path.exists():
        print(f"Trace viewer generated: {index_path}")
    else:
        print("No scenarios found in output directory.")


@main.command()
@click.option(
    "--output-dir",
    required=True,
    type=click.Path(exists=True),
    help="Directory with trace artifacts (single run or multi-run root)",
)
@click.option(
    "--port",
    default=None,
    type=int,
    help="HTTP server port (default: auto-select from 44100-44109)",
)
@click.option(
    "--interval",
    default=15,
    type=int,
    help="Seconds between regeneration checks",
)
@click.option("--log-level", default="INFO")
def serve(output_dir: str, port: int | None, interval: int, log_level: str) -> None:
    """Live trace viewer: serves HTML and auto-regenerates on new results.

    Start this alongside a running eval to watch progress in real-time.
    Press Ctrl+C to stop.
    """
    import socket
    import threading
    from http.server import HTTPServer, SimpleHTTPRequestHandler

    from gaia2_runner.serve import _is_multi_run, _regenerate_loop

    setup_logging(log_level)

    output_dir = os.path.abspath(output_dir)
    multi_run = _is_multi_run(output_dir)

    mode = "Multi-run" if multi_run else "Single-run"
    logger.info("%s mode: %s", mode, output_dir)

    t = threading.Thread(
        target=_regenerate_loop,
        args=(output_dir, multi_run, interval),
        kwargs={"generate_immediately": True},
        daemon=True,
    )
    t.start()

    os.chdir(output_dir)

    # Bind dual-stack (IPv6 + IPv4) for broad network compatibility
    class DualStackHTTPServer(HTTPServer):
        address_family = socket.AF_INET6

        def server_bind(self) -> None:
            self.socket.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 0)
            super().server_bind()

    if port is not None:
        server = DualStackHTTPServer(("::", port), SimpleHTTPRequestHandler)
    else:
        server = None
        for p in range(44100, 44110):
            try:
                server = DualStackHTTPServer(("::", p), SimpleHTTPRequestHandler)
                port = p
                break
            except OSError:
                continue
        if server is None:
            raise click.ClickException(
                "No free port in 44100-44109. Kill stale processes or pass --port."
            )

    hostname = socket.getfqdn() or socket.gethostname()
    click.echo(f"Serving trace viewer at http://{hostname}:{port}")
    click.echo(f"Auto-regenerating every {interval}s when trace artifacts change")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        click.echo("\nStopped")


@main.command("run-dataset")
@click.option(
    "--dataset",
    "-d",
    required=True,
    type=str,
    help="Directory of scenario JSONs, a JSONL file, or a HuggingFace dataset ID (e.g. 'org/dataset').",
)
@click.option(
    "--splits",
    default=None,
    help="Comma-separated splits to download (e.g. 'search,time' or 'all'). Only used with HuggingFace datasets.",
)
@click.option("--image", "-i", required=True, help="Container image name")
@click.option(
    "--runtime",
    default="podman",
    help="Container runtime command (e.g. 'podman', 'docker')",
)
@click.option("--adapter-port", default=8090, type=int)
@click.option(
    "--timeout", default=600, type=int, help="Max seconds to wait per scenario"
)
@click.option("--health-timeout", default=120, type=int)
@click.option("--limit", default=None, type=int, help="Max number of scenarios to run")
@click.option(
    "--concurrency",
    "-j",
    default=1,
    type=int,
    help="Number of scenarios to run concurrently (default: 1 = sequential)",
)
@click.option(
    "--provider",
    default=None,
    help="LLM provider (e.g. anthropic, openai, google)",
)
@click.option("--model", default=None, help="Model ID override for the provider")
@click.option("--api-key", default=None, help="API key for the provider")
@click.option(
    "--base-url",
    default=None,
    help="Base URL override for the LLM API",
)
@click.option(
    "--thinking",
    default="low",
    type=click.Choice(["off", "low", "medium", "high"]),
    help="Reasoning/thinking strength (default: low)",
)
@click.option(
    "--judge-model",
    default=None,
    help="Model for the in-container judge. Required unless "
    "GAIA2_JUDGE_MODEL is already set.",
)
@click.option(
    "--judge-provider",
    default=None,
    help="Provider for the in-container judge. Required unless "
    "GAIA2_JUDGE_PROVIDER is already set.",
)
@click.option(
    "--judge-base-url",
    default=None,
    help="Optional API base URL for the in-container judge. "
    "Passed as GAIA2_JUDGE_BASE_URL env var.",
)
@click.option(
    "--judge-api-key",
    default=None,
    help="Optional API key for the in-container judge. "
    "Passed as GAIA2_JUDGE_API_KEY env var.",
)
@click.option(
    "--volume",
    "volumes",
    multiple=True,
    help="Extra bind mount for the local launcher, e.g. /host:/container[:opts]",
)
@click.option(
    "--output",
    "-o",
    default=None,
    type=click.Path(),
    help="Output JSONL file for results",
)
@click.option(
    "--output-dir",
    default=None,
    type=click.Path(),
    help="Directory to save per-scenario run artifacts",
)
@click.option("--log-level", default="INFO")
@click.option(
    "--pass-at",
    "pass_at",
    default=1,
    type=int,
    help="Number of times to run the full dataset (for variance measurement). "
    "When > 1, creates run_1/ ... run_N/ subdirs under --output-dir.",
)
@click.option(
    "--retry",
    is_flag=True,
    default=False,
    help="Re-run only errored scenarios (startup failures, timeouts) from a "
    "previous run. Keeps failures as-is. Requires --output-dir.",
)
@click.option(
    "--non-recursive",
    is_flag=True,
    default=False,
    help="Only search the top-level dataset directory for scenario JSON files "
    "(no subdirectory traversal). By default, subdirectories are searched recursively.",
)
@click.option(
    "--notification-mode",
    type=click.Choice(["message", "native"]),
    default="message",
    help="ENV notification delivery: 'message' (bundled user msg) or 'native' (OpenClaw system event)",
)
@click.option(
    "--time-speed",
    default=None,
    type=float,
    help="Time speed multiplier for fast-forward mode (e.g. 5 = 5x faster). "
    "Speeds up ENV event delays in time scenarios.",
)
@click.option(
    "--subset",
    default=None,
    type=click.Path(exists=True),
    help="Path to a subset manifest JSON. Only scenarios listed will be run.",
)
def run_dataset(
    dataset: str,
    splits: str | None,
    image: str,
    runtime: str,
    adapter_port: int,
    timeout: int,
    health_timeout: int,
    limit: int | None,
    concurrency: int,
    provider: str | None,
    model: str | None,
    api_key: str | None,
    base_url: str | None,
    thinking: str,
    judge_model: str | None,
    judge_provider: str | None,
    judge_base_url: str | None,
    judge_api_key: str | None,
    volumes: tuple[str, ...],
    output: str | None,
    output_dir: str | None,
    log_level: str,
    pass_at: int,
    retry: bool,
    non_recursive: bool,
    notification_mode: str,
    time_speed: float | None,
    subset: str | None,
) -> None:
    """Run multiple scenarios from a dataset directory, JSONL file, or HuggingFace dataset."""
    (
        execution_config,
        resolved_provider,
        resolved_model,
        resolved_judge_model,
        resolved_judge_provider,
        resolved_judge_base_url,
    ) = _build_execution_config(
        image=image,
        runtime=runtime,
        timeout=timeout,
        health_timeout=health_timeout,
        provider=provider,
        model=model,
        api_key=api_key,
        base_url=base_url,
        thinking=thinking,
        judge_model=judge_model,
        judge_provider=judge_provider,
        judge_base_url=judge_base_url,
        judge_api_key=judge_api_key,
        volumes=volumes,
        notification_mode=notification_mode,
        time_speed=time_speed,
    )
    setup_logging(log_level)

    from gaia2_runner.hf_dataset import download_hf_dataset, is_hf_dataset

    effective_dataset = dataset
    split_list: list[str] | None = None
    is_hf_source = is_hf_dataset(dataset)
    if is_hf_source:
        split_list = _normalize_hf_cli_splits(splits)
        effective_dataset = download_hf_dataset(dataset, splits=split_list)
    elif not Path(dataset).exists():
        raise click.UsageError(f"Dataset path does not exist: {dataset}")

    run_config_base: JsonDict = {
        "command": "run-dataset",
        "dataset": dataset,
        "dataset_cache_dir": effective_dataset if is_hf_source else None,
        "scenario": None,
        "image": image,
        "runtime": runtime,
        "provider": resolved_provider,
        "model": resolved_model,
        "base_url": base_url,
        "judge_model": resolved_judge_model,
        "judge_provider": resolved_judge_provider,
        "judge_base_url": resolved_judge_base_url,
        "timeout": timeout,
        "health_timeout": health_timeout,
        "concurrency": concurrency,
        "limit": limit,
        "splits": _resolved_dataset_splits_for_metadata(dataset, split_list),
    }

    scenario_paths, dataset_root, tmpdir = _load_dataset_scenarios(
        effective_dataset, limit, recursive=not non_recursive, subset=subset
    )
    try:
        _execute_dataset_selection(
            dataset_label=dataset,
            scenario_paths=scenario_paths,
            dataset_root=dataset_root,
            execution_config=execution_config,
            adapter_port=adapter_port,
            concurrency=concurrency,
            output_dir=output_dir,
            output_file=output,
            pass_at=pass_at,
            retry=retry,
            run_config_base=run_config_base,
        )

    finally:
        if tmpdir:
            import shutil

            shutil.rmtree(tmpdir, ignore_errors=True)


@main.command("run-config")
@click.option(
    "--config",
    "config_path",
    required=True,
    type=click.Path(exists=True),
    help="Path to an eval TOML config file.",
)
@click.option(
    "--retry",
    is_flag=True,
    default=False,
    help="Override [run].retry and rerun only errored scenarios from the configured output directory.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Validate the config, resolve the target selection, and print a summary without launching.",
)
def run_config(config_path: str, retry: bool, dry_run: bool) -> None:
    """Run a scenario or dataset from a TOML config file."""
    from gaia2_runner.config import load_runner_toml_config

    config = load_runner_toml_config(config_path)
    setup_logging(config.run.log_level)
    effective_retry = config.run.retry or retry

    if config.target.is_single_scenario:
        if effective_retry:
            raise click.UsageError("--retry is only supported for dataset targets")
        _print_run_config_summary(config, effective_retry=effective_retry)
        if dry_run:
            return

        (
            execution_config,
            resolved_provider,
            resolved_model,
            resolved_judge_model,
            resolved_judge_provider,
            resolved_judge_base_url,
        ) = _build_execution_config(
            image=config.agent.image,
            runtime=config.agent.runtime,
            timeout=config.run.timeout,
            health_timeout=config.run.health_timeout,
            provider=config.agent.provider,
            model=config.agent.model,
            api_key=config.agent.api_key,
            base_url=config.agent.base_url,
            thinking=config.agent.thinking,
            judge_model=config.judge.model,
            judge_provider=config.judge.provider,
            judge_base_url=config.judge.base_url,
            judge_api_key=config.judge.api_key,
            volumes=config.agent.volumes,
            notification_mode=config.run.notification_mode,
            time_speed=config.run.time_speed,
        )

        run_config_base: JsonDict = {
            "command": "run-config",
            "config_path": config.config_path,
            "dataset": None,
            "scenario": config.target.scenario,
            "image": config.agent.image,
            "runtime": config.agent.runtime,
            "provider": resolved_provider,
            "model": resolved_model,
            "base_url": config.agent.base_url,
            "judge_model": resolved_judge_model,
            "judge_provider": resolved_judge_provider,
            "judge_base_url": resolved_judge_base_url,
            "timeout": config.run.timeout,
            "health_timeout": config.run.health_timeout,
            "concurrency": 1,
            "limit": None,
        }

        exit_code = _execute_single_scenario_run(
            scenario=config.target.scenario or "",
            execution_config=execution_config,
            adapter_port=config.run.adapter_port,
            output_dir=config.run.output_dir,
            run_config_base=run_config_base,
        )
        sys.exit(exit_code)

    scenario_paths, dataset_root, tmpdir, dataset_cache_dir = (
        _load_run_config_dataset_scenarios(config)
    )
    try:
        _print_run_config_summary(
            config,
            scenario_count=len(scenario_paths),
            effective_retry=effective_retry,
        )
        if dry_run:
            return

        (
            execution_config,
            resolved_provider,
            resolved_model,
            resolved_judge_model,
            resolved_judge_provider,
            resolved_judge_base_url,
        ) = _build_execution_config(
            image=config.agent.image,
            runtime=config.agent.runtime,
            timeout=config.run.timeout,
            health_timeout=config.run.health_timeout,
            provider=config.agent.provider,
            model=config.agent.model,
            api_key=config.agent.api_key,
            base_url=config.agent.base_url,
            thinking=config.agent.thinking,
            judge_model=config.judge.model,
            judge_provider=config.judge.provider,
            judge_base_url=config.judge.base_url,
            judge_api_key=config.judge.api_key,
            volumes=config.agent.volumes,
            notification_mode=config.run.notification_mode,
            time_speed=config.run.time_speed,
        )

        run_config_base = {
            "command": "run-config",
            "config_path": config.config_path,
            "dataset": config.target.dataset or config.target.dataset_root,
            "dataset_cache_dir": str(dataset_cache_dir) if dataset_cache_dir else None,
            "scenario": None,
            "subset": config.target.subset_manifest,
            "splits": _resolved_dataset_splits_for_metadata(
                config.target.dataset or config.target.dataset_root,
                config.target.splits,
            ),
            "image": config.agent.image,
            "runtime": config.agent.runtime,
            "provider": resolved_provider,
            "model": resolved_model,
            "base_url": config.agent.base_url,
            "judge_model": resolved_judge_model,
            "judge_provider": resolved_judge_provider,
            "judge_base_url": resolved_judge_base_url,
            "timeout": config.run.timeout,
            "health_timeout": config.run.health_timeout,
            "concurrency": config.run.concurrency,
            "limit": config.target.limit,
        }

        _execute_dataset_selection(
            dataset_label=config.target.dataset
            or config.target.dataset_root
            or config.config_path,
            scenario_paths=scenario_paths,
            dataset_root=dataset_root,
            execution_config=execution_config,
            adapter_port=config.run.adapter_port,
            concurrency=config.run.concurrency,
            output_dir=config.run.output_dir,
            output_file=config.run.output,
            pass_at=config.run.pass_at,
            retry=effective_retry,
            run_config_base=run_config_base,
        )
    finally:
        if tmpdir:
            shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    main()
