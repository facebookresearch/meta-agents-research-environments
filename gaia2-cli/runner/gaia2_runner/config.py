# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""TOML-backed config loading for the Gaia2 runner."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import click

from gaia2_runner.container_env import detect_profile

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11 fallback
    import tomli as tomllib

CANONICAL_SPLITS: tuple[str, ...] = (
    "execution",
    "search",
    "ambiguity",
    "adaptability",
    "time",
)
_ENV_VAR_PATTERN = re.compile(r"\$(?:[A-Za-z_][A-Za-z0-9_]*|\{[^}]+\})")


@dataclass(frozen=True, slots=True)
class TargetConfig:
    scenario: str | None
    dataset_root: str | None
    dataset: str | None
    splits: tuple[str, ...]
    subset_manifest: str | None
    limit: int | None
    recursive: bool

    @property
    def is_single_scenario(self) -> bool:
        return self.scenario is not None

    @property
    def is_hf_dataset(self) -> bool:
        return self.dataset is not None


@dataclass(frozen=True, slots=True)
class AgentConfig:
    image: str
    runtime: str
    provider: str | None
    model: str | None
    api_key: str | None
    base_url: str | None
    thinking: str
    volumes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class JudgeConfig:
    model: str
    provider: str
    base_url: str | None
    api_key: str | None


@dataclass(frozen=True, slots=True)
class RunConfig:
    timeout: int
    health_timeout: int
    adapter_port: int
    concurrency: int
    pass_at: int
    retry: bool
    output_dir: str | None
    output: str | None
    log_level: str
    notification_mode: str
    time_speed: float | None


@dataclass(frozen=True, slots=True)
class RunnerTomlConfig:
    config_path: str
    target: TargetConfig
    agent: AgentConfig
    judge: JudgeConfig
    run: RunConfig


def _as_table(value: Any, name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise click.UsageError(f"[{name}] must be a TOML table")
    return value


def _as_optional_str(value: Any, name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise click.UsageError(f"{name} must be a string")
    text = value.strip()
    return text or None


def _as_required_str(value: Any, name: str) -> str:
    text = _as_optional_str(value, name)
    if not text:
        raise click.UsageError(f"{name} is required")
    return text


def _as_optional_int(value: Any, name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise click.UsageError(f"{name} must be an integer")
    return value


def _as_int(value: Any, name: str, *, default: int) -> int:
    result = _as_optional_int(value, name)
    return default if result is None else result


def _as_optional_float(value: Any, name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise click.UsageError(f"{name} must be a number")
    return float(value)


def _as_bool(value: Any, name: str, *, default: bool) -> bool:
    if value is None:
        return default
    if not isinstance(value, bool):
        raise click.UsageError(f"{name} must be true or false")
    return value


def _as_string_list(value: Any, name: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise click.UsageError(f"{name} must be a list of strings")
    return tuple(item.strip() for item in value if item.strip())


def _as_choice(
    value: Any,
    name: str,
    *,
    choices: tuple[str, ...],
    default: str,
) -> str:
    text = _as_optional_str(value, name) or default
    if text not in choices:
        raise click.UsageError(f"{name} must be one of: {', '.join(choices)}")
    return text


def _resolve_path(
    value: str | None,
    *,
    base_dir: Path,
    name: str,
    must_exist: bool,
    must_be_dir: bool = False,
) -> str | None:
    if value is None:
        return None

    expanded_value = os.path.expanduser(os.path.expandvars(value))
    unresolved = _ENV_VAR_PATTERN.findall(expanded_value)
    if unresolved:
        missing = ", ".join(
            sorted(
                {
                    token[2:-1] if token.startswith("${") else token[1:]
                    for token in unresolved
                }
            )
        )
        raise click.UsageError(f"{name} references unset env var(s): {missing}")

    path = Path(expanded_value)
    if not path.is_absolute():
        path = base_dir / path
    path = path.resolve()

    if must_exist and not path.exists():
        raise click.UsageError(f"{name} does not exist: {path}")
    if must_be_dir and path.exists() and not path.is_dir():
        raise click.UsageError(f"{name} must be a directory: {path}")

    return str(path)


def _resolve_secret(
    table: dict[str, Any],
    *,
    section_name: str,
) -> str | None:
    api_key = _as_optional_str(table.get("api_key"), f"[{section_name}].api_key")
    api_key_env = _as_optional_str(
        table.get("api_key_env"), f"[{section_name}].api_key_env"
    )

    if api_key and api_key_env:
        raise click.UsageError(
            f"[{section_name}] cannot set both api_key and api_key_env"
        )

    if api_key_env:
        value = os.environ.get(api_key_env, "").strip()
        if not value:
            raise click.UsageError(
                f"[{section_name}].api_key_env references an unset env var: {api_key_env}"
            )
        click.secho(
            f"{section_name.capitalize()} API key pulled from env var {api_key_env}",
            fg="red",
            err=True,
        )
        return value

    return api_key


def _normalize_splits(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()

    if isinstance(value, str):
        raw = (value.strip(),)
    else:
        raw = _as_string_list(value, "[target].splits")

    if not raw:
        return ()

    lowered = tuple(item.lower() for item in raw)
    if lowered == ("all",):
        return CANONICAL_SPLITS
    if "all" in lowered:
        raise click.UsageError("[target].splits cannot combine 'all' with other splits")

    unknown = sorted(set(lowered) - set(CANONICAL_SPLITS))
    if unknown:
        raise click.UsageError(
            f"Unknown split(s) in [target].splits: {', '.join(unknown)}"
        )
    return tuple(dict.fromkeys(lowered))


def _validate_allowed_keys(
    table: dict[str, Any],
    *,
    table_name: str,
    allowed_keys: set[str],
) -> None:
    unknown = sorted(set(table) - allowed_keys)
    if unknown:
        raise click.UsageError(
            f"[{table_name}] has unknown field(s): {', '.join(unknown)}"
        )


def load_runner_toml_config(config_path: str) -> RunnerTomlConfig:
    """Load and validate a runner config TOML file."""
    cfg_path = Path(config_path).expanduser().resolve()
    with open(cfg_path, "rb") as fh:
        raw = tomllib.load(fh)

    base_dir = cfg_path.parent
    target_table = _as_table(raw.get("target"), "target")
    agent_table = _as_table(raw.get("agent"), "agent")
    judge_table = _as_table(raw.get("judge"), "judge")
    run_table = _as_table(raw.get("run"), "run")

    if "endpoint" in judge_table:
        raise click.UsageError("[judge].endpoint has been renamed to [judge].base_url")
    if "smart_pass_at" in run_table:
        raise click.UsageError(
            "[run].smart_pass_at has been removed; use [run].pass_at"
        )

    _validate_allowed_keys(
        target_table,
        table_name="target",
        allowed_keys={
            "scenario",
            "dataset_root",
            "dataset",
            "splits",
            "subset_manifest",
            "limit",
            "recursive",
        },
    )
    _validate_allowed_keys(
        agent_table,
        table_name="agent",
        allowed_keys={
            "image",
            "runtime",
            "provider",
            "model",
            "api_key",
            "api_key_env",
            "base_url",
            "thinking",
            "volumes",
        },
    )
    _validate_allowed_keys(
        judge_table,
        table_name="judge",
        allowed_keys={"model", "provider", "base_url", "api_key", "api_key_env"},
    )
    _validate_allowed_keys(
        run_table,
        table_name="run",
        allowed_keys={
            "timeout",
            "health_timeout",
            "adapter_port",
            "concurrency",
            "pass_at",
            "retry",
            "output_dir",
            "output",
            "log_level",
            "notification_mode",
            "time_speed",
        },
    )

    scenario = _resolve_path(
        _as_optional_str(target_table.get("scenario"), "[target].scenario"),
        base_dir=base_dir,
        name="[target].scenario",
        must_exist=True,
        must_be_dir=False,
    )
    dataset_root = _resolve_path(
        _as_optional_str(target_table.get("dataset_root"), "[target].dataset_root"),
        base_dir=base_dir,
        name="[target].dataset_root",
        must_exist=True,
        must_be_dir=True,
    )
    dataset = _as_optional_str(target_table.get("dataset"), "[target].dataset")
    splits = _normalize_splits(target_table.get("splits"))
    subset_manifest = _resolve_path(
        _as_optional_str(
            target_table.get("subset_manifest"), "[target].subset_manifest"
        ),
        base_dir=base_dir,
        name="[target].subset_manifest",
        must_exist=True,
        must_be_dir=False,
    )
    limit = _as_optional_int(target_table.get("limit"), "[target].limit")
    recursive = _as_bool(
        target_table.get("recursive"), "[target].recursive", default=True
    )

    target_count = sum(1 for x in (scenario, dataset_root, dataset) if x)
    if target_count != 1:
        raise click.UsageError(
            "Config must set exactly one of [target].scenario, "
            "[target].dataset_root, or [target].dataset"
        )
    if limit is not None and limit < 1:
        raise click.UsageError("[target].limit must be >= 1")
    if scenario and splits:
        raise click.UsageError("[target].splits cannot be used with [target].scenario")
    if scenario and subset_manifest:
        raise click.UsageError(
            "[target].subset_manifest cannot be used with [target].scenario"
        )
    if scenario and target_table.get("recursive") is not None:
        raise click.UsageError(
            "[target].recursive is only supported for dataset targets"
        )

    target = TargetConfig(
        scenario=scenario,
        dataset_root=dataset_root,
        dataset=dataset,
        splits=splits,
        subset_manifest=subset_manifest,
        limit=limit,
        recursive=recursive,
    )

    image = _as_required_str(agent_table.get("image"), "[agent].image")
    runtime = (
        _as_optional_str(agent_table.get("runtime"), "[agent].runtime") or "podman"
    )
    provider = _as_optional_str(agent_table.get("provider"), "[agent].provider")
    model = _as_optional_str(agent_table.get("model"), "[agent].model")
    base_url = _as_optional_str(agent_table.get("base_url"), "[agent].base_url")
    thinking = _as_choice(
        agent_table.get("thinking"),
        "[agent].thinking",
        choices=("off", "low", "medium", "high"),
        default="low",
    )
    volumes = _as_string_list(agent_table.get("volumes"), "[agent].volumes")
    api_key = _resolve_secret(agent_table, section_name="agent")

    profile = detect_profile(image)
    if profile.requires_agent_llm and (not provider or not model):
        raise click.UsageError(
            "[agent].provider and [agent].model are required for this image"
        )

    agent = AgentConfig(
        image=image,
        runtime=runtime,
        provider=provider,
        model=model,
        api_key=api_key,
        base_url=base_url,
        thinking=thinking,
        volumes=volumes,
    )

    judge = JudgeConfig(
        model=_as_required_str(judge_table.get("model"), "[judge].model"),
        provider=_as_required_str(judge_table.get("provider"), "[judge].provider"),
        base_url=_as_optional_str(judge_table.get("base_url"), "[judge].base_url"),
        api_key=_resolve_secret(judge_table, section_name="judge"),
    )

    output_dir = _resolve_path(
        _as_optional_str(run_table.get("output_dir"), "[run].output_dir"),
        base_dir=base_dir,
        name="[run].output_dir",
        must_exist=False,
        must_be_dir=False,
    )
    output = _resolve_path(
        _as_optional_str(run_table.get("output"), "[run].output"),
        base_dir=base_dir,
        name="[run].output",
        must_exist=False,
        must_be_dir=False,
    )
    if not output and output_dir and not target.is_single_scenario:
        output = str(Path(output_dir) / "results.jsonl")

    run = RunConfig(
        timeout=_as_int(run_table.get("timeout"), "[run].timeout", default=600),
        health_timeout=_as_int(
            run_table.get("health_timeout"), "[run].health_timeout", default=120
        ),
        adapter_port=_as_int(
            run_table.get("adapter_port"), "[run].adapter_port", default=8090
        ),
        concurrency=_as_int(
            run_table.get("concurrency"), "[run].concurrency", default=1
        ),
        pass_at=_as_int(run_table.get("pass_at"), "[run].pass_at", default=1),
        retry=_as_bool(run_table.get("retry"), "[run].retry", default=False),
        output_dir=output_dir,
        output=output,
        log_level=_as_optional_str(run_table.get("log_level"), "[run].log_level")
        or "INFO",
        notification_mode=_as_choice(
            run_table.get("notification_mode"),
            "[run].notification_mode",
            choices=("message", "native"),
            default="message",
        ),
        time_speed=_as_optional_float(run_table.get("time_speed"), "[run].time_speed"),
    )

    if run.timeout < 1:
        raise click.UsageError("[run].timeout must be >= 1")
    if run.health_timeout < 1:
        raise click.UsageError("[run].health_timeout must be >= 1")
    if run.adapter_port < 1:
        raise click.UsageError("[run].adapter_port must be >= 1")
    if run.concurrency < 1:
        raise click.UsageError("[run].concurrency must be >= 1")
    if run.pass_at < 1:
        raise click.UsageError("[run].pass_at must be >= 1")
    if run.time_speed is not None and run.time_speed <= 0:
        raise click.UsageError("[run].time_speed must be > 0")
    if target.is_single_scenario and target.limit is not None:
        raise click.UsageError("[target].limit is only supported for dataset targets")
    if target.is_single_scenario and run.concurrency != 1:
        raise click.UsageError(
            "[run].concurrency is only supported for dataset targets"
        )
    if target.is_single_scenario and run.pass_at > 1:
        raise click.UsageError("[run].pass_at is only supported for dataset targets")
    if target.is_single_scenario and run.retry:
        raise click.UsageError("[run].retry is only supported for dataset targets")

    return RunnerTomlConfig(
        config_path=str(cfg_path),
        target=target,
        agent=agent,
        judge=judge,
        run=run,
    )
