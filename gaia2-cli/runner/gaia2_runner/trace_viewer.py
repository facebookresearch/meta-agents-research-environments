# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Static HTML generator for Gaia2 OpenClaw benchmark trace viewing.

Generates self-contained HTML files (inline CSS + JS, no external
dependencies) from run artifacts:

- Per-scenario trace page (``trace.html``): LLM call timeline with
  collapsible cards, tool call summary, judge results.
- Dataset index page (``index.html``): summary stats, sortable and
  filterable scenario table with links to individual traces.

Supports multiple trace formats transparently:

- **Format A** (raw): ``raw_response`` is an SSE or JSON string — parsed
  at render time via ``_normalize_entry()``.
- **Format B** (pre-parsed): ``response`` is a structured dict — used
  directly.
- **Format C** (no trace): no ``trace.jsonl`` at all — the viewer still
  renders judge results and tool calls, just skips the LLM trace panel.
"""

from __future__ import annotations

import html
import json
import logging
import math
import re
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any

from gaia2_runner.trace_parser import (
    canonicalize_entry as _canonicalize_entry,
)
from gaia2_runner.trace_parser import (
    extract_model as _extract_model,
)
from gaia2_runner.trace_parser import (
    extract_provider as _extract_provider,
)
from gaia2_runner.trace_parser import (
    load_jsonl as _load_jsonl,
)
from gaia2_runner.trace_parser import (
    parse_jsonl_text as _parse_jsonl_text,
)

logger: logging.Logger = logging.getLogger(__name__)

_NATURAL_SORT_RE = re.compile(r"(\d+)")
_SPLIT_DISPLAY_ORDER = ("search", "execution", "adaptability", "ambiguity", "time")
_SPLIT_DISPLAY_ORDER_INDEX = {
    split_name: index for index, split_name in enumerate(_SPLIT_DISPLAY_ORDER)
}
_DATASET_PARTITION_DIR_NAMES = frozenset({"train", "validation", "test"})
_SCENARIO_DISCOVERY_FILES = frozenset(
    {
        "result.json",
        "trace.jsonl",
        "events.jsonl",
        "daemon_status.json",
    }
)


def _looks_like_run_dir(path: Path) -> bool:
    """Return whether an immediate child directory is a pass@N run dir."""
    return path.is_dir() and (
        path.name.startswith("run_") or (path / "run_config.json").exists()
    )


def _natural_sort_key(p: Path) -> list[int | str]:
    """Sort key that orders numeric segments numerically (run_1, run_2, ..., run_10)."""
    return [
        int(s) if s.isdigit() else s.lower() for s in _NATURAL_SORT_RE.split(p.name)
    ]


def _coerce_timestamp_text(value: Any) -> str:
    """Normalize timestamps to a comparable text form.

    Trace artifacts are not fully consistent today:
    - detailed LLM traces use ISO timestamps (e.g. ``2024-10-15T07:00:05.000Z``)
    - Hermes summary traces use UNIX epoch floats
    """
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float)):
        from datetime import datetime, timezone

        try:
            return datetime.fromtimestamp(value, tz=timezone.utc).strftime(
                "%Y-%m-%d %H:%M:%S"
            )
        except (OverflowError, OSError, ValueError):
            return ""
    if value is None:
        return ""
    return str(value)


def _format_short_timestamp(value: Any) -> str:
    ts = _coerce_timestamp_text(value)
    return ts[:19].replace("T", " ") if ts else ""


def _format_compact_timestamp(value: Any) -> str:
    ts = _format_short_timestamp(value)
    return ts[:16] if ts else ""


def _timestamp_hms(value: Any) -> str:
    ts = _coerce_timestamp_text(value)
    return ts[11:19] if len(ts) >= 19 else ""


def _int_or_zero(value: Any) -> int:
    """Coerce nullable numeric trace fields to ints.

    Some OpenAI-compatible providers emit token counters as ``null`` for
    certain calls. The trace viewer should treat those as zero instead of
    crashing while aggregating usage.
    """

    if value in (None, ""):
        return 0
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            try:
                return int(float(value))
            except ValueError:
                return 0
    return 0


def _is_renderable_trace_entry(entry: dict[str, Any]) -> bool:
    """Return whether an entry contains an actual per-call LLM trace."""
    return (
        isinstance(entry.get("request"), dict)
        or isinstance(entry.get("response"), dict)
        or bool(entry.get("raw_response"))
    )


def _discover_scenario_ids(output_dir: Path) -> list[str]:
    """Return scenario directories discovered from runner artifacts.

    A scenario may be in progress before ``result.json`` exists, so discovery
    must consider early-written artifacts like ``trace.jsonl`` and
    ``daemon_status.json`` as well.
    """
    scenario_ids = {
        str(path.parent.relative_to(output_dir))
        for path in output_dir.rglob("*")
        if path.is_file()
        and path.name in _SCENARIO_DISCOVERY_FILES
        and path.parent != output_dir
    }
    return sorted(scenario_ids)


def _result_status(result: dict[str, Any], *, has_result: bool) -> tuple[str, str]:
    """Return a user-facing status label and matching badge class."""
    if not has_result:
        return "RUNNING", "badge-running"
    success = _get_success(result)
    status = "PASS" if success is True else ("FAIL" if success is False else "ERROR")
    return status, f"badge-{status.lower()}"


def _summary_status(summary: dict[str, Any]) -> str:
    """Return normalized scenario status for index rendering."""
    status = summary.get("_status")
    if isinstance(status, str) and status:
        return status.lower()
    success = _get_success(summary)
    return "pass" if success is True else ("fail" if success is False else "error")


def _infer_provider_from_model(model: str) -> str:
    if model.startswith("anthropic/") or model.startswith("claude-"):
        return "anthropic"
    if model.startswith("openai/") or model.startswith("gpt-"):
        return "openai"
    if model.startswith("google/") or model.startswith("gemini-"):
        return "google"
    return ""


def _render_tool_call(tc: dict[str, Any]) -> str:
    fn = tc.get("function", {})
    name = fn.get("name", "")
    args_str = fn.get("arguments", "")
    if name == "exec" and args_str:
        try:
            args_obj = json.loads(args_str)
            cmd = args_obj.get("command", args_str)
        except (json.JSONDecodeError, TypeError):
            cmd = args_str
        return f'<div class="tool-exec" data-msg-type="exec">$ {_esc(cmd)}</div>'

    parts = [f'<div class="tool-exec" data-msg-type="exec">{_esc(name)}']
    if args_str:
        try:
            args_str = json.dumps(json.loads(args_str), indent=2)
        except (json.JSONDecodeError, TypeError):
            pass
        parts.append(f" {_esc(args_str)}")
    parts.append("</div>")
    return "".join(parts)


def _render_trace_blocks(blocks: list[dict[str, Any]], *, role: str) -> list[str]:
    parts: list[str] = []
    text_parts: list[str] = []
    css_cls = f"msg-{role}" if role in ("user", "assistant", "tool") else "msg-user"
    role_label = _esc(role).title()

    def flush_text() -> None:
        if not text_parts:
            return
        text = "".join(text_parts)
        text_parts.clear()
        if role == "tool":
            parts.append(f'<div class="tool-output">{text}</div>')
            return
        parts.append(f'<div class="msg {css_cls}">')
        parts.append(f'<div class="msg-label">{role_label}</div>')
        parts.append(text)
        parts.append("</div>")

    for block in blocks:
        if not isinstance(block, dict):
            text_parts.append(_esc(str(block)[:500]))
            continue

        btype = block.get("type", "")
        if btype == "text":
            text = block.get("text", "")
            if text:
                text_parts.append(_esc(_truncate(text, 20000)))
            continue

        flush_text()

        if btype == "thinking":
            text = block.get("thinking", "")
            if text:
                parts.append(
                    '<details class="thinking-details" data-msg-type="thinking">'
                )
                parts.append(
                    f'<summary class="thinking-summary">Thinking ({len(text):,} chars)</summary>'
                )
                parts.append(
                    f'<div class="thinking-content">{_esc(_truncate(text, 20000))}</div>'
                )
                parts.append("</details>")
        elif btype == "tool_use":
            parts.append(_render_tool_exec(block))
        elif btype == "tool_result":
            parts.append(
                f'<div class="tool-output">{_esc(_truncate(str(block.get("content", "")), 20000))}</div>'
            )
        else:
            raw_type = block.get("raw_type", btype or "raw")
            parts.append(f"<em>[{_esc(raw_type)}]</em>")

    flush_text()
    return parts


def generate_all(
    output_dir: str | Path,
    *,
    home_prefix: str = "../..",
) -> list[dict[str, Any]]:
    """Generate HTML viewer for all scenarios in output_dir.

    Scans ``output_dir`` recursively for scenario subdirectories (those
    containing ``result.json``) and generates:
    - ``{scenario_id}/trace.html`` for each scenario
    - ``index.html`` at the top level

    If *dataset_path* is provided, it is used to look up oracle DAG data
    for scenarios that don't have ``scenario_dag.json`` (past runs).

    *home_prefix* is the relative path from a trace page back to the
    root index (default ``../..`` for ``run/scenario/trace.html``).
    For nested output structures, this is computed dynamically per scenario.
    """
    output_dir = Path(output_dir)
    if not output_dir.is_dir():
        logger.warning("Output directory does not exist: %s", output_dir)
        return []

    # Discover scenarios recursively — supports both flat and nested layouts.
    # Each scenario_id is the relative path from output_dir to the scenario
    # directory (e.g. "scenario_1" for flat, "search/validation/scenario_1"
    # for nested).
    scenario_ids = _discover_scenario_ids(output_dir)

    if not scenario_ids:
        # Generate a placeholder page so links don't 404
        _generate_waiting_page(output_dir)
        return []

    # Generate per-scenario trace pages, collecting summaries for the index
    scenario_summaries: list[dict[str, Any]] = []
    for i, sid in enumerate(scenario_ids):
        prev_id = scenario_ids[i - 1] if i > 0 else None
        next_id = scenario_ids[i + 1] if i < len(scenario_ids) - 1 else None

        # Compute home_prefix based on scenario depth. A scenario at
        # depth N (e.g. "a/b/scenario" → 3 parts) needs N+1 "../" to
        # reach the root from its trace.html.
        depth = len(Path(sid).parts)
        scenario_home_prefix = "/".join([".."] * (depth + 1))

        # Compute relative prev/next URLs from this trace.html location
        scenario_dir = output_dir / sid
        prev_rel: str | None = None
        next_rel: str | None = None
        if prev_id:
            prev_rel = _relpath_url(output_dir / prev_id / "trace.html", scenario_dir)
        if next_id:
            next_rel = _relpath_url(output_dir / next_id / "trace.html", scenario_dir)

        try:
            summary = _generate_trace_page(
                output_dir,
                sid,
                prev_id,
                next_id,
                home_prefix=scenario_home_prefix,
                prev_rel=prev_rel,
                next_rel=next_rel,
            )
            if summary is not None:
                scenario_summaries.append(summary)
        except Exception as exc:
            logger.warning("Failed to generate trace for %s: %s", sid, exc)

    # Generate index page using pre-computed summaries (no re-reading files)
    _generate_index_page(
        output_dir,
        scenario_summaries,
        run_metadata=_load_json(output_dir / "run_config.json"),
    )
    logger.info(
        "Generated HTML viewer: %d scenarios, index at %s/index.html",
        len(scenario_ids),
        output_dir,
    )
    return scenario_summaries


def _relpath_url(target: Path, base_dir: Path) -> str:
    """Compute a relative URL from *base_dir* to *target*.

    Both paths must share a common ancestor. Uses POSIX separators for URLs.
    """
    import os

    return os.path.relpath(str(target), str(base_dir)).replace(os.sep, "/")


def _generate_waiting_page(output_dir: Path) -> None:
    """Generate a placeholder index.html for a run with no results yet."""
    config = _load_json(output_dir / "run_config.json")
    run_name = output_dir.name
    model = config.get("model", "—")
    provider = config.get("provider", "—")
    num = config.get("num_scenarios", "?")
    started = config.get("started_at", "")
    started_short = _format_short_timestamp(started) if started else "—"

    parts: list[str] = []
    parts.append('<!DOCTYPE html><html><head><meta charset="utf-8">')
    parts.append(f"<title>{_esc(run_name)} — Starting</title>")
    parts.append(f"<style>{_CSS}</style>")
    parts.append('<meta http-equiv="refresh" content="10">')
    parts.append("</head><body>")

    # Header
    parts.append('<div class="header">')
    parts.append('<nav class="breadcrumb">')
    parts.append('<a href="../index.html">Home</a>')
    parts.append('<span class="sep">/</span>')
    parts.append(f"<span>{_esc(run_name)}</span>")
    parts.append("</nav>")
    parts.append(
        '<span class="badge" style="background:#f59e0b;color:#fff">STARTING</span>'
    )
    parts.append("</div>")

    parts.append('<div class="container">')
    parts.append(
        '<div style="text-align:center;padding:80px 20px">'
        '<div style="font-size:48px;margin-bottom:16px">&#x23F3;</div>'
        '<h2 style="font-size:20px;color:#334155;margin-bottom:8px">'
        "Waiting for results&hellip;</h2>"
        '<p style="color:#64748b;font-size:14px;margin-bottom:24px">'
        "No scenarios have completed yet. This page auto-refreshes.</p>"
    )

    # Run info
    parts.append(
        '<div style="display:inline-block;text-align:left;background:#f8fafc;'
        "border:1px solid #e2e8f0;border-radius:12px;padding:16px 24px;"
        'font-size:13px;color:#475569">'
    )
    parts.append(f"<div><strong>Model:</strong> {_esc(model)}</div>")
    parts.append(f"<div><strong>Provider:</strong> {_esc(provider)}</div>")
    parts.append(f"<div><strong>Scenarios:</strong> {_esc(str(num))}</div>")
    parts.append(f"<div><strong>Started:</strong> {_esc(started_short)}</div>")
    parts.append("</div>")

    parts.append("</div></div>")
    parts.append("</body></html>")

    (output_dir / "index.html").write_text("\n".join(parts))
    logger.info("Generated waiting page at %s/index.html", output_dir)


def _split_sort_key(split_name: str) -> tuple[int, str]:
    return (
        _SPLIT_DISPLAY_ORDER_INDEX.get(split_name, len(_SPLIT_DISPLAY_ORDER)),
        split_name,
    )


@lru_cache(maxsize=32)
def _dataset_split_counts(dataset_path: str) -> dict[str, int]:
    """Return expected scenario counts per split from a local dataset path."""
    if not dataset_path:
        return {}

    root = Path(dataset_path).expanduser()
    if not root.is_dir():
        return {}

    def count_json_files(path: Path) -> int:
        try:
            return sum(
                1
                for child in path.iterdir()
                if child.is_file() and child.suffix == ".json"
            )
        except OSError:
            return 0

    counts: dict[str, int] = {}

    if root.name in _SPLIT_DISPLAY_ORDER_INDEX:
        counts[root.name] = count_json_files(root)
    elif root.parent.name in _SPLIT_DISPLAY_ORDER_INDEX:
        counts[root.parent.name] = count_json_files(root)
    else:
        try:
            children = list(root.iterdir())
        except OSError:
            children = []
        for child in children:
            if not child.is_dir() or child.name not in _SPLIT_DISPLAY_ORDER_INDEX:
                continue
            direct_count = count_json_files(child)
            if direct_count:
                counts[child.name] = direct_count
                continue
            partition_count = 0
            for partition in child.iterdir():
                if (
                    partition.is_dir()
                    and partition.name in _DATASET_PARTITION_DIR_NAMES
                ):
                    partition_count += count_json_files(partition)
            if partition_count:
                counts[child.name] = partition_count

    return {
        split_name: counts[split_name]
        for split_name in sorted(counts, key=_split_sort_key)
    }


def _run_metadata_dataset_path(run_metadata: dict[str, Any]) -> str:
    """Return the local dataset directory referenced by run metadata."""
    for key in ("dataset_cache_dir", "dataset"):
        value = run_metadata.get(key)
        if not isinstance(value, str) or not value:
            continue
        candidate = Path(value).expanduser()
        if candidate.is_dir():
            return str(candidate)
    return ""


def _infer_summary_split(
    summary: dict[str, Any],
    run_metadata: dict[str, Any],
) -> str:
    sid = str(summary.get("_scenario_id") or summary.get("scenario_id") or "")
    sid_parts = Path(sid).parts
    if sid_parts and sid_parts[0] in _SPLIT_DISPLAY_ORDER_INDEX:
        return sid_parts[0]

    configured_splits = [
        split_name
        for split_name in (run_metadata.get("splits") or [])
        if isinstance(split_name, str) and split_name
    ]
    if len(configured_splits) == 1:
        return configured_splits[0]

    dataset_counts = _dataset_split_counts(_run_metadata_dataset_path(run_metadata))
    if len(dataset_counts) == 1:
        return next(iter(dataset_counts))

    dataset_split = run_metadata.get("dataset_split")
    if isinstance(dataset_split, str) and dataset_split in _SPLIT_DISPLAY_ORDER_INDEX:
        return dataset_split

    if len(sid_parts) > 1:
        return sid_parts[0]

    return ""


def _short_scenario_name(scenario_id: str, split_name: str) -> str:
    parts = Path(scenario_id).parts
    if split_name and parts and parts[0] == split_name:
        remainder = parts[1:]
        if remainder:
            return "/".join(remainder)
    return scenario_id


def _aggregate_summaries(
    summaries: list[dict[str, Any]],
    *,
    expected_total: int | None = None,
) -> dict[str, Any]:
    pass_count = sum(1 for summary in summaries if _summary_status(summary) == "pass")
    fail_count = sum(1 for summary in summaries if _summary_status(summary) == "fail")
    error_count = sum(1 for summary in summaries if _summary_status(summary) == "error")
    running_count = sum(
        1 for summary in summaries if _summary_status(summary) == "running"
    )
    total = len(summaries)
    completed_count = pass_count + fail_count + error_count
    normalized_expected = (
        max(expected_total, total) if isinstance(expected_total, int) else None
    )
    pending_count = (
        max(normalized_expected - total, 0)
        if isinstance(normalized_expected, int)
        else 0
    )
    total_tokens_in = sum(
        int(summary.get("total_input_tokens", 0) or 0) for summary in summaries
    )
    total_tokens_out = sum(
        int(summary.get("total_output_tokens", 0) or 0) for summary in summaries
    )
    total_latency_ms = sum(
        int(summary.get("total_latency_ms", 0) or 0) for summary in summaries
    )
    total_llm_calls = sum(
        int(summary.get("num_llm_calls", 0) or 0) for summary in summaries
    )
    total_tool_calls = sum(
        int(summary.get("_num_tool_calls", 0) or 0) for summary in summaries
    )
    denom = completed_count or total or 1
    pass_rate = (pass_count / completed_count * 100) if completed_count > 0 else 0.0

    return {
        "total": total,
        "expected_total": normalized_expected,
        "completed_count": completed_count,
        "pending_count": pending_count,
        "pass_count": pass_count,
        "fail_count": fail_count,
        "error_count": error_count,
        "running_count": running_count,
        "pass_rate": pass_rate,
        "total_tokens_in": total_tokens_in,
        "total_tokens_out": total_tokens_out,
        "total_tokens": total_tokens_in + total_tokens_out,
        "avg_tokens": (total_tokens_in + total_tokens_out) / denom,
        "total_latency_ms": total_latency_ms,
        "avg_latency_ms": total_latency_ms / denom,
        "total_llm_calls": total_llm_calls,
        "avg_llm_calls": total_llm_calls / denom,
        "total_tool_calls": total_tool_calls,
        "avg_tool_calls": total_tool_calls / denom,
    }


def _weighted_avg_stats(stats_by_run: list[dict[str, Any]]) -> dict[str, Any]:
    completed = [
        stats for stats in stats_by_run if int(stats.get("completed_count", 0) or 0) > 0
    ]
    if not completed:
        return {
            "rate": None,
            "stddev": None,
            "text": "—",
            "coverage_text": "No completed runs",
        }

    total_weight = sum(int(stats["completed_count"]) for stats in completed)
    mean_rate = (
        sum(
            float(stats["pass_rate"]) * int(stats["completed_count"])
            for stats in completed
        )
        / total_weight
    )
    stddev: float | None = None
    if len(completed) > 1:
        variance = (
            sum(
                int(stats["completed_count"])
                * (float(stats["pass_rate"]) - mean_rate) ** 2
                for stats in completed
            )
            / total_weight
        )
        stddev = math.sqrt(variance)

    text = f"{mean_rate:.1f}%"
    if stddev is not None:
        text = f"{mean_rate:.1f}% \u00b1 {stddev:.1f}%"

    return {
        "rate": mean_rate,
        "stddev": stddev,
        "text": text,
        "coverage_text": f"{len(completed)}/{len(stats_by_run)} runs completed",
    }


def _pass_at_stats(
    scenario_results: dict[str, dict[str, dict[str, Any]]],
    run_names: list[str],
    *,
    expected_total: int | None = None,
) -> dict[str, Any]:
    normalized_expected = (
        max(expected_total, len(scenario_results))
        if isinstance(expected_total, int)
        else len(scenario_results)
    )
    fully_observed: dict[str, dict[str, dict[str, Any]]] = {}
    for scenario_id, per_run in scenario_results.items():
        if all(
            run_name in per_run and _summary_status(per_run[run_name]) != "running"
            for run_name in run_names
        ):
            fully_observed[scenario_id] = per_run

    observed_count = len(fully_observed)
    passed_any = sum(
        1
        for per_run in fully_observed.values()
        if any(_summary_status(summary) == "pass" for summary in per_run.values())
    )
    rate: float | None = None
    std_error: float | None = None
    if observed_count > 0:
        rate = passed_any / observed_count * 100
        p = passed_any / observed_count
        std_error = math.sqrt(p * (1 - p) / observed_count) * 100

    if rate is None:
        text = "—"
    elif std_error is None:
        text = f"{rate:.1f}%"
    else:
        text = f"{rate:.1f}% \u00b1 {std_error:.1f}%"

    return {
        "rate": rate,
        "std_error": std_error,
        "text": text,
        "coverage_text": f"{observed_count}/{normalized_expected} fully observed",
        "fully_observed": observed_count,
        "expected_total": normalized_expected,
        "passed_any": passed_any,
    }


def _status_chip(status: str, label: str | None = None) -> str:
    badge_cls = f"badge-{status}"
    return f'<span class="badge {badge_cls}">{_esc(label or status.upper())}</span>'


def _aggregate_status(stats: dict[str, Any]) -> dict[str, str]:
    expected_total = stats.get("expected_total")
    total = int(stats.get("total", 0) or 0)
    completed_count = int(stats.get("completed_count", 0) or 0)
    running_count = int(stats.get("running_count", 0) or 0)

    if total == 0 and running_count == 0:
        if isinstance(expected_total, int) and expected_total > 0:
            return {"key": "pending", "label": "Not Started"}
        return {"key": "pending", "label": "Not Started"}

    if running_count > 0:
        return {"key": "running", "label": "Running"}

    if isinstance(expected_total, int) and completed_count < expected_total:
        return {"key": "running", "label": "Running"}

    return {"key": "complete", "label": "Complete"}


def _status_chip_for_stats(stats: dict[str, Any]) -> str:
    status = _aggregate_status(stats)
    return _status_chip(status["key"], status["label"])


def _completion_chip(stats: dict[str, Any]) -> str:
    expected_total = stats.get("expected_total")
    completed_count = int(stats.get("completed_count", 0) or 0)
    if isinstance(expected_total, int) and expected_total > 0:
        ratio = completed_count / expected_total if expected_total else 0
        if ratio >= 1:
            color = "#3f5c4d"
        elif ratio >= 0.5:
            color = "#8a6d3b"
        else:
            color = "#8a5348"
        return (
            f'<span class="metric-pill" style="--pill-accent:{color}">'
            f"{completed_count}/{expected_total}</span>"
        )
    return f'<span class="metric-pill">{completed_count}</span>'


def _format_rate_text(stats: dict[str, Any], *, label: str) -> str:
    value = stats.get("text") or "—"
    return f"{label} {value}"


def _pass_rate_text(stats: dict[str, Any]) -> str:
    completed_count = int(stats.get("completed_count", 0) or 0)
    if completed_count == 0:
        return "—"
    return f"{float(stats.get('pass_rate', 0.0)):.1f}%"


def _avg_latency_text(stats: dict[str, Any]) -> str:
    completed_count = int(stats.get("completed_count", 0) or 0)
    if completed_count == 0:
        return "—"
    return f"{float(stats.get('avg_latency_ms', 0.0)) / 1000:.1f}s"


def _progress_text(stats: dict[str, Any]) -> str:
    expected_total = stats.get("expected_total")
    completed_count = int(stats.get("completed_count", 0) or 0)
    if isinstance(expected_total, int) and expected_total > 0:
        return f"{completed_count}/{expected_total}"
    return str(completed_count)


def _rate_text(rate: float | None) -> str:
    if rate is None:
        return "—"
    return f"{rate:.1f}%"


def _display_image_name(image: Any) -> str:
    if not image:
        return "—"
    text = str(image).replace("localhost/", "").strip()
    lower = text.lower()
    if "openclaw" in lower or "gaia2-oc" in lower:
        return "openclaw"
    if "hermes" in lower:
        return "hermes"
    if "oracle" in lower:
        return "oracle"
    if "jarvis" in lower:
        return "jarvis"
    return text.split(":", 1)[0]


def _render_progress_meter(
    completed_count: int,
    total: int,
    *,
    running_count: int = 0,
    label: str | None = None,
    title: str | None = None,
    variant: str = "default",
) -> str:
    total = (
        max(total, completed_count + running_count, 1)
        if total > 0
        else max(completed_count + running_count, 1)
    )
    completed_pct = max(0.0, min(100.0, completed_count / total * 100))
    running_pct = max(0.0, min(100.0 - completed_pct, running_count / total * 100))
    if label is None:
        label = f"{completed_count}/{total}"
    stack_classes = ["progress-stack"]
    if variant == "compact":
        stack_classes.append("progress-stack-compact")
    elif variant == "landing":
        stack_classes.append("progress-stack-landing")
    title_attr = f' title="{_esc(title)}"' if title else ""
    return (
        f'<div class="{" ".join(stack_classes)}"{title_attr}>'
        '<div class="progress-track">'
        f'<span class="progress-fill progress-fill-complete" style="width:{completed_pct:.1f}%"></span>'
        f'<span class="progress-fill progress-fill-running" style="left:{completed_pct:.1f}%; width:{running_pct:.1f}%"></span>'
        "</div>"
        f'<div class="progress-meta"><span class="progress-label">{_esc(label)}</span></div>'
        "</div>"
    )


def _progress_meter(stats: dict[str, Any], *, variant: str = "default") -> str:
    expected_total = stats.get("expected_total")
    completed_count = int(stats.get("completed_count", 0) or 0)
    running_count = int(stats.get("running_count", 0) or 0)
    total = (
        int(expected_total)
        if isinstance(expected_total, int) and expected_total > 0
        else max(int(stats.get("total", 0) or 0), completed_count + running_count, 1)
    )
    return _render_progress_meter(
        completed_count,
        total,
        running_count=running_count,
        label=_progress_text(stats),
        variant=variant,
    )


def _pass_at_coverage_meter(
    pass_at_stats: dict[str, Any],
    *,
    run_count: int,
    variant: str = "default",
) -> str:
    expected_total = int(pass_at_stats.get("expected_total", 0) or 0)
    fully_observed = int(pass_at_stats.get("fully_observed", 0) or 0)
    total = max(expected_total, fully_observed, 1)
    title = (
        f"{fully_observed}/{total} scenarios completed in all {run_count} runs"
        if run_count > 0
        else None
    )
    return _render_progress_meter(
        fully_observed,
        total,
        label=f"{fully_observed}/{total}",
        title=title,
        variant=variant,
    )


def _render_split_chips(
    split_rows: dict[str, dict[str, Any]],
    *,
    href_prefix: str | None = None,
    variant: str = "default",
) -> str:
    if not split_rows:
        return '<span style="color:#8a8f89">—</span>'

    row_class = "split-chip-row"
    link_class = "split-chip-link"
    chip_extra_class = ""
    if variant == "landing":
        row_class += " split-chip-row-landing"
        link_class += " split-chip-link-landing"
        chip_extra_class = " split-chip-landing"

    parts: list[str] = [f'<div class="{row_class}">']
    for split_name in sorted(split_rows, key=_split_sort_key):
        row = split_rows[split_name]
        stats = row["stats"]
        display_text = row.get("display_text")
        if variant == "landing":
            display_text = row.get("display_compact_text", display_text)
        if not isinstance(display_text, str):
            display_text = _pass_rate_text(stats)
        status = row.get("display_status")
        if not isinstance(status, dict):
            status = _aggregate_status(stats)
        display_title = row.get("display_title")
        if not isinstance(display_title, str):
            display_title = (
                row.get("display_text")
                if isinstance(row.get("display_text"), str)
                else display_text
            )
        href_open = ""
        href_close = ""
        if href_prefix:
            href_open = f'<a class="{link_class}" href="{_esc(href_prefix)}{_esc(split_name)}.html">'
            href_close = "</a>"
        parts.append(href_open)
        parts.append(
            f'<span class="split-chip split-chip-{_esc(status["key"])}{chip_extra_class}" title="{_esc(display_title)}">'
            f'<span class="split-chip-name">{_esc(split_name[:3].upper())}</span>'
            f'<span class="split-chip-value">{_esc(display_text)}</span>'
            "</span>"
        )
        parts.append(href_close)
    parts.append("</div>")
    return "".join(parts)


def _build_run_view_model(
    run_dir: Path,
    run_metadata: dict[str, Any],
    scenario_summaries: list[dict[str, Any]],
) -> dict[str, Any]:
    dataset_counts = _dataset_split_counts(_run_metadata_dataset_path(run_metadata))
    configured_splits = [
        split_name
        for split_name in (run_metadata.get("splits") or [])
        if isinstance(split_name, str) and split_name
    ]

    split_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for summary in scenario_summaries:
        split_name = _infer_summary_split(summary, run_metadata)
        summary["_split"] = split_name
        summary["_scenario_label"] = _short_scenario_name(
            str(summary.get("_scenario_id", "")),
            split_name,
        )
        split_groups[split_name].append(summary)

    split_names = sorted(
        {
            split_name
            for split_name in (
                configured_splits
                + list(dataset_counts.keys())
                + list(split_groups.keys())
            )
            if split_name
        },
        key=_split_sort_key,
    )

    split_rows: dict[str, dict[str, Any]] = {}
    for split_name in split_names:
        summaries = split_groups.get(split_name, [])
        split_rows[split_name] = {
            "name": split_name,
            "summaries": summaries,
            "stats": _aggregate_summaries(
                summaries,
                expected_total=dataset_counts.get(split_name),
            ),
        }

    expected_total = run_metadata.get("num_scenarios")
    if not isinstance(expected_total, int) and dataset_counts:
        expected_total = sum(dataset_counts.values())

    return {
        "name": run_dir.name,
        "path": run_dir,
        "config": run_metadata,
        "scenario_summaries": scenario_summaries,
        "overall": _aggregate_summaries(
            scenario_summaries,
            expected_total=expected_total,
        ),
        "split_rows": split_rows,
        "split_names": split_names,
        "expected_by_split": dataset_counts,
    }


def _build_experiment_view_model(
    experiment_dir: Path,
    run_models: list[dict[str, Any]],
) -> dict[str, Any]:
    run_names = [run_model["name"] for run_model in run_models]
    expected_by_split: dict[str, int] = {}
    full_matrix: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    split_matrices: dict[str, dict[str, dict[str, dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    all_attempts: list[dict[str, Any]] = []

    for run_model in run_models:
        for split_name, count in run_model["expected_by_split"].items():
            expected_by_split[split_name] = max(
                expected_by_split.get(split_name, 0), count
            )
        all_attempts.extend(run_model["scenario_summaries"])
        for summary in run_model["scenario_summaries"]:
            split_name = str(summary.get("_split") or "")
            full_scenario_id = str(summary.get("_scenario_id") or "")
            short_name = str(
                summary.get("_scenario_label")
                or _short_scenario_name(full_scenario_id, split_name)
            )
            full_matrix[full_scenario_id][run_model["name"]] = summary
            if split_name:
                split_matrices[split_name][short_name][run_model["name"]] = summary

    configured_splits = [
        split_name
        for split_name in (run_models[0]["config"].get("splits") or [])
        if run_models and isinstance(split_name, str) and split_name
    ]
    split_names = sorted(
        {
            split_name
            for split_name in (
                configured_splits
                + list(expected_by_split.keys())
                + list(split_matrices.keys())
            )
            if split_name
        },
        key=_split_sort_key,
    )

    total_expected_scenarios = (
        sum(expected_by_split.values()) if expected_by_split else len(full_matrix)
    )
    total_expected_attempts = (
        total_expected_scenarios * len(run_models) if total_expected_scenarios else None
    )

    split_models: dict[str, dict[str, Any]] = {}
    for split_name in split_names:
        run_split_stats = [
            run_model["split_rows"].get(
                split_name,
                {
                    "name": split_name,
                    "summaries": [],
                    "stats": _aggregate_summaries(
                        [],
                        expected_total=run_model["expected_by_split"].get(split_name),
                    ),
                },
            )
            for run_model in run_models
        ]
        attempt_summaries = [
            summary
            for run_model in run_models
            for summary in run_model["split_rows"]
            .get(split_name, {})
            .get("summaries", [])
        ]
        split_expected_total = expected_by_split.get(split_name)
        split_models[split_name] = {
            "name": split_name,
            "scenario_results": split_matrices.get(split_name, {}),
            "attempts": _aggregate_summaries(
                attempt_summaries,
                expected_total=(
                    split_expected_total * len(run_models)
                    if split_expected_total
                    else None
                ),
            ),
            "avg": _weighted_avg_stats([row["stats"] for row in run_split_stats]),
            "pass_at": _pass_at_stats(
                split_matrices.get(split_name, {}),
                run_names,
                expected_total=split_expected_total,
            ),
            "per_run": {
                run_model["name"]: row["stats"]
                for run_model, row in zip(run_models, run_split_stats)
            },
            "expected_total": split_expected_total,
        }

    return {
        "name": experiment_dir.name,
        "path": experiment_dir,
        "config": run_models[0]["config"] if run_models else {},
        "run_models": run_models,
        "run_names": run_names,
        "split_names": split_names,
        "split_models": split_models,
        "full_matrix": full_matrix,
        "attempts": _aggregate_summaries(
            all_attempts,
            expected_total=total_expected_attempts,
        ),
        "avg": _weighted_avg_stats([run_model["overall"] for run_model in run_models]),
        "pass_at": _pass_at_stats(
            full_matrix,
            run_names,
            expected_total=total_expected_scenarios,
        ),
    }


def _landing_entry_from_run_model(run_model: dict[str, Any]) -> dict[str, Any]:
    config = run_model["config"]
    return {
        "kind": "run",
        "name": run_model["name"],
        "href": f"{_esc(run_model['name'])}/index.html",
        "run_count": 1,
        "config": config,
        "overall": run_model["overall"],
        "avg": {
            "text": f"{run_model['overall']['pass_rate']:.1f}%",
            "coverage_text": _completion_chip(run_model["overall"]),
        },
        "pass_at": {
            "text": f"{run_model['overall']['pass_rate']:.1f}%",
            "coverage_text": _completion_chip(run_model["overall"]),
        },
        "split_rows": run_model["split_rows"],
    }


def _landing_entry_from_experiment_model(
    experiment_model: dict[str, Any],
) -> dict[str, Any]:
    return {
        "kind": "experiment",
        "name": experiment_model["name"],
        "href": f"{_esc(experiment_model['name'])}/index.html",
        "run_count": len(experiment_model["run_models"]),
        "config": experiment_model["config"],
        "overall": experiment_model["attempts"],
        "avg": experiment_model["avg"],
        "pass_at": experiment_model["pass_at"],
        "split_rows": {
            split_name: {
                "name": split_name,
                "stats": split_model["attempts"],
                "display_text": split_model["avg"]["text"],
                "display_compact_text": _rate_text(split_model["avg"].get("rate")),
                "display_title": split_model["avg"]["text"],
                "display_status": _aggregate_status(split_model["attempts"]),
            }
            for split_name, split_model in experiment_model["split_models"].items()
        },
    }


def generate_runs_index(
    traces_root: str | Path,
    *,
    is_root: bool = True,
    depth: int = 0,
) -> dict[str, Any] | None:
    """Generate a landing page listing all runs in a traces directory.

    A directory whose immediate children are ``run_*`` subdirectories is
    treated as an experiment root and rendered as an experiment dashboard.
    Otherwise the directory is treated as a collection of experiments and/or
    standalone runs and rendered as a landing page.
    """
    traces_root = Path(traces_root)
    if not traces_root.is_dir():
        logger.warning("Traces root does not exist: %s", traces_root)
        return None

    immediate_run_dirs = sorted(
        [
            child
            for child in traces_root.iterdir()
            if child.is_dir() and child.name.startswith("run_")
        ],
        key=_natural_sort_key,
    )
    if immediate_run_dirs:
        run_models: list[dict[str, Any]] = []
        for run_dir in immediate_run_dirs:
            run_metadata = _load_json(run_dir / "run_config.json")
            scenario_summaries = generate_all(run_dir)
            run_models.append(
                _build_run_view_model(run_dir, run_metadata, scenario_summaries)
            )

        experiment_model = _build_experiment_view_model(traces_root, run_models)
        _generate_experiment_dashboard(
            traces_root,
            experiment_model,
            show_parent_link=not is_root,
        )
        _generate_split_dashboards(traces_root, experiment_model)
        logger.info(
            "Generated experiment dashboard: %d runs at %s/index.html",
            len(run_models),
            traces_root,
        )
        return _landing_entry_from_experiment_model(experiment_model)

    runs: list[dict[str, Any]] = []
    for d in sorted(traces_root.iterdir(), key=_natural_sort_key):
        if not d.is_dir():
            continue
        if d.name == "splits":
            continue
        child_run_dirs = [
            child
            for child in d.iterdir()
            if child.is_dir() and child.name.startswith("run_")
        ]
        if child_run_dirs:
            nested_summary = generate_runs_index(d, is_root=False, depth=depth + 1)
            if nested_summary is not None:
                runs.append(nested_summary)
            continue

        has_config = (d / "run_config.json").exists()
        has_scenarios = (
            any((d / filename).exists() for filename in _SCENARIO_DISCOVERY_FILES)
            or any(d.rglob("result.json"))
            or any(d.rglob("trace.jsonl"))
        )
        if not has_config and not has_scenarios:
            continue

        run_metadata = _load_json(d / "run_config.json")
        scenario_summaries = generate_all(d)
        run_model = _build_run_view_model(d, run_metadata, scenario_summaries)
        runs.append(_landing_entry_from_run_model(run_model))

    if not runs:
        logger.warning("No runs found in %s", traces_root)
        return None

    _generate_runs_landing(traces_root, runs, is_root=is_root)
    logger.info(
        "Generated runs landing: %d entries at %s/index.html",
        len(runs),
        traces_root,
    )
    return None


def _generate_runs_landing(
    traces_root: Path,
    runs: list[dict[str, Any]],
    *,
    is_root: bool = True,
) -> None:
    """Generate a landing page for experiments and standalone runs."""
    dir_name = traces_root.name
    parts: list[str] = []
    parts.append('<!DOCTYPE html><html><head><meta charset="utf-8">')
    if is_root:
        parts.append("<title>Gaia2 Eval Runs</title>")
    else:
        parts.append(f"<title>{_esc(dir_name)} — Gaia2 Eval</title>")
    parts.append(f"<style>{_CSS}</style>")
    parts.append('<meta http-equiv="refresh" content="30">')
    parts.append("</head><body>")

    parts.append('<div class="header shell-header">')
    if is_root:
        parts.append('<nav class="breadcrumb">')
        parts.append("<span>Home</span>")
        parts.append("</nav>")
    else:
        parts.append('<nav class="breadcrumb">')
        parts.append('<a href="../index.html">Home</a>')
        parts.append('<span class="sep">/</span>')
        parts.append(f"<span>{_esc(dir_name)}</span>")
        parts.append("</nav>")
    parts.append(f'<span class="header-kicker">{len(runs)} entries</span>')
    parts.append("</div>")

    parts.append('<div class="container">')
    parts.append('<section class="page-title">')
    parts.append(f"<h1>{_esc('All Experiments' if is_root else dir_name)}</h1>")
    parts.append("</section>")

    # Filters
    parts.append('<div class="filters panel">')
    parts.append(
        '<input type="text" id="searchInput" placeholder="Search experiments, runs, models, or providers..." oninput="filterTable()">'
    )
    parts.append("</div>")

    # Runs table
    parts.append('<section class="panel">')
    parts.append(
        '<div class="panel-header"><h2 class="section-title">Benchmarks</h2></div>'
    )
    parts.append(
        '<div class="table-wrap"><table id="scenarioTable" class="runs-table">'
    )
    parts.append("<thead><tr>")
    col = 0
    parts.append(
        f'<th onclick="sortTable({col})">Started <span class="sort-arrow"></span></th>'
    )
    col += 1
    parts.append(
        f'<th onclick="sortTable({col})">Name <span class="sort-arrow"></span></th>'
    )
    col += 1
    parts.append(
        f'<th onclick="sortTable({col})">Status <span class="sort-arrow"></span></th>'
    )
    col += 1
    parts.append(
        f'<th onclick="sortTable({col})">Model <span class="sort-arrow"></span></th>'
    )
    col += 1
    parts.append(
        f'<th onclick="sortTable({col})">Provider <span class="sort-arrow"></span></th>'
    )
    col += 1
    parts.append(
        f'<th onclick="sortTable({col})">Image <span class="sort-arrow"></span></th>'
    )
    col += 1
    parts.append(
        f'<th onclick="sortTable({col})">Splits <span class="sort-arrow"></span></th>'
    )
    col += 1
    parts.append(
        f'<th onclick="sortTable({col})">Progress <span class="sort-arrow"></span></th>'
    )
    col += 1
    parts.append(
        f'<th onclick="sortTable({col})">Avg <span class="sort-arrow"></span></th>'
    )
    col += 1
    parts.append(
        f'<th onclick="sortTable({col})">Pass <span class="sort-arrow"></span></th>'
    )
    parts.append("</tr></thead><tbody>")

    for r in runs:
        cfg = r["config"]
        name = r["name"]
        kind = r["kind"]
        model = cfg.get("model", "")
        provider = cfg.get("provider", "")
        started = cfg.get("started_at", "")
        started_short = _format_compact_timestamp(started) if started else "—"
        image_name = _display_image_name(cfg.get("image"))
        run_count = max(int(r.get("run_count", 1) or 1), 1)
        overall = r["overall"]
        status_chip = _status_chip_for_stats(overall)
        progress = _progress_meter(overall, variant="landing")
        primary_label = f"avg@{run_count}"
        primary_value = r["avg"]["text"]
        secondary_value = r["pass_at"]["text"] if r.get("pass_at") else "—"
        secondary_label = f"pass@{run_count}"
        secondary_title = r["pass_at"]["coverage_text"] if r.get("pass_at") else ""

        parts.append(f'<tr data-status="all" data-sid="{_esc(name.lower())}">')
        parts.append(f"<td>{_esc(started_short)}</td>")
        parts.append(f'<td><a href="{_esc(name)}/index.html">{_esc(name)}</a></td>')
        parts.append(f"<td>{status_chip}</td>")
        parts.append(f"<td>{_esc(model)}</td>")
        parts.append(f"<td>{_esc(provider)}</td>")
        parts.append(f"<td>{_esc(image_name)}</td>")
        parts.append(
            f"<td>{_render_split_chips(r['split_rows'], href_prefix=(f'{name}/splits/' if kind == 'experiment' else None), variant='landing')}</td>"
        )
        parts.append(f"<td>{progress}</td>")
        parts.append(
            f'<td><div class="metric-stack"><span class="metric-main">{_esc(primary_value)}</span><span class="metric-sub">{_esc(primary_label)}</span></div></td>'
        )
        parts.append(
            f'<td title="{_esc(secondary_title)}"><div class="metric-stack"><span class="metric-main">{_esc(secondary_value)}</span><span class="metric-sub">{_esc(secondary_label)}</span></div></td>'
        )
        parts.append("</tr>")

    parts.append("</tbody></table></div>")
    parts.append("</section>")
    parts.append("</div>")  # container

    parts.append("<script>")
    parts.append(_RUNS_JS)
    parts.append("</script>")

    parts.append("</body></html>")

    (traces_root / "index.html").write_text("\n".join(parts))


def _generate_experiment_dashboard(
    experiment_dir: Path,
    experiment_model: dict[str, Any],
    *,
    show_parent_link: bool,
) -> None:
    config = experiment_model["config"]
    run_models = experiment_model["run_models"]
    split_models = experiment_model["split_models"]
    model = config.get("model", "")
    provider = config.get("provider", "")
    avg_label = f"avg@{len(run_models)}"
    pass_label = f"pass@{len(run_models)}"
    parts: list[str] = []
    parts.append('<!DOCTYPE html><html><head><meta charset="utf-8">')
    parts.append(f"<title>{_esc(experiment_dir.name)} — Gaia2 Benchmark</title>")
    parts.append(f"<style>{_CSS}</style>")
    parts.append('<meta http-equiv="refresh" content="20">')
    parts.append("</head><body>")

    parts.append('<div class="header shell-header">')
    parts.append('<nav class="breadcrumb">')
    if show_parent_link:
        parts.append('<a href="../index.html">Home</a>')
        parts.append('<span class="sep">/</span>')
    else:
        parts.append("<span>Home</span>")
        parts.append('<span class="sep">/</span>')
    parts.append(f"<span>{_esc(experiment_dir.name)}</span>")
    parts.append("</nav>")
    parts.append('<span class="header-kicker">Experiment</span>')
    parts.append("</div>")

    parts.append('<div class="container">')
    coverage_label = f"Pass@{len(run_models)} Coverage"
    parts.append('<section class="hero hero-compact">')
    parts.append('<div class="hero-copy">')
    parts.append('<div class="hero-kicker">Experiment Dashboard</div>')
    parts.append(f"<h1>{_esc(experiment_dir.name)}</h1>")
    subtitle = []
    if model:
        subtitle.append(model)
    if provider:
        subtitle.append(provider)
    if config.get("image"):
        subtitle.append(str(config["image"]).replace("localhost/", ""))
    parts.append(
        f'<p class="hero-subtitle">{_esc(" • ".join(subtitle) or "Gaia2 evaluation run")}</p>'
    )
    parts.append("</div>")
    parts.append('<div class="hero-stats hero-stats-compact">')
    parts.append(
        f'<div class="hero-stat"><span class="hero-stat-value">{len(run_models)}</span><span class="hero-stat-label">Runs</span></div>'
    )
    parts.append(
        f'<div class="hero-stat"><span class="hero-stat-value">{_pass_at_coverage_meter(experiment_model["pass_at"], run_count=len(run_models), variant="compact")}</span><span class="hero-stat-label">{_esc(coverage_label)}</span></div>'
    )
    parts.append(
        f'<div class="hero-stat"><span class="hero-stat-value">{_progress_meter(experiment_model["attempts"], variant="compact")}</span><span class="hero-stat-label">Scenario Runs</span></div>'
    )
    parts.append(
        f'<div class="hero-stat"><span class="hero-stat-value">{_status_chip_for_stats(experiment_model["attempts"])}</span><span class="hero-stat-label">Status</span></div>'
    )
    parts.append("</div>")
    parts.append("</section>")

    parts.append('<div class="summary summary-grid">')
    parts.append(
        f'<div class="stat"><div class="value">{_esc(experiment_model["avg"]["text"])}</div><div class="label">{_esc(avg_label)}</div></div>'
    )
    parts.append(
        f'<div class="stat"><div class="value">{_esc(experiment_model["pass_at"]["text"])}</div><div class="label">{_esc(pass_label)}</div></div>'
    )
    parts.append(
        f'<div class="stat"><div class="value">{experiment_model["attempts"]["pass_count"]}</div><div class="label">Passed Attempts</div></div>'
    )
    parts.append(
        f'<div class="stat"><div class="value">{experiment_model["attempts"]["fail_count"]}</div><div class="label">Failed Attempts</div></div>'
    )
    parts.append(
        f'<div class="stat"><div class="value">{experiment_model["attempts"]["error_count"]}</div><div class="label">Errors</div></div>'
    )
    parts.append(
        f'<div class="stat"><div class="value">{_fmt_tokens(int(experiment_model["attempts"]["total_tokens"]))}</div><div class="label">Total Tokens</div></div>'
    )
    parts.append(
        f'<div class="stat"><div class="value">{experiment_model["attempts"]["avg_latency_ms"] / 1000:.1f}s</div><div class="label">Avg Latency</div></div>'
    )
    parts.append("</div>")

    parts.append('<section class="panel">')
    parts.append(
        '<div class="panel-header"><h2 class="section-title">Split Overview</h2></div>'
    )
    parts.append('<div class="table-wrap"><table>')
    parts.append(
        f"<thead><tr><th>Split</th><th>Status</th><th>{_esc(avg_label)}</th><th>{_esc(pass_label)}</th><th>{_esc(coverage_label)}</th><th>Scenario Runs</th><th>Errors</th><th>Tokens</th></tr></thead><tbody>"
    )
    for split_name in experiment_model["split_names"]:
        split_model = split_models[split_name]
        parts.append("<tr>")
        parts.append(
            f'<td><a href="splits/{_esc(split_name)}.html">{_esc(split_name)}</a></td>'
        )
        parts.append(f"<td>{_status_chip_for_stats(split_model['attempts'])}</td>")
        parts.append(
            f'<td><span class="metric-main">{_esc(split_model["avg"]["text"])}</span></td>'
        )
        parts.append(
            f'<td><span class="metric-main">{_esc(split_model["pass_at"]["text"])}</span></td>'
        )
        parts.append(
            f"<td>{_pass_at_coverage_meter(split_model['pass_at'], run_count=len(run_models), variant='compact')}</td>"
        )
        parts.append(
            f"<td>{_progress_meter(split_model['attempts'], variant='compact')}</td>"
        )
        parts.append(f"<td>{split_model['attempts']['error_count']}</td>")
        parts.append(
            f"<td>{_fmt_tokens(int(split_model['attempts']['total_tokens']))}</td>"
        )
        parts.append("</tr>")
    parts.append("</tbody></table></div>")
    parts.append("</section>")

    parts.append('<section class="panel">')
    parts.append(
        '<div class="panel-header"><h2 class="section-title">Run Status</h2></div>'
    )
    parts.append('<div class="table-wrap"><table>')
    parts.append(
        "<thead><tr><th>Run</th><th>Status</th><th>Progress</th><th>Pass Rate</th><th>Pass</th><th>Fail</th><th>Error</th><th>Latency</th><th>Split Breakdown</th></tr></thead><tbody>"
    )
    for run_model in run_models:
        overall = run_model["overall"]
        parts.append("<tr>")
        parts.append(
            f'<td><a href="{_esc(run_model["name"])}/index.html">{_esc(run_model["name"])}</a></td>'
        )
        parts.append(f"<td>{_status_chip_for_stats(overall)}</td>")
        parts.append(f"<td>{_progress_meter(overall, variant='compact')}</td>")
        parts.append(
            f'<td><span class="metric-main">{_esc(_pass_rate_text(overall))}</span></td>'
        )
        parts.append(f"<td>{overall['pass_count']}</td>")
        parts.append(f"<td>{overall['fail_count']}</td>")
        parts.append(f"<td>{overall['error_count']}</td>")
        parts.append(f"<td>{_esc(_avg_latency_text(overall))}</td>")
        parts.append(
            f"<td>{_render_split_chips(run_model['split_rows'], href_prefix='splits/')}</td>"
        )
        parts.append("</tr>")
    parts.append("</tbody></table></div>")
    parts.append("</section>")

    parts.append("</div>")
    parts.append("</body></html>")
    (experiment_dir / "index.html").write_text("\n".join(parts))


def _generate_split_dashboards(
    experiment_dir: Path,
    experiment_model: dict[str, Any],
) -> None:
    splits_dir = experiment_dir / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    run_names = experiment_model["run_names"]

    for split_name in experiment_model["split_names"]:
        split_model = experiment_model["split_models"][split_name]
        split_status = _aggregate_status(split_model["attempts"])
        coverage_label = f"Pass@{len(run_names)} Coverage"
        parts: list[str] = []
        parts.append('<!DOCTYPE html><html><head><meta charset="utf-8">')
        parts.append(f"<title>{_esc(split_name)} — {_esc(experiment_dir.name)}</title>")
        parts.append(f"<style>{_CSS}</style>")
        parts.append('<meta http-equiv="refresh" content="20">')
        parts.append("</head><body>")
        parts.append('<div class="header shell-header">')
        parts.append('<nav class="breadcrumb">')
        parts.append('<a href="../index.html">Home</a>')
        parts.append('<span class="sep">/</span>')
        parts.append(f"<span>{_esc(split_name)}</span>")
        parts.append("</nav>")
        parts.append('<span class="header-kicker">Split</span>')
        parts.append("</div>")

        parts.append('<div class="container">')
        parts.append('<section class="hero hero-compact">')
        parts.append('<div class="hero-copy">')
        parts.append('<div class="hero-kicker">Split Dashboard</div>')
        parts.append(f"<h1>{_esc(split_name)}</h1>")
        parts.append(
            f'<p class="hero-subtitle">{_esc(experiment_dir.name)} across {len(run_names)} runs. Status distinguishes not-started work from live attempts and completed results.</p>'
        )
        parts.append("</div>")
        parts.append('<div class="hero-stats hero-stats-compact">')
        parts.append(
            f'<div class="hero-stat"><span class="hero-stat-value">{len(run_names)}</span><span class="hero-stat-label">Runs</span></div>'
        )
        parts.append(
            f'<div class="hero-stat"><span class="hero-stat-value">{_pass_at_coverage_meter(split_model["pass_at"], run_count=len(run_names), variant="compact")}</span><span class="hero-stat-label">{_esc(coverage_label)}</span></div>'
        )
        parts.append(
            f'<div class="hero-stat"><span class="hero-stat-value">{_progress_meter(split_model["attempts"], variant="compact")}</span><span class="hero-stat-label">Scenario Runs</span></div>'
        )
        parts.append(
            f'<div class="hero-stat"><span class="hero-stat-value">{_status_chip(split_status["key"], split_status["label"])}</span><span class="hero-stat-label">Status</span></div>'
        )
        parts.append("</div>")
        parts.append("</section>")

        parts.append('<div class="summary summary-grid">')
        parts.append(
            f'<div class="stat"><div class="value">{_esc(split_model["avg"]["text"])}</div><div class="label">avg@{len(run_names)}</div></div>'
        )
        parts.append(
            f'<div class="stat"><div class="value">{_esc(split_model["pass_at"]["text"])}</div><div class="label">pass@{len(run_names)}</div></div>'
        )
        parts.append(
            f'<div class="stat"><div class="value">{split_model["attempts"]["pass_count"]}</div><div class="label">Passed Attempts</div></div>'
        )
        parts.append(
            f'<div class="stat"><div class="value">{split_model["attempts"]["fail_count"]}</div><div class="label">Failed Attempts</div></div>'
        )
        parts.append(
            f'<div class="stat"><div class="value">{split_model["attempts"]["error_count"]}</div><div class="label">Errors</div></div>'
        )
        parts.append("</div>")

        parts.append('<section class="panel">')
        parts.append(
            '<div class="panel-header"><h2 class="section-title">Run Breakdown</h2></div>'
        )
        parts.append('<div class="table-wrap"><table>')
        parts.append(
            "<thead><tr><th>Run</th><th>Status</th><th>Progress</th><th>Pass Rate</th><th>Pass</th><th>Fail</th><th>Error</th><th>Latency</th></tr></thead><tbody>"
        )
        for run_model in experiment_model["run_models"]:
            run_stats = split_model["per_run"].get(
                run_model["name"],
                _aggregate_summaries([], expected_total=split_model["expected_total"]),
            )
            parts.append("<tr>")
            parts.append(
                f'<td><a href="../{_esc(run_model["name"])}/index.html">{_esc(run_model["name"])}</a></td>'
            )
            parts.append(f"<td>{_status_chip_for_stats(run_stats)}</td>")
            parts.append(f"<td>{_progress_meter(run_stats, variant='compact')}</td>")
            parts.append(
                f'<td><span class="metric-main">{_esc(_pass_rate_text(run_stats))}</span></td>'
            )
            parts.append(f"<td>{run_stats['pass_count']}</td>")
            parts.append(f"<td>{run_stats['fail_count']}</td>")
            parts.append(f"<td>{run_stats['error_count']}</td>")
            parts.append(f"<td>{_esc(_avg_latency_text(run_stats))}</td>")
            parts.append("</tr>")
        parts.append("</tbody></table></div>")
        parts.append("</section>")

        parts.append('<section class="panel">')
        parts.append(
            '<div class="panel-header"><h2 class="section-title">Scenario Matrix</h2></div>'
        )
        parts.append(
            '<div class="filters panel" style="margin-bottom:16px"><input type="text" id="scenarioSearch" placeholder="Search scenario..." oninput="filterSplitTable()"></div>'
        )
        parts.append('<div class="table-wrap"><table id="comparisonTable">')
        parts.append("<thead><tr><th>Scenario</th>")
        for run_name in run_names:
            parts.append(f"<th>{_esc(run_name)}</th>")
        parts.append("<th>Passes</th><th>Detail</th></tr></thead><tbody>")
        for scenario_name in sorted(split_model["scenario_results"]):
            per_run = split_model["scenario_results"][scenario_name]
            pass_count = 0
            detail = ""
            parts.append(f'<tr data-sid="{_esc(scenario_name.lower())}">')
            parts.append(f"<td>{_esc(scenario_name)}</td>")
            for run_name in run_names:
                summary = per_run.get(run_name)
                if summary is None:
                    parts.append(f"<td>{_status_chip('pending', 'Not Started')}</td>")
                    continue
                status = _summary_status(summary)
                if status == "pass":
                    pass_count += 1
                if not detail and summary.get("failure_reasons"):
                    detail = str(summary["failure_reasons"][0])
                if not detail and summary.get("error"):
                    detail = str(summary["error"])
                trace_href = f"../{_esc(run_name)}/{_esc(summary.get('_scenario_id', scenario_name))}/trace.html"
                parts.append(
                    f'<td><a href="{trace_href}">{_status_chip(status)}</a></td>'
                )
            parts.append(f"<td>{pass_count}/{len(run_names)}</td>")
            parts.append(
                f'<td class="truncated" title="{_esc(detail)}">{_esc(_truncate(detail, 140))}</td>'
            )
            parts.append("</tr>")
        parts.append("</tbody></table></div>")
        parts.append("</section>")

        parts.append("</div>")
        parts.append("<script>")
        parts.append(_SPLIT_JS)
        parts.append("</script>")
        parts.append("</body></html>")
        (splits_dir / f"{split_name}.html").write_text("\n".join(parts))


# ---------------------------------------------------------------------------
# CSS shared across all pages
# ---------------------------------------------------------------------------

_CSS = """\
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&family=IBM+Plex+Mono:wght@400;500;600&display=swap');
:root {
  --bg: #f3f0ea;
  --bg-accent: #ece7de;
  --panel: rgba(255, 252, 247, 0.92);
  --panel-strong: #fffdf8;
  --line: #ddd5c8;
  --line-strong: #c8beaf;
  --text: #1f2722;
  --muted: #6a726d;
  --muted-strong: #525851;
  --accent: #516b5a;
  --accent-soft: #d8e2d7;
  --warn: #8e6c34;
  --fail: #9a5447;
  --pass: #476454;
  --info: #58708d;
  --shadow: 0 18px 44px rgba(41, 36, 29, 0.06);
  --radius: 22px;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: 'Manrope', 'Avenir Next', 'Segoe UI', sans-serif; line-height: 1.6; color: var(--text); background:
  radial-gradient(circle at top left, rgba(108, 136, 118, 0.08), transparent 28%),
  radial-gradient(circle at top right, rgba(148, 123, 89, 0.08), transparent 24%),
  linear-gradient(180deg, #f7f4ee 0%, var(--bg) 42%, #efebe3 100%);
  -webkit-font-smoothing: antialiased; }
a { color: var(--accent); text-decoration: none; font-weight: 600; }
a:hover { color: #324a3c; text-decoration: none; }

.header { background: rgba(247, 243, 236, 0.82); color: var(--text); padding: 16px 28px; display: flex; align-items: center; gap: 16px; flex-wrap: wrap; position: sticky; top: 0; z-index: 100; border-bottom: 1px solid rgba(200, 190, 175, 0.7); backdrop-filter: blur(12px); }
.header h1 { font-size: 15px; font-weight: 700; letter-spacing: -0.3px; color: var(--text); }
.shell-header { box-shadow: 0 12px 32px rgba(40, 36, 31, .05); }
.header-kicker { margin-left: auto; color: var(--muted); font-size: 11px; font-weight: 700; letter-spacing: 0.14em; text-transform: uppercase; }
.badge { display: inline-block; padding: 4px 14px; border-radius: 20px; font-size: 10px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.8px; border: 1px solid transparent; }
.badge-pass { background: var(--pass); color: #f7f8f5; }
.badge-fail { background: var(--fail); color: #fff7f5; }
.badge-running { background: rgba(88, 112, 141, 0.14); border-color: rgba(88, 112, 141, 0.28); color: #35526a; }
.badge-error { background: #b97247; color: #fff8f1; }
.badge-pending { background: rgba(108, 118, 128, 0.12); border-color: rgba(108, 118, 128, 0.26); color: #56616b; }
.badge-complete { background: rgba(71, 100, 84, 0.14); border-color: rgba(71, 100, 84, 0.26); color: #355140; }
.stats { display: flex; gap: 20px; font-size: 12px; color: var(--muted); }
.stats span { white-space: nowrap; }
.nav-links { margin-left: auto; display: flex; gap: 12px; font-size: 12px; }
.nav-links a { color: var(--accent); font-weight: 600; }

.container { max-width: 1440px; margin: 0 auto; padding: 28px 22px 48px; }
.page-title { display: flex; align-items: center; justify-content: space-between; gap: 16px; margin: 2px 0 20px; padding: 2px 2px 0; }
.page-title h1 { font-size: clamp(28px, 4vw, 42px); line-height: 1.05; letter-spacing: -0.05em; color: var(--text); }
.hero { display: grid; grid-template-columns: minmax(0, 1.8fr) minmax(280px, 0.9fr); gap: 28px; background: linear-gradient(140deg, rgba(255,255,255,0.68), rgba(247,243,236,0.88)); border: 1px solid rgba(200,190,175,0.8); box-shadow: var(--shadow); border-radius: calc(var(--radius) + 2px); padding: 28px 30px; margin-bottom: 24px; }
.hero-compact { grid-template-columns: minmax(0, 1.9fr) minmax(340px, 1.1fr); gap: 22px; padding: 22px 24px; }
.hero-copy h1 { font-size: clamp(28px, 4vw, 42px); line-height: 1.05; letter-spacing: -0.05em; color: var(--text); margin-top: 6px; }
.hero-kicker { color: var(--muted); font-size: 11px; letter-spacing: 0.18em; text-transform: uppercase; font-weight: 800; }
.hero-subtitle { color: var(--muted); font-size: 14px; max-width: 760px; margin-top: 12px; }
.hero-stats { display: grid; grid-template-columns: repeat(1, minmax(0, 1fr)); gap: 12px; align-self: center; }
.hero-stats-compact { grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 10px; align-self: start; }
.hero-stat { background: rgba(255, 253, 248, 0.92); border: 1px solid rgba(200, 190, 175, 0.75); border-radius: 18px; padding: 16px 18px; }
.hero-compact .hero-stat { padding: 12px 14px; border-radius: 16px; }
.hero-stat-value { display: block; font-size: 18px; font-weight: 800; color: var(--text); line-height: 1.2; }
.hero-compact .hero-stat-value { font-size: 16px; }
.hero-stat-label { display: block; margin-top: 4px; color: var(--muted); font-size: 10px; letter-spacing: 0.14em; text-transform: uppercase; font-weight: 700; }
.hero-compact .hero-stat-label { margin-top: 6px; }
.panel { background: var(--panel); border: 1px solid rgba(200, 190, 175, 0.85); box-shadow: var(--shadow); border-radius: var(--radius); padding: 22px 24px; margin-bottom: 24px; }
.panel-header { display: flex; align-items: center; justify-content: space-between; gap: 16px; margin-bottom: 14px; }
.summary-grid { gap: 18px; }
.split-card-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 14px; }
.split-card { display: block; background: rgba(255, 253, 248, 0.86); border: 1px solid rgba(200, 190, 175, 0.7); border-radius: 20px; padding: 16px 18px; box-shadow: 0 10px 26px rgba(41, 36, 29, 0.04); transition: transform .16s ease, border-color .16s ease, box-shadow .16s ease; color: inherit; }
.split-card:hover { transform: translateY(-2px); border-color: rgba(81, 107, 90, 0.55); box-shadow: 0 18px 34px rgba(41, 36, 29, 0.08); }
.split-card-title { color: var(--muted); font-size: 11px; font-weight: 800; letter-spacing: 0.14em; text-transform: uppercase; }
.split-card-value { font-size: 28px; font-weight: 800; letter-spacing: -0.06em; color: var(--text); margin-top: 10px; }
.split-card-meta { display: flex; flex-direction: column; gap: 4px; margin-top: 10px; color: var(--muted); font-size: 12px; }
.metric-main { font-size: 16px; font-weight: 800; color: var(--text); }
.metric-sub { font-size: 10px; text-transform: uppercase; letter-spacing: 0.14em; color: var(--muted); font-weight: 800; }
.metric-stack { display: flex; flex-direction: column; gap: 3px; }
.metric-pill { display: inline-flex; align-items: center; justify-content: center; min-width: 74px; padding: 5px 10px; border-radius: 999px; background: rgba(219, 227, 219, 0.45); color: var(--text); font-size: 11px; font-weight: 700; border: 1px solid rgba(200, 190, 175, 0.72); }
.progress-stack { min-width: 180px; display: flex; flex-direction: column; gap: 8px; }
.progress-track { position: relative; height: 8px; border-radius: 999px; overflow: hidden; background: rgba(210, 203, 193, 0.62); }
.progress-fill { position: absolute; top: 0; bottom: 0; border-radius: 999px; }
.progress-fill-complete { left: 0; background: linear-gradient(90deg, #648170, #476454); }
.progress-fill-running { background: linear-gradient(90deg, #88a2bc, #58708d); }
.progress-meta { display: flex; align-items: center; justify-content: space-between; gap: 8px; }
.progress-label { font-size: 11px; font-weight: 800; color: var(--muted-strong); }
.progress-stack-compact { min-width: 120px; gap: 6px; }
.progress-stack-compact .progress-track { height: 6px; }
.progress-stack-compact .progress-label { font-size: 10px; }
.progress-stack-landing { min-width: 96px; max-width: 112px; gap: 6px; }
.progress-stack-landing .progress-track { height: 6px; }
.progress-stack-landing .progress-label { font-size: 10px; }
.hero-compact .progress-stack,
.hero-compact .progress-stack-compact { min-width: 0; }
.split-chip-row { display: flex; flex-wrap: wrap; gap: 8px; }
.split-chip-link { display: inline-flex; }
.split-chip { display: inline-flex; align-items: center; gap: 8px; border-radius: 999px; padding: 5px 10px; background: rgba(232, 236, 230, 0.92); border: 1px solid rgba(200, 190, 175, 0.72); color: var(--text); }
.split-chip-row-landing { display: flex; flex-wrap: wrap; gap: 9px; min-width: 320px; }
.split-chip-link-landing { display: inline-flex; }
.split-chip-landing { min-width: 84px; justify-content: space-between; padding: 6px 10px; background: rgba(250, 248, 242, 0.96); border-color: rgba(196, 186, 172, 0.74); box-shadow: inset 0 1px 0 rgba(255,255,255,0.8); }
.split-chip-pending { background: rgba(235, 237, 239, 0.96); border-color: rgba(177, 184, 191, 0.7); }
.split-chip-running { background: rgba(227, 236, 244, 0.96); border-color: rgba(133, 158, 181, 0.72); }
.split-chip-complete { background: rgba(230, 238, 233, 0.96); border-color: rgba(133, 163, 145, 0.72); }
.split-chip-name { font-size: 10px; font-weight: 800; letter-spacing: 0.16em; text-transform: uppercase; color: var(--muted); }
.split-chip-value { font-size: 11px; font-weight: 800; white-space: nowrap; }
.split-card-status-row { display: flex; align-items: center; gap: 8px; margin-top: 10px; }
.split-card-value-empty { color: var(--muted); }

/* Two-column trace layout — breaks out of .container max-width */
.has-trace .container { max-width: none; padding: 24px 5vw; }
.trace-layout { display: flex; gap: 0; align-items: flex-start; }
.trace-main { flex: 3; min-width: 0; padding-right: 14px; }
.trace-splitter { width: 2px; cursor: col-resize; background: #e2e8f0; flex-shrink: 0; align-self: stretch; min-height: 200px; transition: background 0.15s; padding: 0 3px; background-clip: content-box; }
.trace-splitter:hover, .trace-splitter.active { background: #3b82f6; background-clip: content-box; }
.trace-sidebar { flex: 2; position: sticky; top: 60px; max-height: calc(100vh - 80px); overflow-y: auto; padding-left: 14px; }
.trace-main > .section-title, .trace-sidebar > .section-title:first-child { margin-top: 0; }
@media (max-width: 1100px) { .trace-layout { flex-direction: column; } .trace-splitter { display: none; } .trace-sidebar { position: static; flex: none; width: 100%; max-height: none; } }

/* Trace cards */
.card { background: var(--panel-strong); border-radius: 18px; box-shadow: var(--shadow); margin-bottom: 20px; overflow: hidden; border: 1px solid rgba(200, 190, 175, 0.85); }
.card-header { padding: 14px 20px; background: #f7f2e8; border-bottom: 1px solid rgba(200, 190, 175, 0.8); cursor: pointer; display: flex; align-items: center; gap: 12px; user-select: none; position: sticky; top: 52px; z-index: 10; transition: background .15s; }
.card-header:hover { background: #efe8da; }
.card-header .arrow { transition: transform .2s ease; font-size: 11px; color: var(--muted); }
.card-header.open .arrow { transform: rotate(90deg); }
.card-header .title { font-weight: 700; font-size: 14px; color: var(--text); }
.card-header .meta { font-size: 12px; color: var(--muted); margin-left: auto; display: flex; gap: 16px; }
.card-body { padding: 24px; display: none; }
.card-header.open + .card-body { display: block; }

/* Messages */
.msg { margin-bottom: 8px; padding: 10px 14px; border-radius: 8px; font-size: 13px; white-space: pre-wrap; word-break: break-word; line-height: 1.6; max-width: 900px; }
.msg-system { background: #fefce8; border-left: 3px solid #eab308; color: #713f12; }
.msg-user { background: #eff6ff; border-left: 3px solid #3b82f6; color: #1e3a5f; }
.msg-assistant { background: #f0fdf4; border-left: 3px solid #22c55e; color: #14532d; }
.msg-tool { background: #f5f3ff; border-left: 3px solid #8b5cf6; color: #3b0764; }
.msg-label { font-weight: 700; font-size: 10px; text-transform: uppercase; letter-spacing: 0.8px; margin-bottom: 3px; color: var(--muted); }

/* Tool pair (exec + output side by side) */
.tool-pair { display: flex; margin: 12px 0; border-radius: 12px; overflow: hidden; border: 1px solid #1e293b; min-height: 42px; }
.tool-pair .tool-exec { border: none; border-radius: 0; margin: 0; border-right: 1px solid #334155; flex: 0 0 auto; max-width: 45%; min-width: 200px; display: flex; align-items: flex-start; }
.tool-pair .tool-output { border: none; border-radius: 0; margin: 0; flex: 1; max-height: 250px; overflow-y: auto; }
/* Tool exec (command) — dark terminal */
.tool-exec { background: #1d2320; border-radius: 12px; margin: 12px 0; padding: 12px 16px; font-family: 'IBM Plex Mono', 'JetBrains Mono', monospace; font-size: 13px; color: #9dd5ad; border: 1px solid #2a312d; }
/* Tool output (result) — dark terminal */
.tool-output { background: #1d2320; border-radius: 12px; margin: 0 0 12px 0; padding: 14px 16px; font-family: 'IBM Plex Mono', 'JetBrains Mono', monospace; font-size: 12px; color: #c3cbc6; border: 1px solid #2a312d; max-height: 250px; overflow-y: auto; white-space: pre-wrap; word-break: break-word; line-height: 1.6; }
/* Standalone tool-call (fallback) */
.tool-call { background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 12px; margin: 12px 0; padding: 14px 16px; }
.tool-call .tool-name { font-weight: 600; color: #7c3aed; font-size: 13px; font-family: 'SF Mono', 'JetBrains Mono', 'Fira Code', 'Consolas', monospace; }
/* ENV event pair — amber themed, same layout as .tool-pair */
.env-pair { display: flex; margin: 12px 0; border-radius: 12px; overflow: hidden; border: 1px solid #f59e0b; min-height: 42px; }
.env-pair .env-exec { flex: 0 0 auto; max-width: 50%; min-width: 200px; padding: 12px 16px; background: #fffbeb; border-right: 1px solid #f59e0b; font-family: 'SF Mono', 'JetBrains Mono', 'Fira Code', 'Consolas', monospace; font-size: 13px; color: #78350f; word-break: break-all; }
.env-pair .env-output { flex: 1; padding: 12px 16px; background: #fffef5; max-height: 250px; overflow-y: auto; font-family: 'SF Mono', 'JetBrains Mono', 'Fira Code', 'Consolas', monospace; font-size: 13px; color: #374151; white-space: pre-wrap; word-break: break-word; }
.env-label { font-size: 10px; font-weight: 700; color: #b45309; margin-bottom: 4px; }
.env-badge { background: #fef3c7; color: #92400e; padding: 1px 6px; border-radius: 3px; font-size: 10px; font-weight: 600; margin-left: 6px; }
.tool-call pre { background: #0f172a; color: #cbd5e1; padding: 12px 14px; border-radius: 8px; font-size: 12px; overflow-x: auto; margin-top: 8px; white-space: pre-wrap; word-break: break-word; font-family: 'SF Mono', 'JetBrains Mono', 'Fira Code', 'Consolas', monospace; line-height: 1.6; }
/* Thinking (collapsed) */
.thinking-details { margin-bottom: 16px; max-width: 900px; }
.thinking-summary { cursor: pointer; padding: 10px 16px; background: #fefce8; border-left: 3px solid #eab308; border-radius: 12px; font-weight: 600; font-size: 11px; text-transform: uppercase; letter-spacing: 0.6px; color: #92400e; list-style: none; display: flex; align-items: center; gap: 8px; transition: background .15s; }
.thinking-summary:hover { background: #fef9c3; }
.thinking-summary::before { content: '\\25B6'; font-size: 8px; transition: transform .2s; }
.thinking-details[open] .thinking-summary::before { transform: rotate(90deg); }
.thinking-summary::-webkit-details-marker { display: none; }
details[open] .ignore-arrow { transform: rotate(90deg); }
.thinking-content { padding: 14px 16px; background: #fefce8; border-left: 3px solid #eab308; border-radius: 0 0 12px 12px; font-size: 13px; white-space: pre-wrap; word-break: break-word; line-height: 1.7; color: #713f12; margin-top: -12px; }

/* Token bar */
.token-bar { display: flex; gap: 2px; height: 4px; border-radius: 4px; overflow: hidden; margin: 6px 0; background: #e2e8f0; }
.token-bar .in { background: #3b82f6; }
.token-bar .out { background: #22c55e; }
.token-bar .cache { background: #64748b; }
.token-legend { font-size: 11px; color: #64748b; display: flex; gap: 14px; }
.token-legend span::before { content: ''; display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 5px; vertical-align: middle; }
.token-legend .l-in::before { background: #3b82f6; }
.token-legend .l-out::before { background: #22c55e; }
.token-legend .l-cache::before { background: #64748b; }

/* Table */
.table-wrap { overflow-x: auto; }
table { width: 100%; border-collapse: collapse; font-size: 13px; }
th { background: #f2ece1; padding: 12px 16px; text-align: left; border-bottom: 2px solid rgba(200, 190, 175, 0.9); cursor: pointer; white-space: nowrap; user-select: none; font-weight: 700; color: var(--muted-strong); font-size: 11px; letter-spacing: 0.08em; text-transform: uppercase; }
th:hover { background: #ebe3d4; }
th .sort-arrow { font-size: 10px; margin-left: 4px; }
td { padding: 12px 16px; border-bottom: 1px solid rgba(221, 213, 200, 0.8); color: var(--text); vertical-align: top; }
tr:hover td { background: rgba(243, 238, 228, 0.7); }

/* Comparison table */
#comparisonTable th:last-child,
#comparisonTable td:last-child {
  border-left: 3px solid #e2e8f0;
  background: #f8fafc;
  text-align: center;
}
#comparisonTable td:not(:first-child):not(:last-child) {
  text-align: center;
}
#comparisonTable tr:nth-child(even) td {
  background: #f8fafc;
}
#comparisonTable tr:nth-child(even) td:last-child {
  background: #f1f5f9;
}
#comparisonTable th:first-child,
#comparisonTable td:first-child {
  position: sticky;
  left: 0;
  background: #fff;
  z-index: 1;
}
#comparisonTable tr:nth-child(even) td:first-child {
  background: #f8fafc;
}

/* Filters */
.filters { display: flex; gap: 12px; margin-bottom: 20px; align-items: center; flex-wrap: wrap; }
.filters select, .filters input { padding: 11px 14px; border: 1px solid rgba(200, 190, 175, 0.9); border-radius: 14px; font-size: 13px; background: rgba(255, 252, 247, 0.92); color: var(--text); transition: border-color .15s, box-shadow .15s; }
.filters select:focus, .filters input:focus { outline: none; border-color: rgba(81, 107, 90, 0.7); box-shadow: 0 0 0 3px rgba(81,107,90,.08); }
.filters input { flex: 1; max-width: 300px; }
.filters input::placeholder { color: var(--muted); }

/* Summary banner */
.summary { background: var(--panel); border-radius: var(--radius); box-shadow: var(--shadow); padding: 24px 26px; margin-bottom: 24px; display: flex; gap: 22px; flex-wrap: wrap; border: 1px solid rgba(200, 190, 175, 0.85); }
.summary .stat { text-align: center; }
.summary .stat .value { font-size: 30px; font-weight: 800; color: var(--text); letter-spacing: -0.06em; }
.summary .stat .label { font-size: 10px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.14em; font-weight: 700; margin-top: 4px; }

/* Turn divider */
.turn-divider { display: flex; align-items: center; gap: 14px; margin: 20px 0 16px; color: #94a3b8; font-size: 10px; font-weight: 600; letter-spacing: 0.5px; text-transform: uppercase; }
.turn-divider::before, .turn-divider::after { content: ''; flex: 1; border-top: 1px dashed #cbd5e1; }
.turn-divider .turn-info { white-space: nowrap; display: flex; gap: 12px; background: #f1f5f9; padding: 4px 14px; border-radius: 20px; }

/* Section titles */
.section-title { font-size: 13px; font-weight: 800; margin: 0; color: var(--muted-strong); letter-spacing: 0.12em; text-transform: uppercase; }

/* Rationale */
.rationale { background: #f8fafc; border-radius: 12px; padding: 16px; font-size: 13px; white-space: pre-wrap; word-break: break-word; border: 1px solid #e2e8f0; line-height: 1.7; color: #334155; }

/* System prompt details */
.system-details summary { background: #fefce8; border-left: 3px solid #eab308; color: #92400e; border-radius: 12px; }
.system-details summary:hover { background: #fef9c3; }

/* Truncated text with hover */
.truncated { max-width: 300px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }

/* Fold summary */
.fold-summary { cursor: pointer; padding: 6px 14px; border-radius: 8px; font-size: 11px; font-weight: 600; color: #64748b; letter-spacing: 0.3px; margin-bottom: 8px; list-style: none; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.fold-summary::-webkit-details-marker { display: none; }
.fold-user { background: #eff6ff; border-left: 3px solid #3b82f6; }
.fold-assistant { background: #f0fdf4; border-left: 3px solid #22c55e; }
.fold-exec { background: #0f172a; color: #4ade80; font-family: 'SF Mono', 'JetBrains Mono', monospace; }
.fold-output { background: #f1f5f9; }

/* Prev/Next navigation arrows */
.nav-arrow { position: fixed; top: 50%; transform: translateY(-50%); z-index: 200; width: 44px; height: 44px; border-radius: 50%; background: #0f172a; color: #e2e8f0; display: flex; align-items: center; justify-content: center; font-size: 20px; text-decoration: none; box-shadow: 0 2px 8px rgba(0,0,0,.2); transition: background .15s, transform .15s; }
.nav-arrow:hover { background: #1e293b; transform: translateY(-50%) scale(1.1); text-decoration: none; color: #fff; }
.nav-arrow-left { left: 12px; }
.nav-arrow-right { right: 12px; }

/* Scrollbar styling */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #cbd5e1; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #94a3b8; }
.tool-output::-webkit-scrollbar-thumb { background: #334155; }
.tool-output::-webkit-scrollbar-thumb:hover { background: #475569; }

/* Breadcrumb navigation */
.breadcrumb { display:flex; align-items:center; gap:6px; font-size:12px; color:var(--muted); margin-right:auto; }
.breadcrumb a { color:var(--accent); font-weight:600; text-decoration:none; }
.breadcrumb a:hover { color:#324a3c; }
.breadcrumb .sep { color:#938b80; }

/* Progress bar (runs index) */
.progress-bar-wrap { position:relative; width:80px; height:20px; background:#d7d0c4; border-radius:10px; overflow:hidden; display:inline-block; }
.progress-bar-fill { height:100%; background:var(--accent); border-radius:10px; transition:width .3s; }
.progress-bar-text { position:absolute; inset:0; display:flex; align-items:center; justify-content:center; font-size:10px; font-weight:700; color:#fff; }

@media (max-width: 980px) {
  .hero { grid-template-columns: 1fr; }
  .hero-stats { grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); }
}
"""


# ---------------------------------------------------------------------------
# Per-scenario trace page
# ---------------------------------------------------------------------------


def _extract_dag_from_scenario_data(
    scenario_data: dict[str, Any],
) -> list[dict[str, Any]]:
    """Extract a compact oracle DAG from scenario data (no runner dependency)."""
    raw_events = scenario_data.get("events") or scenario_data.get(
        "completed_events", []
    )
    dag: list[dict[str, Any]] = []
    # Fields that carry user-visible text get a higher truncation limit
    _long_fields = {"content", "message", "body", "text", "description"}

    for ev in raw_events:
        action = ev.get("action", {})
        raw_args = action.get("args", {})
        if isinstance(raw_args, list):
            args_summary = {
                a["name"]: str(a["value"])[: 500 if a["name"] in _long_fields else 80]
                for a in raw_args
                if "name" in a
            }
        elif isinstance(raw_args, dict):
            args_summary = {
                k: str(v)[: 500 if k in _long_fields else 80]
                for k, v in raw_args.items()
            }
        else:
            args_summary = {}
        dag.append(
            {
                "event_id": ev.get("event_id", ""),
                "event_type": ev.get("event_type", ""),
                "dependencies": ev.get("dependencies", []),
                "app": action.get("app", ""),
                "function": action.get("function_name") or action.get("function", ""),
                "args": args_summary,
            }
        )
    return dag


def _load_scenario_dag(
    dataset_path: str, scenario_id: str
) -> list[dict[str, Any]] | None:
    """Load the oracle DAG for a single scenario by direct file lookup.

    Looks for ``{dataset_path}/{scenario_id}.json`` and parses only that file,
    avoiding the need to scan the entire dataset directory.
    """
    ds = Path(dataset_path)
    if not ds.is_dir():
        return None
    sf = ds / f"{scenario_id}.json"
    if not sf.exists():
        return None
    try:
        data = json.loads(sf.read_text())
        dag = _extract_dag_from_scenario_data(data)
        return dag if dag else None
    except Exception:
        return None


def _resolve_dataset_path(scenario_dir: Path) -> str | None:
    """Resolve the dataset path from the parent run's ``run_config.json``."""
    for parent in (scenario_dir,) + tuple(scenario_dir.parents):
        run_config = parent / "run_config.json"
        if not run_config.exists():
            continue
        try:
            config = json.loads(run_config.read_text())
            resolved = _run_metadata_dataset_path(config)
            if resolved:
                return resolved
        except Exception:
            continue
    return None


def _normalize_ad_judgments(judgments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert agent-driven per-action judgments to turn-based format.

    The AD judge writes one record per agent write action with an optional
    ``turn`` field.  The trace viewer expects turn-grouped records with
    ``id_mapping``, ``match_details``, ``oracle_events``, and
    ``agent_events``.  This function groups by turn and builds that format.
    """
    from collections import defaultdict

    # Group records by turn index.  Records with turn=None are either:
    # - post-completion actions (all oracle matched, extra write) → skip
    # - mid-turn rejections (agent did wrong action) → assign to last turn
    turns: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    last_turn = 0
    for j in judgments:
        turn = j.get("turn")
        if turn is not None:
            last_turn = turn
            turns[turn].append(j)
        else:
            # Check if all oracle events were already matched
            mc = j.get("matched_count", 0) or 0
            to = j.get("total_oracle", 0) or 0
            if mc >= to and to > 0:
                continue  # post-completion action, skip
            # Mid-turn rejection — assign to last known turn
            turns[last_turn].append(j)

    result: list[dict[str, Any]] = []
    for turn_idx in sorted(turns.keys(), key=lambda t: (isinstance(t, str), t)):
        records = turns[turn_idx]
        id_mapping: dict[str, str] = {}
        match_details: list[dict[str, Any]] = []
        agent_events: list[dict[str, Any]] = []
        oracle_events: list[dict[str, Any]] = []
        seen_oracle_ids: set[str] = set()
        last_failure: str | None = None

        for j in records:
            ae = j.get("agent_event") or {}
            oe = j.get("oracle_event")
            success = j.get("success", False)

            agent_events.append(
                {
                    "id": ae.get("id", ""),
                    "tool": ae.get("tool", ""),
                    "args_summary": ae.get("args_summary", {}),
                }
            )

            if oe and oe.get("id") not in seen_oracle_ids:
                oracle_events.append(
                    {
                        "id": oe.get("id", ""),
                        "tool": oe.get("tool", ""),
                        "args": oe.get("args_summary", {}),
                    }
                )
                seen_oracle_ids.add(oe.get("id", ""))

            if success and oe:
                id_mapping[ae.get("id", "")] = oe.get("id", "")
                match_details.append(
                    {
                        "oracle_id": oe["id"],
                        "oracle_tool": oe["tool"],
                        "matched": True,
                        "reason": "",
                        "judge_output": j.get("judge_output") or "",
                    }
                )
            elif not success and oe:
                last_failure = j.get("failure_reason", "")
                match_details.append(
                    {
                        "oracle_id": oe["id"],
                        "oracle_tool": oe["tool"],
                        "matched": False,
                        "reason": last_failure,
                        "judge_output": j.get("judge_output") or "",
                    }
                )
            elif not success:
                last_failure = j.get("failure_reason", "")

        all_success = all(j.get("success", False) for j in records)
        result.append(
            {
                "turn": turn_idx,
                "success": all_success,
                "failure_reason": last_failure if not all_success else None,
                "id_mapping": id_mapping,
                "match_details": match_details,
                "oracle_events": oracle_events,
                "agent_events": agent_events,
            }
        )

    return result


def _is_ad_judgments(judgments: list[dict[str, Any]]) -> bool:
    """Check if judgments are from the agent-driven judge."""
    return bool(judgments) and judgments[0].get("type") == "agent_driven"


def _load_dag_data(scenario_dir: Path, scenario_id: str) -> dict[str, Any] | None:
    """Load DAG visualization data for a scenario.

    Returns a dict with ``oracle_dag``, ``judgments`` (matching info),
    or ``None`` if no DAG data is available.
    """
    # Look up oracle DAG from dataset (auto-discovered from run_config.json)
    oracle_dag: list[dict[str, Any]] | None = None
    resolved = _resolve_dataset_path(scenario_dir)
    if resolved:
        oracle_dag = _load_scenario_dag(resolved, scenario_id)

    if not oracle_dag:
        return None

    # 3. Load judgments for matching info (fall back to daemon judgments)
    judgments_raw = _load_text(scenario_dir / "judgments.jsonl")
    if not judgments_raw.strip():
        judgments_raw = _load_text(scenario_dir / "daemon_judgments.jsonl")
    judgments = _parse_jsonl_text(judgments_raw) if judgments_raw.strip() else []
    if _is_ad_judgments(judgments):
        judgments = _normalize_ad_judgments(judgments)

    # Merge matching data across all judgment turns (multi-turn scenarios
    # have one entry per turn, each with its own id_mapping/match_details)
    id_mapping: dict[str, str] = {}
    match_details: list[dict[str, Any]] = []
    agent_events: list[dict[str, Any]] = []
    seen_oracle_ids: set[str] = set()
    for j in judgments:
        for k, v in (j.get("id_mapping") or {}).items():
            id_mapping[k] = v
        for md in j.get("match_details") or []:
            oid = md.get("oracle_id", "")
            if oid not in seen_oracle_ids:
                match_details.append(md)
                seen_oracle_ids.add(oid)
        # Use agent_events from the last turn (cumulative)
        if j.get("agent_events"):
            agent_events = j["agent_events"]

    return {
        "oracle_dag": oracle_dag,
        "id_mapping": id_mapping,
        "match_details": match_details,
        "agent_events": agent_events,
    }


def _generate_trace_page(
    output_dir: Path,
    scenario_id: str,
    prev_id: str | None,
    next_id: str | None,
    *,
    home_prefix: str = "../..",
    prev_rel: str | None = None,
    next_rel: str | None = None,
) -> dict[str, Any] | None:
    """Generate trace.html for a single scenario.

    Returns a summary dict with pre-computed stats for use by
    ``_generate_index_page``, avoiding a second read of the same files.
    Returns ``None`` if the scenario cannot be processed.
    """
    scenario_dir = output_dir / scenario_id

    # Load artifacts
    result_path = scenario_dir / "result.json"
    has_result = result_path.exists()
    result = _load_json(result_path)
    raw_trace_entries = _load_jsonl(scenario_dir / "trace.jsonl")
    events_raw = _load_text(scenario_dir / "events.jsonl")
    events = _parse_jsonl_text(events_raw) if events_raw else []
    diagnostics = _parse_diagnostics(scenario_dir)
    daemon_turns = _parse_daemon_turns(scenario_dir)

    explorer_url = ""

    # Load DAG data for visualization
    dag_data = _load_dag_data(scenario_dir, scenario_id)

    # Normalize trace entries (Format A → Format B) and filter out
    # non-LLM entries (e.g. npmjs.org registry requests)
    normalized_trace_entries = [
        _canonicalize_entry(e)
        for e in raw_trace_entries
        if "npmjs.org" not in (e.get("url") or "")
    ]
    trace_entries = [
        entry for entry in normalized_trace_entries if _is_renderable_trace_entry(entry)
    ]

    success = _get_success(result) if has_result else None
    status, badge_cls = _result_status(result, has_result=has_result)

    model = _extract_model(trace_entries, result)
    provider = _extract_provider(trace_entries, result)
    if not provider:
        provider = _infer_provider_from_model(model)

    # Compute stats from normalized entries
    num_llm_calls = (
        len(trace_entries) if trace_entries else int(result.get("num_llm_calls") or 0)
    )
    total_input = int(result.get("total_input_tokens") or 0)
    total_output = int(result.get("total_output_tokens") or 0)
    total_latency = int(result.get("total_latency_ms") or 0)
    num_tool_calls = len(events)

    if trace_entries:
        total_input = 0
        total_output = 0
        total_latency = 0
        for e in trace_entries:
            u = e.get("response", {}).get("usage", {})
            total_input += (
                _int_or_zero(u.get("input_tokens"))
                + _int_or_zero(u.get("prompt_tokens"))
                + _int_or_zero(u.get("cache_creation_input_tokens"))
            )
            total_output += _int_or_zero(u.get("output_tokens")) + _int_or_zero(
                u.get("completion_tokens")
            )
            total_latency += _int_or_zero(e.get("latency_ms"))

    parts: list[str] = []
    parts.append('<!DOCTYPE html><html><head><meta charset="utf-8">')
    parts.append(f"<title>Trace: {_esc(scenario_id)}</title>")
    parts.append(f"<style>{_CSS}</style>")
    if dag_data:
        parts.append(f"<style>{_DAG_CSS}</style>")
        parts.append('<script src="https://d3js.org/d3.v7.min.js"></script>')
    parts.append('</head><body class="has-trace">')

    # Header
    run_name = output_dir.name
    # Compute the relative path from this trace.html to the run's index.html.
    # For flat scenarios (depth=1) this is "../index.html".
    # For nested scenarios the depth is greater.
    sid_depth = len(Path(scenario_id).parts)
    run_index_rel = "/".join([".."] * sid_depth) + "/index.html"

    parts.append('<div class="header">')
    parts.append('<nav class="breadcrumb">')
    parts.append(f'<a href="{home_prefix}/index.html">Home</a>')
    parts.append('<span class="sep">/</span>')
    parts.append(f'<a href="{run_index_rel}">{_esc(run_name)}</a>')
    parts.append('<span class="sep">/</span>')
    parts.append(f"<span>{_esc(scenario_id)}</span>")
    parts.append("</nav>")
    parts.append(f'<span class="badge {badge_cls}">{status}</span>')
    if explorer_url:
        parts.append(
            f' <a href="{_esc(explorer_url)}" target="_blank" '
            f'style="font-size:13px;color:#58a6ff;text-decoration:none;margin-left:8px"'
            f' title="Open in Gaia2 Explorer">&#x1F50D; Explorer</a>'
        )
    parts.append('<div class="stats">')
    parts.append(f"<span>{num_llm_calls} LLM calls</span>")
    parts.append(
        f"<span>{_fmt_tokens(total_input)} in / {_fmt_tokens(total_output)} out</span>"
    )
    parts.append(f"<span>{total_latency / 1000:.1f}s latency</span>")
    parts.append(f"<span>{num_tool_calls} tool calls</span>")
    if model:
        parts.append(f"<span>{_esc(model)}</span>")
    parts.append("</div>")

    # Navigation — use pre-computed relative URLs when available (nested
    # output), falling back to sibling-relative links for flat output.
    prev_href = prev_rel or (f"../{_esc(prev_id)}/trace.html" if prev_id else None)
    next_href = next_rel or (f"../{_esc(next_id)}/trace.html" if next_id else None)
    parts.append('<div class="nav-links">')
    if prev_href:
        parts.append(f'<a href="{prev_href}">&larr; Prev</a>')
    if next_href:
        parts.append(f'<a href="{next_href}">Next &rarr;</a>')
    parts.append("</div>")
    parts.append("</div>")  # header

    # Fixed prev/next arrows on screen edges
    if prev_href:
        parts.append(
            f'<a class="nav-arrow nav-arrow-left" href="{prev_href}" title="Previous scenario">&#8592;</a>'
        )
    if next_href:
        parts.append(
            f'<a class="nav-arrow nav-arrow-right" href="{next_href}" title="Next scenario">&#8594;</a>'
        )

    parts.append('<div class="container">')

    # ---- DAG Visualization (full width, above trace layout) ----
    if dag_data:
        parts.append(
            '<details class="dag-section" open>'
            '<summary class="section-title" style="cursor:pointer">'
            "Oracle DAG</summary>"
        )
        parts.append('<div id="dag-container"></div>')
        # Legend
        parts.append(
            '<div class="dag-legend">'
            '<span class="dag-legend-item">'
            '<span class="dag-swatch" style="background:#c8e6c9;border-color:#4CAF50"></span> Matched</span>'
            '<span class="dag-legend-item">'
            '<span class="dag-swatch" style="background:#ffcdd2;border-color:#EF5350;border-style:dashed"></span> Unmatched</span>'
            '<span class="dag-legend-item">'
            '<span class="dag-swatch" style="background:#bbdefb;border-color:#00ABFF"></span> User</span>'
            '<span class="dag-legend-item">'
            '<span class="dag-swatch" style="background:#b2dfdb;border-color:#009688"></span> Env</span>'
            '<span class="dag-legend-item">'
            '<span class="dag-swatch" style="background:#e1bee7;border-color:#C930C8"></span> Agent Extra</span>'
            "</div>"
        )
        parts.append("</details>")
        # Embed DAG data as JSON for the JS to pick up
        parts.append("<script>")
        parts.append(f"window.__DAG_DATA = {json.dumps(dag_data, default=str)};")
        parts.append("</script>")

    parts.append('<div class="trace-layout">')

    # ---- Left: LLM Trace ----
    parts.append('<div class="trace-main">')
    if trace_entries:
        parts.append('<h2 class="section-title">LLM Trace</h2>')
        parts.append('<div class="card"><div class="card-body" style="display:block">')
        parts.append(_render_unified_trace(trace_entries, daemon_turns=daemon_turns))
        parts.append("</div></div>")

    # Daemon turn events (shown below trace if no trace entries)
    if daemon_turns and not trace_entries:
        parts.append('<h2 class="section-title">Daemon Turns</h2>')
        for dt in daemon_turns:
            parts.append(_render_daemon_turn_block(dt))

    # Raw Logs tabs
    raw_logs_html = _render_raw_logs_tabs(scenario_dir)
    if raw_logs_html:
        parts.append('<h2 class="section-title">Raw Logs</h2>')
        parts.append(raw_logs_html)

    parts.append("</div>")

    # ---- Draggable splitter ----
    parts.append('<div class="trace-splitter" id="traceSplitter"></div>')

    # ---- Right: Sidebar (Tool Calls + Judge) ----
    parts.append('<div class="trace-sidebar" id="traceSidebar">')
    # Judge Result (first — most important)
    parts.append('<h2 class="section-title">Judge Result</h2>')
    parts.append(_render_judge_result(result, has_result=has_result))
    # Judge Log (detailed matching info from judgments.jsonl,
    # fall back to daemon_judgments.jsonl for in-container judging)
    judgments_raw = _load_text(scenario_dir / "judgments.jsonl")
    if not judgments_raw.strip():
        judgments_raw = _load_text(scenario_dir / "daemon_judgments.jsonl")
    if judgments_raw.strip():
        judgments = _parse_jsonl_text(judgments_raw)
        if _is_ad_judgments(judgments):
            judgments = _normalize_ad_judgments(judgments)
        if judgments:
            parts.append('<h2 class="section-title">Judge Log</h2>')
            parts.append(_render_judge_log(judgments, events))
    # Diagnostics (only when errors exist)
    if diagnostics.get("has_errors"):
        parts.append('<h2 class="section-title">Diagnostics</h2>')
        parts.append(_render_diagnostics(diagnostics))
    # Tool Call Summary
    if events:
        parts.append('<h2 class="section-title">Tool Calls</h2>')
        parts.append(_render_events_table(events))
    parts.append("</div>")

    parts.append("</div>")  # trace-layout
    parts.append("</div>")  # container

    # Comparison modal (single instance, reused by JS)
    parts.append(
        '<div id="jl-compare-overlay" onclick="jlCloseCompare(event)" '
        'style="display:none;position:fixed;inset:0;z-index:1000;'
        "background:rgba(0,0,0,.45);backdrop-filter:blur(2px);"
        'align-items:center;justify-content:center">'
        '<div id="jl-compare-modal" onclick="event.stopPropagation()" '
        'style="background:#fff;border-radius:16px;'
        "box-shadow:0 8px 32px rgba(0,0,0,.2);"
        "width:min(90vw,960px);max-height:85vh;overflow-y:auto;"
        'padding:24px 28px">'
        "</div></div>"
    )

    # JS for collapsible cards
    parts.append("<script>")
    parts.append(_TRACE_JS)
    parts.append("</script>")

    # DAG visualization JS (after d3 is loaded)
    if dag_data:
        parts.append("<script>")
        parts.append(_DAG_JS)
        parts.append("</script>")

    # Draggable splitter JS
    parts.append("""<script>
(function() {
  const splitter = document.getElementById('traceSplitter');
  if (!splitter) return;
  const layout = splitter.parentElement;
  const main = splitter.previousElementSibling;
  const sidebar = splitter.nextElementSibling;
  if (!layout || !main || !sidebar) return;

  let dragging = false;
  splitter.addEventListener('mousedown', function(e) {
    e.preventDefault();
    dragging = true;
    splitter.classList.add('active');
    document.body.style.cursor = 'col-resize';
    document.body.style.userSelect = 'none';
  });
  document.addEventListener('mousemove', function(e) {
    if (!dragging) return;
    const rect = layout.getBoundingClientRect();
    const pct = ((e.clientX - rect.left) / rect.width) * 100;
    const clamped = Math.max(20, Math.min(80, pct));
    main.style.flex = 'none';
    main.style.width = clamped + '%';
    sidebar.style.flex = 'none';
    sidebar.style.width = (100 - clamped - 1) + '%';
  });
  document.addEventListener('mouseup', function() {
    if (!dragging) return;
    dragging = false;
    splitter.classList.remove('active');
    document.body.style.cursor = '';
    document.body.style.userSelect = '';
  });
})();
</script>""")

    parts.append("</body></html>")

    html_path = scenario_dir / "trace.html"
    html_path.write_text("\n".join(parts))

    # Return pre-computed summary so _generate_index_page can skip re-reading
    return {
        "_scenario_id": scenario_id,
        "_status": status.lower(),
        "_is_complete": has_result,
        "success": success,
        "num_llm_calls": num_llm_calls,
        "total_input_tokens": total_input,
        "total_output_tokens": total_output,
        "total_latency_ms": total_latency,
        "_num_tool_calls": num_tool_calls,
        "model": model,
        "provider": provider,
        "_diagnostics": diagnostics,
        "error": result.get("error"),
        "failure_reasons": result.get("failure_reasons"),
        "rationale": result.get("rationale"),
        "_explorer_url": explorer_url,
    }


def _render_unified_trace(
    entries: list[dict[str, Any]],
    *,
    daemon_turns: list[dict[str, Any]] | None = None,
) -> str:
    """Render all LLM calls as a single continuous conversation.

    Shows the system prompt once, then for each turn only the new
    messages and the response — no repeated history, no per-turn cards.
    A thin divider with turn number and latency separates each turn.

    When *daemon_turns* is provided, daemon turn-boundary blocks are
    inserted between LLM calls at the matching positions (after the
    LLM call that contains a ``send_message_to_user`` tool use).
    """
    parts: list[str] = []
    daemon_turn_idx = 0  # next daemon turn to render

    for i, entry in enumerate(entries):
        resp = entry.get("response", {})
        usage = resp.get("usage", {})
        request_messages = entry.get("request_messages", [])
        seq = entry.get("seq", i + 1)
        latency_ms = entry.get("latency_ms", 0)
        input_tokens = (
            _int_or_zero(usage.get("input_tokens"))
            + _int_or_zero(usage.get("prompt_tokens"))
            + _int_or_zero(usage.get("cache_read_input_tokens"))
            + _int_or_zero(usage.get("cache_creation_input_tokens"))
        )
        output_tokens = _int_or_zero(usage.get("output_tokens")) + _int_or_zero(
            usage.get("completion_tokens")
        )

        # ── System prompt (first turn only, collapsed by default) ──
        if i == 0:
            system = entry.get("system_prompt", "")
            if system:
                system_text = str(system)
                parts.append('<details class="system-details" data-msg-type="system">')
                parts.append(
                    f'<summary class="msg-label" style="cursor:pointer;padding:10px 14px;background:#fefce8;border-left:3px solid #eab308;border-radius:10px;margin-bottom:14px">System Prompt ({len(system_text):,} chars)</summary>'
                )
                parts.append('<div class="msg msg-system">')
                parts.append(_esc(system_text[:20000]))
                if len(system_text) > 20000:
                    parts.append(f"\n... ({len(system_text)} chars total)")
                parts.append("</div>")
                parts.append("</details>")

        # ── Iteration divider (each LLM call is an iteration within a turn) ──
        parts.append('<div class="turn-divider">')
        parts.append('<span class="turn-info">')
        parts.append(f"<span>Iteration {seq}</span>")
        parts.append(f"<span>{latency_ms}ms</span>")
        parts.append(
            f"<span>{_fmt_tokens(input_tokens)} in / {_fmt_tokens(output_tokens)} out</span>"
        )
        parts.append("</span>")
        parts.append("</div>")

        # ── Inject ENV events at the start of each iteration ──
        if daemon_turns:
            entry_hms = _timestamp_hms(entry.get("timestamp", ""))
            while daemon_turn_idx < len(daemon_turns):
                dt = daemon_turns[daemon_turn_idx]
                if dt.get("type") == "env":
                    dt_hms = _timestamp_hms(dt.get("wall_time", ""))
                    if dt_hms and entry_hms and dt_hms <= entry_hms:
                        parts.append(_render_daemon_turn_block(dt))
                        daemon_turn_idx += 1
                        continue
                break

        # ── New input messages (skip those from previous turn) ──
        prev_msg_count = 0
        if i > 0:
            prev_msg_count = len(entries[i - 1].get("request_messages", []))

        new_messages = request_messages[prev_msg_count:]
        # Skip system message at position 0 — already rendered above.
        if i == 0 and new_messages and new_messages[0].get("role") == "system":
            new_messages = new_messages[1:]
        # The first new message is typically the previous turn's assistant
        # response re-echoed as input — skip it to avoid duplication.
        if i > 0 and new_messages and new_messages[0].get("role") == "assistant":
            new_messages = new_messages[1:]
        for msg in new_messages:
            role = msg.get("role", "unknown")
            parts.extend(_render_trace_blocks(msg.get("content_blocks", []), role=role))
            for tc in msg.get("tool_calls", []):
                parts.append(_render_tool_call(tc))

        # ── Reasoning (OpenAI-style flat field) ──
        reasoning = entry.get("response_reasoning", "")
        if reasoning:
            parts.append('<details class="thinking-details" data-msg-type="thinking">')
            parts.append(
                f'<summary class="thinking-summary">Thinking ({len(reasoning):,} chars)</summary>'
            )
            parts.append(
                f'<div class="thinking-content">{_esc(_truncate(reasoning, 20000))}</div>'
            )
            parts.append("</details>")

        # ── Response ──
        parts.extend(
            _render_trace_blocks(entry.get("response_blocks", []), role="assistant")
        )

        # Response tool_calls (OpenAI format)
        for tc in entry.get("response_tool_calls", []):
            parts.append(_render_tool_call(tc))

        # ── API error (non-2xx status or error fields) ──
        http_status = entry.get("response_status", 0) or entry.get("http_status", 0)
        response_error = entry.get("response_error", {})
        error_title = response_error.get("title", "")
        error_detail = response_error.get("detail", "")
        if (http_status and http_status >= 400) or error_title or error_detail:
            parts.append(
                '<div style="margin:12px 0;padding:14px 18px;background:#fef2f2;'
                'border:1px solid #fca5a5;border-radius:10px;font-size:13px;color:#991b1b">'
            )
            status_str = f" {http_status}" if http_status else ""
            parts.append(
                f'<div style="font-weight:700;margin-bottom:4px">'
                f"API Error{status_str}"
                f"{': ' + _esc(error_title) if error_title else ''}"
                f"</div>"
            )
            if error_detail:
                parts.append(
                    f'<div style="font-family:monospace;font-size:12px;'
                    f'white-space:pre-wrap;word-break:break-word;color:#7f1d1d">'
                    f"{_esc(_truncate(error_detail, 2000))}</div>"
                )
            parts.append("</div>")

        # ── Inject turn boundary blocks on end_turn ──
        if daemon_turns and daemon_turn_idx < len(daemon_turns):
            dt = daemon_turns[daemon_turn_idx]
            if dt.get("type") != "env":
                stop = entry.get("response_stop_reason", "")
                if stop in ("end_turn", "stop"):
                    parts.append(_render_daemon_turn_block(dt))
                    daemon_turn_idx += 1

    # Flush remaining daemon events (ENV events after the last trace entry)
    if daemon_turns:
        while daemon_turn_idx < len(daemon_turns):
            parts.append(_render_daemon_turn_block(daemon_turns[daemon_turn_idx]))
            daemon_turn_idx += 1

    return "\n".join(parts)


def _render_tool_exec(block: dict[str, Any]) -> str:
    """Render a tool_use block as a terminal command line."""
    name = block.get("name", "unknown")
    inp = block.get("input", {})
    if name == "exec" and isinstance(inp, dict) and "command" in inp:
        cmd = inp["command"]
        return f'<div class="tool-exec" data-msg-type="exec">$ {_esc(cmd)}</div>'
    # Non-exec tool calls: show name + args
    parts = [f'<div class="tool-exec" data-msg-type="exec">{_esc(name)}']
    if inp:
        try:
            args_str = json.dumps(inp, indent=2)
        except (TypeError, ValueError):
            args_str = str(inp)
        parts.append(f" {_esc(args_str)}")
    parts.append("</div>")
    return "".join(parts)


def _render_events_table(events: list[dict[str, Any]]) -> str:
    """Render events.jsonl entries as an HTML table."""
    parts = ['<div class="table-wrap"><table>']
    parts.append("<thead><tr>")
    parts.append(
        '<th style="white-space:nowrap">Sim&nbsp;Time</th>'
        '<th style="white-space:nowrap">Wall&nbsp;Time</th>'
        "<th>App</th><th>Function</th><th>Args</th><th>Return</th><th>Write</th>"
    )
    parts.append("</tr></thead><tbody>")

    for ev in events:
        app = ev.get("app", "")
        fn = ev.get("fn", "")
        args = ev.get("args", {})
        ret = ev.get("ret", "")
        write = ev.get("w", False)
        sim_t = ev.get("sim_t", "")
        t = ev.get("t", "")
        is_env = str(ev.get("event_id", "")).startswith("Event-ENV-")

        # Format wall time as human-readable
        wall_str = ""
        if isinstance(t, (int, float)) and t > 0:
            from datetime import datetime, timezone

            wall_str = datetime.fromtimestamp(t, tz=timezone.utc).strftime(
                "%Y-%m-%d %H:%M:%S"
            )

        args_str = json.dumps(args, default=str) if args else ""
        ret_str = str(ret) if ret else ""
        row_style = ' style="background:#fffbeb"' if is_env else ""
        env_badge = (
            ' <span style="color:#b45309;font-weight:600;font-size:10px">ENV</span>'
            if is_env
            else ""
        )

        parts.append(f"<tr{row_style}>")
        parts.append(f'<td style="white-space:nowrap">{_esc(sim_t)}</td>')
        parts.append(f'<td style="white-space:nowrap">{_esc(wall_str)}</td>')
        parts.append(f"<td>{_esc(app)}{env_badge}</td>")
        parts.append(f"<td>{_esc(fn)}</td>")
        parts.append(
            f'<td class="truncated" title="{_esc(args_str)}">{_esc(_truncate(args_str, 80))}</td>'
        )
        parts.append(
            f'<td class="truncated" title="{_esc(ret_str)}">{_esc(_truncate(ret_str, 80))}</td>'
        )
        parts.append(f"<td>{'Y' if write else 'N'}</td>")
        parts.append("</tr>")

    parts.append("</tbody></table></div>")
    return "\n".join(parts)


def _render_judge_result(result: dict[str, Any], *, has_result: bool = True) -> str:
    """Render the judge result section."""
    parts: list[str] = []
    status, badge_cls = _result_status(result, has_result=has_result)

    parts.append(f'<span class="badge {badge_cls}">{status}</span>')

    if not has_result:
        parts.append(
            '<div style="margin-top:12px;padding:12px;background:#eff6ff;'
            "border-left:3px solid #3b82f6;border-radius:8px;font-size:13px;"
            'color:#1e3a8a">'
            "Scenario is still running. Judgment will appear after completion."
            "</div>"
        )
        return "\n".join(parts)

    # Infrastructure error (container/LLM failure)
    if result.get("error"):
        parts.append(
            f'<div style="margin-top:12px;padding:12px;background:#fef2f2;'
            f'border-left:3px solid #ef4444;border-radius:8px;font-size:13px;color:#991b1b">'
            f"<strong>Error:</strong> {_esc(result['error'])}"
            f"</div>"
        )

    # Judge failure reasons
    if result.get("failure_reasons"):
        parts.append('<div style="margin-top:12px">')
        parts.append(
            '<h3 style="font-size:12px;color:#c92a2a;margin-bottom:6px">'
            "Failure Reasons</h3>"
        )
        for reason in result["failure_reasons"]:
            table_html = _format_tool_mismatch(reason)
            if table_html:
                parts.append(
                    f'<div style="margin:4px 0;padding:8px 12px;background:#fef2f2;'
                    f'border-left:3px solid #c92a2a;border-radius:8px;color:#7f1d1d">'
                    f"{table_html}</div>"
                )
            else:
                parts.append(
                    f'<div style="margin:4px 0;padding:8px 12px;background:#fef2f2;'
                    f'border-left:3px solid #c92a2a;border-radius:8px;font-size:12px;color:#7f1d1d">'
                    f"{_esc(reason)}</div>"
                )
        parts.append("</div>")

    if result.get("rationale"):
        parts.append(
            f'<div style="margin-top:12px;padding:16px;background:#f8fafc;'
            f"border:1px solid #e2e8f0;border-radius:12px;font-size:13px;"
            f'white-space:pre-wrap;word-break:break-word;line-height:1.7;color:#334155">'
            f"{_esc(result['rationale'])}</div>"
        )

    # Per-tool counts
    per_tool = result.get("per_tool_counts")
    if per_tool:
        parts.append('<h3 style="margin-top:16px;font-size:14px">Per-Tool Counts</h3>')
        parts.append('<div class="table-wrap"><table>')
        parts.append(
            "<thead><tr><th>App</th><th>Function</th>"
            "<th>Agent</th><th>Oracle</th></tr></thead><tbody>"
        )
        # Group tools by app (part before __)
        by_app: dict[str, list[tuple[str, int, int]]] = defaultdict(list)
        for tool, counts in sorted(per_tool.items()):
            a = counts.get("agent", 0)
            o = counts.get("oracle", 0)
            if "__" in tool:
                app, fn = tool.split("__", 1)
            else:
                app, fn = "", tool
            by_app[app].append((fn, a, o))
        for app in sorted(by_app):
            fns = by_app[app]
            for i, (fn, a, o) in enumerate(fns):
                match = (
                    "style='color:#2b8a3e'"
                    if a == o
                    else "style='color:#c92a2a;font-weight:600'"
                )
                app_cell = (
                    f"<td rowspan='{len(fns)}'>{_esc(app)}</td>" if i == 0 else ""
                )
                parts.append(
                    f"<tr>{app_cell}<td>{_esc(fn)}</td>"
                    f"<td {match}>{a}</td><td {match}>{o}</td></tr>"
                )
        parts.append("</tbody></table></div>")

    # Stats
    parts.append('<div style="margin-top:12px;font-size:13px;color:#868e96">')
    parts.append(f"Agent events: {result.get('num_agent_events', '?')} | ")
    parts.append(f"Oracle events: {result.get('num_oracle_events', '?')}")
    parts.append("</div>")

    return "\n".join(parts)


def _render_judge_log(
    judgments: list[dict[str, Any]],
    raw_events: list[dict[str, Any]] | None = None,
) -> str:
    """Render structured judge log with side-by-side oracle/agent matching.

    Shows each oracle event paired with its matched agent event.
    Unmatched write events are shown prominently as "Unmatched Write"
    entries. Unmatched read events are collapsed under "Ignored".
    """
    # Build a set of write tool signatures from events.jsonl so we can
    # distinguish read vs write among unmatched agent events.
    # events.jsonl has {app, fn, w}, judge log has tool="ClassName__function_name".
    write_tools: set[str] = set()
    if raw_events:
        for ev in raw_events:
            if ev.get("w"):
                app = ev.get("app", "")
                fn = ev.get("fn", "")
                # tool_name format: "ClassName__function_name"
                write_tools.add(f"{app}__{fn}")

    parts: list[str] = []

    for j in judgments:
        turn = j.get("turn", "?")
        success = j.get("success", False)
        failure = j.get("failure_reason")
        id_mapping = j.get("id_mapping") or {}
        match_details = j.get("match_details") or []
        agent_evs = j.get("agent_events", [])
        oracle_evs = j.get("oracle_events", [])

        # Invert mapping: oracle_id → agent_id
        oracle_to_agent: dict[str, str] = {v: k for k, v in id_mapping.items()}
        # Set of matched agent IDs
        matched_agent_ids = set(id_mapping.keys())
        # Index agent events by ID for lookup
        agent_by_id: dict[str, dict[str, Any]] = {
            ae.get("id", ""): ae for ae in agent_evs
        }

        # ── Turn header ──
        verdict_color = "#166534" if success else "#991b1b"
        verdict_bg = "#f0fdf4" if success else "#fef2f2"
        verdict_border = "#bbf7d0" if success else "#fecaca"
        verdict_text = "ACCEPT" if success else "REJECT"
        parts.append(
            f'<div style="margin-bottom:16px;padding:12px 14px;'
            f"background:{verdict_bg};border:1px solid {verdict_border};"
            f'border-radius:10px">'
        )
        parts.append(
            f'<div style="font-weight:700;font-size:13px;color:{verdict_color};'
            f'margin-bottom:8px">Turn {turn}: {verdict_text}</div>'
        )

        if failure:
            table_html = _format_tool_mismatch(failure)
            if table_html:
                parts.append(
                    f'<div style="font-size:12px;color:#991b1b;margin-bottom:10px;'
                    f'padding:6px 10px;background:#fff5f5;border-radius:6px">'
                    f"{table_html}</div>"
                )
            else:
                parts.append(
                    f'<div style="font-size:12px;color:#991b1b;margin-bottom:10px;'
                    f'padding:6px 10px;background:#fff5f5;border-radius:6px">'
                    f"{_esc(failure)}</div>"
                )

        # ── Matched oracle events (side by side) ──
        for oe in oracle_evs:
            oe_id = oe.get("id", "?")
            oe_tool = oe.get("tool", "?")
            oe_args = oe.get("args", oe.get("args_summary", {}))
            oe_args_brief = _fmt_args_brief(oe_args)

            # Find match status from match_details
            md = next(
                (m for m in match_details if m.get("oracle_id") == oe_id),
                None,
            )
            matched = md.get("matched", False) if md else oe_id in oracle_to_agent
            reason = (md.get("reason") or "") if md else ""
            judge_output = (md.get("judge_output") or "") if md else ""
            icon = "&#x2714;" if matched else "&#x2718;"
            icon_color = "#166534" if matched else "#991b1b"
            row_bg = "#f8fdf8" if matched else "#fef8f8"

            # Both matched and unmatched oracle events are clickable
            click_attrs = ""
            extra_style = ""
            args_json = _esc(json.dumps(oe_args, default=str))
            reason_esc = _esc(reason) if reason else ""
            agent_id = oracle_to_agent.get(oe_id, "")
            if matched and agent_id and agent_id in agent_by_id:
                # Matched: clicking opens a direct comparison popup
                ae = agent_by_id[agent_id]
                ae_tool_esc = _esc(ae.get("tool", "?"))
                ae_args_json = _esc(
                    json.dumps(ae.get("args", ae.get("args_summary", {})), default=str)
                )
                judge_output_esc = _esc(judge_output) if judge_output else ""
                click_attrs = (
                    f' onclick="jlOpenMatched(this)" '
                    f'data-oracle-tool="{_esc(oe_tool)}" '
                    f'data-oracle-args="{args_json}" '
                    f'data-agent-tool="{ae_tool_esc}" '
                    f'data-agent-args="{ae_args_json}" '
                    f'data-judge-output="{judge_output_esc}" '
                    f'title="Click to compare agent vs oracle"'
                )
                extra_style = "cursor:pointer;transition:outline .1s;"
            elif not matched:
                judge_output_esc = _esc(judge_output) if judge_output else ""
                click_attrs = (
                    f' class="jl-oracle-ev" data-tool="{_esc(oe_tool)}" '
                    f'data-args="{args_json}" data-turn="{turn}" '
                    f'data-reason="{reason_esc}" '
                    f'data-judge-output="{judge_output_esc}" '
                    f'onclick="jlSelectOracle(this)" '
                    f'title="Click to compare with an agent event"'
                )
                extra_style = "cursor:pointer;transition:outline .1s;"

            parts.append(
                f'<div{click_attrs} style="{extra_style}margin:6px 0;padding:8px 10px;background:{row_bg};'
                f"border-radius:8px;border:1px solid {'#dcfce7' if matched else '#fee2e2'};"
                f'font-size:11px">'
            )
            # Oracle side
            parts.append('<div style="display:flex;align-items:flex-start;gap:6px">')
            if not matched:
                parts.append(
                    f'<input type="radio" name="jl-oracle-{turn}" '
                    f'style="margin-top:2px;flex-shrink:0;pointer-events:none;accent-color:#6366f1">'
                )
            parts.append(
                f'<span style="color:{icon_color};flex-shrink:0">{icon}</span>'
                f'<div style="min-width:0">'
            )
            parts.append(
                f'<div style="font-weight:600;color:#475569;word-break:break-all">'
                f"Oracle: {_esc(oe_tool)}</div>"
            )
            if oe_args_brief:
                parts.append(
                    f'<div style="color:#64748b;font-family:monospace;font-size:10px;'
                    f'margin-top:2px;word-break:break-all">{oe_args_brief}</div>'
                )

            # Agent matched side
            agent_id = oracle_to_agent.get(oe_id)
            if agent_id and agent_id in agent_by_id:
                ae = agent_by_id[agent_id]
                ae_tool = ae.get("tool", "?")
                ae_args = ae.get("args", ae.get("args_summary", {}))
                ae_args_brief = _fmt_args_brief(ae_args)
                parts.append(
                    '<div style="margin-top:6px;padding-top:6px;'
                    'border-top:1px dashed #d1d5db">'
                )
                parts.append(
                    f'<div style="font-weight:600;color:#166534;word-break:break-all">'
                    f"&rarr; Agent: {_esc(ae_tool)}</div>"
                )
                if ae_args_brief:
                    parts.append(
                        f'<div style="color:#64748b;font-family:monospace;'
                        f'font-size:10px;margin-top:2px;word-break:break-all">'
                        f"{ae_args_brief}</div>"
                    )
                parts.append("</div>")
            elif not matched and reason:
                parts.append(
                    f'<div style="margin-top:4px;color:#991b1b;font-size:10px;'
                    f'white-space:pre-wrap;word-break:break-word">'
                    f"<strong>Reason:</strong> {_esc(reason)}</div>"
                )

            # Judge rationale (collapsible, shown for both matched and unmatched)
            if judge_output:
                parts.append(
                    '<details style="margin-top:6px">'
                    '<summary style="font-size:10px;color:#7c3aed;cursor:pointer;'
                    'font-weight:600;user-select:none">'
                    "Judge Rationale</summary>"
                    f'<div style="margin-top:4px;padding:8px 10px;background:#f5f3ff;'
                    f"border-radius:6px;border:1px solid #e9d5ff;"
                    f"font-size:11px;color:#4c1d95;white-space:pre-wrap;"
                    f'word-break:break-word;line-height:1.6">'
                    f"{_esc(judge_output)}</div></details>"
                )

            parts.append("</div></div>")  # close flex item + flex container
            parts.append("</div>")  # close row

        # ── Unmatched agent events ──
        unmatched = [
            ae for ae in agent_evs if ae.get("id", "") not in matched_agent_ids
        ]

        # Filter out ENV events entirely, split rest into writes and reads
        def _is_env_event(ae: dict[str, Any]) -> bool:
            eid = ae.get("id", "")
            return eid.startswith("Event-ENV-") or eid.startswith("Event-USER-")

        agent_unmatched = [ae for ae in unmatched if not _is_env_event(ae)]
        unmatched_writes = [
            ae for ae in agent_unmatched if ae.get("tool", "") in write_tools
        ]
        unmatched_reads = [
            ae for ae in agent_unmatched if ae.get("tool", "") not in write_tools
        ]

        # Collect unmatched oracle events for the comparison UI
        unmatched_oracle = [  # noqa: F841
            oe for oe in oracle_evs if oe.get("id", "") not in oracle_to_agent
        ]

        # Show unmatched writes prominently (these are unexpected mutations)
        if unmatched_writes:
            parts.append(
                '<div style="margin-top:10px;padding-top:8px;'
                'border-top:1px solid #e2e8f0">'
            )
            parts.append(
                '<div style="font-size:10px;font-weight:600;color:#991b1b;'
                'text-transform:uppercase;letter-spacing:0.5px;margin-bottom:6px">'
                f"Unmatched Writes ({len(unmatched_writes)})</div>"
            )
            for ae in unmatched_writes:
                ae_tool = ae.get("tool", "?")
                ae_args = ae.get("args", ae.get("args_summary", {}))
                ae_args_full = _fmt_args_brief(ae_args, max_val=300)
                args_json = _esc(json.dumps(ae_args, default=str))
                parts.append(
                    f'<div class="jl-agent-ev" data-tool="{_esc(ae_tool)}" '
                    f'data-args="{args_json}" data-turn="{turn}" '
                    f'style="margin:4px 0;padding:6px 10px;font-size:11px;'
                    f"color:#991b1b;background:#fef2f2;border-radius:6px;"
                    f"border:1px solid #fecaca;word-break:break-all;"
                    f"cursor:pointer;transition:outline .1s;"
                    f'display:flex;align-items:flex-start;gap:8px" '
                    f'onclick="jlSelectAgent(this)" '
                    f'title="Select to compare with an oracle event">'
                    f'<input type="radio" name="jl-agent-{turn}" '
                    f'style="margin-top:2px;flex-shrink:0;pointer-events:none;accent-color:#6366f1">'
                    f'<div style="min-width:0">'
                    f'<div style="font-weight:600">{_esc(ae_tool)}</div>'
                )
                if ae_args_full:
                    parts.append(
                        f'<div style="color:#64748b;font-family:monospace;'
                        f'font-size:10px;margin-top:2px">'
                        f"{ae_args_full}</div>"
                    )
                parts.append("</div></div>")
            parts.append("</div>")

        # Collapse unmatched reads as "Ignored"
        if unmatched_reads:
            parts.append(
                '<details style="margin-top:10px;padding-top:8px;'
                'border-top:1px solid #e2e8f0">'
            )
            parts.append(
                f'<summary style="font-size:10px;font-weight:600;color:#94a3b8;'
                f"text-transform:uppercase;letter-spacing:0.5px;cursor:pointer;"
                f'list-style:none;display:flex;align-items:center;gap:6px">'
                f'<span style="font-size:8px;transition:transform .2s" '
                f'class="ignore-arrow">&#x25B6;</span>'
                f"Ignored ({len(unmatched_reads)})</summary>"
            )
            for ae in unmatched_reads:
                ae_tool = ae.get("tool", "?")
                ae_args = ae.get("args", ae.get("args_summary", {}))
                ae_args_brief = _fmt_args_brief(ae_args)
                parts.append(
                    f'<div style="margin:2px 0;padding:4px 8px;font-size:10px;'
                    f"color:#94a3b8;background:#f8fafc;border-radius:4px;"
                    f'word-break:break-all">'
                    f"{_esc(ae_tool)}"
                )
                if ae_args_brief:
                    parts.append(
                        f'<span style="font-family:monospace">({ae_args_brief})</span>'
                    )
                parts.append("</div>")
            parts.append("</details>")

        parts.append("</div>")  # close turn card

    return "\n".join(parts)


def _fmt_args_brief(args: dict[str, Any], max_val: int = 60) -> str:
    """Format args dict as a brief comma-separated string for display."""
    if not args:
        return ""
    pieces = []
    for k, v in args.items():
        v_str = str(v) if not isinstance(v, str) else v
        if len(v_str) > max_val:
            v_str = v_str[:max_val] + "..."
        pieces.append(f"{_esc(k)}={_esc(v_str)}")
    return ", ".join(pieces)


# ---------------------------------------------------------------------------
# Index page
# ---------------------------------------------------------------------------


def _generate_index_page(
    output_dir: Path,
    scenario_summaries: list[dict[str, Any]],
    *,
    run_metadata: dict[str, Any] | None = None,
) -> None:
    """Generate index.html with summary stats and scenario table.

    *scenario_summaries* is a list of pre-computed dicts returned by
    ``_generate_trace_page``, each containing ``_scenario_id``, ``success``,
    ``num_llm_calls``, token counts, diagnostics, etc.  This avoids
    re-reading and re-parsing the same per-scenario artifact files.
    """
    all_results = scenario_summaries
    run_metadata = run_metadata or {}
    dataset_counts = _dataset_split_counts(_run_metadata_dataset_path(run_metadata))
    configured_splits = [
        split_name
        for split_name in (run_metadata.get("splits") or [])
        if isinstance(split_name, str) and split_name
    ]

    split_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for summary in all_results:
        split_name = _infer_summary_split(summary, run_metadata)
        summary["_split"] = split_name
        summary["_scenario_label"] = _short_scenario_name(
            str(summary.get("_scenario_id", "")),
            split_name,
        )
        split_groups[split_name].append(summary)

    split_names = sorted(
        {
            split_name
            for split_name in (
                configured_splits
                + list(dataset_counts.keys())
                + list(split_groups.keys())
            )
            if split_name
        },
        key=_split_sort_key,
    )
    split_rows = {
        split_name: {
            "name": split_name,
            "summaries": split_groups.get(split_name, []),
            "stats": _aggregate_summaries(
                split_groups.get(split_name, []),
                expected_total=dataset_counts.get(split_name),
            ),
        }
        for split_name in split_names
    }

    total_tokens_in = sum(
        _int_or_zero(r.get("total_input_tokens")) for r in all_results
    )
    total_tokens_out = sum(
        _int_or_zero(r.get("total_output_tokens")) for r in all_results
    )
    total = len(all_results)
    avg_tokens = (total_tokens_in + total_tokens_out) // total if total > 0 else 0

    model = next((r["model"] for r in all_results if r.get("model")), "") or str(
        run_metadata.get("model") or ""
    )
    provider = next(
        (r["provider"] for r in all_results if r.get("provider")),
        "",
    ) or str(run_metadata.get("provider") or "")

    dir_name = output_dir.name
    expected_total = run_metadata.get("num_scenarios")
    if not isinstance(expected_total, int) and dataset_counts:
        expected_total = sum(dataset_counts.values())
    if isinstance(expected_total, int):
        expected_total = max(expected_total, total)
    overall_stats = _aggregate_summaries(all_results, expected_total=expected_total)

    parts: list[str] = []
    parts.append('<!DOCTYPE html><html><head><meta charset="utf-8">')
    parts.append(f"<title>{_esc(dir_name)} — Gaia2 Benchmark</title>")
    parts.append(f"<style>{_CSS}</style>")
    parts.append('<meta http-equiv="refresh" content="30">')
    parts.append("</head><body>")

    try:
        parent_has_runs = any(
            child.is_dir() and child.name.startswith("run_")
            for child in output_dir.parent.iterdir()
        )
    except OSError:
        parent_has_runs = False

    parts.append('<div class="header shell-header">')
    parts.append('<nav class="breadcrumb">')
    parts.append('<a href="../index.html">Home</a>')
    parts.append('<span class="sep">/</span>')
    if parent_has_runs:
        parts.append(f"<span>{_esc(output_dir.parent.name)}</span>")
        parts.append('<span class="sep">/</span>')
    parts.append(f"<span>{_esc(dir_name)}</span>")
    parts.append("</nav>")
    parts.append('<span class="header-kicker">Run</span>')
    parts.append("</div>")

    parts.append('<div class="container">')
    parts.append('<section class="hero hero-compact">')
    parts.append('<div class="hero-copy">')
    parts.append('<div class="hero-kicker">Run Dashboard</div>')
    parts.append(f"<h1>{_esc(dir_name)}</h1>")
    subtitle = []
    if model:
        subtitle.append(model)
    if provider:
        subtitle.append(provider)
    if run_metadata.get("image"):
        subtitle.append(str(run_metadata["image"]).replace("localhost/", ""))
    parts.append(
        f'<p class="hero-subtitle">{_esc(" • ".join(subtitle) or "Gaia2 run output")}</p>'
    )
    parts.append("</div>")
    parts.append('<div class="hero-stats hero-stats-compact">')
    parts.append(
        f'<div class="hero-stat"><span class="hero-stat-value">{len(split_names) or 1}</span><span class="hero-stat-label">Splits</span></div>'
    )
    parts.append(
        f'<div class="hero-stat"><span class="hero-stat-value">{_progress_meter(overall_stats, variant="compact")}</span><span class="hero-stat-label">Progress</span></div>'
    )
    parts.append(
        f'<div class="hero-stat"><span class="hero-stat-value">{_status_chip_for_stats(overall_stats)}</span><span class="hero-stat-label">Status</span></div>'
    )
    parts.append(
        f'<div class="hero-stat"><span class="hero-stat-value">{_esc(_pass_rate_text(overall_stats))}</span><span class="hero-stat-label">Pass Rate</span></div>'
    )
    parts.append("</div>")
    parts.append("</section>")

    parts.append('<div class="summary summary-grid">')
    if isinstance(expected_total, int) and expected_total > 0:
        parts.append(
            f'<div class="stat"><div class="value">{expected_total}</div><div class="label">Expected</div></div>'
        )
        parts.append(
            f'<div class="stat"><div class="value">{overall_stats["completed_count"]}</div><div class="label">Completed</div></div>'
        )
        if overall_stats["running_count"]:
            parts.append(
                f'<div class="stat"><div class="value" style="color:#2563eb">{overall_stats["running_count"]}</div><div class="label">Running</div></div>'
            )
        if overall_stats["pending_count"]:
            parts.append(
                f'<div class="stat"><div class="value" style="color:#64748b">{overall_stats["pending_count"]}</div><div class="label">Pending</div></div>'
            )
    else:
        parts.append(
            f'<div class="stat"><div class="value">{total}</div><div class="label">Scenarios</div></div>'
        )
    parts.append(
        f'<div class="stat"><div class="value" style="color:#2b8a3e">{overall_stats["pass_count"]}</div><div class="label">Passed</div></div>'
    )
    parts.append(
        f'<div class="stat"><div class="value" style="color:#c92a2a">{overall_stats["fail_count"]}</div><div class="label">Failed</div></div>'
    )
    if overall_stats["error_count"]:
        parts.append(
            f'<div class="stat"><div class="value" style="color:#e8590c">{overall_stats["error_count"]}</div><div class="label">Errors</div></div>'
        )
    elif overall_stats["running_count"] and not isinstance(expected_total, int):
        parts.append(
            f'<div class="stat"><div class="value" style="color:#2563eb">{overall_stats["running_count"]}</div><div class="label">Running</div></div>'
        )
    parts.append(
        f'<div class="stat"><div class="value">{_esc(_pass_rate_text(overall_stats))}</div><div class="label">Pass Rate</div></div>'
    )
    parts.append(
        f'<div class="stat"><div class="value">{_fmt_tokens(total_tokens_in + total_tokens_out)}</div><div class="label">Total Tokens</div></div>'
    )
    parts.append(
        f'<div class="stat"><div class="value">{_fmt_tokens(avg_tokens)}</div><div class="label">Avg Tokens</div></div>'
    )
    parts.append("</div>")

    if split_names:
        parts.append('<section class="panel">')
        parts.append(
            '<div class="panel-header"><h2 class="section-title">Split Breakdown</h2></div>'
        )
        parts.append('<div class="split-card-grid">')
        for split_name in split_names:
            split_stats = split_rows[split_name]["stats"]
            split_rate_text = _pass_rate_text(split_stats)
            split_href = f"../splits/{_esc(split_name)}.html" if parent_has_runs else ""
            if split_href:
                parts.append(f'<a class="split-card" href="{split_href}">')
            else:
                parts.append('<div class="split-card">')
            parts.append(f'<div class="split-card-title">{_esc(split_name)}</div>')
            parts.append(
                f'<div class="split-card-status-row">{_status_chip_for_stats(split_stats)}</div>'
            )
            parts.append(
                f'<div class="split-card-value{" split-card-value-empty" if split_rate_text == "—" else ""}">{_esc(split_rate_text)}</div>'
            )
            parts.append('<div class="split-card-meta">')
            parts.append(
                f"<span>{split_stats['pass_count']}/{split_stats['completed_count'] or split_stats['total'] or 0} passed</span>"
            )
            if isinstance(split_stats.get("expected_total"), int):
                parts.append(
                    f"<span>{split_stats['completed_count']}/{split_stats['expected_total']} complete</span>"
                )
            parts.append("</div>")
            parts.append("</div>" if not split_href else "</a>")
        parts.append("</div>")
        parts.append("</section>")

    # Filters
    parts.append('<section class="panel">')
    parts.append(
        '<div class="panel-header"><h2 class="section-title">Scenarios</h2></div>'
    )
    parts.append('<div class="filters">')
    parts.append('<select id="statusFilter" onchange="filterTable()">')
    parts.append('<option value="all">All</option>')
    parts.append('<option value="running">Running</option>')
    parts.append('<option value="pass">Pass</option>')
    parts.append('<option value="fail">Fail</option>')
    parts.append('<option value="error">Error</option>')
    parts.append("</select>")
    if split_names:
        parts.append('<select id="splitFilter" onchange="filterTable()">')
        parts.append('<option value="all">All splits</option>')
        for split_name in split_names:
            parts.append(
                f'<option value="{_esc(split_name.lower())}">{_esc(split_name)}</option>'
            )
        parts.append("</select>")
    else:
        parts.append('<input type="hidden" id="splitFilter" value="all">')
    parts.append(
        '<input type="text" id="searchInput" placeholder="Search scenario ID..." oninput="filterTable()">'
    )
    parts.append("</div>")

    # Scenario table
    parts.append('<div class="table-wrap"><table id="scenarioTable">')
    parts.append("<thead><tr>")
    parts.append(
        '<th onclick="sortTable(0)">Status <span class="sort-arrow"></span></th>'
    )
    parts.append(
        '<th onclick="sortTable(1)">Scenario ID <span class="sort-arrow"></span></th>'
    )
    parts.append(
        '<th onclick="sortTable(2)">Split <span class="sort-arrow"></span></th>'
    )
    parts.append(
        '<th onclick="sortTable(3)">LLM Calls <span class="sort-arrow"></span></th>'
    )
    parts.append(
        '<th onclick="sortTable(4)">Tokens (in) <span class="sort-arrow"></span></th>'
    )
    parts.append(
        '<th onclick="sortTable(5)">Tokens (out) <span class="sort-arrow"></span></th>'
    )
    parts.append(
        '<th onclick="sortTable(6)">Latency <span class="sort-arrow"></span></th>'
    )
    parts.append(
        '<th onclick="sortTable(7)">Tool Calls <span class="sort-arrow"></span></th>'
    )
    parts.append(
        '<th onclick="sortTable(8)">Errors <span class="sort-arrow"></span></th>'
    )
    parts.append("<th>Info</th>")
    parts.append("</tr></thead><tbody>")

    for r in all_results:
        sid = r.get("_scenario_id", r.get("scenario_id", ""))
        split_name = str(r.get("_split") or "")
        scenario_label = str(r.get("_scenario_label") or sid)
        status = _summary_status(r)
        badge_cls = f"badge-{status}"
        status_label = status.upper()

        llm_calls = r.get("num_llm_calls", 0)
        tokens_in = _int_or_zero(r.get("total_input_tokens"))
        tokens_out = _int_or_zero(r.get("total_output_tokens"))
        latency = _int_or_zero(r.get("total_latency_ms"))
        tool_calls = _int_or_zero(r.get("_num_tool_calls"))

        # Diagnostics
        diag = r.get("_diagnostics", {})
        err_count = diag.get("error_count", 0)

        # Info column: prioritised — error > failure_reasons > log errors > rationale
        info = ""
        if r.get("error"):
            info = r["error"]
        elif r.get("failure_reasons"):
            info = "; ".join(r["failure_reasons"])
        elif diag.get("error_summary"):
            info = diag["error_summary"]
        elif r.get("rationale"):
            info = r["rationale"]
        elif status == "running":
            info = "Scenario is still running."

        parts.append(
            f'<tr data-status="{status}" data-sid="{_esc(str(sid).lower())}" data-split="{_esc(split_name.lower())}">'
        )
        parts.append(f'<td><span class="badge {badge_cls}">{status_label}</span></td>')
        explorer = r.get("_explorer_url", "")
        explorer_link = (
            f' <a href="{_esc(explorer)}" target="_blank" title="Open in Gaia2 Explorer">&#x1F50D;</a>'
            if explorer
            else ""
        )
        parts.append(
            f'<td><a href="{_esc(sid)}/trace.html">{_esc(scenario_label)}</a>{explorer_link}</td>'
        )
        parts.append(f"<td>{_esc(split_name or '—')}</td>")
        parts.append(f"<td>{llm_calls}</td>")
        parts.append(f"<td>{tokens_in:,}</td>")
        parts.append(f"<td>{tokens_out:,}</td>")
        parts.append(f"<td>{latency / 1000:.1f}s</td>")
        parts.append(f"<td>{tool_calls}</td>")
        if err_count > 0:
            parts.append(f'<td style="color:#ef4444;font-weight:700">{err_count}</td>')
        else:
            parts.append('<td style="color:#64748b">0</td>')
        info_short = _truncate(info, 120)
        parts.append(
            f'<td class="truncated" title="{_esc(info)}">{_esc(info_short)}</td>'
        )
        parts.append("</tr>")

    parts.append("</tbody></table></div>")
    parts.append("</section>")
    parts.append("</div>")  # container

    # JS
    parts.append("<script>")
    parts.append(_INDEX_JS)
    parts.append("</script>")

    parts.append("</body></html>")

    index_path = output_dir / "index.html"
    index_path.write_text("\n".join(parts))


# ---------------------------------------------------------------------------
# JavaScript
# ---------------------------------------------------------------------------

_TRACE_JS = """\
function toggleCard(header) {
  header.classList.toggle('open');
}
// Switch raw log tabs
function switchLogTab(btn, idx) {
  var tabs = document.querySelectorAll('.log-tab-content');
  var btns = btn.parentElement.querySelectorAll('button');
  tabs.forEach(function(t, i) { t.style.display = i === idx ? 'block' : 'none'; });
  btns.forEach(function(b, i) {
    b.style.background = i === idx ? '#334155' : '#1e293b';
    b.style.color = i === idx ? '#f1f5f9' : '#94a3b8';
  });
}
// Keyboard navigation: left/right arrows
document.addEventListener('keydown', function(e) {
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
  var arrow = null;
  if (e.key === 'ArrowLeft') arrow = document.querySelector('.nav-arrow-left');
  if (e.key === 'ArrowRight') arrow = document.querySelector('.nav-arrow-right');
  if (arrow) { e.preventDefault(); arrow.click(); }
});
// Pair tool-exec with tool-output 1:1 across turn dividers
(function() {
  var body = document.querySelector('.card-body');
  if (!body) return;
  // Snapshot all tool-execs and tool-outputs in DOM order
  var allExecs = Array.from(body.querySelectorAll('.tool-exec'));
  var allOutputs = Array.from(body.querySelectorAll('.tool-output'));
  // Match by index: exec[0]->output[0], exec[1]->output[1], etc.
  var count = Math.min(allExecs.length, allOutputs.length);
  for (var i = 0; i < count; i++) {
    var exec = allExecs[i];
    var output = allOutputs[i];
    // Skip if already paired
    if (exec.parentElement.classList.contains('tool-pair')) continue;
    var pair = document.createElement('div');
    pair.className = 'tool-pair';
    exec.parentNode.insertBefore(pair, exec);
    pair.appendChild(exec);
    // Move output next to exec (remove from its current position)
    pair.appendChild(output);
  }
})();
// Judge log: compare unmatched agent vs oracle events (modal)
var _jlSelectedAgent = null;
var _jlSelectedOracle = null;
function _jlClearSelection(cls) {
  document.querySelectorAll('.' + cls + '.jl-selected').forEach(function(el) {
    el.classList.remove('jl-selected');
    el.style.outline = '';
    var r = el.querySelector('input[type=radio]');
    if (r) r.checked = false;
  });
}
function jlSelectAgent(el) {
  if (_jlSelectedAgent === el) {
    el.classList.remove('jl-selected');
    el.style.outline = '';
    var r = el.querySelector('input[type=radio]');
    if (r) r.checked = false;
    _jlSelectedAgent = null;
  } else {
    _jlClearSelection('jl-agent-ev');
    el.classList.add('jl-selected');
    el.style.outline = '2px solid #6366f1';
    var r = el.querySelector('input[type=radio]');
    if (r) r.checked = true;
    _jlSelectedAgent = el;
  }
  _jlTryCompare();
}
function jlSelectOracle(el) {
  if (_jlSelectedOracle === el) {
    el.classList.remove('jl-selected');
    el.style.outline = '';
    var r = el.querySelector('input[type=radio]');
    if (r) r.checked = false;
    _jlSelectedOracle = null;
  } else {
    _jlClearSelection('jl-oracle-ev');
    el.classList.add('jl-selected');
    el.style.outline = '2px solid #6366f1';
    var r = el.querySelector('input[type=radio]');
    if (r) r.checked = true;
    _jlSelectedOracle = el;
  }
  _jlTryCompare();
}
function jlCloseCompare(e) {
  if (e && e.target !== document.getElementById('jl-compare-overlay')) return;
  document.getElementById('jl-compare-overlay').style.display = 'none';
  _jlClearSelection('jl-agent-ev');
  _jlClearSelection('jl-oracle-ev');
  _jlSelectedAgent = null;
  _jlSelectedOracle = null;
}
document.addEventListener('keydown', function(e) {
  if (e.key === 'Escape') {
    var ov = document.getElementById('jl-compare-overlay');
    if (ov && ov.style.display === 'flex') jlCloseCompare(null);
  }
});
function _jlTryCompare() {
  var overlay = document.getElementById('jl-compare-overlay');
  var modal = document.getElementById('jl-compare-modal');
  if (!_jlSelectedAgent || !_jlSelectedOracle) { overlay.style.display = 'none'; return; }
  var aTool = _jlSelectedAgent.getAttribute('data-tool');
  var oTool = _jlSelectedOracle.getAttribute('data-tool');
  var aArgs, oArgs;
  try { aArgs = JSON.parse(_jlSelectedAgent.getAttribute('data-args') || '{}'); } catch(e) { aArgs = {}; }
  try { oArgs = JSON.parse(_jlSelectedOracle.getAttribute('data-args') || '{}'); } catch(e) { oArgs = {}; }
  var html = '<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:16px">'
    + '<div style="font-weight:700;font-size:15px;color:#0f172a">Event Comparison</div>'
    + '<button onclick="jlCloseCompare(null)" style="background:none;border:none;font-size:20px;color:#94a3b8;cursor:pointer;padding:4px 8px;line-height:1">&times;</button></div>';
  if (aTool !== oTool) {
    html += '<div style="color:#991b1b;font-size:12px;margin-bottom:12px;padding:8px 12px;background:#fef2f2;border-radius:8px">Different tools: <b>' + esc(aTool) + '</b> vs <b>' + esc(oTool) + '</b></div>';
    html += '<div style="display:flex;gap:16px"><div style="flex:1;min-width:0">';
    html += '<div style="font-weight:600;color:#991b1b;font-size:11px;text-transform:uppercase;letter-spacing:.5px;margin-bottom:8px">Agent: ' + esc(aTool) + '</div>';
    for (var k in aArgs) html += _jlParamRow(k, String(aArgs[k]));
    html += '</div><div style="flex:1;min-width:0">';
    html += '<div style="font-weight:600;color:#475569;font-size:11px;text-transform:uppercase;letter-spacing:.5px;margin-bottom:8px">Oracle: ' + esc(oTool) + '</div>';
    for (var k in oArgs) html += _jlParamRow(k, String(oArgs[k]));
    html += '</div></div>';
  } else {
    html += '<div style="font-size:13px;color:#475569;margin-bottom:12px"><b>' + esc(aTool) + '</b></div>';
    var allKeys = {};
    for (var k in aArgs) allKeys[k] = true;
    for (var k in oArgs) allKeys[k] = true;
    html += '<table style="width:100%;border-collapse:collapse;font-size:12px">';
    html += '<tr style="border-bottom:2px solid #e2e8f0"><th style="text-align:left;padding:6px 10px;color:#64748b;font-size:10px;text-transform:uppercase;letter-spacing:.5px;width:20%">Param</th><th style="text-align:left;padding:6px 10px;color:#991b1b;font-size:10px;text-transform:uppercase;letter-spacing:.5px;width:40%">Agent</th><th style="text-align:left;padding:6px 10px;color:#475569;font-size:10px;text-transform:uppercase;letter-spacing:.5px;width:40%">Oracle</th></tr>';
    for (var k in allKeys) {
      var av = k in aArgs ? String(aArgs[k]) : '';
      var ov = k in oArgs ? String(oArgs[k]) : '';
      var same = av === ov;
      var bg = same ? '#f0fdf4' : '#fef2f2';
      html += '<tr style="background:' + bg + ';border-bottom:1px solid #f1f5f9">';
      html += '<td style="padding:6px 10px;color:#64748b;font-weight:600;vertical-align:top;font-family:monospace;font-size:11px">' + esc(k) + '</td>';
      html += '<td style="padding:6px 10px;word-break:break-word;white-space:pre-wrap;font-family:monospace;font-size:11px;color:' + (same ? '#166534' : '#991b1b') + '">' + esc(av || '(missing)') + '</td>';
      html += '<td style="padding:6px 10px;word-break:break-word;white-space:pre-wrap;font-family:monospace;font-size:11px;color:#334155">' + esc(ov || '(missing)') + '</td>';
      html += '</tr>';
    }
    html += '</table>';
  }
  var oJudgeOutput = _jlSelectedOracle.getAttribute('data-judge-output') || '';
  if (oJudgeOutput) {
    html += '<div style="margin-top:14px;padding:10px 12px;background:#f5f3ff;border-radius:8px;border:1px solid #e9d5ff">'
      + '<div style="font-size:10px;font-weight:700;color:#7c3aed;text-transform:uppercase;letter-spacing:.5px;margin-bottom:6px">Judge Rationale</div>'
      + '<div style="font-size:12px;color:#4c1d95;white-space:pre-wrap;word-break:break-word;line-height:1.6">' + esc(oJudgeOutput) + '</div></div>';
  }
  var oReason = _jlSelectedOracle.getAttribute('data-reason') || '';
  if (oReason) {
    html += '<div style="margin-top:12px;padding:10px 14px;background:#fef2f2;border-left:3px solid #ef4444;border-radius:8px;font-size:12px;color:#991b1b;white-space:pre-wrap;word-break:break-word"><strong>Rejection reason:</strong> ' + esc(oReason) + '</div>';
  }
  html += '<div style="margin-top:12px;font-size:10px;color:#94a3b8;text-align:center">Press Esc or click outside to close</div>';
  modal.innerHTML = html;
  overlay.style.display = 'flex';
}
function _jlParamRow(k, v) {
  return '<div style="margin-bottom:6px"><div style="font-size:10px;color:#64748b;font-weight:600;font-family:monospace">' + esc(k) + '</div><div style="font-size:11px;font-family:monospace;white-space:pre-wrap;word-break:break-word;color:#334155;padding:4px 8px;background:#f8fafc;border-radius:4px;margin-top:2px">' + esc(v) + '</div></div>';
}
function _openCompare(aTool, aArgs, oTool, oArgs, isMatched, judgeOutput) {
  var overlay = document.getElementById('jl-compare-overlay');
  var modal = document.getElementById('jl-compare-modal');
  if (!overlay || !modal) return;
  var titleColor = isMatched ? '#166534' : '#0f172a';
  var titleBadge = isMatched ? ' <span style="font-size:11px;padding:2px 8px;background:#dcfce7;color:#166534;border-radius:6px;font-weight:600;margin-left:8px">MATCHED</span>' : '';
  var html = '<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:16px">'
    + '<div style="font-weight:700;font-size:15px;color:' + titleColor + '">Event Comparison' + titleBadge + '</div>'
    + '<button onclick="jlCloseCompare(null)" style="background:none;border:none;font-size:20px;color:#94a3b8;cursor:pointer;padding:4px 8px;line-height:1">&times;</button></div>';
  if (aTool !== oTool) {
    html += '<div style="color:#991b1b;font-size:12px;margin-bottom:12px;padding:8px 12px;background:#fef2f2;border-radius:8px">Different tools: <b>' + esc(aTool) + '</b> vs <b>' + esc(oTool) + '</b></div>';
  }
  html += '<div style="font-size:13px;color:#475569;margin-bottom:12px"><b>' + esc(oTool) + '</b></div>';
  var allKeys = {};
  for (var k in aArgs) allKeys[k] = true;
  for (var k in oArgs) allKeys[k] = true;
  html += '<table style="width:100%;border-collapse:collapse;font-size:12px">';
  html += '<tr style="border-bottom:2px solid #e2e8f0"><th style="text-align:left;padding:6px 10px;color:#64748b;font-size:10px;text-transform:uppercase;letter-spacing:.5px;width:20%">Param</th><th style="text-align:left;padding:6px 10px;color:#166534;font-size:10px;text-transform:uppercase;letter-spacing:.5px;width:40%">Agent</th><th style="text-align:left;padding:6px 10px;color:#475569;font-size:10px;text-transform:uppercase;letter-spacing:.5px;width:40%">Oracle</th></tr>';
  for (var k in allKeys) {
    var av = k in aArgs ? String(aArgs[k]) : '';
    var ov = k in oArgs ? String(oArgs[k]) : '';
    var same = av === ov;
    // For matched events, differences are expected (semantic match) — use neutral colors
    var bg = same ? '#f0fdf4' : (isMatched ? '#fffbeb' : '#fef2f2');
    var diffColor = isMatched ? '#92400e' : '#991b1b';
    html += '<tr style="background:' + bg + ';border-bottom:1px solid #f1f5f9">';
    html += '<td style="padding:6px 10px;color:#64748b;font-weight:600;vertical-align:top;font-family:monospace;font-size:11px">' + esc(k) + '</td>';
    html += '<td style="padding:6px 10px;word-break:break-word;white-space:pre-wrap;font-family:monospace;font-size:11px;color:' + (same ? '#166534' : diffColor) + '">' + esc(av || '(missing)') + '</td>';
    html += '<td style="padding:6px 10px;word-break:break-word;white-space:pre-wrap;font-family:monospace;font-size:11px;color:#334155">' + esc(ov || '(missing)') + '</td>';
    html += '</tr>';
  }
  html += '</table>';
  if (judgeOutput) {
    html += '<div style="margin-top:14px;padding:10px 12px;background:#f5f3ff;border-radius:8px;border:1px solid #e9d5ff">'
      + '<div style="font-size:10px;font-weight:700;color:#7c3aed;text-transform:uppercase;letter-spacing:.5px;margin-bottom:6px">Judge Rationale</div>'
      + '<div style="font-size:12px;color:#4c1d95;white-space:pre-wrap;word-break:break-word;line-height:1.6">' + esc(judgeOutput) + '</div></div>';
  }
  html += '<div style="margin-top:12px;font-size:10px;color:#94a3b8;text-align:center">Press Esc or click outside to close</div>';
  modal.innerHTML = html;
  overlay.style.display = 'flex';
}
function jlOpenMatched(el) {
  var oTool = (el.getAttribute('data-oracle-tool') || '?').replace(/__/g, '.');
  var aTool = (el.getAttribute('data-agent-tool') || '?').replace(/__/g, '.');
  var oArgs, aArgs;
  try { oArgs = JSON.parse(el.getAttribute('data-oracle-args') || '{}'); } catch(e) { oArgs = {}; }
  try { aArgs = JSON.parse(el.getAttribute('data-agent-args') || '{}'); } catch(e) { aArgs = {}; }
  var judgeOutput = el.getAttribute('data-judge-output') || '';
  _openCompare(aTool, aArgs, oTool, oArgs, true, judgeOutput);
}
function esc(s) { var d = document.createElement('div'); d.textContent = s; return d.innerHTML; }
"""

_INDEX_JS = """\
// Auto-regenerate: poll for new scenarios every 30s
(function() {
  var el = document.createElement('span');
  el.style.cssText = 'color:#64748b;font-size:11px;margin-left:12px';
  el.id = 'autoRefresh';
  el.textContent = 'Auto-refresh: on';
  var hdr = document.querySelector('.header');
  if (hdr) hdr.appendChild(el);
  setInterval(function() {
    fetch(location.href).then(function(r){return r.text()}).then(function(html) {
      var m = html.match(/<tbody>([\\s\\S]*?)<\\/tbody>/);
      if (!m) return;
      var tmp = document.createElement('table');
      tmp.innerHTML = '<tbody>' + m[1] + '</tbody>';
      var newHtml = tmp.querySelector('tbody').innerHTML;
      var current = document.querySelector('#scenarioTable tbody');
      if (current && newHtml !== current.innerHTML) {
        location.reload();
      }
    }).catch(function(){});
  }, 30000);
})();

function filterTable() {
  var status = document.getElementById('statusFilter').value;
  var split = document.getElementById('splitFilter').value;
  var search = document.getElementById('searchInput').value.toLowerCase();
  var rows = document.querySelectorAll('#scenarioTable tbody tr');
  rows.forEach(function(row) {
    var rowStatus = row.getAttribute('data-status');
    var rowSid = row.getAttribute('data-sid');
    var rowSplit = row.getAttribute('data-split') || '';
    var showStatus = (status === 'all' || rowStatus === status);
    var showSplit = (split === 'all' || rowSplit === split);
    var showSearch = (!search || rowSid.indexOf(search) !== -1);
    row.style.display = (showStatus && showSplit && showSearch) ? '' : 'none';
  });
}

var sortDir = {};
function sortTable(col) {
  var table = document.getElementById('scenarioTable');
  var tbody = table.querySelector('tbody');
  var rows = Array.from(tbody.querySelectorAll('tr'));
  var dir = sortDir[col] === 'asc' ? 'desc' : 'asc';
  sortDir[col] = dir;

  rows.sort(function(a, b) {
    var aVal = a.cells[col].textContent.trim();
    var bVal = b.cells[col].textContent.trim();
    // Try numeric sort
    var aNum = parseFloat(aVal.replace(/[,%s]/g, ''));
    var bNum = parseFloat(bVal.replace(/[,%s]/g, ''));
    if (!isNaN(aNum) && !isNaN(bNum)) {
      return dir === 'asc' ? aNum - bNum : bNum - aNum;
    }
    return dir === 'asc' ? aVal.localeCompare(bVal) : bVal.localeCompare(aVal);
  });

  rows.forEach(function(row) { tbody.appendChild(row); });
}
"""

_SPLIT_JS = """\
function filterSplitTable() {
  var search = document.getElementById('scenarioSearch').value.toLowerCase();
  document.querySelectorAll('#comparisonTable tbody tr').forEach(function(row) {
    var scenarioId = row.getAttribute('data-sid') || '';
    row.style.display = (!search || scenarioId.indexOf(search) !== -1) ? '' : 'none';
  });
}
"""

_RUNS_JS = """\
// Auto-regenerate: poll for changes every 30s
(function() {
  var count = document.querySelectorAll('#scenarioTable tbody tr').length;
  var el = document.createElement('span');
  el.style.cssText = 'color:#64748b;font-size:11px;margin-left:12px';
  el.textContent = 'Auto-refresh: on';
  var hdr = document.querySelector('.header');
  if (hdr) hdr.appendChild(el);
  setInterval(function() {
    fetch(location.href).then(function(r){return r.text()}).then(function(html) {
      var m = html.match(/<tbody>([\\s\\S]*?)<\\/tbody>/);
      if (!m) return;
      var tmp = document.createElement('table');
      tmp.innerHTML = '<tbody>' + m[1] + '</tbody>';
      var newHtml = tmp.querySelector('tbody').innerHTML;
      var curHtml = document.querySelector('#scenarioTable tbody').innerHTML;
      if (newHtml !== curHtml) {
        location.reload();
      }
    }).catch(function(){});
  }, 30000);
})();

function filterTable() {
  var search = document.getElementById('searchInput').value.toLowerCase();
  var rows = document.querySelectorAll('#scenarioTable tbody tr');
  rows.forEach(function(row) {
    var rowSid = row.getAttribute('data-sid') || '';
    // Also search in all cell text
    var text = row.textContent.toLowerCase();
    row.style.display = (!search || text.indexOf(search) !== -1) ? '' : 'none';
  });
}

var sortDir = {};
function sortTable(col) {
  var table = document.getElementById('scenarioTable');
  var tbody = table.querySelector('tbody');
  var rows = Array.from(tbody.querySelectorAll('tr'));
  var dir = sortDir[col] === 'asc' ? 'desc' : 'asc';
  sortDir[col] = dir;

  rows.sort(function(a, b) {
    var aVal = a.cells[col].textContent.trim();
    var bVal = b.cells[col].textContent.trim();
    var aNum = parseFloat(aVal.replace(/[,%s]/g, ''));
    var bNum = parseFloat(bVal.replace(/[,%s]/g, ''));
    if (!isNaN(aNum) && !isNaN(bNum)) {
      return dir === 'asc' ? aNum - bNum : bNum - aNum;
    }
    return dir === 'asc' ? aVal.localeCompare(bVal) : bVal.localeCompare(aVal);
  });

  rows.forEach(function(row) { tbody.appendChild(row); });
}

var compSortDir = {};
function sortCompTable(col) {
  var table = document.getElementById('comparisonTable');
  if (!table) return;
  var tbody = table.querySelector('tbody');
  var rows = Array.from(tbody.querySelectorAll('tr'));
  var dir = compSortDir[col] === 'asc' ? 'desc' : 'asc';
  compSortDir[col] = dir;

  rows.sort(function(a, b) {
    var aVal = a.cells[col].textContent.trim();
    var bVal = b.cells[col].textContent.trim();
    return dir === 'asc' ? aVal.localeCompare(bVal) : bVal.localeCompare(aVal);
  });

  rows.forEach(function(row) { tbody.appendChild(row); });
}
"""


# ---------------------------------------------------------------------------
# Diagnostics — parse errors from container logs
# ---------------------------------------------------------------------------


def _parse_diagnostics(scenario_dir: Path) -> dict[str, Any]:
    """Parse errors from all available log sources in a scenario directory.

    Returns a dict with per-source error lists, a total count, and a
    short one-liner summary suitable for the index table.
    """
    diag: dict[str, Any] = {
        "openclaw_errors": [],
        "daemon_errors": [],
        "entrypoint_errors": [],
    }

    # OpenClaw log — JSON lines, ERROR entries have logLevelName="ERROR"
    oc_log = _load_text(scenario_dir / "openclaw.log")
    if oc_log:
        for i, line in enumerate(oc_log.splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            meta = obj.get("_meta", {})
            level = meta.get("logLevelName", obj.get("logLevelName", ""))
            if level not in ("ERROR", "WARN"):
                continue
            # Error text lives in field "0" (primary), "1", or "2" (OpenClaw log)
            msg = obj.get("0", "")
            extra = obj.get("1", "")
            if extra and isinstance(extra, str):
                msg = f"{msg} {extra}" if msg else extra
            elif extra and isinstance(extra, dict):
                msg = extra.get("message", msg or str(extra))
            if not msg:
                nested = obj.get("2")
                if isinstance(nested, str):
                    msg = nested
                elif isinstance(nested, dict):
                    msg = nested.get("message", str(nested))
            if level == "WARN" and not any(
                kw in str(msg).lower()
                for kw in ("fail", "crash", "timeout", "refused", "error")
            ):
                continue
            diag["openclaw_errors"].append(
                {"level": level, "message": str(msg)[:500], "line": i}
            )

    # Daemon log — plain text
    daemon_log = _load_text(scenario_dir / "daemon.log")
    if daemon_log:
        for i, line in enumerate(daemon_log.splitlines(), 1):
            low = line.lower()
            if any(
                kw in low
                for kw in ("error", "critical", "exception", "traceback", "panic")
            ):
                diag["daemon_errors"].append({"message": line.strip()[:500], "line": i})

    # Entrypoint log — plain text
    entry_log = _load_text(scenario_dir / "entrypoint.log")
    if entry_log:
        for i, line in enumerate(entry_log.splitlines(), 1):
            if "ERROR" in line.upper():
                diag["entrypoint_errors"].append(
                    {"message": line.strip()[:500], "line": i}
                )

    total = (
        len(diag["openclaw_errors"])
        + len(diag["daemon_errors"])
        + len(diag["entrypoint_errors"])
    )
    diag["has_errors"] = total > 0
    diag["error_count"] = total

    # One-liner for the index table — first error + count
    if total > 0:
        first = None
        for src in ("openclaw_errors", "daemon_errors", "entrypoint_errors"):
            if diag[src]:
                first = diag[src][0]["message"]
                break
        summary = _truncate(first or "", 120)
        if total > 1:
            summary += f" (+{total - 1} more)"
        diag["error_summary"] = summary
    else:
        diag["error_summary"] = ""

    return diag


def _parse_daemon_turns(scenario_dir: Path) -> list[dict[str, Any]]:
    """Parse turn boundary events from the daemon log.

    Returns a list of turn dicts, each containing:
    - turn: int (turn index that just completed)
    - condition: str or None (condition that triggered)
    - judge: str (e.g. "ACCEPT" or "REJECT turn 0: ...")
    - env_actions: list of str (CLI commands executed)
    - notifications: list of str (messages sent to agent)
    - lines: list of str (all raw log lines for this turn block)
    """
    daemon_log = _load_text(scenario_dir / "eventd.log")
    if not daemon_log:
        daemon_log = _load_text(scenario_dir / "daemon.log")
    if not daemon_log:
        return []

    import re

    turns: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None

    # Load ENV event results from events.jsonl for full CLI output
    env_results: dict[str, str] = {}
    events_jsonl = _load_text(scenario_dir / "events.jsonl")
    if events_jsonl:
        for eline in events_jsonl.splitlines():
            try:
                edata = json.loads(eline)
                eid = edata.get("event_id", "")
                if eid.startswith("Event-ENV-"):
                    env_results[eid] = json.dumps(edata, indent=2)
            except (json.JSONDecodeError, TypeError):
                pass

    # Also collect standalone ENV events as independent timeline entries.
    # Each has type="env", a timestamp, and the action details.
    current_env: dict[str, Any] | None = None

    for line in daemon_log.splitlines():
        # ENV event fired by _advance_time()
        if "Time advance:" in line and "firing" in line:
            m = re.search(r"sim=[\d.]+ \(([^)]+)\)", line)
            ts = m.group(1) if m else ""
            # Extract wall-clock timestamp for ordering
            wall_m = re.match(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", line)
            wall_ts = wall_m.group(1) if wall_m else ""
            current_env = {
                "type": "env",
                "sim_time": ts,
                "wall_time": wall_ts,
                "env_summaries": [],
                "env_actions": [],
                "cmd": None,
                "result": None,
                "notification": None,
            }
            turns.append(current_env)
        elif re.search(r"ENV\[\d+\]:", line) and current_env is not None:
            m = re.search(r"ENV\[\d+\]:\s*(.+)", line)
            if m:
                current_env["env_summaries"].append(m.group(1).strip())
            # Extract event ID for result lookup
            eid_m = re.search(r"\[(Event-ENV-[^\]]+)\]", line)
            if eid_m:
                current_env["event_id"] = eid_m.group(1)
                # Look up full result from events.jsonl
                if eid_m.group(1) in env_results:
                    current_env["result"] = env_results[eid_m.group(1)]
        elif "  cmd:" in line and current_env is not None:
            current_env["cmd"] = line.split("cmd:", 1)[-1].strip()
        elif "  result:" in line and current_env is not None:
            # Only use daemon log result if events.jsonl didn't provide one
            if not current_env.get("result"):
                current_env["result"] = line.split("result:", 1)[-1].strip()
        elif "Sending" in line and "notification" in line and current_env is not None:
            current_env["notification"] = (
                line.split("INFO ", 1)[-1].strip() if "INFO " in line else line.strip()
            )
            current_env = None  # ENV block complete

        # Turn boundary start
        if "TURN BOUNDARY DETECTED" in line:
            current_env = None  # End any pending ENV block
            m = re.search(r"turn (\d+)", line)
            turn_num = int(m.group(1)) if m else len(turns) + 1
            current = {
                "turn": turn_num,
                "condition": None,
                "judge": "",
                "env_actions": [],
                "env_summaries": [],
                "env_timestamps": [],
                "notifications": [],
                "lines": [line],
            }
            turns.append(current)
            continue

        if current is None:
            continue

        current["lines"].append(line)

        if "Condition " in line and "PASSED" in line:
            m = re.search(r"Condition (\S+) PASSED", line)
            current["condition"] = m.group(1) if m else "unknown"
        elif "JUDGE ACCEPT" in line:
            current["judge"] = line.split("JUDGE ", 1)[-1] if "JUDGE " in line else line
        elif "JUDGE REJECT" in line:
            # Capture all rejection details (there may be multiple)
            reject_line = line.split("JUDGE ", 1)[-1] if "JUDGE " in line else line
            if not current["judge"] or "ACCEPT" in current["judge"]:
                current["judge"] = reject_line
            else:
                current.setdefault("judge_details", []).append(reject_line)
        elif "JUDGE REJECTED" in line and "stopping" in line:
            current["stopped"] = True
        elif "Soft checker" in line:
            current.setdefault("soft_checker_details", []).append(
                line.split("INFO ", 1)[-1].strip() if "INFO " in line else line.strip()
            )
        elif "LLM checker" in line:
            current.setdefault("soft_checker_details", []).append(
                line.split("INFO ", 1)[-1].strip() if "INFO " in line else line.strip()
            )
        elif "CLI failed" in line:
            current.setdefault("cli_errors", []).append(
                line.split("ERROR ", 1)[-1].strip()
                if "ERROR " in line
                else line.strip()
            )
        elif "Notification:" in line:
            current["notifications"].append(line.split("Notification:", 1)[-1].strip())

    return turns


def _render_env_event_block(env: dict[str, Any]) -> str:
    """Render a standalone ENV event using CSS classes (amber themed)."""
    sim_time = env.get("sim_time", "")
    cmd = env.get("cmd", "")
    result = (env.get("result") or "").strip()

    ts_html = f'<span class="env-badge">{_esc(sim_time)}</span>' if sim_time else ""
    cmd_text = cmd if cmd else "(unknown ENV action)"

    # Right side: pretty-printed JSON output
    if result and result not in ("{", "{}"):
        try:
            parsed = json.loads(result)
            pretty = json.dumps(parsed, indent=2, ensure_ascii=False)
        except (json.JSONDecodeError, TypeError):
            pretty = result
        right_content = f"<pre style='margin:0'>{_esc(pretty)}</pre>"
    else:
        right_content = '<span style="color:#9ca3af;font-style:italic">ok</span>'

    return (
        f'<div class="env-pair">'
        f'<div class="env-exec">'
        f'<div class="env-label">\u23f1 ENV{ts_html}</div>'
        f"$ {_esc(_truncate(cmd_text, 300))}"
        f"</div>"
        f'<div class="env-output">{right_content}</div>'
        f"</div>"
    )


def _render_daemon_turn_block(turn: dict[str, Any]) -> str:
    """Render a daemon turn event block for inline display in the LLM trace.

    Handles two types:
    - type="env": standalone ENV event fired by _advance_time()
    - regular turn dict: turn boundary with judge verdict
    """
    # Standalone ENV event
    if turn.get("type") == "env":
        return _render_env_event_block(turn)

    parts: list[str] = []
    parts.append(
        '<div style="margin:20px 0;padding:14px 18px;background:#eff6ff;'
        'border:1px solid #93c5fd;border-radius:10px;font-size:12px;color:#1e3a5f">'
    )

    # Header
    parts.append(
        f'<div style="display:flex;align-items:center;gap:10px;margin-bottom:8px">'
        f'<span style="font-weight:700;color:#1d4ed8;font-size:13px">'
        f"Turn {turn['turn']} Boundary</span>"
    )
    if turn.get("condition"):
        parts.append(
            f'<code style="background:#dbeafe;padding:2px 8px;border-radius:4px;'
            f'font-size:10px;color:#1e40af">{_esc(turn["condition"])}</code>'
        )
    parts.append("</div>")

    # Judge verdict
    if turn.get("judge"):
        judge_text = turn["judge"]
        if "ACCEPT" in judge_text:
            color, bg = "#166534", "#f0fdf4"
            icon = "&#x2714; PASS"
        else:
            color, bg = "#991b1b", "#fef2f2"
            icon = "&#x2718; FAIL"
            if turn.get("stopped"):
                icon += " (stopped)"
        parts.append(
            f'<div style="margin:6px 0;padding:8px 12px;background:{bg};'
            f'border-radius:6px;color:{color};font-weight:600;font-size:12px">'
            f"{icon}: {_esc(judge_text)}</div>"
        )
        # Additional rejection details
        for detail in turn.get("judge_details", []):
            parts.append(
                f'<div style="margin:2px 0 2px 16px;padding:4px 10px;background:{bg};'
                f'border-radius:4px;color:{color};font-size:11px">'
                f"&#x2718; {_esc(detail)}</div>"
            )

    # Soft checker details
    for detail in turn.get("soft_checker_details", []):
        parts.append(
            f'<div style="margin:3px 0;padding:6px 10px;background:#fffbeb;'
            f"border-left:3px solid #f59e0b;border-radius:6px;"
            f'font-size:11px;color:#78350f;word-break:break-all">'
            f"{_esc(_truncate(detail, 500))}</div>"
        )

    # ENV events are now shown as standalone timeline entries
    # (rendered by _render_env_event_block), not inside turn blocks.

    # CLI errors
    for err in turn.get("cli_errors", []):
        parts.append(
            f'<div style="margin:3px 0;padding:6px 10px;background:#fef2f2;'
            f"border-left:3px solid #ef4444;border-radius:6px;"
            f"font-family:monospace;font-size:11px;color:#991b1b;"
            f'word-break:break-all">{_esc(_truncate(err, 300))}</div>'
        )

    for notif in turn.get("notifications", []):
        parts.append(
            f'<div style="margin:2px 0;color:#6b7280;font-size:11px">'
            f"Notify agent: {_esc(notif)}</div>"
        )

    parts.append("</div>")
    return "\n".join(parts)


def _render_raw_logs_tabs(scenario_dir: Path) -> str:
    """Render tabbed raw log viewer for all available log files."""
    log_files = [
        ("Daemon", "eventd.log"),
        ("Daemon Judge", "daemon_judgments.jsonl"),
        ("Runner Judge", "judgments.jsonl"),
        ("Events", "events.jsonl"),
        ("OpenClaw", "openclaw.log"),
        ("Daemon Status", "daemon_status.json"),
        ("Result", "result.json"),
    ]

    # Collect available logs
    available: list[tuple[str, str, str]] = []  # (label, fname, content)
    for label, fname in log_files:
        content = _load_text(scenario_dir / fname)
        if content.strip():
            available.append((label, fname, content))

    if not available:
        return ""

    parts: list[str] = []
    parts.append('<div class="log-tabs">')

    # Tab buttons
    parts.append(
        '<div style="display:flex;gap:2px;margin-bottom:0;'
        'border-bottom:2px solid #334155">'
    )
    for i, (label, _, _content) in enumerate(available):
        active = (
            "background:#334155;color:#f1f5f9;border-bottom:2px solid #60a5fa"
            if i == 0
            else "background:#1e293b;color:#64748b;border-bottom:2px solid transparent"
        )
        parts.append(
            f'<button onclick="switchLogTab(this,{i})" '
            f'style="padding:8px 16px;border:none;border-radius:8px 8px 0 0;'
            f'font-size:13px;font-weight:600;cursor:pointer;{active}">'
            f"{_esc(label)}</button>"
        )
    parts.append("</div>")

    # Tab content
    for i, (label, fname, content) in enumerate(available):
        display = "block" if i == 0 else "none"
        if fname.endswith(".jsonl"):
            rendered = _render_jsonl_log(content, fname)
        elif fname == "eventd.log":
            rendered = _render_daemon_log(content)
        elif fname.endswith(".json"):
            try:
                import json as _json

                pretty = _json.dumps(_json.loads(content), indent=2)
            except Exception:
                pretty = content
            rendered = (
                f'<pre style="margin:0;white-space:pre-wrap;'
                f'word-break:break-all">{_esc(pretty)}</pre>'
            )
        else:
            rendered = (
                f'<pre style="margin:0;white-space:pre-wrap;'
                f'word-break:break-all">{_esc(content)}</pre>'
            )
        parts.append(
            f'<div class="log-tab-content" style="display:{display};'
            f"max-height:600px;overflow:auto;background:#0f172a;color:#e2e8f0;"
            f"padding:16px;border-radius:0 0 8px 8px;font-size:13px;"
            f'line-height:1.6">{rendered}</div>'
        )

    parts.append("</div>")
    return "\n".join(parts)


def _render_daemon_log(content: str) -> str:
    """Render daemon log with colored log levels and structured lines."""
    parts: list[str] = []
    for line in content.splitlines():
        if not line.strip():
            continue
        # Color by log level
        if " ERROR " in line:
            color, bg = "#fca5a5", "rgba(239,68,68,0.1)"
            weight = "600"
        elif " WARNING " in line:
            color, bg = "#fcd34d", "rgba(245,158,11,0.08)"
            weight = "600"
        elif "JUDGE ACCEPT" in line:
            color, bg = "#86efac", "rgba(34,197,94,0.1)"
            weight = "600"
        elif "JUDGE REJECT" in line or "REJECTED" in line:
            color, bg = "#fca5a5", "rgba(239,68,68,0.1)"
            weight = "600"
        elif "TURN BOUNDARY" in line or "PROCESS TURN" in line:
            color, bg = "#93c5fd", "rgba(59,130,246,0.1)"
            weight = "700"
        elif "Tick:" in line or "ENV[" in line:
            color, bg = "#a5b4fc", "rgba(99,102,241,0.08)"
            weight = "500"
        elif "Soft checker" in line or "LLM checker" in line:
            color, bg = "#fcd34d", "rgba(245,158,11,0.08)"
            weight = "500"
        elif "CLI failed" in line:
            color, bg = "#fca5a5", "rgba(239,68,68,0.08)"
            weight = "500"
        elif " DEBUG " in line:
            color, bg = "#64748b", "transparent"
            weight = "400"
        else:
            color, bg = "#cbd5e1", "transparent"
            weight = "400"

        parts.append(
            f'<div style="padding:2px 8px;color:{color};background:{bg};'
            f"font-weight:{weight};border-radius:3px;margin:1px 0;"
            f"font-family:monospace;font-size:12px;white-space:pre-wrap;"
            f'word-break:break-all">{_esc(line)}</div>'
        )
    return "\n".join(parts)


def _render_jsonl_log(content: str, fname: str) -> str:
    """Render JSONL files as formatted, collapsible entries."""
    parts: list[str] = []
    entries = _parse_jsonl_text(content)

    if not entries:
        return f'<pre style="margin:0">{_esc(content)}</pre>'

    is_judgment = "judgment" in fname.lower() or "judge" in fname.lower()

    is_ad = is_judgment and _is_ad_judgments(entries)

    for i, entry in enumerate(entries):
        if is_judgment and is_ad:
            # Agent-driven judge: per-action records
            success = entry.get("success")
            matched_count = entry.get("matched_count", "?")
            total_oracle = entry.get("total_oracle", "?")
            ae = entry.get("agent_event") or {}
            oe = entry.get("oracle_event")
            if success is True:
                badge = (
                    '<span style="color:#22c55e;font-weight:700">&#x2714; MATCH</span>'
                )
                border_color = "#22c55e"
            elif success is False:
                badge = (
                    '<span style="color:#ef4444;font-weight:700">&#x2718; REJECT</span>'
                )
                border_color = "#ef4444"
            else:
                badge = '<span style="color:#94a3b8">???</span>'
                border_color = "#64748b"

            header = (
                f"Action {i + 1}: {badge}"
                f' <span style="color:#94a3b8;font-size:11px">'
                f"({matched_count}/{total_oracle})</span>"
            )
            ae_tool = ae.get("tool", "?")
            details_parts: list[str] = []
            details_parts.append(
                f'<div style="padding:3px 0;color:#a5b4fc">'
                f'Agent: <span style="color:#e2e8f0">{_esc(ae_tool)}</span></div>'
            )
            if oe:
                oe_tool = oe.get("tool", "?")
                details_parts.append(
                    f'<div style="padding:3px 0;color:#93c5fd">'
                    f'&rarr; Oracle: <span style="color:#e2e8f0">'
                    f"{_esc(oe_tool)}</span></div>"
                )
            reason = entry.get("failure_reason", "")
            if reason:
                header += (
                    f'<div style="margin-top:4px;color:#fca5a5;font-size:12px">'
                    f"{_esc(_truncate(reason, 200))}</div>"
                )
            judge_output = entry.get("judge_output")
            if judge_output:
                details_parts.append(
                    f'<div style="margin-top:6px;padding:6px 8px;background:'
                    f"rgba(139,92,246,0.1);border-radius:4px;font-size:11px;"
                    f'color:#c4b5fd;white-space:pre-wrap;word-break:break-word">'
                    f"{_esc(judge_output)}</div>"
                )
            details_html = "\n".join(details_parts)
        elif is_judgment:
            # Turn-based judgment entries: show turn, success, and details
            turn = entry.get("turn", "?")
            success = _get_success(entry)
            if success is True:
                badge = (
                    '<span style="color:#22c55e;font-weight:700">&#x2714; PASS</span>'
                )
                border_color = "#22c55e"
            elif success is False:
                badge = (
                    '<span style="color:#ef4444;font-weight:700">&#x2718; FAIL</span>'
                )
                border_color = "#ef4444"
            else:
                badge = '<span style="color:#94a3b8">???</span>'
                border_color = "#64748b"

            header = f"Turn {turn}: {badge}"
            reason = entry.get("failure_reason", "")
            if reason:
                header += (
                    f'<div style="margin-top:4px;color:#fca5a5;font-size:12px">'
                    f"{_esc(_truncate(reason, 200))}</div>"
                )

            # Show match details
            details_parts: list[str] = []
            for md in entry.get("match_details") or []:
                matched = md.get("matched", False)
                icon = "&#x2714;" if matched else "&#x2718;"
                icon_color = "#86efac" if matched else "#fca5a5"
                tool = md.get("oracle_tool", "?")
                mr = md.get("reason", "")
                details_parts.append(
                    f'<div style="padding:3px 0;color:{icon_color}">'
                    f'{icon} <span style="color:#e2e8f0">{_esc(tool)}</span>'
                )
                if mr:
                    details_parts.append(
                        f'<div style="color:#94a3b8;font-size:11px;'
                        f'margin-left:20px;word-break:break-all">'
                        f"{_esc(_truncate(mr, 200))}</div>"
                    )
                details_parts.append("</div>")

            # Oracle/Agent event summaries
            for label, key, color in [
                ("Oracle", "oracle_events", "#93c5fd"),
                ("Agent", "agent_events", "#a5b4fc"),
            ]:
                evts = entry.get(key, [])
                if evts:
                    details_parts.append(
                        f'<div style="margin-top:8px;font-weight:600;'
                        f'color:{color};font-size:11px">{label} Events '
                        f"({len(evts)})</div>"
                    )
                    for ev in evts:
                        tool = ev.get("tool", "?")
                        args = ev.get("args_summary", {})
                        args_brief = _fmt_args_brief(args, max_val=80)
                        details_parts.append(
                            f'<div style="padding:2px 0;font-size:11px">'
                            f'<span style="color:#e2e8f0">{_esc(tool)}</span>'
                        )
                        if args_brief:
                            details_parts.append(
                                f'<div style="color:#64748b;font-size:10px;'
                                f'margin-left:12px;word-break:break-all">'
                                f"{args_brief}</div>"
                            )
                        details_parts.append("</div>")

            details_html = "\n".join(details_parts)
        else:
            # Generic JSONL: format as indented JSON
            header = f"Entry {i + 1}"
            border_color = "#334155"
            formatted = json.dumps(entry, indent=2, ensure_ascii=False, default=str)
            details_html = (
                f'<pre style="margin:0;color:#cbd5e1;font-size:12px;'
                f'white-space:pre-wrap;word-break:break-all">'
                f"{_esc(formatted)}</pre>"
            )

        entry_id = f"log-entry-{fname}-{i}"  # noqa: F841
        parts.append(
            f'<details style="margin:6px 0;border-left:3px solid {border_color};'
            f'border-radius:4px">'
            f'<summary style="padding:8px 12px;cursor:pointer;font-size:13px;'
            f'color:#e2e8f0;font-weight:500">{header}</summary>'
            f'<div style="padding:8px 12px 12px 12px">{details_html}</div>'
            f"</details>"
        )

    return "\n".join(parts)


def _render_diagnostics(diagnostics: dict[str, Any]) -> str:
    """Render error diagnostics for the trace page sidebar."""
    if not diagnostics.get("has_errors"):
        return '<div style="color:#94a3b8;font-size:13px">No errors detected in logs.</div>'

    parts: list[str] = []
    _SOURCES = [
        ("OpenClaw Errors", "openclaw_errors", "#ef4444"),
        ("Daemon Errors", "daemon_errors", "#f97316"),
        ("Entrypoint Errors", "entrypoint_errors", "#eab308"),
    ]
    for title, key, color in _SOURCES:
        errors = diagnostics.get(key, [])
        if not errors:
            continue
        parts.append(
            f'<h3 style="margin-top:12px;font-size:12px;color:{color}">'
            f"{_esc(title)} ({len(errors)})</h3>"
        )
        for err in errors[:10]:
            level = err.get("level", "")
            badge = ""
            if level == "ERROR":
                badge = '<span class="badge badge-fail" style="font-size:8px;padding:2px 8px;margin-right:6px">ERROR</span>'
            elif level == "WARN":
                badge = '<span class="badge badge-error" style="font-size:8px;padding:2px 8px;margin-right:6px">WARN</span>'
            parts.append(
                f'<div style="margin:4px 0;padding:8px 12px;background:#fff5f5;'
                f"border-left:3px solid {color};border-radius:8px;font-size:12px;"
                f'color:#1e293b;word-break:break-word">'
                f"{badge}"
                f'<span style="color:#94a3b8;font-size:10px">line {err.get("line", "?")}</span> '
                f"{_esc(err['message'])}"
                f"</div>"
            )
        if len(errors) > 10:
            parts.append(
                f'<div style="font-size:11px;color:#94a3b8">'
                f"... and {len(errors) - 10} more</div>"
            )
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_TOOL_MISMATCH_RE = re.compile(
    r"Tool count mismatch: agent=\{(.*?)\}, oracle=\{(.*?)\}$"
)


def _format_tool_mismatch(reason: str) -> str | None:
    """If *reason* contains a tool-count mismatch, return an HTML table.

    Returns ``None`` when the string doesn't match, so the caller can
    fall back to the plain-text rendering.
    """
    m = _TOOL_MISMATCH_RE.search(reason)
    if not m:
        return None

    def _parse_counts(s: str) -> dict[str, int]:
        if not s.strip():
            return {}
        counts: dict[str, int] = {}
        for pair in s.split(", "):
            k, _, v = pair.rpartition(": ")
            k = k.strip().strip("'\"")
            counts[k] = int(v)
        return counts

    agent_counts = _parse_counts(m.group(1))
    oracle_counts = _parse_counts(m.group(2))
    all_tools = sorted(set(agent_counts) | set(oracle_counts))

    # Group by app (part before __)
    by_app: dict[str, list[tuple[str, int, int]]] = defaultdict(list)
    for tool in all_tools:
        a = agent_counts.get(tool, 0)
        o = oracle_counts.get(tool, 0)
        if "__" in tool:
            app, fn = tool.split("__", 1)
        else:
            app, fn = "", tool
        by_app[app].append((fn, a, o))

    # Build prefix (e.g. "Turn 1: ") from text before the match
    prefix = _esc(reason[: m.start()].strip())
    if prefix:
        prefix = f"<span>{prefix}</span><br>"

    rows: list[str] = []
    td = "padding:2px 6px"
    for app in sorted(by_app):
        fns = by_app[app]
        for i, (fn, a, o) in enumerate(fns):
            color = "#2b8a3e" if a == o else "#c92a2a"
            bold = "" if a == o else ";font-weight:600"
            app_cell = (
                f"<td rowspan='{len(fns)}' style='{td};font-weight:600;"
                f"border-right:1px solid #e5e7eb'>{_esc(app)}</td>"
                if i == 0
                else ""
            )
            rows.append(
                f"<tr>{app_cell}<td style='{td}'>{_esc(fn)}</td>"
                f"<td style='{td};text-align:right;color:{color}{bold}'>{a}</td>"
                f"<td style='{td};text-align:right;color:{color}{bold}'>{o}</td></tr>"
            )

    th = "text-align:left;padding:2px 6px;border-bottom:1px solid #cbd5e1;font-size:11px;color:#64748b"
    return (
        f"{prefix}"
        f"<table style='margin-top:4px;border-collapse:collapse;font-size:12px;line-height:1.3'>"
        f"<thead><tr><th style='{th}'>App</th><th style='{th}'>Function</th>"
        f"<th style='{th};text-align:right'>Agent</th>"
        f"<th style='{th};text-align:right'>Oracle</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def _esc(s: str) -> str:
    """HTML-escape a string."""
    return html.escape(str(s)) if s else ""


def _truncate(s: str, max_len: int) -> str:
    """Truncate string with ellipsis."""
    if not s:
        return ""
    if len(s) <= max_len:
        return s
    return s[:max_len] + "..."


def _fmt_tokens(n: int) -> str:
    """Format token count (e.g. 1234 -> '1.2k', 1234567 -> '1.2M')."""
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}k"
    return str(n)


def _get_success(res: dict[str, Any]) -> bool | None:
    """Extract success from a result dict.

    Supports both top-level ``success`` (runner format) and
    ``judgment.success`` (daemon/eventd format).
    """
    if "success" in res:
        return res["success"]
    return res.get("judgment", {}).get("success")


def _load_json(path: Path) -> dict[str, Any]:
    """Load a JSON file, returning empty dict on failure."""
    try:
        if path.exists():
            with open(path) as f:
                return json.load(f)
    except Exception:
        pass
    return {}


def _load_text(path: Path) -> str:
    """Load a text file, returning empty string on failure."""
    try:
        if path.exists():
            return path.read_text()
    except Exception:
        pass
    return ""


# ---------------------------------------------------------------------------
# DAG visualization CSS + JS
# ---------------------------------------------------------------------------

_DAG_CSS = """\
.dag-section { margin-bottom: 20px; }
.dag-section > summary { list-style: none; }
.dag-section > summary::-webkit-details-marker { display: none; }
.dag-section > summary::before { content: "▸ "; }
.dag-section[open] > summary::before { content: "▾ "; }
#dag-container {
  width: 100%; min-height: 200px; max-height: 500px;
  border: 1px solid #e2e8f0; border-radius: 12px;
  background: #fafbfc; overflow: hidden; position: relative;
}
#dag-container svg { width: 100%; height: 100%; }
.dag-legend {
  display: flex; gap: 16px; padding: 8px 0; font-size: 12px; color: #64748b;
  flex-wrap: wrap;
}
.dag-legend-item { display: flex; align-items: center; gap: 4px; }
.dag-swatch {
  width: 14px; height: 14px; border-radius: 3px;
  border: 2px solid; display: inline-block;
}
.dag-node { cursor: pointer; }
.dag-node rect { rx: 6; ry: 6; stroke-width: 2; }
.dag-node text { font-size: 11px; font-family: system-ui, sans-serif; }
.dag-node:hover rect { filter: brightness(0.95); }
.dag-tooltip {
  position: absolute; background: #1e293b; color: #f1f5f9;
  padding: 8px 12px; border-radius: 8px; font-size: 11px;
  font-family: monospace; max-width: 400px; pointer-events: none;
  z-index: 50; white-space: pre-wrap; word-break: break-all;
  box-shadow: 0 4px 12px rgba(0,0,0,.3);
}
.dag-badge {
  font-size: 9px; font-weight: 700; fill: #fff;
}
"""

_DAG_JS = """\
(function() {
  var data = window.__DAG_DATA;
  if (!data || !data.oracle_dag || !data.oracle_dag.length) return;

  var oracleEvents = data.oracle_dag;
  var idMapping = data.id_mapping || {};   // agent_id -> oracle_id
  var matchDetails = data.match_details || [];
  var agentEvents = data.agent_events || [];

  // Build reverse mapping: oracle_id -> agent_id
  var oracleToAgent = {};
  for (var aId in idMapping) {
    oracleToAgent[idMapping[aId]] = aId;
  }

  // Build matched oracle IDs set
  var matchedOracleIds = {};
  matchDetails.forEach(function(md) {
    if (md.matched) matchedOracleIds[md.oracle_id] = true;
  });

  // Build node map
  var nodeMap = {};
  oracleEvents.forEach(function(ev) {
    nodeMap[ev.event_id] = ev;
  });

  // Compute levels via longest path from roots
  var levels = {};
  function computeLevel(id) {
    if (levels[id] !== undefined) return levels[id];
    var ev = nodeMap[id];
    if (!ev || !ev.dependencies || ev.dependencies.length === 0) {
      levels[id] = 0;
      return 0;
    }
    var maxDep = 0;
    ev.dependencies.forEach(function(depId) {
      if (nodeMap[depId]) {
        maxDep = Math.max(maxDep, computeLevel(depId) + 1);
      }
    });
    levels[id] = maxDep;
    return maxDep;
  }
  oracleEvents.forEach(function(ev) { computeLevel(ev.event_id); });

  // Group by level
  var byLevel = {};
  var maxLevel = 0;
  oracleEvents.forEach(function(ev) {
    var lv = levels[ev.event_id] || 0;
    if (!byLevel[lv]) byLevel[lv] = [];
    byLevel[lv].push(ev);
    maxLevel = Math.max(maxLevel, lv);
  });

  // Layout params
  var nodeW = 180, nodeH = 44, padX = 60, padY = 24;
  var marginLeft = 30, marginTop = 30;

  // Assign positions
  var positions = {};
  for (var lv = 0; lv <= maxLevel; lv++) {
    var nodes = byLevel[lv] || [];
    nodes.forEach(function(ev, idx) {
      positions[ev.event_id] = {
        x: marginLeft + lv * (nodeW + padX),
        y: marginTop + idx * (nodeH + padY)
      };
    });
  }

  // Calculate SVG size
  var maxY = 0;
  oracleEvents.forEach(function(ev) {
    var pos = positions[ev.event_id];
    if (pos) maxY = Math.max(maxY, pos.y + nodeH);
  });
  var svgW = marginLeft + (maxLevel + 1) * (nodeW + padX);
  var svgH = maxY + marginTop;

  // Determine container height (capped)
  var container = document.getElementById("dag-container");
  var displayH = Math.min(svgH + 20, 500);
  container.style.height = displayH + "px";

  // Create SVG with zoom
  var svg = d3.select("#dag-container").append("svg")
    .attr("viewBox", "0 0 " + svgW + " " + svgH);

  var g = svg.append("g");

  // Zoom behavior
  var zoom = d3.zoom()
    .scaleExtent([0.3, 3])
    .on("zoom", function(event) { g.attr("transform", event.transform); });
  svg.call(zoom);

  // Arrow marker
  g.append("defs").append("marker")
    .attr("id", "dag-arrow")
    .attr("viewBox", "0 0 10 10")
    .attr("refX", 10).attr("refY", 5)
    .attr("markerWidth", 8).attr("markerHeight", 8)
    .attr("orient", "auto")
    .append("path").attr("d", "M 0 0 L 10 5 L 0 10 z").attr("fill", "#94a3b8");

  // Draw edges
  oracleEvents.forEach(function(ev) {
    (ev.dependencies || []).forEach(function(depId) {
      if (!positions[depId] || !positions[ev.event_id]) return;
      var src = positions[depId];
      var tgt = positions[ev.event_id];
      var x1 = src.x + nodeW, y1 = src.y + nodeH / 2;
      var x2 = tgt.x, y2 = tgt.y + nodeH / 2;
      var mx = (x1 + x2) / 2;
      g.append("path")
        .attr("d", "M" + x1 + "," + y1 + " C" + mx + "," + y1 + " " + mx + "," + y2 + " " + x2 + "," + y2)
        .attr("fill", "none").attr("stroke", "#94a3b8").attr("stroke-width", 1.5)
        .attr("marker-end", "url(#dag-arrow)");
    });
  });

  // Color scheme
  function nodeColor(ev) {
    if (ev.event_type === "USER") return { fill: "#bbdefb", stroke: "#00ABFF", dash: "" };
    if (ev.event_type === "ENV") return { fill: "#b2dfdb", stroke: "#009688", dash: "" };
    var isMatched = matchedOracleIds[ev.event_id];
    // If no match_details at all but id_mapping exists, check mapping
    if (matchDetails.length === 0 && Object.keys(oracleToAgent).length > 0) {
      isMatched = !!oracleToAgent[ev.event_id];
    }
    if (isMatched) return { fill: "#c8e6c9", stroke: "#4CAF50", dash: "" };
    return { fill: "#ffcdd2", stroke: "#EF5350", dash: "4,3" };
  }

  // Tooltip div
  var tooltip = d3.select("#dag-container").append("div")
    .attr("class", "dag-tooltip").style("display", "none");

  // Draw nodes
  oracleEvents.forEach(function(ev) {
    var pos = positions[ev.event_id];
    if (!pos) return;
    var col = nodeColor(ev);
    var label = (ev.app ? ev.app + "." : "") + (ev.function || "?");
    if (label.length > 24) label = label.slice(0, 22) + "..";

    var node = g.append("g")
      .attr("class", "dag-node")
      .attr("transform", "translate(" + pos.x + "," + pos.y + ")");

    node.append("rect")
      .attr("width", nodeW).attr("height", nodeH)
      .attr("fill", col.fill).attr("stroke", col.stroke)
      .attr("stroke-dasharray", col.dash);

    node.append("text")
      .attr("x", 8).attr("y", 17)
      .attr("fill", "#334155").attr("font-weight", "600")
      .text(label);

    // Type label
    node.append("text")
      .attr("x", 8).attr("y", 34)
      .attr("fill", "#64748b").attr("font-size", "10px")
      .text(ev.event_type);

    // Agent execution order badge
    var agentId = oracleToAgent[ev.event_id];
    if (agentId) {
      // Find the index in agent_events
      var idx = -1;
      for (var i = 0; i < agentEvents.length; i++) {
        if (agentEvents[i].id === agentId) { idx = i; break; }
      }
      if (idx >= 0) {
        node.append("circle")
          .attr("cx", nodeW - 14).attr("cy", 14).attr("r", 10)
          .attr("fill", "#7c3aed");
        node.append("text")
          .attr("class", "dag-badge")
          .attr("x", nodeW - 14).attr("y", 17.5)
          .attr("text-anchor", "middle")
          .text(idx + 1);
      }
    }

    // Hover tooltip
    node.on("mouseover", function(event) {
      var argsStr = Object.keys(ev.args || {}).map(function(k) {
        return k + ": " + ev.args[k];
      }).join("\\n");
      var fullLabel = (ev.app ? ev.app + "." : "") + (ev.function || "?");
      tooltip.style("display", "block")
        .html("<b>" + fullLabel + "</b>\\n" + (argsStr || "(no args)"));
    }).on("mousemove", function(event) {
      var rect = container.getBoundingClientRect();
      tooltip.style("left", (event.clientX - rect.left + 12) + "px")
        .style("top", (event.clientY - rect.top + 12) + "px");
    }).on("mouseout", function() {
      tooltip.style("display", "none");
    });

    // Click: open compare popup for matched nodes
    var matchedAgentId = oracleToAgent[ev.event_id];
    if (matchedAgentId) {
      node.style("cursor", "pointer");
      node.on("click", (function(oracleEv, agentId) {
        return function() {
          // Find agent event data
          var agentEv = null;
          for (var i = 0; i < agentEvents.length; i++) {
            if (agentEvents[i].id === agentId) { agentEv = agentEvents[i]; break; }
          }
          if (!agentEv) return;
          var oTool = (oracleEv.app + "." + oracleEv.function).replace(/__/g, ".");
          var aTool = (agentEv.tool || "").replace(/__/g, ".");
          var oArgs = oracleEv.args || {};
          var aArgs = agentEv.args || agentEv.args_summary || {};
          // Parse stringified args if needed
          if (typeof aArgs === "string") try { aArgs = JSON.parse(aArgs); } catch(e) {}
          // Find judge_output from matchDetails
          var judgeOutput = "";
          for (var mi = 0; mi < matchDetails.length; mi++) {
            if (matchDetails[mi].oracle_id === oracleEv.event_id) {
              judgeOutput = matchDetails[mi].judge_output || "";
              break;
            }
          }
          _openCompare(aTool, aArgs, oTool, oArgs, true, judgeOutput);
        };
      })(ev, matchedAgentId));
    }
  });

  // Agent-only actions (not matched to any oracle event)
  var unmatchedAgent = agentEvents.filter(function(ae) {
    return !idMapping[ae.id];
  });
  if (unmatchedAgent.length > 0) {
    var extraY = svgH;
    var extraG = g.append("g").attr("transform", "translate(" + marginLeft + "," + extraY + ")");
    extraG.append("text").attr("x", 0).attr("y", -4)
      .attr("fill", "#64748b").attr("font-size", "11px").attr("font-weight", "600")
      .text("Agent Extra Actions (" + unmatchedAgent.length + ")");

    unmatchedAgent.forEach(function(ae, i) {
      var x = i * (nodeW + 16);
      var label = (ae.tool || "?").replace(/__/g, ".");
      if (label.length > 24) label = label.slice(0, 22) + "..";
      var ng = extraG.append("g").attr("transform", "translate(" + x + ",4)");
      ng.append("rect")
        .attr("width", nodeW).attr("height", nodeH)
        .attr("fill", "#f3e8ff").attr("stroke", "#C930C8").attr("stroke-width", 2)
        .attr("rx", 6).attr("ry", 6);
      ng.append("text").attr("x", 8).attr("y", 17)
        .attr("fill", "#334155").attr("font-weight", "600").attr("font-size", "11px")
        .text(label);
      ng.append("text").attr("x", 8).attr("y", 34)
        .attr("fill", "#64748b").attr("font-size", "10px")
        .text("AGENT (extra)");
    });

    // Expand SVG to fit extras
    var newH = extraY + nodeH + 40;
    svg.attr("viewBox", "0 0 " + Math.max(svgW, marginLeft + unmatchedAgent.length * (nodeW + 16)) + " " + newH);
  }

  // Turn indicators — vertical bars after send_message_to_user nodes
  // and before ENV events that depend on them
  var turnNodes = oracleEvents.filter(function(ev) {
    return ev.function === "send_message_to_user" || ev.function === "send_message_to_agent";
  });
  turnNodes.forEach(function(tn, ti) {
    var pos = positions[tn.event_id];
    if (!pos) return;
    var barX = pos.x + nodeW + padX / 2;
    var label = tn.function === "send_message_to_agent" ? "User Task" : "Turn " + (ti + 1);
    g.append("line")
      .attr("x1", barX).attr("y1", 0)
      .attr("x2", barX).attr("y2", svgH)
      .attr("stroke", "#cbd5e1").attr("stroke-width", 1.5)
      .attr("stroke-dasharray", "6,4");
    g.append("text")
      .attr("x", barX).attr("y", 12)
      .attr("text-anchor", "middle")
      .attr("fill", "#94a3b8").attr("font-size", "10px").attr("font-weight", "600")
      .text(label);
  });

  // Fit to view on load
  svg.call(zoom.transform, d3.zoomIdentity);
})();
"""
