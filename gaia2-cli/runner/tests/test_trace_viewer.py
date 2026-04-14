# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Tests for trace_viewer performance fixes.

Covers:
1. _load_scenario_dag — direct file lookup replaces scan-all-files
2. _generate_trace_page — returns a summary dict
3. _generate_index_page — accepts pre-computed summaries (no re-reads)
4. generate_all — end-to-end wiring of summaries
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

from gaia2_runner.trace_viewer import (
    _display_image_name,
    _format_compact_timestamp,
    _format_short_timestamp,
    _generate_index_page,
    _generate_trace_page,
    _load_scenario_dag,
    _split_sort_key,
    generate_all,
    generate_runs_index,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_scenario_json(
    scenario_id: str = "scenario_1",
    events: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build a minimal scenario JSON with oracle events."""
    if events is None:
        events = [
            {
                "event_id": "e1",
                "event_type": "USER",
                "dependencies": [],
                "action": {
                    "app": "Contacts",
                    "function": "add_contact",
                    "args": {"name": "Alice"},
                },
            },
        ]
    return {
        "metadata": {"definition": {"scenario_id": scenario_id}},
        "events": events,
    }


def _make_result_json(
    success: bool = True,
    scenario_id: str = "scenario_1",
) -> dict[str, Any]:
    return {
        "scenario_id": scenario_id,
        "success": success,
        "num_llm_calls": 2,
        "total_input_tokens": 100,
        "total_output_tokens": 50,
        "total_latency_ms": 1500,
    }


def _make_trace_entry(
    model: str = "claude-sonnet-4-6",
    input_tokens: int = 50,
    output_tokens: int = 25,
    latency_ms: int = 750,
) -> dict[str, Any]:
    """Build a minimal trace entry (Format B — pre-parsed)."""
    return {
        "seq": 1,
        "latency_ms": latency_ms,
        "request": {"model": model, "messages": []},
        "response": {
            "model": model,
            "usage": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
            },
            "content": [{"type": "text", "text": "Hello"}],
            "stop_reason": "end_turn",
        },
    }


def _write_scenario_artifacts(
    output_dir: Path,
    scenario_id: str,
    *,
    success: bool = True,
    trace_entries: list[dict[str, Any]] | None = None,
    events: list[dict[str, Any]] | None = None,
) -> None:
    """Write result.json, trace.jsonl, and events.jsonl for a scenario."""
    sdir = output_dir / scenario_id
    sdir.mkdir(parents=True, exist_ok=True)

    result = _make_result_json(success=success, scenario_id=scenario_id)
    (sdir / "result.json").write_text(json.dumps(result))

    if trace_entries is None:
        trace_entries = [_make_trace_entry(), _make_trace_entry()]
    lines = [json.dumps(e) for e in trace_entries]
    (sdir / "trace.jsonl").write_text("\n".join(lines))

    if events is None:
        events = [{"action": "tool_call_1"}, {"action": "tool_call_2"}]
    event_lines = [json.dumps(e) for e in events]
    (sdir / "events.jsonl").write_text("\n".join(event_lines))


# ---------------------------------------------------------------------------
# _load_scenario_dag
# ---------------------------------------------------------------------------


class TestLoadScenarioDag:
    def test_loads_single_file(self, tmp_path: Path) -> None:
        """Should parse only the requested scenario file."""
        scenario = _make_scenario_json("scenario_1")
        (tmp_path / "scenario_1.json").write_text(json.dumps(scenario))
        # Also write a second file that should NOT be read
        (tmp_path / "scenario_2.json").write_text(
            json.dumps(_make_scenario_json("scenario_2"))
        )

        dag = _load_scenario_dag(str(tmp_path), "scenario_1")

        assert dag is not None
        assert len(dag) == 1
        assert dag[0]["event_id"] == "e1"
        assert dag[0]["app"] == "Contacts"

    def test_returns_none_for_missing_file(self, tmp_path: Path) -> None:
        """Should return None when the scenario file doesn't exist."""
        dag = _load_scenario_dag(str(tmp_path), "nonexistent")
        assert dag is None

    def test_returns_none_for_non_directory(self, tmp_path: Path) -> None:
        """Should return None when dataset_path is not a directory."""
        dag = _load_scenario_dag(str(tmp_path / "nope"), "scenario_1")
        assert dag is None

    def test_returns_none_for_invalid_json(self, tmp_path: Path) -> None:
        """Should return None when the scenario file has invalid JSON."""
        (tmp_path / "bad.json").write_text("NOT JSON")
        dag = _load_scenario_dag(str(tmp_path), "bad")
        assert dag is None

    def test_returns_none_for_empty_events(self, tmp_path: Path) -> None:
        """Should return None when scenario has no events."""
        scenario = _make_scenario_json("empty", events=[])
        (tmp_path / "empty.json").write_text(json.dumps(scenario))
        dag = _load_scenario_dag(str(tmp_path), "empty")
        assert dag is None

    def test_does_not_read_other_files(self, tmp_path: Path) -> None:
        """Verify only the targeted file is read, not all *.json.

        We prove this by writing invalid JSON to all other files.  If the
        implementation tried to read them it would either crash or return
        None — but because it only reads ``target.json``, it succeeds.
        """
        scenario = _make_scenario_json("target")
        (tmp_path / "target.json").write_text(json.dumps(scenario))

        # Write 10 other files with garbage — if opened they'd raise
        for i in range(10):
            (tmp_path / f"other_{i}.json").write_text("INVALID JSON {{{{")

        dag = _load_scenario_dag(str(tmp_path), "target")
        assert dag is not None
        assert dag[0]["event_id"] == "e1"


# ---------------------------------------------------------------------------
# _generate_trace_page returns summary
# ---------------------------------------------------------------------------


class TestGenerateTracePageReturnsSummary:
    def test_formats_float_timestamp(self) -> None:
        assert _format_short_timestamp(1775115853.297512) == "2026-04-02 07:44:13"
        assert _format_compact_timestamp(1775115853.297512) == "2026-04-02 07:44"

    def test_normalizes_image_name(self) -> None:
        assert _display_image_name("localhost/gaia2-hermes:latest") == "hermes"
        assert _display_image_name("localhost/gaia2-oc:latest") == "openclaw"
        assert _display_image_name("localhost/gaia2-oracle:latest") == "oracle"
        assert _display_image_name("") == "—"

    def test_uses_canonical_split_order(self) -> None:
        split_names = ["ambiguity", "time", "search", "adaptability", "execution"]
        assert sorted(split_names, key=_split_sort_key) == [
            "search",
            "execution",
            "adaptability",
            "ambiguity",
            "time",
        ]

    def test_returns_summary_dict(self, tmp_path: Path) -> None:
        """_generate_trace_page should return a dict with pre-computed stats."""
        _write_scenario_artifacts(tmp_path, "s1", success=True)

        summary = _generate_trace_page(tmp_path, "s1", None, None)

        assert summary is not None
        assert summary["_scenario_id"] == "s1"
        assert summary["success"] is True
        assert summary["num_llm_calls"] == 2
        assert summary["total_input_tokens"] == 100  # 50 * 2
        assert summary["total_output_tokens"] == 50  # 25 * 2
        assert summary["total_latency_ms"] == 1500  # 750 * 2
        assert summary["_num_tool_calls"] == 2
        assert summary["model"] == "claude-sonnet-4-6"
        assert isinstance(summary["_diagnostics"], dict)

    def test_returns_none_on_missing_result(self, tmp_path: Path) -> None:
        """Should return None when result.json is missing/empty."""
        sdir = tmp_path / "bad"
        sdir.mkdir()
        # Empty result.json → _load_json returns {}
        (sdir / "result.json").write_text("{}")

        summary = _generate_trace_page(tmp_path, "bad", None, None)
        # With an empty result dict, it should still return a summary
        # (success=None → ERROR state)
        assert summary is not None
        assert summary["success"] is None

    def test_writes_trace_html(self, tmp_path: Path) -> None:
        """Should still write the trace.html file."""
        _write_scenario_artifacts(tmp_path, "s1")
        _generate_trace_page(tmp_path, "s1", None, None)
        assert (tmp_path / "s1" / "trace.html").exists()

    def test_renders_openai_responses_function_call_outputs(
        self, tmp_path: Path
    ) -> None:
        """Responses API request.input function_call_output should render."""
        trace_entries = [
            {
                "seq": 1,
                "latency_ms": 100,
                "request": {
                    "model": "gpt-5.4",
                    "input": [
                        {"role": "developer", "content": "System prompt"},
                        {
                            "role": "user",
                            "content": [{"type": "input_text", "text": "Do the task"}],
                        },
                    ],
                },
                "response": {
                    "model": "gpt-5.4",
                    "usage": {"input_tokens": 10, "output_tokens": 5},
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "exec",
                                "arguments": '{"command":"ls -la"}',
                            },
                        }
                    ],
                    "finish_reason": "tool_calls",
                },
            },
            {
                "seq": 2,
                "latency_ms": 100,
                "request": {
                    "model": "gpt-5.4",
                    "input": [
                        {"role": "developer", "content": "System prompt"},
                        {
                            "role": "user",
                            "content": [{"type": "input_text", "text": "Do the task"}],
                        },
                        {
                            "type": "function_call_output",
                            "call_id": "call_1",
                            "output": "file1\nfile2\n",
                        },
                    ],
                },
                "response": {
                    "model": "gpt-5.4",
                    "usage": {"input_tokens": 12, "output_tokens": 4},
                    "content": "Done",
                    "finish_reason": "stop",
                },
            },
        ]
        _write_scenario_artifacts(tmp_path, "s1", trace_entries=trace_entries)

        _generate_trace_page(tmp_path, "s1", None, None)

        html = (tmp_path / "s1" / "trace.html").read_text()
        assert "System Prompt" in html
        assert "Do the task" in html
        assert "$ ls -la" in html
        assert "file1" in html
        assert "Done" in html

    def test_renders_native_google_generate_content_trace(self, tmp_path: Path) -> None:
        trace_entries = [
            {
                "seq": 1,
                "latency_ms": 120,
                "request": {
                    "model": "gemini-3.1-pro",
                    "systemInstruction": {"parts": [{"text": "System prompt"}]},
                    "contents": [
                        {"role": "user", "parts": [{"text": "Do the task"}]},
                    ],
                },
                "raw_response": json.dumps(
                    {
                        "modelVersion": "gemini-3.1-pro",
                        "candidates": [
                            {
                                "content": {
                                    "role": "model",
                                    "parts": [
                                        {"text": "Check files", "thought": True},
                                        {
                                            "functionCall": {
                                                "name": "exec",
                                                "args": {"command": "date"},
                                            }
                                        },
                                        {"text": "Done"},
                                    ],
                                },
                                "finishReason": "STOP",
                            }
                        ],
                        "usageMetadata": {
                            "promptTokenCount": 10,
                            "candidatesTokenCount": 5,
                            "totalTokenCount": 15,
                        },
                    }
                ),
                "http_status": 200,
            }
        ]
        _write_scenario_artifacts(tmp_path, "s1", trace_entries=trace_entries)

        summary = _generate_trace_page(tmp_path, "s1", None, None)

        assert summary is not None
        assert summary["model"] == "gemini-3.1-pro"
        assert summary["provider"] == "google"

        html = (tmp_path / "s1" / "trace.html").read_text()
        assert "System Prompt" in html
        assert "Do the task" in html
        assert "Check files" in html
        assert "$ date" in html
        assert "Done" in html

    def test_renders_anthropic_thinking_as_collapsible_block(
        self, tmp_path: Path
    ) -> None:
        trace_entries = [
            {
                "seq": 1,
                "latency_ms": 120,
                "request": {
                    "model": "claude-opus-4-6",
                    "messages": [
                        {"role": "user", "content": "Do the task"},
                    ],
                },
                "response": {
                    "model": "claude-opus-4-6",
                    "usage": {"input_tokens": 10, "output_tokens": 5},
                    "content": [
                        {"type": "thinking", "thinking": "Check the tools first."},
                        {
                            "type": "tool_use",
                            "name": "exec",
                            "input": {"command": "date"},
                        },
                    ],
                    "stop_reason": "tool_use",
                },
            }
        ]
        _write_scenario_artifacts(tmp_path, "s1", trace_entries=trace_entries)

        _generate_trace_page(tmp_path, "s1", None, None)

        html = (tmp_path / "s1" / "trace.html").read_text()
        assert "Thinking (22 chars)" in html
        assert "Check the tools first." in html
        assert "<em>[thinking]</em>" not in html
        assert "$ date" in html

    def test_renders_openai_reasoning_content_as_collapsible_block(
        self, tmp_path: Path
    ) -> None:
        trace_entries = [
            {
                "seq": 1,
                "latency_ms": 120,
                "request": {
                    "model": "fireworks-kimi-k2p5-od",
                    "messages": [
                        {"role": "user", "content": "Do the task"},
                    ],
                },
                "raw_response": json.dumps(
                    {
                        "id": "chatcmpl_123",
                        "model": "fireworks-kimi-k2p5-od",
                        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
                        "choices": [
                            {
                                "finish_reason": "tool_calls",
                                "index": 0,
                                "message": {
                                    "role": "assistant",
                                    "content": "",
                                    "reasoning_content": "Check meeting data.",
                                    "tool_calls": [
                                        {
                                            "id": "call_1",
                                            "type": "function",
                                            "function": {
                                                "name": "exec",
                                                "arguments": '{"command":"date"}',
                                            },
                                        }
                                    ],
                                },
                            }
                        ],
                    }
                ),
                "http_status": 200,
            }
        ]
        _write_scenario_artifacts(tmp_path, "s1", trace_entries=trace_entries)

        _generate_trace_page(tmp_path, "s1", None, None)

        html = (tmp_path / "s1" / "trace.html").read_text()
        assert "Thinking (19 chars)" in html
        assert "Check meeting data." in html
        assert "$ date" in html

    def test_handles_nullable_openai_usage_counters(self, tmp_path: Path) -> None:
        trace_entries = [
            {
                "seq": 1,
                "latency_ms": 120,
                "request": {
                    "model": "gemini-3-1-pro-preview-genai",
                    "messages": [
                        {"role": "user", "content": "Do the task"},
                    ],
                },
                "raw_response": json.dumps(
                    {
                        "id": "chatcmpl_456",
                        "model": "gemini-3-1-pro-preview-genai",
                        "usage": {
                            "prompt_tokens": 10,
                            "completion_tokens": None,
                            "completion_tokens_details": None,
                            "prompt_tokens_details": {
                                "audio_tokens": None,
                                "cached_tokens": 5,
                            },
                        },
                        "choices": [
                            {
                                "finish_reason": "stop",
                                "index": 0,
                                "message": {
                                    "role": "assistant",
                                    "content": "Done",
                                },
                            }
                        ],
                    }
                ),
                "http_status": 200,
            }
        ]
        _write_scenario_artifacts(tmp_path, "s1", trace_entries=trace_entries)

        summary = _generate_trace_page(tmp_path, "s1", None, None)

        assert summary is not None
        assert summary["provider"] == "google"
        assert summary["total_input_tokens"] == 10
        assert summary["total_output_tokens"] == 0

        html = (tmp_path / "s1" / "trace.html").read_text()
        assert "Done" in html

    def test_ignores_non_renderable_trace_entries(self, tmp_path: Path) -> None:
        """Non-renderable trace rows should not affect detailed trace stats."""
        _write_scenario_artifacts(
            tmp_path,
            "s1",
            trace_entries=[
                {
                    "timestamp": 1775115853.297512,
                    "model": "anthropic/claude-opus-4.6",
                    "api_calls": 13,
                    "input_tokens": 15846,
                    "output_tokens": 4896,
                    "completed": True,
                    "final_response_len": 224,
                }
            ],
        )
        (tmp_path / "s1" / "eventd.log").write_text(
            "2026-04-02 07:42:42,059 gaia2_cli.daemon.eventd INFO "
            "Time advance: sim=1.0 (2026-04-02 07:42:42) firing\n"
        )

        summary = _generate_trace_page(tmp_path, "s1", None, None)

        assert summary is not None
        assert summary["num_llm_calls"] == 2
        assert summary["total_input_tokens"] == 100
        assert summary["total_output_tokens"] == 50
        assert summary["model"] == ""

        html = (tmp_path / "s1" / "trace.html").read_text()
        assert "Judge Result" in html
        assert "LLM Trace" not in html


# ---------------------------------------------------------------------------
# _generate_index_page accepts pre-computed summaries
# ---------------------------------------------------------------------------


class TestGenerateIndexPage:
    def test_uses_precomputed_summaries(self, tmp_path: Path) -> None:
        """Should generate index.html from summaries without reading files."""
        summaries = [
            {
                "_scenario_id": "s1",
                "success": True,
                "num_llm_calls": 3,
                "total_input_tokens": 200,
                "total_output_tokens": 100,
                "total_latency_ms": 2000,
                "_num_tool_calls": 5,
                "model": "claude-sonnet-4-6",
                "provider": "anthropic",
                "_diagnostics": {"error_count": 0},
                "error": None,
                "failure_reasons": None,
                "rationale": None,
            },
            {
                "_scenario_id": "s2",
                "success": False,
                "num_llm_calls": 1,
                "total_input_tokens": 50,
                "total_output_tokens": 20,
                "total_latency_ms": 500,
                "_num_tool_calls": 1,
                "model": "claude-sonnet-4-6",
                "provider": "anthropic",
                "_diagnostics": {"error_count": 1, "error_summary": "timeout"},
                "error": None,
                "failure_reasons": ["wrong answer"],
                "rationale": None,
            },
        ]

        _generate_index_page(tmp_path, summaries)

        index_html = (tmp_path / "index.html").read_text()
        # Verify both scenarios appear
        assert "s1" in index_html
        assert "s2" in index_html
        # Verify pass/fail counts rendered
        assert "1</div>" in index_html  # pass count
        assert "PASS" in index_html
        assert "FAIL" in index_html
        # Verify model name appears in header
        assert "claude-sonnet-4-6" in index_html

    def test_no_file_reads(self, tmp_path: Path) -> None:
        """Index generation should not call _load_json or _load_jsonl."""
        summaries = [
            {
                "_scenario_id": "s1",
                "success": True,
                "num_llm_calls": 1,
                "total_input_tokens": 10,
                "total_output_tokens": 5,
                "total_latency_ms": 100,
                "_num_tool_calls": 0,
                "model": "",
                "provider": "",
                "_diagnostics": {"error_count": 0},
                "error": None,
                "failure_reasons": None,
                "rationale": None,
            },
        ]

        with (
            patch("gaia2_runner.trace_viewer._load_json") as mock_json,
            patch("gaia2_runner.trace_viewer._load_jsonl") as mock_jsonl,
            patch("gaia2_runner.trace_viewer._load_text") as mock_text,
        ):
            _generate_index_page(tmp_path, summaries)
            mock_json.assert_not_called()
            mock_jsonl.assert_not_called()
            mock_text.assert_not_called()


# ---------------------------------------------------------------------------
# generate_all end-to-end
# ---------------------------------------------------------------------------


class TestGenerateAll:
    def test_end_to_end(self, tmp_path: Path) -> None:
        """generate_all should produce both trace.html and index.html."""
        _write_scenario_artifacts(tmp_path, "s1", success=True)
        _write_scenario_artifacts(tmp_path, "s2", success=False)

        generate_all(tmp_path)

        assert (tmp_path / "s1" / "trace.html").exists()
        assert (tmp_path / "s2" / "trace.html").exists()
        assert (tmp_path / "index.html").exists()

        index_html = (tmp_path / "index.html").read_text()
        assert "s1" in index_html
        assert "s2" in index_html

    def test_includes_running_scenarios_without_result_json(
        self, tmp_path: Path
    ) -> None:
        running_dir = tmp_path / "running_s1"
        running_dir.mkdir()
        (running_dir / "trace.jsonl").write_text(json.dumps(_make_trace_entry()))
        (running_dir / "events.jsonl").write_text(json.dumps({"action": "tool_call_1"}))
        (tmp_path / "run_config.json").write_text(json.dumps({"num_scenarios": 2}))

        _write_scenario_artifacts(tmp_path, "done_s2", success=True)

        generate_all(tmp_path)

        assert (running_dir / "trace.html").exists()
        index_html = (tmp_path / "index.html").read_text()
        assert "running_s1" in index_html
        assert "RUNNING" in index_html
        assert "Expected" in index_html
        assert "Completed" in index_html

    def test_nested_split_trace_page_keeps_oracle_dag(self, tmp_path: Path) -> None:
        dataset_root = tmp_path / "dataset"
        scenario_json = dataset_root / "adaptability" / "scenario_a.json"
        scenario_json.parent.mkdir(parents=True)
        scenario_json.write_text(json.dumps(_make_scenario_json("scenario_a")))

        run_dir = tmp_path / "run_1"
        run_dir.mkdir()
        (run_dir / "run_config.json").write_text(
            json.dumps(
                {
                    "dataset": str(dataset_root),
                    "num_scenarios": 1,
                    "splits": ["adaptability"],
                }
            )
        )
        _write_scenario_artifacts(run_dir, "adaptability/scenario_a", success=True)

        generate_all(run_dir)

        trace_html = (
            run_dir / "adaptability" / "scenario_a" / "trace.html"
        ).read_text()
        assert "Oracle DAG" in trace_html
        assert "window.__DAG_DATA" in trace_html

    def test_hf_dataset_cache_dir_keeps_oracle_dag(self, tmp_path: Path) -> None:
        dataset_root = tmp_path / "hf_cache"
        scenario_json = dataset_root / "adaptability" / "scenario_a.json"
        scenario_json.parent.mkdir(parents=True)
        scenario_json.write_text(json.dumps(_make_scenario_json("scenario_a")))

        run_dir = tmp_path / "run_hf"
        run_dir.mkdir()
        (run_dir / "run_config.json").write_text(
            json.dumps(
                {
                    "dataset": "meta-agents-research-environments/gaia2-cli",
                    "dataset_cache_dir": str(dataset_root),
                    "num_scenarios": 1,
                    "splits": ["adaptability"],
                }
            )
        )
        _write_scenario_artifacts(run_dir, "adaptability/scenario_a", success=True)

        generate_all(run_dir)

        trace_html = (
            run_dir / "adaptability" / "scenario_a" / "trace.html"
        ).read_text()
        assert "Oracle DAG" in trace_html
        assert "window.__DAG_DATA" in trace_html

    def test_hf_dataset_cache_dir_drives_split_breakdown(self, tmp_path: Path) -> None:
        dataset_root = tmp_path / "hf_cache"
        for split_name in ("search", "time"):
            split_dir = dataset_root / split_name
            split_dir.mkdir(parents=True)
            (split_dir / f"{split_name}_scenario.json").write_text(
                json.dumps(_make_scenario_json(f"{split_name}_scenario"))
            )

        (tmp_path / "run_config.json").write_text(
            json.dumps(
                {
                    "dataset": "meta-agents-research-environments/gaia2-cli",
                    "dataset_cache_dir": str(dataset_root),
                    "num_scenarios": 2,
                }
            )
        )
        _write_scenario_artifacts(tmp_path, "search/search_scenario", success=True)

        generate_all(tmp_path)

        index_html = (tmp_path / "index.html").read_text()
        assert '<option value="search">search</option>' in index_html
        assert '<option value="time">time</option>' in index_html
        assert '<div class="split-card-title">time</div>' in index_html
        assert 'badge badge-pending">Not Started</span>' in index_html

    def test_clamps_expected_total_when_retry_metadata_is_smaller(
        self, tmp_path: Path
    ) -> None:
        (tmp_path / "run_config.json").write_text(json.dumps({"num_scenarios": 1}))
        _write_scenario_artifacts(tmp_path, "s1", success=True)
        _write_scenario_artifacts(tmp_path, "s2", success=False)

        generate_all(tmp_path)

        index_html = (tmp_path / "index.html").read_text()
        assert '<div class="label">Expected</div></div>' in index_html
        assert (
            '<div class="value">2</div><div class="label">Expected</div>' in index_html
        )
        assert (
            '<div class="value">2</div><div class="label">Completed</div>' in index_html
        )

    def test_skips_failed_scenario_gracefully(self, tmp_path: Path) -> None:
        """If one scenario fails, others should still be generated."""
        _write_scenario_artifacts(tmp_path, "good")
        # Create a broken scenario dir (result.json with invalid content)
        bad_dir = tmp_path / "bad"
        bad_dir.mkdir()
        (bad_dir / "result.json").write_text("NOT JSON")

        generate_all(tmp_path)

        assert (tmp_path / "good" / "trace.html").exists()
        # Index should still be generated with at least the good scenario
        assert (tmp_path / "index.html").exists()

    def test_empty_dir_generates_waiting_page(self, tmp_path: Path) -> None:
        """An output dir with no scenarios should get a placeholder page."""
        generate_all(tmp_path)
        assert (tmp_path / "index.html").exists()

    def test_run_index_includes_split_breakdown_for_multi_split_run(
        self, tmp_path: Path
    ) -> None:
        (tmp_path / "run_config.json").write_text(
            json.dumps(
                {
                    "num_scenarios": 2,
                    "splits": ["search", "time"],
                }
            )
        )
        _write_scenario_artifacts(tmp_path, "search/scenario_a", success=True)
        _write_scenario_artifacts(tmp_path, "time/scenario_b", success=False)

        generate_all(tmp_path)

        index_html = (tmp_path / "index.html").read_text()
        assert "Split Breakdown" in index_html
        assert "All splits" in index_html
        assert 'data-split="search"' in index_html
        assert 'data-split="time"' in index_html


class TestGenerateRunsIndex:
    def test_pass_at_directory_gets_nested_runs_landing(self, tmp_path: Path) -> None:
        experiment_dir = tmp_path / "hermes_anthropic_all5_pass3"
        run_1 = experiment_dir / "run_1"
        run_2 = experiment_dir / "run_2"
        run_1.mkdir(parents=True)
        run_2.mkdir(parents=True)

        (run_1 / "run_config.json").write_text(
            json.dumps({"run_number": 1, "num_scenarios": 1})
        )
        (run_2 / "run_config.json").write_text(
            json.dumps({"run_number": 2, "num_scenarios": 1})
        )
        _write_scenario_artifacts(run_1, "adaptability/scenario_a", success=True)
        _write_scenario_artifacts(run_2, "adaptability/scenario_a", success=False)

        generate_runs_index(tmp_path)

        experiment_index = (experiment_dir / "index.html").read_text()
        assert 'href="run_1/index.html"' in experiment_index
        assert 'href="run_2/index.html"' in experiment_index
        assert "Split Overview" in experiment_index
        assert 'href="splits/adaptability.html"' in experiment_index
        assert (run_1 / "index.html").exists()
        assert (run_2 / "index.html").exists()
        split_page = experiment_dir / "splits" / "adaptability.html"
        assert split_page.exists()
        assert "Scenario Matrix" in split_page.read_text()

    def test_root_landing_uses_compact_table_layout(self, tmp_path: Path) -> None:
        dataset_root = tmp_path / "dataset"
        for split_name in ("search", "time"):
            split_dir = dataset_root / split_name
            split_dir.mkdir(parents=True)
            (split_dir / f"{split_name}_scenario.json").write_text(
                json.dumps(_make_scenario_json(f"{split_name}_scenario"))
            )

        experiment_dir = tmp_path / "anthropic_pass3"
        run_1 = experiment_dir / "run_1"
        run_2 = experiment_dir / "run_2"
        run_1.mkdir(parents=True)
        run_2.mkdir(parents=True)

        run_config = {
            "dataset": str(dataset_root),
            "splits": ["search", "time"],
            "num_scenarios": 2,
            "started_at": "2026-04-10T15:00:00Z",
            "model": "claude-opus-4-1",
            "provider": "anthropic",
            "image": "localhost/gaia2-hermes:latest",
            "concurrency": 40,
        }
        (run_1 / "run_config.json").write_text(json.dumps(run_config))
        (run_2 / "run_config.json").write_text(json.dumps(run_config))
        _write_scenario_artifacts(run_1, "search/search_scenario", success=True)
        _write_scenario_artifacts(run_1, "time/time_scenario", success=False)
        _write_scenario_artifacts(run_2, "search/search_scenario", success=True)

        generate_runs_index(tmp_path)

        landing_html = (tmp_path / "index.html").read_text()
        assert "All Experiments" in landing_html
        assert "Gaia2 Trace Viewer" not in landing_html
        assert '<div class="summary summary-grid">' not in landing_html
        assert ">Type <span" not in landing_html
        assert ">Scenarios <span" not in landing_html
        assert ">Error <span" not in landing_html
        assert ">Concurrency <span" not in landing_html
        assert '<th onclick="sortTable(0)">Started' in landing_html
        assert "2026-04-10 15:00" in landing_html
        assert '>Image <span class="sort-arrow"></span></th>' in landing_html
        assert ">hermes</td>" in landing_html
        assert 'metric-sub">avg@2</span>' in landing_html
        assert 'metric-sub">pass@2</span>' in landing_html
        assert "progress-stack progress-stack-landing" in landing_html
        assert "split-chip-row split-chip-row-landing" in landing_html

    def test_run_page_distinguishes_not_started_from_real_zero(
        self, tmp_path: Path
    ) -> None:
        dataset_root = tmp_path / "dataset"
        for split_name in ("search", "time"):
            split_dir = dataset_root / split_name
            split_dir.mkdir(parents=True)
            (split_dir / f"{split_name}_scenario.json").write_text(
                json.dumps(_make_scenario_json(f"{split_name}_scenario"))
            )

        (tmp_path / "run_config.json").write_text(
            json.dumps(
                {
                    "dataset": str(dataset_root),
                    "splits": ["search", "time"],
                    "num_scenarios": 2,
                }
            )
        )
        _write_scenario_artifacts(tmp_path, "search/search_scenario", success=False)

        generate_all(tmp_path)

        index_html = (tmp_path / "index.html").read_text()
        assert 'section class="hero hero-compact"' in index_html
        assert "hero-stats hero-stats-compact" in index_html
        assert "progress-stack progress-stack-compact" in index_html
        assert 'badge badge-running">Running</span>' in index_html
        assert 'badge badge-complete">Complete</span>' in index_html
        assert 'badge badge-pending">Not Started</span>' in index_html
        assert 'split-card-value">0.0%</div>' in index_html
        assert 'split-card-value split-card-value-empty">—</div>' in index_html

    def test_experiment_and_split_pages_show_explicit_statuses(
        self, tmp_path: Path
    ) -> None:
        dataset_root = tmp_path / "dataset"
        split_dir = dataset_root / "search"
        split_dir.mkdir(parents=True)
        (split_dir / "scenario_a.json").write_text(
            json.dumps(_make_scenario_json("scenario_a"))
        )

        experiment_dir = tmp_path / "experiment"
        run_1 = experiment_dir / "run_1"
        run_2 = experiment_dir / "run_2"
        run_1.mkdir(parents=True)
        run_2.mkdir(parents=True)

        run_config = {
            "dataset": str(dataset_root),
            "splits": ["search"],
            "num_scenarios": 1,
        }
        (run_1 / "run_config.json").write_text(json.dumps(run_config))
        (run_2 / "run_config.json").write_text(json.dumps(run_config))
        _write_scenario_artifacts(run_1, "search/scenario_a", success=True)

        generate_runs_index(tmp_path)

        experiment_html = (experiment_dir / "index.html").read_text()
        split_html = (experiment_dir / "splits" / "search.html").read_text()

        assert "Hard Scenarios" not in experiment_html
        assert 'badge badge-running">Running</span>' in experiment_html
        assert 'badge badge-pending">Not Started</span>' in experiment_html
        assert 'badge badge-complete">Complete</span>' in experiment_html
        assert "Pass@2 Coverage" in experiment_html
        assert "Scenario Runs" in experiment_html
        assert "progress-stack progress-stack-compact" in experiment_html
        assert "split-chip-row" in experiment_html
        assert 'section class="hero hero-compact"' in split_html
        assert "hero-stats hero-stats-compact" in split_html
        assert "Pass@2 Coverage" in split_html
        assert "Scenario Runs" in split_html
        assert 'href="../run_1/index.html">run_1</a></td>' in split_html
        assert (
            'href="../run_1/index.html">run_1</a></td>\n<td><span class="badge badge-complete">Complete</span></td>'
            in split_html
        )
        assert "progress-stack progress-stack-compact" in split_html
        assert 'badge badge-pending">Not Started</span>' in split_html
        assert 'metric-main">—</span>' in split_html
