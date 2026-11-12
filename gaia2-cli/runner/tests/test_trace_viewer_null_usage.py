# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Regression tests for scenarios disappearing from the viewer pass rate.

Two defects, both reported in the same issue:

1. A chat reply with an explicit ``"usage": null`` crashed trace page
   generation. ``dict.get("usage", {})`` only falls back for a *missing*
   key, so the null came through and the usage aggregation raised
   ``AttributeError: 'NoneType' object has no attribute 'get'``.
   OpenAI-compatible endpoints emit this both for a ``200`` carrying an
   ``error`` object instead of ``choices`` and for replies that simply omit
   token counters.

2. ``generate_all`` dropped any scenario whose page failed to render, so
   the scenario vanished from ``index.html`` and from the pass rate
   computed over the returned summaries, while the console summary read
   ``result.json`` and kept the right number.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

from gaia2_runner.trace_viewer import _generate_trace_page, generate_all


def _entry(usage: dict[str, Any] | None) -> dict[str, Any]:
    """A trace entry in the pre-parsed shape the runner writes."""
    return {
        "seq": 1,
        "latency_ms": 750,
        "request": {"model": "m", "messages": []},
        "response": {
            "model": "m",
            "usage": usage,
            "content": [{"type": "text", "text": "Hello"}],
            "stop_reason": "end_turn",
        },
    }


def _raw_entry(raw_response: str) -> dict[str, Any]:
    """A trace entry holding a raw provider body, as the HTTP recorder writes."""
    return {
        "seq": 1,
        "http_status": 200,
        "latency_ms": 750,
        "request": {"model": "m", "messages": []},
        "raw_response": raw_response,
    }


def _write_scenario(
    root: Path,
    scenario_id: str,
    *,
    success: bool,
    entries: list[dict[str, Any]],
) -> None:
    sdir = root / scenario_id
    sdir.mkdir(parents=True, exist_ok=True)
    (sdir / "result.json").write_text(
        json.dumps(
            {
                "scenario_id": scenario_id,
                "success": success,
                "model": "claude-sonnet-4-6",
                "num_llm_calls": len(entries),
                "total_input_tokens": 50,
                "total_output_tokens": 25,
                "total_latency_ms": 750,
            }
        )
    )
    (sdir / "events.jsonl").write_text(json.dumps({"action": "tool_call_1"}))
    (sdir / "trace.jsonl").write_text("\n".join(json.dumps(e) for e in entries))


class TestNullUsage:
    def test_null_usage_object_renders(self, tmp_path: Path) -> None:
        """An explicit "usage": null must not break the trace page."""
        _write_scenario(tmp_path, "s1", success=False, entries=[_entry(None)])

        summary = _generate_trace_page(tmp_path, "s1", None, None)

        assert summary is not None
        assert summary["_scenario_id"] == "s1"
        assert summary["total_input_tokens"] == 0
        assert summary["total_output_tokens"] == 0
        assert (tmp_path / "s1" / "trace.html").exists()

    def test_error_body_with_null_choices_and_usage_renders(
        self, tmp_path: Path
    ) -> None:
        """A 200 carrying an error object instead of choices must still render."""
        body = json.dumps(
            {
                "id": None,
                "choices": None,
                "usage": None,
                "error": {"code": 502, "message": "Provider returned error"},
            }
        )
        _write_scenario(tmp_path, "s1", success=False, entries=[_raw_entry(body)])

        summary = _generate_trace_page(tmp_path, "s1", None, None)

        assert summary is not None
        assert (tmp_path / "s1" / "trace.html").exists()

    def test_null_response_renders(self, tmp_path: Path) -> None:
        """An entry with no response body at all must not break the page."""
        entry = {
            "seq": 1,
            "latency_ms": 750,
            "request": {"model": "m", "messages": []},
            "response": None,
        }
        _write_scenario(tmp_path, "s1", success=False, entries=[entry])

        summary = _generate_trace_page(tmp_path, "s1", None, None)

        assert summary is not None
        assert (tmp_path / "s1" / "trace.html").exists()

    def test_scenario_with_null_usage_stays_in_pass_rate(self, tmp_path: Path) -> None:
        """The reported case: one pass, two fails, s2's reply has usage null."""
        _write_scenario(
            tmp_path,
            "s1",
            success=True,
            entries=[_entry({"input_tokens": 50, "output_tokens": 25})],
        )
        _write_scenario(tmp_path, "s2", success=False, entries=[_entry(None)])
        _write_scenario(
            tmp_path,
            "s3",
            success=False,
            entries=[_entry({"input_tokens": 50, "output_tokens": 25})],
        )

        summaries = generate_all(tmp_path)

        assert sorted(s["_scenario_id"] for s in summaries) == ["s1", "s2", "s3"]
        assert sum(1 for s in summaries if s["success"] is True) == 1
        assert len(summaries) == 3


class TestRenderFailureDoesNotDropScenario:
    def test_unrenderable_scenario_still_counted(self, tmp_path: Path) -> None:
        """A page that cannot be rendered must not shrink the denominator."""
        for sid, ok in (("s1", True), ("s2", False), ("s3", False)):
            _write_scenario(
                tmp_path,
                sid,
                success=ok,
                entries=[_entry({"input_tokens": 50, "output_tokens": 25})],
            )

        real = _generate_trace_page

        def boom(output_dir, sid, *args, **kwargs):
            if sid == "s2":
                raise AttributeError("'NoneType' object has no attribute 'get'")
            return real(output_dir, sid, *args, **kwargs)

        with patch("gaia2_runner.trace_viewer._generate_trace_page", side_effect=boom):
            summaries = generate_all(tmp_path)

        assert sorted(s["_scenario_id"] for s in summaries) == ["s1", "s2", "s3"]
        assert sum(1 for s in summaries if s["success"] is True) == 1
        assert len(summaries) == 3

        # The fallback carries result.json's verdict and says why it is partial.
        s2 = next(s for s in summaries if s["_scenario_id"] == "s2")
        assert s2["success"] is False
        assert s2["_status"] == "fail"
        assert s2["_trace_render_failed"] is True
        assert "could not be rendered" in s2["error"]

        # index.html is still written and still lists the scenario.
        index = (tmp_path / "index.html").read_text()
        assert "s2" in index
