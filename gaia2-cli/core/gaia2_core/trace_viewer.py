# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Static HTML trace viewer for Gaia2 ExportedTrace JSON files.

Generates self-contained HTML pages (inline CSS + JS, no external deps
except D3 for the DAG) from ExportedTrace artifacts.  Shows:

- **Conversation timeline**: system prompt, task, LLM outputs with token
  counts, tool calls paired with observations, agent reasoning.
- **Oracle DAG**: interactive D3.js dependency graph of expected events.
- **Judge result**: per-tool counts table, validation verdict, annotation
  rationale when available.

Usage::

    python -m gaia2_core.trace_viewer view /path/to/trace.json
    python -m gaia2_core.trace_viewer view-dir /path/to/traces/
    python -m gaia2_core.trace_viewer serve /path/to/traces/
"""

from __future__ import annotations

import html as html_mod
import json
import logging
import os
import re
import socket
from collections import defaultdict
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Trace loading (ExportedTrace JSON → dicts)
# ---------------------------------------------------------------------------


def _load_json(path: Path) -> dict[str, Any]:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def load_trace(path: Path) -> dict[str, Any]:
    """Load an ExportedTrace JSON, deserializing world_logs in place."""
    data = _load_json(path)
    data["_world_logs"] = []
    for raw in data.get("world_logs", []):
        try:
            data["_world_logs"].append(json.loads(raw) if isinstance(raw, str) else raw)
        except Exception:
            data["_world_logs"].append({"log_type": "unknown", "content": str(raw)})
    return data


def load_traces(directory: Path) -> list[tuple[Path, dict[str, Any]]]:
    """Discover and load all ExportedTrace JSON files in *directory*."""
    results = []
    for p in sorted(directory.rglob("*.json")):
        try:
            with open(p) as f:
                peek = json.load(f)
            v = peek.get("version", "")
            if isinstance(v, str) and (
                v.startswith("are_simulation") or v.startswith("gaia2_")
            ):
                results.append((p, load_trace(p)))
        except Exception:
            continue
    return results


def _sid(data: dict[str, Any]) -> str:
    return data.get("metadata", {}).get("definition", {}).get("scenario_id", "unknown")


def _validation(data: dict[str, Any]) -> str | None:
    return data.get("metadata", {}).get("annotation", {}).get("validation_decision")


def _model(data: dict[str, Any]) -> str:
    return data.get("metadata", {}).get("simulation", {}).get("model_id", "")


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

_ESC = html_mod.escape
_NATURAL_RE = re.compile(r"(\d+)")
_TOOL_MISMATCH_RE = re.compile(
    r"Tool count mismatch: agent=\{(.*?)\}, oracle=\{(.*?)\}$"
)


def _esc(s: str | None) -> str:
    return _ESC(str(s)) if s else ""


def _trunc(s: str, n: int = 200) -> str:
    return s[: n - 1] + "\u2026" if len(s) > n else s


def _fmt_tok(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1e6:.1f}M"
    if n >= 1_000:
        return f"{n / 1e3:.1f}k"
    return str(n)


def _fmt_dur(ms: float) -> str:
    if ms < 1000:
        return f"{ms:.0f}ms"
    return f"{ms / 1000:.1f}s"


def _decode_value(v: Any) -> str:
    """Decode JSON-encoded strings and unicode escapes for display."""
    if v is None:
        return ""
    if not isinstance(v, str):
        return str(v)
    if not v:
        return v
    # JSON arrays/objects: parse and re-serialize with proper Unicode
    if (v.startswith("[") or v.startswith("{")) and (
        v.endswith("]") or v.endswith("}")
    ):
        try:
            return json.dumps(json.loads(v), ensure_ascii=False)
        except Exception:
            pass
    # Strings with JSON unicode escapes: decode via JSON string parsing
    if "\\u" in v:
        try:
            decoded = json.loads(f'"{v}"')
            if isinstance(decoded, str):
                return decoded
        except Exception:
            pass
    return v


def _extract_args(act: dict[str, Any]) -> dict[str, str]:
    """Extract args dict from an event action, decoding values for display."""
    return {
        a["name"]: _decode_value(a.get("value", "")) for a in (act.get("args") or [])
    }


def _fmt_args(args: dict[str, Any], mx: int = 60) -> str:
    if not args:
        return ""
    parts = []
    for k, v in args.items():
        vs = str(v) if not isinstance(v, str) else v
        if len(vs) > mx:
            vs = vs[:mx] + "..."
        parts.append(f"{_esc(k)}={_esc(vs)}")
    return ", ".join(parts)


def _badge(status: str | None) -> str:
    if not status:
        return '<span class="badge badge-unknown">UNKNOWN</span>'
    s = status.lower()
    if s in ("valid", "pass", "passed", "true", "success"):
        return '<span class="badge badge-pass">PASS</span>'
    if s in ("invalid", "fail", "failed", "false"):
        return '<span class="badge badge-fail">FAIL</span>'
    return f'<span class="badge badge-error">{_esc(status)}</span>'


def _format_tool_mismatch(reason: str) -> str | None:
    """Parse tool-count mismatch string into an HTML table."""
    m = _TOOL_MISMATCH_RE.search(reason)
    if not m:
        return None

    def _pc(s):
        if not s.strip():
            return {}
        c = {}
        for pair in s.split(", "):
            k, _, v = pair.rpartition(": ")
            c[k.strip().strip("'\"")] = int(v)
        return c

    ac, oc = _pc(m.group(1)), _pc(m.group(2))
    tools = sorted(set(ac) | set(oc))
    by_app: dict[str, list] = defaultdict(list)
    for t in tools:
        a, o = ac.get(t, 0), oc.get(t, 0)
        app, fn = (t.split("__", 1) + [""])[:2] if "__" in t else ("", t)
        by_app[app].append((fn, a, o))
    prefix = _esc(reason[: m.start()].strip())
    if prefix:
        prefix = f"<span>{prefix}</span><br>"
    td = "padding:2px 6px"
    th = "text-align:left;padding:2px 6px;border-bottom:1px solid #cbd5e1;font-size:11px;color:#64748b"
    rows = []
    for app in sorted(by_app):
        fns = by_app[app]
        for i, (fn, a, o) in enumerate(fns):
            c = "#2b8a3e" if a == o else "#c92a2a"
            b = "" if a == o else ";font-weight:600"
            ac2 = (
                f"<td rowspan='{len(fns)}' style='{td};font-weight:600;border-right:1px solid #e5e7eb'>{_esc(app)}</td>"
                if i == 0
                else ""
            )
            rows.append(
                f"<tr>{ac2}<td style='{td}'>{_esc(fn)}</td><td style='{td};text-align:right;color:{c}{b}'>{a}</td><td style='{td};text-align:right;color:{c}{b}'>{o}</td></tr>"
            )
    return (
        f"{prefix}<table style='margin-top:4px;border-collapse:collapse;font-size:12px;line-height:1.3'>"
        f"<thead><tr><th style='{th}'>App</th><th style='{th}'>Function</th>"
        f"<th style='{th};text-align:right'>Agent</th><th style='{th};text-align:right'>Oracle</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


# ---------------------------------------------------------------------------
# Conversation renderer
# ---------------------------------------------------------------------------

_MAX = 20_000


def _render_conversation(wls: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    seen_sys = False
    for w in wls:
        lt = w.get("log_type", "")
        content = w.get("content", "")
        if isinstance(content, list):
            content = json.dumps(content, indent=2, default=str)
        elif content is None:
            content = ""
        else:
            content = str(content)

        if lt == "system_prompt" and not seen_sys:
            seen_sys = True
            if len(content) > _MAX:
                content = content[:_MAX] + "\n... (truncated)"
            parts.append(
                f'<details class="thinking-details system-details"><summary class="thinking-summary">System Prompt</summary><div class="thinking-content">{_esc(content)}</div></details>'
            )
        elif lt == "task":
            parts.append(
                f'<div class="msg msg-user"><div class="msg-label">Task</div>{_esc(content)}</div>'
            )
        elif lt == "step":
            it = w.get("iteration", "?")
            parts.append(
                f'<div class="turn-divider"><span class="turn-info">Step {it}</span></div>'
            )
        elif lt in (
            "llm_output",
            "llm_output_thought_action",
            "llm_output_plan",
            "llm_output_facts",
            "raw_plan",
            "raw_facts",
        ):
            if len(content) > _MAX:
                content = content[:_MAX] + "\n... (truncated)"
            ti = ""
            pt, ct = w.get("prompt_tokens", 0), w.get("completion_tokens", 0)
            if pt or ct:
                ti = f" | in:{_fmt_tok(pt)} out:{_fmt_tok(ct)}"
                rt = w.get("reasoning_tokens", 0)
                if rt:
                    ti += f" reason:{_fmt_tok(rt)}"
                dur = w.get("completion_duration", 0)
                if dur > 0:
                    ti += f" | {_fmt_dur(dur * 1000)}"
            parts.append(
                f'<div class="msg msg-assistant"><div class="msg-label">LLM Output{_esc(ti)}</div>{_esc(content)}</div>'
            )
        elif lt == "tool_call":
            name = w.get("tool_name", "?")
            args = w.get("tool_arguments", "")
            if isinstance(args, dict):
                args = ", ".join(f"{k}={_trunc(str(v), 80)}" for k, v in args.items())
            parts.append(
                f'<div class="tool-exec">{_esc(name)}({_esc(_trunc(str(args), 500))})</div>'
            )
        elif lt == "observation":
            if len(content) > _MAX:
                content = content[:_MAX] + "\n... (truncated)"
            parts.append(f'<div class="tool-output">{_esc(content)}</div>')
        elif lt in ("thought", "plan", "facts", "rationale", "replan", "refacts"):
            label = {
                "thought": "Thought",
                "plan": "Plan",
                "facts": "Facts",
                "rationale": "Rationale",
                "replan": "Replan",
                "refacts": "Updated Facts",
            }.get(lt, lt.title())
            parts.append(
                f'<details class="thinking-details"><summary class="thinking-summary">{_esc(label)}</summary><div class="thinking-content">{_esc(content)}</div></details>'
            )
        elif lt == "final_answer":
            parts.append(
                f'<div class="final-answer"><span class="badge badge-pass">Final Answer</span><div class="msg msg-assistant" style="border:none;margin:0">{_esc(content)}</div></div>'
            )
        elif lt == "error":
            parts.append(
                f'<div class="error-block"><div class="error-title">Error: {_esc(w.get("error", ""))}</div><div class="error-detail">{_esc(w.get("exception", ""))}</div></div>'
            )
        elif lt == "action":
            app, fn = w.get("app_name", ""), w.get("action_name", "")
            inp = (
                json.dumps(w.get("input", {}), indent=2, default=str)
                if w.get("input")
                else ""
            )
            out = str(w.get("output", ""))
            parts.append(
                f'<div class="env-pair"><div class="env-exec"><div class="env-label">ENV Action</div>{_esc(app)}.{_esc(fn)}<br><pre>{_esc(_trunc(inp, 500))}</pre></div><div class="env-output">{_esc(_trunc(out, 500))}</div></div>'
            )
        elif lt == "environment_notifications":
            parts.append(
                f'<div class="hint-block" style="border-color:#6366f1"><div class="msg-label">Environment Notification</div>{_esc(content)}</div>'
            )
        elif lt == "hint":
            parts.append(
                f'<div class="hint-block"><div class="msg-label">Hint</div>{_esc(content)}</div>'
            )
        elif lt == "agent_user_interface":
            parts.append(
                f'<div class="msg msg-assistant"><div class="msg-label">Agent → User</div>{_esc(content)}</div>'
            )
        elif lt == "subagent":
            label = w.get("name") or w.get("group_id") or "subagent"
            ch = _render_conversation(w.get("children", []))
            parts.append(
                f'<div class="subagent-block"><div class="subagent-label">Subagent: {_esc(label)}</div>{ch}</div>'
            )
        elif lt == "llm_input":
            parts.append(
                f'<details class="thinking-details"><summary class="thinking-summary">LLM Input</summary><div class="thinking-content">{_esc(_trunc(content, _MAX))}</div></details>'
            )
        elif content:
            parts.append(
                f'<details class="thinking-details"><summary class="thinking-summary">{_esc(lt)}</summary><div class="thinking-content">{_esc(_trunc(content, 2000))}</div></details>'
            )
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Judge result renderer
# ---------------------------------------------------------------------------


_MATCHING_RE = re.compile(
    r"-Failure matching agent event \(ID: ([^)]+)\) with oracle event \(ID: ([^)]+)\), reason: (.+)"
)
_ORACLE_TOOL_RE = re.compile(r"tool name: (.+)")
_ORACLE_ARGS_RE = re.compile(r"^-([^:]+): (.+)$", re.MULTILINE)


def _parse_failure_rationale(comment: str) -> dict[str, Any] | None:
    """Parse structured failure rationale into components."""
    if "List of matching attempts:" not in comment:
        return None

    result: dict[str, Any] = {}

    # Extract the unmatched oracle tool
    tool_m = _ORACLE_TOOL_RE.search(comment)
    result["oracle_tool"] = tool_m.group(1) if tool_m else None

    # Extract oracle args
    args_section = (
        comment.split("tool args:")[1].split("List of matching attempts:")[0]
        if "tool args:" in comment
        else ""
    )
    oracle_args = {}
    for m in _ORACLE_ARGS_RE.finditer(args_section):
        oracle_args[m.group(1).strip()] = m.group(2).strip()
    result["oracle_args"] = oracle_args

    # Extract matching attempts
    attempts = []
    for m in _MATCHING_RE.finditer(comment):
        attempts.append(
            {
                "agent_id": m.group(1),
                "oracle_id": m.group(2),
                "reason": m.group(3).strip(),
            }
        )
    result["attempts"] = attempts
    return result


def _render_failure_details(parsed: dict[str, Any]) -> str:
    """Render parsed failure rationale as structured HTML."""
    p: list[str] = []

    # Unmatched oracle tool
    oracle_tool = parsed.get("oracle_tool", "?")
    oracle_args = parsed.get("oracle_args", {})
    p.append(
        '<div style="padding:10px 14px;background:#fef2f2;border-left:3px solid #ef4444;'
        'border-radius:8px;margin-bottom:12px">'
        '<div style="font-size:10px;font-weight:700;color:#991b1b;text-transform:uppercase;'
        'letter-spacing:0.5px;margin-bottom:6px">Unmatched Oracle Event</div>'
        f'<div style="font-weight:600;font-size:13px;color:#991b1b;font-family:monospace">{_esc(oracle_tool)}</div>'
    )
    if oracle_args:
        p.append(
            '<div style="margin-top:6px;font-size:12px;color:#7f1d1d;font-family:monospace">'
        )
        for k, v in oracle_args.items():
            p.append(
                f'<div><span style="color:#991b1b">{_esc(k)}:</span> {_esc(_trunc(str(v), 200))}</div>'
            )
        p.append("</div>")
    p.append("</div>")

    # Matching attempts
    attempts = parsed.get("attempts", [])
    if attempts:
        p.append(
            f'<div style="margin-bottom:12px"><div style="font-size:10px;font-weight:700;color:#475569;'
            f'text-transform:uppercase;letter-spacing:0.5px;margin-bottom:6px">'
            f"Matching Attempts ({len(attempts)})</div>"
        )

        for att in attempts:
            reason = att["reason"]
            agent_id = att["agent_id"]

            # Determine styling based on reason
            if reason == "already matched":
                bg, border, color = "#f8fafc", "#e2e8f0", "#94a3b8"
                icon = "&#x2212;"  # minus
            elif "tool judge reject" in reason:
                bg, border, color = "#fef8f8", "#fee2e2", "#991b1b"
                icon = "&#x2718;"  # cross
            else:
                bg, border, color = "#f8fafc", "#e2e8f0", "#64748b"
                icon = "&#x25CB;"  # circle

            p.append(
                f'<div style="margin:4px 0;padding:6px 10px;background:{bg};'
                f'border-radius:6px;border:1px solid {border};font-size:11px">'
                f'<div style="display:flex;align-items:flex-start;gap:6px">'
                f'<span style="color:{color};flex-shrink:0">{icon}</span>'
                f'<div style="min-width:0">'
                f'<div style="font-weight:600;color:{color};word-break:break-all;font-family:monospace;font-size:10px">'
                f"{_esc(agent_id)}</div>"
            )

            # Parse checker details from reason
            if "tool judge reject" in reason:
                # Extract checker info: "tool judge reject (checkers: email_checker (agent_args={...}))"
                checker_m = re.search(r"checkers: (.+)", reason)
                if checker_m:
                    checker_info = checker_m.group(1)
                    p.append(
                        f'<div style="margin-top:4px;padding:6px 8px;background:#fff5f5;'
                        f'border-radius:4px;font-size:10px;color:#991b1b;word-break:break-all">'
                        f"<strong>Checker:</strong> {_esc(_trunc(checker_info, 500))}</div>"
                    )
                else:
                    p.append(
                        f'<div style="margin-top:2px;color:{color};font-size:10px">{_esc(reason)}</div>'
                    )
            elif reason != "already matched":
                p.append(
                    f'<div style="margin-top:2px;color:{color};font-size:10px">{_esc(reason)}</div>'
                )

            p.append("</div></div></div>")

        p.append("</div>")

    return "\n".join(p)


def _render_judge(data: dict[str, Any]) -> str:
    parts: list[str] = []
    ann = data.get("metadata", {}).get("annotation", {})
    defn = data.get("metadata", {}).get("definition", {})
    verdict = ann.get("validation_decision")
    comment = ann.get("comment")

    # Verdict badge
    parts.append(f'<div style="margin-bottom:12px">{_badge(verdict)}</div>')

    # Exception info
    if defn.get("has_exception"):
        parts.append(
            f'<div style="padding:10px 14px;background:#fef2f2;border-left:3px solid #ef4444;'
            f'border-radius:8px;font-size:13px;color:#991b1b;margin-bottom:12px">'
            f"<strong>{_esc(defn.get('exception_type', 'Exception'))}:</strong> "
            f"{_esc(_trunc(defn.get('exception_message', ''), 500))}</div>"
        )

    # Extract judge matching from comment if present
    id_mapping: dict[str, str] = {}
    matching_data: dict[str, Any] | None = None
    rationale_text = comment or ""
    if rationale_text and "__JUDGE_MATCHING__" in rationale_text:
        rationale_part, _, matching_part = rationale_text.partition(
            "__JUDGE_MATCHING__"
        )
        rationale_text = rationale_part.strip()
        try:
            matching_data = json.loads(matching_part.strip())
            id_mapping = matching_data.get("id_mapping", {})
        except Exception:
            pass

    # Build rejection map: {oracle_id: {agent_id: {"reason": ..., "judge_output": ...}}}
    rejection_map: dict[str, dict[str, dict[str, str]]] = {}
    # First, check for structured match_details from the judge
    exported_match_details = (
        matching_data.get("match_details", []) if matching_data else []
    )
    for md in exported_match_details:
        oid = md.get("oracle_id", "")
        aid = md.get("agent_id", "")
        if oid and aid:
            rejection_map.setdefault(oid, {})[aid] = {
                "reason": md.get("reason", ""),
                "judge_output": md.get("judge_output", ""),
            }
    # Fallback: parse from rationale text if no structured match_details
    if not rejection_map and rationale_text:
        for m in _MATCHING_RE.finditer(rationale_text):
            agent_id, oracle_id, reason = m.group(1), m.group(2), m.group(3).strip()
            if reason != "already matched":
                rejection_map.setdefault(oracle_id, {})[agent_id] = {
                    "reason": reason,
                    "judge_output": "",
                }

    # Show judge reasoning from match_details with inline comparison
    if exported_match_details:
        # Build event lookups for comparison rendering
        _all_events = data.get("events", [])
        _all_completed = data.get("completed_events", [])
        _ev_by_id: dict[str, dict] = {}
        for _e in _all_events:
            _ev_by_id[_e.get("event_id", "")] = _e
        for _c in _all_completed:
            _ev_by_id[_c.get("event_id", "")] = _c

        for md in exported_match_details:
            judge_output = md.get("judge_output", "")
            if not judge_output:
                continue
            oid, aid = md.get("oracle_id", ""), md.get("agent_id", "")
            oe = _ev_by_id.get(oid, {})
            ae = _ev_by_id.get(aid, {})
            o_act = oe.get("action") or {}
            a_act = ae.get("action") or {}
            o_fn = f"{o_act.get('app', '?')}.{o_act.get('function', '?')}"
            a_fn = f"{a_act.get('app', '?')}.{a_act.get('function', '?')}"
            o_args = _extract_args(o_act)
            a_args = _extract_args(a_act)
            all_keys = list(dict.fromkeys(list(o_args) + list(a_args)))

            parts.append(
                '<div style="background:#fff;border-radius:12px;border:1px solid #e9d5ff;'
                'margin-bottom:12px;overflow:hidden;max-width:100%">'
            )
            # Header
            parts.append(
                '<div style="padding:10px 14px;background:#f5f3ff;border-bottom:1px solid #e9d5ff;'
                'display:flex;justify-content:space-between;align-items:center">'
                '<div style="font-size:10px;font-weight:700;color:#7c3aed;text-transform:uppercase;'
                'letter-spacing:0.5px">Rejection Detail</div>'
                '<span class="badge badge-fail" style="font-size:9px;padding:2px 8px">REJECTED</span>'
                "</div>"
            )
            # Tool labels
            parts.append(
                '<div style="display:flex;gap:8px;padding:8px 14px">'
                f'<div style="flex:1;padding:6px 10px;background:#f0fdf4;border-radius:6px;font-size:11px">'
                f'<strong style="color:#475569">Oracle:</strong> <span style="color:#166534">{_esc(o_fn)}</span></div>'
                f'<div style="flex:1;padding:6px 10px;background:#fef2f2;border-radius:6px;font-size:11px">'
                f'<strong style="color:#475569">Agent:</strong> <span style="color:#991b1b">{_esc(a_fn)}</span></div>'
                "</div>"
            )
            # Comparison table
            if all_keys:
                td = "padding:5px 10px;font-family:monospace;font-size:11px;word-break:break-word;white-space:pre-wrap;vertical-align:top;border-bottom:1px solid #f1f5f9;max-width:250px"
                th = "padding:5px 10px;text-align:left;font-size:10px;color:#64748b;text-transform:uppercase;letter-spacing:.5px;border-bottom:2px solid #e2e8f0"
                parts.append(
                    f'<div style="padding:0 14px;overflow-x:auto"><table style="width:100%;border-collapse:collapse;table-layout:fixed">'
                    f'<tr><th style="{th};width:20%">Param</th><th style="{th};width:40%;color:#166534">Oracle</th>'
                    f'<th style="{th};width:40%;color:#991b1b">Agent</th></tr>'
                )
                for k in all_keys:
                    ov = o_args.get(k, "")
                    av = a_args.get(k, "")
                    same = str(ov) == str(av)
                    bg = "#f0fdf4" if same else "#fef2f2"
                    dc = "#166534" if same else "#991b1b"
                    parts.append(
                        f'<tr style="background:{bg}">'
                        f'<td style="{td};color:#64748b;font-weight:600">{_esc(k)}</td>'
                        f'<td style="{td};color:#334155">{_esc(str(ov) or "(missing)")}</td>'
                        f'<td style="{td};color:{dc}">{_esc(str(av) or "(missing)")}</td></tr>'
                    )
                parts.append("</table></div>")
            # Judge reasoning
            parts.append(
                f'<div style="padding:10px 14px;margin:8px 14px;background:#f5f3ff;border-radius:8px;'
                f'border:1px solid #e9d5ff">'
                f'<div style="font-size:10px;font-weight:700;color:#7c3aed;text-transform:uppercase;'
                f'letter-spacing:0.5px;margin-bottom:6px">Judge Reasoning</div>'
                f'<div style="font-size:12px;color:#4c1d95;white-space:pre-wrap;'
                f'word-break:break-word;line-height:1.6">{_esc(judge_output)}</div>'
                "</div>"
            )
            parts.append("</div>")

    # Annotation rationale — parse structured failure info if available
    elif rationale_text and rationale_text != "None":
        parsed = _parse_failure_rationale(rationale_text)
        if parsed:
            parts.append(_render_failure_details(parsed))
        else:
            parts.append(
                f'<div class="rationale"><div class="msg-label">Judge Rationale</div>{_esc(rationale_text)}</div>'
            )

    # Per-tool counts — only AGENT events, only WRITE operations for agent side
    events = data.get("events", [])
    completed = data.get("completed_events", [])
    oracle_agent = [e for e in events if e.get("event_type") == "AGENT"]
    completed_agent = [c for c in completed if c.get("event_type") == "AGENT"]
    completed_writes = [
        c
        for c in completed_agent
        if (c.get("action") or {}).get("operation_type") in ("WRITE", None)
    ]

    per_tool: dict[str, dict[str, int]] = defaultdict(lambda: {"agent": 0, "oracle": 0})
    for ev in oracle_agent:
        act = ev.get("action") or {}
        if act:
            per_tool[f"{act.get('app', '')}__{act.get('function', '')}"]["oracle"] += 1
    for cev in completed_writes:
        act = cev.get("action") or {}
        if act:
            per_tool[f"{act.get('app', '')}__{act.get('function', '')}"]["agent"] += 1

    if per_tool:
        parts.append('<h3 style="margin-top:16px;font-size:14px">Per-Tool Counts</h3>')
        parts.append(
            '<div class="table-wrap"><table><thead><tr><th>App</th><th>Function</th><th>Agent</th><th>Oracle</th></tr></thead><tbody>'
        )
        by_app: dict[str, list] = defaultdict(list)
        for tool, counts in sorted(per_tool.items()):
            a, o = counts["agent"], counts["oracle"]
            app, fn = (tool.split("__", 1) + [""])[:2]
            by_app[app].append((fn, a, o))
        for app in sorted(by_app):
            fns = by_app[app]
            for i, (fn, a, o) in enumerate(fns):
                m = (
                    "style='color:#2b8a3e'"
                    if a == o
                    else "style='color:#c92a2a;font-weight:600'"
                )
                ac2 = f"<td rowspan='{len(fns)}'>{_esc(app)}</td>" if i == 0 else ""
                parts.append(
                    f"<tr>{ac2}<td>{_esc(fn)}</td><td {m}>{a}</td><td {m}>{o}</td></tr>"
                )
        parts.append("</tbody></table></div>")

    parts.append(
        f'<div style="margin-top:8px;font-size:13px;color:#868e96">'
        f"Agent writes: {len(completed_writes)} | Oracle: {len(oracle_agent)} | "
        f"Total agent events: {len(completed_agent)}</div>"
    )

    # Oracle events — show matched/unmatched using id_mapping
    # id_mapping is agent_event_id → oracle_event_id
    # Invert: oracle_id → agent_id
    oracle_to_agent = {v: k for k, v in id_mapping.items()}

    # Build completed event lookup — completed_events use EITHER the oracle
    # event ID (for matched events) or the agent event ID, so index by both.
    completed_by_id: dict[str, dict[str, Any]] = {}
    for c in completed:
        completed_by_id[c.get("event_id", "")] = c
    # Also index by agent ID from the mapping for cross-reference
    for agent_id, oracle_id in id_mapping.items():
        if oracle_id in completed_by_id:
            completed_by_id[agent_id] = completed_by_id[oracle_id]
        elif agent_id in completed_by_id:
            completed_by_id[oracle_id] = completed_by_id[agent_id]

    # Determine which unmatched oracle events were actually evaluated by the judge.
    # The judge stops at the first failure, so events after that are "not evaluated".
    # Events in rejection_map were evaluated (judge attempted matching).
    # The rationale text describes ONE failed oracle event — extract its ID.
    rejected_oracle_ids: set[str] = set(rejection_map.keys())
    # Also check the rationale's matching attempts for the oracle ID
    if rationale_text:
        for m in _MATCHING_RE.finditer(rationale_text):
            rejected_oracle_ids.add(m.group(2))  # oracle_id from matching attempt

    if oracle_agent:
        # Split oracle events into three categories
        matched_oe_html: list[str] = []
        rejected_oe_html: list[str] = []
        skipped_oe_html: list[str] = []
        n_matched = 0

        for oe in oracle_agent:
            oe_id = oe.get("event_id", "")
            act = oe.get("action") or {}
            fn = f"{act.get('app', '?')}.{act.get('function', '?')}"
            args = _extract_args(act)
            agent_id = oracle_to_agent.get(oe_id)
            matched = agent_id is not None
            # Determine state: matched / rejected / skipped
            rejected = not matched and id_mapping and oe_id in rejected_oracle_ids
            skipped = not matched and id_mapping and oe_id not in rejected_oracle_ids
            row: list[str] = []

            if matched:
                icon, icon_color = "&#x2714;", "#166534"
                bg, border = "#f8fdf8", "#dcfce7"
            elif rejected:
                icon, icon_color = "&#x2718;", "#991b1b"
                bg, border = "#fef8f8", "#fee2e2"
            elif skipped:
                icon, icon_color = "&#x25CB;", "#94a3b8"
                bg, border = "#f8fafc", "#e2e8f0"
            else:
                icon, icon_color, bg, border = "", "#475569", "#f8fafc", "#e2e8f0"

            o_args_json = _esc(json.dumps(args, default=str))
            ae_args_json = ""
            ae_fn_str = ""
            if matched and agent_id:
                n_matched += 1
                ae = completed_by_id.get(agent_id, {})
                ae_act = ae.get("action") or {}
                ae_fn_str = f"{ae_act.get('app', '?')}.{ae_act.get('function', '?')}"
                ae_args_raw = _extract_args(ae_act)
                ae_args_json = _esc(json.dumps(ae_args_raw, default=str))

            if matched and agent_id:
                click_attr = (
                    f' onclick="jlOpenMatched(this)" '
                    f'data-oracle-tool="{_esc(fn)}" data-oracle-args="{o_args_json}" '
                    f'data-agent-tool="{_esc(ae_fn_str)}" data-agent-args="{ae_args_json}" '
                    f'title="Click to compare" '
                    f'style="cursor:pointer;transition:outline .1s;'
                )
            elif rejected:
                rejections_attr = ""
                if oe_id in rejection_map:
                    rejections_attr = f' data-rejections="{_esc(json.dumps(rejection_map[oe_id], default=str))}"'
                click_attr = (
                    f' class="jl-oracle-ev" data-tool="{_esc(fn)}" '
                    f'data-args="{o_args_json}" '
                    f"{rejections_attr} "
                    f'onclick="jlSelectOracle(this)" '
                    f'title="Select to compare with an agent event" '
                    f'style="cursor:pointer;transition:outline .1s;'
                )
            else:
                click_attr = 'style="'

            row.append(
                f"<div{click_attr}margin:4px 0;padding:8px 10px;background:{bg};"
                f'border-radius:8px;border:1px solid {border};font-size:11px">'
                f'<div style="display:flex;align-items:flex-start;gap:6px">'
            )
            if rejected:
                row.append(
                    '<input type="radio" name="jl-oracle" '
                    'style="margin-top:2px;flex-shrink:0;pointer-events:none;accent-color:#6366f1">'
                )
            if icon:
                row.append(
                    f'<span style="color:{icon_color};flex-shrink:0">{icon}</span>'
                )
            row.append(
                f'<div style="min-width:0">'
                f'<div style="font-weight:600;color:#475569;word-break:break-all">Oracle: {_esc(fn)}</div>'
            )
            if skipped:
                row.append(
                    '<div style="color:#94a3b8;font-size:10px;font-style:italic;margin-top:2px">'
                    "Not evaluated (judge stopped at earlier failure)</div>"
                )
            ab = _fmt_args(args)
            if ab:
                row.append(
                    f'<div style="color:#64748b;font-family:monospace;font-size:10px;margin-top:2px;word-break:break-all">{ab}</div>'
                )

            if matched and agent_id:
                ae = completed_by_id.get(agent_id, {})
                ae_act = ae.get("action") or {}
                ae_fn_display = (
                    f"{ae_act.get('app', '?')}.{ae_act.get('function', '?')}"
                )
                ae_args_display = _extract_args(ae_act)
                ae_ab = _fmt_args(ae_args_display)
                row.append(
                    '<div style="margin-top:6px;padding-top:6px;border-top:1px dashed #d1d5db">'
                    f'<div style="font-weight:600;color:#166534;word-break:break-all">&rarr; Agent: {_esc(ae_fn_display)}</div>'
                )
                if ae_ab:
                    row.append(
                        f'<div style="color:#64748b;font-family:monospace;font-size:10px;margin-top:2px;word-break:break-all">{ae_ab}</div>'
                    )
                row.append("</div>")

            row.append("</div></div></div>")
            row_html = "\n".join(row)
            if matched:
                matched_oe_html.append(row_html)
            elif rejected:
                rejected_oe_html.append(row_html)
            else:
                skipped_oe_html.append(row_html)

        parts.append(
            f'<h3 style="margin-top:16px;font-size:14px">Oracle Events ({len(oracle_agent)})</h3>'
        )

        # Rejected oracle events first (always visible, with radio select)
        for h in rejected_oe_html:
            parts.append(h)

        # Skipped/not-evaluated oracle events are hidden (judge never reached them)

        # Matched oracle events folded
        if matched_oe_html:
            parts.append(
                f'<details style="margin-top:8px">'
                f'<summary style="font-size:10px;font-weight:600;color:#166534;'
                f"text-transform:uppercase;letter-spacing:0.5px;cursor:pointer;"
                f'list-style:none;display:flex;align-items:center;gap:6px">'
                f'<span style="font-size:8px;transition:transform .2s">&#x25B6;</span>'
                f"Matched ({n_matched})</summary>"
            )
            for h in matched_oe_html:
                parts.append(h)
            parts.append("</details>")

    # Unmatched agent events
    # completed_events may use either the agent ID or oracle ID, so exclude both
    if id_mapping and completed_agent:
        matched_ids = set(id_mapping.keys()) | set(id_mapping.values())
        unmatched_agent = [
            c for c in completed_agent if c.get("event_id", "") not in matched_ids
        ]
        unmatched_writes_list = [
            c
            for c in unmatched_agent
            if (c.get("action") or {}).get("operation_type") in ("WRITE", None)
        ]
        unmatched_reads_list = [
            c
            for c in unmatched_agent
            if (c.get("action") or {}).get("operation_type") == "READ"
        ]

        if unmatched_writes_list:
            parts.append(
                '<details style="margin-top:10px;padding-top:8px;border-top:1px solid #e2e8f0">'
                '<summary style="font-size:10px;font-weight:600;color:#991b1b;'
                "text-transform:uppercase;letter-spacing:0.5px;cursor:pointer;"
                'list-style:none;display:flex;align-items:center;gap:6px">'
                '<span style="font-size:8px;transition:transform .2s">&#x25B6;</span>'
                f"Unmatched Agent Writes ({len(unmatched_writes_list)})</summary>"
            )
            for ae in unmatched_writes_list:
                ae_act = ae.get("action") or {}
                ae_fn = f"{ae_act.get('app', '?')}.{ae_act.get('function', '?')}"
                ae_args = _extract_args(ae_act)
                ae_args_json = _esc(json.dumps(ae_args, default=str))
                ae_ab = _fmt_args(ae_args, 300)
                ae_eid = ae.get("event_id", "")
                parts.append(
                    f'<div class="jl-agent-ev" data-tool="{_esc(ae_fn)}" '
                    f'data-args="{ae_args_json}" '
                    f'data-event-id="{_esc(ae_eid)}" '
                    f'onclick="jlSelectAgent(this)" '
                    f'title="Select to compare with an oracle event" '
                    f'style="margin:4px 0;padding:6px 10px;font-size:11px;'
                    f"color:#991b1b;background:#fef2f2;border-radius:6px;"
                    f"border:1px solid #fecaca;word-break:break-all;"
                    f"cursor:pointer;transition:outline .1s;"
                    f'display:flex;align-items:flex-start;gap:8px">'
                    f'<input type="radio" name="jl-agent" '
                    f'style="margin-top:2px;flex-shrink:0;pointer-events:none;accent-color:#6366f1">'
                    f'<div style="min-width:0">'
                    f'<div style="font-weight:600">{_esc(ae_fn)}</div>'
                )
                if ae_ab:
                    parts.append(
                        f'<div style="color:#64748b;font-family:monospace;'
                        f'font-size:10px;margin-top:2px">{ae_ab}</div>'
                    )
                parts.append("</div></div>")
            parts.append("</details>")

        if unmatched_reads_list:
            parts.append(
                '<details style="margin-top:10px;padding-top:8px;border-top:1px solid #e2e8f0">'
                '<summary style="font-size:10px;font-weight:600;color:#94a3b8;'
                "text-transform:uppercase;letter-spacing:0.5px;cursor:pointer;"
                'list-style:none;display:flex;align-items:center;gap:6px">'
                '<span style="font-size:8px;transition:transform .2s" '
                'class="ignore-arrow">&#x25B6;</span>'
                f"Ignored ({len(unmatched_reads_list)})</summary>"
            )
            for ae in unmatched_reads_list:
                ae_act = ae.get("action") or {}
                ae_fn = f"{ae_act.get('app', '?')}.{ae_act.get('function', '?')}"
                ae_args = _extract_args(ae_act)
                ae_ab = _fmt_args(ae_args)
                parts.append(
                    f'<div style="margin:2px 0;padding:4px 8px;font-size:10px;'
                    f"color:#94a3b8;background:#f8fafc;border-radius:4px;"
                    f'word-break:break-all">'
                    f"{_esc(ae_fn)}"
                )
                if ae_ab:
                    parts.append(
                        f'<span style="font-family:monospace">({ae_ab})</span>'
                    )
                parts.append("</div>")
            parts.append("</details>")

    # Hints
    hints = defn.get("hints", [])
    if hints:
        parts.append(
            f'<h3 style="margin-top:16px;font-size:14px">Hints ({len(hints)})</h3>'
        )
        for h in hints:
            parts.append(
                f'<div class="hint-block"><div class="msg-label">{_esc(h.get("hint_type", ""))}</div>{_esc(h.get("content", ""))}</div>'
            )

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# DAG
# ---------------------------------------------------------------------------

_DAG_CSS = """\
#dag-container { width:100%;min-height:200px;max-height:500px;border:1px solid #e2e8f0;border-radius:12px;background:#fafbfc;overflow:hidden;position:relative; }
#dag-container svg { width:100%;height:100%; }
.dag-legend { display:flex;gap:16px;padding:8px 0;font-size:12px;color:#64748b;flex-wrap:wrap; }
.dag-legend-item { display:flex;align-items:center;gap:4px; }
.dag-swatch { width:14px;height:14px;border-radius:3px;border:2px solid;display:inline-block; }
.dag-node { cursor:pointer; }
.dag-node rect { rx:6;ry:6;stroke-width:2; }
.dag-node text { font-size:11px;font-family:system-ui,sans-serif; }
.dag-node:hover rect { filter:brightness(0.95); }
.dag-tooltip { position:absolute;background:#1e293b;color:#f1f5f9;padding:8px 12px;border-radius:8px;font-size:11px;font-family:monospace;max-width:400px;pointer-events:none;z-index:50;white-space:pre-wrap;word-break:break-all;box-shadow:0 4px 12px rgba(0,0,0,.3); }
"""

_DAG_JS = """\
(function(){
var data=window.__DAG_DATA;if(!data||!data.length)return;
var nodeMap={};data.forEach(function(ev){nodeMap[ev.event_id]=ev});
var levels={};
function cl(id){if(levels[id]!==undefined)return levels[id];var ev=nodeMap[id];if(!ev||!ev.deps||!ev.deps.length){levels[id]=0;return 0}var mx=0;ev.deps.forEach(function(d){if(nodeMap[d])mx=Math.max(mx,cl(d)+1)});levels[id]=mx;return mx}
data.forEach(function(ev){cl(ev.event_id)});
var byLv={},maxLv=0;data.forEach(function(ev){var l=levels[ev.event_id]||0;if(!byLv[l])byLv[l]=[];byLv[l].push(ev);maxLv=Math.max(maxLv,l)});
var W=180,H=44,px=60,py=24,ml=30,mt=30,pos={};
for(var l=0;l<=maxLv;l++){var ns=byLv[l]||[];ns.forEach(function(ev,i){pos[ev.event_id]={x:ml+l*(W+px),y:mt+i*(H+py)}})}
var maxY=0;data.forEach(function(ev){var p=pos[ev.event_id];if(p)maxY=Math.max(maxY,p.y+H)});
var svgW=ml+(maxLv+1)*(W+px),svgH=maxY+mt;
var ct=document.getElementById("dag-container");ct.style.height=Math.min(svgH+20,500)+"px";
var svg=d3.select("#dag-container").append("svg").attr("viewBox","0 0 "+svgW+" "+svgH);
var g=svg.append("g");
svg.call(d3.zoom().scaleExtent([.3,3]).on("zoom",function(e){g.attr("transform",e.transform)}));
g.append("defs").append("marker").attr("id","arr").attr("viewBox","0 0 10 10").attr("refX",10).attr("refY",5).attr("markerWidth",8).attr("markerHeight",8).attr("orient","auto").append("path").attr("d","M0 0L10 5L0 10z").attr("fill","#94a3b8");
data.forEach(function(ev){(ev.deps||[]).forEach(function(d){if(!pos[d]||!pos[ev.event_id])return;var s=pos[d],t=pos[ev.event_id],x1=s.x+W,y1=s.y+H/2,x2=t.x,y2=t.y+H/2,mx=(x1+x2)/2;g.append("path").attr("d","M"+x1+","+y1+" C"+mx+","+y1+" "+mx+","+y2+" "+x2+","+y2).attr("fill","none").attr("stroke","#94a3b8").attr("stroke-width",1.5).attr("marker-end","url(#arr)")})});
function nc(ev){if(ev.et==="USER")return{f:"#bbdefb",s:"#00ABFF"};if(ev.et==="ENV")return{f:"#b2dfdb",s:"#009688"};return{f:"#e8eaf6",s:"#5c6bc0"}}
var tt=d3.select("#dag-container").append("div").attr("class","dag-tooltip").style("display","none");
data.forEach(function(ev){var p=pos[ev.event_id];if(!p)return;var c=nc(ev);var lb=(ev.app?ev.app+".":"")+( ev.fn||"?");if(lb.length>24)lb=lb.slice(0,22)+"..";
var n=g.append("g").attr("class","dag-node").attr("transform","translate("+p.x+","+p.y+")");
n.append("rect").attr("width",W).attr("height",H).attr("fill",c.f).attr("stroke",c.s);
n.append("text").attr("x",8).attr("y",17).attr("fill","#334155").attr("font-weight","600").text(lb);
n.append("text").attr("x",8).attr("y",34).attr("fill","#64748b").attr("font-size","10px").text(ev.et);
n.on("mouseover",function(event){var as=Object.keys(ev.args||{}).map(function(k){return k+": "+ev.args[k]}).join("\\n");tt.style("display","block").html("<b>"+((ev.app?ev.app+".":"")+( ev.fn||"?"))+"</b>\\n"+(as||"(no args)"))}).on("mousemove",function(event){var r=ct.getBoundingClientRect();tt.style("left",(event.clientX-r.left+12)+"px").style("top",(event.clientY-r.top+12)+"px")}).on("mouseout",function(){tt.style("display","none")})});
var tns=data.filter(function(ev){return ev.fn==="send_message_to_user"||ev.fn==="send_message_to_agent"});
tns.forEach(function(tn,ti){var p=pos[tn.event_id];if(!p)return;var bx=p.x+W+px/2;var lb=tn.fn==="send_message_to_agent"?"User Task":"Turn "+(ti+1);g.append("line").attr("x1",bx).attr("y1",0).attr("x2",bx).attr("y2",svgH).attr("stroke","#cbd5e1").attr("stroke-width",1.5).attr("stroke-dasharray","6,4");g.append("text").attr("x",bx).attr("y",12).attr("text-anchor","middle").attr("fill","#94a3b8").attr("font-size","10px").attr("font-weight","600").text(lb)});
svg.call(d3.zoom().transform,d3.zoomIdentity);
})();
"""


def _build_dag_json(data: dict[str, Any]) -> str | None:
    events = data.get("events", [])
    if not events:
        return None

    # Build set of real event IDs
    real_ids = {ev["event_id"] for ev in events}

    # Resolve condition_* dependencies: conditions mark turn boundaries.
    # ENV events that depend on condition_turn_N depend on all turn N-1
    # completing. The turn-ending event is send_message_to_user/send_message_to_agent.
    # To avoid cycles (turn 1 AGENT events also lack dependents), only use
    # events that DON'T transitively depend on any ENV event as bridges.
    env_ids = {ev["event_id"] for ev in events if ev.get("event_type") == "ENV"}
    # BFS: find all events reachable from ENV events (turn 1+ events)
    dep_map = {ev["event_id"]: ev.get("dependencies", []) for ev in events}
    after_env: set[str] = set()
    for eid in real_ids:
        # Check if this event transitively depends on any ENV event
        visited: set[str] = set()
        stack = [eid]
        while stack:
            cur = stack.pop()
            if cur in visited:
                continue
            visited.add(cur)
            if cur in env_ids:
                after_env.add(eid)
                break
            for d in dep_map.get(cur, []):
                if d in real_ids:
                    stack.append(d)
    # Turn boundary events: AGENT events NOT after any ENV, not depended on by
    # other pre-ENV AGENT events
    pre_env_agents = [
        ev
        for ev in events
        if ev.get("event_type") == "AGENT" and ev["event_id"] not in after_env
    ]
    depended_on_by_pre_env: set[str] = set()
    for ev in pre_env_agents:
        for d in ev.get("dependencies", []):
            depended_on_by_pre_env.add(d)
    turn_boundary_ids = [
        ev["event_id"]
        for ev in pre_env_agents
        if ev["event_id"] not in depended_on_by_pre_env
    ]

    dag = []
    for ev in events:
        # Skip bare CONDITION nodes (no action, just markers)
        if ev.get("event_type") == "CONDITION":
            continue
        act = ev.get("action", {})
        args = {}
        if act:
            for a in act.get("args") or []:
                args[a["name"]] = _trunc(str(a.get("value", "")), 80)
        # Resolve dependencies: replace condition_* refs with AGENT leaf events
        deps = []
        for d in ev.get("dependencies", []):
            if d in real_ids:
                deps.append(d)
            elif d.startswith("condition_"):
                deps.extend(turn_boundary_ids)
            # else: skip unknown refs
        dag.append(
            {
                "event_id": ev["event_id"],
                "et": ev.get("event_type", ""),
                "deps": deps,
                "app": act.get("app") if act else None,
                "fn": act.get("function") if act else None,
                "args": args,
            }
        )
    return json.dumps(dag)


# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------

_CSS = """\
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:'Inter',-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;line-height:1.6;color:#1e293b;background:#f8fafc;-webkit-font-smoothing:antialiased}
a{color:#6366f1;text-decoration:none;font-weight:500}a:hover{color:#4f46e5}
.header{background:#0f172a;color:#fff;padding:16px 28px;display:flex;align-items:center;gap:16px;flex-wrap:wrap;position:sticky;top:0;z-index:100;box-shadow:0 2px 8px rgba(0,0,0,.15)}
.badge{display:inline-block;padding:4px 14px;border-radius:20px;font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.8px}
.badge-pass{background:#10b981;color:#fff}.badge-fail{background:#ef4444;color:#fff}.badge-error{background:#f97316;color:#fff}.badge-unknown{background:#94a3b8;color:#fff}
.stats{display:flex;gap:20px;font-size:12px;color:#94a3b8}.stats span{white-space:nowrap}
.nav-links{margin-left:auto;display:flex;gap:12px;font-size:12px}.nav-links a{color:#818cf8;font-weight:500}
.container{max-width:none;margin:0 auto;padding:24px 5vw}
.trace-layout{display:flex;gap:0;align-items:flex-start}
.trace-main{flex:3;min-width:0;padding-right:14px}
.trace-splitter{width:2px;cursor:col-resize;background:#e2e8f0;flex-shrink:0;align-self:stretch;min-height:200px;transition:background .15s;padding:0 3px;background-clip:content-box}
.trace-splitter:hover,.trace-splitter.active{background:#3b82f6;background-clip:content-box}
.trace-sidebar{flex:2;position:sticky;top:60px;max-height:calc(100vh - 80px);overflow-y:auto;padding-left:14px}
@media(max-width:1100px){.trace-layout{flex-direction:column}.trace-splitter{display:none}.trace-sidebar{position:static;flex:none;width:100%;max-height:none}}
.card{background:#fff;border-radius:16px;box-shadow:0 1px 3px rgba(0,0,0,.04),0 4px 16px rgba(0,0,0,.04);margin-bottom:20px;overflow:hidden;border:1px solid #e2e8f0}
.card-header{padding:14px 20px;background:#f8fafc;border-bottom:1px solid #e2e8f0;cursor:pointer;display:flex;align-items:center;gap:12px;user-select:none;position:sticky;top:52px;z-index:10;transition:background .15s}
.card-header:hover{background:#f1f5f9}.card-header .arrow{transition:transform .2s ease;font-size:11px;color:#94a3b8}
.card-header.open .arrow{transform:rotate(90deg)}.card-header .title{font-weight:600;font-size:14px;color:#334155}
.card-header .meta{font-size:12px;color:#94a3b8;margin-left:auto;display:flex;gap:16px}
.card-body{padding:24px;display:none}.card-header.open + .card-body{display:block}
.msg{margin-bottom:8px;padding:10px 14px;border-radius:8px;font-size:13px;white-space:pre-wrap;word-break:break-word;line-height:1.6;max-width:900px}
.msg-system{background:#fefce8;border-left:3px solid #eab308;color:#713f12}
.msg-user{background:#eff6ff;border-left:3px solid #3b82f6;color:#1e3a5f}
.msg-assistant{background:#f0fdf4;border-left:3px solid #22c55e;color:#14532d}
.msg-label{font-weight:700;font-size:10px;text-transform:uppercase;letter-spacing:.8px;margin-bottom:3px;color:#64748b}
.tool-pair{display:flex;margin:12px 0;border-radius:12px;overflow:hidden;border:1px solid #1e293b;min-height:42px}
.tool-pair .tool-exec{border:none;border-radius:0;margin:0;border-right:1px solid #334155;flex:0 0 auto;max-width:45%;min-width:200px;display:flex;align-items:flex-start}
.tool-pair .tool-output{border:none;border-radius:0;margin:0;flex:1;max-height:250px;overflow-y:auto}
.tool-exec{background:#0f172a;border-radius:12px;margin:12px 0;padding:12px 16px;font-family:'SF Mono','JetBrains Mono','Fira Code','Consolas',monospace;font-size:13px;color:#4ade80;border:1px solid #1e293b}
.tool-output{background:#0f172a;border-radius:12px;margin:0 0 12px;padding:14px 16px;font-family:'SF Mono','JetBrains Mono','Fira Code','Consolas',monospace;font-size:12px;color:#94a3b8;border:1px solid #1e293b;max-height:250px;overflow-y:auto;white-space:pre-wrap;word-break:break-word;line-height:1.6}
.env-pair{display:flex;margin:12px 0;border-radius:12px;overflow:hidden;border:1px solid #f59e0b;min-height:42px}
.env-pair .env-exec{flex:0 0 auto;max-width:50%;min-width:200px;padding:12px 16px;background:#fffbeb;border-right:1px solid #f59e0b;font-family:monospace;font-size:13px;color:#78350f;word-break:break-all}
.env-pair .env-output{flex:1;padding:12px 16px;background:#fffef5;max-height:250px;overflow-y:auto;font-family:monospace;font-size:13px;color:#374151;white-space:pre-wrap;word-break:break-word}
.env-label{font-size:10px;font-weight:700;color:#b45309;margin-bottom:4px}
.thinking-details{margin-bottom:16px;max-width:900px}
.thinking-summary{cursor:pointer;padding:10px 16px;background:#fefce8;border-left:3px solid #eab308;border-radius:12px;font-weight:600;font-size:11px;text-transform:uppercase;letter-spacing:.6px;color:#92400e;list-style:none;display:flex;align-items:center;gap:8px;transition:background .15s}
.thinking-summary:hover{background:#fef9c3}
.thinking-summary::before{content:'\\25B6';font-size:8px;transition:transform .2s}
.thinking-details[open] .thinking-summary::before{transform:rotate(90deg)}
.thinking-summary::-webkit-details-marker{display:none}
.thinking-content{padding:14px 16px;background:#fefce8;border-left:3px solid #eab308;border-radius:0 0 12px 12px;font-size:13px;white-space:pre-wrap;word-break:break-word;line-height:1.7;color:#713f12;margin-top:-12px}
.turn-divider{display:flex;align-items:center;gap:14px;margin:20px 0 16px;color:#94a3b8;font-size:10px;font-weight:600;letter-spacing:.5px;text-transform:uppercase}
.turn-divider::before,.turn-divider::after{content:'';flex:1;border-top:1px dashed #cbd5e1}
.turn-divider .turn-info{white-space:nowrap;display:flex;gap:12px;background:#f1f5f9;padding:4px 14px;border-radius:20px}
.section-title{font-size:13px;font-weight:700;margin:32px 0 16px;padding-bottom:10px;border-bottom:2px solid #e2e8f0;color:#475569;letter-spacing:.5px;text-transform:uppercase}
.rationale{background:#f8fafc;border-radius:12px;padding:16px;font-size:13px;white-space:pre-wrap;word-break:break-word;border:1px solid #e2e8f0;line-height:1.7;color:#334155;margin-bottom:12px}
.system-details summary{background:#fefce8;border-left:3px solid #eab308;color:#92400e;border-radius:12px}
.final-answer{border:2px solid #22c55e;border-radius:12px;padding:14px;margin:12px 0;background:#f0fdf4}
.error-block{background:#fef2f2;border-left:3px solid #ef4444;border-radius:8px;padding:12px 16px;margin:8px 0;font-size:13px;color:#991b1b}
.error-block .error-title{font-weight:700;margin-bottom:4px}
.error-block .error-detail{font-family:monospace;font-size:12px;color:#b91c1c;white-space:pre-wrap;word-break:break-word}
.hint-block{background:#fffbeb;border-left:3px solid #f59e0b;border-radius:8px;padding:10px 14px;margin:8px 0;font-size:13px;color:#92400e}
.subagent-block{margin:12px 0;padding:8px 0 8px 16px;border-left:3px solid #8b5cf6}
.subagent-label{font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.8px;color:#7c3aed;margin-bottom:8px}
.summary{background:#fff;border-radius:16px;box-shadow:0 1px 3px rgba(0,0,0,.04),0 4px 16px rgba(0,0,0,.04);padding:28px 32px;margin-bottom:28px;display:flex;gap:40px;flex-wrap:wrap;border:1px solid #e2e8f0}
.summary .stat{text-align:center}.summary .stat .value{font-size:32px;font-weight:800;color:#0f172a;letter-spacing:-1.5px}
.summary .stat .label{font-size:10px;color:#94a3b8;text-transform:uppercase;letter-spacing:1px;font-weight:600;margin-top:4px}
table{width:100%;border-collapse:collapse;font-size:13px}
th{background:#f8fafc;padding:12px 16px;text-align:left;border-bottom:2px solid #e2e8f0;cursor:pointer;white-space:nowrap;user-select:none;font-weight:600;color:#475569;font-size:11px;letter-spacing:.5px;text-transform:uppercase}
th:hover{background:#f1f5f9}td{padding:12px 16px;border-bottom:1px solid #f1f5f9;color:#334155}
tr:hover td{background:#f8fafc}
.table-wrap{overflow-x:auto}
.breadcrumb{display:flex;align-items:center;gap:6px;font-size:12px;color:#94a3b8;margin-right:auto}
.breadcrumb a{color:#818cf8;font-weight:500}.breadcrumb .sep{color:#475569}
.nav-arrow{position:fixed;top:50%;transform:translateY(-50%);z-index:200;width:44px;height:44px;border-radius:50%;background:#0f172a;color:#e2e8f0;display:flex;align-items:center;justify-content:center;font-size:20px;text-decoration:none;box-shadow:0 2px 8px rgba(0,0,0,.2);transition:background .15s,transform .15s}
.nav-arrow:hover{background:#1e293b;transform:translateY(-50%) scale(1.1);color:#fff}
.nav-arrow-left{left:12px}.nav-arrow-right{right:12px}
.filters{display:flex;gap:12px;margin-bottom:20px;align-items:center}
.filters select,.filters input{padding:10px 14px;border:1px solid #e2e8f0;border-radius:10px;font-size:13px;background:#fff;color:#334155}
.filters input{flex:1;max-width:300px}.filters input::placeholder{color:#94a3b8}
::-webkit-scrollbar{width:6px;height:6px}::-webkit-scrollbar-track{background:transparent}::-webkit-scrollbar-thumb{background:#cbd5e1;border-radius:3px}
"""

_JS = """\
document.querySelectorAll('.card-header').forEach(h=>{h.addEventListener('click',()=>{h.classList.toggle('open')})});
(function(){var execs=document.querySelectorAll('.tool-exec:not(.tool-pair .tool-exec)');execs.forEach(function(ex){var next=ex.nextElementSibling;if(next&&next.classList.contains('tool-output')){var pair=document.createElement('div');pair.className='tool-pair';ex.parentNode.insertBefore(pair,ex);pair.appendChild(ex);pair.appendChild(next)}})})();
document.addEventListener('keydown',function(e){if(e.target.tagName==='INPUT'||e.target.tagName==='TEXTAREA')return;var p=document.getElementById('nav-prev'),n=document.getElementById('nav-next');if(e.key==='ArrowLeft'&&p)window.location=p.href;if(e.key==='ArrowRight'&&n)window.location=n.href;if(e.key==='Escape'){var ov=document.getElementById('jl-compare-overlay');if(ov&&ov.style.display==='flex')jlCloseCompare(null)}});
(function(){var sp=document.querySelector('.trace-splitter');if(!sp)return;var main=sp.previousElementSibling,side=sp.nextElementSibling,drag=false;sp.addEventListener('mousedown',function(e){drag=true;sp.classList.add('active');e.preventDefault()});document.addEventListener('mousemove',function(e){if(!drag)return;var r=sp.parentElement.getBoundingClientRect(),p=(e.clientX-r.left)/r.width*100;p=Math.max(20,Math.min(80,p));main.style.flex='0 0 '+p+'%';side.style.flex='0 0 '+(100-p-1)+'%'});document.addEventListener('mouseup',function(){if(drag){drag=false;sp.classList.remove('active')}})})();

/* HTML escape (DOM-based, handles UTF-8 properly) */
function esc(s){var d=document.createElement('div');d.textContent=s;return d.innerHTML}

/* Judge log: compare events */
var _jlSelectedAgent=null;
var _jlSelectedOracle=null;

function _jlClearSelection(cls){
  document.querySelectorAll('.'+cls+'.jl-selected').forEach(function(el){
    el.classList.remove('jl-selected');el.style.outline='';
    var r=el.querySelector('input[type=radio]');if(r)r.checked=false;
  });
}
function jlSelectAgent(el){
  if(_jlSelectedAgent===el){
    el.classList.remove('jl-selected');el.style.outline='';
    var r=el.querySelector('input[type=radio]');if(r)r.checked=false;
    _jlSelectedAgent=null;
  }else{
    _jlClearSelection('jl-agent-ev');
    el.classList.add('jl-selected');el.style.outline='2px solid #6366f1';
    var r=el.querySelector('input[type=radio]');if(r)r.checked=true;
    _jlSelectedAgent=el;
  }
  _jlTryCompare();
}
function jlSelectOracle(el){
  if(_jlSelectedOracle===el){
    el.classList.remove('jl-selected');el.style.outline='';
    var r=el.querySelector('input[type=radio]');if(r)r.checked=false;
    _jlSelectedOracle=null;
  }else{
    _jlClearSelection('jl-oracle-ev');
    el.classList.add('jl-selected');el.style.outline='2px solid #6366f1';
    var r=el.querySelector('input[type=radio]');if(r)r.checked=true;
    _jlSelectedOracle=el;
  }
  _jlTryCompare();
}
function jlCloseCompare(e){
  if(e&&e.target!==document.getElementById('jl-compare-overlay'))return;
  var ov=document.getElementById('jl-compare-overlay');
  if(ov)ov.style.display='none';
  _jlClearSelection('jl-agent-ev');
  _jlClearSelection('jl-oracle-ev');
  _jlSelectedAgent=null;_jlSelectedOracle=null;
}
function _jlTryCompare(){
  var overlay=document.getElementById('jl-compare-overlay');
  if(!overlay)return;
  if(!_jlSelectedAgent||!_jlSelectedOracle){overlay.style.display='none';return;}
  var aTool=_jlSelectedAgent.getAttribute('data-tool');
  var oTool=_jlSelectedOracle.getAttribute('data-tool');
  var aArgs,oArgs;
  try{aArgs=JSON.parse(_jlSelectedAgent.getAttribute('data-args')||'{}');}catch(e){aArgs={};}
  try{oArgs=JSON.parse(_jlSelectedOracle.getAttribute('data-args')||'{}');}catch(e){oArgs={};}
  /* Look up rejection reason and judge output for this agent/oracle pair */
  var reason='',judgeOut='';
  var rejectionsRaw=_jlSelectedOracle.getAttribute('data-rejections');
  if(rejectionsRaw){
    try{
      var rejections=JSON.parse(rejectionsRaw);
      var agentEid=_jlSelectedAgent.getAttribute('data-event-id')||'';
      if(agentEid&&rejections[agentEid]){
        var entry=rejections[agentEid];
        reason=entry.reason||entry||'';
        judgeOut=entry.judge_output||'';
      }
    }catch(e){}
  }
  _showCompare(aTool,aArgs,oTool,oArgs,false,judgeOut,reason);
}
function jlOpenMatched(el){
  var oTool=el.getAttribute('data-oracle-tool')||'?';
  var aTool=el.getAttribute('data-agent-tool')||'?';
  var oArgs,aArgs;
  try{oArgs=JSON.parse(el.getAttribute('data-oracle-args')||'{}');}catch(e){oArgs={};}
  try{aArgs=JSON.parse(el.getAttribute('data-agent-args')||'{}');}catch(e){aArgs={};}
  var judgeOutput=el.getAttribute('data-judge-output')||'';
  _showCompare(aTool,aArgs,oTool,oArgs,true,judgeOutput,'');
}
function _showCompare(aTool,aArgs,oTool,oArgs,isMatched,judgeOutput,rejectReason){
  var overlay=document.getElementById('jl-compare-overlay');
  var modal=document.getElementById('jl-compare-modal');
  if(!overlay||!modal)return;
  var titleBadge=isMatched?' <span style="font-size:11px;padding:2px 8px;background:#dcfce7;color:#166534;border-radius:6px;font-weight:600;margin-left:8px">MATCHED</span>':'';
  var html='<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:16px">'
    +'<div style="font-weight:700;font-size:15px;color:#0f172a">Event Comparison'+titleBadge+'</div>'
    +'<button onclick="jlCloseCompare(null)" style="background:none;border:none;font-size:20px;color:#94a3b8;cursor:pointer;padding:4px 8px;line-height:1">&times;</button></div>';
  if(aTool!==oTool){
    html+='<div style="color:#991b1b;font-size:12px;margin-bottom:12px;padding:8px 12px;background:#fef2f2;border-radius:8px">Different tools: <b>'+esc(aTool)+'</b> vs <b>'+esc(oTool)+'</b></div>';
  }
  html+='<div style="font-size:13px;color:#475569;margin-bottom:12px"><b>'+esc(oTool)+'</b></div>';
  var allKeys={};
  for(var k in aArgs)allKeys[k]=true;
  for(var k in oArgs)allKeys[k]=true;
  html+='<table style="width:100%;border-collapse:collapse;font-size:12px">';
  html+='<tr style="border-bottom:2px solid #e2e8f0"><th style="text-align:left;padding:6px 10px;color:#64748b;font-size:10px;text-transform:uppercase;letter-spacing:.5px;width:20%">Param</th><th style="text-align:left;padding:6px 10px;color:#166534;font-size:10px;text-transform:uppercase;letter-spacing:.5px;width:40%">Agent</th><th style="text-align:left;padding:6px 10px;color:#475569;font-size:10px;text-transform:uppercase;letter-spacing:.5px;width:40%">Oracle</th></tr>';
  for(var k in allKeys){
    var av=k in aArgs?String(aArgs[k]):'';
    var ov=k in oArgs?String(oArgs[k]):'';
    var same=av===ov;
    var bg=same?'#f0fdf4':(isMatched?'#fffbeb':'#fef2f2');
    var diffColor=isMatched?'#92400e':'#991b1b';
    html+='<tr style="background:'+bg+';border-bottom:1px solid #f1f5f9">';
    html+='<td style="padding:6px 10px;color:#64748b;font-weight:600;vertical-align:top;font-family:monospace;font-size:11px">'+esc(k)+'</td>';
    html+='<td style="padding:6px 10px;word-break:break-word;white-space:pre-wrap;font-family:monospace;font-size:11px;color:'+(same?'#166534':diffColor)+'">'+esc(av||'(missing)')+'</td>';
    html+='<td style="padding:6px 10px;word-break:break-word;white-space:pre-wrap;font-family:monospace;font-size:11px;color:#334155">'+esc(ov||'(missing)')+'</td>';
    html+='</tr>';
  }
  html+='</table>';
  if(judgeOutput){
    html+='<div style="margin-top:14px;padding:10px 12px;background:#f5f3ff;border-radius:8px;border:1px solid #e9d5ff">'
      +'<div style="font-size:10px;font-weight:700;color:#7c3aed;text-transform:uppercase;letter-spacing:.5px;margin-bottom:6px">Judge Rationale</div>'
      +'<div style="font-size:12px;color:#4c1d95;white-space:pre-wrap;word-break:break-word;line-height:1.6">'+esc(judgeOutput)+'</div></div>';
  }
  if(rejectReason){
    html+='<div style="margin-top:12px;padding:10px 14px;background:#fef2f2;border-left:3px solid #ef4444;border-radius:8px">'
      +'<div style="font-size:10px;font-weight:700;color:#991b1b;text-transform:uppercase;letter-spacing:.5px;margin-bottom:6px">Rejection Reason</div>'
      +'<div style="font-size:12px;color:#991b1b;white-space:pre-wrap;word-break:break-word;line-height:1.6">'+esc(rejectReason)+'</div></div>';
  }
  html+='<div style="margin-top:12px;font-size:10px;color:#94a3b8;text-align:center">Press Esc or click outside to close</div>';
  modal.innerHTML=html;
  overlay.style.display='flex';
}
"""

_INDEX_JS = """\
var sortDir={};function sortTable(c){var t=document.getElementById('scenarioTable');if(!t)return;var tb=t.querySelector('tbody'),rows=Array.from(tb.querySelectorAll('tr')),d=sortDir[c]||'asc';rows.sort(function(a,b){var av=a.cells[c].textContent.trim(),bv=b.cells[c].textContent.trim(),an=parseFloat(av.replace(/[^0-9.\\-]/g,'')),bn=parseFloat(bv.replace(/[^0-9.\\-]/g,''));if(!isNaN(an)&&!isNaN(bn))return d==='asc'?an-bn:bn-an;return d==='asc'?av.localeCompare(bv):bv.localeCompare(av)});rows.forEach(function(r){tb.appendChild(r)});sortDir[c]=d==='asc'?'desc':'asc'}
function filterTable(){var q=(document.getElementById('searchInput')||{}).value||'';q=q.toLowerCase();document.querySelectorAll('#scenarioTable tbody tr').forEach(function(r){r.style.display=r.textContent.toLowerCase().includes(q)?'':'none'})}
function filterStatus(){var s=(document.getElementById('statusFilter')||{}).value||'all';document.querySelectorAll('#scenarioTable tbody tr').forEach(function(r){if(s==='all'){r.style.display='';return}r.style.display=r.dataset.status===s?'':'none'})}
"""


# ---------------------------------------------------------------------------
# Page generators
# ---------------------------------------------------------------------------


def _card(title: str, body: str, meta: str = "", open: bool = False) -> str:
    oc = " open" if open else ""
    ms = f'<span class="meta">{meta}</span>' if meta else ""
    return (
        f'<div class="card"><div class="card-header{oc}"><span class="arrow">&#x25B6;</span>'
        f'<span class="title">{_esc(title)}</span>{ms}</div><div class="card-body">{body}</div></div>'
    )


def generate_trace_page(
    data: dict[str, Any],
    *,
    prev_url: str | None = None,
    next_url: str | None = None,
    home_url: str = "../index.html",
) -> str:
    sid = _sid(data)
    verdict = _validation(data)
    model = _model(data)
    wls = data.get("_world_logs", [])

    # Token stats
    llm_calls = sum(
        1
        for w in wls
        if w.get("log_type") in ("llm_output", "llm_output_thought_action")
    )
    total_pt = sum(
        w.get("prompt_tokens", 0)
        for w in wls
        if w.get("log_type") in ("llm_output", "llm_output_thought_action")
    )
    total_ct = sum(
        w.get("completion_tokens", 0)
        for w in wls
        if w.get("log_type") in ("llm_output", "llm_output_thought_action")
    )

    p: list[str] = []
    # Header
    p.append(
        f'<div class="header"><nav class="breadcrumb"><a href="{_esc(home_url)}">Home</a><span class="sep">/</span><span>{_esc(sid)}</span></nav>'
    )
    p.append(_badge(verdict))
    p.append('<div class="stats">')
    if llm_calls:
        p.append(f"<span>{llm_calls} LLM calls</span>")
    if total_pt + total_ct:
        p.append(f"<span>{_fmt_tok(total_pt + total_ct)} tokens</span>")
    if model:
        p.append(f"<span>{_esc(model)}</span>")
    p.append("</div>")
    if prev_url or next_url:
        p.append('<div class="nav-links">')
        if prev_url:
            p.append(f'<a href="{_esc(prev_url)}">&#x2190; Prev</a>')
        if next_url:
            p.append(f'<a href="{_esc(next_url)}">Next &#x2192;</a>')
        p.append("</div>")
    p.append("</div>")

    # Two-column layout
    p.append('<div class="container"><div class="trace-layout">')
    p.append('<div class="trace-main"><h2 class="section-title">Conversation</h2>')
    p.append(_render_conversation(wls))
    p.append("</div>")
    p.append('<div class="trace-splitter"></div>')
    p.append('<div class="trace-sidebar">')

    # Judge result card
    p.append(_card("Judge Result", _render_judge(data), open=True))

    # DAG card
    dag_json = _build_dag_json(data)
    if dag_json:
        legend = (
            '<div class="dag-legend">'
            '<div class="dag-legend-item"><div class="dag-swatch" style="background:#bbdefb;border-color:#00ABFF"></div>USER</div>'
            '<div class="dag-legend-item"><div class="dag-swatch" style="background:#b2dfdb;border-color:#009688"></div>ENV</div>'
            '<div class="dag-legend-item"><div class="dag-swatch" style="background:#e8eaf6;border-color:#5c6bc0"></div>AGENT</div>'
            '</div><div id="dag-container"></div>'
        )
        p.append(_card("Oracle DAG", legend, open=True))

    p.append("</div></div></div>")  # sidebar, layout, container

    # Nav arrows
    if prev_url:
        p.append(
            f'<a id="nav-prev" class="nav-arrow nav-arrow-left" href="{_esc(prev_url)}">&#x2190;</a>'
        )
    if next_url:
        p.append(
            f'<a id="nav-next" class="nav-arrow nav-arrow-right" href="{_esc(next_url)}">&#x2192;</a>'
        )

    # Comparison modal overlay
    p.append(
        '<div id="jl-compare-overlay" onclick="jlCloseCompare(event)" '
        'style="display:none;position:fixed;inset:0;z-index:1000;background:rgba(0,0,0,.5);'
        'align-items:center;justify-content:center">'
        '<div id="jl-compare-modal" style="background:#fff;border-radius:16px;max-width:900px;'
        "width:90%;max-height:80vh;overflow-y:auto;padding:24px;"
        'box-shadow:0 8px 32px rgba(0,0,0,.3)"></div></div>'
    )

    # Assemble
    js = _JS
    extra_css = ""
    if dag_json:
        js += f"\nwindow.__DAG_DATA={dag_json};\n" + _DAG_JS
        extra_css = _DAG_CSS

    body = "\n".join(p)
    return (
        f'<!DOCTYPE html><html><head><meta charset="utf-8"><title>{_esc(sid)} — Gaia2 Trace</title>'
        f"<style>{_CSS}\n{extra_css}</style>"
        f'<script src="https://d3js.org/d3.v7.min.js"></script>'
        f'</head><body class="has-trace">{body}<script>{js}</script></body></html>'
    )


def generate_index_page(
    traces: list[tuple[str, dict[str, Any]]], title: str = "Gaia2 Traces"
) -> str:
    total = len(traces)
    passed = sum(
        1
        for _, d in traces
        if _validation(d) and _validation(d).lower() in ("valid", "pass")
    )
    failed = total - passed
    rate = (passed / total * 100) if total else 0

    p: list[str] = []
    p.append(
        f'<div class="header"><nav class="breadcrumb"><span>Home</span></nav>'
        f'<span style="color:#94a3b8;font-size:13px">{total} scenarios</span></div>'
    )
    p.append('<div class="container" style="max-width:1400px">')
    p.append(
        f'<div class="summary"><div class="stat"><div class="value">{total}</div><div class="label">Scenarios</div></div>'
        f'<div class="stat"><div class="value" style="color:#2b8a3e">{passed}</div><div class="label">Passed</div></div>'
        f'<div class="stat"><div class="value" style="color:#c92a2a">{failed}</div><div class="label">Failed</div></div>'
        f'<div class="stat"><div class="value">{rate:.1f}%</div><div class="label">Pass Rate</div></div></div>'
    )
    p.append(
        '<div class="filters"><select id="statusFilter" onchange="filterStatus()"><option value="all">All</option><option value="pass">Pass</option><option value="fail">Fail</option></select>'
        '<input type="text" id="searchInput" placeholder="Search..." oninput="filterTable()"></div>'
    )
    p.append('<div class="table-wrap"><table id="scenarioTable"><thead><tr>')
    for i, col in enumerate(
        ["Scenario", "Status", "Model", "LLM Calls", "Oracle Events"]
    ):
        p.append(
            f'<th onclick="sortTable({i})">{col} <span class="sort-arrow"></span></th>'
        )
    p.append("</tr></thead><tbody>")
    for href, d in traces:
        sid = _sid(d)
        v = _validation(d)
        ds = "pass" if v and v.lower() in ("valid", "pass") else "fail"
        wls = d.get("_world_logs", [])
        llm = sum(
            1
            for w in wls
            if w.get("log_type") in ("llm_output", "llm_output_thought_action")
        )
        n_oracle = len(
            [e for e in d.get("events", []) if e.get("event_type") == "AGENT"]
        )
        p.append(
            f'<tr data-status="{ds}"><td><a href="{_esc(href)}">{_esc(sid)}</a></td><td>{_badge(v)}</td>'
            f"<td>{_esc(_model(d))}</td><td>{llm}</td><td>{n_oracle}</td></tr>"
        )
    p.append("</tbody></table></div></div>")
    return (
        f'<!DOCTYPE html><html><head><meta charset="utf-8"><title>{_esc(title)}</title>'
        f"<style>{_CSS}</style></head><body>{''.join(p)}<script>{_INDEX_JS}</script></body></html>"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _generate_all(traces, out, title):
    entries = []
    for i, (path, data) in enumerate(traces):
        sid = _sid(data)
        prev_url = f"../{_sid(traces[i - 1][1])}/trace.html" if i > 0 else None
        next_url = (
            f"../{_sid(traces[i + 1][1])}/trace.html" if i < len(traces) - 1 else None
        )
        html = generate_trace_page(data, prev_url=prev_url, next_url=next_url)
        d = out / sid
        d.mkdir(parents=True, exist_ok=True)
        (d / "trace.html").write_text(html, encoding="utf-8")
        entries.append((f"{sid}/trace.html", data))
    (out / "index.html").write_text(
        generate_index_page(entries, title=title), encoding="utf-8"
    )
    return entries


def main():
    import sys
    import webbrowser

    args = sys.argv[1:]
    if not args or args[0] in ("-h", "--help"):
        print("Usage: python -m gaia2_core.trace_viewer <command> [options]")
        print("Commands:")
        print("  view <trace.json>       View a single trace")
        print("  view-dir <directory>    View all traces in a directory")
        print("  serve <directory> [-p PORT]  Serve with live regen")
        return

    cmd = args[0]

    if cmd == "view":
        path = Path(args[1])
        data = load_trace(path)
        html = generate_trace_page(data)
        out = path.with_suffix(".html")
        out.write_text(html, encoding="utf-8")
        print(f"Generated: {out}")
        webbrowser.open(f"file://{out.resolve()}")

    elif cmd == "view-dir":
        d = Path(args[1])
        out = Path(args[3]) if len(args) > 3 and args[2] == "-o" else d / "_viewer"
        out.mkdir(parents=True, exist_ok=True)
        traces = load_traces(d)
        if not traces:
            print(f"No traces found in {d}")
            return
        print(f"Found {len(traces)} trace(s)")
        _generate_all(traces, out, title=f"Gaia2 Traces — {d.name}")
        print(f"Generated {len(traces)} trace page(s) at {out}")

    elif cmd == "serve":
        import threading
        import time
        from http.server import HTTPServer, SimpleHTTPRequestHandler

        d = Path(args[1])
        port = int(args[3]) if len(args) > 3 and args[2] == "-p" else None
        out = d / "_viewer"
        out.mkdir(parents=True, exist_ok=True)

        # If already has index.html, serve static
        if (d / "index.html").exists():
            out = d

        traces = load_traces(d)
        if traces:
            print(f"Generating {len(traces)} trace(s)...")
            _generate_all(traces, out, title=f"Gaia2 Traces — {d.name}")
        else:
            (out / "index.html").write_text(
                '<html><body><h2>Waiting for traces...</h2><meta http-equiv="refresh" content="10"></body></html>'
            )

        def regen():
            n = len(traces)
            while True:
                time.sleep(15)
                try:
                    cur = load_traces(d)
                    if len(cur) != n:
                        print(f"Regenerating: {len(cur)} traces")
                        _generate_all(cur, out, title=f"Gaia2 Traces — {d.name}")
                        n = len(cur)
                except Exception:
                    pass

        threading.Thread(target=regen, daemon=True).start()

        class DS(HTTPServer):
            address_family = socket.AF_INET6

            def server_bind(self):
                self.socket.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 0)
                super().server_bind()

        class Q(SimpleHTTPRequestHandler):
            def log_message(self, *a):
                pass

        os.chdir(str(out))
        ports = [port] if port else list(range(44100, 44110))
        srv = None
        for p in ports:
            try:
                srv = DS(("::", p), Q)
                port = p
                break
            except OSError:
                continue
        if not srv:
            print("No available port")
            return

        host = socket.getfqdn() or socket.gethostname()
        print(
            f"\n  http://{host}:{port}\n  http://localhost:{port}\n\n  Ctrl+C to stop\n"
        )
        try:
            srv.serve_forever()
        except KeyboardInterrupt:
            print("\nStopped")


if __name__ == "__main__":
    main()
