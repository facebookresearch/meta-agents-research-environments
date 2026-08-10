# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
"""Replay a completed scenario's events.jsonl through Judge with a swappable engine.

Reproduces exactly what the live daemon (eventd.py) does at judgment time,
without re-executing ENV reactions. Reads the recorded action stream, groups
into turns on send_message_to_user boundaries, and calls Judge.judge_turn
per turn. The Judge instance is built with the same args daemon uses
(_create_judge in eventd.py) so behavior is byte-identical modulo the
swapped engine.

Ground truth for validation: pointing at the original judge engine should
reproduce (up to LLM nondeterminism) the daemon_judgments.jsonl already on
disk. A judge with native reasoning (e.g. gpt-oss-120b) is a stable target
for this check.

Usage::

    python -m gaia2_cli.judge.rejudge \\
        --scenario-dir <path-with-events.jsonl> \\
        --out /tmp/replay/replay_judgments.jsonl \\
        --judge-model gpt-oss-120b \\
        --judge-provider openai-compat \\
        --judge-base-url http://<host>:<port>/v1 \\
        --judge-api-key <key> \\
        --judge-prompt-version omnigaia
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path


def _load_events_jsonl(path: Path) -> list[dict]:
    entries = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def _resolve_scenario_json(scenario_dir: Path) -> Path:
    """result.json records the absolute scenario_file used at run time."""
    result = json.loads((scenario_dir / "result.json").read_text())
    sf = result.get("scenario_file")
    if not sf:
        raise SystemExit("result.json missing scenario_file field")
    p = Path(sf)
    if not p.exists():
        raise SystemExit(f"scenario_file does not exist: {p}")
    return p


def _build_judge(
    scenario_json: Path,
    state_dir: Path,
    replay_state_dir: Path,
    judge_model: str | None,
    judge_provider: str | None,
    judge_base_url: str | None,
    judge_api_key: str | None,
    judge_prompt_version: str | None,
    judge_extra_body: dict | None = None,
):
    """Mirror eventd._create_judge without side effects (no ENV dispatch).

    Deliberately duplicated rather than shared: eventd._create_judge reads its
    config off ``self`` and writes judgments into a single ``state_dir``, while
    replay needs a distinct read dir and write dir. Extracting a common helper
    would mean reshaping the daemon's judge setup for the benefit of an offline
    tool, so keep the two in sync by hand — if you change one, check the other.

    state_dir         : original trajectory dir — read user_details.json etc.
    replay_state_dir  : where Judge is allowed to write judgments.jsonl (its
                        own side effect). Kept distinct so we never mutate
                        the original trajectory dir.
    """
    from gaia2_core.event_loop import EventProcessor
    from gaia2_core.loader import ScenarioLoader
    from gaia2_core.types import UserDetails

    from gaia2_cli.judge import Judge, create_litellm_engine

    loader = ScenarioLoader(str(scenario_json))
    processor = EventProcessor(
        events=loader.events,
        start_time=loader.start_time,
        duration=loader.duration,
        time_increment=loader.time_increment,
        app_name_to_class=loader.app_name_to_class,
    )
    processor.build_turn_triggers()

    oracle_data = loader.extract_oracle_data(
        event_id_to_turn_idx=processor.event_id_to_turn_idx,
        nb_turns=processor.nb_turns,
    )
    turn_oracle_events, turn_oracle_graph, tasks, _ = oracle_data

    user_details = None
    ud_path = state_dir / "user_details.json"
    if ud_path.exists():
        ud = json.loads(ud_path.read_text())
        user_details = UserDetails(
            first_name=ud.get("first_name", ""),
            last_name=ud.get("last_name", ""),
            address=ud.get("address", ""),
        )

    total_oracle = sum(len(t) for t in turn_oracle_events)
    if total_oracle == 0:
        return None, processor.nb_turns

    engine = None
    if judge_model:
        engine = create_litellm_engine(
            model=judge_model,
            provider=judge_provider,
            base_url=judge_base_url,
            validate=False,
            api_key=judge_api_key,
            extra_body=judge_extra_body,
        )

    prompt_overrides = None
    if judge_prompt_version:
        from gaia2_core.judge.prompt_overrides import resolve_prompt_overrides

        prompt_overrides = resolve_prompt_overrides(judge_prompt_version)

    judge = Judge(
        turn_to_oracle_events=turn_oracle_events,
        turn_to_oracle_graph=turn_oracle_graph,
        tasks=tasks,
        user_details=user_details,
        start_time=loader.start_time,
        engine=engine,
        app_name_to_class=loader.app_name_to_class,
        state_dir=str(replay_state_dir),
        prompt_overrides=prompt_overrides,
    )
    return judge, processor.nb_turns


def _split_turns(events: list[dict]) -> list[list[dict]]:
    """Split events.jsonl entries into turns on send_message_to_user boundaries.

    A turn ends AT the send_message_to_user event (inclusive). Mirrors the
    live daemon: judge_turn is called with all agent events up to and
    including the SMU boundary. Trailing entries without an SMU are an
    incomplete final turn — dropped, matching daemon behavior.
    """
    turns: list[list[dict]] = []
    cur: list[dict] = []
    for e in events:
        cur.append(e)
        if (
            e.get("app") == "AgentUserInterface"
            and e.get("fn") == "send_message_to_user"
        ):
            turns.append(cur)
            cur = []
    return turns


def _to_completed_event(entry: dict):
    """Convert events.jsonl entry to CompletedEvent.

    Mirrors eventd._add_events_to_processor: uses sim_t as event_time when
    present (so times align with oracle event_time epoch), derives
    event_type from raw event_id prefix.
    """
    from gaia2_core.types import CompletedEvent, EventAction

    from gaia2_cli.daemon.eventd import _action_to_event_dict

    if "app" not in entry or "fn" not in entry:
        return None

    event_time = entry["t"]
    sim_t = entry.get("sim_t")
    if sim_t:
        try:
            dt = datetime.strptime(sim_t, "%Y-%m-%d %H:%M:%S")
            event_time = dt.replace(tzinfo=timezone.utc).timestamp()
        except (ValueError, TypeError):
            pass

    raw_event_id = str(
        entry.get("event_id") or _action_to_event_dict(entry)["event_id"]
    )
    if raw_event_id.startswith("Event-ENV-"):
        event_type = "ENV"
    elif raw_event_id.startswith("Event-USER-"):
        event_type = "USER"
    else:
        event_type = "AGENT"

    return CompletedEvent(
        event_id=raw_event_id,
        event_type=event_type,
        event_time=event_time,
        action=EventAction(
            app_name=entry["app"],
            class_name=entry["app"],
            function_name=entry["fn"],
            args=entry.get("args", {}),
            operation_type="write" if entry.get("w") else "read",
        ),
        return_value=entry.get("ret"),
    )


def rejudge(scenario_dir: Path, out_path: Path, judge_kwargs: dict) -> dict:
    """Replay one scenario. Returns a summary dict; also writes out_path."""
    events = _load_events_jsonl(scenario_dir / "events.jsonl")
    scenario_json = _resolve_scenario_json(scenario_dir)

    # Judge writes judgments.jsonl into whatever state_dir we hand it. Point
    # it at the parent of out_path (never the original scenario dir) so we
    # cannot mutate on-disk trajectory artifacts.
    replay_state_dir = out_path.parent
    replay_state_dir.mkdir(parents=True, exist_ok=True)

    judge, _nb_turns = _build_judge(
        scenario_json=scenario_json,
        state_dir=scenario_dir,
        replay_state_dir=replay_state_dir,
        **judge_kwargs,
    )
    if judge is None:
        out_path.write_text("")
        return {"scenario_id": scenario_dir.name, "no_oracle_events": True}

    turns = _split_turns(events)
    logging.info(
        "Replaying %d turns (%d total events) against %s",
        len(turns),
        len(events),
        judge_kwargs.get("judge_model") or "no-engine",
    )

    results = []
    with out_path.open("w") as f:
        for turn_idx, turn_events in enumerate(turns):
            completed = [
                ev
                for ev in (_to_completed_event(e) for e in turn_events)
                if ev is not None
            ]
            result = judge.judge_turn(turn_idx=turn_idx, agent_events=completed)
            rec = {
                "turn": turn_idx,
                "success": result.success,
                "failure_reason": result.failure_reason,
            }
            f.write(json.dumps(rec) + "\n")
            results.append(rec)

    return {
        "scenario_id": scenario_dir.name,
        "num_turns_replayed": len(results),
        "success": all(r["success"] for r in results),
        "results": results,
    }


def _diff_vs_daemon(scenario_dir: Path, replay_results: list[dict]) -> dict:
    """Compare replayed turn verdicts to the on-disk daemon_judgments.jsonl."""
    dj = scenario_dir / "daemon_judgments.jsonl"
    if not dj.exists():
        return {"note": "no daemon_judgments.jsonl on disk (nothing to compare)"}
    orig = [json.loads(line) for line in dj.read_text().splitlines() if line.strip()]
    n = min(len(orig), len(replay_results))
    matches = sum(
        1 for i in range(n) if orig[i].get("success") == replay_results[i]["success"]
    )
    return {
        "num_daemon_turns": len(orig),
        "num_replayed_turns": len(replay_results),
        "num_success_match": matches,
        "total_compared": n,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--scenario-dir", required=True, type=Path)
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output judgments.jsonl (default: <scenario-dir>/replay_judgments.jsonl). "
        "NEVER points inside the scenario dir by default when the dir already "
        "has judgments.jsonl — use a fresh directory to be safe.",
    )
    ap.add_argument("--judge-model")
    ap.add_argument("--judge-provider")
    ap.add_argument("--judge-base-url")
    ap.add_argument("--judge-api-key")
    ap.add_argument("--judge-prompt-version")
    ap.add_argument(
        "--judge-extra-body",
        default=None,
        help="JSON dict forwarded to litellm.completion as extra_body. "
        'Required for Qwen3 thinking checkpoints: \'{"chat_template_kwargs":{"enable_thinking":true}}\'.',
    )
    ap.add_argument("--verbose", "-v", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    scenario_dir = args.scenario_dir.resolve()
    if not scenario_dir.is_dir():
        raise SystemExit(f"Not a directory: {scenario_dir}")
    if not (scenario_dir / "events.jsonl").exists():
        raise SystemExit(f"No events.jsonl in {scenario_dir}")

    out_path = args.out or (scenario_dir / "replay_judgments.jsonl")

    parsed_extra_body = None
    if args.judge_extra_body:
        parsed_extra_body = json.loads(args.judge_extra_body)
        if not isinstance(parsed_extra_body, dict):
            raise SystemExit("--judge-extra-body must be a JSON object")

    summary = rejudge(
        scenario_dir=scenario_dir,
        out_path=out_path,
        judge_kwargs=dict(
            judge_model=args.judge_model,
            judge_provider=args.judge_provider,
            judge_base_url=args.judge_base_url,
            judge_api_key=args.judge_api_key,
            judge_prompt_version=args.judge_prompt_version,
            judge_extra_body=parsed_extra_body,
        ),
    )

    if isinstance(summary.get("results"), list):
        summary["daemon_diff"] = _diff_vs_daemon(scenario_dir, summary["results"])

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
