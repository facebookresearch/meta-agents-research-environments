# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
"""gaia2-eventd: Always-on Gaia2 environment daemon.

Runs a lightweight scenario engine that watches events.jsonl for agent actions,
detects turn boundaries, fires ENV reactions (via HTTP or CLI subprocess), and
delivers notifications to the agent via a file-based channel.

No Gaia2 framework imports needed — uses gaia2_cli.scenario instead.
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Any

import click

from gaia2_cli.app_registry import (
    APP_TO_AGENT_CLI as _APP_TO_AGENT_CLI,
)
from gaia2_cli.app_registry import (
    APP_TO_CLI as _APP_TO_CLI,
)
from gaia2_cli.app_registry import (
    NOTIFICATION_FORMATTERS as _NOTIFICATION_FORMATTERS,
)
from gaia2_cli.daemon.channel import FileChannelAdapter
from gaia2_cli.daemon.cli_executor import build_cli_cmd, run_cli

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _action_to_event_dict(action: dict[str, Any]) -> dict[str, Any]:
    """Convert a minimal CLI action entry to a full CompletedEvent dict.

    The CLI writes compact entries to events.jsonl::

        {"t": <float>, "app": "Contacts", "fn": "add_new_contact",
         "args": {...}, "w": true, "ret": "abc123"}

    The Gaia2 framework expects full CompletedEvent dicts.  This function
    bridges the two formats so that CompletedEvent.from_dict() can
    deserialize them.
    """
    app = action["app"]
    fn = action["fn"]
    return {
        "class_name": "CompletedEvent",
        "event_type": "AGENT",
        "event_time": action["t"],
        "event_id": f"AGENT-{app}.{fn}-{uuid.uuid4().hex[:8]}",
        "action": {
            "class_name": app,
            "app_name": app,
            "function_name": fn,
            "args": action["args"],
            "resolved_args": {},
            "operation_type": "write" if action.get("w") else "read",
            "action_id": f"{app}.{fn}-{uuid.uuid4().hex[:8]}",
        },
        "metadata": {
            "return_value": action.get("ret"),
            "exception": None,
            "exception_stack_trace": None,
            "completed": True,
        },
        "successors": [],
        "dependencies": [],
    }


def _is_send_message_to_user(entry: dict[str, Any]) -> bool:
    """Check if an events.jsonl entry is a send_message_to_user (turn boundary)."""
    return (
        entry.get("app") == "AgentUserInterface"
        and entry.get("fn") == "send_message_to_user"
    )


# ---------------------------------------------------------------------------
# Daemon core
# ---------------------------------------------------------------------------


class Gaia2EventDaemon:
    """Always-on Gaia2 environment daemon.

    Watches events.jsonl for agent actions, detects turn boundaries
    (send_message_to_user), fires ENV reactions (via HTTP in Docker mode
    or CLI subprocess in file mode), and writes notifications for the agent.
    """

    def __init__(
        self,
        scenario_path: str,
        state_dir: str,
        events_jsonl: str | None = None,
        notifications_path: str | None = None,
        responses_path: str | None = None,
        judge_model: str | None = None,
        judge_provider: str | None = None,
        judge_base_url: str | None = None,
        judge_api_key: str | None = None,
        poll_interval: float = 1.0,
        log_path: str | None = None,
        notify_url: str | None = None,
        faketime_path: str | None = None,
        notification_mode: str = "message",
        time_speed: float | None = None,
    ) -> None:
        self.scenario_path = Path(scenario_path)
        self.state_dir = Path(state_dir)
        self.events_jsonl = (
            Path(events_jsonl) if events_jsonl else self.state_dir / "events.jsonl"
        )
        self.notifications_path = (
            Path(notifications_path)
            if notifications_path
            else self.state_dir / "notifications.jsonl"
        )
        self.responses_path = (
            Path(responses_path)
            if responses_path
            else self.state_dir / "agent_responses.jsonl"
        )
        self.judge_model = judge_model
        self.judge_provider = judge_provider
        self.judge_base_url = judge_base_url
        self.judge_api_key = judge_api_key
        self.poll_interval = poll_interval
        self.notify_url = notify_url
        self.faketime_path = Path(faketime_path) if faketime_path else None
        self._notification_mode = notification_mode
        self._time_speed = time_speed

        self._last_event_offset: int = 0  # byte offset into events.jsonl
        self._turn_count: int = 0
        self._total_events: int = 0  # total agent events processed
        self._running: bool = False
        self._last_agent_response: str | None = None

        # Channel adapter (created in setup)
        self._channel: FileChannelAdapter | None = None

        # Populated by setup()
        self._processor: Any = None  # EventProcessor
        self._task: str | None = None
        self._final_judgment: Any = None
        self._app_name_to_class: dict[str, str] = {}
        self._app_to_agent_cli: dict[str, str] = {}  # scenario app → agent CLI name

        # --- File logging ---
        if log_path:
            self._setup_file_logging(log_path)

    def _setup_file_logging(self, log_path: str) -> None:
        """Add a file handler to the eventd logger for persistent logs."""
        fh = logging.FileHandler(log_path, mode="w")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(
            logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s")
        )
        logger.addHandler(fh)
        logger.setLevel(logging.DEBUG)

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def setup(self) -> None:
        """Load scenario, create EventProcessor with turn trigger conditions."""
        from gaia2_core.event_loop import EventProcessor
        from gaia2_core.loader import ScenarioLoader

        # ---- Load scenario ------------------------------------------------
        logger.info("Loading scenario from %s", self.scenario_path)
        loader = ScenarioLoader(str(self.scenario_path))
        self._task = loader.task
        self._app_name_to_class = loader.app_name_to_class

        # Build scenario app name → agent-visible CLI command mapping
        # using the app registry's agent_cli field (no hardcoded branded check).
        for app_name in self._app_name_to_class:
            self._app_to_agent_cli[app_name] = _APP_TO_AGENT_CLI.get(
                app_name, _APP_TO_CLI.get(app_name, app_name)
            )

        logger.info(
            "Scenario loaded: %d events, %d apps",
            len(loader.events),
            len(loader.app_name_to_class),
        )

        # ---- Create EventProcessor and build turn triggers ----------------
        processor = EventProcessor(
            events=loader.events,
            start_time=loader.start_time,
            duration=loader.duration,
            time_increment=loader.time_increment,
            app_name_to_class=loader.app_name_to_class,
        )
        processor.build_turn_triggers()

        # Apply time speed multiplier (fast-forward mode)
        if self._time_speed and self._time_speed != 1.0:
            original = processor.time_increment
            processor.time_increment *= self._time_speed
            logger.info(
                "Time speed %.1fx: time_increment %.1f → %.1f",
                self._time_speed,
                original,
                processor.time_increment,
            )

        self._processor = processor

        # ---- Dump user details from runtime contacts state ----------------
        # The contacts CLI reads from the state files created by gaia2-init.
        # Dump to a known path so the runner can extract it for post-hoc
        # judging (soft checkers need user name + address).
        self._dump_user_details()

        # ---- Create Judge (optional) ------------------------------------
        # Judge is needed for multi-turn scenarios (inter-turn grading) and
        # for single-turn scenarios when GAIA2_JUDGE_FINAL_TURN is set
        # (final-turn in-container grading).
        judge = None
        if processor.nb_turns > 1 or os.environ.get("GAIA2_JUDGE_FINAL_TURN"):
            judge = self._create_judge(loader, processor)
        if judge is not None:
            self._processor._judge = judge

        self._processor.schedule_initial_events()

        logger.info(
            "EventProcessor ready: %d turns, %d events, judge=%s",
            self._processor.nb_turns,
            len(loader.events),
            "yes" if judge else "no",
        )

        # Log scheduled events for debugging
        for event in loader.events:
            logger.info(
                "  %s id=%s type=%s deps=%d succs=%d",
                type(event).__name__,
                event.event_id,
                event.event_type,
                len(event.dependency_ids),
                len(event.successor_ids),
            )

        # ---- Create channel adapter --------------------------------------
        self._channel = FileChannelAdapter(
            notifications_path=self.notifications_path,
            responses_path=self.responses_path,
            events_jsonl_path=self.events_jsonl,
        )
        logger.info("Using file channel adapter")
        logger.info("Setup complete")
        self._write_status("running")

    def _create_judge(self, loader: Any, processor: Any) -> Any:
        """Create a Judge from oracle data.

        Returns None only when there are no oracle events (single-turn
        scenarios with nothing to judge).  Raises on any other failure
        so misconfigurations (bad LLM provider, missing deps) surface
        immediately instead of silently running without a judge.
        """
        from gaia2_cli.judge import Judge, create_litellm_engine

        oracle_data = loader.extract_oracle_data(
            event_id_to_turn_idx=processor.event_id_to_turn_idx,
            nb_turns=processor.nb_turns,
        )
        turn_oracle_events, turn_oracle_graph, tasks, _ = oracle_data

        # Load user details from the file we dumped in _dump_user_details
        # (ground truth from the container's runtime contacts state).
        from gaia2_core.types import UserDetails

        user_details = None
        ud_path = self.state_dir / "user_details.json"
        if ud_path.exists():
            ud = json.loads(ud_path.read_text())
            user_details = UserDetails(
                first_name=ud.get("first_name", ""),
                last_name=ud.get("last_name", ""),
                address=ud.get("address", ""),
            )

        # Check if there are any oracle events to judge
        total_oracle = sum(len(t) for t in turn_oracle_events)
        if total_oracle == 0:
            logger.info("No oracle events found, judge disabled")
            return None

        # Create LLM engine if model specified
        engine = None
        if self.judge_model:
            engine = create_litellm_engine(
                model=self.judge_model,
                provider=self.judge_provider,
                base_url=self.judge_base_url,
                validate=False,
                api_key=self.judge_api_key,
            )
            logger.info(
                "Judge LLM engine: model=%s provider=%s base_url=%s",
                self.judge_model,
                self.judge_provider,
                self.judge_base_url,
            )

        judge = Judge(
            turn_to_oracle_events=turn_oracle_events,
            turn_to_oracle_graph=turn_oracle_graph,
            tasks=tasks,
            user_details=user_details,
            start_time=loader.start_time,
            engine=engine,
            app_name_to_class=loader.app_name_to_class,
            state_dir=str(self.state_dir),
        )
        logger.info(
            "Judge created: %d turns, %d oracle events",
            len(turn_oracle_events),
            total_oracle,
        )
        return judge

    def _dump_user_details(self) -> None:
        """Read user contact from runtime state and write to state dir.

        Uses the same contacts state files that gaia2-init created from
        the scenario JSON.  Writes ``user_details.json`` so the runner
        can extract it for post-hoc soft judging without needing to
        exec into the container.
        """
        try:
            contacts_json = self.state_dir / "contacts.json"
            if not contacts_json.exists():
                logger.warning(
                    "No contacts.json in state dir, skipping user_details dump"
                )
                return
            contacts = json.loads(contacts_json.read_text()).get("contacts", {})
            for contact in contacts.values():
                if contact.get("is_user"):
                    user = {
                        "first_name": contact.get("first_name", ""),
                        "last_name": contact.get("last_name", ""),
                        "address": contact.get("address", ""),
                    }
                    out = self.state_dir / "user_details.json"
                    out.write_text(json.dumps(user))
                    logger.info(
                        "User details: %s %s (%s)",
                        user["first_name"],
                        user["last_name"],
                        user["address"][:40],
                    )
                    return
            logger.warning("No is_user contact found in contacts.json")
        except Exception as e:
            logger.warning("Failed to dump user_details: %s", e)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Main daemon loop."""
        assert self._channel is not None, "Call setup() before run()"
        self._running = True

        # Time tracking for simulated time advancement.
        # Real elapsed * time_increment = simulated elapsed.
        self._sim_start: float | None = None
        if self._processor:
            self._sim_start = self._processor.start_time

        try:
            self._run_file()
        except Exception:
            logger.exception("Daemon crashed")
            self.shutdown("error")

    def _noproxy_urlopen(
        self, url: str, data: bytes | None = None, timeout: float = 10
    ):
        """urlopen bypassing proxy (for localhost communication)."""
        import urllib.request

        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        req = urllib.request.Request(url)
        if data is not None:
            req.data = data
            req.add_header("Content-Type", "application/json")
        return opener.open(req, timeout=timeout)

    def _wait_for_notify_url(self, timeout: float = 120.0) -> bool:
        """Wait for the notify_url to be fully connected (adapter + gateway)."""
        health_url = f"{self.notify_url.rstrip('/')}/health"
        start = time.time()
        while time.time() - start < timeout:
            try:
                with self._noproxy_urlopen(health_url, timeout=2) as resp:
                    data = json.loads(resp.read())
                    if data.get("connected"):
                        logger.info(
                            "notify_url ready and connected: %s", self.notify_url
                        )
                        return True
                    logger.debug("Adapter up but not connected to gateway yet...")
            except Exception:
                pass
            time.sleep(1)
        logger.error("notify_url not ready after %.0fs: %s", timeout, self.notify_url)
        return False

    def _run_file(self) -> None:
        """File mode: poll events.jsonl for turn boundaries (send_message_to_user)."""
        logger.info(
            "gaia2-eventd running in file mode, watching %s (poll=%.2fs)",
            self.events_jsonl,
            self.poll_interval,
        )

        # Wait for notify_url (adapter) if configured
        if self.notify_url:
            if not self._wait_for_notify_url():
                logger.error("Aborting: adapter not available")
                return

        # Send initial user message if available
        if self._task:
            self._write_notification(
                {"type": "user_message", "content": self._task, "turn": 0}
            )
            logger.info("Sent initial user message (%d chars)", len(self._task))

        poll_count = 0
        last_activity = time.time()
        idle_timeout = 300.0  # shutdown if no events for 5 minutes
        while self._running:
            # Advance simulated time and fire due ENV events.
            # This runs on every poll iteration so ENV events fire
            # while the agent is working, not just at turn boundaries.
            if self._advance_time():
                last_activity = time.time()

            # Fail fast if judge rejected (condition may fire in _advance_time)
            if self._processor and self._processor.stopped:
                logger.info("Judge rejected — fail fast")
                if self._processor._last_judgment is not None:
                    self._final_judgment = self._processor._last_judgment
                self.shutdown("stopped")
                return

            new_entries = self._read_new_events()

            if not new_entries:
                poll_count += 1
                if poll_count % 50 == 0:
                    logger.debug(
                        "Still polling... (%d polls, no new events)", poll_count
                    )
                # Don't timeout while there are queued events waiting to fire
                has_pending = self._processor and bool(self._processor._queue)

                # Check if scenario completed while draining the queue
                # (the completion check at turn boundaries may have found
                # pending events, but _advance_time has since drained them).
                if not has_pending and self._check_scenario_complete():
                    return

                if not has_pending and time.time() - last_activity > idle_timeout:
                    logger.error("No activity for %.0fs — shutting down", idle_timeout)
                    self.shutdown("error")
                    return
                time.sleep(self.poll_interval)
                continue

            poll_count = 0
            last_activity = time.time()
            logger.info("Read %d new event(s) from events.jsonl", len(new_entries))

            # Keep ENV entries for the processor/judge so their real sim_t is
            # available for time checks, but exclude them from agent-facing
            # turn-boundary logic.
            agent_entries = [
                e
                for e in new_entries
                if not str(e.get("event_id", "")).startswith("Event-ENV-")
            ]
            if len(agent_entries) < len(new_entries):
                logger.debug(
                    "Including %d ENV event(s) for judge timing",
                    len(new_entries) - len(agent_entries),
                )

            for e in agent_entries:
                logger.debug(
                    "  event: %s.%s (w=%s)", e.get("app"), e.get("fn"), e.get("w")
                )

            # Add all polled entries immediately so the processor sees the
            # exact sim_t of ENV actions written to events.jsonl.
            self._total_events += len(agent_entries)
            self._add_events_to_processor(new_entries)
            self._write_status("running")

            if not agent_entries:
                time.sleep(self.poll_interval)
                continue

            # Check for turn boundary
            has_turn_boundary = any(_is_send_message_to_user(e) for e in agent_entries)

            if has_turn_boundary:
                # Capture the agent's response text from the turn boundary
                for e in agent_entries:
                    if _is_send_message_to_user(e):
                        self._last_agent_response = e.get("args", {}).get("content", "")

                self._turn_count += 1
                logger.info(
                    ">>> TURN BOUNDARY DETECTED — turn %d <<<", self._turn_count
                )
                self._write_status("running")
                self._process_turn([])

                # Check if scenario is complete
                if self._processor:
                    if self._processor.stopped:
                        logger.info(
                            "Judge stopped scenario at turn %d", self._turn_count
                        )
                        if self._processor._last_judgment is not None:
                            self._final_judgment = self._processor._last_judgment
                        self.shutdown("stopped")
                        return
                    if self._check_scenario_complete():
                        return
                    # Guard: agent taking way more turns than expected.
                    max_turns = self._processor.nb_turns * 3
                    if self._turn_count > max_turns and not has_pending:
                        logger.error(
                            "Agent exceeded %d turns (expected %d) — aborting",
                            self._turn_count,
                            self._processor.nb_turns,
                        )
                        self.shutdown("error")
                        return

                # After processing a turn, check if a follow-up user
                # message arrived via agent_responses.jsonl (the channel
                # adapter's response path).
                follow_ups = self._channel.read_notifications()
                for fu in follow_ups:
                    if fu.get("type") == "user_message":
                        self._write_notification(fu)
                        logger.info("Forwarded follow-up user_message from channel")
            else:
                logger.debug("No turn boundary in batch — waiting for more events")

                # Check completion outside turn boundary — the agent may
                # have stopped sending SMUs while the daemon finished all
                # condition-based turns.  Without this, the daemon waits
                # forever for the next SMU that never comes.
                if self._processor and self._turn_count >= self._processor.nb_turns:
                    has_pending = bool(self._processor._queue)
                    if self._processor.all_conditions_fired and not has_pending:
                        if os.environ.get("GAIA2_JUDGE_FINAL_TURN"):
                            result = self._processor.judge_final_turn()
                            if result is not None:
                                logger.info(
                                    "Final turn judge: %s%s",
                                    "PASS" if result.success else "FAIL",
                                    (
                                        f" — {result.failure_reason}"
                                        if result.failure_reason
                                        else ""
                                    ),
                                )
                                self._final_judgment = result
                                if not result.success:
                                    self.shutdown("stopped")
                                    time.sleep(self.poll_interval)
                                    continue
                        logger.info(
                            "All turns judged (%d/%d) — scenario complete",
                            self._turn_count,
                            self._processor.nb_turns,
                        )
                        self.shutdown()
                        time.sleep(self.poll_interval)
                        continue

            time.sleep(self.poll_interval)

    def shutdown(self, status: str = "complete") -> None:
        """Stop the daemon."""
        self._running = False
        self._write_status(status)
        logger.info("gaia2-eventd shut down (status=%s)", status)

    # ------------------------------------------------------------------
    # Completion check
    # ------------------------------------------------------------------

    def _check_scenario_complete(self) -> bool:
        """Check if all turns are done and judge the final turn.

        Returns True if the scenario is complete (caller should return).
        Called both at turn boundaries and after the event queue drains.
        """
        if not self._processor:
            return False
        has_pending = bool(self._processor._queue)
        if not (
            self._processor.all_conditions_fired
            and self._turn_count >= self._processor.nb_turns
            and not has_pending
        ):
            return False

        logger.info(
            "Completion check: all_conditions=%s, turns=%d/%d, pending=%d",
            self._processor.all_conditions_fired,
            self._turn_count,
            self._processor.nb_turns,
            len(self._processor._queue),
        )
        if os.environ.get("GAIA2_JUDGE_FINAL_TURN"):
            result = self._processor.judge_final_turn()
            if result is not None:
                logger.info(
                    "Final turn judge: %s%s",
                    "PASS" if result.success else "FAIL",
                    f" — {result.failure_reason}" if result.failure_reason else "",
                )
                self._final_judgment = result
                if not result.success:
                    self.shutdown("stopped")
                    return True
        logger.info(
            "All turns judged (%d/%d) — scenario complete",
            self._turn_count,
            self._processor.nb_turns,
        )
        self.shutdown()
        return True

    # ------------------------------------------------------------------
    # Status file
    # ------------------------------------------------------------------

    def _write_status(self, status: str) -> None:
        """Atomically write daemon_status.json to state_dir.

        The adapter serves this via GET /status so the runner can poll
        scenario progress without independently counting turns.
        """
        nb_turns = self._processor.nb_turns if self._processor else 1
        data = {
            "status": status,
            "turn": self._turn_count,
            "nb_turns": nb_turns,
            "num_events": self._total_events,
        }
        if self._last_agent_response:
            data["last_response"] = self._last_agent_response
        if self._final_judgment is not None:
            data["judgment"] = {
                "success": self._final_judgment.success,
                "failure_reason": self._final_judgment.failure_reason,
            }
        status_path = self.state_dir / "daemon_status.json"
        tmp_path = status_path.with_suffix(".json.tmp")
        try:
            tmp_path.write_text(json.dumps(data) + "\n")
            tmp_path.replace(status_path)
        except OSError as e:
            logger.warning("Failed to write daemon_status.json: %s", e)

    # ------------------------------------------------------------------
    # Event reading
    # ------------------------------------------------------------------

    def _read_new_events(self) -> list[dict[str, Any]]:
        """Read new lines from events.jsonl since the last read."""
        if not self.events_jsonl.exists():
            return []

        entries: list[dict[str, Any]] = []
        with open(self.events_jsonl, "r") as f:
            f.seek(self._last_event_offset)
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    logger.warning(f"Skipping malformed line: {line[:100]}")
            self._last_event_offset = f.tell()

        return entries

    # ------------------------------------------------------------------
    # Turn processing
    # ------------------------------------------------------------------

    def _add_events_to_processor(self, entries: list[dict[str, Any]]) -> None:
        """Convert events.jsonl entries and add to the EventProcessor."""
        from datetime import datetime, timezone

        from gaia2_core.types import (
            CompletedEvent as ScenarioCompletedEvent,
        )
        from gaia2_core.types import (
            EventAction,
        )

        for entry in entries:
            if "app" not in entry or "fn" not in entry:
                continue
            # Use simulated time (sim_t) when available so event_time is
            # in the same epoch as oracle event_time (scenario start_time).
            event_time = entry["t"]
            sim_t = entry.get("sim_t")
            if sim_t:
                try:
                    dt = datetime.strptime(sim_t, "%Y-%m-%d %H:%M:%S")
                    event_time = dt.replace(tzinfo=timezone.utc).timestamp()
                except (ValueError, TypeError):
                    logger.warning("Failed to parse sim_t %r, using wall-clock", sim_t)
            if self._processor.stopped:
                break
            raw_event_id = str(
                entry.get("event_id") or _action_to_event_dict(entry)["event_id"]
            )
            if raw_event_id.startswith("Event-ENV-"):
                event_type = "ENV"
            elif raw_event_id.startswith("Event-USER-"):
                event_type = "USER"
            else:
                event_type = "AGENT"
            completed = ScenarioCompletedEvent(
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
            self._processor.add_agent_event(completed)

    def _process_turn(self, entries: list[dict[str, Any]]) -> None:
        """Process a turn boundary: tick the processor and execute ENV actions.

        Agent events are already added to the processor via
        ``_add_events_to_processor()`` as they arrive. This method
        evaluates conditions and fires zero-delay ENV reactions.

        ENV events with nonzero event_relative_time are queued and
        fired by ``_advance_time()`` in the main poll loop.
        """
        logger.info("=== PROCESS TURN %d ===", self._turn_count)

        # 3: Tick the processor — ConditionChecks evaluate and
        # trigger successor ENV events.
        try:
            env_actions = self._processor.tick()
        except Exception as e:
            logger.error("Processor tick FAILED: %s", e, exc_info=True)
            return

        logger.info("TICK complete: %d ENV actions", len(env_actions))

        # 4: Execute immediate ENV actions (zero-delay events).
        # Events with nonzero event_relative_time are queued and will
        # be fired by _advance_time() in the main poll loop.
        self._execute_env_actions(env_actions)

        logger.info(
            "=== TURN %d DONE ===",
            self._turn_count,
        )

        # Update faketime for next turn's CLI calls
        if self.faketime_path and self._processor:
            self._update_faketime()

    def _execute_env_actions(
        self,
        env_actions: list[tuple[str, str, dict[str, Any], str]],
    ) -> None:
        """Execute ENV actions and send notifications.

        In Docker mode (``notify_url`` set), dispatches actions via
        ``POST /execute_action`` to the adapter so the container can
        route to CLI tools, Python app methods, or any other backend.
        In file mode, calls CLI tools directly via subprocess.
        """
        if not env_actions:
            return

        logger.info("Executing %d ENV actions:", len(env_actions))
        pending_notifications: list[str] = []
        for i, (app_name, fn_name, args, env_event_id) in enumerate(env_actions):
            # AUI.send_message_to_agent -> route as user message (not a CLI action)
            if app_name == "AgentUserInterface" and fn_name == "send_message_to_agent":
                aui_content = args.get("content", "")
                if aui_content:
                    logger.info("Routing as user message (%d chars)", len(aui_content))
                    self._send_to_endpoint("/send_user_message", aui_content)
                continue

            logger.info(
                "  ENV[%d]: %s.%s(%s) [%s]",
                i,
                app_name,
                fn_name,
                ", ".join(f"{k}={repr(v)[:30]}" for k, v in args.items()),
                env_event_id,
            )

            result = self._dispatch_env_action(app_name, fn_name, args, env_event_id)

            if result is not None:
                logger.info("  result: %s", result[:120] if result else "(empty)")
                # Write ENV return value back to processor and judge so that
                # {{Event-ENV-xxx}} placeholders resolve in future turns.
                # Use the `ret` field from events.jsonl (written by log_action)
                # rather than raw stdout — log_action writes the semantic return
                # value (e.g. email_id) not the full JSON response.
                env_ret = self._read_env_ret(env_event_id)
                if env_ret is None:
                    env_ret = result  # fallback to stdout / HTTP result
                if self._processor and env_event_id in self._processor._completed:
                    self._processor._completed[env_event_id].return_value = env_ret
                    if self._processor._judge is not None:
                        self._processor._judge.env_return_values[env_event_id] = env_ret
                # Collect formatted notification (bundled after the loop)
                formatted = self._format_notification(app_name, fn_name, args)
                if formatted:
                    pending_notifications.append(formatted)

        # Send bundled notifications
        if pending_notifications:
            self._send_notifications(pending_notifications)

    def _dispatch_env_action(
        self,
        app_name: str,
        fn_name: str,
        args: dict[str, Any],
        env_event_id: str,
    ) -> str | None:
        """Dispatch a single ENV action via HTTP or direct CLI.

        Returns the result string on success, *None* on failure.
        """
        # Docker mode: route through the adapter's /execute_action endpoint
        if self.notify_url:
            return self._execute_via_http(app_name, fn_name, args, env_event_id)

        # File mode: call CLI tools directly (standalone / test usage)
        cmd = build_cli_cmd(app_name, fn_name, args)
        if not cmd:
            logger.error(
                "ENV_EXEC_FAILED app=%s fn=%s event_id=%s reason=no_cli_mapping",
                app_name,
                fn_name,
                env_event_id,
            )
            return None
        import shlex

        logger.info("  cmd: %s", shlex.join(cmd))
        return run_cli(cmd, event_id=env_event_id, state_dir=str(self.state_dir))

    def _execute_via_http(
        self,
        app_name: str,
        fn_name: str,
        args: dict[str, Any],
        env_event_id: str,
    ) -> str | None:
        """POST /execute_action to the adapter and return the result."""
        # Log in the same format as the CLI path so the trace viewer can
        # parse ENV actions (it looks for "  cmd: <command>" lines).
        subcmd = fn_name.replace("_", "-")
        arg_parts = " ".join(f"--{k.replace('_', '-')} ..." for k in args)
        logger.info("  cmd: %s %s %s", app_name, subcmd, arg_parts)
        url = f"{self.notify_url.rstrip('/')}/execute_action"
        payload = json.dumps(
            {
                "app": app_name,
                "action": fn_name,
                "args": args,
                "event_id": env_event_id,
            }
        ).encode()
        try:
            with self._noproxy_urlopen(url, data=payload, timeout=60) as resp:
                result = json.loads(resp.read())
                if result.get("ok"):
                    return result.get("result", "")
                logger.error(
                    "ENV_EXEC_FAILED app=%s fn=%s event_id=%s http_error=%s",
                    app_name,
                    fn_name,
                    env_event_id,
                    result.get("error", "unknown"),
                )
                return None
        except Exception as e:
            logger.error(
                "ENV_EXEC_FAILED app=%s fn=%s event_id=%s reason=%s",
                app_name,
                fn_name,
                env_event_id,
                e,
            )
            return None

    def _advance_time(self) -> bool:
        """Advance simulated time and fire any due ENV events.

        Called on every poll iteration in the main loop.  Advances
        ``_current_time`` by a fixed step (``poll_interval * time_increment``)
        each tick, decoupled from wall-clock time.  This makes time dilation
        purely a function of ``time_increment`` — high multipliers advance
        faster without needing poll interval adjustments.

        Also drains ConditionChecks so they can evaluate and schedule
        ENV successor events when conditions are met.

        Returns True if any ENV actions were fired.
        """
        if not self._processor:
            return False

        sim_step = self.poll_interval * self._processor.time_increment
        sim_time = self._processor._current_time + sim_step

        env_actions = self._processor.tick_to(sim_time)

        # Always update faketime so CLI tools see advancing simulated
        # time, even when no ENV events fire this tick.
        if self.faketime_path:
            self._update_faketime()

        if not env_actions:
            return False

        # Count real ENV actions (exclude user messages routed via AUI)
        real_env = sum(
            1
            for app, fn, _, _ in env_actions
            if not (app == "AgentUserInterface" and fn == "send_message_to_agent")
        )
        if real_env > 0:
            logger.info(
                "Time advance: sim=%.1f (+%.1fs), firing %d ENV actions",
                sim_time,
                sim_time - self._sim_start,
                real_env,
            )

        self._execute_env_actions(env_actions)
        return True

    # ------------------------------------------------------------------
    # Faketime
    # ------------------------------------------------------------------

    def _update_faketime(self) -> None:
        """Write current simulated time to the faketime file.

        libfaketime reads /tmp/faketime.rc to freeze time for CLI tools.
        This advances the simulated clock between turns so the agent and
        ENV actions see correct scenario timestamps.
        """
        from datetime import datetime, timezone

        current = self._processor._current_time
        sim_time = datetime.fromtimestamp(current, tz=timezone.utc)
        time_str = sim_time.strftime("%Y-%m-%d %H:%M:%S")
        tmp_path = self.faketime_path.with_suffix(".tmp")
        tmp_path.write_text(time_str + "\n")
        tmp_path.rename(self.faketime_path)
        logger.info("Faketime updated: %s", time_str)

    def _read_env_ret(self, env_event_id: str) -> Any:
        """Read the `ret` value for an ENV event from events.jsonl.

        After an ENV CLI call, ``log_action()`` writes the semantic return
        value (e.g. an email_id string) to events.jsonl.  Reading it back
        gives us the same value the replay tool's backfill uses, rather
        than the raw stdout JSON.
        """
        try:
            with open(self.events_jsonl) as f:
                for line in reversed(f.readlines()):
                    line = line.strip()
                    if not line:
                        continue
                    entry = json.loads(line)
                    if entry.get("event_id") == env_event_id:
                        return entry.get("ret")
        except Exception:
            pass
        return None

    # ------------------------------------------------------------------
    # Notifications
    # ------------------------------------------------------------------

    def _format_notification(
        self, app: str, fn: str, args: dict[str, Any]
    ) -> str | None:
        """Format an ENV action as a human-readable notification.

        Returns a clean one-liner like ``EmailClientV2: New email received
        from alice@example.com``, or *None* if this app/function combo is
        not configured for notifications.  Lookup uses the module-level
        ``_NOTIFICATION_FORMATTERS`` registry (data-driven, easy to extend).
        """
        fns = _NOTIFICATION_FORMATTERS.get(app)
        if not fns or fn not in fns:
            return None
        # Use the agent-visible CLI command name (e.g. "chats" not "Chats")
        label = self._app_to_agent_cli.get(app, app)
        try:
            return f"{label}: {fns[fn](args)}"
        except Exception as exc:
            logger.warning("Formatter error for %s.%s: %s", app, fn, exc)
            return f"{label}: {fn} completed"

    def _send_notifications(self, notifications: list[str]) -> None:
        """Bundle and send ENV notifications via the adapter."""
        if len(notifications) == 1:
            content = f"[Notification] {notifications[0]}"
        else:
            lines = "\n".join(f"- {n}" for n in notifications)
            content = f"[Notifications]\n{lines}"
        logger.info(
            "Sending %d bundled notification(s) (turn %d)",
            len(notifications),
            self._turn_count,
        )
        self._send_to_endpoint("/send_notifications", content)

    def _send_to_endpoint(self, endpoint: str, content: str, retries: int = 5) -> bool:
        """POST a message to an adapter endpoint with retries."""
        if not self.notify_url:
            logger.warning("No notify_url configured, cannot send to %s", endpoint)
            return False
        url = f"{self.notify_url.rstrip('/')}{endpoint}"
        payload = json.dumps({"message": content}).encode()
        for attempt in range(retries):
            try:
                with self._noproxy_urlopen(url, data=payload, timeout=30) as resp:
                    result = json.loads(resp.read())
                    if result.get("ok"):
                        logger.info("Sent to %s (%d chars)", endpoint, len(content))
                        return True
                    logger.error("Adapter rejected %s: %s", endpoint, result)
            except Exception as e:
                logger.warning(
                    "%s attempt %d/%d failed: %s",
                    endpoint,
                    attempt + 1,
                    retries,
                    e,
                )
            if attempt < retries - 1:
                time.sleep(2)
        logger.error("Failed to send to %s after %d attempts", endpoint, retries)
        return False

    def _send_via_notify_url(self, content: str, retries: int = 5) -> bool:
        """Send a user message via POST /send_user_message (or /notify alias)."""
        return self._send_to_endpoint("/send_user_message", content, retries)

    def _write_notification(self, notification: dict[str, Any]) -> None:
        """Write a notification to the agent (file mode fallback).

        In Docker mode, notifications go through the adapter endpoints
        (``_send_notifications`` / ``_send_to_endpoint``).  This method
        is only used for file-mode adapters and the initial task message.
        """
        ntype = notification.get("type", "unknown")
        logger.info("Notification: %s (turn %s)", ntype, notification.get("turn", "?"))

        content = notification.get("content", "")
        if ntype == "env_action":
            app = notification.get("app", "")
            fn = notification.get("function", "")
            formatted = self._format_notification(app, fn, notification.get("args", {}))
            content = formatted or f"[ENV] {app}.{fn}() completed"
        elif ntype == "user_message" and not content:
            content = str(notification)

        if self.notify_url:
            if content:
                self._send_via_notify_url(content)
        else:
            with open(self.notifications_path, "a") as f:
                f.write(json.dumps(notification) + "\n")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


@click.command()
@click.option(
    "--scenario",
    required=True,
    type=click.Path(exists=True),
    help="Path to scenario JSON file.",
)
@click.option(
    "--state-dir",
    required=True,
    type=click.Path(exists=True),
    help="Path to state directory (from gaia2-init).",
)
@click.option(
    "--events-jsonl",
    default=None,
    type=click.Path(),
    help="Path to events.jsonl (default: <state-dir>/events.jsonl).",
)
@click.option(
    "--notifications",
    default=None,
    type=click.Path(),
    help="Path to notifications output file.",
)
@click.option(
    "--responses",
    default=None,
    type=click.Path(),
    help="Path to agent_responses.jsonl (default: <state-dir>/agent_responses.jsonl).",
)
@click.option(
    "--judge-model",
    default=None,
    help="LLM model name for the judge (e.g. claude-opus-4-6 or gpt-4.1).",
)
@click.option(
    "--judge-provider",
    default=None,
    help="LLM provider for the judge (e.g. openai, anthropic).",
)
@click.option(
    "--judge-base-url",
    default=None,
    help="API base URL for the judge LLM.",
)
@click.option(
    "--judge-api-key",
    default=None,
    help="Optional API key override for the judge LLM.",
)
@click.option(
    "--poll-interval",
    default=1.0,
    type=float,
    help="Seconds between events.jsonl polls.",
)
@click.option(
    "--notify-url",
    default=None,
    help="HTTP URL for sending notifications (POST /notify). "
    "When set, user messages and ENV notifications are sent via HTTP "
    "while events.jsonl is still used for turn detection (file mode). "
    "Use in Docker where gaia2-adapter.mjs bridges to the agent gateway.",
)
@click.option(
    "--faketime-path",
    default=None,
    type=click.Path(),
    help="Path to faketime.rc file. When set, writes simulated time after each turn.",
)
@click.option(
    "--notification-mode",
    type=click.Choice(["message", "native"]),
    default="message",
    help="ENV notification delivery mode.",
)
@click.option(
    "--time-speed",
    default=None,
    type=float,
    help="Time speed multiplier (e.g. 5 = 5x faster). Multiplies scenario time_increment.",
)
def main(
    scenario: str,
    state_dir: str,
    events_jsonl: str | None,
    notifications: str | None,
    responses: str | None,
    judge_model: str | None,
    judge_provider: str | None,
    judge_base_url: str | None,
    judge_api_key: str | None,
    poll_interval: float,
    notify_url: str | None,
    faketime_path: str | None,
    notification_mode: str,
    time_speed: float | None,
) -> None:
    """Run the Gaia2 event daemon.

    Watches events.jsonl for agent tool calls, detects turn boundaries
    (send_message_to_user), fires ENV reactions as CLI subprocess calls,
    and writes notifications to the agent.
    """
    # Console logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    # File logging to {state_dir}/eventd.log
    log_path = str(Path(state_dir) / "eventd.log")

    daemon = Gaia2EventDaemon(
        scenario_path=scenario,
        state_dir=state_dir,
        events_jsonl=events_jsonl,
        notifications_path=notifications,
        responses_path=responses,
        judge_model=judge_model,
        judge_provider=judge_provider,
        judge_base_url=judge_base_url,
        judge_api_key=judge_api_key,
        poll_interval=poll_interval,
        log_path=log_path,
        notify_url=notify_url,
        faketime_path=faketime_path,
        notification_mode=notification_mode,
        time_speed=time_speed,
    )

    logger.info("Setting up daemon... (log: %s)", log_path)
    daemon.setup()

    logger.info("Starting daemon...")
    try:
        daemon.run()
    except KeyboardInterrupt:
        logger.info("Interrupted, shutting down...")
        daemon.shutdown()


if __name__ == "__main__":
    main()
