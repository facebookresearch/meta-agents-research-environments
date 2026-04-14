# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Container-based scenario runner for Gaia2.

Orchestrates running a single Gaia2 scenario inside a container:
launch → init → send task → wait for completion → extract events → collect result.

Judging is done entirely in-container by gaia2-eventd (final-turn judge).
The runner reads the verdict from daemon_status.json and constructs the result.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

from gaia2_runner.events_converter import convert_events_jsonl
from gaia2_runner.launcher import ContainerLauncher

logger: logging.Logger = logging.getLogger(__name__)

_RED = "\033[31m"
_RESET = "\033[0m"

# Timeout for polling agent response (seconds)
DEFAULT_RESPONSE_TIMEOUT = 600
# Timeout for waiting for adapter health (seconds)
DEFAULT_HEALTH_TIMEOUT = 120

# Container path where the LLM trace is written (read back by _extract_trace_file)
_CONTAINER_TRACE_PATH = "/tmp/trace.jsonl"


class ContainerRunner:
    """Run a single Gaia2 scenario inside a container and grade the result."""

    def __init__(
        self,
        launcher: ContainerLauncher,
        image: str,
        adapter_port: int = 8090,
    ) -> None:
        self.launcher = launcher
        self.image = image
        self.adapter_port = adapter_port
        # Use a session that bypasses proxy for localhost (the adapter runs
        # inside the container on 127.0.0.1). Without this, inherited proxy
        # env vars can route adapter requests through an outbound proxy and
        # break local health/status polling.
        self._local_session = requests.Session()
        self._local_session.trust_env = False
        self._last_daemon_status: dict | None = None

    def run_scenario(
        self,
        scenario_json_path: str,
        *,
        response_timeout: int = DEFAULT_RESPONSE_TIMEOUT,
        health_timeout: int = DEFAULT_HEALTH_TIMEOUT,
        container_env: dict[str, str] | None = None,
        provider: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        extra_volumes: tuple[str, ...] | None = None,
        output_dir: str | None = None,
        gateway_port: int | None = None,
    ) -> dict[str, Any]:
        """Run a scenario end-to-end and return the result.

        Args:
            scenario_json_path: Path to the scenario JSON file.
            response_timeout: Max seconds to wait for agent response.
            health_timeout: Max seconds to wait for adapter health.
            container_env: Extra environment variables for the container.
            provider: LLM provider name (e.g. "anthropic", "openai").
                When set, provider/model/key are injected as env vars.
            model: Model ID override for the provider.
            api_key: API key for the provider.
            extra_volumes: Extra bind mounts passed to the local container launcher.
            output_dir: Directory to save run artifacts (events, response,
                result, openclaw log). Each scenario gets a subdirectory
                named by its scenario_id. Pass ``None`` to skip saving.
            gateway_port: OpenClaw gateway port override.  When set, the
                container binds the gateway to this port instead of the
                default (18789).  Required for concurrent execution with
                ``--network=host``.

        Returns:
            Dict with keys: scenario_id, success, reward, num_agent_events,
            num_oracle_events, agent_response, error (if any).
        """
        scenario_path, scenario_data, scenario_id = self._load_scenario(
            scenario_json_path
        )
        # Resolve the original scenario file path for explorer links
        scenario_file = str(Path(scenario_json_path).resolve())

        # Use scenario duration as timeout if declared, otherwise CLI default.
        scenario_duration = scenario_data.get("metadata", {}).get("definition", {}).get(
            "duration"
        ) or scenario_data.get("duration")
        if scenario_duration is not None:
            scenario_duration = float(scenario_duration)
            effective_timeout = max(int(scenario_duration) + 60, response_timeout)
            logger.info(
                "Scenario duration=%ds, effective timeout=%ds (cli=%ds)",
                int(scenario_duration),
                effective_timeout,
                response_timeout,
            )
        else:
            effective_timeout = response_timeout

        container_id = None
        artifact_dir: Path | None = None
        try:
            effective_env = self._build_container_env(
                scenario_data, container_env, output_dir
            )

            container_id = self.launcher.launch(
                self.image,
                str(scenario_path),
                env=effective_env or None,
                provider=provider,
                model=model,
                api_key=api_key,
                extra_volumes=extra_volumes,
                adapter_port=self.adapter_port,
                gateway_port=gateway_port,
            )

            host_adapter_port = (
                self.launcher.get_host_adapter_port() or self.adapter_port
            )

            self.launcher.wait_for_adapter(
                container_id,
                port=host_adapter_port,
                timeout=health_timeout,
            )

            adapter_url = f"http://127.0.0.1:{host_adapter_port}"
            user_task = self._extract_user_task(scenario_data)

            # gaia2-eventd sends the initial user message for scenario-driven
            # runs (where events exist).  Only send from the runner when
            # there is no scenario / daemon.
            if user_task and not scenario_data.get("events"):
                logger.info(
                    "No scenario events; sending user task (%d chars) via /notify",
                    len(user_task),
                )
                resp = self._local_session.post(
                    f"{adapter_url}/notify",
                    json={"message": user_task},
                    timeout=10,
                )
                resp.raise_for_status()
                notify_result = resp.json()
                if not notify_result.get("ok"):
                    return {
                        "scenario_id": scenario_id,
                        "success": None,
                        "error": f"Failed to send task: {notify_result}",
                    }
                logger.info("Task sent, runId=%s", notify_result.get("runId"))
            else:
                logger.info(
                    "Daemon will send initial task (%d chars); waiting for response",
                    len(user_task or ""),
                )

            # 4. Poll for scenario completion via daemon status.
            # The daemon is the single authority on turn counting and
            # scenario lifecycle. The runner just polls GET /status
            # until the daemon reports completion (or timeout expires).
            poll_start = time.monotonic()
            agent_response, daemon_status = self._poll_for_response(
                adapter_url,
                timeout=effective_timeout,
            )
            poll_elapsed = time.monotonic() - poll_start
            timed_out = poll_elapsed >= effective_timeout - 1

            events_raw, agent_response = self._collect_events(
                container_id, agent_response
            )

            artifact_dir = self._ensure_artifact_dir(output_dir, scenario_id)

            # Extract daemon artifacts (logs, status, judgments) for debugging.
            if container_id and artifact_dir:
                self._extract_daemon_logs(container_id, artifact_dir)

            daemon_status = self._last_daemon_status or {}
            daemon_judgment = daemon_status.get("judgment")
            if daemon_judgment is not None:
                success = daemon_judgment.get("success", False)
                result = {
                    "scenario_id": scenario_id,
                    "success": success,
                    "reward": 1.0 if success else 0.0,
                    "num_agent_events": len(convert_events_jsonl(events_raw)),
                    "num_oracle_events": 0,
                    "agent_response": agent_response,
                    "judged_in_container": True,
                }
                if not success:
                    reason = daemon_judgment.get("failure_reason", "")
                    result["failure_reasons"] = (
                        [reason]
                        if reason
                        else [f"daemon status: {daemon_status.get('status')}"]
                    )
                logger.info(
                    "In-container judgment: %s",
                    "PASS" if success else "FAIL",
                )
            else:
                # No judgment — the daemon never saw a turn boundary
                # (no send_message_to_user in events.jsonl), or it
                # hit the idle timeout waiting for agent activity.
                ds = daemon_status.get("status", "unknown")
                num_events = len(convert_events_jsonl(events_raw))
                if ds == "error":
                    error_msg = (
                        f"Daemon error: no turn boundary detected "
                        f"({num_events} tool calls, idle timeout or agent stuck)"
                    )
                elif ds == "unknown":
                    error_msg = "No daemon status — container may lack judge support"
                else:
                    error_msg = (
                        f"No judgment (daemon status: {ds}, {num_events} agent events)"
                    )
                logger.warning(
                    "No in-container judgment for %s: %s",
                    scenario_id,
                    error_msg,
                )
                if timed_out and scenario_duration is not None:
                    # Agent exceeded the scenario's declared duration
                    # (e.g. 330s for time scenarios) plus a 60s buffer.
                    # This is a model failure, not an infra error.
                    result = {
                        "scenario_id": scenario_id,
                        "success": False,
                        "reward": 0.0,
                        "num_agent_events": num_events,
                        "agent_response": agent_response,
                        "failure_reasons": [
                            f"Scenario timeout: agent did not complete within "
                            f"{int(scenario_duration)}s scenario duration "
                            f"({num_events} tool calls)"
                        ],
                    }
                    logger.info(
                        "Scenario timeout → FAIL (duration=%ds)",
                        int(scenario_duration),
                    )
                else:
                    result = {
                        "scenario_id": scenario_id,
                        "success": None,
                        "error": error_msg,
                        "num_agent_events": num_events,
                        "agent_response": agent_response,
                    }
            if timed_out:
                result["timed_out"] = True
                result["timeout_seconds"] = effective_timeout
            result["daemon_status"] = daemon_status
            result["scenario_file"] = scenario_file

            self._extract_trace_file(container_id, artifact_dir)

            if output_dir:
                self._save_artifacts(
                    output_dir=output_dir,
                    scenario_id=scenario_id,
                    events_raw=events_raw,
                    agent_response=agent_response,
                    result=result,
                    container_id=container_id,
                )
                # Verify result.json was persisted — _save_artifacts
                # swallows exceptions, so an in-memory pass/fail that
                # never reaches disk would silently corrupt --retry.
                result_path = Path(output_dir) / scenario_id / "result.json"
                if not result_path.exists():
                    logger.error(
                        "result.json NOT written for %s — marking as error",
                        scenario_id,
                    )
                    result = {
                        "scenario_id": scenario_id,
                        "success": None,
                        "error": "Artifacts failed to save (result.json missing)",
                    }

            return result

        except Exception as exc:
            logger.error("Scenario %s failed: %s", scenario_id, exc, exc_info=True)
            error_result = {
                "scenario_id": scenario_id,
                "success": None,
                "error": str(exc),
                "scenario_file": scenario_file,
            }
            if output_dir:
                self._save_artifacts(
                    output_dir=output_dir,
                    scenario_id=scenario_id,
                    events_raw="",
                    agent_response=None,
                    result=error_result,
                    container_id=container_id,
                )
            return error_result
        finally:
            if container_id:
                try:
                    self.launcher.stop(container_id)
                except Exception:
                    logger.warning("Failed to stop container %s", container_id)

    # ── Helpers extracted from run_scenario ──────────────────────────────

    def _load_scenario(
        self,
        scenario_json_path: str,
    ) -> tuple[Path, dict[str, Any], str]:
        """Load and validate the scenario JSON.

        Returns (scenario_path, scenario_data, scenario_id).
        """
        scenario_path = Path(scenario_json_path).resolve()
        if not scenario_path.exists():
            raise FileNotFoundError(f"Scenario not found: {scenario_path}")

        with open(scenario_path) as f:
            scenario_data = json.load(f)

        scenario_id = (
            scenario_data.get("metadata", {})
            .get("definition", {})
            .get("scenario_id", "unknown")
        )
        logger.info("Running scenario %s from %s", scenario_id, scenario_path.name)

        return scenario_path, scenario_data, scenario_id

    def _build_container_env(
        self,
        scenario_data: dict[str, Any],
        container_env: dict[str, str] | None,
        output_dir: str | None,
    ) -> dict[str, str]:
        """Build the effective environment dict for the container."""
        from gaia2_runner.container_env import resolve_api_key_details

        effective_env = dict(container_env) if container_env else {}

        if "BASE_URL" not in effective_env and "BASE_URL" in os.environ:
            effective_env["BASE_URL"] = os.environ["BASE_URL"]

        if output_dir:
            effective_env["GAIA2_TRACE_FILE"] = _CONTAINER_TRACE_PATH

        start_time = (
            scenario_data.get("metadata", {}).get("definition", {}).get("start_time")
        )
        if start_time and "FAKETIME" not in effective_env:
            dt = datetime.fromtimestamp(float(start_time), tz=timezone.utc)
            effective_env["FAKETIME"] = dt.strftime("%Y-%m-%d %H:%M:%S")

        # Always enable in-container final-turn judging.
        effective_env["GAIA2_JUDGE_FINAL_TURN"] = "1"

        judge_model = (
            effective_env.get("GAIA2_JUDGE_MODEL")
            or os.environ.get("GAIA2_JUDGE_MODEL", "")
        ).strip()
        judge_provider = (
            effective_env.get("GAIA2_JUDGE_PROVIDER")
            or os.environ.get("GAIA2_JUDGE_PROVIDER", "")
        ).strip()
        if not judge_model or not judge_provider:
            missing = []
            if not judge_model:
                missing.append("GAIA2_JUDGE_MODEL")
            if not judge_provider:
                missing.append("GAIA2_JUDGE_PROVIDER")
            raise RuntimeError(
                "In-container judge configuration is required for benchmark runs. "
                f"Missing: {', '.join(missing)}"
            )

        effective_env["GAIA2_JUDGE_MODEL"] = judge_model
        effective_env["GAIA2_JUDGE_PROVIDER"] = judge_provider

        judge_api_key = (
            effective_env.get("GAIA2_JUDGE_API_KEY")
            or os.environ.get("GAIA2_JUDGE_API_KEY", "")
        ).strip()
        if not judge_api_key:
            resolved_judge_key = resolve_api_key_details(
                judge_provider, env=effective_env
            )
            judge_api_key = resolved_judge_key.value
            if resolved_judge_key.from_env and resolved_judge_key.source:
                logger.warning(
                    "%sJudge API key not passed via --judge-api-key; pulling from %s%s",
                    _RED,
                    resolved_judge_key.source,
                    _RESET,
                )
        if judge_api_key:
            effective_env["GAIA2_JUDGE_API_KEY"] = judge_api_key

        judge_base_url = effective_env.get("GAIA2_JUDGE_BASE_URL") or os.environ.get(
            "GAIA2_JUDGE_BASE_URL", ""
        )
        if judge_base_url:
            effective_env["GAIA2_JUDGE_BASE_URL"] = judge_base_url

        return effective_env

    def _collect_events(
        self,
        container_id: str,
        agent_response: str | None,
    ) -> tuple[str, str | None]:
        """Extract events.jsonl and resolve the agent response.

        Falls back to extracting the response from events.jsonl if the
        polling loop didn't catch it (e.g. race condition).

        Returns (events_raw, resolved_agent_response).
        Raises RuntimeError if the agent produced no response and no events.
        """
        events_raw = self._read_events_jsonl(container_id)

        if not agent_response:
            agent_response = self._extract_response_from_events(events_raw)

        if not agent_response and not events_raw.strip():
            raise RuntimeError(
                "Agent produced no response and no events (container or LLM failure)"
            )

        return events_raw, agent_response

    @staticmethod
    def _ensure_artifact_dir(
        output_dir: str | None,
        scenario_id: str,
    ) -> Path | None:
        """Create and return the artifact directory, or None if unset."""
        if not output_dir:
            return None
        artifact_dir = Path(output_dir) / scenario_id
        artifact_dir.mkdir(parents=True, exist_ok=True)
        return artifact_dir

    def _extract_trace_file(
        self,
        container_id: str | None,
        artifact_dir: Path | None,
    ) -> None:
        """Copy the LLM trace file from the container to the artifact dir."""
        if not artifact_dir or not container_id:
            return
        trace_path = str(artifact_dir / "trace.jsonl")
        try:
            self.launcher.copy_from(container_id, _CONTAINER_TRACE_PATH, trace_path)
            logger.info("Extracted trace: %s", trace_path)
        except Exception:
            logger.debug("No trace file in container (no LLM calls?)")

    def _extract_user_task(self, scenario_data: dict[str, Any]) -> str | None:
        """Extract the initial user task message from the scenario JSON.

        Looks for the first send_message_to_agent event in the scenario's
        events list (oracle events with EventType USER).

        Handles two action formats:
        - Legacy: ``action.function_name``, ``action.args`` is a plain dict
        - Current: ``action.function``, ``action.args`` is a list of
          ``{name, value}`` dicts
        """
        for events_key in ("events", "completed_events"):
            for event in scenario_data.get(events_key, []):
                action = event.get("action", {})
                fn = action.get("function_name") or action.get("function", "")
                if fn != "send_message_to_agent":
                    continue
                args = action.get("args", {})
                # args may be a plain dict or a list of {name, value} dicts
                if isinstance(args, dict):
                    content = args.get("content", "")
                elif isinstance(args, list):
                    content = ""
                    for arg in args:
                        if arg.get("name") == "content":
                            content = arg.get("value", "")
                            break
                else:
                    continue
                if content:
                    return content
        return None

    def _poll_for_response(
        self,
        adapter_url: str,
        timeout: int = DEFAULT_RESPONSE_TIMEOUT,
        interval: float = 2.0,
        activity_timeout: float = 600.0,
    ) -> tuple[str | None, str]:
        """Poll GET /status until the daemon reports scenario completion.

        The daemon writes ``daemon_status.json`` with its progress:
        ``{"status": "running"|"complete"|"stopped"|"error", "turn": N,
        "nb_turns": M, "last_response": "..."}``.

        The adapter serves it via ``GET /status``.  This method polls
        until ``status`` is a terminal value or *timeout* expires.

        Returns ``(agent_response, daemon_status)`` where *daemon_status*
        is the last observed status string (``"complete"``, ``"error"``,
        ``"timeout"``, etc.).
        """
        deadline = time.monotonic() + timeout
        activity_deadline = time.monotonic() + activity_timeout
        last_turn = -1
        last_num_events = -1
        last_response: str | None = None
        daemon_status = "unknown"

        while time.monotonic() < deadline:
            try:
                resp = self._local_session.get(f"{adapter_url}/status", timeout=5)
                data = resp.json()
                daemon_status = data.get("status", "waiting")
                turn = data.get("turn", 0)
                nb_turns = data.get("nb_turns", 1)
                num_events = data.get("num_events", 0)

                if turn != last_turn or num_events != last_num_events:
                    last_turn = turn
                    last_num_events = num_events
                    activity_deadline = time.monotonic() + activity_timeout
                    logger.info(
                        "Daemon status: %s (turn %d/%d, events=%d)",
                        daemon_status,
                        turn,
                        nb_turns,
                        num_events,
                    )

                if data.get("last_response"):
                    last_response = data["last_response"]

                if daemon_status in ("complete", "stopped", "error"):
                    logger.info(
                        "Daemon finished: status=%s turn=%d/%d",
                        daemon_status,
                        turn,
                        nb_turns,
                    )
                    self._last_daemon_status = data
                    return last_response, daemon_status

            except Exception as exc:
                logger.debug("Status poll error: %s", exc)

            # Activity timeout — daemon may be dead
            if time.monotonic() > activity_deadline:
                logger.warning(
                    "No activity for %.0fs — daemon may be stuck", activity_timeout
                )
                return last_response, "activity_timeout"

            time.sleep(interval)

        logger.warning("Timed out waiting for daemon completion after %ds", timeout)
        return last_response, "timeout"

    @staticmethod
    def _extract_message_text(raw: Any) -> str:
        """Extract plain text from an adapter message payload.

        Handles:
        - Plain string → returned as-is
        - Dict with ``content`` list of ``{type, text}`` blocks → concatenated
        - Dict with ``content`` string → returned directly
        - Anything else → str() fallback
        """
        if isinstance(raw, str):
            return raw
        if isinstance(raw, dict):
            content = raw.get("content")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                parts = []
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        parts.append(block.get("text", ""))
                    elif isinstance(block, str):
                        parts.append(block)
                return "\n".join(parts)
            # Fallback: try errorMessage for error states
            return raw.get("errorMessage", str(raw))
        return str(raw) if raw else ""

    @staticmethod
    def _extract_response_from_events(events_raw: str) -> str | None:
        """Extract the agent's final send_message_to_user from events.jsonl.

        The gaia2-adapter writes AUI.send_message_to_user entries to
        events.jsonl for terminal states (final/error/aborted).  Returns
        the last such message, or None if none found.
        """
        last_response = None
        for line in events_raw.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if (
                entry.get("app") == "AgentUserInterface"
                and entry.get("fn") == "send_message_to_user"
            ):
                content = entry.get("args", {}).get("content", "")
                if content:
                    last_response = content
        return last_response

    def _read_events_jsonl(self, container_id: str) -> str:
        """Read events.jsonl from the container.

        Tries ``exec cat`` first (fast, works while container is running).
        Falls back to ``podman cp`` if the container has already stopped
        (which happens when the gateway exits after the agent finishes).
        Returns empty string if the file doesn't exist in the container
        (daemon crashed before writing any events).
        """
        events_path = "/var/gaia2/state/events.jsonl"
        try:
            return self.launcher.exec(
                container_id,
                ["cat", events_path],
                user="gaia2",
            )
        except Exception as exc:
            logger.info("exec failed (%s), falling back to copy_from", exc)
            with tempfile.NamedTemporaryFile(
                mode="r",
                suffix=".jsonl",
                prefix="events-",
                delete=False,
            ) as tmp:
                tmp_path = tmp.name
            try:
                self.launcher.copy_from(container_id, events_path, tmp_path)
                with open(tmp_path) as f:
                    return f.read()
            except Exception as cp_exc:
                logger.warning(
                    "copy_from also failed (%s) — events.jsonl may not exist",
                    cp_exc,
                )
                return ""
            finally:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

    def _save_artifacts(
        self,
        output_dir: str,
        scenario_id: str,
        events_raw: str,
        agent_response: str | None,
        result: dict[str, Any],
        container_id: str | None,
    ) -> None:
        """Save run artifacts to disk for debugging and analysis.

        Creates ``<output_dir>/<scenario_id>/`` and writes:
        - events.jsonl — raw tool call log from the agent
        - agent_response.txt — agent's final text answer
        - result.json — grading result dict
        - openclaw.log — OpenClaw agent trace (if extractable)

        Failures are logged but never propagate to the caller.
        """
        try:
            artifact_dir = Path(output_dir) / scenario_id
            artifact_dir.mkdir(parents=True, exist_ok=True)

            # events.jsonl
            (artifact_dir / "events.jsonl").write_text(events_raw)

            # agent_response.txt
            (artifact_dir / "agent_response.txt").write_text(agent_response or "")

            # result.json
            (artifact_dir / "result.json").write_text(
                json.dumps(result, indent=2, default=str) + "\n"
            )

            # openclaw.log — find the log filename, then copy it out
            if container_id:
                self._extract_openclaw_log(container_id, artifact_dir)
                # daemon logs already extracted before judging

            logger.info("Artifacts saved to %s", artifact_dir)
        except Exception as exc:
            logger.warning("Failed to save artifacts: %s", exc)

    def _extract_openclaw_log(
        self,
        container_id: str,
        artifact_dir: Path,
    ) -> None:
        """Extract the OpenClaw agent log from the container.

        The log lives at ``/tmp/openclaw/openclaw-<date>.log`` inside the
        container.  We list the directory to discover the filename, then
        use ``copy_from`` to extract it.
        """
        try:
            ls_output = self.launcher.exec(
                container_id,
                ["ls", "/tmp/openclaw/"],
            )
            log_files = [
                f
                for f in ls_output.strip().split("\n")
                if f.startswith("openclaw-") and f.endswith(".log")
            ]
            if not log_files:
                logger.debug("No openclaw log found in container")
                return
            log_filename = log_files[0]
            src_path = f"/tmp/openclaw/{log_filename}"
            dst_path = str(artifact_dir / "openclaw.log")
            self.launcher.copy_from(container_id, src_path, dst_path)
        except Exception:
            # exec may fail if container already stopped; try copy_from
            # with a known pattern via a temp dir
            try:
                with tempfile.TemporaryDirectory(prefix="oc-log-") as tmpdir:
                    self.launcher.copy_from(container_id, "/tmp/openclaw/", tmpdir)
                    copied = list(Path(tmpdir).glob("openclaw-*.log"))
                    if copied:
                        shutil.copy2(
                            str(copied[0]),
                            str(artifact_dir / "openclaw.log"),
                        )
            except Exception as exc:
                logger.debug("Could not extract openclaw log: %s", exc)

    def _extract_daemon_logs(
        self,
        container_id: str,
        artifact_dir: Path,
    ) -> None:
        """Extract daemon log and in-container judgments for debugging.

        Copies:
        - ``/tmp/gaia2-eventd.log`` → ``eventd.log``
        - ``/tmp/entrypoint.log`` → ``entrypoint.log``
        - ``/var/gaia2/state/judgments.jsonl`` → ``daemon_judgments.jsonl``
        - ``/var/gaia2/state/user_details.json`` → ``user_details.json``
        - ``/var/gaia2/state/daemon_status.json`` → ``daemon_status.json``
        - ``/var/gaia2/state/notifications.jsonl`` → ``notifications.jsonl``
        - ``/tmp/faketime.rc`` → ``faketime.rc``
        """
        for src, dst in [
            ("/tmp/gaia2-eventd.log", "eventd.log"),
            ("/tmp/entrypoint.log", "entrypoint.log"),
            ("/var/gaia2/state/judgments.jsonl", "daemon_judgments.jsonl"),
            ("/var/gaia2/state/user_details.json", "user_details.json"),
            ("/var/gaia2/state/daemon_status.json", "daemon_status.json"),
            ("/var/gaia2/state/notifications.jsonl", "notifications.jsonl"),
            ("/tmp/faketime.rc", "faketime.rc"),
        ]:
            try:
                self.launcher.copy_from(container_id, src, str(artifact_dir / dst))
            except Exception:
                logger.debug("Could not extract %s from container", src)
