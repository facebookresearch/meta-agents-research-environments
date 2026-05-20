#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
"""Mini-SWE-Agent worker — runs as the sandboxed agent user.

Connects to the GAIA2 adapter (gaia2 user) via a Unix socket and processes
conversation requests using mini-swe-agent's DefaultAgent.

Mini agent contract:
    - DefaultAgent.run(task) loops: bash command → output → bash command ...
    - Terminates when env raises Submitted (agent ran
      `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\n<final text>`).
    - Returns dict with {exit_status, submission}.

Conversation model:
    - First notify per scenario → agent.run(task) (fresh history).
    - Subsequent notifies (e.g. ENV-bundled notifications mid-scenario for
      adaptability/time) → ``_resume_agent``: preserves agent.messages,
      synthesises a tool_result for the dangling submission tool_use, then
      re-enters the step loop. The agent keeps full conversation history
      across notifies for the lifetime of the scenario.

Protocol (JSON lines over Unix socket):
    Adapter → Worker:
        {"type": "message",   "text": "...", "run_id": "..."}
        {"type": "interrupt", "text": "..."}    # ignored (no native support)
    Worker → Adapter:
        {"type": "ready"}
        {"type": "response", "run_id": "...", "state": "final"|"error",
         "message": "..."}
"""

from __future__ import annotations

import json
import logging
import os
import queue
import socket
import sys
import threading
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

WORKER_SOCK = os.environ.get("MINI_WORKER_SOCK", "/tmp/mini-worker.sock")
AGENTS_MD = os.path.expanduser("~/AGENTS.md")
TRACE_FILE = os.environ.get("GAIA2_TRACE_FILE", "/tmp/trace.jsonl")
MINI_TRAJECTORY_FILE = os.environ.get(
    "MINI_TRAJECTORY_FILE", "/tmp/mini-trajectory.json"
)
_FAKETIME_FILE = "/tmp/faketime.rc"

logging.basicConfig(level=os.environ.get("MINI_LOG_LEVEL", "INFO"))
log = logging.getLogger("mini-worker")

_trace_seq = 0
_trace_lock = threading.Lock()


def _send(sock: socket.socket, obj: dict) -> None:
    payload = json.dumps(obj).encode() + b"\n"
    sock.sendall(payload)


def _recv_lines(sock: socket.socket):
    buf = b""
    while True:
        chunk = sock.recv(65536)
        if not chunk:
            return
        buf += chunk
        while b"\n" in buf:
            line, buf = buf.split(b"\n", 1)
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                log.warning("dropping unparseable line: %s (%s)", line[:200], e)


# ═══════════════════════════════════════════════════════════════════════
#  Agent setup
# ═══════════════════════════════════════════════════════════════════════


def _trace_timestamp() -> str:
    """ISO 8601 timestamp; prefers scenario faketime when active so trace
    entries are anchored to simulated wall-clock (matches hermes/openclaw)."""
    try:
        with open(_FAKETIME_FILE) as f:
            ts = f.read().strip()
        if ts:
            return (
                datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
                .replace(tzinfo=timezone.utc)
                .isoformat()
            )
    except (OSError, ValueError):
        pass
    return datetime.now(timezone.utc).isoformat()


def _jsonify_trace(value: Any) -> Any:
    """Best-effort JSON-safe coercion for arbitrary nested values."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _jsonify_trace(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify_trace(v) for v in value]
    if hasattr(value, "model_dump"):
        try:
            return _jsonify_trace(value.model_dump(mode="json"))
        except TypeError:
            return _jsonify_trace(value.model_dump())
    return str(value)


def _append_trace_entry(
    *,
    url: str,
    request: Any,
    latency_ms: float,
    http_status: int,
    raw_response: str,
) -> None:
    """Write one line to ``/tmp/trace.jsonl`` matching the GAIA2 trace-viewer
    schema (same keys hermes/openclaw emit). Per-call, append-only."""
    if not TRACE_FILE:
        return
    global _trace_seq
    with _trace_lock:
        _trace_seq += 1
        seq = _trace_seq
    entry = {
        "seq": seq,
        "timestamp": _trace_timestamp(),
        "type": "llm_call",
        "url": url,
        "latency_ms": round(latency_ms),
        "http_status": http_status,
        "request": request,
        "raw_response": raw_response,
    }
    try:
        with open(TRACE_FILE, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except OSError as e:
        log.warning("could not write trace: %s", e)


def _response_to_trace_body(response: Any, limit: int = 50_000) -> str:
    try:
        if hasattr(response, "model_dump_json"):
            body = response.model_dump_json()
        else:
            body = json.dumps(_jsonify_trace(response))
    except Exception as e:
        body = json.dumps({"_trace_serialize_failure": str(e)})
    return body[:limit]


def _install_trace_logging(model, base_url: str, model_name: str) -> None:
    """Wrap ``LitellmModel._query`` so every LLM call appends to trace.jsonl.

    Idempotent — sets a sentinel on the model after first install.
    """
    if not TRACE_FILE:
        return
    if getattr(model, "_gaia2_trace_logging_installed", False):
        return

    # Synthesize a URL-like identifier for the viewer's "Endpoint" column.
    if base_url:
        if "/anthropic" in base_url or "/messages" in base_url:
            trace_url = base_url.rstrip("/") + "/messages"
        else:
            trace_url = base_url.rstrip("/") + "/chat/completions"
    else:
        trace_url = f"litellm:{model_name}"

    orig_query = model._query

    def wrapped_query(messages, **kwargs):
        t0 = time.monotonic()
        request_dump = _jsonify_trace(
            {
                "model": model_name,
                "messages": messages,
                **({"kwargs": kwargs} if kwargs else {}),
            }
        )
        try:
            response = orig_query(messages, **kwargs)
        except Exception as e:
            _append_trace_entry(
                url=trace_url,
                request=request_dump,
                latency_ms=(time.monotonic() - t0) * 1000,
                http_status=getattr(e, "status_code", 0) or 500,
                raw_response=json.dumps({"error": str(e), "type": type(e).__name__}),
            )
            raise
        _append_trace_entry(
            url=trace_url,
            request=request_dump,
            latency_ms=(time.monotonic() - t0) * 1000,
            http_status=200,
            raw_response=_response_to_trace_body(response),
        )
        return response

    model._query = wrapped_query
    model._gaia2_trace_logging_installed = True


def _read_system_prompt() -> str:
    try:
        with open(AGENTS_MD) as f:
            return f.read()
    except OSError as e:
        log.warning("could not read %s: %s", AGENTS_MD, e)
        return ""


_FAKETIME_BASH = "/home/agent/bin/bash"


def _build_mini_agent(system_prompt: str):
    """Construct DefaultAgent wired to LitellmModel + LocalEnvironment.

    Reads MODEL, PROVIDER, API_KEY, BASE_URL, MAX_STEPS, COST_LIMIT,
    AGENT_TIMEOUT_SECONDS from env.
    """
    # Custom or fine-tuned model names may not be in litellm's pricing
    # table, so calculate_cost would throw. We don't care about cost
    # accounting for benchmark runs — ignore errors.
    os.environ.setdefault("MSWEA_COST_TRACKING", "ignore_errors")
    import subprocess

    from minisweagent.agents.default import DefaultAgent
    from minisweagent.environments.local import LocalEnvironment
    from minisweagent.models.litellm_model import LitellmModel

    class FaketimeLocalEnvironment(LocalEnvironment):
        """Run bash commands through /home/agent/bin/bash (libfaketime wrapper).

        Mini's vanilla LocalEnvironment uses ``shell=True`` which on Linux
        invokes ``/bin/sh`` (dash) — that bypasses the per-container faketime
        bash wrapper at /home/agent/bin/bash, so the agent sees REAL wall-
        clock time instead of the scenario's simulated time. Override
        ``execute`` to invoke the wrapper explicitly so ``date``, calendar
        queries, etc. return scenario time.
        """

        def execute(self, action: dict, cwd: str = "", *, timeout=None) -> dict:
            command = action.get("command", "")
            cwd = cwd or self.config.cwd or os.getcwd()
            try:
                result = subprocess.run(
                    [_FAKETIME_BASH, "-c", command],
                    text=True,
                    cwd=cwd,
                    env=os.environ | self.config.env,
                    timeout=timeout or self.config.timeout,
                    encoding="utf-8",
                    errors="replace",
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                )
                output = {
                    "output": result.stdout,
                    "returncode": result.returncode,
                    "exception_info": "",
                }
            except Exception as e:
                raw = getattr(e, "output", "") or ""
                if isinstance(raw, bytes):
                    raw = raw.decode("utf-8", errors="replace")
                output = {
                    "output": raw,
                    "returncode": -1,
                    "exception_info": f"An error occurred while executing the command: {e}",
                    "extra": {"exception_type": type(e).__name__, "exception": str(e)},
                }
            self._check_finished(output)
            return output

    model_name = os.environ.get("MODEL", "anthropic/claude-opus-4-6")
    api_key = os.environ.get("API_KEY", "")
    base_url = os.environ.get("BASE_URL", "")
    max_steps = int(os.environ.get("MINI_STEP_LIMIT", "200"))
    cost_limit = float(os.environ.get("MINI_COST_LIMIT", "10.0"))
    cmd_timeout = int(os.environ.get("MINI_COMMAND_TIMEOUT", "120"))

    model_kwargs: dict[str, Any] = {}
    if api_key:
        model_kwargs["api_key"] = api_key
    if base_url:
        # litellm/anthropic uses api_base; openai uses base_url. Set both,
        # litellm picks the right one per provider.
        model_kwargs["api_base"] = base_url
        model_kwargs["base_url"] = base_url

    log.info(
        "creating LitellmModel: model=%s base_url=%s api_key_set=%s",
        model_name,
        base_url or "<default>",
        bool(api_key),
    )
    model = LitellmModel(model_name=model_name, model_kwargs=model_kwargs)
    _install_trace_logging(model, base_url, model_name)

    log.info(
        "creating FaketimeLocalEnvironment (timeout=%ds, shell=%s)",
        cmd_timeout,
        _FAKETIME_BASH,
    )
    env = FaketimeLocalEnvironment(timeout=cmd_timeout)

    # The system_template is rendered once when the agent first formats messages.
    # mini's StrictUndefined Jinja blows up on {{...}} placeholders in our
    # AGENTS.md, so feed it as a *literal* string (no Jinja substitutions
    # needed — render_agent_prompt.py already filled them all in).
    instance_template = (
        "User task:\n\n{{task}}\n\n"
        "Use the GAIA2 CLI tools listed in your system prompt. Issue exactly ONE "
        "bash tool call per turn. When you have the final answer, call the "
        "bash tool with this command (exactly):\n"
        "  echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && echo '<your final reply to the user>'\n"
        "The text after the marker becomes your reply. Do NOT include explanations "
        "or markdown after the marker — just the literal user-facing answer."
    )

    agent = DefaultAgent(
        model,
        env,
        system_template=system_prompt or "You are a helpful assistant.",
        instance_template=instance_template,
        step_limit=max_steps,
        cost_limit=cost_limit,
    )

    # Mini's DefaultAgent.run() auto-saves the full trajectory via
    # `agent.save(config.output_path)` after every step. Point at a file in
    # /tmp so the runner can extract it as a per-scenario artifact (native
    # mini ``mini-swe-agent-1.1`` format — complements our GAIA2 trace.jsonl).
    if MINI_TRAJECTORY_FILE:
        agent.config.output_path = Path(MINI_TRAJECTORY_FILE)
        log.info("native mini trajectory will be saved to %s", MINI_TRAJECTORY_FILE)

    return agent


# ═══════════════════════════════════════════════════════════════════════
#  Main loop
# ═══════════════════════════════════════════════════════════════════════


def _run_task(agent, task: str, *, fresh: bool) -> tuple[str, str]:
    """Run a single agent task. Returns (state, message) for the adapter.

    ``fresh=True``  → first notify of the scenario. Call ``agent.run(task)``
                     which sets system + user messages and steps to exit.
    ``fresh=False`` → subsequent notify (mid-scenario, e.g. an ENV-bundled
                     notification from the GAIA2 daemon for adaptability
                     scenarios). Mini's ``DefaultAgent.run()`` would reset
                     ``self.messages``, losing prior context. Instead we
                     pop the trailing ``exit`` message from the previous
                     submission, append the new notification as a user
                     message, and re-enter the step loop until the agent
                     produces a new exit (next submission). The agent
                     keeps full conversation history across notifies.
    """
    try:
        if fresh:
            result = agent.run(task)
        else:
            result = _resume_agent(agent, task)
    except Exception as e:
        log.exception("agent step loop raised")
        return "error", f"Error: {type(e).__name__}: {e}"

    submission = (result or {}).get("submission") or ""
    exit_status = (result or {}).get("exit_status") or ""

    if not submission.strip():
        return (
            "error",
            f"Agent ended with exit_status={exit_status} and empty submission",
        )

    return "final", submission.strip()


def _resume_agent(agent, notification_text: str) -> dict:
    """Resume an already-run agent with a new user message, preserving history.

    Implementation mirrors ``DefaultAgent.run`` (same step loop + exception
    handling) but skips the ``self.messages = []`` reset and the system/
    instance template re-render. Side effect: ``agent.messages`` keeps
    growing across notifies for the lifetime of the scenario.
    """
    from minisweagent.exceptions import InterruptAgentFlow

    # Drop the trailing "exit" message from the previous run so the step
    # loop's termination check doesn't fire immediately.
    while agent.messages and agent.messages[-1].get("role") == "exit":
        agent.messages.pop()

    # Mini's env raises Submitted mid-execute_actions, so the LAST assistant
    # turn may contain a tool_use with no matching tool_result — Anthropic
    # rejects that on the next request ("tool_use ids were found without
    # tool_result blocks"). Synthesise a placeholder tool_result for any
    # dangling tool calls on the last assistant message so the conversation
    # is well-formed before we add the new user notification.
    if agent.messages:
        last = agent.messages[-1]
        if last.get("role") == "assistant":
            tool_calls = last.get("tool_calls") or []
            if tool_calls:
                observations = [
                    {
                        "output": "[COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT acknowledged — conversation continues with notification]",
                        "returncode": 0,
                        "exception_info": "",
                    }
                    for _ in tool_calls
                ]
                obs_msgs = agent.model.format_observation_messages(
                    last, observations, agent.get_template_vars()
                )
                agent.add_messages(*obs_msgs)

    # Wrap the notification as a follow-up user turn. GAIA2 daemon prefixes
    # bundled notifications with "[Notifications]"; we keep that as-is so
    # the agent recognises mid-execution updates vs the original task.
    agent.add_messages(
        agent.model.format_message(role="user", content=notification_text),
    )

    while True:
        try:
            agent.step()
        except InterruptAgentFlow as e:
            agent.add_messages(*e.messages)
        except Exception as e:
            agent.handle_uncaught_exception(e)
            raise
        finally:
            agent.save(agent.config.output_path)
        if agent.messages[-1].get("role") == "exit":
            break
    return agent.messages[-1].get("extra", {}) or {}


def _connect_socket() -> socket.socket | None:
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    log.info("connecting to %s", WORKER_SOCK)
    for attempt in range(60):
        try:
            sock.connect(WORKER_SOCK)
            log.info("connected")
            return sock
        except (FileNotFoundError, ConnectionRefusedError):
            if attempt == 59:
                log.error("failed to connect to adapter after 60 tries")
                return None
            time.sleep(1)
    return None


def _spawn_listener(sock: socket.socket, msg_queue: queue.Queue) -> None:
    def _listener():
        try:
            for msg in _recv_lines(sock):
                if msg.get("type") == "interrupt":
                    log.info("interrupt received (no-op): %s", msg.get("text", "")[:80])
                else:
                    msg_queue.put(msg)
        finally:
            msg_queue.put({"type": "__eof__"})

    threading.Thread(target=_listener, daemon=True).start()


def _process_message(sock: socket.socket, agent, msg: dict, *, is_fresh: bool) -> None:
    text = msg.get("text", "")
    run_id = msg.get("run_id", "unknown")
    log.info("starting task (run_id=%s, fresh=%s): %s", run_id, is_fresh, text[:100])
    try:
        state, message = _run_task(agent, text, fresh=is_fresh)
        log.info(
            "task complete (run_id=%s, state=%s): %s", run_id, state, message[:100]
        )
        payload = {
            "type": "response",
            "run_id": run_id,
            "state": state,
            "message": message,
        }
        if state == "error":
            payload["errorMessage"] = message
        _send(sock, payload)
    except Exception as e:
        log.exception("dispatch loop crashed")
        _send(
            sock,
            {
                "type": "response",
                "run_id": run_id,
                "state": "error",
                "message": f"worker crash: {e}\n{traceback.format_exc()}",
                "errorMessage": str(e),
            },
        )


def main() -> int:
    system_prompt = _read_system_prompt()
    agent = _build_mini_agent(system_prompt)

    sock = _connect_socket()
    if sock is None:
        return 1

    _send(sock, {"type": "ready"})
    msg_queue: queue.Queue = queue.Queue()
    _spawn_listener(sock, msg_queue)

    # First notify per scenario sets up the agent's system + user; subsequent
    # notifies (e.g. ENV-bundled notifications in adaptability scenarios)
    # are continuation turns that must preserve conversation history.
    is_fresh = True

    while True:
        msg = msg_queue.get()
        mtype = msg.get("type")
        if mtype == "__eof__":
            log.info("adapter disconnected; exiting")
            return 0
        if mtype != "message":
            log.info("unknown message type: %s", mtype)
            continue
        _process_message(sock, agent, msg, is_fresh=is_fresh)
        is_fresh = False


if __name__ == "__main__":
    sys.exit(main())
