#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
"""GAIA2 Adapter for Mini-SWE-Agent: HTTP bridge + Unix-socket client to mini_worker.

Runs as the ``gaia2`` user. Does NOT host the mini DefaultAgent — that lives in
``mini_worker.py`` as the ``agent`` user, so the agent's bash tool cannot
reach ``/var/gaia2/state`` (700, gaia2-owned).

Inbound  (GAIA2 -> Agent):  POST /notify   — forwarded over Unix socket
Outbound (Agent -> GAIA2):  GET  /events   — SSE stream of agent responses
                            GET  /messages — poll buffered agent responses
                            GET  /health   — connection status
                            GET  /status   — daemon scenario progress

Unix-socket protocol (JSON lines on /tmp/mini-worker.sock):
    adapter → worker: {"type": "message",   "text": "...", "run_id": "..."}
                      {"type": "interrupt", "text": "..."}
    worker → adapter: {"type": "ready"}
                      {"type": "response",  "run_id": "...", "state": "final"|"error", "message": "..."}
"""

import asyncio
import json
import os
import sys
import uuid

# Shared adapter base — HTTP server, message buffer, SSE, route dispatch.
# In Docker: both files are in /opt/. Locally: shared/ is a sibling directory.
_this_dir = os.path.dirname(os.path.abspath(__file__))
for _p in [_this_dir, os.path.join(_this_dir, "..", "shared")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
from gaia2_adapter_base import (  # noqa: E402
    AdapterState,
    create_client_handler,
    run_adapter,
    write_aui_event,
)
from gaia2_cli.daemon.cli_executor import execute_cli_action  # noqa: E402

# ── Configuration ───────────────────────────────────────────────────────
STATE_DIR = os.environ.get("GAIA2_STATE_DIR", "/var/gaia2/state")
WORKER_SOCK = os.environ.get("MINI_WORKER_SOCK", "/tmp/mini-worker.sock")

# ── Backend state ───────────────────────────────────────────────────────
_writer: asyncio.StreamWriter | None = None
_connected: bool = False
_active_run_id: str | None = None
_run_lock: asyncio.Lock | None = None  # initialised in main() once loop exists

_state = AdapterState(
    buffer_size=int(os.environ.get("GAIA2_BUFFER_SIZE", "200")),
)


# ═══════════════════════════════════════════════════════════════════════
#  Worker Unix-socket server
# ═══════════════════════════════════════════════════════════════════════


def _handle_worker_response(msg: dict) -> None:
    """Buffer a response from the worker and broadcast it to SSE clients."""
    global _active_run_id

    run_id = msg.get("run_id", msg.get("runId", ""))
    state = msg.get("state", "")

    if run_id and run_id == _active_run_id:
        _active_run_id = None

    if state == "final":
        write_aui_event("send_message_to_user", msg.get("message", "") or "")

    entry = _state.buffer_and_broadcast(
        {
            "run_id": run_id,
            "runId": run_id,
            "state": state,
            "message": msg.get("message", ""),
            **(
                {"errorMessage": msg["errorMessage"]} if msg.get("errorMessage") else {}
            ),
        }
    )
    print(f"[gaia2-adapter] Buffered {state} message seq={entry['seq']} runId={run_id}")


async def _on_worker_conn(
    reader: asyncio.StreamReader, writer: asyncio.StreamWriter
) -> None:
    """Handle a worker connection. Only one worker may be connected at a time."""
    global _writer, _connected

    if _writer is not None:
        print("[gaia2-adapter] new worker connected; dropping previous connection")
        try:
            _writer.close()
        except Exception:
            pass
    _writer = writer
    print("[gaia2-adapter] worker connected")

    try:
        async for raw in reader:
            line = raw.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except (json.JSONDecodeError, ValueError) as e:
                print(f"[gaia2-adapter] worker sent non-JSON: {e}: {raw!r}")
                continue

            mtype = msg.get("type")
            if mtype == "ready":
                _connected = True
                print("[gaia2-adapter] worker ready")
            elif mtype == "response":
                _handle_worker_response(msg)
            else:
                print(f"[gaia2-adapter] unknown worker message type: {mtype!r}")
    finally:
        _connected = False
        if _writer is writer:
            _writer = None
        try:
            writer.close()
        except Exception:
            pass
        print("[gaia2-adapter] worker disconnected")


async def _worker_listener() -> None:
    """Bind the Unix socket and serve worker connections forever."""
    try:
        os.unlink(WORKER_SOCK)
    except FileNotFoundError:
        pass
    server = await asyncio.start_unix_server(_on_worker_conn, path=WORKER_SOCK)
    os.chmod(WORKER_SOCK, 0o666)  # let the agent user connect
    print(f"[gaia2-adapter] listening for mini worker on {WORKER_SOCK}")
    async with server:
        await server.serve_forever()


# ═══════════════════════════════════════════════════════════════════════
#  /notify → worker
# ═══════════════════════════════════════════════════════════════════════


def _write_line(writer: asyncio.StreamWriter, msg: dict) -> None:
    writer.write((json.dumps(msg) + "\n").encode())


async def send_message(text: str) -> dict:
    """Forward a user message (or daemon notification) to the worker."""
    global _active_run_id

    if not _connected or _writer is None or _run_lock is None:
        raise ConnectionError("Mini worker not connected")

    async with _run_lock:
        if _active_run_id is not None:
            # Daemon notification arrived while a run is in flight. Mini's
            # worker logs interrupts but does not preempt the step loop, so
            # the new message will be processed once the current run returns.
            print(
                f"[gaia2-adapter] Queueing notification during active run {_active_run_id}"
            )
            _write_line(_writer, {"type": "interrupt", "text": text})

        run_id = str(uuid.uuid4())
        _active_run_id = run_id
        _write_line(_writer, {"type": "message", "text": text, "run_id": run_id})
        await _writer.drain()

    return {"run_id": run_id}


def is_connected() -> bool:
    return _connected


def get_health_info() -> dict:
    return {
        "backend": "mini",
        "worker_connected": _connected,
        "activeRun": _active_run_id,
    }


async def backend_connect() -> None:
    """Start the Unix-socket listener. Worker comes up later via entrypoint.sh."""
    asyncio.create_task(_worker_listener())


# ═══════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════


def _on_notify_sent(text: str) -> None:
    """Called after POST /notify successfully hands off to the worker."""
    write_aui_event("send_message_to_agent", text)


async def _execute_action(app: str, action: str, args: dict, event_id: str) -> dict:
    """Route ENV actions to CLI tools (runs as gaia2 user, has state access)."""
    return execute_cli_action(app, action, args, event_id, state_dir=STATE_DIR)


async def main() -> None:
    global _run_lock
    _run_lock = asyncio.Lock()

    handler = create_client_handler(
        state=_state,
        send_message=send_message,
        is_connected=is_connected,
        get_health_info=get_health_info,
        on_notify_sent=_on_notify_sent,
        execute_action=_execute_action,
    )

    await run_adapter(_state, handler, backend_connect, backend_name="Mini")


if __name__ == "__main__":
    asyncio.run(main())
