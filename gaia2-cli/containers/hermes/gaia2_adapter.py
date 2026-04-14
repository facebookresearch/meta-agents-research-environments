#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
"""Gaia2 Adapter for Hermes: HTTP bridge + Unix socket worker transport.

Inbound  (Gaia2 -> Agent):  POST /notify   — sends user messages to Hermes
Outbound (Agent -> Gaia2):  GET  /events   — SSE stream of agent responses
                            GET  /messages — poll buffered agent responses (?after=<seq>)
                            GET  /health   — connection status
                            GET  /status   — daemon scenario progress

Hermes integration:
    - The adapter runs as the gaia2 user and owns the HTTP surface.
    - The worker runs as the agent user and owns the AIAgent instance.
    - Adapter and worker communicate over a Unix socket using JSON lines.
    - Worker responses are buffered for /messages and /events SSE.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import uuid
from collections import deque
from pathlib import Path
from typing import Any

# Shared adapter base — HTTP server, message buffer, SSE, route dispatch.
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
WORKER_SOCK = os.environ.get("HERMES_WORKER_SOCK", "/tmp/hermes-worker.sock")
WORKER_READY_TIMEOUT = float(os.environ.get("HERMES_WORKER_READY_TIMEOUT", "60"))

# ── Backend state ───────────────────────────────────────────────────────
_worker_server: asyncio.AbstractServer | None = None
_worker_reader: asyncio.StreamReader | None = None
_worker_writer: asyncio.StreamWriter | None = None
_worker_ready = False
_worker_ready_event: asyncio.Event | None = None
_worker_send_lock: asyncio.Lock | None = None
_pending_run_ids: deque[str] = deque()

# Shared adapter state
_state = AdapterState(
    buffer_size=int(os.environ.get("GAIA2_BUFFER_SIZE", "200")),
)


# ═══════════════════════════════════════════════════════════════════════
#  Worker bridge
# ═══════════════════════════════════════════════════════════════════════


def _message_text(message: Any) -> str:
    """Extract visible assistant text from the worker response payload."""
    if isinstance(message, str):
        return message
    if not isinstance(message, dict):
        return ""

    for value in (message.get("text"), message.get("content")):
        if isinstance(value, str):
            return value

    content = message.get("content")
    if not isinstance(content, list):
        return ""

    parts: list[str] = []
    for item in content:
        if isinstance(item, str):
            parts.append(item)
            continue
        if not isinstance(item, dict):
            continue
        if item.get("type") in ("text", "input_text", "output_text"):
            text = item.get("text")
            if isinstance(text, str):
                parts.append(text)
    return "".join(parts)


def _response_text(response: dict[str, Any]) -> str:
    return _message_text(response.get("message", ""))


def _clear_pending_run(run_id: str) -> None:
    if not run_id:
        return
    try:
        _pending_run_ids.remove(run_id)
    except ValueError:
        pass


def _reset_worker_connection() -> None:
    global _worker_reader, _worker_writer, _worker_ready
    _worker_reader = None
    _worker_writer = None
    _worker_ready = False
    _pending_run_ids.clear()


def on_backend_response(response: dict[str, Any]) -> None:
    """Buffer a worker response and emit a turn boundary when needed."""
    state = response.get("state", "")
    run_id = response.get("run_id", response.get("runId", ""))
    text = _response_text(response)

    _clear_pending_run(str(run_id))
    entry = _state.buffer_and_broadcast(response)
    print(f"[gaia2-adapter] Buffered {state} message seq={entry['seq']} runId={run_id}")

    if state in ("final", "error"):
        write_aui_event("send_message_to_user", text)


def _handle_worker_frame(frame: dict[str, Any]) -> None:
    """Dispatch a single worker frame received over the Unix socket."""
    global _worker_ready

    frame_type = frame.get("type", "")
    if frame_type == "ready":
        _worker_ready = True
        if _worker_ready_event is not None:
            _worker_ready_event.set()
        print("[gaia2-adapter] Hermes worker connected and ready")
        return

    if frame_type == "response":
        response = dict(frame)
        response.pop("type", None)
        on_backend_response(response)
        return

    print(f"[gaia2-adapter] Ignoring unknown worker frame type: {frame_type}")


async def _handle_worker_client(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
) -> None:
    """Accept the single Hermes worker connection and relay its frames."""
    global _worker_reader, _worker_writer

    if _worker_writer is not None and not _worker_writer.is_closing():
        print("[gaia2-adapter] Rejecting duplicate Hermes worker connection")
        writer.close()
        await writer.wait_closed()
        return

    _worker_reader = reader
    _worker_writer = writer

    try:
        while True:
            line = await reader.readline()
            if not line:
                break
            try:
                frame = json.loads(line)
            except json.JSONDecodeError:
                print(f"[gaia2-adapter] Ignoring invalid worker frame: {line[:200]!r}")
                continue
            _handle_worker_frame(frame)
    finally:
        if writer is _worker_writer:
            print("[gaia2-adapter] Hermes worker disconnected")
            _reset_worker_connection()
        writer.close()
        await writer.wait_closed()


async def _shutdown_worker_bridge() -> None:
    """Close the worker socket server and remove the socket path."""
    global _worker_server

    writer = _worker_writer
    if writer is not None and not writer.is_closing():
        writer.close()
        await writer.wait_closed()
    _reset_worker_connection()

    server = _worker_server
    _worker_server = None
    if server is not None:
        server.close()
        await server.wait_closed()

    Path(WORKER_SOCK).unlink(missing_ok=True)


async def _write_worker_frame(frame: dict[str, Any]) -> None:
    """Write one JSONL frame to the worker."""
    if _worker_writer is None or _worker_writer.is_closing() or not _worker_ready:
        raise ConnectionError("Hermes worker is not connected")

    payload = (json.dumps(frame) + "\n").encode()
    try:
        _worker_writer.write(payload)
        await _worker_writer.drain()
    except OSError as exc:
        await _shutdown_worker_bridge()
        raise ConnectionError("Hermes worker connection dropped") from exc


async def send_message(text: str) -> dict[str, str]:
    """Send a user or daemon message to the Hermes worker."""
    global _worker_send_lock

    if _worker_send_lock is None:
        _worker_send_lock = asyncio.Lock()

    async with _worker_send_lock:
        if not is_connected():
            raise ConnectionError("Hermes worker not initialized")

        had_pending_run = bool(_pending_run_ids)
        run_id = str(uuid.uuid4())

        if had_pending_run:
            print(
                "[gaia2-adapter] Interrupting active Hermes run before new message "
                f"(next={run_id})"
            )
            await _write_worker_frame({"type": "interrupt", "text": text})

        await _write_worker_frame({"type": "message", "text": text, "run_id": run_id})
        _pending_run_ids.append(run_id)

    return {"run_id": run_id}


def is_connected() -> bool:
    return (
        _worker_ready and _worker_writer is not None and not _worker_writer.is_closing()
    )


def get_health_info() -> dict[str, Any]:
    return {
        "backend": "hermes",
        "model": os.environ.get("MODEL", "claude-opus-4-6"),
        "activeRun": _pending_run_ids[0] if _pending_run_ids else None,
    }


async def backend_connect() -> None:
    """Start the worker socket server and wait for Hermes to connect."""
    global _worker_ready_event, _worker_send_lock, _worker_server

    await _shutdown_worker_bridge()
    _worker_send_lock = asyncio.Lock()
    _worker_ready_event = asyncio.Event()

    sock_path = Path(WORKER_SOCK)
    sock_path.parent.mkdir(parents=True, exist_ok=True)
    sock_path.unlink(missing_ok=True)

    _worker_server = await asyncio.start_unix_server(
        _handle_worker_client,
        path=WORKER_SOCK,
    )
    os.chmod(WORKER_SOCK, 0o666)
    print(f"[gaia2-adapter] Waiting for Hermes worker on {WORKER_SOCK}")

    try:
        await asyncio.wait_for(_worker_ready_event.wait(), timeout=WORKER_READY_TIMEOUT)
    except asyncio.TimeoutError as exc:
        await _shutdown_worker_bridge()
        raise TimeoutError(
            f"Hermes worker did not connect within {WORKER_READY_TIMEOUT:.0f}s"
        ) from exc


# ═══════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════


def _on_notify_sent(text: str) -> None:
    write_aui_event("send_message_to_agent", text)


async def _execute_action(app: str, action: str, args: dict, event_id: str) -> dict:
    return execute_cli_action(app, action, args, event_id, state_dir=STATE_DIR)


async def main():
    handler = create_client_handler(
        state=_state,
        send_message=send_message,
        is_connected=is_connected,
        get_health_info=get_health_info,
        on_notify_sent=_on_notify_sent,
        execute_action=_execute_action,
    )

    await run_adapter(_state, handler, backend_connect, backend_name="Hermes worker")


if __name__ == "__main__":
    asyncio.run(main())
