#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
"""Shared HTTP adapter base for Gaia2 agent backends (OpenClaw, Hermes).

Provides the HTTP server layer, message buffering, SSE broadcasting,
and route dispatch. Backend-specific adapters import from here and
supply their own ``send_message``, ``is_connected``, ``get_health_info``,
and ``on_backend_response`` implementations.

Usage in a backend adapter::

    from gaia2_adapter_base import AdapterState, create_client_handler, run_adapter

    state = AdapterState()

    async def my_send_message(text: str) -> dict: ...
    def my_is_connected() -> bool: ...
    def my_get_health_info() -> dict: ...
    def my_on_response(response: dict) -> None:
        state.buffer_and_broadcast(response)

    handler = create_client_handler(
        state=state,
        send_message=my_send_message,
        is_connected=my_is_connected,
        get_health_info=my_get_health_info,
        extra_routes=my_extra_routes,  # optional
    )

    await run_adapter(state, handler)
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time as _time
from collections import deque
from datetime import datetime, timezone
from typing import Any, Callable, Coroutine
from urllib.parse import parse_qs, urlparse

# ── Configuration ───────────────────────────────────────────────────────
STATUS_FILE = "daemon_status.json"
STATE_DIR = os.environ.get("GAIA2_STATE_DIR", "/var/gaia2/state")
EVENTS_JSONL = os.environ.get(
    "GAIA2_EVENTS_JSONL",
    os.path.join(STATE_DIR, "events.jsonl"),
)


# ═══════════════════════════════════════════════════════════════════════
#  AUI event helper (shared across all adapters)
# ═══════════════════════════════════════════════════════════════════════


def write_aui_event(fn: str, content: str) -> None:
    """Append an AgentUserInterface event to events.jsonl.

    These synthetic tool-call entries let the daemon detect turn boundaries
    (send_message_to_user) and track messages sent to the agent
    (send_message_to_agent).  Format matches ``gaia2_cli.shared.log_action()``.
    """
    # send_message_to_user is an agent write action (turn boundary).
    # send_message_to_agent is a user/system message TO the agent — not
    # an agent action, so it's classified as a read to match the original
    # Gaia2 framework's AgentEventFilter (operation_type=WRITE only).
    entry = {
        "t": _time.time(),
        "app": "AgentUserInterface",
        "fn": fn,
        "args": {"content": content},
        "w": fn != "send_message_to_agent",
        "ret": "Message sent.",
    }
    # Include simulated time from faketime.rc if available
    try:
        with open("/tmp/faketime.rc") as f:
            entry["sim_t"] = f.read().strip()
    except OSError:
        pass
    try:
        with open(EVENTS_JSONL, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except OSError as e:
        print(f"[gaia2-adapter] Warning: could not write AUI event: {e}")


# ═══════════════════════════════════════════════════════════════════════
#  Adapter state (message buffer, SSE clients)
# ═══════════════════════════════════════════════════════════════════════


class AdapterState:
    """Shared mutable state for the HTTP adapter layer."""

    def __init__(self, buffer_size: int = 200) -> None:
        self.message_buffer: deque = deque(maxlen=buffer_size)
        self.message_seq: int = 0
        self.sse_clients: set[asyncio.StreamWriter] = set()
        self.port: int = int(os.environ.get("GAIA2_ADAPTER_PORT", "8090"))

    def buffer_message(self, response: dict) -> dict:
        self.message_seq += 1
        entry = {
            "seq": self.message_seq,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **response,
        }
        self.message_buffer.append(entry)
        return entry

    async def broadcast_to_sse(self, entry: dict) -> None:
        data = json.dumps(entry)
        payload = f"event: chat\ndata: {data}\n\n".encode()
        dead = set()
        for writer in self.sse_clients:
            try:
                writer.write(payload)
                await writer.drain()
            except OSError:
                dead.add(writer)
        for w in dead:
            self.sse_clients.discard(w)
            try:
                w.close()
            except Exception:
                pass

    def buffer_and_broadcast(self, response: dict) -> dict:
        """Buffer a response and schedule SSE broadcast. Returns the entry."""
        entry = self.buffer_message(response)
        asyncio.ensure_future(self.broadcast_to_sse(entry))
        return entry


# ═══════════════════════════════════════════════════════════════════════
#  HTTP utilities
# ═══════════════════════════════════════════════════════════════════════


async def read_http_request(reader: asyncio.StreamReader):
    """Parse an HTTP/1.1 request. Returns (method, path, headers, body)."""
    request_line = await reader.readline()
    if not request_line:
        return None, None, {}, b""
    parts = request_line.decode("utf-8", errors="replace").strip().split(" ", 2)
    if len(parts) < 2:
        return None, None, {}, b""
    method, path = parts[0], parts[1]

    headers = {}
    while True:
        line = await reader.readline()
        if line in (b"\r\n", b"\n", b""):
            break
        decoded = line.decode("utf-8", errors="replace").strip()
        if ": " in decoded:
            key, value = decoded.split(": ", 1)
            headers[key.lower()] = value

    content_length = int(headers.get("content-length", "0"))
    body = b""
    if content_length > 0:
        body = await reader.readexactly(min(content_length, 1_048_576))

    return method, path, headers, body


def http_response(
    writer: asyncio.StreamWriter,
    status: int,
    body: dict | str,
    content_type: str = "application/json",
    extra_headers: dict | None = None,
) -> None:
    """Write a complete HTTP response and close."""
    if isinstance(body, dict):
        payload = json.dumps(body).encode()
    else:
        payload = body.encode() if isinstance(body, str) else body

    status_text = {
        200: "OK",
        400: "Bad Request",
        404: "Not Found",
        502: "Bad Gateway",
        503: "Service Unavailable",
    }.get(status, "Error")

    lines = [
        f"HTTP/1.1 {status} {status_text}",
        f"Content-Type: {content_type}",
        f"Content-Length: {len(payload)}",
        "Connection: close",
    ]
    if extra_headers:
        for k, v in extra_headers.items():
            lines.append(f"{k}: {v}")
    lines.append("")
    lines.append("")
    writer.write("\r\n".join(lines).encode() + payload)


# ═══════════════════════════════════════════════════════════════════════
#  Shared route handlers
# ═══════════════════════════════════════════════════════════════════════


async def handle_health(
    writer: asyncio.StreamWriter,
    state: AdapterState,
    is_connected: Callable[[], bool],
    get_health_info: Callable[[], dict],
) -> None:
    """Handle GET /health."""
    health = {
        "ok": is_connected(),
        "connected": is_connected(),
        **get_health_info(),
        "buffer": {"size": len(state.message_buffer), "lastSeq": state.message_seq},
        "sseClients": len(state.sse_clients),
    }
    http_response(writer, 200, health)


async def handle_status(writer: asyncio.StreamWriter) -> None:
    """Handle GET /status — read daemon_status.json from state dir."""
    status_path = os.path.join(STATE_DIR, STATUS_FILE)
    try:
        with open(status_path) as f:
            status = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        status = {"status": "waiting"}
    http_response(writer, 200, status)


async def handle_messages(
    writer: asyncio.StreamWriter,
    state: AdapterState,
    query_string: str,
) -> None:
    """Handle GET /messages?after=<seq>."""
    qs = parse_qs(query_string)
    after_seq = int(qs.get("after", ["0"])[0])
    messages = [m for m in state.message_buffer if m["seq"] > after_seq]
    http_response(
        writer,
        200,
        {"ok": True, "messages": messages, "lastSeq": state.message_seq},
    )


async def handle_notify(
    writer: asyncio.StreamWriter,
    body: bytes,
    send_message: Callable[[str], Coroutine[Any, Any, dict]],
    is_connected: Callable[[], bool],
    on_sent: Callable[[str], None] | None = None,
) -> None:
    """Handle POST /notify — send a user message to the agent.

    *on_sent* is called after a successful send with the message text
    (e.g. to write an AUI event to events.jsonl).
    """
    if not is_connected():
        http_response(
            writer, 503, {"ok": False, "error": "Not connected to agent backend"}
        )
        return

    try:
        data = json.loads(body)
    except (json.JSONDecodeError, ValueError):
        http_response(writer, 400, {"ok": False, "error": "Invalid JSON"})
        return

    message = data.get("message", "")
    if not isinstance(message, str) or not message.strip():
        http_response(
            writer, 400, {"ok": False, "error": 'Missing or empty "message" field'}
        )
        return

    try:
        result = await send_message(message.strip())
        if on_sent:
            on_sent(message.strip())
        http_response(writer, 200, {"ok": True, "runId": result.get("run_id", "")})
    except Exception as e:
        detail = getattr(e, "detail", None)
        status = 502 if detail else 503
        resp: dict[str, Any] = {"ok": False, "error": str(e)}
        if detail:
            resp["detail"] = detail
        http_response(writer, status, resp)


async def handle_sse(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    state: AdapterState,
) -> None:
    """Handle GET /events — SSE stream of agent responses."""
    header = (
        "HTTP/1.1 200 OK\r\n"
        "Content-Type: text/event-stream\r\n"
        "Cache-Control: no-cache\r\n"
        "Connection: keep-alive\r\n"
        "X-Accel-Buffering: no\r\n"
        "\r\n"
        ": connected to gaia2-adapter SSE stream\n\n"
    )
    writer.write(header.encode())
    await writer.drain()
    state.sse_clients.add(writer)
    print(f"[gaia2-adapter] SSE client connected ({len(state.sse_clients)} total)")
    try:
        while not reader.at_eof():
            await asyncio.sleep(1)
    except OSError:
        pass
    finally:
        state.sse_clients.discard(writer)
        print(
            f"[gaia2-adapter] SSE client disconnected ({len(state.sse_clients)} total)"
        )
        try:
            writer.close()
        except Exception:
            pass


# ═══════════════════════════════════════════════════════════════════════
#  Client handler factory & server runner
# ═══════════════════════════════════════════════════════════════════════

# Type for extra route handlers: async (method, route, parsed, reader, writer, body) -> bool
# Returns True if handled, False to fall through to 404.
ExtraRouteHandler = Callable[
    [str, str, Any, asyncio.StreamReader, asyncio.StreamWriter, bytes],
    Coroutine[Any, Any, bool],
]


# Type for execute_action callback:
#   async (app, action, args, event_id) -> {"ok": bool, "result"?: str, "error"?: str}
ExecuteActionHandler = Callable[
    [str, str, dict, str],
    Coroutine[Any, Any, dict],
]


def create_client_handler(
    state: AdapterState,
    send_message: Callable[[str], Coroutine[Any, Any, dict]],
    is_connected: Callable[[], bool],
    get_health_info: Callable[[], dict],
    extra_routes: ExtraRouteHandler | None = None,
    on_notify_sent: Callable[[str], None] | None = None,
    execute_action: ExecuteActionHandler | None = None,
) -> Callable:
    """Create a ``handle_client`` coroutine with the standard route table.

    Routes:
        GET  /health           — connection & buffer status
        GET  /status           — daemon scenario progress (daemon_status.json)
        GET  /messages         — poll buffered agent responses
        GET  /events           — SSE stream
        POST /notify           — send user message to agent
        POST /send_user_message — alias for /notify
        POST /execute_action   — execute an ENV action (eventd → adapter)

    *extra_routes* is called for any route not in the table above.
    *execute_action* handles ENV action dispatch from gaia2-eventd.
    """

    async def _handle_client(
        reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ):
        try:
            method, path, headers, body = await read_http_request(reader)
            if method is None:
                writer.close()
                return

            parsed = urlparse(path)
            route = parsed.path

            # GET /health
            if method == "GET" and route == "/health":
                await handle_health(writer, state, is_connected, get_health_info)
                await writer.drain()
                writer.close()
                return

            # GET /status
            if method == "GET" and route == "/status":
                await handle_status(writer)
                await writer.drain()
                writer.close()
                return

            # POST /execute_action — ENV action dispatch from eventd
            if method == "POST" and route == "/execute_action":
                await _handle_execute_action(writer, body)
                await writer.drain()
                writer.close()
                return

            # POST /notify or /send_user_message or /send_notifications
            if method == "POST" and route in (
                "/notify",
                "/send_user_message",
                "/send_notifications",
            ):
                # Let extra_routes handle /send_notifications if the backend
                # wants to override delivery (e.g. native notification mode).
                if route == "/send_notifications" and extra_routes:
                    handled = await extra_routes(
                        method, route, parsed, reader, writer, body
                    )
                    if handled:
                        return
                await handle_notify(
                    writer, body, send_message, is_connected, on_notify_sent
                )
                await writer.drain()
                writer.close()
                return

            # GET /events — SSE stream
            if method == "GET" and route == "/events":
                await handle_sse(reader, writer, state)
                return

            # GET /messages?after=<seq>
            if method == "GET" and route == "/messages":
                await handle_messages(writer, state, parsed.query)
                await writer.drain()
                writer.close()
                return

            # Backend-specific routes
            if extra_routes:
                handled = await extra_routes(
                    method, route, parsed, reader, writer, body
                )
                if handled:
                    return

            # 404
            http_response(writer, 404, {"error": "Not found"})
            await writer.drain()
            writer.close()

        except OSError:
            pass
        except Exception as e:
            print(f"[gaia2-adapter] Error handling request: {e}", file=sys.stderr)
            try:
                writer.close()
            except Exception:
                pass

    async def _handle_execute_action(writer: asyncio.StreamWriter, body: bytes) -> None:
        """Handle POST /execute_action — dispatch an ENV action."""
        if not execute_action:
            http_response(
                writer, 501, {"ok": False, "error": "execute_action not configured"}
            )
            return
        try:
            data = json.loads(body)
        except (json.JSONDecodeError, ValueError):
            http_response(writer, 400, {"ok": False, "error": "Invalid JSON"})
            return

        app = data.get("app", "")
        action = data.get("action", "")
        args = data.get("args", {})
        event_id = data.get("event_id", "")

        if not app or not action:
            http_response(
                writer,
                400,
                {"ok": False, "error": "Missing 'app' or 'action' field"},
            )
            return

        try:
            result = await execute_action(app, action, args, event_id)
            http_response(writer, 200, result)
        except Exception as e:
            http_response(writer, 500, {"ok": False, "error": str(e)})

    return _handle_client


async def run_adapter(
    state: AdapterState,
    client_handler: Callable,
    backend_connect: Callable[[], Coroutine],
    backend_name: str = "agent",
) -> None:
    """Connect to the backend and start the HTTP server."""
    print(f"[gaia2-adapter] Connecting to {backend_name} backend...")
    await backend_connect()

    server = await asyncio.start_server(client_handler, "0.0.0.0", state.port)
    print(f"[gaia2-adapter] HTTP server listening on 0.0.0.0:{state.port}")
    print(
        "[gaia2-adapter] Endpoints:  POST /notify, POST /execute_action, "
        "GET /events (SSE), GET /messages, GET /health, GET /status"
    )

    async with server:
        await server.serve_forever()
