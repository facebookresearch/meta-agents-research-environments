#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
"""Gaia2 Adapter for OpenClaw: HTTP bridge + OpenClaw WebSocket client.

Inbound  (Gaia2 -> Agent):  POST /notify   — sends user messages to OpenClaw
Outbound (Agent -> Gaia2):  GET  /events   — SSE stream of agent responses
                            GET  /messages — poll buffered agent responses (?after=<seq>)
                            GET  /health   — connection status
                            GET  /status   — daemon scenario progress

OpenClaw protocol:
    1. WebSocket connect to ws://host:18789
    2. Gateway sends connect.challenge event
    3. Client responds with connect RPC (token auth, role, scopes)
    4. Gateway responds with hello-ok
    5. chat.send — send user messages (sessionKey, message, idempotencyKey)
    6. event:chat — terminal responses (state: final/error/aborted)

Uses only asyncio + websockets (no framework dependencies).
"""

import asyncio
import json
import os
import sys
import uuid
from pathlib import Path
from typing import Any, Callable

import websockets
from websockets.exceptions import ConnectionClosed

# Shared adapter base — HTTP server, message buffer, SSE, route dispatch
# In Docker: both files are in /opt/. Locally: shared/ is a sibling directory.
_this_dir = os.path.dirname(os.path.abspath(__file__))
for _p in [_this_dir, os.path.join(_this_dir, "..", "shared")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
from gaia2_adapter_base import (  # noqa: E402
    AdapterState,
    create_client_handler,
    handle_notify,
    http_response,
    run_adapter,
    write_aui_event,
)
from gaia2_cli.daemon.cli_executor import execute_cli_action  # noqa: E402

# ── Configuration ───────────────────────────────────────────────────────
NOTIFICATION_MODE = os.environ.get("GAIA2_NOTIFICATION_MODE", "message")
GATEWAY_PORT = os.environ.get("OPENCLAW_GATEWAY_PORT", "18789")
HOOKS_TOKEN = os.environ.get("OPENCLAW_HOOKS_TOKEN", "gaia2-hooks")
STATE_DIR = os.environ.get("GAIA2_STATE_DIR", "/var/gaia2/state")

# ── Backend state ───────────────────────────────────────────────────────
_ws = None
_connected = False
_rpc_id = 0
_reconnect_task = None
_listen_task: asyncio.Task | None = None

_gateway_url = ""
_session_key = ""
_auth_token = ""
_latest_chat_delta: dict[str, str] = {}

_pending: dict[str, asyncio.Future] = {}

# Shared adapter state
_state = AdapterState(
    buffer_size=int(os.environ.get("GAIA2_BUFFER_SIZE", "200")),
)


# ═══════════════════════════════════════════════════════════════════════
#  OpenClaw WebSocket backend
# ═══════════════════════════════════════════════════════════════════════


def _next_rpc_id() -> str:
    global _rpc_id
    _rpc_id += 1
    return str(_rpc_id)


async def _send_rpc(
    method: str, params: dict | None = None, timeout: float = 30.0
) -> dict:
    """Send a JSON-RPC request and wait for the response."""
    if _ws is None:
        raise ConnectionError("Not connected to OpenClaw gateway")

    rpc_id = _next_rpc_id()
    frame: dict[str, Any] = {"type": "req", "id": rpc_id, "method": method}
    if params is not None:
        frame["params"] = params

    loop = asyncio.get_running_loop()
    future = loop.create_future()
    _pending[rpc_id] = future

    try:
        await _ws.send(json.dumps(frame))
        return await asyncio.wait_for(future, timeout=timeout)
    except asyncio.TimeoutError:
        _pending.pop(rpc_id, None)
        raise TimeoutError(f"RPC timeout ({timeout}s) for {method}")
    except Exception:
        _pending.pop(rpc_id, None)
        raise


async def _handle_challenge(ws) -> None:
    """Respond to connect.challenge with auth credentials."""
    print("[openclaw] Got challenge, authenticating...")
    await ws.send(
        json.dumps(
            {
                "type": "req",
                "id": "auth-1",
                "method": "connect",
                "params": {
                    "minProtocol": 3,
                    "maxProtocol": 3,
                    "client": {
                        "id": "openclaw-tui",
                        "version": "1.0.0",
                        "platform": "linux",
                        "mode": "ui",
                    },
                    "role": "operator",
                    "scopes": ["operator.read", "operator.write"],
                    "caps": [],
                    "commands": [],
                    "permissions": {},
                    "auth": {"token": _auth_token},
                    "locale": "en-US",
                    "userAgent": "gaia2-adapter/1.0.0",
                },
            }
        )
    )


def _message_text(message: Any) -> str:
    """Extract visible assistant text from the current OpenClaw message shape."""
    if isinstance(message, str):
        return message
    if not isinstance(message, dict):
        return ""

    for value in (message.get("text"), message.get("content")):
        if isinstance(value, str):
            return value

    content = message.get("content")
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""

    return "".join(
        item
        if isinstance(item, str)
        else item.get("text", "")
        if isinstance(item, dict)
        else ""
        for item in content
    )


def _is_tool_message(message: Any) -> bool:
    """Return True when the payload is a tool-use event, not a user reply."""
    if not isinstance(message, dict):
        return False

    for key in (
        "toolCalls",
        "tool_calls",
        "toolUses",
        "tool_uses",
        "toolUse",
        "tool_use",
    ):
        if message.get(key):
            return True

    content = message.get("content")
    if not isinstance(content, list):
        return False

    for item in content:
        if not isinstance(item, dict):
            continue
        if item.get("type") in (
            "toolCall",
            "tool_call",
            "toolcall",
            "toolUse",
            "tool_use",
            "toolResult",
            "tool_result",
        ):
            return True
        for key in ("toolCall", "tool_call", "toolCalls", "tool_calls"):
            if item.get(key):
                return True
    return False


def _handle_chat_event(payload: dict) -> None:
    """Track chat deltas and emit the terminal assistant reply."""
    state = payload.get("state")

    if state == "delta":
        run_id = payload.get("runId", "")
        if isinstance(run_id, str) and run_id:
            text = _message_text(payload.get("message", "")).strip()
            if text:
                _latest_chat_delta[run_id] = text
        return

    if state not in ("final", "error", "aborted"):
        return
    if not _on_response:
        return

    resp = {
        "run_id": payload.get("runId", ""),
        "runId": payload.get("runId", ""),
        "sessionKey": payload.get("sessionKey", ""),
        "state": state,
        "message": payload.get("message", ""),
    }
    if payload.get("errorMessage"):
        resp["errorMessage"] = payload["errorMessage"]
    if payload.get("usage"):
        resp["usage"] = payload["usage"]

    run_id = resp.get("runId", "")
    cached_delta = ""
    if isinstance(run_id, str) and run_id:
        cached_delta = _latest_chat_delta.pop(run_id, "")

    # Ignore tool-use terminal events. The user-visible assistant reply should
    # arrive as a later final event after the tool loop completes.
    if state == "final" and _is_tool_message(resp["message"]):
        print(
            "[openclaw] Ignoring tool-use chat event "
            f"(runId={resp.get('runId', '')}, sessionKey={resp.get('sessionKey', '')})"
        )
        return

    # Some runs end with an empty final payload even though the last delta
    # already contains the full user-visible answer. Reuse that delta text.
    if state == "final" and cached_delta:
        inline_text = _message_text(resp["message"]).strip()
        if not inline_text or inline_text != cached_delta:
            resp["message"] = {
                "role": "assistant",
                "content": [{"type": "text", "text": cached_delta}],
            }

    _on_response(resp)


async def _dispatch_message(ws, msg: dict) -> None:
    """Route a single parsed WebSocket message to the appropriate handler."""
    global _connected

    msg_payload = msg.get("payload")
    # Normalize: some gateway messages send payload as a string
    if isinstance(msg_payload, str):
        try:
            msg_payload = json.loads(msg_payload)
        except (json.JSONDecodeError, ValueError):
            msg_payload = None

    # Handle connect.challenge -> send connect auth
    if (
        msg.get("event") == "connect.challenge"
        or msg.get("method") == "connect.challenge"
    ):
        await _handle_challenge(ws)
        return

    # Handle hello-ok -> mark as connected
    if (
        msg.get("id") == "auth-1"
        and msg.get("ok")
        and isinstance(msg_payload, dict)
        and msg_payload.get("type") == "hello-ok"
    ):
        print("[openclaw] Authenticated and connected.")
        _connected = True
        return

    # Auth failure
    if msg.get("id") == "auth-1" and msg.get("ok") is False:
        print(f"[openclaw] Auth failed: {json.dumps(msg.get('error'))}")
        _connected = False
        await ws.close()
        return

    # Broadcast events from the gateway
    if msg.get("type") == "event" and msg.get("event") == "chat":
        if isinstance(msg_payload, dict):
            _handle_chat_event(msg_payload)
        return

    # RPC responses (chat.send, etc.)
    if msg.get("id") and msg["id"] in _pending:
        future = _pending.pop(msg["id"])
        if not future.done():
            future.set_result(msg)


async def _listen(ws) -> None:
    """Listen for messages from the OpenClaw gateway."""
    try:
        async for raw in ws:
            try:
                msg = json.loads(raw)
            except (json.JSONDecodeError, ValueError):
                continue
            if not isinstance(msg, dict):
                continue
            await _dispatch_message(ws, msg)
    except ConnectionClosed:
        pass
    except Exception as e:
        print(f"[openclaw] Listen error: {e}")


async def _ws_connect() -> None:
    """Establish WebSocket connection to OpenClaw gateway."""
    global _ws, _connected, _listen_task

    print(f"[openclaw] Connecting to {_gateway_url}...")

    try:
        _ws = await websockets.connect(_gateway_url, max_size=10_485_760, proxy=None)
        print("[openclaw] WebSocket open, waiting for challenge...")

        # Start listener — it handles the challenge/auth handshake
        _listen_task = asyncio.create_task(_listen(_ws))

    except Exception as e:
        print(f"[openclaw] Connection failed: {e}")
        _connected = False
        _schedule_reconnect()


def _schedule_reconnect() -> None:
    global _reconnect_task
    if _reconnect_task and not _reconnect_task.done():
        return

    async def _reconnect():
        global _reconnect_task
        await asyncio.sleep(2)
        _reconnect_task = None
        await _ws_connect()

    _reconnect_task = asyncio.create_task(_reconnect())


async def _on_disconnect() -> None:
    """Handle WebSocket disconnection."""
    global _connected
    print("[openclaw] WebSocket closed.")
    _connected = False
    _latest_chat_delta.clear()

    for _rpc_id, future in list(_pending.items()):
        if not future.done():
            future.set_exception(ConnectionError("WebSocket closed"))
    _pending.clear()

    _schedule_reconnect()


def _resolve_auth_token() -> str:
    """Read auth token from env or openclaw config file."""
    token = os.environ.get("OPENCLAW_GATEWAY_TOKEN", "")
    if token:
        return token
    try:
        config_path = Path.home() / ".openclaw" / "openclaw.json"
        cfg = json.loads(config_path.read_text())
        return cfg.get("gateway", {}).get("auth", {}).get("token", "")
    except Exception:
        return ""


async def backend_connect() -> None:
    global _gateway_url, _session_key, _auth_token
    _gateway_url = os.environ.get(
        "OPENCLAW_GATEWAY_URL",
        f"ws://127.0.0.1:{os.environ.get('OPENCLAW_GATEWAY_PORT', '18789')}",
    )
    _session_key = os.environ.get("GAIA2_SESSION_KEY", "agent:main:main")
    _auth_token = _resolve_auth_token()

    # Don't block on initial connection — the gateway may not be up yet
    # (adapter starts as gaia2 before gateway starts as agent).
    # The reconnect loop will keep retrying in the background.
    asyncio.ensure_future(_ws_connect())

    async def _monitor():
        while True:
            if _ws is not None:
                try:
                    await _ws.wait_closed()
                except Exception:
                    pass
                await _on_disconnect()
            await asyncio.sleep(1)

    asyncio.create_task(_monitor())


async def send_message(text: str) -> dict:
    if not _connected or _ws is None:
        raise ConnectionError("Not connected to OpenClaw gateway")

    res = await _send_rpc(
        "chat.send",
        {
            "sessionKey": _session_key,
            "message": text,
            "idempotencyKey": str(uuid.uuid4()),
        },
    )

    if res.get("ok") is False:
        exc = ConnectionError("Gateway rejected message")
        exc.detail = res.get("error")  # type: ignore[attr-defined]
        raise exc

    return {"run_id": res.get("payload", {}).get("runId", "")}


def is_connected() -> bool:
    return _connected


def get_health_info() -> dict:
    return {
        "backend": "openclaw",
        "gateway": _gateway_url,
        "sessionKey": _session_key,
    }


# ═══════════════════════════════════════════════════════════════════════
#  OpenClaw-specific: response handling
# ═══════════════════════════════════════════════════════════════════════


async def _send_wake_hook(text: str) -> None:
    """POST to OpenClaw /hooks/wake with a system event.

    The wake hook enqueues the text as a system event in the main session
    and triggers a heartbeat. The agent sees the notification as a
    ``System: [timestamp] <text>`` line prepended to the heartbeat prompt,
    with full conversation context from the main session transcript.
    """
    import urllib.request

    url = f"http://127.0.0.1:{GATEWAY_PORT}/hooks/wake"
    payload = json.dumps({"text": text, "mode": "now"}).encode()
    req = urllib.request.Request(
        url,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {HOOKS_TOKEN}",
        },
    )
    loop = asyncio.get_running_loop()
    resp_data = await loop.run_in_executor(
        None, lambda: urllib.request.urlopen(req, timeout=10).read()
    )
    result = json.loads(resp_data)
    if not result.get("ok"):
        raise ConnectionError(f"Wake hook rejected: {result}")
    print(f"[gaia2-adapter] Wake hook sent ({len(text)} chars)")


def _response_text(response: dict) -> str:
    """Extract plain text from a buffered OpenClaw response."""
    return _message_text(response.get("message", ""))


def _terminal_response_text(response: dict) -> str:
    """Extract the terminal user-visible text for final/error responses."""
    text = _response_text(response)
    if text.strip():
        return text
    error_text = response.get("errorMessage", "")
    return error_text if isinstance(error_text, str) else ""


_on_response: Callable | None = None


def on_backend_response(response: dict) -> None:
    state = response.get("state", "")
    run_id = response.get("run_id", response.get("runId", ""))

    # Drop OpenClaw internal responses — heartbeat acks, memory compaction
    # flushes, and empty turns that are not part of the scenario.
    # NOTE: Do NOT drop empty text ("") — that's a real agent completion
    # when maxToolRoundtrips is exhausted without producing text output.
    text_preview = _response_text(response).strip()
    if state == "final" and text_preview in ("HEARTBEAT_OK", "NO_REPLY"):
        print(f"[gaia2-adapter] Dropped internal ack '{text_preview}' (runId={run_id})")
        return

    entry = _state.buffer_and_broadcast(response)
    print(f"[gaia2-adapter] Buffered {state} message seq={entry['seq']} runId={run_id}")

    # Write AUI.send_message_to_user for terminal states so the daemon
    # sees a turn boundary in events.jsonl.
    # Final runs can legitimately finish with empty assistant text after a
    # tool-use turn. Emit that empty boundary so the daemon can still close
    # the turn cleanly.
    if state == "final":
        write_aui_event("send_message_to_user", _response_text(response))
    elif state in ("error", "aborted"):
        text = _terminal_response_text(response)
        if text.strip():
            write_aui_event("send_message_to_user", text)


# ═══════════════════════════════════════════════════════════════════════
#  OpenClaw-specific extra routes
# ═══════════════════════════════════════════════════════════════════════


async def _extra_routes(method, route, parsed, reader, writer, body) -> bool:
    """Handle POST /send_notifications (OpenClaw-specific)."""
    if method == "POST" and route == "/send_notifications":
        if NOTIFICATION_MODE == "native":
            # Native mode: use /hooks/wake instead of chat.send
            try:
                data = json.loads(body)
            except (json.JSONDecodeError, ValueError):
                http_response(writer, 400, {"ok": False, "error": "Invalid JSON"})
                await writer.drain()
                writer.close()
                return True
            text = data.get("message", "").strip()
            if not text:
                http_response(writer, 400, {"ok": False, "error": "Empty message"})
                await writer.drain()
                writer.close()
                return True
            try:
                await _send_wake_hook(text)
                http_response(writer, 200, {"ok": True, "mode": "native"})
            except Exception as e:
                http_response(writer, 502, {"ok": False, "error": str(e)})
            await writer.drain()
            writer.close()
            return True
        else:
            await handle_notify(
                writer,
                body,
                send_message,
                is_connected,
                on_sent=lambda text: write_aui_event(
                    "send_message_to_agent", text[:500]
                ),
            )
            await writer.drain()
            writer.close()
            return True
    return False


def _on_notify_sent(text: str) -> None:
    """Called after POST /notify successfully sends a message."""
    write_aui_event("send_message_to_agent", text)


# ═══════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════


async def _execute_action(app: str, action: str, args: dict, event_id: str) -> dict:
    """Route ENV actions to CLI tools."""
    return execute_cli_action(app, action, args, event_id, state_dir=STATE_DIR)


async def main():
    global _on_response
    _on_response = on_backend_response

    handler = create_client_handler(
        state=_state,
        send_message=send_message,
        is_connected=is_connected,
        get_health_info=get_health_info,
        extra_routes=_extra_routes,
        on_notify_sent=_on_notify_sent,
        execute_action=_execute_action,
    )

    await run_adapter(_state, handler, backend_connect, backend_name="OpenClaw")


if __name__ == "__main__":
    asyncio.run(main())
