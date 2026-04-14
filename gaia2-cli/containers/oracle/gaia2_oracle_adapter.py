#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
"""Oracle adapter — replays ground-truth actions and serves HTTP status.

Starts a minimal HTTP server on :8090 (same endpoints as OC/Hermes adapters)
and runs replay_scenario(with_daemon=False) in a background thread. The daemon
runs as a separate process started by gaia2-init-entrypoint.sh (same architecture
as OC/Hermes). The runner interacts with this container identically to a real
agent container.

Endpoints:
    GET  /health  — always returns ok (no agent backend to check)
    GET  /status  — reads daemon_status.json (written by gaia2-eventd)
    POST /notify  — accepts but ignores (daemon sends task, oracle ignores it)
    POST /send_user_message — same as /notify
    POST /execute_action — execute ENV actions via CLI tools (from gaia2-eventd)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import threading
import time

# gaia2_adapter_base is copied alongside this file in /opt/
sys.path.insert(0, "/opt")
from gaia2_adapter_base import (
    http_response,
    read_http_request,
)

logger = logging.getLogger(__name__)

STATE_DIR = os.environ.get("GAIA2_STATE_DIR", "/var/gaia2/state")
SCENARIO = os.environ.get("GAIA2_SCENARIO", "/var/gaia2/custom_scenario.json")
STATUS_FILE = os.path.join(STATE_DIR, "daemon_status.json")


def _wait_for_daemon_ready(timeout: float = 30.0) -> bool:
    """Poll daemon_status.json until the daemon reports 'running'."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with open(STATUS_FILE) as f:
                data = json.load(f)
            status = data.get("status", "")
            if status in ("running", "complete", "stopped", "error"):
                logger.info("Daemon ready (status=%s)", status)
                return True
        except (FileNotFoundError, json.JSONDecodeError):
            pass
        time.sleep(0.2)
    logger.warning("Daemon not ready after %.0fs", timeout)
    return False


def _wait_for_daemon_complete(timeout: float = 120.0) -> dict:
    """Poll daemon_status.json until the daemon reaches a terminal state."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with open(STATUS_FILE) as f:
                data = json.load(f)
            status = data.get("status", "")
            if status in ("complete", "stopped", "error"):
                logger.info("Daemon finished (status=%s)", status)
                return data
        except (FileNotFoundError, json.JSONDecodeError):
            pass
        time.sleep(0.5)
    logger.warning("Daemon did not complete after %.0fs", timeout)
    return {"status": "timeout"}


def _run_replay() -> None:
    """Run oracle replay in a background thread.

    The daemon runs as a separate process (started by gaia2-init-entrypoint.sh).
    We use with_daemon=False so replay just executes CLI tools, writing events
    to events.jsonl. The daemon independently watches events.jsonl, detects
    turn boundaries, fires ENV reactions, and runs the judge.
    """
    from gaia2_cli.daemon.replay import replay_scenario

    turn_delay = float(os.environ.get("GAIA2_ORACLE_TURN_DELAY", "2.0"))
    time_speed_str = os.environ.get("GAIA2_TIME_SPEED")
    time_speed = float(time_speed_str) if time_speed_str else None

    # Wait for the daemon process to be ready before replaying events.
    logger.info("Waiting for daemon to be ready...")
    if not _wait_for_daemon_ready():
        logger.error("Daemon never became ready — aborting replay")
        try:
            with open(STATUS_FILE, "w") as f:
                json.dump({"status": "error"}, f)
        except Exception:
            pass
        return

    logger.info(
        "Starting oracle replay: %s (turn_delay=%.1fs, time_speed=%s)",
        SCENARIO,
        turn_delay,
        time_speed,
    )
    try:
        ok = replay_scenario(
            scenario=SCENARIO,
            state_dir=STATE_DIR,
            turn_delay=turn_delay,
            with_daemon=False,
            external_daemon=True,
            verbose=False,
            quiet=True,
            time_speed=time_speed,
            skip_init=True,
        )
        logger.info("Oracle replay finished: %s", "ok" if ok else "error")
    except Exception as exc:
        logger.error("Oracle replay failed: %s", exc, exc_info=True)

    # Replay is done — wait for daemon to finish judging.
    # The daemon detects the final turn boundary and runs judge_final_turn()
    # before self-shutting down.
    logger.info("Replay complete, waiting for daemon to finish judging...")
    _wait_for_daemon_complete(timeout=120.0)


async def _handle_client(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
) -> None:
    """Handle a single HTTP request."""
    try:
        method, path, headers, body = await read_http_request(reader)
        if method is None:
            writer.close()
            return

        route = path.split("?")[0] if path else ""

        if method == "GET" and route == "/health":
            http_response(writer, 200, {"ok": True, "connected": True, "oracle": True})

        elif method == "GET" and route == "/status":
            try:
                with open(STATUS_FILE) as f:
                    status = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                status = {"status": "waiting"}
            http_response(writer, 200, status)

        elif method == "POST" and route == "/execute_action":
            # ENV action dispatch from gaia2-eventd — execute via CLI tools.
            try:
                data = json.loads(body)
                from gaia2_cli.daemon.cli_executor import execute_cli_action

                result = execute_cli_action(
                    app=data.get("app", ""),
                    action=data.get("action", ""),
                    args=data.get("args", {}),
                    event_id=data.get("event_id", ""),
                    state_dir=STATE_DIR,
                )
                http_response(writer, 200, result)
            except Exception as exc:
                http_response(writer, 500, {"ok": False, "error": str(exc)})

        elif method == "POST" and route in (
            "/notify",
            "/send_user_message",
            "/send_notifications",
        ):
            # Accept daemon messages but don't act on them — oracle drives actions.
            http_response(writer, 200, {"ok": True})

        else:
            http_response(writer, 404, {"error": "Not found"})

        await writer.drain()
        writer.close()

    except OSError:
        pass
    except Exception as exc:
        logger.error("Request error: %s", exc)
        try:
            writer.close()
        except Exception:
            pass


async def _main() -> None:
    port = int(os.environ.get("GAIA2_ADAPTER_PORT", "8090"))

    # Start HTTP server first so the runner can connect.
    server = await asyncio.start_server(_handle_client, "0.0.0.0", port)
    logger.info("Oracle adapter listening on 0.0.0.0:%d", port)

    # Start replay in a background thread (it blocks on subprocess calls).
    replay_thread = threading.Thread(
        target=_run_replay, daemon=True, name="oracle-replay"
    )
    replay_thread.start()

    async with server:
        await server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )
    asyncio.run(_main())
