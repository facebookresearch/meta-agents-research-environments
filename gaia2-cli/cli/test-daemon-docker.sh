#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
# test-daemon-docker.sh — Test gaia2-eventd daemon with Docker container
#
# Architecture:
#   Container (podman):
#     - OpenClaw gateway (port 18789)
#     - gaia2-adapter.mjs (port 8090)
#     - Agent runs as restricted user with Gaia2 CLI tools
#     - State dir: /var/gaia2/state/ (bind-mounted from host)
#
#   Host:
#     - gaia2-eventd daemon in HTTP mode
#     - Reads events.jsonl from bind-mounted state dir
#     - Sends notifications via POST http://127.0.0.1:8090/notify
#     - Polls agent responses via GET http://127.0.0.1:8090/messages
#
# The daemon runs on the host for dev testing convenience — it reads
# events.jsonl from the bind-mounted state dir and forwards notifications
# over HTTP. The container runs only the adapter + agent; the in-container
# daemon is explicitly disabled for this workflow.
#
# Usage:
#   PROVIDER=anthropic MODEL=claude-opus-4-6 ANTHROPIC_API_KEY=<key> \
#     bash test-daemon-docker.sh <scenario.json>
#
#   PROVIDER=openai-compat MODEL=my-model BASE_URL=https://api.example.com/v1 \
#     API_KEY=<key> bash test-daemon-docker.sh <scenario.json>
#
# Monitoring:
#   # Watch daemon logs (live)
#   tail -f /tmp/gaia2-daemon-test/eventd.log
#
#   # Watch events.jsonl (agent tool calls + ENV actions)
#   tail -f /tmp/gaia2-daemon-test/state/events.jsonl | jq .
#
#   # Check adapter health
#   curl -s http://127.0.0.1:8090/health | jq .
#
#   # Poll agent messages manually
#   curl -s 'http://127.0.0.1:8090/messages?after=0' | jq .
#
#   # Send a manual notification (bypass daemon)
#   curl -X POST http://127.0.0.1:8090/notify \
#     -H 'Content-Type: application/json' \
#     -d '{"message":"Hello from test"}'

set -eu

SCENARIO="${1:?Usage: PROVIDER=<provider> MODEL=<model> [API_KEY or provider key] bash $0 <scenario.json>}"
CONTAINER_NAME="gaia2-daemon-test"
IMAGE="${GAIA2_OC_IMAGE:-localhost/gaia2-oc:latest}"
TEST_DIR="/tmp/gaia2-daemon-test"
FAKETIME_FILE="$TEST_DIR/faketime.rc"
GAIA2_CLI_DIR="$(cd "$(dirname "$0")" && pwd)"
PROVIDER="${PROVIDER:?Error: PROVIDER is required.}"
MODEL="${MODEL:?Error: MODEL is required.}"

# ── Setup test directory ──────────────────────────────────────────────────
echo "Setting up test directory: $TEST_DIR"
rm -rf "$TEST_DIR"
mkdir -p "$TEST_DIR/state"

# Initialize Gaia2 state
echo "Initializing Gaia2 state..."
export PATH="$GAIA2_CLI_DIR/.venv/bin:$PATH"
export GAIA2_STATE_DIR="$TEST_DIR/state"
gaia2-init --scenario "$SCENARIO" --state-dir "$TEST_DIR/state"
cp "$SCENARIO" "$TEST_DIR/scenario.json"

# Seed a shared faketime file so the containerized agent and the host daemon
# observe the same simulated clock. The daemon updates this file in-place.
python3 - "$SCENARIO" "$FAKETIME_FILE" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

scenario_path = Path(sys.argv[1])
faketime_path = Path(sys.argv[2])
data = json.loads(scenario_path.read_text())
start_time = data.get("metadata", {}).get("definition", {}).get(
    "start_time",
    data.get("start_time", 0),
)
if start_time and float(start_time) > 0:
    value = datetime.fromtimestamp(float(start_time), tz=timezone.utc).strftime(
        "%Y-%m-%d %H:%M:%S"
    )
    faketime_path.write_text(value + "\n")
else:
    faketime_path.touch()
PY
chmod 644 "$FAKETIME_FILE"

echo "  State dir: $TEST_DIR/state"
echo "  Apps: $(ls "$TEST_DIR/state/"*.json 2>/dev/null | wc -l) state files"

# ── Start container ───────────────────────────────────────────────────────
echo ""
echo "Starting container '$CONTAINER_NAME'..."
podman rm -f "$CONTAINER_NAME" 2>/dev/null || true

# Bind-mount the state directory so both container and host daemon share it
CONTAINER_ARGS=(
    --network=host
    --name "$CONTAINER_NAME"
    -d
    -e "PROVIDER=$PROVIDER"
    -e "MODEL=$MODEL"
    -e OPENCLAW_FORCE_RECONFIG=1
    -e GAIA2_DAEMON_DISABLE=1
    -v "$SCENARIO:/var/gaia2/custom_scenario.json:ro,z"
    -v "$TEST_DIR/state:/var/gaia2/state:z"
    -v "$FAKETIME_FILE:/tmp/faketime.rc:z"
)
for var in \
    API_KEY \
    ANTHROPIC_API_KEY \
    OPENAI_API_KEY \
    OPENAI_COMPAT_API_KEY \
    OPENAI_COMPLETIONS_API_KEY \
    GEMINI_API_KEY \
    GOOGLE_API_KEY \
    OPENROUTER_API_KEY \
    BASE_URL \
    OPENAI_BASE_URL \
    ANTHROPIC_BASE_URL \
    GEMINI_BASE_URL \
    GOOGLE_BASE_URL \
    OPENROUTER_BASE_URL \
    THINKING \
    GAIA2_TRACE_FILE \
; do
    if [ -n "${!var:-}" ]; then
        CONTAINER_ARGS+=(-e "$var=${!var}")
    fi
done
podman run "${CONTAINER_ARGS[@]}" "$IMAGE"

echo "  Waiting for adapter..."
RETRIES=0
while ! curl -sf http://127.0.0.1:8090/health >/dev/null 2>&1; do
    RETRIES=$((RETRIES + 1))
    if [ $RETRIES -ge 60 ]; then
        echo "ERROR: Adapter did not start after 60s"
        echo "Container logs:"
        podman logs "$CONTAINER_NAME" 2>&1 | tail -20
        exit 1
    fi
    sleep 1
done

echo "  Adapter ready: $(curl -s http://127.0.0.1:8090/health | python3 -c 'import sys,json; d=json.load(sys.stdin); print(f"connected={d.get(\"connected\")}")')"

# ── Start daemon on host ──────────────────────────────────────────────────
echo ""
echo "Starting gaia2-eventd daemon (HTTP mode)..."
echo "  Scenario: $SCENARIO"
echo "  State dir: $TEST_DIR/state"
echo "  Adapter: http://127.0.0.1:8090"
echo "  Log: $TEST_DIR/eventd.log"
echo ""

# The daemon runs on the host and forwards notifications to the adapter.
gaia2-eventd \
    --scenario "$SCENARIO" \
    --state-dir "$TEST_DIR/state" \
    --notify-url http://127.0.0.1:8090 \
    --faketime-path "$FAKETIME_FILE" \
    --poll-interval 2.0 \
    2>&1 | tee "$TEST_DIR/eventd.log" &
DAEMON_PID=$!

echo "  Daemon PID: $DAEMON_PID"
echo ""
echo "══════════════════════════════════════════════════════════════"
echo "  DAEMON RUNNING — monitoring commands:"
echo ""
echo "  # Watch daemon output (this terminal)"
echo "  # Press Ctrl+C to stop"
echo ""
echo "  # In another terminal:"
echo "  tail -f $TEST_DIR/eventd.log              # daemon log"
echo "  tail -f $TEST_DIR/state/events.jsonl      # tool calls"
echo "  curl -s 'http://127.0.0.1:8090/messages?after=0' | jq .  # agent msgs"
echo "  curl -s http://127.0.0.1:8090/health | jq .              # health"
echo "══════════════════════════════════════════════════════════════"
echo ""

# Wait for daemon (Ctrl+C to stop)
cleanup() {
    echo ""
    echo "Shutting down..."
    kill $DAEMON_PID 2>/dev/null || true
    wait $DAEMON_PID 2>/dev/null || true
    podman stop "$CONTAINER_NAME" 2>/dev/null || true
    echo ""
    echo "Logs saved to: $TEST_DIR/"
    echo "  eventd.log:     $(wc -l < "$TEST_DIR/eventd.log" 2>/dev/null || echo 0) lines"
    echo "  events.jsonl:   $(wc -l < "$TEST_DIR/state/events.jsonl" 2>/dev/null || echo 0) events"
}
trap cleanup EXIT INT TERM

wait $DAEMON_PID
