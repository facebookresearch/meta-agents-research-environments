#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
# gaia2-init-entrypoint.sh — Runtime scenario init + Gaia2 infrastructure startup.
#
# Runs as root, then drops to agent for the Hermes adapter.
#
# Architecture (privilege separation):
#   GAIA2 side (gaia2 user):  gaia2-eventd + state files + events.jsonl
#   Agent side (agent user): Hermes adapter (in-process AIAgent)
#
# The agent interacts with the simulated environment exclusively through:
#   - CLI tools (via gaia2-exec setuid wrapper → gaia2 user)
#   - Daemon messages (adapter bridges to/from the Gaia2 daemon)
#
# Environment variables:
#   GAIA2_SCENARIO       — path to scenario JSON (default: /var/gaia2/custom_scenario.json)
#   GAIA2_FS_BACKING_DIR — filesystem backing directory (baked into image at /opt/gaia2_filesystem)
#   GAIA2_DAEMON_DISABLE — set to "1" to skip daemon startup
#   GAIA2_ADAPTER_PORT   — adapter HTTP port (default: 8090)
#   GAIA2_JUDGE_BASE_URL — optional API base URL override for the in-container judge
#   GAIA2_JUDGE_API_KEY  — optional API key override for the in-container judge
#   FAKETIME            — simulated start time (e.g., "2025-09-01 07:00:00")
#   PROVIDER/MODEL/API_KEY/BASE_URL/THINKING/MAX_TOKENS — Hermes runtime config

set -eo pipefail

export PATH="/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"

CUSTOM_SCENARIO="${GAIA2_SCENARIO:-/var/gaia2/custom_scenario.json}"
ADAPTER_PORT="${GAIA2_ADAPTER_PORT:-8090}"

# ── 1. Initialise Gaia2 state ──────────────────────────────────────────────
if [ ! -f "$CUSTOM_SCENARIO" ]; then
    echo "[gaia2-init] No scenario found at $CUSTOM_SCENARIO, skipping init" >&2
else
    echo "[gaia2-init] Initialising from $CUSTOM_SCENARIO ..." >&2
    rm -rf /var/gaia2/state/*
    gaia2-init --scenario "$CUSTOM_SCENARIO" \
              --state-dir /var/gaia2/state \
              ${GAIA2_FS_BACKING_DIR:+--fs-backing-dir "$GAIA2_FS_BACKING_DIR"}
    chown -R gaia2:gaia2 /var/gaia2/state
    # Lock down /var/gaia2 — agent must not read scenario (ground truth) or
    # state files (must use CLI tools via gaia2-exec instead).
    chmod 700 /var/gaia2
    chmod -R go-rwx /var/gaia2/state

    # Extract start_time from scenario and set FAKETIME if not already set
    if [ -z "$FAKETIME" ]; then
        FAKETIME=$(python3 -c "
import json, sys
from datetime import datetime, timezone
d = json.load(open('$CUSTOM_SCENARIO'))
st = d.get('metadata', {}).get('definition', {}).get('start_time', d.get('start_time', 0))
if st and float(st) > 0:
    print(datetime.fromtimestamp(float(st), tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S'))
" 2>/dev/null)
        if [ -n "$FAKETIME" ]; then
            export FAKETIME
            echo "[gaia2-init] Auto-detected FAKETIME from scenario: $FAKETIME" >&2
        fi
    fi

    # Remove CLI symlinks for apps not in this scenario.
    _REMOVE=$(python3 -c "
from gaia2_cli.app_registry import resolve_scenario_tools
import json
r = resolve_scenario_tools('$CUSTOM_SCENARIO')
print(' '.join(r.get('remove', [])))
" 2>/dev/null)
    if [ -n "$_REMOVE" ]; then
        for cmd in $_REMOVE; do
            rm -f "/home/agent/bin/$cmd"
        done
        echo "[gaia2-init] Removed unused CLI symlinks: $_REMOVE" >&2
    fi

    # Render agent system prompt from template + scenario
    /usr/local/bin/python3 /opt/render_agent_prompt.py \
        --scenario "$CUSTOM_SCENARIO" \
        --template /opt/AGENTS_TEMPLATE.md \
        --exec-tool terminal \
        --output-prompt /home/agent/AGENTS.md 2>/dev/null \
        && echo "[gaia2-init] Rendered AGENTS.md" >&2

    echo "[gaia2-init] Done" >&2
fi

# ── 2. Start gaia2-eventd daemon as gaia2 user ──────────────────────────────
# The daemon watches events.jsonl for turn boundaries, runs the judge,
# fires ENV reactions via CLI, and sends follow-up messages via the adapter.
if [ "${GAIA2_DAEMON_DISABLE:-0}" != "1" ] && [ -f "$CUSTOM_SCENARIO" ]; then
    touch /tmp/gaia2-eventd.log && chown gaia2:gaia2 /tmp/gaia2-eventd.log
    # faketime.rc: gaia2-owned, world-readable.
    # Daemon (gaia2) writes it between turns. CLI tools (gaia2 via gaia2-exec)
    # and agent bash wrapper read it via libfaketime. Agent cannot modify it.
    if [ -n "$FAKETIME" ]; then
        echo "$FAKETIME" > /tmp/faketime.rc
    else
        touch /tmp/faketime.rc
    fi
    chown gaia2:gaia2 /tmp/faketime.rc
    chmod 644 /tmp/faketime.rc

    echo "[gaia2-init] Starting gaia2-eventd as gaia2 user..." >&2
    su -s /usr/bin/bash gaia2 -c "
        export PATH=/usr/local/bin:/usr/bin:/bin
        export GAIA2_STATE_DIR=/var/gaia2/state
        ${GAIA2_JUDGE_FINAL_TURN:+export GAIA2_JUDGE_FINAL_TURN='$GAIA2_JUDGE_FINAL_TURN'}
        /usr/local/bin/gaia2-eventd \
            --scenario '$CUSTOM_SCENARIO' \
            --state-dir /var/gaia2/state \
            --notify-url 'http://127.0.0.1:$ADAPTER_PORT' \
            --poll-interval 0.5 \
            ${FAKETIME:+--faketime-path /tmp/faketime.rc} \
            ${GAIA2_NOTIFICATION_MODE:+--notification-mode $GAIA2_NOTIFICATION_MODE} \
            ${GAIA2_JUDGE_MODEL:+--judge-model $GAIA2_JUDGE_MODEL} \
            ${GAIA2_JUDGE_PROVIDER:+--judge-provider $GAIA2_JUDGE_PROVIDER} \
            ${GAIA2_JUDGE_BASE_URL:+--judge-base-url $GAIA2_JUDGE_BASE_URL} \
            ${GAIA2_JUDGE_API_KEY:+--judge-api-key '$GAIA2_JUDGE_API_KEY'} \
            ${GAIA2_TIME_SPEED:+--time-speed $GAIA2_TIME_SPEED} \
            >> /tmp/gaia2-eventd.log 2>&1 &
    "
    echo "[gaia2-init] Daemon launched (log: /tmp/gaia2-eventd.log)" >&2
else
    echo "[gaia2-init] gaia2-eventd skipped (disable=${GAIA2_DAEMON_DISABLE:-0})" >&2
fi

# ── 2b. Start GAIA2 adapter as gaia2 user ───────────────────────────────────
# The adapter bridges HTTP (runner) ↔ Unix socket (hermes worker). It writes
# AUI events to events.jsonl for turn boundary detection. Runs as gaia2 so it
# can access /var/gaia2/state (700) without exposing it to the agent.
if [ -f "$CUSTOM_SCENARIO" ]; then
    touch /tmp/gaia2-adapter.log && chown gaia2:gaia2 /tmp/gaia2-adapter.log
    echo "[gaia2-init] Starting GAIA2 adapter as gaia2 user..." >&2
    su -s /usr/bin/bash gaia2 -c "
        export PATH=/usr/local/bin:/usr/bin:/bin
        export GAIA2_STATE_DIR=/var/gaia2/state
        export GAIA2_ADAPTER_PORT='$ADAPTER_PORT'
        PYTHONUNBUFFERED=1 /usr/local/bin/python3 /opt/gaia2_adapter.py \
            >> /tmp/gaia2-adapter.log 2>&1 &
    "
    echo "[gaia2-init] Adapter launched (log: /tmp/gaia2-adapter.log)" >&2
fi

# ── 3. Drop to agent and exec the real entrypoint ────────────────────────
# Write env vars to a file to avoid quoting issues with special characters
# (e.g. API keys that may contain problematic chars).
ENV_FILE=/tmp/hermes-env.sh
cat > "$ENV_FILE" << 'STATIC'
export HOME=/home/agent
export PATH=/home/agent/bin
export GAIA2_STATE_DIR=/var/gaia2/state
export DONT_FAKE_MONOTONIC=1
STATIC

# Append runtime env vars (each safely printf-quoted to handle special chars)
for var in \
    PROVIDER MODEL API_KEY BASE_URL THINKING MAX_TOKENS \
    no_proxy NO_PROXY http_proxy https_proxy HTTP_PROXY HTTPS_PROXY \
    FAKETIME GAIA2_TRACE_FILE \
; do
    val="${!var:-}"
    [ -z "$val" ] && continue
    printf 'export %s=%q\n' "$var" "$val" >> "$ENV_FILE"
done

chown agent:agent "$ENV_FILE"
chmod 600 "$ENV_FILE"
exec su -s /usr/bin/bash agent -c "source /tmp/hermes-env.sh && exec /usr/bin/bash /opt/entrypoint.sh"
