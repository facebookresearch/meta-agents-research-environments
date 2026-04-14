#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
# gaia2-init-entrypoint.sh — Oracle container init.
#
# Runs as root: initializes state, starts gaia2-eventd daemon as a separate
# process, then drops to gaia2 user for the oracle adapter.
#
# Architecture matches OC/Jarvis: daemon runs independently, oracle replay
# writes events to events.jsonl, daemon detects turn boundaries and judges.
#
# Environment variables:
#   GAIA2_SCENARIO       — path to scenario JSON (default: /var/gaia2/custom_scenario.json)
#   GAIA2_FS_BACKING_DIR — filesystem backing directory (baked into image)
#   GAIA2_ADAPTER_PORT   — adapter HTTP port (default: 8090)
#   FAKETIME            — simulated start time (e.g., "2025-09-01 07:00:00")
#   GAIA2_JUDGE_MODEL    — LLM model for in-container judge
#   GAIA2_JUDGE_PROVIDER — LLM provider for judge
#   GAIA2_JUDGE_BASE_URL — optional custom API base URL for judge
#   GAIA2_JUDGE_API_KEY  — optional API key override for the in-container judge
#   GAIA2_JUDGE_FINAL_TURN — enable final-turn judging ("1")

set -eo pipefail

export PATH="/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"

CUSTOM_SCENARIO="${GAIA2_SCENARIO:-/var/gaia2/custom_scenario.json}"
ADAPTER_PORT="${GAIA2_ADAPTER_PORT:-8090}"

# ── 1. Initialize state directory ─────────────────────────────────────────
if [ ! -f "$CUSTOM_SCENARIO" ]; then
    echo "[gaia2-init] No scenario found at $CUSTOM_SCENARIO" >&2
else
    echo "[gaia2-init] Initializing state for $CUSTOM_SCENARIO ..." >&2
    rm -rf /var/gaia2/state/*
    chown -R gaia2:gaia2 /var/gaia2/state
    chmod 711 /var/gaia2
    chmod 711 /var/gaia2/state

    # Run gaia2-init to populate state directory from scenario JSON.
    # The daemon expects initialized state (contacts, emails, etc.).
    su -s /usr/bin/bash gaia2 -c "
        export PATH=/usr/local/bin:/usr/bin:/bin
        export GAIA2_STATE_DIR=/var/gaia2/state
        /usr/local/bin/gaia2-init --scenario '$CUSTOM_SCENARIO' --state-dir /var/gaia2/state
    "

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

    echo "[gaia2-init] Done" >&2
fi

# ── 2. Prepare faketime + start gaia2-eventd daemon ────────────────────────
if [ -f "$CUSTOM_SCENARIO" ]; then
    if [ -n "$FAKETIME" ]; then
        echo "$FAKETIME" > /tmp/faketime.rc
    else
        touch /tmp/faketime.rc
    fi
    chown gaia2:gaia2 /tmp/faketime.rc
    chmod 644 /tmp/faketime.rc

    # Ensure /dev/shm exists for FAKETIME_NO_CACHE semaphores
    mkdir -p /dev/shm 2>/dev/null && chmod 1777 /dev/shm 2>/dev/null || true

    # Start gaia2-eventd as a separate background process (same pattern as OC).
    # The daemon watches events.jsonl, detects turn boundaries, fires ENV
    # reactions, and runs the in-container judge.
    touch /tmp/gaia2-eventd.log && chown gaia2:gaia2 /tmp/gaia2-eventd.log
    echo "[gaia2-init] Starting gaia2-eventd as gaia2 user..." >&2
    su -s /usr/bin/bash gaia2 -c "
        export PATH=/usr/local/bin:/usr/bin:/bin
        export GAIA2_STATE_DIR=/var/gaia2/state
        ${GAIA2_JUDGE_FINAL_TURN:+export GAIA2_JUDGE_FINAL_TURN='$GAIA2_JUDGE_FINAL_TURN'}
        ${http_proxy:+export http_proxy='$http_proxy'}
        ${https_proxy:+export https_proxy='$https_proxy'}
        ${no_proxy:+export no_proxy='$no_proxy'}
        /usr/local/bin/gaia2-eventd \
            --scenario '$CUSTOM_SCENARIO' \
            --state-dir /var/gaia2/state \
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
fi

# ── 3. Drop to gaia2 user and exec the oracle adapter ─────────────────────
exec su -s /usr/bin/bash gaia2 -c "
    export PATH=/usr/local/bin:/usr/bin:/bin
    export GAIA2_STATE_DIR=/var/gaia2/state
    export GAIA2_SCENARIO='$CUSTOM_SCENARIO'
    export GAIA2_ADAPTER_PORT='$ADAPTER_PORT'
    ${FAKETIME:+export FAKETIME='$FAKETIME'}
    ${http_proxy:+export http_proxy='$http_proxy'}
    ${https_proxy:+export https_proxy='$https_proxy'}
    ${no_proxy:+export no_proxy='$no_proxy'}
    exec /usr/bin/bash /opt/entrypoint.sh
"
