#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
# entrypoint.sh — Starts the Hermes agent worker.
#
# The GAIA2 adapter and daemon run as the gaia2 user, started by
# gaia2-init-entrypoint.sh before dropping to agent.
#
# This script only manages the Hermes worker process, which connects
# to the adapter via a Unix socket and runs the AIAgent.

set -o pipefail

LOG=/tmp/entrypoint.log

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a $LOG; }

log "=== entrypoint start (hermes) ==="

# Set up libfaketime if FAKETIME is set (for scenario time simulation).
# Python's time.time() / datetime.now() will return simulated time.
# HTTP calls to LLM APIs use real time natively (no TLS proxy needed).
if [ -n "${FAKETIME:-}" ]; then
    /usr/bin/mkdir -p /dev/shm 2>/dev/null && /usr/bin/chmod 1777 /dev/shm 2>/dev/null || true
    log "Faketime enabled: $FAKETIME (CLI: frozen via /tmp/faketime.rc)"
fi

log "User: $(/usr/bin/id -un), PATH: $PATH, State: ${GAIA2_STATE_DIR:-unset}"

# ── Start Hermes worker (connects to adapter via Unix socket) ─────────
# Disable Hermes's built-in PII redaction — it masks phone numbers in CLI
# output, which breaks tool calls that pass those numbers back as arguments.
export HERMES_REDACT_SECRETS=0

log "Starting Hermes worker..."
PYTHONUNBUFFERED=1 /usr/local/bin/python3 /opt/hermes_worker.py >> $LOG 2>&1 &
WORKER_PID=$!
log "Worker PID: $WORKER_PID"

# Brief wait to check worker didn't crash immediately
/usr/bin/sleep 2
if ! kill -0 $WORKER_PID 2>/dev/null; then
    log "ERROR: Hermes worker exited unexpectedly"
    log "=== Last 40 lines of entrypoint.log ==="
    /usr/bin/tail -40 $LOG 2>/dev/null || true
    exit 1
fi
log "Hermes worker running"
log "GAIA2 adapter + daemon running as gaia2 user (see /tmp/gaia2-adapter.log, /tmp/gaia2-eventd.log)"

# ── Wait for process to exit ─────────────────────────────────────────
# shellcheck disable=SC2317  # cleanup is invoked indirectly via trap
cleanup() {
    log "Shutting down..."
    kill "$WORKER_PID" 2>/dev/null || true
    wait "$WORKER_PID" 2>/dev/null || true
}
trap cleanup EXIT TERM INT

wait $WORKER_PID 2>/dev/null
EXIT_CODE=$?
log "Worker (PID $WORKER_PID) exited with code $EXIT_CODE"
exit $EXIT_CODE
