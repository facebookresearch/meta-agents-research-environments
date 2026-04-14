#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
# Oracle adapter entrypoint — runs as gaia2 user.

set -o pipefail

LOG=/tmp/entrypoint.log

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a $LOG; }

log "=== entrypoint start (oracle) ==="
log "User: $(/usr/bin/id -un), PATH: $PATH, State: ${GAIA2_STATE_DIR:-unset}"

# ── Start oracle adapter ────────────────────────────────────────────────
log "Starting oracle adapter..."
PYTHONUNBUFFERED=1 /usr/local/bin/python3 /opt/gaia2_oracle_adapter.py >> $LOG 2>&1 &
ADAPTER_PID=$!
log "Adapter PID: $ADAPTER_PID"

# Brief wait to check adapter didn't crash immediately
/usr/bin/sleep 2
if ! kill -0 $ADAPTER_PID 2>/dev/null; then
    log "ERROR: Oracle adapter exited unexpectedly"
    log "=== Last 40 lines of entrypoint.log ==="
    /usr/bin/tail -40 $LOG 2>/dev/null || true
    exit 1
fi
log "Oracle adapter ready (port ${GAIA2_ADAPTER_PORT:-8090})"

# ── Wait for process to exit ────────────────────────────────────────────
# shellcheck disable=SC2317  # cleanup is invoked indirectly via trap
cleanup() {
    log "Shutting down..."
    kill "$ADAPTER_PID" 2>/dev/null || true
    wait "$ADAPTER_PID" 2>/dev/null || true
}
trap cleanup EXIT TERM INT

wait $ADAPTER_PID 2>/dev/null
EXIT_CODE=$?
log "Adapter (PID $ADAPTER_PID) exited with code $EXIT_CODE"
exit $EXIT_CODE
