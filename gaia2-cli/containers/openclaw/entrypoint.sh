#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
# entrypoint.sh — Starts the OpenClaw gateway (agent side only).
#
# The GAIA2 adapter and daemon run as the gaia2 user, started by
# gaia2-init-entrypoint.sh before dropping to agent.
#
# This script only manages the gateway process.

set -o pipefail

LOG=/tmp/entrypoint.log

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a $LOG; }

log "=== entrypoint start ==="

# Source openclaw config (generates config if needed)
# shellcheck disable=SC1091
. /etc/profile.d/openclaw-setup.sh 2>/dev/null || true
# Only force reconfiguration during initial startup. If this leaks into
# descendant shells spawned by the exec tool, /etc/profile.d/openclaw-setup.sh
# rewrites openclaw.json mid-run and triggers a gateway reload/restart.
unset OPENCLAW_FORCE_RECONFIG

# Log faketime status. The gateway runs under libfaketime (via oc wrapper)
# so new Date() returns simulated time natively. CLI tools get frozen faked
# time via gaia2-exec and bash wrapper.
if [ -n "${FAKETIME:-}" ]; then
    # Ensure /dev/shm exists for FAKETIME_NO_CACHE semaphores
    /usr/bin/mkdir -p /dev/shm 2>/dev/null && /usr/bin/chmod 1777 /dev/shm 2>/dev/null || true
    log "Faketime enabled: $FAKETIME (gateway: LD_PRELOAD, CLI: frozen via /tmp/faketime.rc)"
fi

log "User: $(/usr/bin/id -un), PATH: $PATH, State: ${GAIA2_STATE_DIR:-unset}"

# ── Pick up MITM TLS proxy started by gaia2-init-entrypoint.sh (root) ─────
# The proxy runs as root to read ca.key (600). It writes its port to a file.
if [ -f /tmp/tls-proxy-port ]; then
    TLS_PROXY_PORT=$(cat /tmp/tls-proxy-port)
    log "TLS proxy running (root, port: $TLS_PROXY_PORT)"
    export https_proxy="http://127.0.0.1:${TLS_PROXY_PORT}"
    export HTTPS_PROXY="$https_proxy"
fi

# ── Start OpenClaw gateway ────────────────────────────────────────────────
log "Starting OC gateway..."
/home/agent/bin/oc gateway run ${OPENCLAW_GATEWAY_PORT:+--port $OPENCLAW_GATEWAY_PORT} >> $LOG 2>&1 &
OC_PID=$!
log "Gateway PID: $OC_PID"

# ── Wait for gateway to be ready ─────────────────────────────────────────
GATEWAY_PORT="${OPENCLAW_GATEWAY_PORT:-18789}"
RETRIES=0
MAX_RETRIES=60
while ! /usr/bin/curl -sf -m 2 --noproxy 127.0.0.1 "http://127.0.0.1:${GATEWAY_PORT}/health" >/dev/null 2>&1; do
    CURL_RC=$?
    RETRIES=$((RETRIES + 1))
    if [ $((RETRIES % 10)) -eq 1 ] || [ $RETRIES -ge $((MAX_RETRIES - 1)) ]; then
        log "Health check attempt $RETRIES (curl exit=$CURL_RC):"
        /usr/bin/curl -v -m 2 --noproxy 127.0.0.1 "http://127.0.0.1:${GATEWAY_PORT}/health" >> $LOG 2>&1 || true
    fi
    if [ $RETRIES -ge $MAX_RETRIES ]; then
        log "ERROR: Gateway did not become ready after ${MAX_RETRIES}s"
        kill $OC_PID 2>/dev/null || true
        exit 1
    fi
    if ! kill -0 $OC_PID 2>/dev/null; then
        log "ERROR: Gateway process exited unexpectedly"
        wait $OC_PID 2>/dev/null
        log "Gateway exit code: $?"
        exit 1
    fi
    /usr/bin/sleep 1
done
log "Gateway ready (port ${GATEWAY_PORT})"
log "GAIA2 adapter + daemon running as gaia2 user (see /tmp/gaia2-adapter.log, /tmp/gaia2-eventd.log)"
log "All services running. Gateway=$OC_PID"

# ── Wait for gateway to exit ─────────────────────────────────────────
# shellcheck disable=SC2317  # cleanup is invoked indirectly via trap
cleanup() {
    log "Shutting down..."
    kill "$OC_PID" 2>/dev/null || true
    wait "$OC_PID" 2>/dev/null || true
}
trap cleanup EXIT TERM INT

wait $OC_PID 2>/dev/null
EXIT_CODE=$?
log "Gateway (PID $OC_PID) exited with code $EXIT_CODE"
exit $EXIT_CODE
