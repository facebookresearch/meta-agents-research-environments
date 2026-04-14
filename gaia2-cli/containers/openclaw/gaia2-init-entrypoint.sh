#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
# gaia2-init-entrypoint.sh — Runtime scenario init + Gaia2 infrastructure startup.
#
# Runs as root, then drops to agent for the OpenClaw gateway.
#
# Architecture (privilege separation):
#   GAIA2 side (gaia2 user):  gaia2-eventd + state files + events.jsonl
#   Agent side (agent user): OpenClaw gateway + gaia2-adapter
#
# The agent interacts with the simulated environment exclusively through:
#   - CLI tools (via gaia2-exec setuid wrapper → gaia2 user)
#   - Gateway messages (adapter bridges to/from the Gaia2 daemon)
#
# Environment variables:
#   GAIA2_SCENARIO       — path to scenario JSON (default: /var/gaia2/custom_scenario.json)
#   GAIA2_FS_BACKING_DIR — filesystem backing directory (baked into image at /opt/gaia2_filesystem)
#   GAIA2_DAEMON_DISABLE — set to "1" to skip daemon startup
#   GAIA2_ADAPTER_PORT   — adapter HTTP port (default: 8090)
#   GAIA2_JUDGE_BASE_URL — optional API base URL override for the in-container judge
#   GAIA2_JUDGE_API_KEY  — optional API key override for the in-container judge
#   FAKETIME             — simulated start time (e.g., "2025-09-01 07:00:00")
#   API_KEY / provider-specific keys — forwarded to the agent runtime

set -eo pipefail

export PATH="/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"

SCENARIO_PATH="${GAIA2_SCENARIO:-/var/gaia2/custom_scenario.json}"
ADAPTER_PORT="${GAIA2_ADAPTER_PORT:-8090}"
STATE_DIR=/var/gaia2/state
FAKETIME_FILE=/tmp/faketime.rc
EVENTD_LOG=/tmp/gaia2-eventd.log
ADAPTER_LOG=/tmp/gaia2-adapter.log
ENTRYPOINT_LOG=/tmp/entrypoint.log
TLS_PROXY_PORT_FILE=/tmp/tls-proxy-port
AGENT_ENV_FILE=/tmp/openclaw-env.sh


detect_faketime_from_scenario() {
    [ -n "${FAKETIME:-}" ] && return

    FAKETIME=$(python3 -c "
import json, sys
from datetime import datetime, timezone
d = json.load(open('$SCENARIO_PATH'))
st = d.get('metadata', {}).get('definition', {}).get('start_time', d.get('start_time', 0))
if st and float(st) > 0:
    print(datetime.fromtimestamp(float(st), tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S'))
" 2>/dev/null)

    if [ -n "$FAKETIME" ]; then
        export FAKETIME
        echo "[gaia2-init] Auto-detected FAKETIME from scenario: $FAKETIME" >&2
    fi
}


prune_agent_tools() {
    local removed_tools

    removed_tools=$(python3 -c "
from gaia2_cli.app_registry import resolve_scenario_tools
r = resolve_scenario_tools('$SCENARIO_PATH')
print(' '.join(r.get('remove', [])))
" 2>/dev/null)

    if [ -n "$removed_tools" ]; then
        for cmd in $removed_tools; do
            rm -f "/home/agent/bin/$cmd"
        done
        echo "[gaia2-init] Removed unused CLI symlinks: $removed_tools" >&2
    fi
}


render_agent_prompt() {
    /usr/local/bin/python3 /opt/render_agent_prompt.py \
        --scenario "$SCENARIO_PATH" \
        --template /opt/AGENTS_TEMPLATE.md \
        --output-prompt /home/agent/AGENTS.md \
        --output-approvals /home/agent/.openclaw/exec-approvals.json \
        && chown -R agent:agent /home/agent/.openclaw /home/agent/AGENTS.md \
        && echo "[gaia2-init] Rendered AGENTS.md + exec-approvals.json" >&2
}


init_state() {
    if [ ! -f "$SCENARIO_PATH" ]; then
        echo "[gaia2-init] No scenario found at $SCENARIO_PATH, skipping init" >&2
        return
    fi

    echo "[gaia2-init] Initialising from $SCENARIO_PATH ..." >&2
    rm -rf "$STATE_DIR"/*
    gaia2-init --scenario "$SCENARIO_PATH" \
        --state-dir "$STATE_DIR" \
        ${GAIA2_FS_BACKING_DIR:+--fs-backing-dir "$GAIA2_FS_BACKING_DIR"}
    chown -R gaia2:gaia2 "$STATE_DIR"

    # Lock down /var/gaia2 — agent must not read scenario (ground truth) or
    # state files (must use CLI tools via gaia2-exec instead).
    chmod 700 /var/gaia2
    chmod -R go-rwx "$STATE_DIR"

    detect_faketime_from_scenario
    prune_agent_tools
    render_agent_prompt
    echo "[gaia2-init] Done" >&2
}


prepare_faketime_file() {
    if [ -n "${FAKETIME:-}" ]; then
        echo "$FAKETIME" > "$FAKETIME_FILE"
    else
        touch "$FAKETIME_FILE"
    fi
    chown gaia2:gaia2 "$FAKETIME_FILE"
    chmod 644 "$FAKETIME_FILE"
}


start_eventd() {
    if [ "${GAIA2_DAEMON_DISABLE:-0}" = "1" ] || [ ! -f "$SCENARIO_PATH" ]; then
        echo "[gaia2-init] gaia2-eventd skipped (disable=${GAIA2_DAEMON_DISABLE:-0})" >&2
        return
    fi

    touch "$EVENTD_LOG" && chown gaia2:gaia2 "$EVENTD_LOG"
    # faketime.rc: gaia2-owned, world-readable.
    # Daemon (gaia2) writes it between turns. CLI tools (gaia2 via gaia2-exec)
    # and agent bash wrapper read it via libfaketime. Agent cannot modify it.
    prepare_faketime_file

    echo "[gaia2-init] Starting gaia2-eventd as gaia2 user..." >&2
    # shellcheck disable=SC2016
    su -s /usr/bin/bash gaia2 -c "
        export PATH=/usr/local/bin:/usr/bin:/bin
        export GAIA2_STATE_DIR='$STATE_DIR'
        ${GAIA2_JUDGE_FINAL_TURN:+export GAIA2_JUDGE_FINAL_TURN='$GAIA2_JUDGE_FINAL_TURN'}
        ${http_proxy:+export http_proxy='$http_proxy'}
        ${https_proxy:+export https_proxy='$https_proxy'}
        ${no_proxy:+export no_proxy='$no_proxy'}
        /usr/local/bin/gaia2-eventd \
            --scenario '$SCENARIO_PATH' \
            --state-dir '$STATE_DIR' \
            --notify-url 'http://127.0.0.1:$ADAPTER_PORT' \
            --poll-interval 0.5 \
            ${FAKETIME:+--faketime-path $FAKETIME_FILE} \
            ${GAIA2_NOTIFICATION_MODE:+--notification-mode $GAIA2_NOTIFICATION_MODE} \
            ${GAIA2_JUDGE_MODEL:+--judge-model $GAIA2_JUDGE_MODEL} \
            ${GAIA2_JUDGE_PROVIDER:+--judge-provider $GAIA2_JUDGE_PROVIDER} \
            ${GAIA2_JUDGE_BASE_URL:+--judge-base-url $GAIA2_JUDGE_BASE_URL} \
            ${GAIA2_JUDGE_API_KEY:+--judge-api-key '$GAIA2_JUDGE_API_KEY'} \
            ${GAIA2_TIME_SPEED:+--time-speed $GAIA2_TIME_SPEED} \
            >> '$EVENTD_LOG' 2>&1 &
    "
    echo "[gaia2-init] Daemon launched (log: $EVENTD_LOG)" >&2
}


start_adapter() {
    if [ ! -f "$SCENARIO_PATH" ]; then
        return
    fi

    touch "$ADAPTER_LOG" && chown gaia2:gaia2 "$ADAPTER_LOG"
    echo "[gaia2-init] Starting GAIA2 adapter as gaia2 user..." >&2
    # shellcheck disable=SC2016
    su -s /usr/bin/bash gaia2 -c "
        export PATH=/usr/local/bin:/usr/bin:/bin
        export GAIA2_STATE_DIR='$STATE_DIR'
        export GAIA2_ADAPTER_PORT='$ADAPTER_PORT'
        export OPENCLAW_GATEWAY_TOKEN='${OPENCLAW_GATEWAY_TOKEN:-gaia2-notif}'
        export OPENCLAW_HOOKS_TOKEN='${OPENCLAW_HOOKS_TOKEN:-gaia2-hooks}'
        ${GAIA2_NOTIFICATION_MODE:+export GAIA2_NOTIFICATION_MODE='$GAIA2_NOTIFICATION_MODE'}
        ${OPENCLAW_GATEWAY_PORT:+export OPENCLAW_GATEWAY_PORT='$OPENCLAW_GATEWAY_PORT'}
        PYTHONUNBUFFERED=1 /usr/local/bin/python3 /opt/gaia2_adapter.py \
            >> '$ADAPTER_LOG' 2>&1 &
    "
    echo "[gaia2-init] Adapter launched (log: $ADAPTER_LOG)" >&2
}


start_tls_proxy_if_needed() {
    local retries=0
    local upstream_https_proxy_value
    local -a tls_proxy_cmd

    if [ ! -f "$FAKETIME_FILE" ] || [ -z "${FAKETIME:-}" ]; then
        return
    fi

    # Create entrypoint.log writable by agent (TLS proxy logs here as root,
    # then entrypoint.sh continues logging as agent).
    touch "$ENTRYPOINT_LOG" && chmod 666 "$ENTRYPOINT_LOG"

    upstream_https_proxy_value="${HTTPS_PROXY:-${https_proxy:-${UPSTREAM_HTTPS_PROXY:-}}}"
    echo "[gaia2-init] Starting MITM TLS proxy..." >&2
    tls_proxy_cmd=(
        /usr/bin/env
        -u LD_PRELOAD
        -u FAKETIME
        -u FAKETIME_TIMESTAMP_FILE
        -u FAKETIME_NO_CACHE
        -u https_proxy
        -u HTTPS_PROXY
    )
    if [ -n "$upstream_https_proxy_value" ]; then
        tls_proxy_cmd+=(UPSTREAM_HTTPS_PROXY="$upstream_https_proxy_value")
    fi
    tls_proxy_cmd+=(
        TLS_PROXY_PORT_FILE="$TLS_PROXY_PORT_FILE"
        /usr/local/bin/python3
        /opt/tls_proxy.py
        --port
        0
        --ca-cert
        /opt/tls-proxy/ca.crt
        --ca-key
        /opt/tls-proxy/ca.key
        --host-key
        /opt/tls-proxy/host.key
    )
    "${tls_proxy_cmd[@]}" >> "$ENTRYPOINT_LOG" 2>&1 &

    while [ ! -s "$TLS_PROXY_PORT_FILE" ]; do
        retries=$((retries + 1))
        if [ $retries -ge 20 ]; then
            echo "[gaia2-init] ERROR: TLS proxy did not start after 10s" >&2
            exit 1
        fi
        sleep 0.5
    done

    TLS_PROXY_PORT=$(cat "$TLS_PROXY_PORT_FILE")
    export TLS_PROXY_PORT
    echo "[gaia2-init] TLS proxy ready (port: $TLS_PROXY_PORT)" >&2
}


write_agent_env_file() {
    cat > "$AGENT_ENV_FILE" << 'STATIC'
export HOME=/home/agent
export PATH=/home/agent/bin
export GAIA2_STATE_DIR=/var/gaia2/state
export DONT_FAKE_MONOTONIC=1
STATIC

    # Append runtime env vars (each safely printf-quoted to handle special chars)
    for var in \
        PROVIDER MODEL API_KEY ANTHROPIC_API_KEY ANTHROPIC_MODEL OPENCLAW_FORCE_RECONFIG \
        GEMINI_API_KEY GOOGLE_API_KEY GEMINI_MODEL GOOGLE_MODEL GOOGLE_BASE_URL GEMINI_BASE_URL \
        BASE_URL OPENAI_BASE_URL OPENAI_API_KEY OPENAI_MODEL         \
        OPENROUTER_API_KEY OPENROUTER_MODEL OPENROUTER_BASE_URL      \
        THINKING REASONING_EFFORT                                     \
        no_proxy NO_PROXY http_proxy https_proxy HTTP_PROXY HTTPS_PROXY \
        NODE_EXTRA_CA_CERTS FAKETIME                                 \
        OPENCLAW_GATEWAY_PORT OPENCLAW_GATEWAY_URL OPENCLAW_GATEWAY_TOKEN OPENCLAW_HOOKS_TOKEN \
        GAIA2_TRACE_FILE \
        DONT_FAKE_MONOTONIC TLS_PROXY_PORT \
    ; do
        val="${!var:-}"
        [ -z "$val" ] && continue
        printf 'export %s=%q\n' "$var" "$val" >> "$AGENT_ENV_FILE"
    done

    chown agent:agent "$AGENT_ENV_FILE"
    chmod 600 "$AGENT_ENV_FILE"
}


exec_agent_entrypoint() {
    # Always force reconfig so openclaw-setup.sh regenerates
    # auth-profiles.json with the runtime API key; the build-time setup run
    # cannot bake real credentials into the image.
    write_agent_env_file
    exec su -s /usr/bin/bash agent -c "source '$AGENT_ENV_FILE' && exec /usr/bin/bash /opt/entrypoint.sh"
}


main() {
    init_state
    start_eventd
    start_adapter
    start_tls_proxy_if_needed
    exec_agent_entrypoint
}


main "$@"
