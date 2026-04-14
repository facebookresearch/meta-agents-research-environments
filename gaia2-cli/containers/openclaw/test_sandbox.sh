#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
# Runtime smoke test for the gaia2-oc sandbox.
#
# Builds the image with the repo Makefile (unless --skip-build), starts the
# container with a tiny scenario, waits for the adapter health endpoint, and
# validates the runtime sandbox contract:
#   - scenario init actually ran
#   - agent PATH is restricted
#   - only scenario-relevant CLI tools remain in PATH
#   - state and scenario files are not readable by the agent
#   - representative Gaia2 tools still work through gaia2-exec
#   - OpenAI custom BASE_URL survives runtime env forwarding
#
# Usage:
#   bash containers/openclaw/test_sandbox.sh
#   bash containers/openclaw/test_sandbox.sh --skip-build

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
IMAGE="localhost/gaia2-oc:latest"
CONTAINER="gaia2-oc-sandbox-test"
BUILD_LOG="/tmp/gaia2-oc-sandbox-build.log"
TEST_DIR="$(mktemp -d /tmp/gaia2-oc-sandbox.XXXXXX)"
SCENARIO="$TEST_DIR/scenario.json"
PASS=0
FAIL=0

green()  { printf '\033[32m%s\033[0m\n' "$*"; }
red()    { printf '\033[31m%s\033[0m\n' "$*"; }
yellow() { printf '\033[33m%s\033[0m\n' "$*"; }
bold()   { printf '\033[1m%s\033[0m\n' "$*"; }

pick_free_port() {
    python3 - <<'PY'
import socket

with socket.socket() as sock:
    sock.bind(("127.0.0.1", 0))
    print(sock.getsockname()[1])
PY
}

DEFAULT_ADAPTER_PORT="$(pick_free_port)"
DEFAULT_GATEWAY_PORT="$(pick_free_port)"
while [[ "$DEFAULT_GATEWAY_PORT" == "$DEFAULT_ADAPTER_PORT" ]]; do
    DEFAULT_GATEWAY_PORT="$(pick_free_port)"
done
ADAPTER_PORT="${GAIA2_ADAPTER_PORT:-$DEFAULT_ADAPTER_PORT}"
GATEWAY_PORT="${OPENCLAW_GATEWAY_PORT:-$DEFAULT_GATEWAY_PORT}"

cleanup() {
    podman stop "$CONTAINER" >/dev/null 2>&1 || true
    podman rm "$CONTAINER" >/dev/null 2>&1 || true
    rm -rf "$TEST_DIR"
}

cleanup_container() {
    podman stop "$CONTAINER" >/dev/null 2>&1 || true
    podman rm "$CONTAINER" >/dev/null 2>&1 || true
}

trap cleanup EXIT

host_exec() {
    /usr/bin/bash -lc "$1" 2>&1
}

agent_exec() {
    podman exec --user agent "$CONTAINER" /usr/bin/bash -c "$1" 2>&1
}

root_exec() {
    podman exec --user root -e PATH="/usr/local/bin:/usr/bin:/bin" "$CONTAINER" /usr/bin/bash -c "$1" 2>&1
}

run_cmd() {
    local scope="$1"
    local cmd="$2"
    case "$scope" in
        host) host_exec "$cmd" ;;
        agent) agent_exec "$cmd" ;;
        root) root_exec "$cmd" ;;
        *)
            echo "unknown scope: $scope" >&2
            return 2
            ;;
    esac
}

print_failure() {
    local cmd="$1"
    local output="$2"
    red "        command: $cmd"
    printf '%s\n' "$output" | head -5 | sed 's/^/        /'
}

assert_ok() {
    local scope="$1"
    local desc="$2"
    local cmd="$3"
    local output
    if output=$(run_cmd "$scope" "$cmd" 2>&1); then
        green "  PASS: $desc"
        PASS=$((PASS + 1))
    else
        red "  FAIL: $desc"
        print_failure "$cmd" "$output"
        FAIL=$((FAIL + 1))
    fi
}

assert_fail() {
    local scope="$1"
    local desc="$2"
    local cmd="$3"
    local output
    if output=$(run_cmd "$scope" "$cmd" 2>&1); then
        red "  FAIL: $desc (expected failure, got success)"
        print_failure "$cmd" "$output"
        FAIL=$((FAIL + 1))
    else
        green "  PASS: $desc"
        PASS=$((PASS + 1))
    fi
}

assert_contains() {
    local scope="$1"
    local desc="$2"
    local cmd="$3"
    local expected="$4"
    local output
    output=$(run_cmd "$scope" "$cmd" 2>&1) || true
    if printf '%s' "$output" | grep -qF "$expected"; then
        green "  PASS: $desc"
        PASS=$((PASS + 1))
    else
        red "  FAIL: $desc"
        red "        expected output to contain: $expected"
        print_failure "$cmd" "$output"
        FAIL=$((FAIL + 1))
    fi
}

cat > "$SCENARIO" <<'EOF'
{
  "metadata": {
    "definition": {
      "scenario_id": "openclaw_sandbox_smoke"
    }
  },
  "apps": [
    {
      "name": "Calendar",
      "class_name": "CalendarApp",
      "app_state": {
        "events": {}
      }
    },
    {
      "name": "Contacts",
      "class_name": "ContactsApp",
      "app_state": {
        "contacts": {}
      }
    }
  ],
  "events": []
}
EOF

bold "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
bold " Gaia2-OC Runtime Sandbox Smoke Test"
bold "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

if [[ "${1:-}" != "--skip-build" ]]; then
    bold "[build] Building image with make gaia2-oc..."
    if ! (cd "$REPO_ROOT" && make gaia2-oc) >"$BUILD_LOG" 2>&1; then
        red "FATAL: make gaia2-oc failed"
        tail -50 "$BUILD_LOG" || true
        exit 1
    fi
    green "[build] Image built successfully"
else
    yellow "[build] Skipped (using $IMAGE)"
fi
echo ""

bold "[setup] Starting container..."
cleanup_container
podman run --network=host -d --name "$CONTAINER" \
    -v "$SCENARIO:/var/gaia2/custom_scenario.json:ro,z" \
    -e OPENCLAW_FORCE_RECONFIG=1 \
    -e PROVIDER=openai \
    -e MODEL=gpt-4o \
    -e OPENAI_API_KEY=dummy-key \
    -e BASE_URL=https://proxy.example/v1 \
    -e GAIA2_DAEMON_DISABLE=1 \
    -e GAIA2_ADAPTER_PORT="$ADAPTER_PORT" \
    -e OPENCLAW_GATEWAY_PORT="$GATEWAY_PORT" \
    "$IMAGE" >/dev/null

READY=0
for _ in $(seq 1 60); do
    if curl -sf --noproxy 127.0.0.1 "http://127.0.0.1:${ADAPTER_PORT}/health" | python3 -c 'import json,sys; d=json.load(sys.stdin); assert d["ok"] is True and d["connected"] is True' >/dev/null 2>&1; then
        READY=1
        break
    fi
    sleep 1
done

if [[ "$READY" -ne 1 ]]; then
    red "FATAL: adapter health endpoint did not become ready on port $ADAPTER_PORT"
    podman logs "$CONTAINER" | tail -100 || true
    root_exec 'cat /tmp/gaia2-adapter.log 2>/dev/null || true' || true
    root_exec 'tail -100 /tmp/entrypoint.log 2>/dev/null || true' || true
    exit 1
fi
green "[setup] Container running (adapter port $ADAPTER_PORT, gateway port $GATEWAY_PORT)"
echo ""

bold "[1/7] Runtime Init"
assert_ok host "health endpoint reports ok + connected" \
    "curl -sf --noproxy 127.0.0.1 http://127.0.0.1:${ADAPTER_PORT}/health | python3 -c 'import json,sys; d=json.load(sys.stdin); assert d[\"ok\"] is True and d[\"connected\"] is True'"
assert_ok root "state init created events.jsonl" \
    "test -f /var/gaia2/state/events.jsonl"
assert_contains agent "BASE_URL reached openclaw.json at runtime" \
    'jq -r ".models.providers.openai.baseUrl" "$HOME/.openclaw/openclaw.json"' \
    "https://proxy.example/v1"
assert_contains agent "model selection stayed openai/gpt-4o" \
    'jq -r ".agents.defaults.model.primary" "$HOME/.openclaw/openclaw.json"' \
    "openai/gpt-4o"
echo ""

bold "[2/7] Agent Identity"
assert_contains agent "agent user is non-root (uid 1000)" \
    "/usr/bin/id -u" "1000"
assert_contains agent "agent username is agent" \
    "/usr/bin/id -un" "agent"
assert_contains agent "home directory is /home/agent" \
    'echo "$HOME"' "/home/agent"
assert_contains agent "PATH is restricted to /home/agent/bin" \
    'echo "$PATH"' "/home/agent/bin"
echo ""

bold "[3/7] Runtime Tool Pruning"
assert_ok agent "calendar tool is present for the mounted scenario" \
    "command -v calendar >/dev/null"
assert_ok agent "contacts tool is present for the mounted scenario" \
    "command -v contacts >/dev/null"
assert_fail agent "emails tool was removed for an unused app" \
    "command -v emails >/dev/null"
assert_fail agent "cloud-drive tool was removed for an unused app" \
    "command -v cloud-drive >/dev/null"
echo ""

bold "[4/7] State File Isolation"
assert_fail agent "/var/gaia2 is not listable" \
    "ls /var/gaia2"
assert_fail agent "mounted scenario is not readable" \
    "</var/gaia2/custom_scenario.json"
assert_fail agent "contacts state is not readable" \
    "</var/gaia2/state/contacts.json"
assert_fail agent "events.jsonl is not readable" \
    "</var/gaia2/state/events.jsonl"
assert_contains root "root can still verify contacts state exists" \
    'jq -r "keys[0]" /var/gaia2/state/contacts.json' \
    "contacts"
echo ""

bold "[5/7] Gaia2 CLI Tools"
assert_contains agent "contacts get-contacts works via gaia2-exec" \
    "contacts get-contacts" \
    '"contacts"'
assert_contains agent "calendar get-events works via gaia2-exec" \
    'calendar get-events --start-date "2025-01-01 00:00:00" --end-date "2025-01-02 00:00:00"' \
    '"events"'
assert_ok agent "oc wrapper is available" \
    "oc --version >/dev/null 2>&1"
echo ""

bold "[6/7] PATH Restrictions"
assert_fail agent "curl is not exposed on the agent PATH" \
    "command -v curl >/dev/null"
assert_fail agent "wget is not exposed on the agent PATH" \
    "command -v wget >/dev/null"
assert_fail agent "node is not exposed on the agent PATH" \
    "command -v node >/dev/null"
assert_ok agent "python3 remains available" \
    "command -v python3 >/dev/null"
echo ""

bold "[7/7] Write Isolation"
assert_ok agent "agent can write to /home/agent" \
    'echo test > /home/agent/.sandbox_test_file'
assert_fail agent "agent cannot write to /etc" \
    'echo test > /etc/testfile'
assert_fail agent "agent cannot write to /usr" \
    'echo test > /usr/testfile'
assert_fail agent "agent cannot write to /var/gaia2" \
    'echo test > /var/gaia2/testfile'
echo ""

bold "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
TOTAL=$((PASS + FAIL))
if [[ $FAIL -eq 0 ]]; then
    green " ALL $TOTAL TESTS PASSED"
else
    red " $FAIL/$TOTAL TESTS FAILED"
fi
bold "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

exit "$FAIL"
