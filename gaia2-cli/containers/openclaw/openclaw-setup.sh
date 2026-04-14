#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
# Auto-configure OpenClaw on shell login.
#
# The shell side stays intentionally small. Provider/env resolution and JSON
# generation live in openclaw_config.py so the behavior matrix is unit-tested.

# Skip if config exists and no force-reconfig (e.g. bind-mounted from host).
if [ -f "$HOME/.openclaw/openclaw.json" ] && [ -z "${OPENCLAW_FORCE_RECONFIG:-}" ]; then
    return 0 2>/dev/null || exit 0
fi

OPENCLAW_CONFIG_HELPER="/opt/openclaw_config.py"
if [ ! -f "$OPENCLAW_CONFIG_HELPER" ]; then
    OPENCLAW_CONFIG_HELPER="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/openclaw_config.py"
fi

if [ ! -f "$OPENCLAW_CONFIG_HELPER" ]; then
    echo "[openclaw-setup] Error: missing config helper: $OPENCLAW_CONFIG_HELPER" >&2
    return 1 2>/dev/null || exit 1
fi

OPENCLAW_SETUP_EXPORTS="$(python3 "$OPENCLAW_CONFIG_HELPER" --home "$HOME")" || {
    return 1 2>/dev/null || exit 1
}

eval "$OPENCLAW_SETUP_EXPORTS"

unset OPENCLAW_SETUP_EXPORTS
unset OPENCLAW_CONFIG_HELPER
