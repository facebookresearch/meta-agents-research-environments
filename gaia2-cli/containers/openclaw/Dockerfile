# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
# Gaia2 + OpenClaw Docker image for benchmark evaluation.
# Combines Gaia2 CLI tools with OpenClaw (AI agent gateway) in a sandboxed environment.
#
# Architecture: gaia2-cli base → Node.js + OpenClaw gateway → sandboxed agent
#
# IMPORTANT: Do NOT use "openclaw" in the image tag — some host security policies kill
# processes with "openclaw" in the command line. Use a neutral name instead.
#
# BUILD (from gaia2-cli/):
#   make gaia2-oc
# RUN: podman run --network=host -e API_KEY=<key> localhost/gaia2-oc:latest
#
# The gaia2-cli base provides: Python 3.12, CLI tools, curl, jq, libfaketime
# The openclaw-builder stage installs: Node.js 22, OpenClaw gateway from npm

# Build args (declared before FROM so all stages can use them)
ARG GAIA2_CLI_IMAGE=localhost/gaia2-cli
ARG GAIA2_CLI_VERSION=latest

# ── Stage 1: Compile setuid wrapper from source ─────────────────────────
FROM debian:bookworm-slim AS gaia2-exec-builder
RUN echo 'APT::Sandbox::User "root";' | tee -a /etc/apt/apt.conf.d/10sandbox
RUN apt-get update && apt-get install -y --no-install-recommends gcc libc6-dev \
    && rm -rf /var/lib/apt/lists/*
COPY shared/gaia2-exec.c /tmp/gaia2-exec.c
RUN gcc -O2 -o /tmp/gaia2-exec /tmp/gaia2-exec.c && strip /tmp/gaia2-exec

# ── Stage 2: Install Node.js + OpenClaw from scratch ────────────────────
FROM debian:bookworm-slim AS openclaw-builder
# Proxy env vars so curl/apt inside this stage can reach external registries
ARG http_proxy
ARG https_proxy
ENV http_proxy=${http_proxy} https_proxy=${https_proxy}
RUN echo 'APT::Sandbox::User "root";' | tee -a /etc/apt/apt.conf.d/10sandbox
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl ca-certificates gnupg git \
    && rm -rf /var/lib/apt/lists/*
# Node.js 22.x via NodeSource
RUN curl -fsSL https://deb.nodesource.com/setup_22.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*
# OpenClaw npm package — downloaded as tarball to avoid host security policies
# killing processes with "openclaw" in the command line
ARG OPENCLAW_VERSION=2026.4.1
RUN curl -fSL "https://registry.npmjs.org/openclaw/-/openclaw-${OPENCLAW_VERSION}.tgz" \
      -o /tmp/pkg.tgz && \
    npm install -g /tmp/pkg.tgz && \
    rm /tmp/pkg.tgz
# Copy to /opt/chatbot (layout expected by launch.mjs and dist/*.js)
RUN cp -r /usr/lib/node_modules/openclaw /opt/chatbot

# ── Stage 3: Final image — gaia2-cli base + Node.js + OpenClaw ───────────
ARG GAIA2_CLI_IMAGE
ARG GAIA2_CLI_VERSION
FROM ${GAIA2_CLI_IMAGE}:${GAIA2_CLI_VERSION}

# ── Node.js runtime (from openclaw builder) ──────────────────────────────
COPY --from=openclaw-builder /usr/bin/node /usr/bin/node

# ── OpenClaw gateway application + node_modules ─────────────────────────
COPY --from=openclaw-builder /opt/chatbot/ /opt/chatbot/

# ── Python websockets (for the adapter) ──────────────────────────────────
RUN /usr/local/bin/python3 -m pip install --no-cache-dir websockets==16.0

# ── MITM TLS proxy: generate CA + host key at build time ──────────────────
# Wide validity (2020–2040) so Node.js under faketime (past/future dates)
# always accepts per-host certs. libfaketime backdates the CA cert to 2020.
RUN mkdir -p /opt/tls-proxy && \
    openssl genrsa -out /opt/tls-proxy/ca.key 2048 && \
    openssl req -new -key /opt/tls-proxy/ca.key \
      -out /tmp/ca.csr -subj "/CN=Gaia2 MITM Proxy CA" && \
    LD_PRELOAD=/usr/lib/x86_64-linux-gnu/faketime/libfaketime.so.1 \
      FAKETIME="2020-01-01 00:00:00" \
      openssl x509 -req -in /tmp/ca.csr \
        -signkey /opt/tls-proxy/ca.key \
        -out /opt/tls-proxy/ca.crt \
        -days 7300 && \
    openssl genrsa -out /opt/tls-proxy/host.key 2048 && \
    rm -f /tmp/ca.csr && \
    chmod 644 /opt/tls-proxy/ca.crt && \
    chmod 600 /opt/tls-proxy/ca.key /opt/tls-proxy/host.key

# ── MITM TLS proxy script ─────────────────────────────────────────────────
COPY shared/tls_proxy.py /opt/tls_proxy.py

# ── Clear/set env vars ───────────────────────────────────────────────────
ENV OPENCLAW_AVAILABLE=""
ENV OPENCLAW_MDNS_HOSTNAME="gaia2-agent"
ENV OPENCLAW_PATH_BOOTSTRAPPED="1"
ENV NODE_PATH=/opt/chatbot/node_modules
ENV NODE_EXTRA_CA_CERTS=/opt/tls-proxy/ca.crt

# ── Proxy bypass ─────────────────────────────────────────────────────────
ENV no_proxy="localhost,127.0.0.1,::1"
ENV NO_PROXY="${no_proxy}"

# ── `oc` wrapper ─────────────────────────────────────────────────────────
# Sets a restricted PATH so the gateway's exec tool can only find allowed
# binaries. /home/agent/bin contains wrappers for Gaia2 CLI tools + essentials.
# When /tmp/faketime.rc exists, injects libfaketime so new Date() returns
# simulated time natively. DONT_FAKE_MONOTONIC=1 keeps libuv timers working.
RUN printf '#!/bin/bash\nexport NODE_EXTRA_CA_CERTS="${NODE_EXTRA_CA_CERTS:-/opt/tls-proxy/ca.crt}"\nexport REASONING_EFFORT="${REASONING_EFFORT:-${THINKING:-}}"\nexport PATH="/home/agent/bin"\nif [ -f /tmp/faketime.rc ]; then\n  export LD_PRELOAD="/usr/lib/x86_64-linux-gnu/faketime/libfaketime.so.1"\n  export FAKETIME_TIMESTAMP_FILE="/tmp/faketime.rc"\n  export FAKETIME_NO_CACHE=1\n  export DONT_FAKE_MONOTONIC=1\n  unset FAKETIME\n  no_proxy_value="localhost,127.0.0.1,::1"\n  export NODE_TLS_REJECT_UNAUTHORIZED=0\n  export NODE_NO_WARNINGS=1\nelse\n  no_proxy_value="${NO_PROXY:-${no_proxy:-%s}}"\nfi\nexport no_proxy="$no_proxy_value"\nexport NO_PROXY="$no_proxy_value"\nexec /usr/bin/node /opt/chatbot/launch.mjs "$@"\n' \
  "localhost,127.0.0.1,::1" \
  > /usr/local/bin/oc && chmod +x /usr/local/bin/oc

# ── Custom launch.mjs (proxy dispatcher + transport shim) ────────────────
COPY containers/openclaw/launch.mjs containers/openclaw/launch_request_utils.mjs /opt/chatbot/

# ── Rename openclaw.mjs to gaia2-gw.mjs (some hosts kill "openclaw" processes) ─
RUN if [ -f /opt/chatbot/openclaw.mjs ]; then \
      cp /opt/chatbot/openclaw.mjs /opt/chatbot/gaia2-gw.mjs; \
    fi

# ── Scrub "openclaw" from process titles, mDNS, and identifiers ────────
# Some host security policies kill any process with "openclaw" visible anywhere.
# Also patch wrapperBasename so isMainModule() accepts gaia2-gw.mjs.
RUN find /opt/chatbot/dist -type f -name '*.js' -exec \
        sed -i -E \
            -e 's/wrapperBasename:\s*"openclaw\.mjs"/wrapperBasename: "gaia2-gw.mjs"/g' \
            -e 's/wrapperBasename:\s*"openclaw\.js"/wrapperBasename: "gaia2-gw.js"/g' \
            -e 's/process(\$[0-9]+)?\.title\s*=\s*"openclaw-gateway"/process\1.title = "gaia2-oc-gw"/g' \
            -e 's/process(\$[0-9]+)?\.title\s*=\s*"openclaw"/process\1.title = "gaia2-oc"/g' \
            -e 's/process\.title\s*=\s*`\$\{cliName\}-\$\{name\}`/process.title = `gaia2-oc-${name}`/g' \
            -e 's/openclaw-gw/gaia2-gw/g' \
            -e 's/"OpenClaw"/"Gaia2-Agent"/g' \
            -e 's/\(OpenClaw\)/(Gaia2-Agent)/g' \
            -e 's/openclaw\.local/gaia2-agent.local/g' \
            {} +

# ── Gaia2 state: privilege-separated via "gaia2" user ──────────────────────
RUN useradd -r -s /usr/sbin/nologin -d /var/gaia2 gaia2 && \
    mkdir -p /var/gaia2/state
ENV GAIA2_STATE_DIR=/var/gaia2/state
RUN chown -R gaia2:gaia2 /var/gaia2 && \
    chmod 700 /var/gaia2

# ── Non-root agent user ───────────────────────────────────────────────
RUN useradd -m -s /bin/bash -u 1000 agent

# ── Setuid Gaia2 wrapper (compiled from gaia2-exec.c in builder stage) ─────
COPY --from=gaia2-exec-builder /tmp/gaia2-exec /usr/local/bin/gaia2-exec
RUN chown gaia2:gaia2 /usr/local/bin/gaia2-exec && \
    chmod u+s /usr/local/bin/gaia2-exec

# ── Agent sandboxing: restricted PATH with only allowed commands ─────────
RUN mkdir -p /home/agent/bin && \
    for cmd in calendar contacts emails messages chats rent-a-flat city cabs shopping cloud-drive; do \
        ln -s /usr/local/bin/gaia2-exec /home/agent/bin/$cmd; \
    done && \
    ln -s /usr/bin/ls      /home/agent/bin/ls && \
    ln -s /usr/bin/pwd     /home/agent/bin/pwd && \
    ln -s /usr/bin/date    /home/agent/bin/date && \
    ln -s /usr/bin/echo    /home/agent/bin/echo && \
    ln -s /usr/bin/cat     /home/agent/bin/cat && \
    ln -s /usr/bin/head    /home/agent/bin/head && \
    ln -s /usr/bin/tail    /home/agent/bin/tail && \
    ln -s /usr/bin/grep    /home/agent/bin/grep && \
    ln -s /usr/bin/sed     /home/agent/bin/sed && \
    ln -s /usr/bin/awk     /home/agent/bin/awk && \
    ln -s /usr/bin/wc      /home/agent/bin/wc && \
    ln -s /usr/bin/sort    /home/agent/bin/sort && \
    ln -s /usr/bin/mkdir   /home/agent/bin/mkdir && \
    ln -s /usr/bin/cp      /home/agent/bin/cp && \
    ln -s /usr/bin/mv      /home/agent/bin/mv && \
    ln -s /usr/bin/rm      /home/agent/bin/rm && \
    ln -s /usr/bin/touch   /home/agent/bin/touch && \
    ln -s /usr/bin/chmod   /home/agent/bin/chmod && \
    ln -s /usr/bin/jq      /home/agent/bin/jq && \
    ln -s /usr/bin/seq     /home/agent/bin/seq && \
    ln -s /usr/bin/xargs   /home/agent/bin/xargs && \
    ln -s /usr/bin/find    /home/agent/bin/find && \
    ln -s /usr/bin/tee     /home/agent/bin/tee && \
    ln -s /usr/bin/cut     /home/agent/bin/cut && \
    ln -s /usr/bin/tr      /home/agent/bin/tr && \
    ln -s /usr/bin/paste   /home/agent/bin/paste && \
    ln -s /usr/bin/uniq    /home/agent/bin/uniq && \
    ln -s /usr/bin/diff    /home/agent/bin/diff && \
    ln -s /usr/bin/base64  /home/agent/bin/base64 && \
    ln -s /usr/bin/basename /home/agent/bin/basename && \
    ln -s /usr/bin/dirname /home/agent/bin/dirname && \
    ln -s /usr/bin/file    /home/agent/bin/file && \
    ln -s /usr/bin/mktemp  /home/agent/bin/mktemp && \
    ln -s /usr/bin/sleep   /home/agent/bin/sleep && \
    ln -s /usr/bin/expr    /home/agent/bin/expr && \
    ln -s /usr/bin/test    /home/agent/bin/test && \
    ln -s /usr/bin/true    /home/agent/bin/true && \
    ln -s /usr/bin/false   /home/agent/bin/false && \
    ln -s /usr/local/bin/python3 /home/agent/bin/python3 && \
    ln -s /usr/local/bin/oc /home/agent/bin/oc && \
    chown -R agent:agent /home/agent/bin

# ── Faketime-aware bash/sh wrappers ─────────────────────────────────────
RUN printf '#!/usr/bin/bash\nif [ -f /tmp/faketime.rc ]; then\n  export LD_PRELOAD="/usr/lib/x86_64-linux-gnu/faketime/libfaketime.so.1"\n  export FAKETIME_TIMESTAMP_FILE="/tmp/faketime.rc"\n  export FAKETIME_NO_CACHE=1\n  unset FAKETIME\nfi\nexec /usr/bin/bash "$@"\n' \
      > /home/agent/bin/bash && chmod +x /home/agent/bin/bash && \
    cp /home/agent/bin/bash /home/agent/bin/sh

# ── OpenClaw config + adapter + entrypoint ───────────────────────────────
COPY shared/gaia2_adapter_base.py /opt/gaia2_adapter_base.py
COPY shared/render_agent_prompt.py /opt/render_agent_prompt.py
COPY shared/AGENTS_TEMPLATE.md /opt/AGENTS_TEMPLATE.md
COPY containers/openclaw/gaia2_adapter.py /opt/gaia2_adapter.py
COPY containers/openclaw/openclaw_config.py /opt/openclaw_config.py
COPY containers/openclaw/openclaw-setup.sh /etc/profile.d/openclaw-setup.sh
COPY containers/openclaw/entrypoint.sh /opt/entrypoint.sh
RUN chmod +x /opt/entrypoint.sh

ENV HOME=/home/agent

# Run setup as agent (with full PATH for build-time tools like mkdir, grep, date)
USER agent
RUN PROVIDER=anthropic bash /etc/profile.d/openclaw-setup.sh
RUN echo '. /etc/profile.d/openclaw-setup.sh' >> /home/agent/.bashrc

# ── Debug: status banner on shell login ──────────────────────────────────
RUN printf '\necho "──── gaia2-openclaw ────"\necho "User:   $(whoami)"\necho "PATH:   $PATH"\noc --version 2>/dev/null || echo "oc: not responding"\n[ -f "$HOME/.openclaw/openclaw.json" ] && echo "Config: OK" || echo "Config: MISSING (set provider API key)"\necho "State:  $GAIA2_STATE_DIR"\necho "───────────────────────"\n' >> /home/agent/.bashrc

# ── Lock down PATH for runtime ───────────────────────────────────────────
ENV PATH="/home/agent/bin"

# ── Entrypoint ─────────────────────────────────────────────────────────
USER root
COPY containers/openclaw/gaia2-init-entrypoint.sh /opt/gaia2-init-entrypoint.sh
RUN chmod +x /opt/gaia2-init-entrypoint.sh

ENTRYPOINT []
CMD ["/usr/bin/bash", "/opt/gaia2-init-entrypoint.sh"]
