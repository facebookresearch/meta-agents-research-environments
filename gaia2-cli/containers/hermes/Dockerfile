# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
# Gaia2 + Hermes Agent Docker image for benchmark evaluation.
# Combines the Hermes AI agent (Python, by Nous Research) with Gaia2 CLI tools
# in a sandboxed environment.
#
# BUILD (from gaia2-cli/):
#   make gaia2-hermes
# RUN: podman run --network=host -it -e API_KEY=<key> localhost/gaia2-hermes:latest
#
# Hermes is cloned from GitHub at build time (pinned tag).
# The runtime is split: the gaia2 adapter owns HTTP, the Hermes worker owns
# the AIAgent and connects over a Unix socket.

ARG GAIA2_CLI_IMAGE=localhost/gaia2-cli
ARG GAIA2_CLI_VERSION=latest
ARG HERMES_REPO=https://github.com/NousResearch/hermes-agent.git
ARG HERMES_VERSION=v2026.3.28

# ── Compile setuid wrapper from source ────────────────────────────────
FROM debian:bookworm-slim AS gaia2-exec-builder
RUN echo 'APT::Sandbox::User "root";' | tee -a /etc/apt/apt.conf.d/10sandbox
RUN apt-get update && apt-get install -y --no-install-recommends gcc libc6-dev \
    && rm -rf /var/lib/apt/lists/*
COPY shared/gaia2-exec.c /tmp/gaia2-exec.c
RUN gcc -O2 -o /tmp/gaia2-exec /tmp/gaia2-exec.c && strip /tmp/gaia2-exec

# ── Final image: gaia2-cli base + Hermes agent ────────────────────────
FROM ${GAIA2_CLI_IMAGE}:${GAIA2_CLI_VERSION}

ARG http_proxy
ARG https_proxy
ARG HERMES_REPO
ARG HERMES_VERSION

# ── Install git (not in gaia2-cli base) + clone Hermes from GitHub ────
RUN echo 'APT::Sandbox::User "root";' | tee -a /etc/apt/apt.conf.d/10sandbox && \
    apt-get update && apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/*
RUN http_proxy=${http_proxy} https_proxy=${https_proxy} \
    git clone --depth 1 --branch "${HERMES_VERSION}" "${HERMES_REPO}" /opt/hermes
RUN http_proxy=${http_proxy} https_proxy=${https_proxy} \
    pip install --no-cache-dir -e /opt/hermes

# ── Gaia2 state: privilege-separated via "gaia2" user ──────────────────
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
    ln -s /usr/local/bin/python3 /home/agent/bin/python3 && \
    ln -s /usr/bin/curl    /home/agent/bin/curl && \
    chown -R agent:agent /home/agent/bin

# ── Faketime-aware bash/sh wrappers ─────────────────────────────────────
RUN printf '#!/usr/bin/bash\nif [ -f /tmp/faketime.rc ]; then\n  export LD_PRELOAD="/usr/lib/x86_64-linux-gnu/faketime/libfaketime.so.1"\n  export FAKETIME_TIMESTAMP_FILE="/tmp/faketime.rc"\n  export FAKETIME_NO_CACHE=1\n  unset FAKETIME\nfi\nexec /usr/bin/bash "$@"\n' \
      > /home/agent/bin/bash && chmod +x /home/agent/bin/bash && \
    cp /home/agent/bin/bash /home/agent/bin/sh

# ── Adapter + entrypoint ─────────────────────────────────────────────────
COPY shared/gaia2_adapter_base.py /opt/gaia2_adapter_base.py
COPY shared/render_agent_prompt.py /opt/render_agent_prompt.py
COPY shared/AGENTS_TEMPLATE.md /opt/AGENTS_TEMPLATE.md
COPY containers/hermes/gaia2_adapter.py /opt/gaia2_adapter.py
COPY containers/hermes/hermes_worker.py /opt/hermes_worker.py
COPY containers/hermes/entrypoint.sh /opt/entrypoint.sh
RUN chmod +x /opt/entrypoint.sh

ENV HOME=/home/agent

# ── Debug: status banner on shell login ──────────────────────────────
RUN printf '\necho "──── gaia2-hermes ────"\necho "User:   $(whoami)"\necho "PATH:   $PATH"\necho "Hermes: installed"\necho "State:  $GAIA2_STATE_DIR"\necho "────────────────────"\n' >> /home/agent/.bashrc

# ── Lock down PATH for runtime ───────────────────────────────────────────
ENV PATH="/home/agent/bin"

# ── Entrypoint ─────────────────────────────────────────────────────────
USER root
COPY containers/hermes/gaia2-init-entrypoint.sh /opt/gaia2-init-entrypoint.sh
RUN chmod +x /opt/gaia2-init-entrypoint.sh

ENTRYPOINT []
CMD ["/usr/bin/bash", "/opt/gaia2-init-entrypoint.sh"]
