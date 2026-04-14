# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
# Gaia2 Oracle Docker image — replays ground-truth actions without an LLM.
#
# Lightweight image extending gaia2-cli with the oracle replay adapter.
# No Node.js, no Rust, no agent runtime. Just CLI tools + replay + HTTP status.
#
# BUILD (from gaia2-cli/):
#   make gaia2-oracle
# RUN:
#   podman run --rm --network=host \
#     -v "$SCENARIO:/var/gaia2/custom_scenario.json:ro,z" \
#     localhost/gaia2-oracle:latest

ARG GAIA2_CLI_IMAGE=localhost/gaia2-cli
ARG GAIA2_CLI_VERSION=latest

# ── Compile setuid wrapper from source ────────────────────────────────────
FROM debian:bookworm-slim AS gaia2-exec-builder
RUN echo 'APT::Sandbox::User "root";' | tee -a /etc/apt/apt.conf.d/10sandbox
RUN apt-get update && apt-get install -y --no-install-recommends gcc libc6-dev \
    && rm -rf /var/lib/apt/lists/*
COPY shared/gaia2-exec.c /tmp/gaia2-exec.c
RUN gcc -O2 -o /tmp/gaia2-exec /tmp/gaia2-exec.c && strip /tmp/gaia2-exec

# ── Final image: gaia2-cli base + oracle adapter ──────────────────────────
FROM ${GAIA2_CLI_IMAGE}:${GAIA2_CLI_VERSION}

# ── Gaia2 state: privilege-separated via "gaia2" user ──────────────────────
RUN useradd -r -s /usr/sbin/nologin -d /var/gaia2 gaia2 && \
    mkdir -p /var/gaia2/state

ENV GAIA2_STATE_DIR=/var/gaia2/state
RUN chown -R gaia2:gaia2 /var/gaia2 && \
    chmod 700 /var/gaia2

# ── Non-root agent user (needed for gaia2-exec target) ────────────────────
RUN useradd -m -s /bin/bash -u 1000 agent

# ── Setuid Gaia2 wrapper ──────────────────────────────────────────────────
COPY --from=gaia2-exec-builder /tmp/gaia2-exec /usr/local/bin/gaia2-exec
RUN chown gaia2:gaia2 /usr/local/bin/gaia2-exec && \
    chmod u+s /usr/local/bin/gaia2-exec

# ── Agent sandboxing: restricted PATH with CLI tools via gaia2-exec ───────
RUN mkdir -p /home/agent/bin && \
    for cmd in calendar contacts emails messages chats rent-a-flat city cabs shopping cloud-drive; do \
        ln -s /usr/local/bin/gaia2-exec /home/agent/bin/$cmd; \
    done && \
    ln -s /usr/bin/ls      /home/agent/bin/ls && \
    ln -s /usr/bin/cat     /home/agent/bin/cat && \
    ln -s /usr/bin/date    /home/agent/bin/date && \
    ln -s /usr/bin/echo    /home/agent/bin/echo && \
    ln -s /usr/bin/bash    /home/agent/bin/bash && \
    ln -s /usr/bin/sleep   /home/agent/bin/sleep && \
    chown -R agent:agent /home/agent/bin

# ── Oracle adapter + entrypoints ─────────────────────────────────────────
COPY shared/gaia2_adapter_base.py /opt/gaia2_adapter_base.py
COPY containers/oracle/gaia2_oracle_adapter.py /opt/gaia2_oracle_adapter.py
COPY containers/oracle/gaia2-init-entrypoint.sh /opt/gaia2-init-entrypoint.sh
COPY containers/oracle/entrypoint.sh /opt/entrypoint.sh

RUN chmod +x /opt/gaia2-init-entrypoint.sh /opt/entrypoint.sh

ENTRYPOINT []
CMD ["/usr/bin/bash", "/opt/gaia2-init-entrypoint.sh"]
