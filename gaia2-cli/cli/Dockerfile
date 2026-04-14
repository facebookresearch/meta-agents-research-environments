# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

# GAIA2 CLI Docker Image
# Minimal image with CLI entry points for benchmark evaluation.
#
# BUILD (from gaia2-cli/): podman build --network=host -f cli/Dockerfile -t gaia2-cli:latest .
# TEST:  podman run --rm gaia2-cli:latest calendar --help

FROM python:3.12.6-slim

# Configure APT sandbox to prevent permission issues
RUN echo 'APT::Sandbox::User "root";' | tee -a /etc/apt/apt.conf.d/10sandbox

# Install minimal system dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    curl \
    jq \
    libfaketime \
    sudo \
    vim-tiny \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV PIP_ROOT_USER_ACTION=ignore
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Copy gaia2-core (shared dependency) and install it first
COPY core/ /tmp/core/
RUN pip install --no-cache-dir /tmp/core/ && rm -rf /tmp/core/

# Copy project files
COPY cli/pyproject.toml .
COPY cli/gaia2_cli/ ./gaia2_cli/

# Install the package
RUN pip install --no-cache-dir ".[judge]"

# Remove build source (only needed for pip install)
RUN rm -rf /app/gaia2_cli/ /app/pyproject.toml

# Download gaia2_filesystem dataset (public, ~296 files).
# Scenarios reference these via real_path fields; gaia2-init copies
# only the files each scenario needs to /var/gaia2/state/filesystem/.
ARG HF_TOKEN=""
RUN pip install --no-cache-dir "huggingface_hub[cli]" && \
    HF_TOKEN="${HF_TOKEN}" python -c "from huggingface_hub import snapshot_download; snapshot_download('meta-agents-research-environments/gaia2_filesystem', repo_type='dataset', local_dir='/opt/gaia2_filesystem')" && \
    pip uninstall -y huggingface_hub filelock fsspec tqdm && \
    rm -rf /root/.cache/huggingface

ENV GAIA2_FS_BACKING_DIR=/opt/gaia2_filesystem

# Compile gaia2_cli to bytecode (.pyc) and strip .py source
RUN PKG=$(python -c "import gaia2_cli; print(gaia2_cli.__path__[0])") \
    && python -m compileall -b "$PKG" \
    && find "$PKG" -name "*.py" -delete \
    && find "$PKG" -type d -name "__pycache__" -exec rm -rf {} +

# No ENTRYPOINT - container stays alive via reset_entrypoint=True in VMVM config
CMD ["sleep", "infinity"]
