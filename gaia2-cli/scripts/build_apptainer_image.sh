#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
# build_apptainer_image.sh — convert the bundled OCI archive to a SquashFS .sif.
#
# Run on a node that has apptainer + fakeroot subuid mappings (most compute
# nodes; login nodes typically lack both). One-shot: re-run only when
# gaia2-images.tar is rebuilt. Output .sif is portable and read-only thereafter.
#
# Usage:
#   srun --partition=cpu --qos=<qos> --time=15:00 --cpus-per-task=4 --mem=16G \
#        bash gaia2-cli/scripts/build_apptainer_image.sh
#
# Override paths via env:
#   GAIA2_OC_ARCHIVE  (default: gaia2-cli/gaia2-images.tar relative to repo)
#   GAIA2_OC_SIF      (default: gaia2-cli/gaia2-images/gaia2-oc.sif)
#   TMPDIR            scratch space for the intermediate .sif (default: /tmp)
# Apptainer's own APPTAINER_CACHEDIR / APPTAINER_TMPDIR are inherited as-is.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
GAIA2_OC_ARCHIVE="${GAIA2_OC_ARCHIVE:-$REPO_ROOT/gaia2-cli/gaia2-images.tar}"
GAIA2_OC_SIF="${GAIA2_OC_SIF:-$REPO_ROOT/gaia2-cli/gaia2-images/gaia2-oc.sif}"

if ! command -v apptainer >/dev/null 2>&1; then
    echo "ERROR: apptainer not found on this node" >&2
    exit 1
fi
if [[ ! -f "$GAIA2_OC_ARCHIVE" ]]; then
    echo "ERROR: OCI archive missing: $GAIA2_OC_ARCHIVE" >&2
    exit 1
fi

mkdir -p "$(dirname "$GAIA2_OC_SIF")"

# Build to scratch space then move into place — the repo path may be size- or
# permission-constrained. Set TMPDIR to point at somewhere with room for the
# intermediate .sif (roughly the image size).
WORK_SIF="${TMPDIR:-/tmp}/gaia2-apptainer-build/gaia2-oc.$$.sif"
mkdir -p "$(dirname "$WORK_SIF")"
trap 'rm -f "$WORK_SIF"' EXIT

echo "[build] node=$(hostname -f)"
echo "[build] archive=$GAIA2_OC_ARCHIVE"
echo "[build] sif=$GAIA2_OC_SIF"

# Auto-detect archive format: `podman save` defaults to docker-archive
# (top-level manifest.json + flat <sha>.tar layers); `podman save --format=oci-archive`
# produces an OCI image layout (oci-layout + index.json + blobs/sha256/...).
# Apptainer needs the source type to match — passing the wrong one fails with
# obscure "no such file or directory" errors deep in skopeo's internal conversion.
if tar -tf "$GAIA2_OC_ARCHIVE" 2>/dev/null | grep -q '^oci-layout$'; then
    ARCHIVE_SOURCE="oci-archive:$GAIA2_OC_ARCHIVE"
else
    ARCHIVE_SOURCE="docker-archive:$GAIA2_OC_ARCHIVE"
fi
echo "[build] source=$ARCHIVE_SOURCE"

# --fakeroot: needed so chown/setuid layers in the OCI image translate cleanly
# into the SquashFS without permission errors.
apptainer build --fakeroot --force "$WORK_SIF" "$ARCHIVE_SOURCE"

mv -f "$WORK_SIF" "$GAIA2_OC_SIF"
trap - EXIT

ls -lh "$GAIA2_OC_SIF"
echo "[build] done"
