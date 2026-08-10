#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
# Translate the Gaia2 benchmark into a target language.
#
# Produces a `dataset_root` directory of per-scenario JSON files that the
# gaia2-cli runner consumes directly via `[target].dataset_root` (see
# runner/examples/cluster/omnigaia_mt.toml).
#
# Unlike the in-monorepo wrapper, this targets a local OpenAI-compatible
# endpoint (vLLM) instead of the internal Llama API — no `with-proxy` / fwdproxy.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# --- LLM endpoint (local vLLM, OpenAI-compatible) ---------------------------
# Point both translation and review at a served model. Override per-run.
export OMNIGAIA_LLM_BASE_URL="${OMNIGAIA_LLM_BASE_URL:-http://localhost:8000/v1}"
export OMNIGAIA_LLM_API_KEY="${OMNIGAIA_LLM_API_KEY:-EMPTY}"

# --- Translation settings ---------------------------------------------------
SUBSET="${SUBSET:-all}"                 # search | execution | ambiguity | adaptability | all
TGT_LANG="${TGT_LANG:-spa_Latn}"
DATASET_ID="${DATASET_ID:-meta-agents-research-environments/gaia2}"
# When OMNIGAIA_LLM_BASE_URL is set, model names must match `--served-model-name`.
TRANSLATION_MODEL="${TRANSLATION_MODEL:-gemma-4-31b}"
REVIEW_MODEL="${REVIEW_MODEL:-gpt-oss-120b}"
LIMIT="${LIMIT:-}"
LID_CHECK="${LID_CHECK:-1}"

# Output goes on shared NFS so it can be referenced as dataset_root from eval.
OUTPUT_BASE="${OUTPUT_BASE:-/path/to/scratch/${USER}/omnigaia/omnigaia-mt}"
OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_BASE}/${TGT_LANG}/data}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${OUTPUT_BASE}/${TGT_LANG}/checkpoints}"

cd "${MT_DIR}"

CLI_ARGS=(
    --output_dir "${OUTPUT_DIR}"
    --dataset_id "${DATASET_ID}"
    --subset "${SUBSET}"
    --tgt_lang "${TGT_LANG}"
    --translation_model "${TRANSLATION_MODEL}"
    --review_model "${REVIEW_MODEL}"
    --checkpoint_dir "${CHECKPOINT_DIR}"
)
[[ -n "${LIMIT}" ]] && CLI_ARGS+=(--limit "${LIMIT}")
if [[ "${LID_CHECK}" == "1" ]]; then
    CLI_ARGS+=(--lid_check)
else
    CLI_ARGS+=(--nolid_check)
fi

echo "=== OmniGAIA-MT Translation ==="
echo "  Endpoint:          ${OMNIGAIA_LLM_BASE_URL}"
echo "  Subset:            ${SUBSET}"
echo "  Target language:   ${TGT_LANG}"
echo "  Translation model: ${TRANSLATION_MODEL}"
echo "  Review model:      ${REVIEW_MODEL}"
echo "  Limit:             ${LIMIT:-none}"
echo "  Output dir:        ${OUTPUT_DIR}"
echo "========================================"

python -m omnigaia.cli.translate "${CLI_ARGS[@]}"

echo "=== Translation completed. dataset_root: ${OUTPUT_DIR} ==="
