#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
# Translate the Gaia2 benchmark into a target language.
#
# Produces a `dataset_root` directory of per-scenario JSON files that the
# gaia2-cli runner consumes directly via `[target].dataset_root` (see
# runner/examples/omnilingual_gaia2.toml).
#
# Targets a local OpenAI-compatible endpoint (vLLM), so it needs no hosted-API
# credentials.
#
# The defaults here reproduce the published pipeline: a single translator-only
# pass with the term-table contract enforced. Set REVIEW=1 to add the optional
# second review + post-edit pass (a reproducible negative result — it rewrote
# under 9% of fields, almost entirely stylistically, for ~2x the latency).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# --- LLM endpoint (local vLLM, OpenAI-compatible) ---------------------------
# Point at your served model. Override per-run.
export GAIA2_MT_LLM_BASE_URL="${GAIA2_MT_LLM_BASE_URL:-http://localhost:8000/v1}"
export GAIA2_MT_LLM_API_KEY="${GAIA2_MT_LLM_API_KEY:-EMPTY}"

# --- Translation settings ---------------------------------------------------
SUBSET="${SUBSET:-all}"                 # search | execution | ambiguity | adaptability | all
TGT_LANG="${TGT_LANG:-spa_Latn}"
DATASET_ID="${DATASET_ID:-meta-agents-research-environments/gaia2}"
# When GAIA2_MT_LLM_BASE_URL is set, model names must match `--served-model-name`.
TRANSLATION_MODEL="${TRANSLATION_MODEL:-google/gemma-4-31B-it}"
LIMIT="${LIMIT:-}"
# Optional second review + post-edit pass. Off, as published: it changed under
# 9% of fields, almost entirely stylistically, for roughly 2x the latency.
REVIEW="${REVIEW:-0}"
REVIEW_MODEL="${REVIEW_MODEL:-openai/gpt-oss-120b}"
# Advisory GlotLID check. Off, as it was for the released dataset; it gates
# nothing and only adds columns to RESULTS_CSV.
LID_CHECK="${LID_CHECK:-0}"
RESULTS_CSV="${RESULTS_CSV:-}"

# Output goes on shared NFS so it can be referenced as dataset_root from eval.
OUTPUT_BASE="${OUTPUT_BASE:-/path/to/scratch/${USER}/omnilingual-gaia2}"
OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_BASE}/${TGT_LANG}/data}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${OUTPUT_BASE}/${TGT_LANG}/checkpoints}"

cd "${MT_DIR}"

CLI_ARGS=(
    --output_dir "${OUTPUT_DIR}"
    --dataset_id "${DATASET_ID}"
    --subset "${SUBSET}"
    --tgt_lang "${TGT_LANG}"
    --translation_model "${TRANSLATION_MODEL}"
    --checkpoint_dir "${CHECKPOINT_DIR}"
)
[[ -n "${LIMIT}" ]] && CLI_ARGS+=(--limit "${LIMIT}")
[[ -n "${RESULTS_CSV}" ]] && CLI_ARGS+=(--results_csv "${RESULTS_CSV}")
if [[ "${REVIEW}" == "1" ]]; then
    CLI_ARGS+=(--review --review_model "${REVIEW_MODEL}")
fi
if [[ "${LID_CHECK}" == "1" ]]; then
    CLI_ARGS+=(--lid_check)
fi

echo "=== Omnilingual-GAIA2 Translation ==="
echo "  Endpoint:          ${GAIA2_MT_LLM_BASE_URL}"
echo "  Subset:            ${SUBSET}"
echo "  Target language:   ${TGT_LANG}"
echo "  Translation model: ${TRANSLATION_MODEL}"
if [[ "${REVIEW}" == "1" ]]; then
    echo "  Review model:      ${REVIEW_MODEL} (optional pass ENABLED)"
else
    echo "  Review pass:       disabled (translator-only, as published)"
fi
echo "  Limit:             ${LIMIT:-none}"
echo "  Output dir:        ${OUTPUT_DIR}"
echo "========================================"

python -m gaia2_mt.cli.translate "${CLI_ARGS[@]}"

echo "=== Translation completed. dataset_root: ${OUTPUT_DIR} ==="
