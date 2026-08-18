#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

DATASET="$(ensure_dataset_arg "${1:-mvsa-s}")"
CONFIG="${PTCD_CONFIG:-${PTCD_ROOT}/configs/paper/${DATASET}.json}"
DATA_ROOT="${DATA_ROOT:-${PTCD_ROOT}/datasets}"
OUTPUT_DIR="${OUTPUT_DIR:-${PTCD_ROOT}/outputs}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-VL-7B-Instruct}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export ENABLE_GRADIENT_CHECKPOINTING="${ENABLE_GRADIENT_CHECKPOINTING:-0}"

export TEXT_PROMPT_ONLY="${TEXT_PROMPT_ONLY:-0}"
export VISUAL_PROMPT_ONLY="${VISUAL_PROMPT_ONLY:-0}"
export SP_N_TOKENS="${SP_N_TOKENS:-$(json_get "${CONFIG}" sp_n_tokens 8)}"
export VISUAL_SP_N_TOKENS="${VISUAL_SP_N_TOKENS:-$(json_get "${CONFIG}" visual_sp_n_tokens 16)}"
export TARGET_MODE="${TARGET_MODE:-$(json_get "${CONFIG}" target_mode token)}"

# Prompt tuning is the training stage of PTCD. Retrieval demos are used by
# eval_ptcd.sh during contrastive decoding, so training keeps demos off by default.
export USE_DEMO="${USE_DEMO:-0}"
export TRAIN_JSONL="${TRAIN_JSONL:-${DATA_ROOT}/${DATASET}/train_few1.json}"

TRAIN_TSV="$(json_get "${CONFIG}" train_tsv train_few1.tsv)"
DEV_TSV="$(json_get "${CONFIG}" dev_tsv dev_few1.tsv)"
TEST_TSV="$(json_get "${CONFIG}" test_tsv test.tsv)"
SP_LR="$(json_get "${CONFIG}" sp_lr 1e-4)"
SP_STEPS="$(json_get "${CONFIG}" sp_steps 1500)"
SP_ACCUM="$(json_get "${CONFIG}" sp_accum 1)"
SP_WARMUP="$(json_get "${CONFIG}" sp_warmup 200)"
BATCH_SIZE="$(json_get "${CONFIG}" batch_size 4)"
TEMPLATE_ID="$(json_get "${CONFIG}" template_id 2)"
SP_DROPOUT="$(json_get "${CONFIG}" sp_dropout 0.2)"
VISUAL_SP_DROPOUT="$(json_get "${CONFIG}" visual_sp_dropout 0.2)"
MIN_PIXELS="$(json_get "${CONFIG}" min_pixels 50176)"
MAX_PIXELS="$(json_get "${CONFIG}" max_pixels 1048576)"

CKPT_DIR="${CKPT_DIR:-${OUTPUT_DIR}/checkpoints/${DATASET}}"
STEP_CKPT_DIR="${STEP_CKPT_DIR:-${CKPT_DIR}/step_ckpts}"
LOG_DIR="${LOG_DIR:-${OUTPUT_DIR}/logs/${DATASET}}"
mkdir -p "${CKPT_DIR}" "${STEP_CKPT_DIR}" "${LOG_DIR}"

cd "${PTCD_ROOT}"

echo "[INFO] dataset=${DATASET}"
echo "[INFO] config=${CONFIG}"
echo "[INFO] model=${MODEL_NAME}"
echo "[INFO] data_root=${DATA_ROOT}"
echo "[INFO] checkpoint_dir=${CKPT_DIR}"
echo "[INFO] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

python src/prompt_tuning/train_soft_prompt.py \
  --model "${MODEL_NAME}" \
  --data_dir "${DATA_ROOT}/${DATASET}" \
  --img_dir "${DATA_ROOT}/${DATASET}/imgs" \
  --tsv "${TEST_TSV}" \
  --train_tsv "${TRAIN_TSV}" \
  --dev_tsv "${DEV_TSV}" \
  --dtype bf16 \
  --min_pixels "${MIN_PIXELS}" \
  --max_pixels "${MAX_PIXELS}" \
  --batch_size "${BATCH_SIZE}" \
  --sp_mode generic \
  --sp_lr "${SP_LR}" \
  --sp_steps "${SP_STEPS}" \
  --sp_accum "${SP_ACCUM}" \
  --sp_warmup "${SP_WARMUP}" \
  --sp_ckpt "${CKPT_DIR}/prompt_ckpt.pt" \
  --sp_best "${CKPT_DIR}/prompt_ckpt.best.pt" \
  --step_ckpt_dir "${STEP_CKPT_DIR}" \
  --sp_dropout "${SP_DROPOUT}" \
  --visual_sp_dropout "${VISUAL_SP_DROPOUT}" \
  --template_id "${TEMPLATE_ID}" \
  --seed "${SEED:-34}" \
  --eval_every "${EVAL_EVERY:-200}" \
  --log_every "${LOG_EVERY:-200}" \
  2>&1 | tee "${LOG_DIR}/train_ptcd.log"
