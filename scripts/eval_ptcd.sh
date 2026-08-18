#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

DATASET="$(ensure_dataset_arg "${1:-mvsa-s}")"
CONFIG="${PTCD_CONFIG:-${PTCD_ROOT}/configs/paper/${DATASET}.json}"
DATA_ROOT="${DATA_ROOT:-${PTCD_ROOT}/datasets}"
OUTPUT_DIR="${OUTPUT_DIR:-${PTCD_ROOT}/outputs}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-VL-7B-Instruct}"
DEMO_EMB_TAG="${DEMO_EMB_TAG:-sbert-roberta-large}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

export TEXT_PROMPT_ONLY="${TEXT_PROMPT_ONLY:-0}"
export VISUAL_PROMPT_ONLY="${VISUAL_PROMPT_ONLY:-0}"
export SP_N_TOKENS="${SP_N_TOKENS:-$(json_get "${CONFIG}" sp_n_tokens 8)}"
export VISUAL_SP_N_TOKENS="${VISUAL_SP_N_TOKENS:-$(json_get "${CONFIG}" visual_sp_n_tokens 16)}"
export TARGET_MODE="${TARGET_MODE:-$(json_get "${CONFIG}" target_mode token)}"

export USE_DEMO="${USE_DEMO:-1}"
export DEMO_CONTRASTIVE="${DEMO_CONTRASTIVE:-1}"
export DEMO_TOPK="${DEMO_TOPK:-$(json_get "${CONFIG}" demo_topk 1)}"
export DEMO_MODE="${DEMO_MODE:-$(json_get "${CONFIG}" demo_mode perclass)}"
export DEMO_TAU_HIGH="${DEMO_TAU_HIGH:-$(json_get "${CONFIG}" demo_tau_high 0.3)}"
export DEMO_GAMMA="${DEMO_GAMMA:-$(json_get "${CONFIG}" demo_gamma 7.5)}"
export DEMO_LAMBDA_SIM="${DEMO_LAMBDA_SIM:-$(json_get "${CONFIG}" demo_lambda_sim 0.05)}"
export TRAIN_JSONL="${TRAIN_JSONL:-${DATA_ROOT}/${DATASET}/train_few1.json}"
export RUN_SUFFIX="${RUN_SUFFIX:-PTCD}"
export PRED_FIELD="${PRED_FIELD:-pred_ptcd_demo${DEMO_TOPK}_${DEMO_MODE}}"

TEST_TSV="$(json_get "${CONFIG}" test_tsv test.tsv)"
TEMPLATE_ID="$(json_get "${CONFIG}" template_id 2)"
MIN_PIXELS="$(json_get "${CONFIG}" min_pixels 50176)"
MAX_PIXELS="$(json_get "${CONFIG}" max_pixels 1048576)"

CKPT_DIR="${CKPT_DIR:-${OUTPUT_DIR}/checkpoints/${DATASET}}"
DEFAULT_BEST="${CKPT_DIR}/prompt_ckpt.best.pt"
DEFAULT_FINAL="${CKPT_DIR}/prompt_ckpt.pt"
if [[ -z "${SOFT_PROMPT_CKPT:-}" ]]; then
  if [[ -f "${DEFAULT_BEST}" ]]; then
    export SOFT_PROMPT_CKPT="${DEFAULT_BEST}"
  elif [[ -f "${DEFAULT_FINAL}" ]]; then
    export SOFT_PROMPT_CKPT="${DEFAULT_FINAL}"
  else
    echo "[ERR] no soft-prompt checkpoint found under ${CKPT_DIR}" >&2
    echo "      run: bash scripts/train_ptcd.sh ${DATASET}" >&2
    echo "      or set SOFT_PROMPT_CKPT=/path/to/prompt_ckpt.pt" >&2
    exit 1
  fi
fi

if [[ "${USE_DEMO}" == "1" || "${USE_DEMO}" == "true" ]]; then
  case "${DEMO_MODE}" in
    perclass)
      if ! compgen -G "${DATA_ROOT}/${DATASET}/test2train_${DEMO_EMB_TAG}_perclass_top*.npz" >/dev/null; then
        echo "[ERR] missing per-class retrieval index for ${DATASET}" >&2
        echo "      expected: ${DATA_ROOT}/${DATASET}/test2train_${DEMO_EMB_TAG}_perclass_top*.npz" >&2
        echo "      prepare it with: (cd datasets && DATASETS=${DATASET} bash run_data.sh)" >&2
        echo "      or set USE_DEMO=0 for a prompt-only ablation." >&2
        exit 1
      fi
      ;;
    global)
      if [[ ! -f "${DATA_ROOT}/${DATASET}/test2train_${DEMO_EMB_TAG}_top10_idx.npy" ]]; then
        echo "[ERR] missing global retrieval index for ${DATASET}" >&2
        exit 1
      fi
      ;;
  esac
fi

cd "${PTCD_ROOT}"

DIST_ARGS=()
if [[ "${DISTRIBUTED:-1}" == "1" ]]; then
  DIST_ARGS+=(--distributed)
fi
if [[ "${DUMP_RAW:-1}" == "1" ]]; then
  DIST_ARGS+=(--dump_raw)
fi

echo "[INFO] dataset=${DATASET}"
echo "[INFO] config=${CONFIG}"
echo "[INFO] model=${MODEL_NAME}"
echo "[INFO] soft_prompt=${SOFT_PROMPT_CKPT}"
echo "[INFO] demo=${USE_DEMO} mode=${DEMO_MODE} topk=${DEMO_TOPK} contrastive=${DEMO_CONTRASTIVE}"
echo "[INFO] gating tau_high=${DEMO_TAU_HIGH} gamma=${DEMO_GAMMA} lambda_sim=${DEMO_LAMBDA_SIM}"

python src/run.py \
  --data_dir "${DATA_ROOT}/${DATASET}" \
  --img_dir "${DATA_ROOT}/${DATASET}/imgs" \
  --tsv "${TEST_TSV}" \
  --model "${MODEL_NAME}" \
  --dtype bf16 \
  --attn_impl sdpa \
  --lang en \
  --max_new_tokens "${MAX_NEW_TOKENS:-16}" \
  --template_id "${TEMPLATE_ID}" \
  --min_pixels "${MIN_PIXELS}" \
  --max_pixels "${MAX_PIXELS}" \
  "${DIST_ARGS[@]}"
