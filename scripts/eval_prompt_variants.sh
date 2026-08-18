#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PTCD_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
: "${CUDA_VISIBLE_DEVICES:=}" 

detect_gpus() {
  if [[ -n "${CUDA_VISIBLE_DEVICES}" ]]; then
    IFS=',' read -ra IDS <<< "${CUDA_VISIBLE_DEVICES}"
    echo "${#IDS[@]}"
  else
    if command -v nvidia-smi >/dev/null 2>&1; then
      nvidia-smi -L | wc -l | tr -d ' '
    else
      echo "0"
    fi
  fi
}

NUM_GPUS="$(detect_gpus)"
if [[ "${NUM_GPUS}" -lt 1 ]]; then
  echo "[WARN] no GPU detected, exiting."
fi
echo "[INFO] visible gpu num: ${NUM_GPUS}"
[[ -n "${CUDA_VISIBLE_DEVICES}" ]] && echo "[INFO] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=0

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-VL-7B-Instruct}"
DATA_ROOT="${DATA_ROOT:-${PTCD_ROOT}/datasets}"
DATASETS=("mvsa-s" "mvsa-m" "masad" "t2015" "t2017" "tumemo")
VARIANTS=("IMAGE_FIRST" "TEXT_FIRST" "CONFLICT_AWARE" "SARCASM_AWARE" "STRICT")

cd "${PTCD_ROOT}"

for dataset in "${DATASETS[@]}"; do
    echo "start processing dataset: ${dataset}"
    for variant in "${VARIANTS[@]}"; do
      export PROMPT_VARIANT="${variant}"
      python -m src.cli.run \
          --data_dir "${DATA_ROOT}/${dataset}" \
          --img_dir "${DATA_ROOT}/${dataset}/imgs" \
          --tsv "test.tsv" \
          --labels "positive,neutral,negative" \
          --model "${MODEL_NAME}" \
          --dtype "bf16" \
          --attn_impl "sdpa" \
          --lang "en" \
          --max_new_tokens 16 \
          --distributed \
          --dump_raw

      echo "dataset ${dataset} processed."
      echo "----------------------------------------"
    done
done

echo "All datasets processed."
