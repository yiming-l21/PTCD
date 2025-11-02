#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=2,4,5,6,7
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

export USE_DEMO=0
export SP_N_TOKENS=8
DATASETS=("tumemo")

for k in 1 2; do
  # topk=1 时两个模式等价，只跑 perclass；topk=2 时跑 perclass 和 balanced
  if [ "$k" -eq 1 ]; then
    MODES=("perclass")
  else
    MODES=("perclass" "balanced")
  fi
  if [ "$USE_DEMO" -eq 0 ]; then
    MODES=("none")
    if [ "$k" -ne 1 ]; then
      continue  
    fi
  fi
  for mode in "${MODES[@]}"; do
    export DEMO_TOPK="$k"
    export DEMO_MODE="$mode"
    export PRED_FIELD="pred_demo${DEMO_TOPK}_${DEMO_MODE}"

    echo "==== Running with DEMO_TOPK=${DEMO_TOPK}, DEMO_MODE=${DEMO_MODE} (PRED_FIELD=${PRED_FIELD}) ===="

    for dataset in "${DATASETS[@]}"; do
      echo "[start] dataset: ${dataset}"
      export TRAIN_JSONL="/home/lym/VLM-MSA/datasets/${dataset}/train_few1.json"
      export SOFT_PROMPT_CKPT="/home/lym/VLM-MSA/ckpt/${dataset}_prompt_ckpt.best.pt"

      python src/run.py \
        --data_dir "datasets/${dataset}" \
        --img_dir "datasets/${dataset}/imgs" \
        --tsv "test.tsv" \
        --model "/home/lym/models/qwen2.5_vl" \
        --dtype "bf16" \
        --attn_impl "sdpa" \
        --lang "en" \
        --max_new_tokens 16 \
        --distributed \
        --dump_raw

      echo "[done ] dataset: ${dataset}"
      echo "----------------------------------------"
    done

    echo "==== Finished combo: DEMO_TOPK=${DEMO_TOPK}, DEMO_MODE=${DEMO_MODE} ===="
    echo
  done
done

echo "All runs completed."
