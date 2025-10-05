#!/usr/bin/env bash
set -euo pipefail

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

DATASETS=("mvsa-s" "mvsa-m" "masad" "t2017" "t2015" "tumemo")


for dataset in "${DATASETS[@]}"; do
    echo "start processing dataset: ${dataset}"

    python model.py \
        --data_dir "datasets/${dataset}" \
        --img_dir "datasets/${dataset}/imgs" \
        --tsv "test.tsv" \
        --labels "positive,neutral,negative" \
        --model "/home/lym/models/qwen2.5_vl" \
        --dtype "bf16" \
        --attn_impl "sdpa" \
        --lang "en" \
        --max_new_tokens 16 \
        --distributed \
        --dump_raw \

    echo "dataset ${dataset} processed."
    echo "----------------------------------------"
done

echo "All datasets processed."
