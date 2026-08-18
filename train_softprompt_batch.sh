#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Space-separated dataset list, e.g.:
# DATASETS="mvsa-s mvsa-m t2015 t2017" bash train_softprompt_batch.sh
read -r -a DATASETS_ARR <<< "${DATASETS:-mvsa-s mvsa-m t2015 t2017}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -r -a AVAILABLE_GPUS <<< "${CUDA_VISIBLE_DEVICES}"
NUM_GPUS="${#AVAILABLE_GPUS[@]}"

if [[ "${NUM_GPUS}" -lt 1 ]]; then
  echo "[ERR] no visible GPU configured" >&2
  exit 1
fi

echo "[INFO] datasets: ${DATASETS_ARR[*]}"
echo "[INFO] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

idx=0
while [[ "${idx}" -lt "${#DATASETS_ARR[@]}" ]]; do
  pids=()
  for gpu in "${AVAILABLE_GPUS[@]}"; do
    if [[ "${idx}" -ge "${#DATASETS_ARR[@]}" ]]; then
      break
    fi
    dataset="${DATASETS_ARR[$idx]}"
    echo "[INFO] launch dataset=${dataset} on visible GPU id ${gpu}"
    CUDA_VISIBLE_DEVICES="${gpu}" bash "${SCRIPT_DIR}/scripts/train_ptcd.sh" "${dataset}" &
    pids+=("$!")
    idx=$((idx + 1))
  done
  wait "${pids[@]}"
done

echo "[INFO] all batch prompt-tuning jobs completed."
