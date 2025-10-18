#!/usr/bin/env bash
set -euo pipefail

DATASETS=(mvsa-s mvsa-m masad t2015 t2017 tumemo)
SPLITS=(train_few1 dev_few1 test)

for dataset in "${DATASETS[@]}"; do
  for split in "${SPLITS[@]}"; do
    echo "start processing dataset: ${dataset}, split: ${split}"
    export DATASET_NAME="${dataset}"
    export SPLIT="${split}"
    python data_tsv2json.py
  done
  echo "dataset ${dataset} processed."
  echo "----------------------------------------"
done

for dataset in "${DATASETS[@]}"; do
  export DATASET_NAME="$dataset"
  echo "===start generate embedding for ${DATASET_NAME}==="
  bash generate.sh
done

for dataset in "${DATASETS[@]}"; do
  export DATASET_NAME="$dataset"
  echo "===start precompute topk for ${DATASET_NAME}==="
  for mode in val test; do
    export MODE="$mode"
    bash topk.sh
  done
done