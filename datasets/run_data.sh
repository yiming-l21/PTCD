#!/usr/bin/env bash
set -euo pipefail

# Paper datasets by default. Add experimental datasets with:
# DATASETS="mvsa-s mvsa-m t2015 t2017 masad tumemo" bash run_data.sh
read -r -a DATASETS_ARR <<< "${DATASETS:-mvsa-s mvsa-m t2015 t2017}"
read -r -a SPLITS_ARR <<< "${SPLITS:-train_few1 dev_few1 test}"

for dataset in "${DATASETS_ARR[@]}"; do
  for split in "${SPLITS_ARR[@]}"; do
    echo "start processing dataset: ${dataset}, split: ${split}"
    DATASET_NAME="${dataset}" SPLIT="${split}" python data_tsv2json.py
  done
  echo "dataset ${dataset} json conversion done."
  echo "----------------------------------------"
done

for dataset in "${DATASETS_ARR[@]}"; do
  echo "=== start generate embedding for ${dataset} ==="
  DATASET_NAME="${dataset}" bash generate.sh
done

for dataset in "${DATASETS_ARR[@]}"; do
  echo "=== start precompute topk for ${dataset} ==="
  for mode in val test; do
    DATASET_NAME="${dataset}" MODE="${mode}" bash topk.sh
  done
done
