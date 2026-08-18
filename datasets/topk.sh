#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

DATASET_NAME="${DATASET_NAME:-mvsa-s}"
MODE="${MODE:-test}"
DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/datasets}"
DEMO_EMB_TAG="${DEMO_EMB_TAG:-sbert-roberta-large}"
STORE_K="${STORE_K:-10}"
PER_CLASS_K="${PER_CLASS_K:-5}"

DATA="${DATA_ROOT}/${DATASET_NAME}"

python "${SCRIPT_DIR}/precompute_topk.py" \
  --train_emb "${DATA}/train_${DEMO_EMB_TAG}.npy" \
  --query_emb "${DATA}/${MODE}_${DEMO_EMB_TAG}.npy" \
  --store_k "${STORE_K}" \
  --out_prefix "${DATA}/${MODE}2train_${DEMO_EMB_TAG}" \
  --train_file "${DATA}/train_few1.json" \
  --label_field label \
  --dataset_name "${DATASET_NAME}" \
  --per_class_k "${PER_CLASS_K}" \
  --balance_strategy roundrobin
