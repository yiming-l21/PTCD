#!/usr/bin/env bash
set -euo pipefail

# Generate sentence embeddings used by retrieval-augmented demo selection.
# Override DATA_ROOT, SBERT_MODEL, DEMO_EMB_TAG, or EMB_BATCH_SIZE as needed.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

DATASET_NAME="${DATASET_NAME:-mvsa-s}"
DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/datasets}"
SBERT_MODEL="${SBERT_MODEL:-sentence-transformers/sbert-roberta-large}"
DEMO_EMB_TAG="${DEMO_EMB_TAG:-sbert-roberta-large}"
EMB_BATCH_SIZE="${EMB_BATCH_SIZE:-64}"

DATA_DIR="${DATA_ROOT}/${DATASET_NAME}"

python "${SCRIPT_DIR}/generate_emb.py" \
  --jsonl "${DATA_DIR}/train_few1.json" \
  --data_dir "${DATA_DIR}" \
  --split train \
  --model_tag "${DEMO_EMB_TAG}" \
  --hf_model "${SBERT_MODEL}" \
  --batch_size "${EMB_BATCH_SIZE}"

python "${SCRIPT_DIR}/generate_emb.py" \
  --jsonl "${DATA_DIR}/dev_few1.json" \
  --data_dir "${DATA_DIR}" \
  --split val \
  --model_tag "${DEMO_EMB_TAG}" \
  --hf_model "${SBERT_MODEL}" \
  --batch_size "${EMB_BATCH_SIZE}"

python "${SCRIPT_DIR}/generate_emb.py" \
  --jsonl "${DATA_DIR}/test.json" \
  --data_dir "${DATA_DIR}" \
  --split test \
  --model_tag "${DEMO_EMB_TAG}" \
  --hf_model "${SBERT_MODEL}" \
  --batch_size "${EMB_BATCH_SIZE}"
