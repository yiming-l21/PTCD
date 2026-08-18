#!/usr/bin/env bash
set -euo pipefail

DATASET="${DATASET:-mvsa-s}"
PRED_FIELD="${PRED_FIELD:-pred_ptcd_demo1_perclass}"
LOG_PATH="${LOG_PATH:-logs/${DATASET}/${PRED_FIELD}_cd_debug.jsonl}"
OUT_DIR="${OUT_DIR:-analysis/${DATASET}}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python "${SCRIPT_DIR}/analyze_logs.py" \
  --log "${LOG_PATH}" \
  --out_dir "${OUT_DIR}"
