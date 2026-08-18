#!/usr/bin/env bash
set -euo pipefail

export USE_DEMO=0
export DEMO_CONTRASTIVE=0
export RUN_SUFFIX="${RUN_SUFFIX:-PROMPT_ONLY}"
export PRED_FIELD="${PRED_FIELD:-pred_prompt_only}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/eval_ptcd.sh" "$@"
