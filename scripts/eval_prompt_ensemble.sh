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
  echo "[WARN] No GPU detected. The script will continue but may be slow."
fi
echo "[INFO] Visible GPU num: ${NUM_GPUS}"
[[ -n "${CUDA_VISIBLE_DEVICES}" ]] && echo "[INFO] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

# ---- common envs ----
export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=0

# ---- paths & defaults (按需修改) ----
MODEL_DIR="${MODEL_DIR:-${MODEL_NAME:-Qwen/Qwen2.5-VL-7B-Instruct}}"
DATA_ROOT="${DATA_ROOT:-${PTCD_ROOT}/datasets}"
DTYPE="${DTYPE:-bf16}"
ATTN_IMPL="${ATTN_IMPL:-sdpa}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-16}"

# 需要评测的数据集（目录需存在：datasets/<name>）
DATASETS=("t2015" "t2017" "tumemo")

run_one() {
  local dataset="$1"
  local run_suffix="$2"
  local mode="$3"   # STRICT | ENS3 | ENS5

  local data_dir="${DATA_ROOT}/${dataset}"
  local img_opt=()
  if [[ -d "${data_dir}/imgs" ]]; then
    img_opt+=(--img_dir "${data_dir}/imgs")
  fi

  # 每次调用前找一个空闲端口
  local PORT
  PORT="$(python - <<'PY'
import socket
s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()
PY
)"
  # 为了稳定，强制走回环网卡
  local envs="RUN_SUFFIX=${run_suffix} MASTER_ADDR=127.0.0.1 MASTER_PORT=${PORT} GLOO_SOCKET_IFNAME=lo NCCL_SOCKET_IFNAME=lo"

  common_args=(
    --data_dir "${data_dir}"
    --tsv "test.tsv"
    --model "${MODEL_DIR}"
    --dtype "${DTYPE}"
    --attn_impl "${ATTN_IMPL}"
    --lang "en"
    --max_new_tokens "${MAX_NEW_TOKENS}"
    --dump_raw
    --distributed
  )

  echo "----------------------------------------"
  echo "[RUN] dataset=${dataset}  mode=${mode}  RUN_SUFFIX=${run_suffix}  PORT=${PORT}"

  case "${mode}" in
    STRICT)
      env ${envs} PROMPT_VARIANT="STRICT" \
        python -m src.cli.run "${common_args[@]}" "${img_opt[@]}"
      ;;
    ENS3)
      env ${envs} PROMPT_ENSEMBLE="STRICT,IMAGE_FIRST,TEXT_FIRST" \
        python -m src.cli.run "${common_args[@]}" "${img_opt[@]}"
      ;;
    ENS5)
      env ${envs} PROMPT_ENSEMBLE="STRICT,IMAGE_FIRST,TEXT_FIRST,CONFLICT_AWARE,SARCASM_AWARE" \
        python -m src.cli.run "${common_args[@]}" "${img_opt[@]}"
      ;;
    *)
      echo "[ERR] unknown mode=${mode}" >&2
      exit 1
      ;;
  esac
}


cd "${PTCD_ROOT}"

for dataset in "${DATASETS[@]}"; do
  echo "========================================"
  echo "Start processing dataset: ${dataset}"

  run_one "${dataset}" "${dataset}_ENS3" "ENS3"

  run_one "${dataset}" "${dataset}_ENS5" "ENS5"

  echo "dataset ${dataset} processed."
done

echo "All datasets processed."
