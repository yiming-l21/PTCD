#!/usr/bin/env bash
set -euo pipefail

# -------------------- GPU 选择 --------------------
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"5,7"}
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
  exit 1
fi
echo "[INFO] visible gpu num: ${NUM_GPUS}"
[[ -n "${CUDA_VISIBLE_DEVICES}" ]] && echo "[INFO] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

# -------------------- 环境变量 --------------------
export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=0
export TEXT_PROMPT_ONLY="${TEXT_PROMPT_ONLY:-0}"  # 1=仅文本，0=不单独启用
export VISUAL_PROMPT_ONLY="${VISUAL_PROMPT_ONLY:-0}"  # 1=仅视觉，0=不单独启用
export USE_DEMO=0
export SP_N_TOKENS=8
export DEMO_CONTRASTIVE=0
export TARGET_MODE="token"
export MODES="perclass"
DATASETS=("t2015")  

# 可选：指定 ckpt 目录；若不指定，则默认 /home/lym/VLM-MSA/ckpt/${dataset}
# export SOFT_DIR="/home/lym/VLM-MSA/ckpt/mvsa-s"

# 让空通配不报错
shopt -s nullglob

for k in 1; do
  # topk=1 时两个模式等价，只跑 perclass；topk=2 时跑 perclass 和 balanced
  if [ "$k" -eq 1 ]; then
    MODES=("perclass")
  else
    MODES="global"
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
      if [ "$dataset" == "t2015" ] || [ "$dataset" == "t2017" ] || [ "$dataset" == "tumemeo" ]; then
        export TARGET_MODE="json"
      else
        export TARGET_MODE="token"
      fi

      # --------- 修复：遍历主目录+step_ckpts目录下所有ckpt ----------
      # 目录优先级：SOFT_DIR（若设置） -> /home/lym/VLM-MSA/ckpt/${dataset}
      CKPT_DIR="${SOFT_DIR:-/home/lym/VLM-MSA/ckpt2/${dataset}}"
      STEP_CKPT_DIR="${CKPT_DIR}/step_ckpts"  # 按步保存的子目录
      if [[ ! -d "${CKPT_DIR}" ]]; then
        echo "[WARN] ckpt dir not found: ${CKPT_DIR} (skip dataset ${dataset})"
        continue
      fi

      # 修复：正确收集并排序ckpt文件（解决语法错误）
      # 先收集所有.pt文件到临时变量，再排序
      CKPTS_TMP=()
      # 主目录的ckpt（best.pt/final.pt）
      while IFS= read -r -d '' file; do
        CKPTS_TMP+=("$file")
      done < <(find "${CKPT_DIR}" -maxdepth 1 -type f -name "*.pt" -print0)
      # step_ckpts目录的按步保存ckpt
      if [[ -d "${STEP_CKPT_DIR}" ]]; then
        while IFS= read -r -d '' file; do
          CKPTS_TMP+=("$file")
        done < <(find "${STEP_CKPT_DIR}" -maxdepth 1 -type f -name "*.pt" -print0)
      fi

      # 按修改时间升序排序（使用stat获取修改时间，兼容不同系统）
      # 修复：避免空数组排序报错
      if [[ ${#CKPTS_TMP[@]} -eq 0 ]]; then
        echo "[WARN] no *.pt found in ${CKPT_DIR} and ${STEP_CKPT_DIR}"
        continue
      fi

      # 按修改时间排序（升序： oldest -> newest）
      if command -v stat >/dev/null 2>&1; then
        # Linux系统：使用stat获取修改时间（%Y：秒级时间戳）
        CKPTS=($(for file in "${CKPTS_TMP[@]}"; do
          stat -c "%Y %n" "$file"
        done | sort -n | awk '{print $2}'))
      else
        # macOS系统（如需兼容）：使用stat -f
        CKPTS=($(for file in "${CKPTS_TMP[@]}"; do
          stat -f "%m %N" "$file"
        done | sort -n | awk '{print $2}'))
      fi

      echo "[INFO] found ${#CKPTS[@]} soft prompts (main: ${CKPT_DIR}, step: ${STEP_CKPT_DIR})"
      for ckpt in "${CKPTS[@]}"; do
        export SOFT_PROMPT_CKPT="${ckpt}"
        base_ckpt="$(basename "${ckpt}")"
        echo "[INFO] running inference with SOFT_PROMPT_CKPT=${SOFT_PROMPT_CKPT}"

        # 如需区分日志，可在环境或 run.py 里加入 RUN_TAG=base_ckpt；此处仅示例打印
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

        echo "[done ] dataset=${dataset} ckpt=${base_ckpt}"
        echo "----------------------------------------"
      done
      # -------------------------------------------------------------

    done

    echo "==== Finished combo: DEMO_TOPK=${DEMO_TOPK}, DEMO_MODE=${DEMO_MODE} ===="
    echo
  done
done

echo "All runs completed."