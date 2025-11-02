#!/usr/bin/env bash
set -euo pipefail

# -------------------- GPU 选择 --------------------
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"6,7,5"}
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

# -------------------- 环境变量 --------------------
export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=0

export USE_DEMO=0
export SP_N_TOKENS=8
DATASETS=("mvsa-s" "mvsa-m" "masad" "t2015" "t2017" "tumemo")  

# 可选：指定 ckpt 目录；若不指定，则默认 /home/lym/VLM-MSA/ckpt/${dataset}
# export SOFT_DIR="/home/lym/VLM-MSA/ckpt/mvsa-s"

# 让空通配不报错
shopt -s nullglob

for k in 1 2; do
  # topk=1 时两个模式等价，只跑 perclass；topk=2 时跑 perclass 和 balanced
  if [ "$k" -eq 1 ]; then
    MODES=("perclass")
  else
    MODES=("perclass" "balanced")
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

      # --------- 关键修改：遍历目录下所有 soft prompt ckpt ----------
      # 目录优先级：SOFT_DIR（若设置） -> /home/lym/VLM-MSA/ckpt/${dataset}
      CKPT_DIR="${SOFT_DIR:-/home/lym/VLM-MSA/ckpt/${dataset}}"
      if [[ ! -d "${CKPT_DIR}" ]]; then
        echo "[WARN] ckpt dir not found: ${CKPT_DIR} (skip dataset ${dataset})"
        continue
      fi

      # 支持按照修改时间排序；你也可以去掉 sort -V/按名字排序
      mapfile -t CKPTS < <(find "${CKPT_DIR}" -maxdepth 1 -type f -name "*.pt" -printf "%T@ %p\n" | sort -n | awk '{print $2}')
      if (( ${#CKPTS[@]} == 0 )); then
        echo "[WARN] no *.pt under ${CKPT_DIR}"
        continue
      fi

      echo "[INFO] found ${#CKPTS[@]} soft prompts in ${CKPT_DIR}"
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
