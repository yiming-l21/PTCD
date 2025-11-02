#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=4
############################################
# 可改区：设备/soft tokens/并行卡数
############################################
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"   # 单卡默认用第2号卡；多卡时可设成 "0,1,2,3"
export SP_N_TOKENS="${SP_N_TOKENS:-8}"                     # 软提示token个数
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 如果要多卡 DDP，把 NPROC 设成 >1，比如 4；单卡就别设或设成1
NPROC="${NPROC:-1}"

############################################
MODEL="/home/lym/models/qwen2.5_vl"
DATASETS="t2015"   # 可改成 t2015 或 t2017
DATA_DIR="datasets/${DATASETS}"
IMG_DIR="datasets/${DATASETS}/imgs"

# 你的原始参数基本保持不变，只是新增了 --sp_dropout
ARGS=(
  --model         "$MODEL"
  --data_dir      "$DATA_DIR"
  --img_dir       "$IMG_DIR"
  --tsv           "test.tsv"
  --train_tsv     "train_few1.tsv"
  --dev_tsv       "dev_few1.tsv"
  --dtype         bf16
  --min_pixels    50176
  --max_pixels    1048576
  --batch_size    4
  --sp_mode       generic
  --sp_lr         8e-4
  --sp_steps      1000
  --sp_ckpt      /home/lym/VLM-MSA/ckpt/${DATASETS}/
  --sp_dropout    0.20       # ★ 新增：训练时的 prompt-dropout 概率（评估自动关闭）
)
echo "[INFO] Launching single GPU on device(s): ${CUDA_VISIBLE_DEVICES}"
python src/prompt_tuning/train_soft_prompt.py "${ARGS[@]}"
