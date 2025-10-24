#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=6
export SP_N_TOKENS=8
: "${CUDA_VISIBLE_DEVICES:=}" 
python src/prompt_tuning/train_soft_prompt.py \
  --model "/home/lym/models/qwen2.5_vl" \
  --data_dir "datasets/mvsa-s" \
  --img_dir "datasets/mvsa-s/imgs" \
  --tsv "test.tsv" \
  --train_tsv train_few1.tsv \
  --dev_tsv dev_few1.tsv \
  --dtype bf16 \
  --min_pixels 224 --max_pixels 1024 \
  --batch_size 4 \
  --sp_mode generic \
  --sp_lr 5e-3 --sp_steps 1000 --sp_ckpt ./prompt_ckpt.pt
