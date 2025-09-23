#!/usr/bin/env bash
set -euo pipefail

# 定义要处理的数据集列表
DATASETS=("tumemo" "masad" "mvsa-s" "mvsa-m" "t2017" "t2015")

# python -m venv .venv && source .venv/bin/activate

# pip install -U pip
# pip install -r requirements.txt

for dataset in "${DATASETS[@]}"; do
    echo "开始处理数据集: $dataset"
    
    python model.py \
        --data_dir "datasets/$dataset" \
        --img_dir "datasets/$dataset/imgs" \
        --tsv "test.tsv" \
        --labels "positive,neutral,negative" \
        --model "/export/home/liuyiming54/models/qwen2.5_vl" \
        --dtype "bf16" \
        --attn_impl "sdpa" \
        --lang "en" \
        --max_new_tokens 16 \
        --temperature 1.0
    
    echo "dataset $dataset processed."
    echo "----------------------------------------"
done

echo "All datasets processed."
