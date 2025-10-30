#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=2,4,5,6,7
# 数据集列表
DATASETS=("mvsa-m" "masad" "t2015" "t2017" "tumemo")
export SP_N_TOKENS=8

train_script="src/prompt_tuning/train_soft_prompt.py"

# --------------------------
# 自动获取可用GPU列表
# --------------------------
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
  # 若用户已指定CUDA_VISIBLE_DEVICES，优先使用其作为可用GPU（逗号分隔）
  IFS=',' read -ra AVAILABLE_GPUS <<< "$CUDA_VISIBLE_DEVICES"
else
  # 否则从nvidia-smi获取所有可用GPU的ID
  AVAILABLE_GPUS=($(nvidia-smi --list-gpus | awk '{print $2}' | tr -d ':'))
fi

NUM_GPUS=${#AVAILABLE_GPUS[@]}
NUM_DATASETS=${#DATASETS[@]}

# 检查是否有可用GPU
if [ $NUM_GPUS -eq 0 ]; then
  echo "错误：未检测到可用GPU，请检查环境或设置CUDA_VISIBLE_DEVICES"
  exit 1
fi

echo "检测到可用GPU：${AVAILABLE_GPUS[*]}（共$NUM_GPUS张）"
echo "待训练数据集：${DATASETS[*]}（共$NUM_DATASETS个）"


# --------------------------
# 自适应分配任务（分批训练）
# --------------------------
current_idx=0  # 当前处理的数据集索引
while [ $current_idx -lt $NUM_DATASETS ]; do
  # 本轮批次的GPU索引（从可用GPU中取，避免越界）
  batch_gpus=()
  for ((i=0; i<NUM_GPUS && current_idx+i < NUM_DATASETS; i++)); do
    batch_gpus+=("${AVAILABLE_GPUS[$i]}")
  done
  BATCH_SIZE=${#batch_gpus[@]}  # 本批次任务数

  echo "=== 启动第$(( (current_idx / NUM_GPUS) + 1 ))批训练（共$BATCH_SIZE个任务） ==="

  # 启动本批次的所有任务（后台运行）
  pids=()  # 存储本批次进程ID，用于等待
  for ((i=0; i<BATCH_SIZE; i++)); do
    dataset="${DATASETS[$current_idx]}"
    gpu="${batch_gpus[$i]}"
    log_file="${dataset}_train.log"

    echo "启动任务：数据集=$dataset，GPU=$gpu，日志文件=$log_file"
    # 为当前任务指定GPU，后台运行并记录进程ID
    CUDA_VISIBLE_DEVICES="$gpu" python "$train_script" \
      --model "/home/lym/models/qwen2.5_vl" \
      --data_dir "datasets/${dataset}" \
      --img_dir "datasets/${dataset}/imgs" \
      --tsv "test.tsv" \
      --train_tsv train_few1.tsv \
      --dev_tsv dev_few1.tsv \
      --dtype bf16 \
      --min_pixels 224 --max_pixels 1024 \
      --batch_size 4 \
      --sp_mode generic \
      --sp_lr 8e-4 --sp_steps 1000 --sp_ckpt "./ckpt/${dataset}_prompt_ckpt.pt" --sp_best "./ckpt/${dataset}_prompt_ckpt.best.pt" \
      > "$log_file" 2>&1 &

    pids+=("$!")  # 记录后台进程ID
    current_idx=$((current_idx + 1))  # 移动到下一个数据集
  done

  # 等待本批次所有任务完成后，再启动下一批
  echo "等待本批次任务完成（进程ID：${pids[*]}）..."
  wait "${pids[@]}"
  echo "=== 第$(( (current_idx / NUM_GPUS) ))批训练完成 ==="
done

echo "所有数据集训练完成！"