#!/usr/bin/env bash
set -euo pipefail

############################################
# 可改区：核心配置（按需调整）
############################################
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4}"
# 文本软提示数量（默认8个，可按需调整）
export SP_N_TOKENS="${SP_N_TOKENS:-8}"
# 视觉软提示数量（默认8个，建议4-16个）
export VISUAL_SP_N_TOKENS="${VISUAL_SP_N_TOKENS:-8}"
export TEXT_PROMPT_ONLY="${TEXT_PROMPT_ONLY:-0}"  # 1=仅文本，0=不单独启用
export VISUAL_PROMPT_ONLY="${VISUAL_PROMPT_ONLY:-1}"  # 1=仅视觉，0=不单独启用
# 其他环境变量
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
NPROC="${NPROC:-1}"
export SAVE_EVERY_STEP="${SAVE_EVERY_STEP:-100}"

############################################
MODEL="/home/lym/models/qwen2.5_vl"
# 数据集配置（支持 t2015/t2017/tumemo）
DATASETS="mvsa-s"
DATA_DIR="datasets/${DATASETS}"
IMG_DIR="datasets/${DATASETS}/imgs"  # 图像文件夹路径（必须存在，否则视觉Prompt不生效）

# 输出目录：按数据集+日期命名，避免覆盖
CKPT_DIR="/home/lym/VLM-MSA/ckpt/${DATASETS}"
STEP_CKPT_DIR="${CKPT_DIR}/step_ckpts"  # 新增：按步保存的子目录
LOG_FILE="${CKPT_DIR}/train.log"

############################################
# 训练参数（保持原有参数，新增视觉相关配置）
############################################
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
  --sp_lr         8e-4  # 文本Prompt学习率（视觉Prompt自动为 8e-4 * 1.5 = 1.2e-3）
  --sp_steps      1000  # 最大训练步数
  --sp_accum      1     # 梯度累积步数（显存不足时调大，如2/4）
  --sp_warmup     200   # 学习率热身步数（默认200，建议为总步数的10%-20%）
  --sp_ckpt       "$CKPT_DIR"  # 主checkpoint保存目录（最佳模型+最终模型）
  --step_ckpt_dir "$STEP_CKPT_DIR"  # 新增：按步保存checkpoint的目录
  --save_every_step "$SAVE_EVERY_STEP"  # 新增：每隔多少步保存一次
  --sp_dropout    0.20  # 文本Prompt Dropout概率
  --visual_sp_dropout 0.10  # 新增：视觉Prompt Dropout概率（训练时生效）
  --seed          34     # 随机种子，保证可复现
  --template_id   2      # 数据集模板ID（与dataset_info.py对应）
  --eval_every    100    # 每100步评估一次
  --log_every     50     # 每50步打印一次日志
)

############################################
# 预处理：创建输出目录和日志文件
############################################
mkdir -p "$CKPT_DIR"
mkdir -p "$STEP_CKPT_DIR"  # 新增：创建按步保存的子目录
echo "[INFO] 训练日志将保存到：${LOG_FILE}"
echo "[INFO] 主Checkpoint将保存到：${CKPT_DIR}"
echo "[INFO] 按步Checkpoint将保存到：${STEP_CKPT_DIR}"  # 新增提示
echo "[INFO] 启动时间：$(date)"
echo "[INFO] 环境变量："
echo "  CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "  SP_N_TOKENS: ${SP_N_TOKENS}"
echo "  VISUAL_SP_N_TOKENS: ${VISUAL_SP_N_TOKENS}"
echo "  SAVE_EVERY_STEP: ${SAVE_EVERY_STEP}"  # 新增环境变量显示
echo "  NPROC: ${NPROC}"
echo "[INFO] 训练参数："
printf '  %s\n' "${ARGS[@]}"
echo "----------------------------------------" > "$LOG_FILE"
echo "[INFO] 启动时间：$(date)" >> "$LOG_FILE"
echo "[INFO] 环境变量：" >> "$LOG_FILE"
echo "  CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}" >> "$LOG_FILE"
echo "  SP_N_TOKENS: ${SP_N_TOKENS}" >> "$LOG_FILE"
echo "  VISUAL_SP_N_TOKENS: ${VISUAL_SP_N_TOKENS}" >> "$LOG_FILE"
echo "  SAVE_EVERY_STEP: ${SAVE_EVERY_STEP}" >> "$LOG_FILE"  # 新增环境变量日志
echo "[INFO] 训练参数：" >> "$LOG_FILE"
printf '  %s\n' "${ARGS[@]}" >> "$LOG_FILE"
echo "----------------------------------------" >> "$LOG_FILE"

############################################
# 启动训练（单卡/多卡兼容）
############################################
if [ "$NPROC" -eq 1 ]; then
  # 单卡训练（默认）
  echo "[INFO] 启动单卡训练（设备：${CUDA_VISIBLE_DEVICES}）"
  python src/prompt_tuning/train_soft_prompt.py "${ARGS[@]}" 2>&1 | tee -a "$LOG_FILE"
else
  # 多卡DDP训练（如需多卡，设 NPROC=卡数）
  echo "[INFO] 启动多卡DDP训练（设备：${CUDA_VISIBLE_DEVICES}，卡数：${NPROC}）"
  torchrun --nproc_per_node="$NPROC" --master_port=29500 \
    src/prompt_tuning/train_soft_prompt.py "${ARGS[@]}" 2>&1 | tee -a "$LOG_FILE"
fi

############################################
# 训练结束后整理日志
############################################
echo "----------------------------------------" >> "$LOG_FILE"
echo "[INFO] 训练结束时间：$(date)" >> "$LOG_FILE"
echo "[INFO] 训练日志路径：${LOG_FILE}"
echo "[INFO] 主Checkpoint路径：${CKPT_DIR}"
echo "[INFO] 按步Checkpoint路径：${STEP_CKPT_DIR}"  # 新增路径提示
echo "[INFO] 训练完成！"