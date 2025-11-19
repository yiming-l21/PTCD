#!/usr/bin/env bash
set -euo pipefail

############################################
# 可改区：核心配置（按需调整）
############################################
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,7}"
export SP_N_TOKENS="${SP_N_TOKENS:-8}"
export VISUAL_SP_N_TOKENS="${VISUAL_SP_N_TOKENS:-16}"
export TEXT_PROMPT_ONLY="${TEXT_PROMPT_ONLY:-0}"
export VISUAL_PROMPT_ONLY="${VISUAL_PROMPT_ONLY:-1}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
NPROC="${NPROC:-1}"
export SAVE_EVERY_STEP="${SAVE_EVERY_STEP:-100}"

MODEL="/home/lym/models/qwen2.5_vl"

# 数据集名
DATASETS="mvsa-s"
DATA_DIR="datasets/${DATASETS}"
IMG_DIR="datasets/${DATASETS}/imgs"

# RAG 语料（可被 JSON 覆盖）
export TRAIN_JSONL="${TRAIN_JSONL:-/home/lym/VLM-MSA/datasets/${DATASETS}/train_few1.json}"

# 输出目录
CKPT_DIR="/home/lym/VLM-MSA/ckpt2/${DATASETS}"
STEP_CKPT_DIR="${CKPT_DIR}/step_ckpts"
LOG_FILE="${CKPT_DIR}/train.log"

# RAG 默认开关（可被 JSON 覆盖）
export USE_DEMO="${USE_DEMO:-0}"
export DEMO_TOPK="${DEMO_TOPK:-1}"
export DEMO_MODE="${DEMO_MODE:-perclass}"

############################################
# 按数据集读取 JSON 并覆盖超参 (configs/<dataset>.json)
############################################
CFG_DIR="configs"
CFG_FILE="${CFG_DIR}/${DATASETS}.json"

_read_json() {
  # $1: key
  jq -r "(.${1} // empty) // \"\"" "${CFG_FILE}"
}

if command -v jq >/dev/null 2>&1 && [[ -f "${CFG_FILE}" ]]; then
  echo "[INFO] 发现数据集配置：${CFG_FILE}，将覆盖默认参数"

  # 资源/并行
  cfg_cuda=$(_read_json cuda)           # "2" 或 "2,3"
  cfg_nproc=$(_read_json nproc)         # 1 / 2 / 4

  # 超参
  cfg_sp_lr=$(_read_json sp_lr)
  cfg_sp_steps=$(_read_json sp_steps)
  cfg_sp_accum=$(_read_json sp_accum)
  cfg_sp_warmup=$(_read_json sp_warmup)
  cfg_batch_size=$(_read_json batch_size)
  cfg_template_id=$(_read_json template_id)
  cfg_sp_dropout=$(_read_json sp_dropout)
  cfg_vis_sp_dropout=$(_read_json visual_sp_dropout)
  cfg_sp_ntokens=$(_read_json sp_n_tokens)
  cfg_vis_ntokens=$(_read_json visual_sp_n_tokens)

  # RAG
  cfg_use_demo=$(_read_json use_demo)
  cfg_demo_topk=$(_read_json demo_topk)
  cfg_demo_mode=$(_read_json demo_mode)
  cfg_train_jsonl=$(_read_json train_jsonl)

  # 数据切分
  cfg_train_tsv=$(_read_json train_tsv)
  cfg_dev_tsv=$(_read_json dev_tsv)
  cfg_test_tsv=$(_read_json test_tsv)
  cfg_target_mode=$(_read_json target_mode)

  # 应用覆盖（非空才覆盖）
  [[ -n "${cfg_cuda}" ]]           && export CUDA_VISIBLE_DEVICES="${cfg_cuda}"
  [[ -n "${cfg_nproc}" ]]          && NPROC="${cfg_nproc}"

  [[ -n "${cfg_sp_lr}" ]]          && export _OVR_SP_LR="${cfg_sp_lr}"
  [[ -n "${cfg_sp_steps}" ]]       && export _OVR_SP_STEPS="${cfg_sp_steps}"
  [[ -n "${cfg_sp_accum}" ]]       && export _OVR_SP_ACCUM="${cfg_sp_accum}"
  [[ -n "${cfg_sp_warmup}" ]]      && export _OVR_SP_WARMUP="${cfg_sp_warmup}"
  [[ -n "${cfg_batch_size}" ]]     && export _OVR_BATCH_SIZE="${cfg_batch_size}"
  [[ -n "${cfg_template_id}" ]]    && export _OVR_TEMPLATE_ID="${cfg_template_id}"
  [[ -n "${cfg_sp_dropout}" ]]     && export _OVR_SP_DROPOUT="${cfg_sp_dropout}"
  [[ -n "${cfg_vis_sp_dropout}" ]] && export _OVR_VIS_DROPOUT="${cfg_vis_sp_dropout}"
  [[ -n "${cfg_sp_ntokens}" ]]     && export SP_N_TOKENS="${cfg_sp_ntokens}"
  [[ -n "${cfg_vis_ntokens}" ]]    && export VISUAL_SP_N_TOKENS="${cfg_vis_ntokens}"

  [[ -n "${cfg_use_demo}" ]]       && export USE_DEMO="${cfg_use_demo}"
  [[ -n "${cfg_demo_topk}" ]]      && export DEMO_TOPK="${cfg_demo_topk}"
  [[ -n "${cfg_demo_mode}" ]]      && export DEMO_MODE="${cfg_demo_mode}"
  [[ -n "${cfg_train_jsonl}" ]]    && export TRAIN_JSONL="${cfg_train_jsonl}"

  [[ -n "${cfg_train_tsv}" ]]      && export _OVR_TRAIN_TSV="${cfg_train_tsv}"
  [[ -n "${cfg_dev_tsv}" ]]        && export _OVR_DEV_TSV="${cfg_dev_tsv}"
  [[ -n "${cfg_test_tsv}" ]]       && export _OVR_TEST_TSV="${cfg_test_tsv}"
  [[ -n "${cfg_target_mode}" ]]    && export TARGET_MODE="${cfg_target_mode}"
else
  [[ ! -f "${CFG_FILE}" ]] && echo "[INFO] 未找到 ${CFG_FILE}，使用脚本默认参数"
  [[ ! $(command -v jq) ]] && echo "[WARN] 未安装 jq，无法读取 JSON；使用脚本默认参数"
fi

############################################
# 训练参数（默认 + JSON 覆盖）
############################################
SP_LR_VAL="${_OVR_SP_LR:-3e-4}"
SP_STEPS_VAL="${_OVR_SP_STEPS:-1500}"
SP_ACCUM_VAL="${_OVR_SP_ACCUM:-1}"
SP_WARMUP_VAL="${_OVR_SP_WARMUP:-100}"
BATCH_VAL="${_OVR_BATCH_SIZE:-4}"
TPL_ID_VAL="${_OVR_TEMPLATE_ID:-2}"
SP_DROPOUT_VAL="${_OVR_SP_DROPOUT:-0.20}"
VIS_DROPOUT_VAL="${_OVR_VIS_DROPOUT:-0.10}"

TRAIN_TSV_VAL="${_OVR_TRAIN_TSV:-train_few1.tsv}"
DEV_TSV_VAL="${_OVR_DEV_TSV:-test.tsv}"
TEST_TSV_VAL="${_OVR_TEST_TSV:-test.tsv}"

ARGS=(
  --model         "$MODEL"
  --data_dir      "$DATA_DIR"
  --img_dir       "$IMG_DIR"
  --tsv           "$TEST_TSV_VAL"
  --train_tsv     "$TRAIN_TSV_VAL"
  --dev_tsv       "$DEV_TSV_VAL"
  --dtype         bf16
  --min_pixels    50176
  --max_pixels    1048576
  --batch_size    "$BATCH_VAL"
  --sp_mode       generic
  --sp_lr         "$SP_LR_VAL"
  --sp_steps      "$SP_STEPS_VAL"
  --sp_accum      "$SP_ACCUM_VAL"
  --sp_warmup     "$SP_WARMUP_VAL"
  --sp_ckpt       "$CKPT_DIR"
  --step_ckpt_dir "$STEP_CKPT_DIR"
  --save_every_step "$SAVE_EVERY_STEP"
  --sp_dropout    "$SP_DROPOUT_VAL"
  --visual_sp_dropout "$VIS_DROPOUT_VAL"
  --seed          34
  --template_id   "$TPL_ID_VAL"
  --eval_every    500
  --log_every     500
)

############################################
# 预处理：创建输出目录和日志文件
############################################
mkdir -p "$CKPT_DIR" "$STEP_CKPT_DIR"
echo "[INFO] 训练日志将保存到：${LOG_FILE}"
echo "[INFO] 主Checkpoint将保存到：${CKPT_DIR}"
echo "[INFO] 按步Checkpoint将保存到：${STEP_CKPT_DIR}"
echo "[INFO] 启动时间：$(date)"
echo "[INFO] 环境变量："
echo "  CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "  NPROC: ${NPROC}"
echo "  SP_N_TOKENS: ${SP_N_TOKENS}"
echo "  VISUAL_SP_N_TOKENS: ${VISUAL_SP_N_TOKENS}"
echo "  USE_DEMO: ${USE_DEMO} | DEMO_TOPK: ${DEMO_TOPK} | DEMO_MODE: ${DEMO_MODE}"
echo "  TRAIN_JSONL: ${TRAIN_JSONL}"
echo "[INFO] 训练参数："
printf '  %s\n' "${ARGS[@]}"

{
  echo "----------------------------------------"
  echo "[INFO] 启动时间：$(date)"
  echo "[INFO] 环境变量："
  echo "  CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
  echo "  NPROC: ${NPROC}"
  echo "  SP_N_TOKENS: ${SP_N_TOKENS}"
  echo "  VISUAL_SP_N_TOKENS: ${VISUAL_SP_N_TOKENS}"
  echo "  USE_DEMO: ${USE_DEMO} | DEMO_TOPK: ${DEMO_TOPK} | DEMO_MODE: ${DEMO_MODE}"
  echo "  TRAIN_JSONL: ${TRAIN_JSONL}"
  echo "[INFO] 训练参数（最终生效）："
  printf '  %s\n' "${ARGS[@]}"
  echo "----------------------------------------"
} > "$LOG_FILE"

############################################
# 启动训练（单卡/多卡兼容）
############################################
if [ "$NPROC" -eq 1 ]; then
  echo "[INFO] 启动单卡训练（设备：${CUDA_VISIBLE_DEVICES}）"
  python src/prompt_tuning/train_soft_prompt.py "${ARGS[@]}" 2>&1 | tee -a "$LOG_FILE"
else
  echo "[INFO] 启动多卡DDP训练（设备：${CUDA_VISIBLE_DEVICES}，卡数：${NPROC}）"
  torchrun --nproc_per_node="$NPROC" --master_port=29500 \
    src/prompt_tuning/train_soft_prompt.py "${ARGS[@]}" 2>&1 | tee -a "$LOG_FILE"
fi

############################################
# 训练结束后整理日志
############################################
{
  echo "----------------------------------------"
  echo "[INFO] 训练结束时间：$(date)"
  echo "[INFO] 训练日志路径：${LOG_FILE}"
  echo "[INFO] 主Checkpoint路径：${CKPT_DIR}"
  echo "[INFO] 按步Checkpoint路径：${STEP_CKPT_DIR}"
  echo "[INFO] 训练完成！"
} >> "$LOG_FILE"
