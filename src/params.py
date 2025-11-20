import argparse
from pathlib import Path

def build_args():
    p = argparse.ArgumentParser()
    # paths
    p.add_argument('--data_dir', type=str, default='datasets/t2015', help='包含 TSV 的数据目录')
    p.add_argument('--img_dir', type=str, default=None, help='图片目录；为空则文本-only')
    p.add_argument('--tsv', type=str, default='test.tsv', help='要评测的 TSV 文件名')
    p.add_argument('--train_tsv', type=str, default='train_few1.tsv', help='要训练的 TSV 文件名')
    p.add_argument('--dev_tsv', type=str, default='dev_few1.tsv', help='要验证的 TSV 文件名')

    # model
    p.add_argument('--model', type=str, default='Qwen/Qwen2.5-VL-7B-Instruct')
    p.add_argument('--dtype', type=str, default='auto', choices=['auto','bf16','fp16','fp32'])
    p.add_argument('--device_map', type=str, default='cuda:2', help='设备映射（如 cuda:4 或 auto）')
    p.add_argument('--attn_impl', type=str, default='sdpa', choices=['sdpa','flash_attention_2'])
    p.add_argument('--min_pixels', type=int, default=224*224, help='图像最小像素数')
    p.add_argument('--max_pixels', type=int, default=1440*1440, help='图像最大像素数')

    # inference
    p.add_argument('--labels', type=str, default='positive,neutral,negative',
                   help='分类标签集合，英文逗号分隔，例如: "positive,neutral,negative" 或 "positive,negative"')
    p.add_argument('--lang', type=str, default='zh', choices=['zh','en'])
    p.add_argument('--max_new_tokens', type=int, default=16, help='生成最大新token数')
    p.add_argument('--temperature', type=float, default=0.0, help='生成温度（0为确定性）')
    p.add_argument('--top_p', type=float, default=1.0, help='top-p采样参数')
    p.add_argument('--batch_size', type=int, default=16, help='训练/推理批次大小')
    p.add_argument('--seed', type=int, default=34, help='随机种子（保证可复现）')
    p.add_argument('--no_img', action='store_true', help='强制不使用图像（即使提供 img_dir）')
    p.add_argument('--distributed', action='store_true', help='Enable multi-GPU DDP')
    p.add_argument('--dump_raw', action='store_true', help='Dump raw generations to raw_generations.jsonl')

    # ===== 模板配置 =====
    p.add_argument('--template_id', type=int, default=2, help='数据集模板ID（与dataset_info.py对应）')

    # ===== soft-prompt 核心配置 =====
    p.add_argument('--sp_n_tokens', type=int, default=16, help='可学习文本软提示token个数')
    p.add_argument('--sp_mode', type=str, default='combined', choices=['generic','class_specific','combined'])
    p.add_argument('--sp_lambda', type=float, default=0.5, help='combined 模式下的 λ')
    p.add_argument('--sp_per_tpl', action='store_true', help='是否为每个模板维护独立 prompt 头')
    p.add_argument('--sp_vtokens', type=int, default=8, help='视觉软提示token个数（>0 则启用视觉前缀）')
    p.add_argument('--visual_sp_dropout', type=float, default=0.1, help='视觉软提示dropout概率（训练时生效）')

    # ===== sp trainer 训练配置 =====
    p.add_argument('--sp_lr', type=float, default=3e-4, help='文本软提示学习率（视觉自动为1.5倍）')
    p.add_argument('--sp_steps', type=int, default=1000, help='最大训练步数')
    p.add_argument('--sp_accum', type=int, default=1, help='梯度累积步数')
    p.add_argument('--sp_warmup', type=int, default=200, help='学习率热身步数（总步数的10%-20%）')
    p.add_argument('--sp_ckpt', type=str, default='prompt_ckpt.pt', help='最终checkpoint保存路径')
    p.add_argument('--sp_best', type=str, default='prompt_ckpt.best.pt', help='最优checkpoint保存路径')
    p.add_argument('--sp_dropout', type=float, default=0.2, help='文本软提示dropout概率')
    p.add_argument('--log_every', type=int, default=100, help='每多少步打印一次训练日志')
    p.add_argument('--eval_every', type=int, default=500, help='每多少步进行一次验证评估')
    p.add_argument('--save_every_step', type=int, default=100, help='每多少步保存一次中间checkpoint')
    p.add_argument('--step_ckpt_dir', type=str, default=None, help='中间checkpoint保存目录（不保存则为 None）')

    return p.parse_args()

def resolve_paths(args):
    """解析路径为Path对象，确保目录存在"""
    args.data_dir = Path(args.data_dir)
    args.img_dir = None if args.img_dir is None else Path(args.img_dir)
    
    # 确保输出checkpoint目录存在
    if args.sp_ckpt:
        ckpt_dir = Path(args.sp_ckpt).parent
        ckpt_dir.mkdir(parents=True, exist_ok=True)
    if args.sp_best:
        best_ckpt_dir = Path(args.sp_best).parent
        best_ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    # 数据集路径
    args.tsv_path = args.data_dir / args.tsv
    args.train_tsv_path = args.data_dir / args.train_tsv
    args.dev_tsv_path = args.data_dir / args.dev_tsv
    
    return args