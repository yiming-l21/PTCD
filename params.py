import argparse
from pathlib import Path




def build_args():
    p = argparse.ArgumentParser()
    # paths
    p.add_argument('--data_dir', type=str, default='datasets/t2015', help='包含 TSV 的数据目录')
    p.add_argument('--img_dir', type=str, default=None, help='图片目录；为空则文本-only')
    p.add_argument('--tsv', type=str, default='test.tsv', help='要评测的 TSV 文件名')


    # model
    p.add_argument('--model', type=str, default='Qwen/Qwen2.5-VL-7B-Instruct')
    p.add_argument('--dtype', type=str, default='auto', choices=['auto','bf16','fp16','fp32'])
    p.add_argument('--device_map', type=str, default='auto')
    p.add_argument('--attn_impl', type=str, default='sdpa', choices=['sdpa','flash_attention_2'])
    p.add_argument('--min_pixels', type=int, default=224*224)
    p.add_argument('--max_pixels', type=int, default=1024*1024)


    # inference
    p.add_argument('--labels', type=str, default='positive,neutral,negative',
    help='分类标签集合，英文逗号分隔，例如: "positive,neutral,negative" 或 "positive,negative"')
    p.add_argument('--lang', type=str, default='zh', choices=['zh','en'])
    p.add_argument('--max_new_tokens', type=int, default=16)
    p.add_argument('--temperature', type=float, default=0.0)
    p.add_argument('--top_p', type=float, default=1.0)
    p.add_argument('--batch_size', type=int, default=1, help='最小基线默认逐条推理，安全起见设 1')
    p.add_argument('--seed', type=int, default=34)
    p.add_argument('--no_img', action='store_true', help='强制不使用图像（即使提供 img_dir）')

    return p.parse_args()




def resolve_paths(args):
    args.data_dir = Path(args.data_dir)
    args.img_dir = None if args.img_dir is None else Path(args.img_dir)
    args.tsv_path = args.data_dir / args.tsv
    return args