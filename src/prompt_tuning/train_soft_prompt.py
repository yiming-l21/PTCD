# -*- coding: utf-8 -*-
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__))) 
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from dataset import MSADataset
from dataset_info import get_labels_and_template
from params import build_args, resolve_paths
from prompts import build_user_content, build_prompt_variant, build_instruction
from retrieve_demo import DemoProvider
from utils import set_seed, build_msgs, parse_label_from_output
from prompt_tuning.prompt_learner import SoftPromptLearner, TrainCfg
from sp_utils import init_soft_tokens, init_visual_soft_tokens, gpu_mem_snapshot, gpu_mem_reset_peak

def label_to_target_json(label: str) -> str:
    return f'{{"label": "{label}"}}'

def collate(samples, processor, labels, label_map, use_image, img_root, has_aspect, tpl, device="cpu"):
    batch_msgs = []
    gold = []
    for s in samples:
        img_path = (str((img_root / s.img_id)) if (use_image and s.img_id) else None)
        user_text = build_user_content(s.text_s, getattr(s, "text_a", None), has_aspect=has_aspect)
        instruction = build_instruction(labels, use_image, has_aspect, tpl)
        msgs = build_msgs(
            instruction=instruction,
            user_text=user_text,
            use_image=use_image and (img_path is not None),
            img_path=img_path,
            prefix_demo_msgs=[],
        )
        batch_msgs.append(msgs)
        gold.append(label_map[s.label])
    text_list = [processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in batch_msgs]
    images, videos = [], []
    from qwen_vl_utils import process_vision_info
    for m in batch_msgs:
        img_inp, vid_inp = process_vision_info(m)
        images.append(img_inp); videos.append(vid_inp)
    enc = processor(text=text_list, images=images, padding=True, return_tensors="pt").to(device)
    return {
        "hf_inputs": enc,
        "gold_label_str": gold
    }

def find_transformer_layers(model):
    candidate_paths = [
        "model.language_model.model.layers",
        "language_model.model.layers",
        "model.model.layers",
        "model.layers",
        "transformer.h",
    ]
    for path in candidate_paths:
        cur, ok = model, True
        for attr in path.split("."):
            if hasattr(cur, attr):
                cur = getattr(cur, attr)
            else:
                ok = False
                break
        if ok and hasattr(cur, "__len__"):
            return cur

    blocks = []
    for module in model.modules():
        if hasattr(module, "self_attn") or hasattr(module, "attention") or hasattr(module, "attn"):
            blocks.append(module)
    return blocks if blocks else None

def enable_gc_for_last_ratio(model, ratio=0.5):
    if hasattr(model, "config"):
        model.config.use_cache = False

    fn = getattr(model, "gradient_checkpointing_enable", None)
    if fn:
        try:
            fn(gradient_checkpointing_kwargs={"use_reentrant": False})
        except TypeError:
            try:
                fn(use_reentrant=False)
            except TypeError:
                fn()

    layers = find_transformer_layers(model)
    if not layers or not hasattr(layers, "__len__"):
        print("[warn] 未找到 transformer 层列表，跳过按比例设置。")
        return

    n = len(layers)
    start = int(n * (1 - float(ratio)))
    start = max(0, min(n, start))

    for i, block in enumerate(layers):
        try:
            setattr(block, "gradient_checkpointing", i >= start)
        except Exception:
            pass

    print(f"[gc] enabled on last {n - start}/{n} layers (ratio={ratio:.2f})")

def main():
    args = resolve_paths(build_args())
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset_name = Path(args.data_dir).name
    labels, label_map, tmpl = get_labels_and_template(dataset_name, getattr(args, "template_id", 2))
    train_reader = MSADataset(args, args.data_dir / args.train_tsv, dataset_name=dataset_name, label_map=label_map)
    train_samples = train_reader.read()
    dev_reader = MSADataset(args, args.data_dir / args.dev_tsv, dataset_name=dataset_name, label_map=label_map)
    dev_samples = dev_reader.read()

    # 加载模型
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model, 
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map=device
    )
    ratio4dataset={"t2017": 0.8, "t2015":0.3, "tumemo":0.7 }
    if dataset_name in ["t2017", "tumemo", "t2015"]:
        enable_gc_for_last_ratio(model, ratio=ratio4dataset[dataset_name])
    print("model dtype", model.dtype)
    gpu_mem_snapshot(prefix="[after load ] ")

    # 加载processor
    processor = AutoProcessor.from_pretrained(
        args.model, 
        min_pixels=args.min_pixels, 
        max_pixels=args.max_pixels, 
        use_fast=False
    )

    # 初始化文本软提示
    n_text_sp = int(os.getenv("SP_N_TOKENS", "16"))
    soft_tokens, soft_ids = init_soft_tokens(processor.tokenizer, model, n_text_sp)
    print(f"[*] 文本软提示: {soft_tokens} (数量: {n_text_sp})")

    # 视觉配置
    use_image = (args.img_dir is not None) and (not args.no_img)
    n_visual_sp = int(os.getenv("VISUAL_SP_N_TOKENS", "8"))
    print(f"[*] 视觉软提示数量: {n_visual_sp} (使用图像: {use_image})")

    has_aspect = getattr(
        args, "has_aspect",
        True if "fine" in str(tmpl).lower() else ("t2015" in dataset_name or "t2017" in dataset_name or "masad" in dataset_name)
    )

    # 构建DataLoader
    coll = lambda batch: collate(
        batch, processor, labels, label_map, use_image, args.img_dir, has_aspect, "STRICT", device=device
    )
    train_loader = DataLoader(
        train_samples, 
        batch_size=getattr(args, "batch_size", 1), 
        shuffle=True, 
        collate_fn=coll
    )
    dev_loader = DataLoader(
        dev_samples, 
        batch_size=1, 
        shuffle=False, 
        collate_fn=coll
    )
    gpu_mem_snapshot(prefix="[after collate] ")

    # 训练配置
    tr_cfg = TrainCfg(
        lr=float(getattr(args, "sp_lr", 1e-3)),
        max_steps=int(getattr(args, "sp_steps", 3000)),
        grad_accum=int(getattr(args, "sp_accum", 1)),
        save_ckpt=str(getattr(args, "sp_ckpt", "prompt_ckpt.pt")),
        ckpt_best=getattr(args, "sp_best", "prompt_ckpt.best.pt"),
        warmup_steps=int(getattr(args, "sp_warmup", 200)),
        step_ckpt_dir=getattr(args, "step_ckpt_dir", None),
        save_every_step=int(getattr(args, "save_every_step", 100)),
    )

    # 初始化Learner（包含视觉Prompt）
    learner = SoftPromptLearner(
        model=model,
        processor=processor,
        template_variants=["STRICT"],
        demo_provider=DemoProvider.from_env(args, dataset_name=dataset_name, image_base=Path(args.data_dir) / "imgs"),
        label_space=labels,
        train_cfg=tr_cfg,
        device=device,
        use_image=use_image,
        n_visual_sp=n_visual_sp
    )

    # 开始训练
    learner.fit(train_loader, dev_loader, target_builder=label_to_target_json)

if __name__ == "__main__":
    main()