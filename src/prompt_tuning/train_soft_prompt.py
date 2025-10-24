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
from utils import set_seed, build_msgs
from prompt_tuning.prompt_learner import SoftPromptLearner, TrainCfg
from sp_utils import init_soft_tokens

def label_to_target_json(label: str) -> str:
    # 与 parse_label_from_output 对齐：训练时让模型生成 {"label": "<cand>"}
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

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(args.model, torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32)
    model.to(device).eval()
    processor = AutoProcessor.from_pretrained(args.model, min_pixels=args.min_pixels, max_pixels=args.max_pixels)

    n_soft = int(os.getenv("SP_N_TOKENS", "0"))
    soft_tokens, soft_ids = init_soft_tokens(processor.tokenizer, model, n_soft)
    print(f"[*] soft tokens: {soft_tokens}")

    use_image = (args.img_dir is not None) and (not args.no_img)
    has_aspect = getattr(
        args, "has_aspect",
        True if "fine" in str(tmpl).lower() else ("t2015" in dataset_name or "t2017" in dataset_name or "masad" in dataset_name)
    )

    # Determine template variant(s)
    prompt_variants = build_prompt_variant()

    # RAG inference
    demo = DemoProvider.from_env(args, dataset_name=dataset_name, image_base=Path(args.data_dir) / "imgs")


    coll = lambda batch: collate(batch, processor, labels, label_map, use_image, args.img_dir, has_aspect, "STRICT", device=device)
    train_loader = DataLoader(train_samples, batch_size=getattr(args, "batch_size", 1), shuffle=True, collate_fn=coll)
    dev_loader = DataLoader(dev_samples, batch_size= 1, shuffle=False, collate_fn=coll)
    tr_cfg = TrainCfg(
        lr=float(getattr(args, "sp_lr", 5e-2)),
        max_steps=int(getattr(args, "sp_steps", 3000)),
        grad_accum=int(getattr(args, "sp_accum", 1)),
        save_ckpt=str(getattr(args, "sp_ckpt", "prompt_ckpt.pt"))
    )
    learner = SoftPromptLearner(model, processor, ["STRICT"], demo, labels, tr_cfg, device)

    learner.fit(train_loader, dev_loader, target_builder=label_to_target_json)

if __name__ == "__main__":
    main()
