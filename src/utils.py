from __future__ import annotations 
import re
import json
import random
import torch
import os
import numpy as np
from pathlib import Path
from typing import List
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from typing import List, Dict, Any, Tuple, Optional
from prompts import build_instruction
from ensemble import _majority_vote
from qwen_vl_utils import process_vision_info
from prompt_tuning.sp_utils import init_soft_tokens
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_model_and_processor(
    model_id: str,
    dtype: str,
    device: torch.device,
    attn_impl: str,
    min_pixels: int,
    max_pixels: int,
    use_fast_processor: bool = True,
):
    torch_dtype = {
        "auto": "auto",
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[dtype]
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        attn_implementation=attn_impl,
    )
    model.to(device).eval()
    print(f"[*]check processor params min_pixels:{min_pixels}, max_pixels:{max_pixels}, use_fast_processor:{use_fast_processor}", flush=True)
    processor = AutoProcessor.from_pretrained(
        model_id, min_pixels=min_pixels, max_pixels=max_pixels, use_fast=False
    )
    tok = processor.tokenizer
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model.generation_config.pad_token_id = tok.pad_token_id
    n_soft = int(os.getenv("SP_N_TOKENS", "0"))
    soft_tokens, soft_ids = init_soft_tokens(tok, model, n_soft)
    if n_soft > 0:
        print(f"[*] soft tokens registered: {soft_tokens}, soft id registerd: {soft_ids}", flush=True)
    # initialize soft tokens
    return model, processor

def _to_jsonable(x):
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, dict):
        return {k: _to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_to_jsonable(v) for v in x]
    return x

def _first_json(text: str) -> dict | None:
    m = re.search(r"\{[\s\S]*?\}", text)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None

def parse_label_from_output(raw: str, label_space: List[str], target_mode: str = "token") -> str:
    if target_mode == "token":
        if not isinstance(raw, str):
            return label_space[0]
        low_out = raw.strip().lower()

        # 精确匹配：优先检测单词边界，防止 false match（如 'positive' in 'depositive'）
        for cand in label_space:
            pat = rf"\b{re.escape(cand.lower())}\b"
            if re.search(pat, low_out):
                return cand

        # fallback: 如果都没匹配上，就用第一个类（通常是 neutral 或 negative）
        return label_space[0]
    data = _first_json(raw)
    if isinstance(data, dict):
        val = data.get("label")
        if isinstance(val, str):
            low = val.strip().lower()
            for cand in label_space:
                if low == cand.lower():
                    return cand
    low_out = raw.lower()
    for cand in label_space:
        pat = rf"\b{re.escape(cand.lower())}\b"
        if re.search(pat, low_out):
            return cand
    return label_space[0]

def build_msgs(
    *,
    instruction: str,
    user_text: str,
    use_image: bool,
    img_path: Optional[str],
    prefix_demo_msgs: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    
    n_soft = int(os.getenv("SP_N_TOKENS", "0"))
    soft_tokens = [f"<soft{i}>" for i in range(n_soft)]
    # if n_soft > 0:
    #     half = n_soft // 2
    #     pre_soft = "".join(soft_tokens[:half])
    #     post_soft = "".join(soft_tokens[half:])
    #     user_text = f"{pre_soft}{user_text}{post_soft}"
    pre_soft = "".join(soft_tokens) if n_soft > 0 else ""
    user_text = f"{pre_soft}{user_text}"
    system_msg = {"role": "system", "content": instruction}

    if use_image and img_path:
        user_msg = {
            "role": "user",
            "content": [
                {"type": "image", "image": str(img_path)},
                {"type": "text",  "text": user_text},
            ],
        }
    else:
        user_msg = {"role": "user", "content": [{"type": "text", "text": user_text}]}
    return [system_msg] + (prefix_demo_msgs or []) + [user_msg]

def infer_with_variants(
    *,
    model,
    processor,
    prompt_variants: List[str],
    label_space: List[str],
    use_image_flag: bool,
    img_path: Optional[str],
    user_text: str,
    prefix_demo_msgs: List[Dict[str, Any]],
    max_new_tokens: int,
    run_one_fn,                  
    parse_label_fn,   
    has_aspect: bool,          
) -> Tuple[str, List[str], List[Dict[str, Any]]]:
    per_tpl_preds: List[str] = []
    raw_bundle: List[Dict[str, Any]] = []
    target_mode_env = os.getenv("TARGET_MODE", "token")
    for tpl in prompt_variants:
        instruction = build_instruction(
            labels=label_space,
            use_image=use_image_flag and (img_path is not None),
            has_aspect=has_aspect,           
            template_variant=tpl,
            target_mode=target_mode_env,
        )
        msgs = build_msgs(
            instruction=instruction,
            user_text=user_text,
            use_image=use_image_flag and (img_path is not None),
            img_path=img_path,
            prefix_demo_msgs=prefix_demo_msgs,
        )
        raw = run_one_fn(model, processor, msgs, max_new_tokens=max_new_tokens, label_space=label_space)
        pred = parse_label_fn(raw, label_space, target_mode=target_mode_env)
        per_tpl_preds.append(pred)
        raw_bundle.append({"tpl": tpl, "raw": raw})

    final_pred = (
        per_tpl_preds[0]
        if len(prompt_variants) == 1
        else _majority_vote(per_tpl_preds, label_space)
    )
    return final_pred, per_tpl_preds, raw_bundle