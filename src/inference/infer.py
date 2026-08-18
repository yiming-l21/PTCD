# -*- coding: utf-8 -*-
# src/infer_common.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List, Any
import contextlib
import torch

# 与你的 eval / run 共用的允许键集合
GEN_ALLOW = {
    "input_ids", "attention_mask",
    "pixel_values", "pixel_attention_mask",
    "pixel_values_videos", "pixel_values_videos_mask",
    "image_grid_thw", "video_grid_thw", "image_sizes",
    "input_features", "encoder_outputs", "image_embeddings",
}

@contextlib.contextmanager
def prompt_eval_guards(row_replacer: Optional[Any], vp_core: Optional[Any]):
    """
    评估/推理前：临时关闭文本软提示与视觉前缀的 dropout；退出时恢复。
    - row_replacer: SoftPromptLearner._row_replacer 或 None
    - vp_core: VisualPrompt.core 或 None
    """
    old_txt_dp = None
    vp_prev_training, vp_prev_dp = None, None

    try:
        if row_replacer is not None:
            old_txt_dp = float(getattr(row_replacer, "dropout_p", 0.0))
            row_replacer.dropout_p = 0.0

        if vp_core is not None:
            vp_prev_training = bool(vp_core.training)
            vp_prev_dp = float(getattr(vp_core, "dropout_p", 0.0))
            vp_core.dropout_p = 0.0
            vp_core.eval()

        yield
    finally:
        if row_replacer is not None and (old_txt_dp is not None):
            row_replacer.dropout_p = old_txt_dp

        if vp_core is not None:
            vp_core.dropout_p = vp_prev_dp if vp_prev_dp is not None else vp_core.dropout_p
            vp_core.train(vp_prev_training is True)

def set_strict_greedy_generation(model):
    """
    统一生成超参（与推理侧一致）：greedy，无采样，无beam。
    """
    try:
        gc = model.generation_config
        gc.do_sample = False
        gc.temperature = None
        for k in ("top_p", "top_k", "typical_p"):
            if hasattr(gc, k):
                setattr(gc, k, None)
    except Exception:
        pass

def generate_scores_argmax(
    model,
    tokenizer,
    hf_inputs: Dict[str, torch.Tensor],
    *,
    max_new_tokens: int,
    decode_clean: bool = False,
) -> Tuple[str, torch.Tensor]:
    """
    核心：统一的 generate + scores-argmax 解码。
    返回：(decoded_text, gen_ids[1D])
    """
    set_strict_greedy_generation(model)
    out = model.generate(
        **hf_inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        num_beams=1,
        return_dict_in_generate=True,
        output_scores=True,
    )

    step_tokens = []
    for step_scores in out.scores:                      # 每步 [B, vocab]
        probs = torch.softmax(step_scores.float(), dim=-1)
        tok = probs.argmax(dim=-1)                      # [B]
        step_tokens.append(tok)
    if step_tokens:
        gen_ids = torch.stack(step_tokens, dim=1)[0]    # [new_len]
    else:
        gen_ids = torch.empty((0,), dtype=torch.long, device=next(model.parameters()).device)

    text_out = tokenizer.decode(
        gen_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=bool(decode_clean),
    )
    return text_out, gen_ids

def prepare_inputs_from_messages(
    processor,
    messages: List[Dict[str, Any]],
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """
    供推理端使用：从 messages 构建 hf_inputs（不做过滤；由调用侧再过滤到 GEN_ALLOW）。
    """
    from qwen_vl_utils import process_vision_info
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    images, videos = process_vision_info(messages)
    inputs = processor(text=[text], images=images, videos=videos, padding=True, return_tensors="pt")
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in inputs.items()}

# src/infer_common.py
from typing import Sequence

def _slice_like_batch(x, i: int, B: int):
    """
    将 batch-like 数据抽取第 i 条：
    - 张量: 按维度0切片，保留 batch 维度 => x[i:i+1]
    - list/tuple: 取第 i 个，并用 [item] 包一层，保持 batch 尺度一致
    - 其他: 原样返回（通常不是 batch 维；或已是单条）
    """
    if torch.is_tensor(x):
        if x.dim() >= 1 and x.size(0) == B:
            return x[i:i+1]
        return x
    if isinstance(x, (list, tuple)) and len(x) == B:
        item = x[i]
        # Qwen2.5-VL 期望这些字段仍是“一个 batch”的结构，所以包一层列表
        return [item]
    return x

def filter_to_gen_allow(
    batch_like: Dict[str, Any],
    *,
    take_index: Optional[int] = None,
    device: Optional[torch.device] = None,
    compute_dtype: Optional[torch.dtype] = None,
) -> Dict[str, Any]:
    """
    - 仅保留生成所需键 (GEN_ALLOW)
    - 如果给了 take_index，则对张量和 list/tuple 都抽出第 i 条，并保持 batch 维度
    - 将 pixel_values / pixel_values_videos 转到 device，并按 compute_dtype 对齐
    """
    s = {k: v for k, v in batch_like.items() if k in GEN_ALLOW}

    if take_index is not None:
        # 推断 batch 大小（优先从张量推断）
        B = None
        for v in s.values():
            if torch.is_tensor(v) and v.dim() >= 1:
                B = int(v.size(0))
                break
        if B is None:
            # 再从 list/tuple 推断
            for v in s.values():
                if isinstance(v, (list, tuple)):
                    B = len(v)
                    break
        if B is None:
            B = 1

        for k, v in list(s.items()):
            s[k] = _slice_like_batch(v, take_index, B)

    # 设备与 dtype 对齐
    for k, v in list(s.items()):
        if torch.is_tensor(v):
            if device is not None:
                # pixel tensors 需要到指定 device
                s[k] = v.to(device)
            # 视觉张量按模型 dtype 对齐
            if k in ("pixel_values", "pixel_values_videos"):
                if (compute_dtype is not None) and v.dtype != compute_dtype and torch.is_floating_point(v):
                    s[k] = s[k].to(dtype=compute_dtype)
    return s



# utils/trace.py
import json, os, hashlib, time
from dataclasses import asdict, dataclass

def _tensor_hash(t):
    try:
        return hashlib.sha256(t.detach().float().cpu().numpy().tobytes()).hexdigest()[:16]
    except Exception:
        return None

def _stats(t):
    try:
        tt = t.detach().float()
        return {
            "shape": list(tt.shape),
            "dtype": str(t.dtype),
            "device": str(t.device),
            "mean": float(tt.mean().item()),
            "std": float(tt.std().item()),
            "norm": float(tt.norm().item()),
            "sha": _tensor_hash(tt),
        }
    except Exception:
        return {"shape": None, "dtype": None, "device": None, "mean": None, "std": None, "norm": None, "sha": None}

def save_trace(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

@dataclass
class GenCfg:
    max_new_tokens: int
    temperature: float
    top_p: float
    do_sample: bool
    num_beams: int

def now_ms():
    return int(time.time() * 1000)
