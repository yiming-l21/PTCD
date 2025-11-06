# -*- coding: utf-8 -*-
from typing import List, Tuple, Optional
from transformers import PreTrainedTokenizerBase, PreTrainedModel
import torch
import torch.nn as nn
import os

def init_soft_tokens(tokenizer: PreTrainedTokenizerBase,
                     model: PreTrainedModel,
                     n_tokens: int) -> Tuple[List[str], List[int]]:
    if n_tokens <= 0:
        return [], []
    soft_tokens = [f"<soft{i}>" for i in range(n_tokens)]
    added_vocab = tokenizer.get_added_vocab()
    to_add = [t for t in soft_tokens if t not in added_vocab]
    if to_add:
        tokenizer.add_special_tokens({"additional_special_tokens": to_add})
        model.resize_token_embeddings(len(tokenizer))
    ids = tokenizer.convert_tokens_to_ids(soft_tokens)
    return soft_tokens, ids
def init_visual_soft_tokens(model: PreTrainedModel,
                           n_tokens: int,
                           device: torch.device) -> Tuple[nn.Parameter, int]:
    """稳定版视觉软提示初始化：不依赖模型接口，直接高斯分布初始化（避免调用中间层报错）"""
    if n_tokens <= 0:
        return None, 0
    if not (hasattr(model, "model") and hasattr(model.model, "visual")):
        raise RuntimeError("未找到Qwen2.5-VL的视觉编码器模块（正确路径应为 model.model.visual）")

    hidden_size = model.config.vision_config.hidden_size
    print(f"[SUCCESS] 视觉软提示配置：{n_tokens}个Token × {hidden_size}维度")
    visual_sp_emb = torch.randn(
        n_tokens, hidden_size, 
        dtype=model.dtype, 
        device=device
    ) * 0.02

    visual_sp_param = nn.Parameter(visual_sp_emb)
    print(f"[visual-sp] 初始化完成！可训练参数数量：{n_tokens * hidden_size}")

    return visual_sp_param, hidden_size

def soft_string(soft_tokens: List[str]) -> str:
    return "".join(soft_tokens)

# GPU内存工具
def _fmt_bytes(n):
    if n is None: return "n/a"
    units = ["B","KB","MB","GB","TB"]
    i = 0
    n = float(n)
    while n >= 1024 and i < len(units)-1:
        n /= 1024.0; i += 1
    return f"{n:.2f}{units[i]}"

def gpu_mem_snapshot(device=None, prefix=""):
    if not torch.cuda.is_available():
        print(prefix + "[GPU] CUDA not available"); return
    dev = torch.cuda.current_device() if device is None else device
    torch.cuda.synchronize(dev)
    alloc = torch.cuda.memory_allocated(dev)
    reserv = torch.cuda.memory_reserved(dev)
    max_alloc = torch.cuda.max_memory_allocated(dev)
    max_resv = torch.cuda.max_memory_reserved(dev)
    free, total = torch.cuda.mem_get_info(dev)
    print(prefix + f"[GPU:{dev}] "
          f"alloc={_fmt_bytes(alloc)}  reserv={_fmt_bytes(reserv)}  "
          f"max_alloc={_fmt_bytes(max_alloc)}  max_resv={_fmt_bytes(max_resv)}  "
          f"free={_fmt_bytes(free)} / total={_fmt_bytes(total)}")

def gpu_mem_reset_peak(device=None):
    if torch.cuda.is_available():
        dev = torch.cuda.current_device() if device is None else device
        torch.cuda.reset_peak_memory_stats(dev)

def gpu_mem_summary(device=None):
    if torch.cuda.is_available():
        dev = torch.cuda.current_device() if device is None else device
        print(torch.cuda.memory_summary(device=dev, abbreviated=True))