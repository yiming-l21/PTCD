# -*- coding: utf-8 -*-
from typing import List, Tuple
from transformers import PreTrainedTokenizerBase, PreTrainedModel

def init_soft_tokens(tokenizer: PreTrainedTokenizerBase,
                     model: PreTrainedModel,
                     n_tokens: int) -> Tuple[List[str], List[int]]:
    """
    在 tokenizer 中注册 <soft0>..<soft{n-1}>, 并 resize 模型 embedding。
    返回 (token_strs, token_ids)。
    若已注册，重复调用是幂等的。
    """
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

def soft_string(soft_tokens: List[str]) -> str:
    return "".join(soft_tokens)


import os, sys, math, time, subprocess, torch

def _fmt_bytes(n):
    if n is None: return "n/a"
    units = ["B","KB","MB","GB","TB"]
    i = 0
    n = float(n)
    while n >= 1024 and i < len(units)-1:
        n /= 1024.0; i += 1
    return f"{n:.2f}{units[i]}"

def gpu_mem_snapshot(device=None, prefix=""):
    """打印当前/保留/峰值（allocated/reserved/peaks），以及空闲-总量（mem_get_info）。"""
    if not torch.cuda.is_available():
        print(prefix + "[GPU] CUDA not available"); return
    dev = torch.cuda.current_device() if device is None else device
    torch.cuda.synchronize(dev)
    alloc = torch.cuda.memory_allocated(dev)
    reserv = torch.cuda.memory_reserved(dev)
    max_alloc = torch.cuda.max_memory_allocated(dev)
    max_resv = torch.cuda.max_memory_reserved(dev)
    free, total = torch.cuda.mem_get_info(dev)  # NVAPI: free/total on device
    print(prefix + f"[GPU:{dev}] "
          f"alloc={_fmt_bytes(alloc)}  reserv={_fmt_bytes(reserv)}  "
          f"max_alloc={_fmt_bytes(max_alloc)}  max_resv={_fmt_bytes(max_resv)}  "
          f"free={_fmt_bytes(free)} / total={_fmt_bytes(total)}")

def gpu_mem_reset_peak(device=None):
    """把峰值计数器清零（建议每个 step 或每个 epoch 调用一次）"""
    if torch.cuda.is_available():
        dev = torch.cuda.current_device() if device is None else device
        torch.cuda.reset_peak_memory_stats(dev)

def gpu_mem_summary(device=None):
    """打印更详细的 allocator 摘要（一次性，很长）"""
    if torch.cuda.is_available():
        dev = torch.cuda.current_device() if device is None else device
        print(torch.cuda.memory_summary(device=dev, abbreviated=True))

def nvidia_smi_top(n=5):
    """从 nvidia-smi 抓取进程显存排行榜，便于发现“幽灵进程”"""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
             "--format=csv,noheader,nounits"], text=True
        ).strip().splitlines()
        rows = []
        for line in out:
            pid, name, mem = [x.strip() for x in line.split(",", 2)]
            rows.append((int(pid), name, int(mem)))
        rows.sort(key=lambda x: x[2], reverse=True)
        print("[nvidia-smi] top processes by used_memory (MiB):")
        for pid, name, mem in rows[:n]:
            print(f"  pid={pid:<8} mem={mem:>6}MiB  cmd={name}")
    except Exception as e:
        print(f"[nvidia-smi] unavailable: {e}")
