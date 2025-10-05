# -*- coding: utf-8 -*-
import os
import re
import sys
import json
import time
import random
import socket
from time import monotonic
from pathlib import Path
from typing import List, Tuple

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

from params import build_args, resolve_paths
from dataset import MSADataset
from prompts import build_instruction, build_user_content
from utils import get_labels_and_template

# ---------------- Logging / Environment ----------------
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
from transformers.utils import logging as hf_logging  # noqa: E402
hf_logging.set_verbosity_error()
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")

# ---------------- Utility Functions ----------------
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

    processor = AutoProcessor.from_pretrained(
        model_id, min_pixels=min_pixels, max_pixels=max_pixels, use_fast=use_fast_processor
    )
    tok = processor.tokenizer
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model.generation_config.pad_token_id = tok.pad_token_id
    return model, processor

def _first_json(text: str) -> dict | None:
    m = re.search(r"\{[\s\S]*?\}", text)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None

def parse_label_from_output(raw: str, label_space: List[str]) -> str:
    data = _first_json(raw)
    if isinstance(data, dict):
        val = data.get("label")
        if isinstance(val, str):
            low = val.strip().lower()
            for cand in label_space:
                if low == cand.lower():
                    return cand
    # Whole-word matching to avoid partial hits like "not positive"
    low_out = raw.lower()
    for cand in label_space:
        pat = rf"\b{re.escape(cand.lower())}\b"
        if re.search(pat, low_out):
            return cand
    return "neutral" if any(c.lower() == "neutral" for c in label_space) else label_space[0]

def build_messages(instruction: str, use_image: bool, img_path: str | None, user_text: str):
    system_msg = {"role": "system", "content": instruction}
    if use_image and img_path is not None:
        user_content = [{"type": "image", "image": str(img_path)}, {"type": "text", "text": user_text}]
    else:
        user_content = [{"type": "text", "text": user_text}]
    return [system_msg, {"role": "user", "content": user_content}]

def run_one(model, processor, messages, max_new_tokens: int) -> str:
    """Single end-to-end generation (greedy decoding)."""
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
        )

    # Determine split point using attention_mask
    attn = inputs.get("attention_mask", None)
    if attn is not None:
        cut = int(attn.sum(dim=1)[0].item())
    else:
        pad_id = processor.tokenizer.pad_token_id
        cut = int((inputs.input_ids != pad_id).sum(dim=1)[0].item())

    gen_ids = out[0][cut:]
    text_out = processor.decode(gen_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

    # Truncate to first JSON block if applicable
    j = text_out.find("}")
    if j >= 0:
        text_out = text_out[: j + 1]
    return text_out

# ---------------- Progress Display ----------------
def iter_with_clean_progress(
    samples,
    rank: int,
    world_size: int,
    is_dist: bool,
    *,
    show_global: bool = False,
    log_every: int = 10,
    sync_every: int = 10,
    device: str = "cuda",
):
    """
    - If stdout is TTY: rank0 shows minimal ASCII progress bar, others stay silent.
    - If not TTY: rank0 prints progress periodically as plain text.
    - show_global=True: rank0 aggregates progress across ranks using all_reduce.
    """
    local_indices = list(range(rank, len(samples), world_size)) if is_dist else list(range(len(samples)))
    show_bar = (rank == 0)
    is_tty = sys.stdout.isatty()

    if not (show_bar and is_tty):
        total = len(samples) if (show_global and is_dist) else len(local_indices)
        done, last_print, t0 = 0, 0.0, monotonic()
        for i in local_indices:
            yield i
            done += 1
            now = monotonic()
            if rank == 0 and (done == total or done % log_every == 0 or (now - last_print) > 0.5):
                pct = 100.0 * done / max(total, 1)
                print(f"[rank {rank}] {done}/{total} ({pct:.1f}%) elapsed={now - t0:0.1f}s", flush=True)
                last_print = now
        return

    # Simple ASCII progress bar for rank0 only
    from rich.console import Console
    from rich.progress import Progress, BarColumn, TimeElapsedColumn, TimeRemainingColumn, MofNCompleteColumn, TextColumn
    console = Console(force_terminal=True, color_system=None, no_color=True, emoji=False)
    columns = [
        TextColumn(f"Rank {rank}"),
        BarColumn(bar_width=None, pulse=False),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    ]
    total = len(samples) if (show_global and is_dist) else len(local_indices)
    progress = Progress(*columns, console=console, refresh_per_second=5, transient=False)

    with progress:
        task = progress.add_task("generating", total=total)
        if show_global and is_dist:
            # Aggregate progress across all ranks
            pending = 0
            for i in local_indices:
                yield i
                pending += 1
                if pending % sync_every == 0:
                    t = torch.tensor([pending], device=(device if torch.cuda.is_available() else "cpu"), dtype=torch.int32)
                    dist.all_reduce(t, op=dist.ReduceOp.SUM)
                    progress.advance(task, int(t.item()))
                    pending = 0
            # Flush remaining
            t = torch.tensor([pending], device=(device if torch.cuda.is_available() else "cpu"), dtype=torch.int32)
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            progress.advance(task, int(t.item()))
        else:
            # Local progress for rank0
            for i in local_indices:
                yield i
                progress.advance(task, 1)

# ---------------- DDP Worker ----------------
def ddp_worker(rank: int, world_size: int, args_dict: dict):
    class _A: pass
    args = _A(); args.__dict__.update(args_dict)

    is_dist = world_size > 1
    if is_dist:
        torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank}")
        dist.init_process_group(backend="nccl", init_method="env://", rank=rank, world_size=world_size)
    else:
        torch.cuda.set_device(0 if torch.cuda.is_available() else "cpu")
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    set_seed(args.seed + (rank if is_dist else 0))

    # Data
    dataset_name = Path(args.data_dir).name
    template_id = getattr(args, "template_id", 2)
    label_space, label_map, tmpl = get_labels_and_template(dataset_name, template_id)
    tsv_path = args.data_dir / args.tsv
    reader = MSADataset(args, Path(tsv_path), dataset_name=dataset_name)
    samples = reader.read()

    # Model & Processor
    use_fast = getattr(args, "use_fast_processor", True)
    model, processor = load_model_and_processor(
        model_id=args.model, dtype=args.dtype, device=device,
        attn_impl=args.attn_impl, min_pixels=args.min_pixels, max_pixels=args.max_pixels,
        use_fast_processor=use_fast
    )

    # Prompt setup
    use_image_flag = (args.img_dir is not None) and (not args.no_img)
    has_aspect = getattr(
        args, "has_aspect",
        True if "fine" in str(tmpl).lower() else ("t2015" in dataset_name or "t2017" in dataset_name or "masad" in dataset_name)
    )
    instruction = build_instruction(labels=label_space, use_image=use_image_flag, has_aspect=has_aspect)

    local_results: List[Tuple[int, str]] = []
    local_gts: List[Tuple[int, str]] = []
    local_raws: List[Tuple[int, str]] = []

    # Main inference loop
    for i in iter_with_clean_progress(
        samples, rank, world_size, is_dist,
        show_global=False,
        log_every=10,
        sync_every=10,
        device=str(model.device)
    ):
        s = samples[i]
        img_path = None
        if use_image_flag and getattr(s, "img_id", None):
            cand = args.img_dir / s.img_id
            if cand.exists():
                img_path = cand

        user_text = build_user_content(s.text_s, getattr(s, "text_a", None), has_aspect=has_aspect)
        msgs = build_messages(
            instruction=instruction,
            use_image=use_image_flag and (img_path is not None),
            img_path=str(img_path) if img_path is not None else None,
            user_text=user_text,
        )
        raw = run_one(model, processor, msgs, max_new_tokens=args.max_new_tokens)
        pred = parse_label_from_output(raw, label_space)

        local_results.append((i, pred))
        local_gts.append((i, label_map[s.label]))
        if getattr(args, "dump_raw", False):
            local_raws.append((i, raw))

    # Collect results and evaluate
    if is_dist:
        pack = {"preds": local_results, "gts": local_gts, "raws": local_raws}
        gathered = [None for _ in range(world_size)]
        dist.all_gather_object(gathered, pack)

        if rank == 0:
            all_pred, all_gt, all_raw = [], [], []
            for p in gathered:
                all_pred.extend(p["preds"]); all_gt.extend(p["gts"]); all_raw.extend(p["raws"])
            all_pred.sort(key=lambda x: x[0]); all_gt.sort(key=lambda x: x[0]); all_raw.sort(key=lambda x: x[0])

            preds = [p for _, p in all_pred]
            gts   = [g for _, g in all_gt]
            raws  = [r for _, r in all_raw]
            _finalize_and_save(preds, gts, raws)

        dist.barrier()
        dist.destroy_process_group()
    else:
        preds = [p for _, p in sorted(local_results, key=lambda x: x[0])]
        gts   = [g for _, g in sorted(local_gts,     key=lambda x: x[0])]
        raws  = [r for _, r in sorted(local_raws,    key=lambda x: x[0])] if getattr(args, "dump_raw", False) else []
        _finalize_and_save(preds, gts, raws)

def _finalize_and_save(preds: List[str], gts: List[str], raws: List[str | Tuple[int, str]]):
    acc = accuracy_score(gts, preds)
    f1_mac = f1_score(gts, preds, average="macro")
    f1_wtd = f1_score(gts, preds, average="weighted")
    print(f"[Test] size={len(gts)} Acc={acc:.4f} Macro-F1={f1_mac:.4f} Weighted-F1={f1_wtd:.4f}")

    out_dir = Path("."); out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / "out_qwen2_5_vl_preds.txt"
    with pred_path.open("w", encoding="utf-8") as f:
        f.write("#True\t#Pred\n")
        for y, y_ in zip(gts, preds):
            f.write(f"{y}\t{y_}\n")
    print(f"[*] saved -> {pred_path.resolve()}")

    if raws:
        raw_path = out_dir / "raw_generations.jsonl"
        with raw_path.open("w", encoding="utf-8") as f:
            for item in raws:
                if isinstance(item, tuple):
                    idx, txt = item
                else:
                    idx, txt = -1, item
                f.write(json.dumps({"index": idx, "text": txt}, ensure_ascii=False) + "\n")
        print(f"[*] saved -> {raw_path.resolve()}")

def _find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]

def main():
    args = resolve_paths(build_args())
    distributed = getattr(args, "distributed", False)

    if distributed:
        world_size = torch.cuda.device_count()
        if world_size <= 1:
            print("[warn] Only 1 GPU visible. Running single-process.")
            ddp_worker(rank=0, world_size=1, args_dict=args.__dict__)
            return
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", str(_find_free_port()))
        os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
        os.environ.setdefault("NCCL_IB_DISABLE", "1")
        os.environ.setdefault("NCCL_P2P_DISABLE", "0")
        mp.spawn(ddp_worker, args=(world_size, args.__dict__), nprocs=world_size, join=True)
    else:
        ddp_worker(rank=0, world_size=1, args_dict=args.__dict__)

if __name__ == "__main__":
    main()
