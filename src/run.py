# -*- coding: utf-8 -*-
import os
import re
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__))) 

import json
import time
import random
import socket
from time import monotonic
from pathlib import Path
from typing import List, Tuple
from ensemble import _majority_vote
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

from src.params import build_args, resolve_paths
from src.dataset import MSADataset
from src.prompts import build_instruction, build_user_content, build_prompt_variant
from src.dataset_info import get_labels_and_template
from src.utils import set_seed, load_model_and_processor, parse_label_from_output, infer_with_variants
from src.retrieve_demo import DemoProvider
from src.prompt_tuning.prompt_learner import load_prompt_ckpt
from logs.logs import _finalize_and_save,_resolve_pred_field,_write_rag_debug_and_stats
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
from transformers.utils import logging as hf_logging  # noqa: E402
hf_logging.set_verbosity_error()
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")

def _maybe_init_prompt(model, processor, label_space):
    ckpt_path = os.getenv("SOFT_PROMPT_CKPT", "").strip()
    if not ckpt_path:
        return None
    st = load_prompt_ckpt(ckpt_path, map_location=model.device)
    ids = processor.tokenizer.convert_tokens_to_ids(st["soft_tokens"])
    vecs = st["soft_vecs"].to(model.device).to(model.get_input_embeddings().weight.dtype)
    with torch.no_grad():
        model.get_input_embeddings().weight[ids] = vecs
    print(f"[*] soft-prompt rows loaded into embedding from {ckpt_path}", flush=True)
    return True
# ==== metrics helpers (no sklearn) ====
def _compute_confusion_and_metrics(preds, gts, labels):
    """Return cm(np.int64[K,K]), metrics(dict), and per-class acc(dict)."""
    import numpy as _np

    label2id = {l: i for i, l in enumerate(labels)}
    y_true = _np.array([label2id[x] for x in gts], dtype=_np.int64)
    y_pred = _np.array([label2id[x] for x in preds], dtype=_np.int64)

    K = len(labels)
    cm = _np.zeros((K, K), dtype=_np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1

    support = cm.sum(axis=1)                    # per true class
    pred_sum = cm.sum(axis=0)                   # per predicted class
    tp = _np.diag(cm)

    # avoid div-by-zero
    recall = _np.divide(tp, _np.maximum(support, 1), dtype=_np.float64)
    precision = _np.divide(tp, _np.maximum(pred_sum, 1), dtype=_np.float64)
    f1 = _np.divide(2 * precision * recall, _np.maximum(precision + recall, 1e-12))

    acc = float((y_true == y_pred).mean())
    macro_f1 = float(f1.mean())
    weighted_f1 = float((f1 * (support / _np.maximum(support.sum(), 1))).sum())

    per_class_acc = {labels[i]: float(recall[i]) for i in range(K)}

    metrics = {
        "acc": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "per_class_acc": per_class_acc,
        "support": {labels[i]: int(support[i]) for i in range(K)},
    }
    return cm, metrics, per_class_acc


def _print_and_save_confusion(cm, labels, out_dir, prefix):
    """Pretty-print and save CSV/JSON."""
    import os as _os, json as _json
    from pathlib import Path as _Path
    import numpy as _np

    _Path(out_dir).mkdir(parents=True, exist_ok=True)

    # Console pretty print
    K = len(labels)
    header = " " * 12 + " | " + " ".join([f"{l:>8}" for l in labels])
    print("\n[confusion_matrix] rows=gold, cols=pred")
    print(header)
    for i, l in enumerate(labels):
        row = " ".join([f"{int(cm[i, j]):>8d}" for j in range(K)])
        print(f"{l:>10} | {row}")


def run_one(model, processor, messages, max_new_tokens: int, label_space: List[str]) -> str:
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
        BarColumn(bar_width=None),
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
    reader = MSADataset(args, Path(tsv_path), dataset_name=dataset_name, label_map=label_map)
    samples = reader.read()
    samples_meta = reader.get_samples_meta()
    # Model & Processor
    use_fast = getattr(args, "use_fast_processor", False)
    model, processor = load_model_and_processor(
        model_id=args.model, dtype=args.dtype, device=device,
        attn_impl=args.attn_impl, min_pixels=args.min_pixels, max_pixels=args.max_pixels,
        use_fast_processor=use_fast
    )
    _maybe_init_prompt(model, processor, label_space)
    # Prompt setup
    use_image_flag = (args.img_dir is not None) and (not args.no_img)
    has_aspect = getattr(
        args, "has_aspect",
        True if "fine" in str(tmpl).lower() else ("t2015" in dataset_name or "t2017" in dataset_name or "masad" in dataset_name)
    )

    # Determine template variant(s)
    prompt_variants = build_prompt_variant()

    # RAG inference
    demo = DemoProvider.from_env(args, dataset_name=dataset_name, image_base=Path(args.data_dir) / "imgs")

    local_results: List[Tuple[int, str]] = []
    local_gts: List[Tuple[int, str]] = []
    local_raws: List[Tuple[int, str]] = []
    demo_diags_by_idx = {}
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

        # ---------- build demos from offline index + diagnostics----------
        prefix_demo_msgs, demo_diag = demo.for_query(
            i, label_map=label_map, dataset_name=dataset_name
        )
        demo_diags_by_idx[i] = demo_diag

        final_pred, per_tpl_preds, raw_bundle = infer_with_variants(
            model=model,
            processor=processor,
            prompt_variants=prompt_variants,
            label_space=label_space,
            use_image_flag=use_image_flag,
            img_path=(str(img_path) if img_path is not None else None),
            user_text=user_text,
            prefix_demo_msgs=prefix_demo_msgs,
            max_new_tokens=args.max_new_tokens,
            run_one_fn=run_one,
            parse_label_fn=parse_label_from_output,
            has_aspect=has_aspect,  
        )
        local_results.append((i, final_pred))
        local_gts.append((i, label_map[s.label]))
        demo_diags_by_idx[i]["pred"] = final_pred
        demo_diags_by_idx[i]["gold"] = label_map[s.label]
        if len(prompt_variants) > 1:
            demo_diags_by_idx[i]["tpl_preds"] = per_tpl_preds
        if getattr(args, "dump_raw", False):
            if len(prompt_variants) == 1:
                local_raws.append((i, raw_bundle[0]["raw"]))
            else:
                local_raws.append((i, json.dumps(
                    {"tpl_preds": per_tpl_preds, "tpl_raws": raw_bundle}, ensure_ascii=False
                )))
        if rank == 0 and (i % 100 == 0):
            print(f"Sample {i} ground_truth={label_map[s.label]}, final_pred={final_pred}, tpl_preds={per_tpl_preds}", flush=True)
    # Collect results and evaluate
    pred_field = _resolve_pred_field()
    if is_dist:
        pack = {"preds": local_results, "gts": local_gts, "raws": local_raws, "diag":  demo_diags_by_idx,}
        gathered = [None for _ in range(world_size)]
        dist.all_gather_object(gathered, pack)

        if rank == 0:
            all_pred, all_gt, all_raw = [], [], []
            merged_diag = {}  
            for p in gathered:
                all_pred.extend(p["preds"]); all_gt.extend(p["gts"]); all_raw.extend(p["raws"])
                for k, v in p.get("diag", {}).items():
                    merged_diag[k] = v
            all_pred.sort(key=lambda x: x[0]); all_gt.sort(key=lambda x: x[0]); all_raw.sort(key=lambda x: x[0])

            preds = [p for _, p in all_pred]
            gts   = [g for _, g in all_gt]
            raws  = [r for _, r in all_raw]
            _finalize_and_save(preds, gts, raws, dataset_name, samples_meta, pred_field)
            cm, metrics, _ = _compute_confusion_and_metrics(preds, gts, label_space)
            out_dir = Path("logs") / dataset_name
            prefix = pred_field  # e.g., pred_demo1_none / pred_none_none 等
            _print_and_save_confusion(cm, label_space, out_dir, prefix)
            if demo.use_demo:
                _write_rag_debug_and_stats(dataset_name, pred_field, samples_meta, merged_diag, preds, gts)
        dist.barrier()
        dist.destroy_process_group()
    else:
        preds = [p for _, p in sorted(local_results, key=lambda x: x[0])]
        gts   = [g for _, g in sorted(local_gts,     key=lambda x: x[0])]
        raws  = [r for _, r in sorted(local_raws,    key=lambda x: x[0])] if getattr(args, "dump_raw", False) else []
        _finalize_and_save(preds, gts, raws, dataset_name, samples_meta, pred_field)
        cm, metrics, _ = _compute_confusion_and_metrics(preds, gts, label_space)
        out_dir = Path("logs") / dataset_name
        prefix = pred_field  # e.g., pred_demo1_none / pred_none_none 等
        _print_and_save_confusion(cm, label_space, out_dir, prefix)
        if demo.use_demo:
            _write_rag_debug_and_stats(dataset_name, pred_field, samples_meta, demo_diags_by_idx, preds, gts)

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
