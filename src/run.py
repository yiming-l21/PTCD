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
    import os, torch

    def _is_rank0():
        try:
            import torch.distributed as dist
            return (not dist.is_initialized()) or (dist.get_rank() == 0)
        except Exception:
            return True

    LOG_ON = int(os.getenv("VIS_SP_LOG", "1")) > 0 and _is_rank0()
    LOG_N  = int(os.getenv("VIS_SP_LOG_N", "3"))
    STRICT = int(os.getenv("VIS_SP_STRICT", "1")) > 0
    ASSUME_B1 = int(os.getenv("VIS_SP_ASSUME_B1", "1")) > 0  # 2D+无Ns时按B=1处理

    ckpt_path = os.getenv("SOFT_PROMPT_CKPT", "").strip()
    if not ckpt_path or not os.path.exists(ckpt_path):
        if LOG_ON: print("[WARN] 未指定或不存在软提示checkpoint，跳过加载", flush=True)
        return None

    try:
        st = load_prompt_ckpt(ckpt_path, map_location=model.device)
    except Exception as e:
        print(f"[ERROR] 加载checkpoint失败：{ckpt_path}，错误：{str(e)}", flush=True)
        return None

    text_only   = int(st.get("text_only", 0) or 0)
    visual_only = int(st.get("visual_only", 0) or 0)

    # --- 文本软提示：写回 embedding
    if (not visual_only) and st.get("soft_tokens") is not None and st.get("soft_vecs") is not None:
        soft_tokens = list(st["soft_tokens"])
        soft_vecs   = st["soft_vecs"].to(model.device)
        missing = [t for t in soft_tokens if processor.tokenizer.convert_tokens_to_ids(t) == processor.tokenizer.unk_token_id]
        if missing:
            processor.tokenizer.add_special_tokens({"additional_special_tokens": missing})
            model.resize_token_embeddings(len(processor.tokenizer))
            if LOG_ON: print(f"[*] tokenizer 新增 {len(missing)} 个软提示 token 并已 resize", flush=True)
        ids = processor.tokenizer.convert_tokens_to_ids(soft_tokens)
        if len(ids) != soft_vecs.shape[0]:
            msg = f"[ERROR] 文本软提示token数与向量数不匹配：{len(ids)} vs {soft_vecs.shape[0]}"
            print(msg, flush=True)
            if STRICT: raise RuntimeError(msg)
            return None
        with torch.no_grad():
            emb = model.get_input_embeddings().weight
            emb[torch.tensor(ids, device=emb.device, dtype=torch.long)] = soft_vecs.to(emb.dtype)
        if LOG_ON: print(f"[*] 加载文本软提示：{len(ids)} 个（{ckpt_path}）", flush=True)
    else:
        if LOG_ON: print(f"[*] 跳过文本软提示（visual_only={visual_only}）", flush=True)

    # --- 视觉软提示：注册插入
    if text_only or st.get("visual_sp_vecs") is None or int(st.get("visual_sp_n_tokens", 0)) <= 0:
        if LOG_ON: print(f"[*] 跳过视觉软提示（text_only={text_only}）", flush=True)
        return True

    visual_vecs = st["visual_sp_vecs"].to(model.device)  # [n_sp, 1280]
    n_sp = int(st["visual_sp_n_tokens"])
    if visual_vecs.shape != (n_sp, 1280):
        msg = f"[ERROR] 视觉软提示形状非法：{tuple(visual_vecs.shape)}，需 (n_sp,1280)"
        print(msg, flush=True)
        if STRICT: raise RuntimeError(msg)
        return None

    # 找到 visual.merger
    merger = None
    cur = getattr(model, "model", None)
    if cur is not None:
        cur = getattr(cur, "visual", None)
        if cur is not None:
            merger = getattr(cur, "merger", None)
    if merger is None:
        for m in model.modules():
            if hasattr(m, "forward") and "merger" in m.__class__.__name__.lower():
                merger = m; break
    if merger is None:
        msg = "[ERROR] 未找到视觉 merger 模块，无法注册视觉软提示 hook"
        print(msg, flush=True)
        if STRICT: raise RuntimeError(msg)
        return None

    # dtype 对齐
    try:
        pe = getattr(model.model.visual, "patch_embedding", None)
        vis_dtype = (pe.weight.dtype if (pe is not None and hasattr(pe, "weight")) else next(model.parameters()).dtype)
    except Exception:
        vis_dtype = next(model.parameters()).dtype
    visual_vecs = visual_vecs.to(dtype=vis_dtype)

    model._visual_sp_debug = {
        "calls": 0, "last_B": None, "n_sp": n_sp, "vis_dtype": str(vis_dtype),
        "last_before_shape": None, "last_after_shape": None, "max_abs_diff": None,
    }

    def _pre_hook(module, inputs, kwargs):
        if not inputs: return None
        x = inputs[0] if (isinstance(inputs, (list, tuple)) and len(inputs) >= 1) else None

        # try kw/inputs 提取 Ns（THW 或 lengths）
        Ns = None
        thw_keys = ("image_grid_thw", "grid_thw", "thw")
        len_keys = ("tokens_per_image", "lengths", "image_lengths")
        for k in thw_keys:
            t = kwargs.get(k, None)
            if torch.is_tensor(t) and t.dim() == 2 and t.size(-1) == 3:
                Ns = (t.to(dtype=torch.int64).prod(dim=1)).tolist(); break
        if Ns is None:
            for k in len_keys:
                t = kwargs.get(k, None)
                if torch.is_tensor(t) and t.dim() == 1:
                    Ns = t.to(dtype=torch.int64).tolist(); break
        if Ns is None and isinstance(inputs, (list, tuple)):
            for t in inputs[1:5]:
                if torch.is_tensor(t) and t.dim() == 2 and t.size(-1) == 3:
                    Ns = (t.to(dtype=torch.int64).prod(dim=1)).tolist(); break
                if torch.is_tensor(t) and t.dim() == 1:
                    Ns = t.to(dtype=torch.int64).tolist(); break

        # 3D：直接 (B,N,1280) 前拼
        if torch.is_tensor(x) and x.dim() == 3 and x.size(-1) == 1280:
            B, N, D = x.shape
            sp = visual_vecs.unsqueeze(0).expand(B, -1, -1)
            new_first = torch.cat([sp, x], dim=1)
            dbg = model._visual_sp_debug; dbg["calls"] += 1; dbg["last_B"] = int(B)
            dbg["last_before_shape"] = (int(B), int(N), int(D))
            dbg["last_after_shape"]  = (int(new_first.size(0)), int(new_first.size(1)), int(new_first.size(2)))
            with torch.no_grad():
                head = new_first[:, :n_sp, :]; mad = (head - sp).abs().max().item() if head.numel() > 0 else 0.0
            dbg["max_abs_diff"] = float(mad)
            if LOG_ON and dbg["calls"] <= LOG_N:
                print(f"[VIS-SP/3D][{dbg['calls']}] before={(B,N,D)} add=(B,{n_sp},1280) -> after={tuple(dbg['last_after_shape'])} max_abs_diff={mad:.3e}", flush=True)
            if STRICT:
                assert new_first.size(1) == N + n_sp and mad <= 1e-5
            return ((new_first,) + inputs[1:], kwargs)

        # 2D：有 Ns → 按 batch 拆分后逐样本前拼
        if torch.is_tensor(x) and x.dim() == 2 and x.size(-1) == 1280 and (Ns is not None):
            tokens_total, D = x.shape
            sNs = sum(Ns)
            if sNs != tokens_total:
                if LOG_ON and model._visual_sp_debug["calls"] < LOG_N:
                    print(f"[VIS-SP/2D][SKIP] sum(Ns)={sNs} != tokens={tokens_total}", flush=True)
                if STRICT: raise RuntimeError(f"2D 拆分失败：sum(Ns)={sNs} vs tokens={tokens_total}")
                return None
            chunks, off = [], 0
            for nb in Ns:
                seg = x[off:off+nb, :]
                sp  = visual_vecs.to(x.dtype).reshape(n_sp, 1280)
                chunks.append(torch.cat([sp, seg], dim=0))
                off += nb
            new_first = torch.cat(chunks, dim=0)
            dbg = model._visual_sp_debug; dbg["calls"] += 1
            dbg["last_B"] = int(len(Ns))
            dbg["last_before_shape"] = (int(tokens_total), int(D))
            dbg["last_after_shape"]  = (int(new_first.size(0)), int(new_first.size(1)))
            with torch.no_grad():
                mad = (new_first[:n_sp, :] - visual_vecs.to(new_first.dtype)).abs().max().item()
            dbg["max_abs_diff"] = float(mad)
            if LOG_ON and dbg["calls"] <= LOG_N:
                print(f"[VIS-SP/2D][{dbg['calls']}] before={(tokens_total,D)} sum(Ns)={sNs} B={len(Ns)} add_each={n_sp} -> after={tuple(dbg['last_after_shape'])} max_abs_diff={mad:.3e}", flush=True)
            if STRICT:
                expect = tokens_total + len(Ns) * n_sp
                assert new_first.size(0) == expect and mad <= 1e-5
            return ((new_first,) + inputs[1:], kwargs)

        # 2D：无 Ns 且允许 B=1 兜底 → 直接在前面拼 n_sp
        if torch.is_tensor(x) and x.dim() == 2 and x.size(-1) == 1280 and Ns is None and ASSUME_B1:
            tokens_total, D = x.shape
            sp = visual_vecs.to(x.dtype).reshape(n_sp, 1280)
            new_first = torch.cat([sp, x], dim=0)  # (n_sp+tokens_total,1280)
            dbg = model._visual_sp_debug; dbg["calls"] += 1
            dbg["last_B"] = 1
            dbg["last_before_shape"] = (int(tokens_total), int(D))
            dbg["last_after_shape"]  = (int(new_first.size(0)), int(new_first.size(1)))
            with torch.no_grad():
                mad = (new_first[:n_sp, :] - sp).abs().max().item()
            dbg["max_abs_diff"] = float(mad)
            if LOG_ON and dbg["calls"] <= LOG_N:
                print(f"[VIS-SP/2D-B1][{dbg['calls']}] before={(tokens_total,D)} assume_B=1 add={n_sp} -> after={tuple(dbg['last_after_shape'])} max_abs_diff={mad:.3e}", flush=True)
            if STRICT:
                assert new_first.size(0) == tokens_total + n_sp and mad <= 1e-5
            return ((new_first,) + inputs[1:], kwargs)

        # 其他：打印一次签名
        if LOG_ON and model._visual_sp_debug["calls"] < LOG_N:
            def _pp(v):
                if torch.is_tensor(v): return f"Tensor{tuple(v.shape)} dtype={v.dtype}"
                if isinstance(v, (list, tuple)): return [ _pp(t) for t in v[:8] ] + (["..."] if len(v) > 8 else [])
                if isinstance(v, dict): return {k: _pp(val) for k, val in list(v.items())[:8]}
                return repr(type(v).__name__)
            print(f"[VIS-SP][SKIP] unexpected shape={_pp(x)} (no Ns) inputs_sig={_pp(inputs)} kwargs_sig={_pp(kwargs)}", flush=True)
        return None

    h_merger = merger.register_forward_pre_hook(_pre_hook, with_kwargs=True)
    if not hasattr(model, "_visual_sp_handle"): model._visual_sp_handle = []
    model._visual_sp_handle.append(h_merger)
    if LOG_ON: print(f"[*] 视觉软提示 pre_hook 已注册：n_sp={n_sp}, dtype={vis_dtype}, 插入点=visual.merger 前", flush=True)

    return True

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

@torch.inference_mode()
def run_one(
    model,
    processor,
    messages,
    max_new_tokens: int,
    label_space: List[str],
) -> str:
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    images, videos = process_vision_info(messages)
    inputs = processor(text=[text], images=images, videos=videos, padding=True, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            return_dict_in_generate=True,
            output_scores=True,  # 拿每步 logits
        )
    gen_ids = []
    for s in out.scores:  # s: [B, vocab]
        probs = torch.softmax(s.float(), dim=-1)
        token = probs.argmax(dim=-1) 
        gen_ids.append(token)
    gen_ids = torch.stack(gen_ids, dim=1)[0]  # [new_len]
    text_out = processor.decode(gen_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
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
