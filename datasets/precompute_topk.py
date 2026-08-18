#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import argparse
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import sys
parent = Path(__file__).resolve().parent.parent
if str(parent) not in sys.path:
    sys.path.insert(0, str(parent))
from src.data.dataset_info import get_labels_and_template
# ---------------- Defaults ----------------
DEFAULT_CLASS_MAP: dict[str, list[str]] = {
    # coarse-grained
    "mvsa-s": ["negative", "neutral", "positive"],
    "mvsa-m": ["negative", "neutral", "positive"],
    "tumemo": ["angry", "bored", "calm", "fear", "happy", "love", "sad"],
    # fine-grained
    "t2015":  ["negative", "neutral", "positive"],
    "t2017":  ["negative", "neutral", "positive"],
    "masad":  ["negative", "positive"],
}
def get_classes_for_dataset(name: str) -> list[str]:
    return DEFAULT_CLASS_MAP.get((name or "").lower(), ["negative", "neutral", "positive"])

# ---------------- Core utils ----------------
def load_emb(path: Path, dtype=np.float32) -> np.ndarray:
    x = np.load(str(path))
    if x.ndim != 2:
        raise ValueError(f"Embedding must be 2D, got {x.shape} from {path}")
    if x.dtype != dtype:
        x = x.astype(dtype)
    return x

def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n = np.maximum(n, eps)
    return x / n

def read_lines(path: Path) -> List[str]:
    return [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]

def read_train_labels(train_file: Path, *, label_field: str = "label") -> List[str]:
    """
    支持：
      - .jsonl：逐行 JSON，取 label_field
      - .txt  ：每行一个标签
    """
    if train_file.suffix.lower() == ".json":
        labels = []
        with train_file.open("r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    lab = str(obj.get(label_field, "")).strip()
                except Exception:
                    lab = ""
                labels.append(lab)
        return labels
    else:
        # 备用：纯文本标签表
        return read_lines(train_file)

def stable_unique(seq: List[str]) -> List[str]:
    seen, out = set(), []
    for s in seq:
        if s not in seen:
            seen.add(s); out.append(s)
    return out

def argtopk_rows(S: np.ndarray, K: int) -> Tuple[np.ndarray, np.ndarray]:
    N = S.shape[1]
    if K < N:
        part = np.argpartition(-S, kth=K-1, axis=1)[:, :K]
        part_vals = np.take_along_axis(S, part, axis=1)
        order = np.argsort(-part_vals, axis=1)
        topk_idx = np.take_along_axis(part,      order, axis=1)
        topk_val = np.take_along_axis(part_vals, order, axis=1)
    else:
        order = np.argsort(-S, axis=1)
        topk_idx = order
        topk_val = np.take_along_axis(S, order, axis=1)
    return topk_idx, topk_val

def balanced_merge(per_class_map: Dict[str, np.ndarray], strategy: str = "roundrobin") -> np.ndarray:
    classes = list(per_class_map.keys())
    mats = [per_class_map[c] for c in classes]
    Nq, Kc = mats[0].shape
    C = len(mats)
    out = np.full((Nq, C * Kc), -1, dtype=mats[0].dtype)
    if strategy == "concat":
        for i, m in enumerate(mats):
            out[:, i*Kc:(i+1)*Kc] = m
        return out
    # round-robin
    pos = 0
    for r in range(Kc):
        for i in range(C):
            out[:, pos] = mats[i][:, r]
            pos += 1
    return out

# ---------------- Main ----------------
def main():
    ap = argparse.ArgumentParser(description="Build offline TopK (global + per-class + balanced)")
    ap.add_argument("--train_emb",   required=True, help="train_{tag}.npy (Nt, D)")
    ap.add_argument("--query_emb",   required=True, help="{split}_{tag}.npy (Nq, D)")
    ap.add_argument("--store_k",     type=int, default=50, help="Global TopK to store (sorted)")
    ap.add_argument("--out_prefix",  required=True, help="Output prefix, e.g. datasets/mvsa-s/test2train_sbert")
    ap.add_argument("--chunk_size",  type=int, default=0, help="Manual batch size (#queries); 0 = auto")
    ap.add_argument("--no_normalize", action="store_true", help="Do NOT L2-normalize (then sims=dot product)")

    # 类别均衡增强（只需要传训练集文件即可）
    ap.add_argument("--train_file",   type=str, default=None, help="训练集文件：.jsonl（含label字段）或 .txt（每行一个标签）")
    ap.add_argument("--label_field",  type=str, default="label", help="jsonl中标签字段名")
    ap.add_argument("--dataset_name", type=str, default="", help="（可选）数据集名，用内置类别映射；留空则从训练文件推断")
    ap.add_argument("--per_class_k",  type=int, default=0, help="每类TopK（>0时导出按类文件与均衡文件）")
    ap.add_argument("--balance_strategy", type=str, default="roundrobin", choices=["roundrobin","concat"],
                    help="合并各类TopK的策略（均衡序列输出用）")

    args = ap.parse_args()

    # --- load embeddings ---
    train = load_emb(Path(args.train_emb))
    query = load_emb(Path(args.query_emb))
    if not args.no_normalize:
        train = l2_normalize(train); query = l2_normalize(query); sim_name = "cosine"
    else:
        sim_name = "dot"

    Nt, D  = train.shape
    Nq     = query.shape[0]
    K      = min(args.store_k, Nt)

    if args.chunk_size > 0:
        chunk_q = args.chunk_size
    else:
        target_mm = 25_000_000  # ~25M muls/chunk
        chunk_q = max(1, target_mm // max(Nt, 1))
        chunk_q = min(chunk_q, Nq)

    print(f"[info] train: {train.shape}, query: {query.shape}, sim={sim_name}")
    print(f"[info] globalK={K}, chunk_q={chunk_q}")

    # --- outputs: global ---
    idx_out = np.empty((Nq, K), dtype=np.int64)
    sim_out = np.empty((Nq, K), dtype=np.float32)

    # --- per-class setup ---
    has_class = (args.per_class_k > 0) and (args.train_file is not None)
    classes: List[str] = []
    labels: Optional[List[str]] = None
    cls_to_indices: Dict[str, np.ndarray] = {}

    if has_class:
        train_file = Path(args.train_file)
        if not train_file.exists():
            raise FileNotFoundError(f"--train_file not found: {train_file}")
        labels = read_train_labels(train_file, label_field=args.label_field)
        _,label_map,_ = get_labels_and_template(args.dataset_name, template_id=2)
        labels = [label_map.get(lab, lab) for lab in labels]
        if len(labels) != Nt:
            raise ValueError(f"train_file rows (labels) = {len(labels)} != train_emb rows {Nt} "
                             f"(请确保嵌入行序与训练文件一致)")

        if args.dataset_name:
            classes = get_classes_for_dataset(args.dataset_name)
        else:
            classes = stable_unique(labels)

        lbl_arr = np.array(labels, dtype=object)
        for c in classes:
            cls_to_indices[c] = np.where(lbl_arr == c)[0]

        Kc = args.per_class_k
        per_cls_idx = {c: np.full((Nq, Kc), -1, dtype=np.int64) for c in classes}
        per_cls_sim = {c: np.full((Nq, Kc), -np.float32(np.inf), dtype=np.float32) for c in classes}
        print(f"[info] per-class enabled: classes={classes}, per_class_k={Kc}")

    # --- compute by chunks ---
    for st in range(0, Nq, chunk_q):
        ed = min(Nq, st + chunk_q)
        sims = query[st:ed] @ train.T  # (B, Nt)

        # Global TopK
        g_idx, g_sim = argtopk_rows(sims, K)
        idx_out[st:ed] = g_idx
        sim_out[st:ed] = g_sim.astype(np.float32)

        # Per-class TopK
        if has_class:
            for c, t_idx in cls_to_indices.items():
                if t_idx.size == 0:
                    continue
                S = sims[:, t_idx]                     # (B, Nc)
                kc = min(args.per_class_k, t_idx.size)
                pi, ps = argtopk_rows(S, kc)          # (B, kc) in t_idx space
                mapped_idx = t_idx[pi]                # back to global idx
                per_cls_idx[c][st:ed, :kc] = mapped_idx
                per_cls_sim[c][st:ed, :kc] = ps.astype(np.float32)

    # --- save global ---
    idx_path = f"{args.out_prefix}_top{K}_idx.npy"
    sim_path = f"{args.out_prefix}_top{K}_sim.npy"
    np.save(idx_path, idx_out)
    np.save(sim_path, sim_out)
    print(f"[save] global idx -> {idx_path}  shape={idx_out.shape}, dtype=int64")
    print(f"[save] global sim -> {sim_path}  shape={sim_out.shape}, dtype=float32")
    print(f"[peek] global[0, :min(5,K)]: idx={idx_out[0, :min(5,K)].tolist()}, "
          f"sim={np.round(sim_out[0, :min(5,K)],4).tolist()}")

    # --- save per-class & balanced ---
    if has_class:
        # per-class npz（每类各自的 TopK，按相似度已排序）
        per_cls_npz = {"__classes__": np.array(classes, dtype=object)}
        for c in classes:
            per_cls_npz[f"idx::{c}"] = per_cls_idx[c]
            per_cls_npz[f"sim::{c}"] = per_cls_sim[c]
        perclass_path = f"{args.out_prefix}_perclass_top{args.per_class_k}.npz"
        np.savez_compressed(perclass_path, **per_cls_npz)
        print(f"[save] per-class npz -> {perclass_path}")
        for c in classes:
            print(f"       - {c}: idx shape={per_cls_idx[c].shape}, sim shape={per_cls_sim[c].shape}, "
                  f"example idx={per_cls_idx[c][0,:min(3,args.per_class_k)].tolist()}, "
                  f"sim={np.round(per_cls_sim[c][0,:min(3,args.per_class_k)],4).tolist()}")

        # 均衡序列（roundrobin 或 concat）
        bal_idx = balanced_merge({c: per_cls_idx[c] for c in classes}, strategy=args.balance_strategy)
        bal_sim = balanced_merge({c: per_cls_sim[c] for c in classes}, strategy=args.balance_strategy).astype(np.float32)

        C, Kc = len(classes), args.per_class_k
        b_idx_path = f"{args.out_prefix}_balanced_{args.balance_strategy}_top{C}x{Kc}_idx.npy"
        b_sim_path = f"{args.out_prefix}_balanced_{args.balance_strategy}_top{C}x{Kc}_sim.npy"
        print(bal_idx)
        np.save(b_idx_path, bal_idx)
        np.save(b_sim_path, bal_sim)
        print(f"[save] balanced idx -> {b_idx_path}  shape={bal_idx.shape}, dtype=int64, pad=-1")
        print(f"[save] balanced sim -> {b_sim_path}  shape={bal_sim.shape}, dtype=float32, pad=-inf")
        print(f"[peek] balanced[0, :min(9, bal_idx.shape[1])]: idx={bal_idx[0,:min(9,bal_idx.shape[1])].tolist()}, "
              f"sim={np.round(bal_sim[0,:min(9,bal_sim.shape[1])],4).tolist()}")

if __name__ == "__main__":
    main()
