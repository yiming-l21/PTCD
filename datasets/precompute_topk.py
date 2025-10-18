#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, argparse, json
import numpy as np
from pathlib import Path

def load_emb(path: Path, dtype=np.float32) -> np.ndarray:
    x = np.load(str(path))
    if x.ndim != 2:
        raise ValueError(f"Embedding must be 2D, got {x.shape} from {path}")
    if x.dtype != dtype:
        x = x.astype(dtype)
    return x

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_emb",   required=True, help="train_{tag}.npy")
    ap.add_argument("--query_emb",   required=True, help="{split}_{tag}.npy (dev/test)")
    ap.add_argument("--store_k",     type=int, default=50, help="离线存多少个候选(已排序)，在线再取 topk")
    ap.add_argument("--out_prefix",  required=True, help="输出前缀，如 /path/mvsa-s/test2train_sbert")
    ap.add_argument("--save_sims",   action="store_true", help="同时保存相似度")
    ap.add_argument("--chunk_size",  type=int, default=0, help="手动分块大小(按 query 条数)，0 表示自动估计")
    args = ap.parse_args()

    train = load_emb(Path(args.train_emb))      # (Nt, D)
    query = load_emb(Path(args.query_emb))      # (Nq, D)

    Nt, D  = train.shape[0], train.shape[1]
    Nq     = query.shape[0]
    K      = min(args.store_k, Nt)             
    if args.chunk_size > 0:
        chunk_q = args.chunk_size
    else:
        target_mm = 25_000_000
        chunk_q = max(1, target_mm // max(Nt, 1))
        chunk_q = min(chunk_q, Nq)

    idx_out = np.empty((Nq, K), dtype=np.int64)
    sim_out = np.empty((Nq, K), dtype=np.float32) if args.save_sims else None

    print(f"[info] train: {train.shape}, query: {query.shape}, store_k={K}, chunk_q={chunk_q}")

    for st in range(0, Nq, chunk_q):
        ed = min(Nq, st + chunk_q)
        sims = query[st:ed] @ train.T  # (B, Nt)
        if K < Nt:
            part = np.argpartition(-sims, kth=K-1, axis=1)[:, :K] 
            part_sims = np.take_along_axis(sims, part, axis=1)     # (B, K)
            order = np.argsort(-part_sims, axis=1)                
            topk_idx = np.take_along_axis(part,     order, axis=1) 
            topk_sim = np.take_along_axis(part_sims, order, axis=1)
        else:
            order = np.argsort(-sims, axis=1)
            topk_idx = order
            topk_sim = np.take_along_axis(sims, order, axis=1)

        idx_out[st:ed] = topk_idx
        if sim_out is not None:
            sim_out[st:ed] = topk_sim.astype(np.float32)
    idx_path = f"{args.out_prefix}_top{K}_idx.npy"
    np.save(idx_path, idx_out)
    print(f"[save] indices -> {idx_path}  shape={idx_out.shape}, dtype={idx_out.dtype}")

    if sim_out is not None:
        sim_path = f"{args.out_prefix}_top{K}_sim.npy"
        np.save(sim_path, sim_out)
        print(f"[save] sims    -> {sim_path}  shape={sim_out.shape}, dtype={sim_out.dtype}")
if __name__ == "__main__":
    main()
