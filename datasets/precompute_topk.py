#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import argparse
import numpy as np
from pathlib import Path

def load_emb(path: Path, dtype=np.float32) -> np.ndarray:
    x = np.load(str(path))
    if x.ndim != 2:
        raise ValueError(f"Embedding must be 2D, got {x.shape} from {path}")
    if x.dtype != dtype:
        x = x.astype(dtype)
    return x

def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    # row-wise L2 norm -> avoid division by zero
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n = np.maximum(n, eps)
    return x / n
def main():
    ap = argparse.ArgumentParser(description="Build offline TopK (indices + precomputed sims)")
    ap.add_argument("--train_emb",   required=True, help="train_{tag}.npy (Nt, D)")
    ap.add_argument("--query_emb",   required=True, help="{split}_{tag}.npy (Nq, D)")
    ap.add_argument("--store_k",     type=int, default=50, help="TopK to store (sorted)")
    ap.add_argument("--out_prefix",  required=True, help="output prefix, e.g. /path/mvsa-s/test2train_sbert")
    ap.add_argument("--chunk_size",  type=int, default=0, help="manual batch size (#queries); 0 = auto")
    ap.add_argument("--no_normalize", action="store_true", help="do NOT L2-normalize; then sims = dot product")
    # optional id lists so we can align by id later
    ap.add_argument("--queries_id",  type=str, default=None, help="text file: one query id per line")
    ap.add_argument("--train_id",    type=str, default=None, help="text file: one train id per line")
    args = ap.parse_args()

    train = load_emb(Path(args.train_emb))      # (Nt, D)
    query = load_emb(Path(args.query_emb))      # (Nq, D)

    # cosine similarity by default
    if not args.no_normalize:
        train = l2_normalize(train)
        query = l2_normalize(query)
        sim_name = "cosine"
    else:
        sim_name = "dot"

    Nt, D  = train.shape
    Nq     = query.shape[0]
    K      = min(args.store_k, Nt)

    # auto chunk: keep a rough matmul size to limit memory
    if args.chunk_size > 0:
        chunk_q = args.chunk_size
    else:
        target_mm = 25_000_000  # ~25M multiply-adds per chunk
        chunk_q = max(1, target_mm // max(Nt, 1))
        chunk_q = min(chunk_q, Nq)

    idx_out = np.empty((Nq, K), dtype=np.int64)
    sim_out = np.empty((Nq, K), dtype=np.float32)

    print(f"[info] train: {train.shape}, query: {query.shape}, store_k={K}, chunk_q={chunk_q}, sim={sim_name}")

    for st in range(0, Nq, chunk_q):
        ed = min(Nq, st + chunk_q)
        # (B, D) @ (D, Nt) -> (B, Nt)
        sims = query[st:ed] @ train.T

        if K < Nt:
            # partial selection to get TopK (unsorted), then sort inside K
            part = np.argpartition(-sims, kth=K-1, axis=1)[:, :K]  # (B, K)
            part_sims = np.take_along_axis(sims, part, axis=1)     # (B, K)
            order = np.argsort(-part_sims, axis=1)                 # (B, K)
            topk_idx = np.take_along_axis(part,     order, axis=1) # (B, K)
            topk_sim = np.take_along_axis(part_sims, order, axis=1)# (B, K)
        else:
            order = np.argsort(-sims, axis=1)                      # (B, Nt)
            topk_idx = order                                       # (B, Nt)
            topk_sim = np.take_along_axis(sims, order, axis=1)     # (B, Nt)

        idx_out[st:ed] = topk_idx
        sim_out[st:ed] = topk_sim.astype(np.float32)

    idx_path = f"{args.out_prefix}_top{K}_idx.npy"
    sim_path = f"{args.out_prefix}_top{K}_sim.npy"
    np.save(idx_path, idx_out)
    np.save(sim_path, sim_out)
    print(f"[save] indices -> {idx_path}  shape={idx_out.shape}, dtype={idx_out.dtype}")
    print(f"[save] sims    -> {sim_path}  shape={sim_out.shape}, dtype={sim_out.dtype}")

if __name__ == "__main__":
    main()
