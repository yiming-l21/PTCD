#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, os, json, numpy as np
from tqdm import tqdm
from pathlib import Path
# 如果你要用占位(不下载模型)的“零向量”版本，把下面两行注释取消：
# USE_DUMMY = True
# EMBED_DIM = 1024  # 占位维度，自行设定（真实模型一般是 1024）
USE_DUMMY = False

def read_jsonl(path):
    items = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line: 
                continue
            try:
                obj = json.loads(line)
                items.append(obj)
            except Exception:
                # 忽略截断/坏行
                pass
    return items

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", required=True, help="JSONL 文件路径（每行一个样本）")
    parser.add_argument("--data_dir", required=True, help="保存 .npy 的目录（比如 /home/lym/MultiPoint/datasets/mvsa-s）")
    parser.add_argument("--split", required=True, choices=["train","val","test"], help="当前生成哪个 split 的向量")
    parser.add_argument("--model_tag", default="sbert-roberta-large", help="命名到文件名中的模型短名，如 sbert-roberta-large")
    parser.add_argument("--hf_model", default="usc-isi/sbert-roberta-large-anli-mnli-snli", help="HuggingFace sentence-transformers 模型名")
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()
    out_path = os.path.join(args.data_dir, f"{args.split}_{args.model_tag}.npy")
    os.makedirs(args.data_dir, exist_ok=True)
    samples = read_jsonl(args.jsonl)
    texts = [ (s.get("text") or "").strip() for s in samples ]
    print(f"Loaded {len(texts)} rows from {args.jsonl}")

    if USE_DUMMY:
        # 生成占位向量（全零），可以先跑通流程
        emb = np.zeros((len(texts), EMBED_DIM), dtype=np.float32)
    else:
        # 真正计算句向量
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(args.hf_model)
        emb_chunks = []
        for i in tqdm(range(0, len(texts), args.batch_size)):
            batch = texts[i:i+args.batch_size]
            vec = model.encode(batch, convert_to_numpy=True, normalize_embeddings=True)
            emb_chunks.append(vec.astype(np.float32))
        emb = np.concatenate(emb_chunks, axis=0) if emb_chunks else np.zeros((0, 1024), dtype=np.float32)
    np.save(out_path, emb)
    print(f"Saved: {out_path}  shape={emb.shape}")

if __name__ == "__main__":
    main()
