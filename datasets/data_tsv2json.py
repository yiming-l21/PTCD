#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import json
from pathlib import Path
import sys
import os
COARSE = "coarse"
FINE = "fine"

COARSE_DATASETS = {"mvsa-s", "mvsa-m", "tumemo", "tumblr"} 
FINE_DATASETS   = {"t2015", "t2017", "masad"}

def dataset_mode(dataset_name: str) -> str:
    name = (dataset_name or "").strip().lower()
    if name in COARSE_DATASETS:
        return COARSE
    if name in FINE_DATASETS:
        return FINE
    return COARSE

def looks_like_header(row):
    if not row:
        return False
    joined = " ".join(x.lower() for x in row)
    return ("label" in joined) or (row[0].strip().lower() == "index")

# ---------- 解析器 ----------
def parse_row_coarse(row):
    """
    期望列序（有/无表头均可）：
    0:index, 1:label, 2:image_id, 3:text, 4:caption(可缺省)
    若超过5列，则将第5列及之后合并到 caption。
    """
    row = [c.lstrip("\ufeff") if i == 0 else c for i, c in enumerate(row)]
    if len(row) < 4:
        raise ValueError(f"Coarse row has <4 fields: {row}")

    index = row[0].strip()
    label = row[1].strip()
    image_id = row[2].strip()
    text = row[3].strip()
    caption = "\t".join(r.strip() for r in row[4:]).strip() if len(row) >= 5 else ""

    img_stem = Path(image_id).stem
    return {
        "index": index,
        "label": label,
        "image_id": image_id,
        "id": img_stem,
        "text": text,
        "caption": caption,
    }

def parse_row_fine(row):
    """
    期望列序（MASAD/t2015/t2017 常见格式）：
    0:index/id, 1:label, 2:image_id, 3:text, 4:aspect
    若有多余列（>=6），把多余列并入 caption；否则 caption=""。
    """
    row = [c.lstrip("\ufeff") if i == 0 else c for i, c in enumerate(row)]
    if len(row) < 5:
        raise ValueError(f"Fine row needs >=5 fields: {row}")

    index = row[0].strip()
    label = row[1].strip().lower()
    image_id = row[2].strip()
    text = row[3].strip()
    aspect = row[4].strip()
    caption = "\t".join(r.strip() for r in row[5:]).strip() if len(row) >= 6 else ""

    img_stem = Path(image_id).stem
    return {
        "index": index,
        "label": label,
        "image_id": image_id,
        "id": img_stem,
        "text": text,
        "caption": caption,  # fine 也保留 caption 字段（可能为空）
        "aspect": aspect,
    }

def convert(tsv_path: Path, out_path: Path, image_base: Path, mode: str):
    with tsv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(
                f,
                delimiter="\t",
                quoting=csv.QUOTE_NONE,  
                escapechar="\\",
                strict=False,
            )

        try:
            first = next(reader)
        except StopIteration:
            raise SystemExit("Empty TSV file.")

        rows_iter = reader
        if looks_like_header(first):
            # 跳过表头
            pass
        else:
            # 第一行就是数据
            rows_iter = [first] + list(reader)

        out_f = out_path.open("w", encoding="utf-8")  # 输出 JSON Lines
        with out_f as fo:
            for raw in rows_iter:
                if not raw or all(not x.strip() for x in raw):
                    continue
                if mode == COARSE:
                    rec = parse_row_coarse(raw)
                    obj = {
                        "id": rec["id"],
                        "text": rec["text"],
                        "image": str((image_base / rec["image_id"]).as_posix()),
                        "caption": rec["caption"],
                        "label": rec["label"],
                    }
                else:
                    rec = parse_row_fine(raw)
                    obj = {
                        "id": rec["id"],
                        "text": rec["text"],
                        "image": str((image_base / rec["image_id"]).as_posix()),
                        "caption": rec["caption"],
                        "aspect": rec["aspect"],
                        "label": rec["label"],
                    }
                fo.write(json.dumps(obj, ensure_ascii=False) + "\n")

def main():
    dataset_name = os.getenv("DATASET_NAME", "mvsa-s").strip().lower()
    split = os.getenv("SPLIT", "train").strip().lower()
    mode = dataset_mode(dataset_name)

    repo_root = Path(__file__).resolve().parent.parent
    data_root = Path(os.getenv("DATA_ROOT", repo_root / "datasets"))
    image_base = data_root / dataset_name / "imgs"
    tsv = data_root / dataset_name / f"{split}.tsv"
    out = data_root / dataset_name / f"{split}.json"
    print(f"[*] dataset={dataset_name} mode={mode}")
    print(f"[*] tsv={tsv}")
    print(f"[*] out={out}")
    print(f"[*] image_base={image_base}")

    convert(tsv, out, image_base, mode)

if __name__ == "__main__":
    main()
