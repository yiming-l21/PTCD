# -*- coding: utf-8 -*-
"""
分析 Contrastive Decoding with Demo Gating 的日志。

使用方法：
    python analyze_cd_logs.py \
        --log logs/mvsa-s/pred_demo2_cd_debug.jsonl \
        --out_dir analysis/mvsa-s

依赖：
    - numpy
    - matplotlib
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np
import matplotlib.pyplot as plt


# ===================== 基础 I/O =====================

def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


# ===================== 基础统计：整体 acc / base acc =====================

def compute_overall_acc(records: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    计算 base 和 final 的 overall accuracy。
    要求字段：
      - gold
      - y0 (base 预测)
      - y_final 或 pred_final（最终预测）
    """
    n = 0
    base_correct = 0
    final_correct = 0

    for r in records:
        gold = r.get("gold", None)
        y0 = r.get("y0", None)
        y_final = r.get("y_final", r.get("pred_final", None))

        if gold is None or y_final is None:
            continue

        n += 1
        if y0 is not None and y0 == gold:
            base_correct += 1
        if y_final == gold:
            final_correct += 1

    return {
        "n": n,
        "acc_base": base_correct / n if n > 0 else None,
        "acc_final": final_correct / n if n > 0 else None,
        "delta_acc": (final_correct - base_correct) / n if n > 0 else None,
    }


# ===================== 1) α 分桶分析 =====================

def alpha_bucket_stats(
    records: List[Dict[str, Any]],
    buckets: List[Tuple[float, float]] = None,
) -> List[Dict[str, Any]]:
    """
    对 alpha 分桶，统计：
      - 样本数
      - base acc
      - final acc
      - delta acc
      - 平均 c0, cD, delta_c, sim, alpha
    """

    if buckets is None:
        buckets = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.01)]

    # 每个 bucket 用一个 list 收集记录下标
    bucket_indices = [[] for _ in buckets]

    for idx, r in enumerate(records):
        alpha = r.get("alpha", None)
        if alpha is None:
            continue
        for bi, (lo, hi) in enumerate(buckets):
            if lo <= alpha < hi:
                bucket_indices[bi].append(idx)
                break

    stats = []

    for (lo, hi), indices in zip(buckets, bucket_indices):
        if not indices:
            stats.append(
                {
                    "range": (lo, hi),
                    "n": 0,
                    "acc_base": None,
                    "acc_final": None,
                    "delta_acc": None,
                    "c0_mean": None,
                    "cD_mean": None,
                    "delta_c_mean": None,
                    "sim_mean": None,
                    "alpha_mean": None,
                }
            )
            continue

        base_acc = []
        final_acc = []
        c0_list, cD_list, dc_list, sim_list, a_list = [], [], [], [], []

        for idx in indices:
            r = records[idx]
            gold = r.get("gold", None)
            y0 = r.get("y0", None)
            y_final = r.get("y_final", r.get("pred_final", None))

            if gold is None or y_final is None:
                continue

            base_acc.append(1 if (y0 is not None and y0 == gold) else 0)
            final_acc.append(1 if y_final == gold else 0)

            c0 = r.get("c0", None)
            cD = r.get("cD", None)
            dc = r.get("delta_c", None)
            sim = r.get("sim", None)
            alpha = r.get("alpha", None)

            if c0 is not None:
                c0_list.append(c0)
            if cD is not None:
                cD_list.append(cD)
            if dc is not None:
                dc_list.append(dc)
            if sim is not None:
                sim_list.append(sim)
            if alpha is not None:
                a_list.append(alpha)

        n = len(indices)
        acc_base = np.mean(base_acc) if base_acc else None
        acc_final = np.mean(final_acc) if final_acc else None
        delta_acc = (acc_final - acc_base) if (acc_base is not None and acc_final is not None) else None

        stats.append(
            {
                "range": (lo, hi),
                "n": n,
                "acc_base": acc_base,
                "acc_final": acc_final,
                "delta_acc": delta_acc,
                "c0_mean": np.mean(c0_list) if c0_list else None,
                "cD_mean": np.mean(cD_list) if cD_list else None,
                "delta_c_mean": np.mean(dc_list) if dc_list else None,
                "sim_mean": np.mean(sim_list) if sim_list else None,
                "alpha_mean": np.mean(a_list) if a_list else None,
            }
        )

    return stats


def plot_alpha_buckets(stats: List[Dict[str, Any]], out_dir: Path, title: str = ""):
    """
    画 α 分桶的条形图（delta_acc）和样本数
    """

    out_dir.mkdir(parents=True, exist_ok=True)

    labels = [f"[{s['range'][0]:.1f},{s['range'][1]:.1f})" for s in stats]
    n_vals = [s["n"] for s in stats]
    delta_acc_vals = [s["delta_acc"] if s["delta_acc"] is not None else 0.0 for s in stats]

    x = np.arange(len(labels))

    # 图 1：delta acc
    plt.figure(figsize=(8, 4))
    plt.bar(x, delta_acc_vals)
    plt.xticks(x, labels, rotation=45, ha="right")
    plt.xlabel("Alpha bucket")
    plt.ylabel("Δ Acc (final - base)")
    plt.title(title + " - ΔAcc vs alpha bucket")
    plt.tight_layout()
    plt.savefig(out_dir / "alpha_bucket_delta_acc.png", dpi=200)
    plt.close()

    # 图 2：样本数
    plt.figure(figsize=(8, 4))
    plt.bar(x, n_vals)
    plt.xticks(x, labels, rotation=45, ha="right")
    plt.xlabel("Alpha bucket")
    plt.ylabel("#Samples")
    plt.title(title + " - Sample count vs alpha bucket")
    plt.tight_layout()
    plt.savefig(out_dir / "alpha_bucket_counts.png", dpi=200)
    plt.close()


# ===================== 2) Fixed vs Broken 分析 =====================

def analyze_fixed_broken(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    分析：
      - fixed: base 错，final 对
      - broken: base 对，final 错
    返回每类的：n, alpha_mean, delta_c_mean, sim_mean, c0_mean, cD_mean
    """
    fixed = []
    broken = []

    for r in records:
        gold = r.get("gold", None)
        y0 = r.get("y0", None)
        y_final = r.get("y_final", r.get("pred_final", None))

        if gold is None or y_final is None or y0 is None:
            continue

        if y0 != gold and y_final == gold:
            fixed.append(r)
        elif y0 == gold and y_final != gold:
            broken.append(r)

    def summarize(lst: List[Dict[str, Any]], name: str) -> Dict[str, Any]:
        if not lst:
            return {"name": name, "n": 0}

        alpha_list = [r["alpha"] for r in lst if r.get("alpha", None) is not None]
        dc_list = [r["delta_c"] for r in lst if r.get("delta_c", None) is not None]
        sim_list = [r["sim"] for r in lst if r.get("sim", None) is not None]
        c0_list = [r["c0"] for r in lst if r.get("c0", None) is not None]
        cD_list = [r["cD"] for r in lst if r.get("cD", None) is not None]

        return {
            "name": name,
            "n": len(lst),
            "alpha_mean": float(np.mean(alpha_list)) if alpha_list else None,
            "delta_c_mean": float(np.mean(dc_list)) if dc_list else None,
            "sim_mean": float(np.mean(sim_list)) if sim_list else None,
            "c0_mean": float(np.mean(c0_list)) if c0_list else None,
            "cD_mean": float(np.mean(cD_list)) if cD_list else None,
        }

    return {
        "fixed": summarize(fixed, "fixed"),
        "broken": summarize(broken, "broken"),
    }


# ===================== 3) Δc–sim Heatmap（平均 α） =====================

def delta_sim_heatmap(
    records: List[Dict[str, Any]],
    out_dir: Path,
    title: str = "",
    delta_range: Tuple[float, float] = (-0.5, 0.5),
    sim_range: Tuple[float, float] = (0.0, 1.0),
    n_bins: int = 20,
):
    """
    构建 Δc–sim 的二维网格，颜色为平均 alpha。
    """

    out_dir.mkdir(parents=True, exist_ok=True)

    deltas = []
    sims = []
    alphas = []

    for r in records:
        dc = r.get("delta_c", None)
        sim = r.get("sim", None)
        alpha = r.get("alpha", None)

        if dc is None or sim is None or alpha is None:
            continue
        if not (delta_range[0] <= dc <= delta_range[1]):
            continue
        if not (sim_range[0] <= sim <= sim_range[1]):
            continue

        deltas.append(dc)
        sims.append(sim)
        alphas.append(alpha)

    if not deltas:
        print("[WARN] No valid (delta_c, sim, alpha) triples, skip heatmap.")
        return

    deltas = np.array(deltas)
    sims = np.array(sims)
    alphas = np.array(alphas)

    # 2D binning
    delta_bins = np.linspace(delta_range[0], delta_range[1], n_bins + 1)
    sim_bins = np.linspace(sim_range[0], sim_range[1], n_bins + 1)

    # index per dimension
    delta_idx = np.digitize(deltas, delta_bins) - 1
    sim_idx = np.digitize(sims, sim_bins) - 1

    H = np.zeros((n_bins, n_bins), dtype=np.float32)
    C = np.zeros((n_bins, n_bins), dtype=np.int32)

    for d_i, s_i, a in zip(delta_idx, sim_idx, alphas):
        if 0 <= d_i < n_bins and 0 <= s_i < n_bins:
            H[s_i, d_i] += a
            C[s_i, d_i] += 1

    # avoid division by zero
    mask = C > 0
    H_mean = np.zeros_like(H)
    H_mean[mask] = H[mask] / C[mask]

    plt.figure(figsize=(6, 5))
    extent = [
        delta_range[0],
        delta_range[1],
        sim_range[0],
        sim_range[1],
    ]
    # imshow 中 origin 设为 lower，使 y 轴从下往上增大
    im = plt.imshow(
        H_mean,
        origin="lower",
        extent=extent,
        aspect="auto",
    )
    plt.xlabel("Δc (c_D - c_0)")
    plt.ylabel("sim(p0, pD)")
    plt.title(title + " - mean alpha in Δc-sim space")
    plt.colorbar(im, label="mean alpha")
    plt.tight_layout()
    plt.savefig(out_dir / "delta_sim_alpha_heatmap.png", dpi=200)
    plt.close()


# ===================== CLI and main =====================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=str, required=True, help="Path to *_cd_debug.jsonl")
    parser.add_argument("--out_dir", type=str, required=True, help="Directory to save plots and stats")
    args = parser.parse_args()

    log_path = Path(args.log)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Loading log from {log_path}")
    records = load_jsonl(log_path)
    print(f"[INFO] Loaded {len(records)} records")

    # 1) overall acc
    overall = compute_overall_acc(records)
    print("[OVERALL]")
    print(json.dumps(overall, ensure_ascii=False, indent=2))

    # 2) alpha buckets
    alpha_stats = alpha_bucket_stats(records)
    print("\n[ALPHA BUCKETS]")
    for s in alpha_stats:
        print(
            f"alpha in [{s['range'][0]:.1f},{s['range'][1]:.1f}): "
            f"n={s['n']}, "
            f"acc_base={s['acc_base']}, acc_final={s['acc_final']}, delta_acc={s['delta_acc']}, "
            f"alpha_mean={s['alpha_mean']}, delta_c_mean={s['delta_c_mean']}, sim_mean={s['sim_mean']}"
        )

    plot_alpha_buckets(alpha_stats, out_dir, title=log_path.stem)

    # 3) fixed vs broken
    fb_stats = analyze_fixed_broken(records)
    print("\n[FIXED vs BROKEN]")
    print(json.dumps(fb_stats, ensure_ascii=False, indent=2))

    # 4) Δc-sim heatmap
    delta_sim_heatmap(records, out_dir, title=log_path.stem)

    # 5) 保存所有统计到一个 JSON
    all_stats = {
        "overall": overall,
        "alpha_buckets": alpha_stats,
        "fixed_broken": fb_stats,
    }
    (out_dir / "cd_analysis_stats.json").write_text(
        json.dumps(all_stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n[INFO] Saved stats JSON to {out_dir / 'cd_analysis_stats.json'}")
    print(f"[INFO] Plots saved under {out_dir}")


if __name__ == "__main__":
    main()
