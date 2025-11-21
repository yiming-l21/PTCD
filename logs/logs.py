from sklearn.metrics import accuracy_score, f1_score
from pathlib import Path
from typing import List, Tuple
import os,json
from collections import OrderedDict, Counter, defaultdict
def _write_rag_debug_and_stats(
    dataset_name, pred_field, samples_meta, demo_diags_by_idx, preds, gts
):
    """
    新版：
    - 写 JSONL：每行包含完整 CD gating 诊断信息，以及 RAG demos 信息（如有）
    - 写 summary JSON：记录整体 acc 和 gating 超参（如 DEMO_TAU_HIGH）
    - 覆盖所有 CD 分析实验所需字段：y0, yD, y_final, c0, cD, delta_c, sim, alpha, z0, zD
    """

    logs_dir = Path(".").resolve() / "logs" / dataset_name
    logs_dir.mkdir(parents=True, exist_ok=True)

    # === 输出文件路径 ===
    dbg_path = logs_dir / f"{pred_field}_cd_debug.jsonl"
    summary_path = logs_dir / f"{pred_field}_cd_summary.json"

    # === 读取 gating 超参（用于 summary 和日后筛选） ===
    tau_high = float(os.getenv("DEMO_TAU_HIGH", "0.6"))
    lambda_sim = float(os.getenv("DEMO_LAMBDA_SIM", "0.2"))
    gamma = float(os.getenv("DEMO_GAMMA", "5.0"))

    # === 写 JSONL：每行 = 一个样本的完整诊断信息 ===
    with dbg_path.open("w", encoding="utf-8") as f:
        for idx in range(len(preds)):
            meta = samples_meta[idx] if idx < len(samples_meta) else {}
            diag = demo_diags_by_idx.get(idx, {})

            # 统一结构 —— 所有字段都写在这一条 record 中
            row = {
                "idx": idx,
                "sample_id": meta.get("id", str(idx)),

                "text": meta.get("text", ""),
                "image": meta.get("image", ""),

                "gold": gts[idx],
                "pred_final": preds[idx],            # 最终 gated 预测

                # Contrastive Decoding 诊断字段（若无 CD 则为空）
                "y0": diag.get("y0", None),
                "yD": diag.get("yD", None),
                "y_final": diag.get("y_final", preds[idx]),

                "c0": diag.get("c0", None),
                "cD": diag.get("cD", None),
                "delta_c": diag.get("delta_c", None),

                "sim": diag.get("sim", None),
                "alpha": diag.get("alpha", None),

                "z0": diag.get("z0", None),
                "zD": diag.get("zD", None),

                # RAG demos 信息（可选）
                "demos": diag.get("rag", {}).get("demos", diag.get("demos", [])),

                # 多模板诊断（如启用）
                "tpl_preds": diag.get("tpl_preds", None),

                # gating 配置（用于更复杂实验）
                "gating_cfg": {
                    "tau_high": tau_high,
                    "lambda_sim": lambda_sim,
                    "gamma": gamma,
                }
            }

            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"[*] Saved CD debug JSONL → {dbg_path}")

    # === 写 Summary JSON ===
    n = len(preds)
    acc_final = sum(p == g for p, g in zip(preds, gts)) / max(n, 1)

    summary = {
        "dataset": dataset_name,
        "pred_field": pred_field,
        "n_samples": n,
        "acc_final": acc_final,
        "gating_cfg": {
            "tau_high": tau_high,
            "lambda_sim": lambda_sim,
            "gamma": gamma,
        },
        "note": "Full CD diagnostics stored in the JSONL file."
    }

    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[*] Saved summary → {summary_path}")


def _resolve_pred_field():
    key = os.getenv("PRED_FIELD", "").strip()
    if key:
        return key
    suf = os.getenv("RUN_SUFFIX", "").strip()
    if not suf:
        suf = os.getenv("PROMPT_VARIANT", "STRICT").strip().upper()
    return f"pred_{suf.lower()}"

def _ensure_logs_dir(dataset_name: str) -> Path:
    root = Path(".").resolve()
    p = root / "logs" / dataset_name
    p.mkdir(parents=True, exist_ok=True)
    return p

def _load_jsonl_as_dict(jsonl_path: Path, key_field: str = "id") -> OrderedDict:
    data = OrderedDict()
    if jsonl_path.exists():
        with jsonl_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if key_field in obj:
                        data[str(obj[key_field])] = obj
                except Exception:
                    pass
    return data

def _dump_dict_as_jsonl(data_dict: OrderedDict, jsonl_path: Path):
    tmp = jsonl_path.with_suffix(".jsonl.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for _, obj in data_dict.items():
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    tmp.replace(jsonl_path)

def _finalize_and_save(preds: List[str], gts: List[str], raws: List[str | Tuple[int, str]],dataset_name: str,samples_meta: List[dict],pred_field: str,):
    suffix = os.getenv("RUN_SUFFIX", "").strip() or os.getenv("PROMPT_VARIANT", "STRICT").strip().upper()

    acc = accuracy_score(gts, preds)
    f1_mac = f1_score(gts, preds, average="macro")
    f1_wtd = f1_score(gts, preds, average="weighted")
    print(f"[Test:{suffix}] size={len(gts)} Acc={acc:.4f} Macro-F1={f1_mac:.4f} Weighted-F1={f1_wtd:.4f}")
    logs_dir = _ensure_logs_dir(dataset_name)
    pred_jsonl = logs_dir / "predictions.jsonl"
    metrics_path = logs_dir / "metrics.json"
    data_map = _load_jsonl_as_dict(pred_jsonl, key_field="id")
    for idx, pred in enumerate(preds):
        meta = samples_meta[idx] if idx < len(samples_meta) else None
        if meta is None:
            meta = {"id": str(idx), "text": "", "image": "", "caption": "", "label": gts[idx] if idx < len(gts) else ""}
        sid = str(meta.get("id", idx))
        if sid not in data_map:
            data_map[sid] = {
                "id": sid,
                "text": meta.get("text", ""),
                "image": meta.get("image", ""),
                "caption": meta.get("caption", ""),
                "label": meta.get("label", ""),
            }
        data_map[sid][pred_field] = pred
    _dump_dict_as_jsonl(data_map, pred_jsonl)
    print(f"[*] saved -> {pred_jsonl.resolve()}")

    metrics = {}
    if metrics_path.exists():
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        except Exception:
            metrics = {}
    metrics[pred_field] = {
        "size": len(gts),
        "accuracy": acc,
        "macro_f1": f1_mac,
        "weighted_f1": f1_wtd,
    }
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"[*] saved -> {metrics_path.resolve()}")