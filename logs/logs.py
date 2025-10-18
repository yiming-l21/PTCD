from sklearn.metrics import accuracy_score, f1_score
from pathlib import Path
from typing import List, Tuple
import os,json
from collections import OrderedDict, Counter, defaultdict
def _write_rag_debug_and_stats(dataset_name, pred_field, samples_meta, demo_diags_by_idx, preds, gts):
    logs_dir = Path(".").resolve() / "logs" / dataset_name
    logs_dir.mkdir(parents=True, exist_ok=True)
    dbg_path = logs_dir / f"rag_debug_{pred_field}.jsonl"
    stat_path = logs_dir / f"rag_diagnostics_{pred_field}.json"

    with dbg_path.open("w", encoding="utf-8") as f:
        for idx in range(len(preds)):
            meta = samples_meta[idx] if idx < len(samples_meta) else {}
            diag = demo_diags_by_idx.get(idx, {})
            row = {
                "id": meta.get("id", str(idx)),
                "gold": gts[idx] if idx < len(gts) else "",
                "pred": preds[idx],
                "text": meta.get("text", ""),
                "image": meta.get("image", ""),
                "demos": diag.get("demos", []),
                "tpl_preds": diag.get("tpl_preds", []),
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[*] saved -> {dbg_path}")

    n = len(preds)
    top_demo_label = []
    top_demo_sim = []
    agree_top_demo_pred = 0
    agree_top_demo_gold = 0

    bucket_sims = []  
    bucket_acc  = []  
    bucket_demo_eq_gold = [] 
    bucket_demo_eq_pred = [] 

    for idx in range(n):
        pred_i = preds[idx]
        gold_i = gts[idx]
        demos = demo_diags_by_idx.get(idx, {}).get("demos", [])
        if demos:
            d0 = max(demos, key=lambda d: d.get("sim", 0.0))
            dl, ds = d0.get("label", ""), float(d0.get("sim", 0.0))
        else:
            dl, ds = "", 0.0
        top_demo_label.append(dl)
        top_demo_sim.append(ds)

        if dl == pred_i: agree_top_demo_pred += 1
        if dl == gold_i: agree_top_demo_gold += 1

        bucket_sims.append(ds)
        bucket_acc.append(1 if pred_i == gold_i else 0)
        bucket_demo_eq_gold.append(1 if dl == gold_i else 0)
        bucket_demo_eq_pred.append(1 if dl == pred_i else 0)

    def _cond_acc(mask):
        tot = sum(mask)
        if tot == 0:
            return None
        acc_num = sum(a for a, m in zip(bucket_acc, mask) if m == 1)
        return acc_num / tot

    acc_when_demo_eq_gold    = _cond_acc(bucket_demo_eq_gold)
    acc_when_demo_neq_gold   = _cond_acc([1 - m for m in bucket_demo_eq_gold])
    acc_when_demo_eq_pred    = _cond_acc(bucket_demo_eq_pred)
    acc_when_demo_neq_pred   = _cond_acc([1 - m for m in bucket_demo_eq_pred])
    import math
    bins = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.01]
    bin_names = [f"[{bins[i]},{bins[i+1]})" for i in range(len(bins)-1)]
    bin_stats = {bn: {"count":0, "acc":0.0} for bn in bin_names}
    bin_hits  = {bn: 0 for bn in bin_names}
    for sim, ok in zip(bucket_sims, bucket_acc):
        bn = bin_names[-1]
        for i in range(len(bins)-1):
            if bins[i] <= sim < bins[i+1]:
                bn = bin_names[i]; break
        bin_stats[bn]["count"] += 1
        bin_stats[bn]["acc"]   += ok
        bin_hits[bn] += 1
    for bn in bin_names:
        c = bin_stats[bn]["count"]
        if c > 0:
            bin_stats[bn]["acc"] = bin_stats[bn]["acc"] / c

    stats = {
        "N": n,
        "top_demo_label_agree_with_pred_rate": agree_top_demo_pred / n if n else None,
        "top_demo_label_agree_with_gold_rate": agree_top_demo_gold / n if n else None,
        "acc_when_top_demo_label_equals_gold": acc_when_demo_eq_gold,
        "acc_when_top_demo_label_not_equals_gold": acc_when_demo_neq_gold,
        "acc_when_top_demo_label_equals_pred": acc_when_demo_eq_pred,
        "acc_when_top_demo_label_not_equals_pred": acc_when_demo_neq_pred,
        "top_demo_sim_mean": float(sum(top_demo_sim)/n) if n else None,
        "sim_bins": bin_stats,
        "label_distribution_in_top_demo": dict(Counter([l for l in top_demo_label if l!=""])),
    }
    with stat_path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"[*] saved -> {stat_path}")

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