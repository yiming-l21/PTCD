from pathlib import Path
from typing import List, Tuple, Optional,Dict
import numpy as np
import json
import os

COARSE = "coarse"
FINE   = "fine"

COARSE_DATASETS = {"mvsa-s", "mvsa-m", "tumemo", "tumblr"}
FINE_DATASETS   = {"t2015", "t2017", "masad"}

def _to_jsonable(x):
    try:
        return float(x)
    except Exception:
        return 0.0

def _balanced_roundrobin_merge(mats: List[np.ndarray]) -> np.ndarray:
    if not mats:
        return np.zeros((0, 0), dtype=np.int64)
    Nq, Kc = mats[0].shape
    C = len(mats)
    out = np.full((Nq, C * Kc), -1, dtype=mats[0].dtype)
    pos = 0
    for r in range(Kc):
        for i in range(C):
            out[:, pos] = mats[i][:, r]
            pos += 1
    return out

def _load_global_index(prefix: Path, topk: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    idx_p = prefix.parent / f"{prefix.name}_top{topk}_idx.npy"
    sim_p = prefix.parent / f"{prefix.name}_top{topk}_sim.npy"
    if idx_p.exists() and sim_p.exists():
        idx = np.load(str(idx_p))
        sim = np.load(str(sim_p))
        print(f"[*] Loaded GLOBAL demo index: {idx_p}  shape={idx.shape}", flush=True)
        return idx, sim
    raise FileNotFoundError(f"global index not found at {idx_p}")

def _find_perclass_npz(prefix: Path) -> Optional[Path]:
    for p in prefix.parent.glob(f"{prefix.name}_perclass_top*.npz"):
        return p
    return None

def _load_perclass_index(prefix: Path) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], List[str]]:
    percls_p = _find_perclass_npz(prefix)
    if percls_p is None:
        raise FileNotFoundError("per-class npz not found")
    z = np.load(str(percls_p), allow_pickle=True)
    classes = list(z["__classes__"].tolist())
    per_idx = {c: z[f"idx::{c}"] for c in classes}
    per_sim = {c: z[f"sim::{c}"] for c in classes}
    print(f"[*] Loaded PER-CLASS demo npz: {percls_p}  classes={classes}", flush=True)
    return per_idx, per_sim, classes

def _load_balanced_index(prefix: Path) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    per_idx, per_sim, classes = _load_perclass_index(prefix)  
    Kc = next(iter(per_idx.values())).shape[1]
    C = len(classes)

    idx_p = prefix.parent / f"{prefix.name}_balanced_roundrobin_top{C}x{Kc}_idx.npy"
    sim_p = prefix.parent / f"{prefix.name}_balanced_roundrobin_top{C}x{Kc}_sim.npy"
    if idx_p.exists() and sim_p.exists():
        bal_idx = np.load(str(idx_p))
        bal_sim = np.load(str(sim_p))
        print(f"[*] Loaded BALANCED demo index: {idx_p}  shape={bal_idx.shape}", flush=True)
        return bal_idx, bal_sim, classes

    mats_idx = [per_idx[c] for c in classes]
    mats_sim = [per_sim[c] for c in classes]
    bal_idx = _balanced_roundrobin_merge(mats_idx).astype(np.int64)
    bal_sim = _balanced_roundrobin_merge(mats_sim).astype(np.float32)
    print(f"[*] Built BALANCED index on-the-fly from per-class npz: shape={bal_idx.shape}", flush=True)
    return bal_idx, bal_sim, classes

def load_offline_demo(prefix: Path, mode: str = "global", global_topk: int = 10):
    mode = mode.strip().lower()
    if mode == "global":
        idx, sim = _load_global_index(prefix, topk=global_topk)
        return idx, sim, {"mode": "global"}
    elif mode == "balanced":
        bal_idx, bal_sim, classes = _load_balanced_index(prefix)
        return bal_idx, bal_sim, {"mode": "balanced", "classes": classes}
    elif mode == "perclass":
        per_idx, per_sim, classes = _load_perclass_index(prefix)
        return per_idx, per_sim, {"mode": "perclass", "classes": classes}
    else:
        raise ValueError(f"Unknown DEMO_MODE={mode}")

def dataset_mode(dataset_name: str) -> str:
    name = (dataset_name or "").strip().lower()
    if name in COARSE_DATASETS:
        return COARSE
    if name in FINE_DATASETS:
        return FINE
    return COARSE

def _replace_placeholder(text: str, aspect: Optional[str]) -> str:
    if not text: return ""
    if not aspect: return text
    return text.replace("$T$", aspect).replace("$t$", aspect)

def _resolve_image_path(img: str, image_base: Optional[Path], make_abs: bool=True) -> str:
    if not img: return ""
    p = Path(img)
    if not p.is_absolute() and image_base is not None:
        p = image_base / img
    if make_abs:
        p = p.resolve()
    return p.as_posix()

def read_train_items(
    p: Path,
    dataset_name: str = "",
    image_base: Optional[Path] = None,
    make_abs_path: bool = True,
    use_aspect_line: bool = False,  
    replace_placeholder: bool = True 
) -> List[dict]:
    items: List[dict] = []
    mode = dataset_mode(dataset_name)
    if not p or not p.exists():
        print(f"[warn] TRAIN_JSONL not found: {p}", flush=True)
        return items
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: 
                continue
            try:
                obj = json.loads(line)
                txt = obj.get("text").strip()
                lbl = obj.get("label").strip()
                img = obj.get("image").strip()
                asp = obj.get("aspect").strip() if mode == FINE else ""
                if mode == FINE:
                    if replace_placeholder and asp:
                        txt = _replace_placeholder(txt, asp)
                    if use_aspect_line and asp:
                        text_for_demo = f"Text: {txt}\nAspect: {asp}\nReturn JSON only."
                    else:
                        text_for_demo = f"Text: {txt}\nReturn JSON only."
                else:
                    text_for_demo = f"Text: {txt}\nReturn JSON only."

                img_path = _resolve_image_path(img, image_base=image_base, make_abs=make_abs_path)
                items.append({
                    "text": text_for_demo,  
                    "label": lbl,
                    "image": img_path,
                    "aspect": asp if mode == FINE else "",
                })
            except Exception as e:
                continue
    return items

def build_demo_messages(
    demos: List[dict],
    *,
    dataset_name: str = "",
) -> List[dict]:
    msgs: List[dict] = []
    for d in demos:
        txt = (d.get("text") or "").strip()
        lbl = d.get("label") or ""
        img = d.get("image") or ""
        # user
        if img:
            user_content = [
                {"type": "image", "image": img},
                {"type": "text",  "text": txt},
            ]
        else:
            user_content = [{"type": "text", "text": txt}]
        assistant_content = [{"type": "text", "text": json.dumps({"label": lbl}, ensure_ascii=False)}]
        msgs.append({"role": "user", "content": user_content})
        msgs.append({"role": "assistant", "content": assistant_content})
    return msgs
