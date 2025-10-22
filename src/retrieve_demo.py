from pathlib import Path
from typing import List, Tuple, Optional,Dict, Any
import numpy as np
import json
import os
from utils import _to_jsonable
COARSE = "coarse"
FINE   = "fine"

COARSE_DATASETS = {"mvsa-s", "mvsa-m", "tumemo", "tumblr"}
FINE_DATASETS   = {"t2015", "t2017", "masad"}

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

class DemoProvider:
    """
    负责：
      - 从 args/env 读取配置并加载离线 demo 索引（global / balanced / perclass）
      - 针对单个样本 row，产出 prefix_demo_msgs 与诊断字典
    依赖：本文件中的 load_offline_demo / read_train_items / build_demo_messages
    """

    def __init__(
        self,
        *,
        use_demo: bool,
        demo_mode: str,
        demo_topk: int,
        train_items: List[Dict],
        demo_index=None,
        demo_sim=None,
        per_idx=None,
        per_sim=None,
        per_classes: Optional[List[str]] = None,
    ):
        self.use_demo = use_demo
        self.demo_mode = demo_mode
        self.demo_topk = demo_topk
        self.train_items = train_items
        self.demo_index = demo_index
        self.demo_sim = demo_sim
        self.per_idx = per_idx
        self.per_sim = per_sim
        self.per_classes = per_classes or []

    @classmethod
    def from_env(cls, args, dataset_name: str, image_base: Path) -> "DemoProvider":
        """
        读取环境变量并加载离线索引：
          - USE_DEMO / DEMO_MODE / DEMO_TOPK / DEMO_EMB_TAG / TRAIN_JSONL
        """
        use_demo: bool = os.getenv("USE_DEMO", "0").strip().lower() in {"1", "true", "yes"}
        demo_topk: int = int(os.getenv("DEMO_TOPK", "3"))
        emb_tag = os.getenv("DEMO_EMB_TAG", "sbert-roberta-large").strip()
        split_name = "test" if "test" in args.tsv else "val"
        train_jsonl_path = Path(os.getenv("TRAIN_JSONL") or getattr(args, "train_jsonl", ""))

        train_items = (
            read_train_items(
                train_jsonl_path,
                dataset_name=dataset_name,
                image_base=image_base,
                make_abs_path=True,
                use_aspect_line=True,
                replace_placeholder=True,
            )
            if use_demo
            else []
        )

        offline_prefix = Path(args.data_dir) / f"{split_name}2train_{emb_tag}"
        demo_mode = os.getenv("DEMO_MODE", "global").strip().lower()

        demo_index = demo_sim = per_idx = per_sim = None
        per_classes = None

        if use_demo:
            try:
                loaded_idx, loaded_sim, meta = load_offline_demo(
                    offline_prefix, mode=demo_mode, global_topk=10
                )
                if meta["mode"] in {"global", "balanced"}:
                    demo_index, demo_sim = loaded_idx, loaded_sim
                    per_classes = meta.get("classes", None)
                elif meta["mode"] == "perclass":
                    per_idx, per_sim = loaded_idx, loaded_sim
                    per_classes = meta.get("classes", None)
                    print(f"[*] per-class demo classes: {per_classes}", flush=True)
                print(f"[*] DEMO_MODE={demo_mode} ready.", flush=True)
            except Exception as e:
                print(
                    f"[warn] USE_DEMO=TRUE but failed to load DEMO indices ({demo_mode}): {e}",
                    flush=True,
                )

        return cls(
            use_demo=use_demo,
            demo_mode=demo_mode,
            demo_topk=demo_topk,
            train_items=train_items,
            demo_index=demo_index,
            demo_sim=demo_sim,
            per_idx=per_idx,
            per_sim=per_sim,
            per_classes=per_classes,
        )

    def for_query(
        self,
        row: int,
        *,
        label_map: Dict[str, str],
        dataset_name: str,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        针对一个样本行号 row，返回：
            - prefix_demo_msgs: List[chat message dict]
            - demo_diag: 诊断信息（写日志用）
        """
        if not (self.use_demo and self.train_items):
            return [], {"demos": []}

        demos: List[Dict[str, Any]] = []
        mode = self.demo_mode

        if mode in {"global", "balanced"} and self.demo_index is not None and self.demo_sim is not None:
            if 0 <= row < self.demo_index.shape[0]:
                ids = self.demo_index[row]
                sims = self.demo_sim[row]
                if mode == "global":
                    k = min(int(self.demo_topk), ids.shape[-1])
                    for m in range(k):
                        self._push_demo(demos, int(ids[m]), sims[m], label_map)
                else:
                    C = len(self.per_classes) or 1
                    Kc_avail = ids.shape[-1] // C
                    per_cls_take = min(int(self.demo_topk), Kc_avail)
                    for r in range(per_cls_take):
                        for cls_i in range(C):
                            pos = cls_i + r * C
                            self._push_demo(demos, int(ids[pos]), sims[pos], label_map)

        elif mode == "perclass" and self.per_idx is not None and self.per_sim is not None and self.per_classes:
            for c in self.per_classes:
                ids_c = self.per_idx[c][row]   # (Kc,)
                sims_c = self.per_sim[c][row]  # (Kc,)
                take = min(int(self.demo_topk), ids_c.shape[-1])
                for r in range(take):
                    self._push_demo(demos, int(ids_c[r]), sims_c[r], label_map)

        prefix_demo_msgs = build_demo_messages(demos, dataset_name=dataset_name) if demos else []
        demo_diag = {
            "demos": [
                {
                    "train_id": d.get("train_id", -1),
                    "label": d.get("label", ""),
                    "text": d.get("text", ""),
                    "image": (d.get("image") or ""),
                    "sim": d.get("sim"),
                }
                for d in demos
            ]
        }
        return prefix_demo_msgs, demo_diag

    def _push_demo(self, demos: List[Dict[str, Any]], j: int, sim_val, label_map: Dict[str, str]):
        if j < 0 or j >= len(self.train_items):
            return
        it = self.train_items[j]
        imgp = None
        cand = it.get("image")
        if cand:
            p = Path(cand)
            if p.exists():
                imgp = p.as_posix()
        demos.append(
            {
                "text": it.get("text", ""),
                "label": label_map.get(it.get("label", ""), it.get("label", "")),
                "image": imgp,
                "sim": _to_jsonable(sim_val),
                "train_id": j,
            }
        )


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
