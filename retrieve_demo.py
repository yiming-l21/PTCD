from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
import json
import os

COARSE = "coarse"
FINE   = "fine"

COARSE_DATASETS = {"mvsa-s", "mvsa-m", "tumemo", "tumblr"}
FINE_DATASETS   = {"t2015", "t2017", "masad"}

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
    *,
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
                asp = obj.get("aspect").strip()

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
