from pathlib import Path
from typing import List, Tuple
import numpy as np
import json

class ExampleBank:
    def __init__(self, train_jsonl_path: Path, train_emb_path: Path, train_img_dir: Path | None):
        self.train_items = self._read_jsonl(train_jsonl_path)
        self.train_emb = np.load(str(train_emb_path))  # (N_train, D)
        self.img_dir = train_img_dir
        assert len(self.train_items) == self.train_emb.shape[0], \
            f"train_jsonl size {len(self.train_items)} != train_emb rows {self.train_emb.shape[0]}"

    @staticmethod
    def _read_jsonl(p: Path):
        items = []
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    txt = (obj.get("text") or obj.get("#3 String") or obj.get("content") or "").strip()
                    lbl = (obj.get("label") or obj.get("#1 Label") or obj.get("sentiment") or "").strip()
                    img = (obj.get("image") or obj.get("img_id") or obj.get("#2 ImageID") or "").strip()
                    items.append({"text": txt, "label": lbl, "img": img})
                except Exception:
                    pass
        return items

    def _img_path(self, img_id: str | None) -> str | None:
        if not img_id or not self.img_dir:
            return None
        cand = self.img_dir / img_id
        return str(cand) if cand.exists() else None

    def topk(self, query_vec: np.ndarray, k: int = 3, avoid_text: str | None = None):
        sims = self.train_emb @ query_vec.astype(np.float32)  # 已 normalize => 点积即 cosine
        idx = np.argpartition(-sims, kth=min(k*4, len(sims)-1))[:k*4]
        idx = idx[np.argsort(-sims[idx])]
        demos = []
        seen_texts = set()
        if avoid_text:
            seen_texts.add(avoid_text.strip())
        for j in idx:
            it = self.train_items[j]
            t = (it.get("text") or "").strip()
            if (not t) or (t in seen_texts):
                continue
            imgp = self._img_path(it.get("img"))
            demos.append({"text": t, "label": (it.get("label") or "").strip(), "image": imgp})
            seen_texts.add(t)
            if len(demos) >= k:
                break
        return demos


def format_fewshot_block(demos: List[dict]) -> str:
    """
    生成简短、明确且不干扰最终 JSON 的 few-shot 块。
    """
    lines = ["Few-shot examples (read only; do NOT explain):"]
    for i, d in enumerate(demos, 1):
        txt = (d.get("text") or "").replace("\n", " ").strip()
        lbl = (d.get("label") or "").strip()
        lines.append(f"{i}) Text: {txt}\n   Expected JSON: {{\"label\": \"{lbl}\"}}")
    return "\n".join(lines)

def read_train_items(p: Path):
        items = []
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
                    txt = (obj.get("text") or obj.get("#3 String") or obj.get("content") or "").strip()
                    lbl = (obj.get("label") or obj.get("#1 Label") or obj.get("sentiment") or "").strip()
                    img = (obj.get("image") or obj.get("img_id") or obj.get("#2 ImageID") or "").strip()
                    items.append({"text": txt, "label": lbl, "image": img})
                except Exception:
                    pass
        return items

def build_demo_messages(demos: List[dict], has_aspect: bool = True) -> List[dict]:
    msgs = []
    for d in demos:
        txt = (d.get("text") or "").strip()
        lbl = (d.get("label") or "").strip()
        img = d.get("image", None)
        if img:
            user_content = [
                {"type": "image", "image": img},
                {"type": "text", "text": f"Text: {txt}\nReturn JSON only."},
            ]
        else:
            user_content = [{"type": "text", "text": f"Text: {txt}\nReturn JSON only."}]
        assistant_content = [{"type": "text", "text": json.dumps({"label": lbl}, ensure_ascii=False)}]

        msgs.append({"role": "user", "content": user_content})
        msgs.append({"role": "assistant", "content": assistant_content})
    return msgs
