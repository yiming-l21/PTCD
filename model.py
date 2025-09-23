
import re, json, time, random, os
from pathlib import Path
from typing import List

import numpy as np
from rich.progress import track
from sklearn.metrics import accuracy_score, f1_score
import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

from params import build_args, resolve_paths
from dataset import MSADataset
from prompts import build_instruction, build_user_content
from utils import get_labels_and_template

# -------- utils --------
def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_model_and_processor(model_id: str, dtype: str, device_map: str, attn_impl: str,
                             min_pixels: int, max_pixels: int, use_fast_processor: bool = True):
    torch_dtype = {'auto': 'auto','bf16': torch.bfloat16,'fp16': torch.float16,'fp32': torch.float32}[dtype]
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch_dtype, device_map=device_map, attn_implementation=attn_impl,
    )
    processor = AutoProcessor.from_pretrained(
        model_id, min_pixels=min_pixels, max_pixels=max_pixels, use_fast=use_fast_processor
    )
    return model, processor

def _first_json(text: str) -> dict | None:
    m = re.search(r"\{[\s\S]*?\}", text)
    if not m: return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None

def parse_label_from_output(raw: str, label_space: List[str]) -> str:
    data = _first_json(raw)
    if isinstance(data, dict):
        val = data.get('label')
        if isinstance(val, str):
            low = val.strip().lower()
            for cand in label_space:
                if low == cand.lower():
                    return cand
    low_out = raw.lower()
    for cand in label_space:
        if cand.lower() in low_out:
            return cand
    return 'neutral' if any(c.lower()=='neutral' for c in label_space) else label_space[0]

def build_messages(instruction: str, use_image: bool, img_path: str | None, user_text: str):
    system_msg = {"role": "system", "content": instruction}
    if use_image and img_path is not None:
        user_content = [{"type": "image", "image": str(img_path)}, {"type": "text", "text": user_text}]
    else:
        user_content = [{"type": "text", "text": user_text}]
    return [system_msg, {"role": "user", "content": user_content}]

def chunked(it, n):
    it = list(it)
    for i in range(0, len(it), n):
        yield it[i:i+n]


def run_batch(model, processor, messages_list, max_new_tokens: int, temperature: float, top_p: float) -> List[str]:
    texts, images_batch, videos_batch = [], [], []
    for msgs in messages_list:
        t = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(msgs) 
        texts.append(t)
        images_batch.append(image_inputs if image_inputs else None)  
        videos_batch.append(video_inputs if video_inputs else None)  

    has_any_images = any(x is not None for x in images_batch)
    has_any_videos = any(x is not None for x in videos_batch)

    proc_kwargs = dict(
        text=texts,
        padding=True,
        return_tensors='pt',
    )
    if has_any_images:
        proc_kwargs['images'] = images_batch
    if has_any_videos:
        proc_kwargs['videos'] = videos_batch

    inputs = processor(**proc_kwargs).to(model.device)

    do_sample = True if temperature and temperature > 0 else False
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample
        )


    attn = inputs.get('attention_mask', None)
    if attn is None:
        cut = inputs.input_ids.shape[1]
        gens = [out[i][cut:] for i in range(out.shape[0])]
    else:
        in_lens = attn.sum(dim=1)  # [B]
        gens = [out[i][in_lens[i]:] for i in range(out.shape[0])]

    return [
        processor.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        for g in gens
    ]

def run_one(model, processor, messages, max_new_tokens: int, temperature: float, top_p: float) -> str:
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs,
                       padding=True, return_tensors='pt').to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens,
                             temperature=temperature, top_p=top_p)
    gen_ids = out[0][inputs.input_ids.shape[1]:]
    return processor.decode(gen_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

def main():
    args = resolve_paths(build_args())
    set_seed(args.seed)

    dataset_name = Path(args.data_dir).name  # e.g. 'mvsa-s'
    template_id = getattr(args, 'template_id', 2)  
    label_space, label_map, tmpl = get_labels_and_template(dataset_name, template_id)
    # data
    tsv_path = args.data_dir / "test.tsv"
    reader = MSADataset(args, Path(tsv_path), dataset_name=dataset_name)
    samples = reader.read()  # return List[Sample]

    # model
    model, processor = load_model_and_processor(
        model_id=args.model, dtype=args.dtype, device_map=args.device_map, attn_impl=args.attn_impl,
        min_pixels=args.min_pixels, max_pixels=args.max_pixels,
        use_fast_processor=getattr(args, 'use_fast_processor', True)
    )

    # prompt
    use_image_flag = (args.img_dir is not None) and (not args.no_img)
    has_aspect = getattr(args, 'has_aspect', True if 'fine' in str(tmpl).lower() else ('t2015' in dataset_name or 't2017' in dataset_name or 'masad' in dataset_name))
    instruction = build_instruction(labels=label_space, use_image=use_image_flag, has_aspect=has_aspect)
    batch_size = getattr(args, 'batch_size', 8)
    gts, preds = [], []
    gts = [label_map[s.label] for s in samples]
    t0 = time.time()
    for group in track(list(chunked(samples, batch_size)), description='Inferring (batched)'):
        messages_list = []
        meta_idx = []  

        for s in group:
            img_path = None
            if use_image_flag and s.img_id:
                cand = args.img_dir / s.img_id
                if cand.exists():
                    img_path = cand

            user_text = build_user_content(s.text_s, getattr(s, 'text_a', None), has_aspect=has_aspect)
            msgs = build_messages(
                instruction=instruction,
                use_image=use_image_flag and (img_path is not None),
                img_path=str(img_path) if img_path is not None else None,
                user_text=user_text
            )
            messages_list.append(msgs)

        raws = run_batch(
            model, processor, messages_list,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p
        )

        for raw in raws:
            preds.append(parse_label_from_output(raw, label_space))
    acc = accuracy_score(gts, preds)
    f1_mac = f1_score(gts, preds, average='macro')
    f1_wtd = f1_score(gts, preds, average='weighted')
    print(f"[Test] size={len(samples)} Acc={acc:.4f} Macro-F1={f1_mac:.4f} Weighted-F1={f1_wtd:.4f}")

    # save
    out_path = Path('out_qwen2_5_vl_preds.txt')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open('w', encoding='utf-8') as f:
        f.write('#True\t#Pred\n')
        for y, y_ in zip(gts, preds):
            f.write(f'{y}\t{y_}\n')
    print(f"[*] done in {time.time()-t0:.1f}s -> {out_path}")

if __name__ == '__main__':
    main()
