# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import List, Dict, Any, Tuple
import os
import math

import torch

from src.infer import prepare_inputs_from_messages, prompt_eval_guards


@torch.inference_mode()
def score_labels_for_messages(
    model,
    processor,
    messages: List[Dict[str, Any]],
    label_space: List[str],
    target_mode: str = "json",
) -> torch.Tensor:
    """
    给定一条样本（messages），对每个 label 计算一个 score 向量 z ∈ R^C，
    这里 z[c] ≈ log p(label_c | prompt)，通过 -loss 实现（越大越好）。
    """
    device = model.device
    tok = processor.tokenizer

    # 1) 先把当前 messages 编成 prefix（只含用户 query 和可选图像，不含 label）
    raw_inputs = prepare_inputs_from_messages(processor, messages, device)
    input_ids = raw_inputs["input_ids"]            # [1, T]
    attention_mask = raw_inputs["attention_mask"]  # [1, T]
    B, T = input_ids.shape

    # 视觉 / 其它输入保持不动
    others = {
        k: v for k, v in raw_inputs.items()
        if k not in ("input_ids", "attention_mask")
    }

    # 推断模型浮点 dtype，用于 AMP
    compute_dtype = None
    for p in model.parameters():
        if p is not None and p.is_floating_point():
            compute_dtype = p.dtype
            break
    use_amp = (
        device.type == "cuda"
        and compute_dtype is not None
        and compute_dtype in (torch.float16, torch.bfloat16)
    )

    # 推理时关掉 soft prompt / visual prefix 的 dropout
    row_replacer = getattr(getattr(model, "_prompt_learner", None), "_row_replacer", None)
    vp_core = getattr(getattr(model, "_vp_runtime", None), "core", None)

    def label_to_target(label: str) -> str:
        if target_mode == "token":
            return label
        return f'{{"label": "{label}"}}'

    scores: List[float] = []
    with prompt_eval_guards(row_replacer, vp_core):
        for lbl in label_space:
            tgt_text = label_to_target(lbl)
            tgt_ids_list = tok(tgt_text, add_special_tokens=False)["input_ids"]
            if len(tgt_ids_list) == 0:
                scores.append(float("-inf"))
                continue

            tgt_ids = torch.tensor(
                tgt_ids_list, device=device, dtype=torch.long
            ).unsqueeze(0)  # [1, L]
            L = tgt_ids.size(1)
            input_ids2 = torch.cat([input_ids, tgt_ids], dim=1)  # [1, T+L]
            attn2 = torch.cat(
                [
                    attention_mask,
                    torch.ones((B, L), device=device, dtype=attention_mask.dtype),
                ],
                dim=1,
            )
            labels = torch.full_like(input_ids2, fill_value=-100)
            labels[:, T:T + L] = tgt_ids

            model_inputs: Dict[str, Any] = dict(others)
            model_inputs["input_ids"] = input_ids2
            model_inputs["attention_mask"] = attn2
            model_inputs["labels"] = labels

            if use_amp:
                with torch.autocast(device_type="cuda", dtype=compute_dtype):
                    out = model(**model_inputs)
            else:
                out = model(**model_inputs)

            nll = float(out.loss.detach().cpu().item())
            scores.append(-nll) 
    return torch.tensor(scores, device=device)  # [C]


@torch.inference_mode()
def demo_contrastive_decode(
    model,
    processor,
    base_messages: List[Dict[str, Any]],
    demo_messages: List[Dict[str, Any]],
    label_space: List[str],
    target_mode: str = "json",
) -> Tuple[str, Dict[str, Any]]:
    """
    Demo 级对比解码：
    - base_messages: 不带 demo 的 prompt
    - demo_messages: 带 demo 的 prompt
    返回: (最终预测 label, debug 信息字典)
    """
    device = model.device

    # 超参用 env 控制，方便 ablation
    tau_high = float(os.getenv("DEMO_TAU_HIGH", "0.3"))       # base 很自信时的阈值
    lambda_sim = float(os.getenv("DEMO_LAMBDA_SIM", "0.05"))  # 分布相似度权重
    gamma = float(os.getenv("DEMO_GAMMA", "7.5"))             # sigmoid 尖锐度

    # 1) 分别算「无 demo / 有 demo」两套 label logits
    z0 = score_labels_for_messages(
        model, processor, base_messages, label_space, target_mode=target_mode
    )  # [C]
    zD = score_labels_for_messages(
        model, processor, demo_messages, label_space, target_mode=target_mode
    )  # [C]

    p0 = torch.softmax(z0, dim=-1)
    pD = torch.softmax(zD, dim=-1)

    y0_id = int(p0.argmax().item())
    yD_id = int(pD.argmax().item())
    y0 = label_space[y0_id]
    yD = label_space[yD_id]

    # 置信度：top1 - top2 margin
    def conf(p: torch.Tensor) -> float:
        top2 = torch.topk(p, k=2, dim=-1).values
        return float((top2[0] - top2[1]).item())

    c0 = conf(p0)
    cD = conf(pD)
    delta_c = cD - c0

    # 分布相似度（cosine）
    sim = float(
        torch.nn.functional.cosine_similarity(p0, pD, dim=0).detach().cpu().item()
    )

    # 2) gating：如果 base 已经很自信且 demo 想改 label，直接视为 demo 有害
    if c0 > tau_high and yD_id != y0_id:
        alpha = 0.0
    else:
        score = delta_c + lambda_sim * sim
        alpha = 1.0 / (1.0 + math.exp(-gamma * score))  # sigmoid

    # 3) 论文中的概率分布融合：p_final = (1 - alpha) p0 + alpha pD
    p_final = (1.0 - alpha) * p0 + alpha * pD
    y_final_id = int(p_final.argmax().item())
    y_final = label_space[y_final_id]

    debug = {
        "y0": y0,
        "yD": yD,
        "y_final": y_final,
        "c0": c0,
        "cD": cD,
        "delta_c": delta_c,
        "sim": sim,
        "alpha": alpha,
        "z0": z0.detach().cpu().tolist(),
        "zD": zD.detach().cpu().tolist(),
        "p0": p0.detach().cpu().tolist(),
        "pD": pD.detach().cpu().tolist(),
        "p_final": p_final.detach().cpu().tolist(),
        "fusion": "probability",
    }
    return y_final, debug
