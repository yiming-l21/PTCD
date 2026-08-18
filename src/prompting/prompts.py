from __future__ import annotations 
from typing import List, Literal
import os

TemplateVariant = Literal[
    "STRICT",
    "IMAGE_FIRST",
    "TEXT_FIRST",
    "CONFLICT_AWARE",
    "SARCASM_AWARE",
]

def build_instruction(
    labels: List[str],
    use_image: bool = True,
    has_aspect: bool = True,
    template_variant: TemplateVariant = "STRICT",
    target_mode: str = "token",   
) -> str:

    target_mode = target_mode.lower()
    label_str = ", ".join(labels)

    # --- 输出格式部分（唯一修改点） ---
    if target_mode == "json":
        out_head = (
            "You are a multimodal sentiment classifier.\n"
            f'Return JSON only, with a single field "label" whose value is one of [{label_str}].\n'
            "No explanations, no additional fields.\n"
        )
    else:
        out_head = (
            "You are a multimodal sentiment classifier.\n"
            f"Respond with ONE word only: one of [{label_str}].\n"
            "No explanations, no punctuation.\n"
        )

    # --- STRICT 模板 ---
    if template_variant == "STRICT":
        base = out_head
        base += "Task: determine the "
        base += "aspect-based " if has_aspect else "overall "
        base += "sentiment in the text and image." if use_image else "sentiment in the text only."
        return base

    # -------------- 其他模板 --------------
    # 通用头（和 STRICT 的头不完全一样，保持你之前的结构）
    if target_mode == "json":
        head = (
            "You are a multimodal sentiment classifier.\n"
            f'Return JSON only, with a single field "label" whose value is one of [{label_str}].\n'
            "Do not add explanations or extra text.\n"
        )
    else:
        head = (
            "You are a multimodal sentiment classifier.\n"
            f"Respond with ONE word only: one of [{label_str}].\n"
            "Do not add explanations or extra text.\n"
        )

    tail = "Task: "
    if has_aspect:
        tail += "Judge sentiment ONLY toward the given Aspect in the input"
    else:
        tail += "Determine the overall sentiment of the input"
    tail += " (text and image)." if use_image else " (text only)."

    # ---------- IMAGE_FIRST ----------
    if template_variant == "IMAGE_FIRST":
        if use_image:
            body = (
                "Decision protocol:\n"
                "- Inspect IMAGE evidence first (facial expression, product condition, scene cues).\n"
                "- Use TEXT to confirm or correct the image-based impression.\n"
                "- Prioritize IMAGE when visual cues are explicit; break ties with TEXT.\n"
            )
        else:
            body = (
                "Decision protocol:\n"
                "- IMAGE is unavailable for this sample; rely on TEXT only.\n"
            )
        return head + body + tail

    # ---------- TEXT_FIRST ----------
    if template_variant == "TEXT_FIRST":
        body = (
            "Decision protocol:\n"
            "- Derive sentiment from TEXT primarily (including negation and discourse cues).\n"
            "- Use IMAGE only as supporting evidence if clearly relevant; ignore IMAGE if noisy/irrelevant.\n"
        )
        return head + body + tail

    # ---------- CONFLICT_AWARE ----------
    if template_variant == "CONFLICT_AWARE":
        if use_image:
            body = (
                "Conflict handling:\n"
                "- When TEXT and IMAGE conflict, prioritize TEXT for sarcasm/negation/implicit sentiment.\n"
                "- Use IMAGE only when aligned with the described sentiment or when TEXT is truly ambiguous.\n"
                "- If evidence remains insufficient, prefer the cautious label (e.g., 'neutral' if available).\n"
            )
        else:
            body = (
                "Conflict handling:\n"
                "- IMAGE may be absent; resolve conflict within TEXT.\n"
                "- Prefer the cautious label when uncertainty persists.\n"
            )
        if has_aspect:
            body += "- Judge the sentiment ONLY toward the given Aspect; ignore others.\n"
        return head + body + tail

    # ---------- SARCASM_AWARE ----------
    if template_variant == "SARCASM_AWARE":
        body = (
            "Pragmatics-aware guidelines:\n"
            "- Be sensitive to sarcasm and negation.\n"
            "- Consider emojis/memes in context.\n"
            "- If uncertain, prefer a cautious label.\n"
        )
        if has_aspect:
            body += "- Apply rules ONLY to the given Aspect.\n"
        return head + body + tail

    # fallback
    return build_instruction(labels, use_image, has_aspect, "STRICT", target_mode)


def build_user_content(text_s: str, text_a: str | None, has_aspect: bool = True, target_mode: str = "token") -> str:
    # 不再要求返回 JSON，而是一个单词
    if target_mode == "json":
        if has_aspect and text_a:
            return f'Text: {text_s}\nAspect: {text_a}\nRespond in JSON format with a single field "label".'
        return f'Text: {text_s}\nRespond in JSON format with a single field "label".'
    else:
        if has_aspect and text_a:
            return f'Text: {text_s}\nAspect: {text_a}\nRespond with ONE word only.'
        return f'Text: {text_s}\nRespond with ONE word only.'


def build_prompt_variant():
    single_tpl = os.getenv("PROMPT_VARIANT", "STRICT").strip().upper()
    allowed_tpls = {"STRICT","IMAGE_FIRST","TEXT_FIRST","CONFLICT_AWARE","SARCASM_AWARE"}

    ens_env = os.getenv("PROMPT_ENSEMBLE", "").strip()
    if ens_env:
        prompt_variants = [v.strip().upper() for v in ens_env.split(",") if v.strip()]
    else:
        prompt_variants = [single_tpl]

    prompt_variants = [v for v in prompt_variants if v in allowed_tpls]
    if not prompt_variants:
        prompt_variants = ["STRICT"]

    run_suffix = os.getenv("RUN_SUFFIX", "").strip()
    if not run_suffix:
        run_suffix = ("ENS_" + "-".join(prompt_variants)) if len(prompt_variants) > 1 else prompt_variants[0]
    os.environ["RUN_SUFFIX"] = run_suffix

    print(f"[*] using prompt variants: {prompt_variants}  (suffix={run_suffix})", flush=True)
    return prompt_variants
