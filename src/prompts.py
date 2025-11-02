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
) -> str:
    # 把 label list 拼成可读字符串
    label_str = ", ".join(labels)
    if template_variant == "STRICT":
        base = (
            "You are a multimodal sentiment classifier.\n"
            f"Respond with ONE word only: one of [{label_str}].\n"
            "No explanations, no punctuation.\n"
        )
        base += "Task: determine the " + ("aspect-based " if has_aspect else "overall ")
        base += "sentiment in the text and image." if use_image else "sentiment in the text only."
        return base

    # 其余几个模板逻辑保持不变，只把 JSON 语句改掉
    head = (
        "You are a multimodal sentiment classifier.\n"
        f"Respond with ONE word only: one of [{label_str}].\n"
        "Do not add explanations or extra text.\n"
    )
    tail = "Task: " + (
        ("Judge sentiment ONLY toward the given Aspect in the input" if has_aspect
         else "Determine the overall sentiment of the input")
    )
    tail += " (text and image)." if use_image else " (text only)."

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

    if template_variant == "TEXT_FIRST":
        body = (
            "Decision protocol:\n"
            "- Derive sentiment from TEXT primarily (including negation and discourse cues).\n"
            "- Use IMAGE only as supporting evidence if clearly relevant; ignore IMAGE if noisy/irrelevant.\n"
        )
        return head + body + tail

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
                "- IMAGE may be absent; resolve conflict within TEXT (e.g., between literal words and context cues).\n"
                "- Prefer the cautious label when uncertainty persists (e.g., 'neutral' if available).\n"
            )
        if has_aspect:
            body += (
                "- Judge the sentiment ONLY toward the given Aspect; ignore other entities or attributes.\n"
            )
        return head + body + tail

    if template_variant == "SARCASM_AWARE":
        body = (
            "Pragmatics-aware guidelines:\n"
            "- Be sensitive to sarcasm, negation and contrastive wording; positive tokens do not guarantee positive sentiment if used ironically.\n"
            "- Consider emojis/memes in context; textual pragmatics override literal polarity when conflicting.\n"
            "- When uncertain, prefer the cautious label (e.g., 'neutral' if available).\n"
        )
        if has_aspect:
            body += (
                "- Apply these rules ONLY to the given Aspect; ignore sentiment toward other entities.\n"
            )
        return head + body + tail

    return build_instruction(labels, use_image, has_aspect, "STRICT")


def build_user_content(text_s: str, text_a: str | None, has_aspect: bool = True) -> str:
    # 不再要求返回 JSON，而是一个单词
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
