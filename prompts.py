from typing import List, Literal, Optional

TemplateVariant = Literal[
    "STRICT",           # baseline
    "IMAGE_FIRST",      # image first, then text correction
    "TEXT_FIRST",       # text first, image as evidence
    "CONFLICT_AWARE",   # handling conflicts between text and image
    "SARCASM_AWARE",    # handling sarcasm/negation/expressions in text
]
def build_instruction(
    labels: List[str],
    use_image: bool = True,
    has_aspect: bool = True,
    template_variant: TemplateVariant = "STRICT",
) -> str:
    if template_variant == "STRICT":
        base = (
            "You are a multimodal sentiment classifier.\n"
            f"Return exactly ONE JSON: {{\"label\": <one of {labels}>}}.\n"
            "No extra words.\n"
        )
        base += "Task: determine the " + ("aspect-based " if has_aspect else "overall ") + "sentiment in the text"
        base += " and image." if use_image else " only."
        return base
    head = (
        "You are a multimodal sentiment classifier.\n"
        f"Respond with exactly ONE JSON: {{\"label\": <one of {labels}>}}.\n"
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
                "- If evidence remains insufficient, prefer the most cautious label (e.g., 'neutral' if available).\n"
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
    if has_aspect and text_a:
        return f'Text: {text_s}\nAspect: {text_a}\nReturn JSON only.'
    return f'Text: {text_s}\nReturn JSON only.'
