from typing import List

def build_instruction(labels: List[str], use_image: bool = True, has_aspect: bool = True) -> str:
    base = (
        "You are a multimodal sentiment classifier.\n"
        f"Return exactly ONE JSON: {{\"label\": <one of {labels}>}}.\n"
        "No extra words.\n"
    )
    base += "Task: determine the " + ("aspect-based " if has_aspect else "overall ") + "sentiment in the text"
    base += " and image." if use_image else " only."
    return base

def build_user_content(text_s: str, text_a: str | None, has_aspect: bool = True) -> str:
    if has_aspect and text_a:
        return f'Text: {text_s}\nAspect: {text_a}\nReturn JSON only.'
    return f'Text: {text_s}\nReturn JSON only.'
