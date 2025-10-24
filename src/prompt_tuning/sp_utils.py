# -*- coding: utf-8 -*-
from typing import List, Tuple
from transformers import PreTrainedTokenizerBase, PreTrainedModel

def init_soft_tokens(tokenizer: PreTrainedTokenizerBase,
                     model: PreTrainedModel,
                     n_tokens: int) -> Tuple[List[str], List[int]]:
    """
    在 tokenizer 中注册 <soft0>..<soft{n-1}>, 并 resize 模型 embedding。
    返回 (token_strs, token_ids)。
    若已注册，重复调用是幂等的。
    """
    if n_tokens <= 0:
        return [], []
    soft_tokens = [f"<soft{i}>" for i in range(n_tokens)] 
    added_vocab = tokenizer.get_added_vocab()
    to_add = [t for t in soft_tokens if t not in added_vocab]
    if to_add:
        tokenizer.add_special_tokens({"additional_special_tokens": to_add})
        model.resize_token_embeddings(len(tokenizer))
    ids = tokenizer.convert_tokens_to_ids(soft_tokens)
    return soft_tokens, ids

def soft_string(soft_tokens: List[str]) -> str:
    return "".join(soft_tokens)