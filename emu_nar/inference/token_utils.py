from __future__ import annotations

from typing import List

import torch


def get_special_ids(cfg, tokenizer):
    if hasattr(cfg, "special_token_ids"):
        return cfg.special_token_ids
    return {key: tokenizer.encode(value)[0] for key, value in cfg.special_tokens.items()}


def get_digit_token_ids(tokenizer) -> set[int]:
    ids = set()
    for token in [str(i) for i in range(10)] + ["*"]:
        ids.add(tokenizer.encode(token, add_special_tokens=False)[0])
    return ids


def sample_next_token(logits, temperature, top_k, top_p):
    logits = logits / max(temperature, 1e-5)
    if top_k > 0:
        top_k = min(top_k, logits.size(-1))
        cutoff = torch.topk(logits, top_k)[0][..., -1, None]
        logits = torch.where(logits < cutoff, torch.full_like(logits, -float("inf")), logits)
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits = logits.masked_fill(indices_to_remove, -float("inf"))
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


def parse_hw_from_tokens(tokenizer, hw_tokens: List[int]) -> tuple[int, int]:
    hw_str = tokenizer.decode(hw_tokens)
    h_str, w_str = hw_str.split("*")
    return int(h_str), int(w_str)


def collect_hw_tokens(seq: List[int], boi_id: int, img_id: int) -> List[int]:
    last_boi = len(seq) - 1 - seq[::-1].index(boi_id)
    last_img = len(seq) - 1 - seq[::-1].index(img_id)
    return seq[last_boi + 1 : last_img]
