from __future__ import annotations

import os.path as osp
from typing import List, Tuple

import torch
from transformers import AutoTokenizer


def build_text_tokenizer(tokenizer_path: str):
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        special_tokens_file=osp.join(tokenizer_path, "emu3_vision_tokens.txt"),
        trust_remote_code=True,
    )
    tokenizer.bos_token = "<|extra_203|>"
    tokenizer.eos_token = "<|extra_204|>"
    tokenizer.pad_token = "<|endoftext|>"
    tokenizer.eol_token = "<|extra_200|>"
    tokenizer.eof_token = "<|extra_201|>"
    tokenizer.tms_token = "<|extra_202|>"
    tokenizer.img_token = "<|image token|>"
    tokenizer.boi_token = "<|image start|>"
    tokenizer.eoi_token = "<|image end|>"
    tokenizer.bss_token = "<|extra_100|>"
    tokenizer.ess_token = "<|extra_101|>"
    tokenizer.bog_token = "<|extra_60|>"
    tokenizer.eog_token = "<|extra_61|>"
    tokenizer.boc_token = "<|extra_50|>"
    tokenizer.eoc_token = "<|extra_51|>"
    return tokenizer


def build_image_prefix_tokens(tokenizer, height: int, width: int) -> List[int]:
    boi_id = tokenizer.encode(tokenizer.boi_token, add_special_tokens=False)[0]
    img_id = tokenizer.encode(tokenizer.img_token, add_special_tokens=False)[0]
    hw_ids = tokenizer.encode(f"{height}*{width}", add_special_tokens=False)
    return [boi_id, *hw_ids, img_id]


def encode_text_ids(
    tokenizer,
    text_template: str,
    text: str,
    text_max_length: int,
    add_boi: bool,
    height: int,
    width: int,
) -> torch.Tensor:
    prompt = text_template.replace("{text}", text)
    text_ids = tokenizer.encode(prompt, add_special_tokens=False)
    if text_max_length > 0:
        text_ids = text_ids[: text_max_length]
    if add_boi and hasattr(tokenizer, "boi_token"):
        text_ids = list(text_ids) + build_image_prefix_tokens(tokenizer, height, width)
    return torch.tensor(text_ids, dtype=torch.long)


def pad_text_ids(text_ids: List[torch.Tensor], pad_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
    max_len = max(ids.numel() for ids in text_ids)
    padded = []
    mask = []
    for ids in text_ids:
        pad_len = max_len - ids.numel()
        if pad_len > 0:
            pad = torch.full((pad_len,), pad_id, dtype=ids.dtype)
            ids = torch.cat([ids, pad], dim=0)
        padded.append(ids)
        attn = torch.zeros((max_len,), dtype=torch.long)
        attn[: ids.numel() - pad_len] = 1
        mask.append(attn)
    return torch.stack(padded, dim=0), torch.stack(mask, dim=0)

