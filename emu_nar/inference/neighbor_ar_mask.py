# -*- coding: utf-8 -*-
# Build interleaved NAR-style attention masks aligned with NAR-images.

from __future__ import annotations

from typing import List, Optional, Tuple

import torch


def _build_step_id(height: int, width: int, device: torch.device) -> torch.Tensor:
    rows = torch.arange(height, device=device).unsqueeze(1).expand(height, width)
    cols = torch.arange(width, device=device).unsqueeze(0).expand(height, width)
    return (rows + cols).reshape(-1)


def _build_proximity_allow(height: int, width: int, device: torch.device) -> torch.Tensor:
    total = height * width
    allow = torch.zeros((total, total), device=device, dtype=torch.bool)
    previous: List[int] = []
    for c in range(height + width - 1):
        current: List[int] = []
        for h in range(height):
            w = c - h
            if 0 <= w < width:
                idx = h * width + w
                current.append(idx)
                previous.append(idx)
        for idx in current:
            allow[idx, previous] = True
    return allow


def _extract_interleaved_blocks(
    prefix_ids: torch.Tensor,
    img_token_id: int,
    eoi_token_id: int,
    eol_token_id: int,
    visual_token_offset: int,
) -> List[Tuple[List[int], int, int]]:
    seq = prefix_ids.tolist()
    blocks: List[Tuple[List[int], int, int]] = []
    idx = 0
    while idx < len(seq):
        if seq[idx] != img_token_id:
            idx += 1
            continue
        end = idx + 1
        while end < len(seq) and seq[end] != eoi_token_id:
            end += 1
        if end >= len(seq):
            break

        rows: List[List[int]] = []
        current: List[int] = []
        pos = idx + 1
        while pos < end:
            tok = seq[pos]
            if tok == eol_token_id:
                rows.append(current)
                current = []
            elif tok >= visual_token_offset:
                current.append(pos)
            pos += 1
        if current or rows:
            rows.append(current)
        if rows and all(len(r) == len(rows[0]) and len(r) > 0 for r in rows):
            positions = [p for row in rows for p in row]
            blocks.append((positions, len(rows), len(rows[0])))
        idx = end + 1
    return blocks


def _build_interleaved_prefix_allow(
    prefix_ids: torch.Tensor,
    device: torch.device,
    img_token_id: int,
    eoi_token_id: int,
    eol_token_id: int,
    visual_token_offset: int,
) -> torch.Tensor:
    prefix_len = prefix_ids.numel()
    allow = torch.tril(torch.ones(prefix_len, prefix_len, device=device, dtype=torch.bool))
    blocks = _extract_interleaved_blocks(
        prefix_ids=prefix_ids,
        img_token_id=img_token_id,
        eoi_token_id=eoi_token_id,
        eol_token_id=eol_token_id,
        visual_token_offset=visual_token_offset,
    )
    for positions, height, width in blocks:
        if height * width != len(positions):
            continue
        step_id = _build_step_id(height, width, device)
        s_q = step_id[:, None]
        s_k = step_id[None, :]
        block_allow = s_k <= s_q
        pos = torch.tensor(positions, device=device, dtype=torch.long)
        allow[pos[:, None], pos[None, :]] |= block_allow
    return allow


def build_interleaved_neighbor_ar_mask(
    prefix_ids: Optional[torch.Tensor],
    height: int,
    width: int,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
    img_token_id: Optional[int] = None,
    eoi_token_id: Optional[int] = None,
    eol_token_id: Optional[int] = None,
    visual_token_offset: Optional[int] = None,
    text_attention_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create a 4D attention mask for an interleaved prefix (text + previous images)
    plus a new image block decoded with NAR.
    """
    if prefix_ids is None:
        prefix_ids = torch.empty((batch_size, 0), device=device, dtype=torch.long)
    if prefix_ids.dim() == 1:
        prefix_ids = prefix_ids.unsqueeze(0)
    prefix_len = prefix_ids.size(1)

    if prefix_len > 0 and (
        img_token_id is None
        or eoi_token_id is None
        or eol_token_id is None
        or visual_token_offset is None
    ):
        raise ValueError("interleaved prefix requires img/eoi/eol ids and visual_token_offset")

    t_steps = height * width
    total = prefix_len + t_steps
    attn = torch.full((batch_size, 1, total, total), float("-inf"), device=device, dtype=dtype)

    if prefix_len > 0:
        for b in range(batch_size):
            prefix_allow = _build_interleaved_prefix_allow(
                prefix_ids=prefix_ids[b],
                device=device,
                img_token_id=img_token_id,
                eoi_token_id=eoi_token_id,
                eol_token_id=eol_token_id,
                visual_token_offset=visual_token_offset,
            )
            if text_attention_mask is not None:
                valid = text_attention_mask[b].to(device=device).bool()
                invalid = ~valid
                if invalid.any():
                    prefix_allow[invalid, :] = False
                    prefix_allow[:, invalid] = False
                diag = torch.eye(prefix_len, device=device, dtype=torch.bool)
                prefix_allow |= diag
            attn[b, 0, :prefix_len, :prefix_len].masked_fill_(prefix_allow, 0.0)
            attn[b, 0, prefix_len:, :prefix_len] = 0.0

    step_id = _build_step_id(height, width, device)
    allowed = _build_proximity_allow(height, width, device)
    attn[:, 0, prefix_len:, prefix_len:].masked_fill_(allowed, 0.0)
    return attn, step_id


__all__ = ["build_interleaved_neighbor_ar_mask"]
