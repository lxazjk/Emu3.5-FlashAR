# -*- coding: utf-8 -*-
# Build NAR-style proximity attention masks (allow visibility within same step).

from __future__ import annotations

from typing import Optional, Tuple

import torch


def build_neighbor_ar_mask(
    height: int,
    width: int,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create a 4D attention bias mask for NAR proximity attention.
    - Tokens are row-major: index = r * W + c.
    - Step id is wavefront: step = r + c.
    - Allow tokens to attend to all previous steps and the current step.
    Returns:
      attn_mask: [B, 1, T, T] with 0 for allowed and -inf for blocked.
      step_id:  [T] step id per position.
    """
    t_steps = height * width
    rows = torch.arange(height, device=device).unsqueeze(1).expand(height, width)
    cols = torch.arange(width, device=device).unsqueeze(0).expand(height, width)
    step_id = (rows + cols).reshape(-1)  # [T]

    # Allowed if step(k) <= step(q).
    s_q = step_id[:, None]
    s_k = step_id[None, :]
    allowed = s_k <= s_q

    attn = torch.full((batch_size, 1, t_steps, t_steps), float("-inf"), device=device, dtype=dtype)
    attn[:, 0].masked_fill_(allowed, 0.0)
    return attn, step_id


def build_text_neighbor_ar_mask(
    height: int,
    width: int,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
    prefix_len: int,
    text_attention_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create a 4D attention bias mask with a text prefix and NAR image tokens.
    - Text tokens attend causally within text.
    - Image tokens attend to all text tokens and to previous/current image steps.
    Returns:
      attn_mask: [B, 1, P+T, P+T] with 0 for allowed and -inf for blocked.
      step_id:   [T] step id per image position.
    """
    t_steps = height * width
    total = prefix_len + t_steps
    attn = torch.full((batch_size, 1, total, total), float("-inf"), device=device, dtype=dtype)

    if prefix_len > 0:
        text_allowed = torch.tril(torch.ones(prefix_len, prefix_len, device=device, dtype=torch.bool))
        attn[:, 0, :prefix_len, :prefix_len].masked_fill_(text_allowed, 0.0)
        attn[:, 0, prefix_len:, :prefix_len] = 0.0

        if text_attention_mask is not None:
            text_valid = text_attention_mask.to(device=device).bool()
            invalid_keys = ~text_valid
            if invalid_keys.any():
                attn[:, 0, :prefix_len, :prefix_len].masked_fill_(invalid_keys[:, None, :], float("-inf"))
                attn[:, 0, prefix_len:, :prefix_len].masked_fill_(invalid_keys[:, None, :], float("-inf"))
                attn[:, 0, :prefix_len, :prefix_len].masked_fill_(invalid_keys[:, :, None], float("-inf"))
            diag = torch.eye(prefix_len, device=device, dtype=torch.bool).unsqueeze(0)
            attn[:, 0, :prefix_len, :prefix_len].masked_fill_(diag, 0.0)

    rows = torch.arange(height, device=device).unsqueeze(1).expand(height, width)
    cols = torch.arange(width, device=device).unsqueeze(0).expand(height, width)
    step_id = (rows + cols).reshape(-1)  # [T]

    s_q = step_id[:, None]
    s_k = step_id[None, :]
    allowed = s_k <= s_q
    attn[:, 0, prefix_len:, prefix_len:].masked_fill_(allowed, 0.0)
    return attn, step_id
