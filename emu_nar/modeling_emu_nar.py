# -*- coding: utf-8 -*-
"""
Non-Autoregressive (NAR) wrapper for Emu3.5.

Combines proximity attention masking with horizontal/vertical prediction heads
to enable diagonal-step parallel decoding of image tokens.

Usage:
    model = EmuNAR(backbone.model, vocab_size=..., hidden_size=...)
    # Training:
    out = model(input_ids=image_ids, height=H, width=W, text_input_ids=text_ids)
    loss = out["loss"]
    # Generation:
    grid = model.generate(height=H, width=W, device=device, text_input_ids=text_ids)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F


# ============================================================================
# Sampling helpers
# ============================================================================

def _top_k_top_p_filtering(
    logits: torch.Tensor,
    top_k: int = 0,
    top_p: float = 1.0,
    filter_value: float = -float("Inf"),
    min_tokens_to_keep: int = 1,
) -> torch.Tensor:
    if top_k > 0:
        top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        if min_tokens_to_keep > 1:
            sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices_to_remove.scatter(
            1, sorted_indices, sorted_indices_to_remove
        )
        logits[indices_to_remove] = filter_value
    return logits


def _sample_logits(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    sample_logits: bool = True,
) -> torch.Tensor:
    idx = []
    for i in range(logits.shape[1]):
        one_logits = logits[:, i, :] / max(temperature, 1e-5)
        if top_k > 0 or top_p < 1.0:
            one_logits = _top_k_top_p_filtering(one_logits, top_k=top_k, top_p=top_p)
        probs = F.softmax(one_logits, dim=-1)
        if sample_logits:
            one_idx = torch.multinomial(probs, num_samples=1)
        else:
            _, one_idx = torch.topk(probs, k=1, dim=-1)
        idx.append(one_idx.view(-1, 1))
    return torch.cat(idx, dim=-1)


# ============================================================================
# Proximity attention mask
# ============================================================================

def _build_step_id(height: int, width: int, device: torch.device) -> torch.Tensor:
    """Manhattan-distance step id for each position in the H×W grid."""
    rows = torch.arange(height, device=device).unsqueeze(1).expand(height, width)
    cols = torch.arange(width, device=device).unsqueeze(0).expand(height, width)
    return (rows + cols).reshape(-1)


def _build_proximity_allow(
    height: int, width: int, device: torch.device
) -> torch.Tensor:
    """Boolean allow matrix: allow[q, k] = True iff k can be attended by q."""
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
    """Find complete rectangular image blocks inside a prefix sequence."""
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
    """Build allow matrix for a text+image prefix: causal for text, proximity for images."""
    prefix_len = prefix_ids.numel()
    allow = torch.tril(
        torch.ones(prefix_len, prefix_len, device=device, dtype=torch.bool)
    )
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
        block_allow = step_id[None, :] <= step_id[:, None]
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
    Build a 4-D attention mask for [prefix | new_image].

    Returns:
        attn_mask: (B, 1, T, T)  float mask (0 = attend, -inf = block)
        step_id:   (H*W,)  diagonal step index for the new image block
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
        raise ValueError(
            "interleaved prefix requires img/eoi/eol ids and visual_token_offset"
        )

    t_steps = height * width
    total = prefix_len + t_steps
    attn = torch.full(
        (batch_size, 1, total, total), float("-inf"), device=device, dtype=dtype
    )

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
            # new image attends to valid prefix positions only
            if text_attention_mask is not None:
                valid = text_attention_mask[b].to(device=device).bool()
                attn[b, 0, prefix_len:, :prefix_len] = torch.where(
                    valid.unsqueeze(0), 0.0, float("-inf")
                )
            else:
                attn[b, 0, prefix_len:, :prefix_len] = 0.0

    step_id = _build_step_id(height, width, device)
    allowed = _build_proximity_allow(height, width, device)
    attn[:, 0, prefix_len:, prefix_len:].masked_fill_(allowed, 0.0)
    return attn, step_id


# ============================================================================
# EmuNAR
# ============================================================================

class EmuNAR(nn.Module):
    """
    Wrap an AR backbone for NAR training and diagonal-step decoding.

    Architecture:
        backbone  →  hidden states  →  horizontal_head  →  h_logits (right neighbor)
                                    →  [vertical_block] → vertical_head → v_logits (down neighbor)
        fused_logits = shift-and-merge(h_logits, v_logits)
    """

    def __init__(
        self,
        pretrained_backbone: nn.Module,
        vocab_size: int,
        hidden_size: int,
        num_heads: int = 8,
        ff_mult: int = 4,
        dropout: float = 0.1,
        pad_token_id: int = -100,
        mask_token_id: Optional[int] = None,
        visual_token_offset: Optional[int] = None,
        use_vertical_block: bool = True,
        vertical_layers: int = 1,
        lm_head: Optional[nn.Linear] = None,
        img_token_id: Optional[int] = None,
        eol_token_id: Optional[int] = None,
        eoi_token_id: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.backbone = pretrained_backbone
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.pad_token_id = pad_token_id
        self.mask_token_id = (
            mask_token_id if mask_token_id is not None else pad_token_id
        )
        self.visual_token_offset = visual_token_offset
        self.use_vertical_block = use_vertical_block

        cfg = getattr(pretrained_backbone, "config", None)
        self.img_token_id = (
            img_token_id
            if img_token_id is not None
            else getattr(cfg, "img_token_id", None)
        )
        self.eol_token_id = (
            eol_token_id
            if eol_token_id is not None
            else getattr(cfg, "eol_token_id", None)
        )
        self.eoi_token_id = (
            eoi_token_id
            if eoi_token_id is not None
            else getattr(cfg, "eoi_token_id", None)
        )

        # --- prediction heads ---
        self.horizontal_head = nn.Linear(hidden_size, vocab_size)

        if use_vertical_block:
            layers = max(1, int(vertical_layers))
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=num_heads,
                dim_feedforward=hidden_size * ff_mult,
                dropout=dropout,
                batch_first=True,
                activation="gelu",
            )
            if layers == 1:
                self.vertical_block = encoder_layer
                self._vertical_uses_encoder = False
            else:
                self.vertical_block = nn.TransformerEncoder(
                    encoder_layer, num_layers=layers
                )
                self._vertical_uses_encoder = True
            self.vertical_norm = nn.LayerNorm(hidden_size)
        else:
            self.vertical_block = None
            self.vertical_norm = None
            self._vertical_uses_encoder = False

        self.vertical_head = nn.Linear(hidden_size, vocab_size)

        # initialise heads from lm_head if provided
        if lm_head is not None:
            with torch.no_grad():
                self.horizontal_head.weight.copy_(lm_head.weight)
                self.vertical_head.weight.copy_(lm_head.weight)

        self._sync_dtype_with_backbone()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _sync_dtype_with_backbone(self) -> None:
        try:
            backbone_dtype = next(self.backbone.parameters()).dtype
        except StopIteration:
            return
        self.horizontal_head.to(dtype=backbone_dtype)
        if self.vertical_block is not None:
            self.vertical_block.to(dtype=backbone_dtype)
        if self.vertical_norm is not None:
            self.vertical_norm.to(dtype=backbone_dtype)
        self.vertical_head.to(dtype=backbone_dtype)

    def _reshape_grid(
        self, seq: torch.Tensor, height: int, width: int
    ) -> torch.Tensor:
        bsz, seq_len = seq.shape[:2]
        if seq_len != height * width:
            raise ValueError(f"Expected seq_len={height * width}, got {seq_len}")
        return seq.view(bsz, height, width, -1)

    @staticmethod
    def _build_step_mask_2d(step_id: torch.Tensor) -> torch.Tensor:
        """Bool mask for TransformerEncoder: True = blocked."""
        allowed = step_id[None, :] <= step_id[:, None]
        return ~allowed

    def _fuse_logits(
        self, h_logits: torch.Tensor, v_logits: torch.Tensor
    ) -> torch.Tensor:
        """
        Shift-and-merge horizontal / vertical logits into per-position predictions.

        h_logits[b, h, w] predicts position (h, w+1);  after roll → position (h, w).
        v_logits[b, h, w] predicts position (h+1, w);  after roll → position (h, w).
        """
        cond_logits = (h_logits[:, 0, 0, :] + v_logits[:, 0, 0, :]) / 2
        h_shift = h_logits.roll(shifts=1, dims=2)   # prediction for right neighbor
        v_shift = v_logits.roll(shifts=1, dims=1)   # prediction for bottom neighbor
        fused = torch.zeros_like(v_shift)
        fused[:, 0, 0, :] = cond_logits              # corner: average
        fused[:, 0, 1:, :] = h_shift[:, 0, 1:, :]   # first row: horizontal only
        fused[:, 1:, 0, :] = v_shift[:, 1:, 0, :]   # first col: vertical only
        fused[:, 1:, 1:, :] = (                      # interior: average both
            h_shift[:, 1:, 1:, :] + v_shift[:, 1:, 1:, :]
        ) / 2
        return fused

    def _apply_vertical_block(
        self, image_hidden: torch.Tensor, step_mask_2d: torch.Tensor
    ) -> torch.Tensor:
        if self.vertical_block is not None:
            if self._vertical_uses_encoder:
                v_hidden = self.vertical_block(image_hidden, mask=step_mask_2d)
            else:
                v_hidden = self.vertical_block(image_hidden, src_mask=step_mask_2d)
            return self.vertical_norm(v_hidden)
        return image_hidden

    def _compute_logits(
        self,
        image_hidden: torch.Tensor,
        height: int,
        width: int,
        step_mask_2d: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        h_grid = self._reshape_grid(image_hidden, height, width)
        h_logits = self.horizontal_head(h_grid)

        v_hidden = self._apply_vertical_block(image_hidden, step_mask_2d)
        v_grid = self._reshape_grid(v_hidden, height, width)
        v_logits = self.vertical_head(v_grid)

        fused = self._fuse_logits(h_logits, v_logits)
        return {"fused": fused, "h_logits": h_logits, "v_logits": v_logits}

    def _run_backbone(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
            use_cache=False,
        )
        return outputs.last_hidden_state

    # ------------------------------------------------------------------
    # Mask helpers (public, for external callers)
    # ------------------------------------------------------------------

    def build_mask(
        self,
        image_ids: torch.Tensor,
        height: int,
        width: int,
        text_input_ids: Optional[torch.Tensor] = None,
        text_attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, int, torch.Tensor, torch.Tensor]:
        """
        Build full input ids, prefix length, 4-D attention mask, and step mask.
        """
        mask_dtype = next(self.backbone.parameters()).dtype
        if text_input_ids is not None:
            if text_input_ids.dim() == 1:
                text_input_ids = text_input_ids.unsqueeze(0)
            prefix_len = text_input_ids.size(1)
            full_input_ids = torch.cat([text_input_ids, image_ids], dim=1)
            attn_mask, step_id = build_interleaved_neighbor_ar_mask(
                prefix_ids=text_input_ids,
                height=height,
                width=width,
                batch_size=image_ids.size(0),
                device=image_ids.device,
                dtype=mask_dtype,
                img_token_id=self.img_token_id,
                eoi_token_id=self.eoi_token_id,
                eol_token_id=self.eol_token_id,
                visual_token_offset=self.visual_token_offset,
                text_attention_mask=text_attention_mask,
            )
        else:
            prefix_len = 0
            full_input_ids = image_ids
            attn_mask, step_id = build_interleaved_neighbor_ar_mask(
                prefix_ids=None,
                height=height,
                width=width,
                batch_size=image_ids.size(0),
                device=image_ids.device,
                dtype=mask_dtype,
            )
        step_mask_2d = self._build_step_mask_2d(step_id)
        return full_input_ids, prefix_len, attn_mask, step_mask_2d

    # ------------------------------------------------------------------
    # Forward (training)
    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids: torch.Tensor,
        height: int,
        width: int,
        text_input_ids: Optional[torch.Tensor] = None,
        text_attention_mask: Optional[torch.Tensor] = None,
        step_positions: Optional[torch.Tensor] = None,
        prev_positions: Optional[torch.Tensor] = None,
        chunked_loss: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute fused NAR loss.

        Args:
            input_ids:  (B, H*W)  flattened image token ids.
            height, width: spatial dimensions of the image grid.
            text_input_ids:  (B, T) optional text prefix.
            text_attention_mask: (B, T) optional padding mask for prefix.
            step_positions: 1-D positions to predict (generation mode).
            prev_positions: 1-D positions whose hidden states are used.
            chunked_loss: compute loss row-by-row to save memory.
        """
        bsz = input_ids.size(0)

        # --- build mask ---
        full_input_ids, prefix_len, attn_mask, step_mask_2d = self.build_mask(
            image_ids=input_ids,
            height=height,
            width=width,
            text_input_ids=text_input_ids,
            text_attention_mask=text_attention_mask,
        )

        # --- backbone forward ---
        hidden = self._run_backbone(full_input_ids, attn_mask)
        image_hidden = hidden[:, prefix_len:, :] if prefix_len > 0 else hidden

        # === step-based generation path ===
        if step_positions is not None and prev_positions is not None:
            return self._forward_step(
                image_hidden, step_mask_2d, step_positions, prev_positions,
                height, width, input_ids.device,
            )

        # === chunked loss path ===
        if chunked_loss:
            return self._forward_chunked(
                image_hidden, step_mask_2d, input_ids, height, width,
            )

        # === full loss path ===
        return self._forward_full(
            image_hidden, step_mask_2d, input_ids, height, width,
        )

    # ------------------------------------------------------------------
    # Forward sub-paths
    # ------------------------------------------------------------------

    def _forward_step(
        self,
        image_hidden: torch.Tensor,
        step_mask_2d: torch.Tensor,
        step_positions: torch.Tensor,
        prev_positions: torch.Tensor,
        height: int,
        width: int,
        device: torch.device,
    ) -> Dict[str, torch.Tensor]:
        """Per-step logits for generation (no loss)."""
        bsz = image_hidden.size(0)
        if step_positions.dim() != 1 or prev_positions.dim() != 1:
            raise ValueError("step_positions and prev_positions must be 1-D.")
        if step_positions.numel() == 0:
            return {
                "step_logits": torch.empty(
                    (bsz, 0, self.vocab_size), device=device
                )
            }

        v_hidden = self._apply_vertical_block(image_hidden, step_mask_2d)

        h_prev = self.horizontal_head(image_hidden[:, prev_positions, :])
        v_prev = self.vertical_head(v_hidden[:, prev_positions, :])

        total = height * width
        pos_to_idx = torch.full((total,), -1, device=device, dtype=torch.long)
        pos_to_idx[prev_positions] = torch.arange(
            prev_positions.numel(), device=device
        )

        rows = step_positions // width
        cols = step_positions % width
        step_logits = torch.empty(
            (bsz, step_positions.numel(), self.vocab_size),
            device=device,
            dtype=h_prev.dtype,
        )

        left_mask = cols > 0
        up_mask = rows > 0
        both_mask = left_mask & up_mask
        corner_mask = ~left_mask & ~up_mask

        # first-row non-corner: horizontal only
        h_only = left_mask & ~up_mask
        if h_only.any():
            left_idx = pos_to_idx[step_positions[h_only] - 1]
            step_logits[:, h_only, :] = h_prev[:, left_idx, :]

        # first-col non-corner: vertical only
        v_only = up_mask & ~left_mask
        if v_only.any():
            up_idx = pos_to_idx[step_positions[v_only] - width]
            step_logits[:, v_only, :] = v_prev[:, up_idx, :]

        # interior: average
        if both_mask.any():
            left_idx = pos_to_idx[step_positions[both_mask] - 1]
            up_idx = pos_to_idx[step_positions[both_mask] - width]
            step_logits[:, both_mask, :] = (
                h_prev[:, left_idx, :] + v_prev[:, up_idx, :]
            ) / 2

        # corner (0,0): average of self
        if corner_mask.any():
            idx = pos_to_idx[step_positions[corner_mask]]
            step_logits[:, corner_mask, :] = (
                h_prev[:, idx, :] + v_prev[:, idx, :]
            ) / 2

        return {"step_logits": step_logits}

    def _forward_chunked(
        self,
        image_hidden: torch.Tensor,
        step_mask_2d: torch.Tensor,
        input_ids: torch.Tensor,
        height: int,
        width: int,
    ) -> Dict[str, torch.Tensor]:
        """Row-by-row loss to save memory."""
        bsz = input_ids.size(0)
        device = input_ids.device
        target_grid = input_ids.view(bsz, height, width)

        v_hidden = self._apply_vertical_block(image_hidden, step_mask_2d)
        h_grid = image_hidden.view(bsz, height, width, -1)
        v_grid = v_hidden.view(bsz, height, width, -1)

        loss_sum = torch.tensor(0.0, device=device)
        loss_h_sum = torch.tensor(0.0, device=device)
        loss_v_sum = torch.tensor(0.0, device=device)
        v_prev = None

        for r in range(height):
            h_logits_row = self.horizontal_head(h_grid[:, r, :, :])
            v_logits_row = self.vertical_head(v_grid[:, r, :, :])

            fused_row = torch.empty_like(h_logits_row)
            if r == 0:
                fused_row[:, 0, :] = (
                    h_logits_row[:, 0, :] + v_logits_row[:, 0, :]
                ) / 2
                if width > 1:
                    fused_row[:, 1:, :] = h_logits_row[:, :-1, :]
            else:
                fused_row[:, 0, :] = v_prev[:, 0, :]
                if width > 1:
                    fused_row[:, 1:, :] = (
                        h_logits_row[:, :-1, :] + v_prev[:, 1:, :]
                    ) / 2

            loss_sum += F.cross_entropy(
                fused_row.reshape(-1, self.vocab_size),
                target_grid[:, r, :].reshape(-1),
                ignore_index=self.pad_token_id,
                reduction="sum",
            )
            if width > 1:
                loss_h_sum += F.cross_entropy(
                    h_logits_row[:, :-1, :].reshape(-1, self.vocab_size),
                    target_grid[:, r, 1:].reshape(-1),
                    ignore_index=self.pad_token_id,
                    reduction="sum",
                )
            if r > 0:
                loss_v_sum += F.cross_entropy(
                    v_prev.reshape(-1, self.vocab_size),
                    target_grid[:, r, :].reshape(-1),
                    ignore_index=self.pad_token_id,
                    reduction="sum",
                )
            v_prev = v_logits_row

        denom = float(bsz * height * width)
        loss = loss_sum / denom
        loss_h = (
            loss_h_sum / float(bsz * height * (width - 1))
            if width > 1
            else torch.tensor(0.0, device=device, dtype=loss.dtype)
        )
        loss_v = (
            loss_v_sum / float(bsz * (height - 1) * width)
            if height > 1
            else torch.tensor(0.0, device=device, dtype=loss.dtype)
        )
        return {"loss": loss, "loss_h": loss_h, "loss_v": loss_v}

    def _forward_full(
        self,
        image_hidden: torch.Tensor,
        step_mask_2d: torch.Tensor,
        input_ids: torch.Tensor,
        height: int,
        width: int,
    ) -> Dict[str, torch.Tensor]:
        """Standard full-grid loss."""
        bsz = input_ids.size(0)
        device = input_ids.device
        logits = self._compute_logits(image_hidden, height, width, step_mask_2d)
        fused = logits["fused"]
        h_logits = logits["h_logits"]
        v_logits = logits["v_logits"]

        target_grid = input_ids.view(bsz, height, width)
        loss = F.cross_entropy(
            fused.reshape(-1, self.vocab_size),
            target_grid.reshape(-1),
            ignore_index=self.pad_token_id,
        )

        if width > 1:
            loss_h = F.cross_entropy(
                h_logits[:, :, :-1, :].reshape(-1, self.vocab_size),
                target_grid[:, :, 1:].reshape(-1),
                ignore_index=self.pad_token_id,
            )
        else:
            loss_h = torch.tensor(0.0, device=device, dtype=loss.dtype)

        if height > 1:
            loss_v = F.cross_entropy(
                v_logits[:, :-1, :, :].reshape(-1, self.vocab_size),
                target_grid[:, 1:, :].reshape(-1),
                ignore_index=self.pad_token_id,
            )
        else:
            loss_v = torch.tensor(0.0, device=device, dtype=loss.dtype)

        return {
            "loss": loss,
            "loss_h": loss_h,
            "loss_v": loss_v,
            "logits_h": h_logits,
            "logits_v": v_logits,
            "logits": fused,
        }

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        height: int,
        width: int,
        device: torch.device,
        text_input_ids: Optional[torch.Tensor] = None,
        unconditional_text_input_ids: Optional[torch.Tensor] = None,
        cfg_scale: float = 1.0,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        sample_logits: bool = True,
    ) -> torch.Tensor:
        """
        Diagonal-step decoding.

        Returns:
            (H, W) int64 token grid.
        """
        grid = torch.full(
            (1, height, width), self.mask_token_id, device=device, dtype=torch.long
        )
        mask_dtype = next(self.backbone.parameters()).dtype

        # --- conditional mask ---
        if text_input_ids is not None:
            if text_input_ids.dim() == 1:
                text_input_ids = text_input_ids.unsqueeze(0)
            prefix_len = text_input_ids.size(1)
            attn_mask, step_id = build_interleaved_neighbor_ar_mask(
                prefix_ids=text_input_ids,
                height=height, width=width, batch_size=1,
                device=device, dtype=mask_dtype,
                img_token_id=self.img_token_id,
                eoi_token_id=self.eoi_token_id,
                eol_token_id=self.eol_token_id,
                visual_token_offset=self.visual_token_offset,
            )
        else:
            prefix_len = 0
            attn_mask, step_id = build_interleaved_neighbor_ar_mask(
                prefix_ids=None,
                height=height, width=width, batch_size=1,
                device=device, dtype=mask_dtype,
            )

        # --- unconditional mask (for CFG) ---
        if unconditional_text_input_ids is not None:
            if unconditional_text_input_ids.dim() == 1:
                unconditional_text_input_ids = unconditional_text_input_ids.unsqueeze(0)
            u_prefix_len = unconditional_text_input_ids.size(1)
            u_attn_mask, _ = build_interleaved_neighbor_ar_mask(
                prefix_ids=unconditional_text_input_ids,
                height=height, width=width, batch_size=1,
                device=device, dtype=mask_dtype,
                img_token_id=self.img_token_id,
                eoi_token_id=self.eoi_token_id,
                eol_token_id=self.eol_token_id,
                visual_token_offset=self.visual_token_offset,
            )
        else:
            u_prefix_len = 0
            u_attn_mask = None

        step_mask_2d = self._build_step_mask_2d(step_id)
        max_step = int(step_id.max().item())

        for step in range(0, max_step + 1):
            positions = (step_id == step).nonzero(as_tuple=False).view(-1)
            if positions.numel() == 0:
                continue

            image_ids = grid.view(1, -1)
            if text_input_ids is not None:
                full_ids = torch.cat([text_input_ids, image_ids], dim=1)
            else:
                full_ids = image_ids

            hidden = self._run_backbone(full_ids, attn_mask)
            image_hidden = (
                hidden[:, prefix_len:, :] if prefix_len > 0 else hidden
            )
            logits = self._compute_logits(
                image_hidden, height, width, step_mask_2d
            )
            fused = logits["fused"].reshape(1, -1, self.vocab_size)
            step_logits = fused[:, positions, :]

            # classifier-free guidance
            if cfg_scale > 1.0 and unconditional_text_input_ids is not None:
                u_full_ids = torch.cat(
                    [unconditional_text_input_ids, image_ids], dim=1
                )
                u_hidden = self._run_backbone(u_full_ids, u_attn_mask)
                u_image_hidden = (
                    u_hidden[:, u_prefix_len:, :]
                    if u_prefix_len > 0
                    else u_hidden
                )
                u_logits = self._compute_logits(
                    u_image_hidden, height, width, step_mask_2d
                )
                u_fused = u_logits["fused"].reshape(1, -1, self.vocab_size)
                u_step_logits = u_fused[:, positions, :]
                step_logits = (
                    u_step_logits + cfg_scale * (step_logits - u_step_logits)
                )

            # mask non-visual tokens
            if self.visual_token_offset is not None:
                step_logits = step_logits.clone()
                step_logits[:, :, : self.visual_token_offset] = float("-inf")

            step_pred = _sample_logits(
                step_logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                sample_logits=sample_logits,
            )
            grid.view(1, -1)[:, positions] = step_pred

        return grid[0]


__all__ = [
    "EmuNAR",
    "build_interleaved_neighbor_ar_mask",
    "_sample_logits",
]
