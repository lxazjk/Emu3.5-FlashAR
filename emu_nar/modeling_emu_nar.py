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

import copy
import math
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


def build_t2i_neighbor_ar_mask(
    prefix_ids: Optional[torch.Tensor],
    height: int,
    width: int,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
    text_attention_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build a 4-D mask for pure T2I:
      - prefix (text/special tokens) uses causal mask
      - image tokens attend all valid prefix tokens
      - image block uses diagonal-step proximity mask
    """
    if prefix_ids is None:
        prefix_ids = torch.empty((batch_size, 0), device=device, dtype=torch.long)
    if prefix_ids.dim() == 1:
        prefix_ids = prefix_ids.unsqueeze(0)
    prefix_len = prefix_ids.size(1)

    t_steps = height * width
    total = prefix_len + t_steps
    attn = torch.full(
        (batch_size, 1, total, total), float("-inf"), device=device, dtype=dtype
    )

    if prefix_len > 0:
        causal = torch.tril(
            torch.ones(prefix_len, prefix_len, device=device, dtype=torch.bool)
        )
        diag = torch.eye(prefix_len, device=device, dtype=torch.bool)
        for b in range(batch_size):
            allow = causal.clone()
            if text_attention_mask is not None:
                valid = text_attention_mask[b].to(device=device).bool()
                allow &= valid.unsqueeze(0) & valid.unsqueeze(1)
                allow |= diag
                attn[b, 0, prefix_len:, :prefix_len] = torch.where(
                    valid.unsqueeze(0), 0.0, float("-inf")
                )
            else:
                attn[b, 0, prefix_len:, :prefix_len] = 0.0
            attn[b, 0, :prefix_len, :prefix_len].masked_fill_(allow, 0.0)

    step_id = _build_step_id(height, width, device)
    allowed = _build_proximity_allow(height, width, device)
    attn[:, 0, prefix_len:, prefix_len:].masked_fill_(allowed, 0.0)
    return attn, step_id


def build_t2i_causal_mask(
    prefix_ids: Optional[torch.Tensor],
    height: int,
    width: int,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
    text_attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Build a standard AR causal 4-D mask for T2I sequence:
      [text_prefix][image_tokens]
    """
    if prefix_ids is None:
        prefix_ids = torch.empty((batch_size, 0), device=device, dtype=torch.long)
    if prefix_ids.dim() == 1:
        prefix_ids = prefix_ids.unsqueeze(0)
    prefix_len = prefix_ids.size(1)
    image_len = height * width
    total = prefix_len + image_len

    causal = torch.tril(torch.ones((total, total), device=device, dtype=torch.bool))
    eye = torch.eye(total, device=device, dtype=torch.bool)
    attn = torch.full((batch_size, 1, total, total), float("-inf"), device=device, dtype=dtype)
    image_valid = torch.ones((image_len,), device=device, dtype=torch.bool)
    for b in range(batch_size):
        if prefix_len > 0 and text_attention_mask is not None:
            prefix_valid = text_attention_mask[b, :prefix_len].to(device=device).bool()
        else:
            prefix_valid = torch.ones((prefix_len,), device=device, dtype=torch.bool)
        valid = torch.cat([prefix_valid, image_valid], dim=0)
        allow = causal & valid.unsqueeze(0) & valid.unsqueeze(1)
        allow |= eye
        attn[b, 0].masked_fill_(allow, 0.0)
    return attn


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
        learnable_fuse: bool = False,
        fuse_h_init: float = 0.5,
        fuse_corner_h_init: float = -1.0,
        lm_head: Optional[nn.Linear] = None,
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
        self.learnable_fuse = bool(learnable_fuse)

        cfg = getattr(pretrained_backbone, "config", None)

        # --- prediction heads ---
        self.horizontal_head = nn.Linear(hidden_size, vocab_size)

        if use_vertical_block:
            layers = max(1, int(vertical_layers))
            backbone_layers = getattr(pretrained_backbone, "layers", None)
            if backbone_layers is None or len(backbone_layers) < layers:
                raise ValueError(
                    "pretrained_backbone.layers is missing or shorter than vertical_layers; "
                    "cannot initialize vertical block from Emu3 decoder layers."
                )
            # Reuse Emu3.5 transformer parameters by cloning decoder layers from backbone.
            self.vertical_block = nn.ModuleList(
                [copy.deepcopy(layer) for layer in backbone_layers[-layers:]]
            )
            for layer in self.vertical_block:
                if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "is_causal"):
                    layer.self_attn.is_causal = False
            rms_eps = float(getattr(cfg, "rms_norm_eps", 1e-6))
            from src.emu3p5.modeling_emu3 import Emu3RMSNorm

            self.vertical_norm = Emu3RMSNorm(hidden_size, eps=rms_eps)
        else:
            self.vertical_block = None
            self.vertical_norm = None

        self.vertical_head = nn.Linear(hidden_size, vocab_size)

        if fuse_corner_h_init < 0:
            fuse_corner_h_init = fuse_h_init
        self.fuse_h_init = self._clamp_prob(float(fuse_h_init))
        self.fuse_corner_h_init = self._clamp_prob(float(fuse_corner_h_init))
        if self.learnable_fuse:
            self.fuse_h_logit = nn.Parameter(
                torch.tensor(
                    [self._prob_to_logit(self.fuse_h_init)], dtype=torch.float32
                )
            )
            self.fuse_corner_h_logit = nn.Parameter(
                torch.tensor(
                    [self._prob_to_logit(self.fuse_corner_h_init)],
                    dtype=torch.float32,
                )
            )
        else:
            self.fuse_h_logit = None
            self.fuse_corner_h_logit = None

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
    def _clamp_prob(prob: float) -> float:
        return min(max(prob, 1e-4), 1.0 - 1e-4)

    @classmethod
    def _prob_to_logit(cls, prob: float) -> float:
        p = cls._clamp_prob(prob)
        return math.log(p / (1.0 - p))

    def _fuse_weights(
        self, device: torch.device, dtype: torch.dtype
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.learnable_fuse:
            w_h = torch.sigmoid(self.fuse_h_logit).to(device=device, dtype=dtype)
            w_h_corner = torch.sigmoid(self.fuse_corner_h_logit).to(
                device=device, dtype=dtype
            )
        else:
            w_h = torch.tensor(self.fuse_h_init, device=device, dtype=dtype)
            w_h_corner = torch.tensor(
                self.fuse_corner_h_init, device=device, dtype=dtype
            )
        return w_h, (1.0 - w_h), w_h_corner, (1.0 - w_h_corner)

    def get_fuse_stats(self) -> Dict[str, torch.Tensor]:
        device = self.horizontal_head.weight.device
        dtype = self.horizontal_head.weight.dtype
        w_h, w_v, w_h_corner, w_v_corner = self._fuse_weights(device, dtype)
        return {
            "fuse_w_h": w_h.detach().reshape(()),
            "fuse_w_v": w_v.detach().reshape(()),
            "fuse_w_h_corner": w_h_corner.detach().reshape(()),
            "fuse_w_v_corner": w_v_corner.detach().reshape(()),
        }

    @staticmethod
    def _build_step_mask_2d(step_id: torch.Tensor) -> torch.Tensor:
        """Bool mask where True means blocked attention."""
        allowed = step_id[None, :] <= step_id[:, None]
        return ~allowed

    @staticmethod
    def _build_step_attn_4d(
        step_mask_2d: torch.Tensor, batch_size: int, dtype: torch.dtype
    ) -> torch.Tensor:
        seq_len = int(step_mask_2d.size(0))
        mask = torch.zeros(
            (batch_size, 1, seq_len, seq_len),
            device=step_mask_2d.device,
            dtype=dtype,
        )
        if step_mask_2d.any():
            mask_val = torch.finfo(dtype).min
            mask.masked_fill_(step_mask_2d.view(1, 1, seq_len, seq_len), mask_val)
        return mask

    def _fuse_logits(
        self,
        h_logits: torch.Tensor,
        v_logits: torch.Tensor,
        cond_h_logits: torch.Tensor,
        cond_v_logits: torch.Tensor,
    ) -> torch.Tensor:
        """
        Shift-and-merge horizontal / vertical logits into per-position predictions.

        h_logits[b, h, w] predicts position (h, w+1);  after roll → position (h, w).
        v_logits[b, h, w] predicts position (h+1, w);  after roll → position (h, w).
        """
        w_h, w_v, w_h_corner, w_v_corner = self._fuse_weights(
            h_logits.device, h_logits.dtype
        )
        cond_logits = (
            w_h_corner * cond_h_logits[:, 0, :] + w_v_corner * cond_v_logits[:, 0, :]
        )
        h_shift = h_logits.roll(shifts=1, dims=2)   # prediction for right neighbor
        v_shift = v_logits.roll(shifts=1, dims=1)   # prediction for bottom neighbor
        fused = torch.zeros_like(v_shift)
        fused[:, 0, 0, :] = cond_logits              # corner: learnable fuse
        fused[:, 0, 1:, :] = h_shift[:, 0, 1:, :]   # first row: horizontal only
        fused[:, 1:, 0, :] = v_shift[:, 1:, 0, :]   # first col: vertical only
        fused[:, 1:, 1:, :] = (                      # interior: learnable fuse
            w_h * h_shift[:, 1:, 1:, :] + w_v * v_shift[:, 1:, 1:, :]
        )
        return fused

    def _apply_vertical_block(
        self, image_hidden: torch.Tensor, step_mask_2d: torch.Tensor
    ) -> torch.Tensor:
        if self.vertical_block is not None:
            bsz, seq_len, _ = image_hidden.shape
            attn_mask = self._build_step_attn_4d(step_mask_2d, bsz, image_hidden.dtype)
            position_ids = torch.arange(
                seq_len, device=image_hidden.device, dtype=torch.long
            ).unsqueeze(0).expand(bsz, -1)
            v_hidden = image_hidden
            for layer in self.vertical_block:
                v_hidden = layer(
                    hidden_states=v_hidden,
                    attention_mask=attn_mask,
                    position_ids=position_ids,
                    output_attentions=False,
                    use_cache=False,
                )[0]
            return self.vertical_norm(v_hidden)
        return image_hidden

    def _compute_logits(
        self,
        cond_hidden: torch.Tensor,
        image_hidden: torch.Tensor,
        height: int,
        width: int,
        step_mask_2d: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        cond_h_logits = self.horizontal_head(cond_hidden)
        cond_v_logits = self.vertical_head(cond_hidden)
        h_grid = self._reshape_grid(image_hidden, height, width)
        h_logits = self.horizontal_head(h_grid)

        v_hidden = self._apply_vertical_block(image_hidden, step_mask_2d)
        v_grid = self._reshape_grid(v_hidden, height, width)
        v_logits = self.vertical_head(v_grid)

        fused = self._fuse_logits(h_logits, v_logits, cond_h_logits, cond_v_logits)
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
            attn_mask, step_id = build_t2i_neighbor_ar_mask(
                prefix_ids=text_input_ids,
                height=height,
                width=width,
                batch_size=image_ids.size(0),
                device=image_ids.device,
                dtype=mask_dtype,
                text_attention_mask=text_attention_mask,
            )
        else:
            prefix_len = 0
            full_input_ids = image_ids
            attn_mask, step_id = build_t2i_neighbor_ar_mask(
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
        ar_distill: bool = False,
        ar_distill_temperature: float = 2.0,
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
        if prefix_len > 0:
            cond_hidden = hidden[:, prefix_len - 1 : prefix_len, :]
            image_hidden = hidden[:, prefix_len:, :]
        else:
            cond_hidden = hidden[:, :1, :]
            image_hidden = hidden

        # === step-based generation path ===
        if step_positions is not None and prev_positions is not None:
            return self._forward_step(
                cond_hidden, image_hidden, step_mask_2d, step_positions, prev_positions,
                height, width, input_ids.device,
            )

        # === full loss path ===
        outputs = self._forward_full(
            cond_hidden, image_hidden, step_mask_2d, input_ids, height, width,
        )
        if ar_distill:
            outputs["loss_distill"] = self._compute_ar_distill_loss(
                full_input_ids=full_input_ids,
                prefix_len=prefix_len,
                text_input_ids=text_input_ids,
                text_attention_mask=text_attention_mask,
                student_v_logits=outputs["logits_v"],
                height=height,
                width=width,
                temperature=ar_distill_temperature,
            )
        return outputs

    # ------------------------------------------------------------------
    # Forward sub-paths
    # ------------------------------------------------------------------

    def _forward_step(
        self,
        cond_hidden: torch.Tensor,
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
        cond_h_logits = self.horizontal_head(cond_hidden)
        cond_v_logits = self.vertical_head(cond_hidden)

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

        w_h, w_v, w_h_corner, w_v_corner = self._fuse_weights(
            h_prev.device, h_prev.dtype
        )

        # interior: learnable fuse
        if both_mask.any():
            left_idx = pos_to_idx[step_positions[both_mask] - 1]
            up_idx = pos_to_idx[step_positions[both_mask] - width]
            step_logits[:, both_mask, :] = (
                w_h * h_prev[:, left_idx, :] + w_v * v_prev[:, up_idx, :]
            )

        # corner (0,0): learnable corner fuse
        if corner_mask.any():
            cond_logits = (
                w_h_corner * cond_h_logits[:, :1, :] + w_v_corner * cond_v_logits[:, :1, :]
            )
            step_logits[:, corner_mask, :] = cond_logits.expand(-1, int(corner_mask.sum().item()), -1)

        return {"step_logits": step_logits}

    def _forward_full(
        self,
        cond_hidden: torch.Tensor,
        image_hidden: torch.Tensor,
        step_mask_2d: torch.Tensor,
        input_ids: torch.Tensor,
        height: int,
        width: int,
    ) -> Dict[str, torch.Tensor]:
        """Standard full-grid loss."""
        bsz = input_ids.size(0)
        device = input_ids.device
        logits = self._compute_logits(cond_hidden, image_hidden, height, width, step_mask_2d)
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

        fuse_stats = self.get_fuse_stats()
        return {
            "loss": loss,
            "loss_h": loss_h,
            "loss_v": loss_v,
            "logits_h": h_logits,
            "logits_v": v_logits,
            "logits": fused,
            **fuse_stats,
        }

    def _compute_ar_distill_loss(
        self,
        full_input_ids: torch.Tensor,
        prefix_len: int,
        text_input_ids: Optional[torch.Tensor],
        text_attention_mask: Optional[torch.Tensor],
        student_v_logits: torch.Tensor,
        height: int,
        width: int,
        temperature: float,
    ) -> torch.Tensor:
        """
        AR distillation for vertical branch:
          student at (r,c) predicting (r+1,c) vs AR teacher at prev token (q-1).
        """
        bsz = int(full_input_ids.size(0))
        image_len = int(height * width)
        if height <= 1:
            return student_v_logits.new_zeros(())

        student_flat = student_v_logits[:, :-1, :, :].reshape(bsz, -1, self.vocab_size)
        q = torch.arange(width, image_len, device=full_input_ids.device, dtype=torch.long)

        if student_flat.size(1) != int(q.numel()):
            raise RuntimeError(
                f"Distill alignment mismatch: student={student_flat.size(1)} teacher_targets={int(q.numel())}"
            )

        mask_dtype = next(self.backbone.parameters()).dtype
        ar_attn = build_t2i_causal_mask(
            prefix_ids=text_input_ids,
            height=height,
            width=width,
            batch_size=bsz,
            device=full_input_ids.device,
            dtype=mask_dtype,
            text_attention_mask=text_attention_mask,
        )
        with torch.no_grad():
            ar_hidden = self._run_backbone(full_input_ids, ar_attn)
        ar_image_hidden = ar_hidden[:, prefix_len:, :] if prefix_len > 0 else ar_hidden

        # teacher at previous token predicts current target q in AR factorization.
        prev_idx = q - 1

        temp = max(float(temperature), 1e-5)
        chunk = 64
        kl_sum = student_flat.new_zeros(())
        token_count = 0
        for start in range(0, int(prev_idx.numel()), chunk):
            end = min(start + chunk, int(prev_idx.numel()))
            idx = prev_idx[start:end]
            with torch.no_grad():
                teacher_logits = self.horizontal_head(ar_image_hidden[:, idx, :])
            student_logits = student_flat[:, start:end, :]
            log_p = F.log_softmax(student_logits.float() / temp, dim=-1)
            q_prob = F.softmax(teacher_logits.float() / temp, dim=-1)
            kl = F.kl_div(log_p, q_prob, reduction="none").sum(dim=-1)
            kl_sum = kl_sum + kl.sum()
            token_count += int(kl.numel())

        if token_count == 0:
            return student_v_logits.new_zeros(())
        return (kl_sum / float(token_count)) * (temp * temp)

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
            attn_mask, step_id = build_t2i_neighbor_ar_mask(
                prefix_ids=text_input_ids,
                height=height, width=width, batch_size=1,
                device=device, dtype=mask_dtype,
            )
        else:
            prefix_len = 0
            attn_mask, step_id = build_t2i_neighbor_ar_mask(
                prefix_ids=None,
                height=height, width=width, batch_size=1,
                device=device, dtype=mask_dtype,
            )

        # --- unconditional mask (for CFG) ---
        if unconditional_text_input_ids is not None:
            if unconditional_text_input_ids.dim() == 1:
                unconditional_text_input_ids = unconditional_text_input_ids.unsqueeze(0)
            u_prefix_len = unconditional_text_input_ids.size(1)
            u_attn_mask, _ = build_t2i_neighbor_ar_mask(
                prefix_ids=unconditional_text_input_ids,
                height=height, width=width, batch_size=1,
                device=device, dtype=mask_dtype,
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
            if prefix_len > 0:
                cond_hidden = hidden[:, prefix_len - 1 : prefix_len, :]
                image_hidden = hidden[:, prefix_len:, :]
            else:
                cond_hidden = hidden[:, :1, :]
                image_hidden = hidden
            logits = self._compute_logits(
                cond_hidden, image_hidden, height, width, step_mask_2d
            )
            fused = logits["fused"].reshape(1, -1, self.vocab_size)
            step_logits = fused[:, positions, :]

            # classifier-free guidance
            if cfg_scale > 1.0 and unconditional_text_input_ids is not None:
                u_full_ids = torch.cat(
                    [unconditional_text_input_ids, image_ids], dim=1
                )
                u_hidden = self._run_backbone(u_full_ids, u_attn_mask)
                if u_prefix_len > 0:
                    u_cond_hidden = u_hidden[:, u_prefix_len - 1 : u_prefix_len, :]
                    u_image_hidden = u_hidden[:, u_prefix_len:, :]
                else:
                    u_cond_hidden = u_hidden[:, :1, :]
                    u_image_hidden = u_hidden
                u_logits = self._compute_logits(
                    u_cond_hidden, u_image_hidden, height, width, step_mask_2d
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
    "build_t2i_neighbor_ar_mask",
    "build_t2i_causal_mask",
    "_sample_logits",
]
