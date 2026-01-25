# -*- coding: utf-8 -*-
# NAR wrapper for Emu3.5 with proximity attention and fused logits.

from __future__ import annotations

from typing import Dict, Optional

import torch
from torch import nn
from torch.nn import functional as F

from .neighbor_ar_mask import build_interleaved_neighbor_ar_mask


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
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
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


class NeighborARWrapper(nn.Module):
    """
    Wrap a pretrained AR backbone for NAR-style training and decoding.

    - Proximity mask allows tokens within the same diagonal step to attend each other.
    - Two heads predict right/down neighbors and are fused into per-position logits.
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
        img_token_id: Optional[int] = None,
        eol_token_id: Optional[int] = None,
        eoi_token_id: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.backbone = pretrained_backbone
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.pad_token_id = pad_token_id
        self.mask_token_id = mask_token_id if mask_token_id is not None else pad_token_id
        self.visual_token_offset = visual_token_offset
        self.use_vertical_block = use_vertical_block
        cfg = getattr(pretrained_backbone, "config", None)
        self.img_token_id = img_token_id if img_token_id is not None else getattr(cfg, "img_token_id", None)
        self.eol_token_id = eol_token_id if eol_token_id is not None else getattr(cfg, "eol_token_id", None)
        self.eoi_token_id = eoi_token_id if eoi_token_id is not None else getattr(cfg, "eoi_token_id", None)

        self.horizontal_head = nn.Linear(hidden_size, vocab_size)
        if use_vertical_block:
            self.vertical_block = nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=num_heads,
                dim_feedforward=hidden_size * ff_mult,
                dropout=dropout,
                batch_first=True,
                activation="gelu",
            )
            self.vertical_norm = nn.LayerNorm(hidden_size)
        else:
            self.vertical_block = None
            self.vertical_norm = None
        self.vertical_head = nn.Linear(hidden_size, vocab_size)
        self._sync_dtype_with_backbone()

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

    def _reshape_grid(self, seq: torch.Tensor, height: int, width: int) -> torch.Tensor:
        bsz, seq_len = seq.shape[:2]
        if seq_len != height * width:
            raise ValueError(f"Expected seq_len={height * width}, got {seq_len}")
        return seq.view(bsz, height, width, -1)

    def _build_step_mask_2d(self, step_id: torch.Tensor) -> torch.Tensor:
        t_steps = step_id.numel()
        s_q = step_id[:, None]
        s_k = step_id[None, :]
        allowed = s_k <= s_q
        return ~allowed

    def _fuse_logits(self, h_logits: torch.Tensor, v_logits: torch.Tensor) -> torch.Tensor:
        cond_logits = (h_logits[:, 0, 0, :] + v_logits[:, 0, 0, :]) / 2
        h_shift = h_logits.roll(shifts=1, dims=2)
        v_shift = v_logits.roll(shifts=1, dims=1)
        fused = torch.zeros_like(v_shift)
        fused[:, 0, 0, :] = cond_logits
        fused[:, 0, 1:, :] = h_shift[:, 0, 1:, :]
        fused[:, 1:, 0, :] = v_shift[:, 1:, 0, :]
        fused[:, 1:, 1:, :] = (h_shift[:, 1:, 1:, :] + v_shift[:, 1:, 1:, :]) / 2
        return fused

    def _compute_logits(
        self,
        image_hidden: torch.Tensor,
        height: int,
        width: int,
        step_mask_2d: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        h_grid = self._reshape_grid(image_hidden, height, width)
        h_logits = self.horizontal_head(h_grid)

        if self.vertical_block is not None:
            v_hidden = self.vertical_block(image_hidden, src_mask=step_mask_2d)
            v_hidden = self.vertical_norm(v_hidden)
        else:
            v_hidden = image_hidden
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

    def _get_fused_logits(
        self,
        input_ids: torch.Tensor,
        height: int,
        width: int,
        text_len: int,
        step_mask_2d: torch.Tensor,
    ) -> torch.Tensor:
        hidden = self._run_backbone(input_ids, attention_mask=None)
        image_hidden = hidden[:, text_len:, :] if text_len > 0 else hidden
        logits = self._compute_logits(image_hidden, height, width, step_mask_2d)
        return logits["fused"]

    def _build_full_inputs(
        self,
        text_input_ids: Optional[torch.Tensor],
        image_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, int, torch.Tensor, torch.Tensor]:
        mask_dtype = next(self.backbone.parameters()).dtype
        if text_input_ids is not None:
            if text_input_ids.dim() == 1:
                text_input_ids = text_input_ids.unsqueeze(0)
            prefix_len = text_input_ids.size(1)
            full_input_ids = torch.cat([text_input_ids, image_ids], dim=1)
            attn_mask, step_id = build_interleaved_neighbor_ar_mask(
                prefix_ids=text_input_ids,
                height=int(image_ids.size(1) ** 0.5),
                width=int(image_ids.size(1) ** 0.5),
                batch_size=image_ids.size(0),
                device=image_ids.device,
                dtype=mask_dtype,
                img_token_id=self.img_token_id,
                eoi_token_id=self.eoi_token_id,
                eol_token_id=self.eol_token_id,
                visual_token_offset=self.visual_token_offset,
                text_attention_mask=None,
            )
        else:
            prefix_len = 0
            full_input_ids = image_ids
            attn_mask, step_id = build_interleaved_neighbor_ar_mask(
                prefix_ids=None,
                height=int(image_ids.size(1) ** 0.5),
                width=int(image_ids.size(1) ** 0.5),
                batch_size=image_ids.size(0),
                device=image_ids.device,
                dtype=mask_dtype,
            )
        step_mask_2d = self._build_step_mask_2d(step_id)
        return full_input_ids, prefix_len, attn_mask, step_mask_2d

    def forward(
        self,
        input_ids: torch.Tensor,
        height: int,
        width: int,
        text_input_ids: Optional[torch.Tensor] = None,
        text_attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute fused NAR loss over image tokens.
        input_ids: [B, H*W] flattened image tokens.
        """
        bsz = input_ids.size(0)
        mask_dtype = next(self.backbone.parameters()).dtype
        if text_input_ids is not None:
            if text_input_ids.dim() == 1:
                text_input_ids = text_input_ids.unsqueeze(0)
            if text_input_ids.size(0) != bsz:
                raise ValueError("text_input_ids batch size must match image batch size.")
            prefix_len = text_input_ids.size(1)
            full_input_ids = torch.cat([text_input_ids, input_ids], dim=1)
            attn_mask, step_id = build_interleaved_neighbor_ar_mask(
                prefix_ids=text_input_ids,
                height=height,
                width=width,
                batch_size=bsz,
                device=input_ids.device,
                dtype=mask_dtype,
                img_token_id=self.img_token_id,
                eoi_token_id=self.eoi_token_id,
                eol_token_id=self.eol_token_id,
                visual_token_offset=self.visual_token_offset,
                text_attention_mask=text_attention_mask,
            )
        else:
            prefix_len = 0
            full_input_ids = input_ids
            attn_mask, step_id = build_interleaved_neighbor_ar_mask(
                prefix_ids=None,
                height=height,
                width=width,
                batch_size=bsz,
                device=input_ids.device,
                dtype=mask_dtype,
            )

        step_mask_2d = self._build_step_mask_2d(step_id)
        hidden = self._run_backbone(full_input_ids, attn_mask)
        image_hidden = hidden[:, prefix_len:, :] if prefix_len > 0 else hidden

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
            h_targets = target_grid[:, :, 1:]
            loss_h = F.cross_entropy(
                h_logits[:, :, :-1, :].reshape(-1, self.vocab_size),
                h_targets.reshape(-1),
                ignore_index=self.pad_token_id,
            )
        else:
            loss_h = torch.tensor(0.0, device=input_ids.device, dtype=loss.dtype)

        if height > 1:
            v_targets = target_grid[:, 1:, :]
            loss_v = F.cross_entropy(
                v_logits[:, :-1, :, :].reshape(-1, self.vocab_size),
                v_targets.reshape(-1),
                ignore_index=self.pad_token_id,
            )
        else:
            loss_v = torch.tensor(0.0, device=input_ids.device, dtype=loss.dtype)

        return {
            "loss": loss,
            "loss_h": loss_h,
            "loss_v": loss_v,
            "logits_h": h_logits,
            "logits_v": v_logits,
            "logits": fused,
        }

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
        Diagonal-step decoding with fused logits.
        Returns a [H, W] token grid.
        """
        grid = torch.full((1, height, width), self.mask_token_id, device=device, dtype=torch.long)
        image_ids = grid.view(1, -1)
        mask_dtype = next(self.backbone.parameters()).dtype
        if text_input_ids is not None:
            if text_input_ids.dim() == 1:
                text_input_ids = text_input_ids.unsqueeze(0)
            prefix_len = text_input_ids.size(1)
            attn_mask, step_id = build_interleaved_neighbor_ar_mask(
                prefix_ids=text_input_ids,
                height=height,
                width=width,
                batch_size=1,
                device=device,
                dtype=mask_dtype,
                img_token_id=self.img_token_id,
                eoi_token_id=self.eoi_token_id,
                eol_token_id=self.eol_token_id,
                visual_token_offset=self.visual_token_offset,
                text_attention_mask=None,
            )
        else:
            prefix_len = 0
            attn_mask, step_id = build_interleaved_neighbor_ar_mask(
                prefix_ids=None,
                height=height,
                width=width,
                batch_size=1,
                device=device,
                dtype=mask_dtype,
            )
        if unconditional_text_input_ids is not None and unconditional_text_input_ids.dim() == 1:
            unconditional_text_input_ids = unconditional_text_input_ids.unsqueeze(0)
        if unconditional_text_input_ids is not None:
            u_prefix_len = unconditional_text_input_ids.size(1)
            u_attn_mask, _ = build_interleaved_neighbor_ar_mask(
                prefix_ids=unconditional_text_input_ids,
                height=height,
                width=width,
                batch_size=1,
                device=device,
                dtype=mask_dtype,
                img_token_id=self.img_token_id,
                eoi_token_id=self.eoi_token_id,
                eol_token_id=self.eol_token_id,
                visual_token_offset=self.visual_token_offset,
                text_attention_mask=None,
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
            image_hidden = hidden[:, prefix_len:, :] if prefix_len > 0 else hidden
            logits = self._compute_logits(image_hidden, height, width, step_mask_2d)
            fused = logits["fused"].reshape(1, -1, self.vocab_size)
            step_logits = fused[:, positions, :]

            if cfg_scale > 1.0 and unconditional_text_input_ids is not None:
                u_full_ids = torch.cat([unconditional_text_input_ids, image_ids], dim=1)
                u_hidden = self._run_backbone(u_full_ids, u_attn_mask)
                u_image_hidden = u_hidden[:, u_prefix_len:, :] if u_prefix_len > 0 else u_hidden
                u_logits = self._compute_logits(u_image_hidden, height, width, step_mask_2d)
                u_fused = u_logits["fused"].reshape(1, -1, self.vocab_size)
                u_step_logits = u_fused[:, positions, :]
                step_logits = u_step_logits + cfg_scale * (step_logits - u_step_logits)

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
