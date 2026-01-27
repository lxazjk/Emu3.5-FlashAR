# -*- coding: utf-8 -*-
# Minimal LoRA utilities for Emu3 backbone fine-tuning.

from __future__ import annotations

import math
from typing import Iterable, Sequence

import torch
from torch import nn


class LoRALinear(nn.Module):
    def __init__(
        self,
        base: nn.Linear,
        r: int,
        alpha: float,
        dropout: float,
    ) -> None:
        super().__init__()
        if r <= 0:
            raise ValueError("LoRA rank must be > 0.")
        self.base = base
        self.r = int(r)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.r
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.lora_A = nn.Linear(base.in_features, self.r, bias=False)
        self.lora_B = nn.Linear(self.r, base.out_features, bias=False)

        nn.init.normal_(self.lora_A.weight, std=0.02)
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = self.base(x)
        lora = self.lora_B(self.lora_A(self.dropout(x))) * self.scaling
        return result + lora


def _replace_linear(parent: nn.Module, name: str, r: int, alpha: float, dropout: float) -> int:
    mod = getattr(parent, name, None)
    if not isinstance(mod, nn.Linear) or isinstance(mod, LoRALinear):
        return 0
    setattr(parent, name, LoRALinear(mod, r=r, alpha=alpha, dropout=dropout))
    return 1


def apply_lora_to_backbone(
    backbone: nn.Module,
    num_layers: int,
    r: int,
    alpha: float,
    dropout: float,
    target_modules: Sequence[str] | None = None,
) -> int:
    if r <= 0 or num_layers <= 0:
        return 0
    layers = getattr(getattr(backbone, "model", backbone), "layers", None)
    if layers is None:
        raise ValueError("Backbone has no layers to apply LoRA.")
    num_layers = min(int(num_layers), len(layers))
    targets = target_modules or (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    )
    count = 0
    for layer in list(layers)[-num_layers:]:
        for name in targets:
            count += _replace_linear(layer.self_attn, name, r, alpha, dropout)
            count += _replace_linear(layer.mlp, name, r, alpha, dropout)
    return count


def _build_r_schedule(num_layers: int, r_min: int, r_max: int) -> list[int]:
    if num_layers <= 0:
        return []
    if num_layers == 1:
        return [int(r_max)]
    rs: list[int] = []
    for idx in range(num_layers):
        frac = idx / (num_layers - 1)
        r = int(round(r_min + frac * (r_max - r_min)))
        r = max(1, min(int(r_max), r))
        rs.append(r)
    return rs


def apply_progressive_lora_to_backbone(
    backbone: nn.Module,
    num_layers: int,
    r_min: int,
    r_max: int,
    alpha_scale: float,
    dropout: float,
    target_modules: Sequence[str] | None = None,
) -> int:
    if r_min <= 0 or r_max <= 0 or num_layers <= 0:
        return 0
    if r_min > r_max:
        raise ValueError("LoRA r_min must be <= r_max.")
    layers = getattr(getattr(backbone, "model", backbone), "layers", None)
    if layers is None:
        raise ValueError("Backbone has no layers to apply LoRA.")
    num_layers = min(int(num_layers), len(layers))
    targets = target_modules or (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    )
    selected = list(layers)[-num_layers:]
    rs = _build_r_schedule(len(selected), int(r_min), int(r_max))
    count = 0
    for layer, r in zip(selected, rs):
        alpha = float(alpha_scale) * float(r)
        for name in targets:
            count += _replace_linear(layer.self_attn, name, r, alpha, dropout)
            count += _replace_linear(layer.mlp, name, r, alpha, dropout)
    return count


def iter_lora_parameters(module: nn.Module) -> Iterable[nn.Parameter]:
    for sub in module.modules():
        if isinstance(sub, LoRALinear):
            yield from sub.lora_A.parameters()
            yield from sub.lora_B.parameters()


def collect_lora_modules(module: nn.Module) -> list[nn.Module]:
    return [m for m in module.modules() if isinstance(m, LoRALinear)]


__all__ = [
    "LoRALinear",
    "apply_lora_to_backbone",
    "apply_progressive_lora_to_backbone",
    "iter_lora_parameters",
    "collect_lora_modules",
]
