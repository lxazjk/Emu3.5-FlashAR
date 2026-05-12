from __future__ import annotations

import gc
import importlib.util
import os
import os.path as osp
import time
from pathlib import Path
from typing import Any, Dict

import torch
from PIL import Image

from flashar.utils.text_utils import build_image_prefix_tokens
from src.utils.generation_utils import multimodal_decode


def load_benchmark_cfg(cfg_path: str) -> Any:
    cfg_path = osp.abspath(cfg_path)
    module_name = f"bench_cfg_{Path(cfg_path).stem}_{int(time.time() * 1e6)}"
    spec = importlib.util.spec_from_file_location(module_name, cfg_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load config from {cfg_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepare_benchmark_cfg(cfg: Any, save_path: str) -> Any:
    cfg.rank = 0
    cfg.world_size = 1
    cfg.save_path = save_path
    if isinstance(cfg.prompts, dict):
        cfg.prompts = [(name, prompt) for name, prompt in cfg.prompts.items()]
    else:
        cfg.prompts = [(f"{idx:03d}", prompt) for idx, prompt in enumerate(cfg.prompts)]
    cfg.num_prompts = len(cfg.prompts)
    return cfg


def build_special_token_ids(cfg: Any, tokenizer: Any) -> Dict[str, int]:
    return {key: tokenizer.encode(value)[0] for key, value in cfg.special_tokens.items()}


def ensure_bos(input_ids: torch.Tensor, bos_id: int) -> torch.Tensor:
    if input_ids[0, 0].item() == bos_id:
        return input_ids
    bos = torch.tensor([[bos_id]], device=input_ids.device, dtype=input_ids.dtype)
    return torch.cat([bos, input_ids], dim=1)

def build_prompt_tensor(
    cfg: Any,
    tokenizer: Any,
    text: str,
    *,
    add_image_prefix: bool,
    device: torch.device,
) -> torch.Tensor:
    prompt = cfg.template.format(question=text)
    input_ids = tokenizer.encode(
        prompt,
        return_tensors="pt",
        add_special_tokens=False,
    ).to(device)
    input_ids = ensure_bos(input_ids, cfg.special_token_ids["BOS"])
    if add_image_prefix:
        prefix = torch.tensor(
            [build_image_prefix_tokens(tokenizer, int(cfg.target_height), int(cfg.target_width))],
            device=device,
            dtype=input_ids.dtype,
        )
        input_ids = torch.cat([input_ids, prefix], dim=1)
    return input_ids


def build_unconditional_tensor(
    cfg: Any,
    tokenizer: Any,
    *,
    add_image_prefix: bool,
    device: torch.device,
) -> torch.Tensor:
    input_ids = tokenizer.encode(
        cfg.unc_prompt,
        return_tensors="pt",
        add_special_tokens=False,
    ).to(device)
    input_ids = ensure_bos(input_ids, cfg.special_token_ids["BOS"])
    if add_image_prefix:
        prefix = torch.tensor(
            [build_image_prefix_tokens(tokenizer, int(cfg.target_height), int(cfg.target_width))],
            device=device,
            dtype=input_ids.dtype,
        )
        input_ids = torch.cat([input_ids, prefix], dim=1)
    return input_ids


def save_image(image: Image.Image, out_path: str) -> None:
    os.makedirs(osp.dirname(out_path), exist_ok=True)
    image.save(out_path)


def decode_flashar_grid_to_image(
    *,
    vq_model: Any,
    grid: torch.Tensor,
    visual_token_offset: int,
    height: int,
    width: int,
) -> Image.Image:
    device = next(vq_model.parameters()).device
    embed_dim = getattr(vq_model.quantize, "e_dim", 256)
    with torch.no_grad():
        vq_tokens = (grid - visual_token_offset).clamp_min(0)
        image = vq_model.decode_code(
            vq_tokens[None].to(device),
            shape=(1, height, width, embed_dim),
        ).float()
    image = image[0].permute(1, 2, 0)
    image_array = (
        ((image + 1.0) * 127.5).clamp(0, 255).detach().cpu().numpy().astype("uint8")
    )
    return Image.fromarray(image_array)


def decode_ar_tokens_to_image(
    *,
    tokenizer: Any,
    vq_model: Any,
    token_ids: torch.Tensor,
) -> Image.Image:
    decoded = tokenizer.batch_decode(token_ids, skip_special_tokens=False)[0]
    mm_out = multimodal_decode(decoded, tokenizer, vq_model)
    for kind, payload in mm_out:
        if kind == "image" and isinstance(payload, Image.Image):
            return payload
    raise RuntimeError("AR decode produced no image.")


def sync_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def cleanup_cuda(*objs: Any) -> None:
    for obj in objs:
        del obj
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
