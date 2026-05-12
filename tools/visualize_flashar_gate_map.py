#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from transformers.cache_utils import DynamicCache

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from flashar.model.modeling_emu_flashar import _build_step_id, _sample_logits

_GENEVAL_SCRIPT = REPO_ROOT / "tools" / "generate_geneval_flashar.py"
_SPEC = importlib.util.spec_from_file_location("generate_geneval_flashar_local", _GENEVAL_SCRIPT)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"Cannot load {_GENEVAL_SCRIPT}")
_GENEVAL_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_GENEVAL_MODULE)
decode_image = _GENEVAL_MODULE.decode_image
load_model = _GENEVAL_MODULE.load_model
make_prefix = _GENEVAL_MODULE.make_prefix


def parse_args():
    p = argparse.ArgumentParser(
        description="Run flashar inference and save the horizontal-head gate map."
    )
    p.add_argument(
        "--prompt",
        default="a cinematic red sports car on a rain-soaked city street at night",
    )
    p.add_argument("--outdir", default="outputs/gate_maps/default")
    p.add_argument("--model_path", default="./weights/Emu3.5-Image")
    p.add_argument("--tokenizer_path", default="./src/tokenizer_emu3_ibq")
    p.add_argument("--vq_path", default="./weights/Emu3.5-VisionTokenizer")
    p.add_argument("--ckpt_path", default="./weights/flashar_step74000_geneval077798_cfg5")
    p.add_argument("--height", type=int, default=32)
    p.add_argument("--width", type=int, default=32)
    p.add_argument(
        "--text_template",
        default="<|extra_203|>You are a helpful assistant for t2i task. USER: {text} ASSISTANT: <|extra_100|>",
    )
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_k", type=int, default=0)
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument("--cfg_scale", type=float, default=5.0)
    p.add_argument("--greedy", action="store_true")
    p.add_argument("--seed", type=int, default=20260506)
    p.add_argument("--visual_token_offset", type=int, default=-1)
    p.add_argument("--use_vertical_block", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--vertical_layers", type=int, default=0)
    p.add_argument("--vertical_start_layer", type=int, default=-1)
    p.add_argument("--split_backbone", action="store_true")
    p.add_argument("--cell_size", type=int, default=18)
    return p.parse_args()


def _select_visual_gate_batch(gate: torch.Tensor, use_cfg: bool) -> torch.Tensor:
    if use_cfg:
        return gate.chunk(2, dim=0)[1]
    return gate[:1]


@torch.no_grad()
def generate_with_gate_map(
    wrapper,
    *,
    height: int,
    width: int,
    device: torch.device,
    text_input_ids: torch.Tensor,
    text_attention_mask: torch.Tensor | None,
    unconditional_text_input_ids: torch.Tensor | None,
    unconditional_text_attention_mask: torch.Tensor | None,
    cfg_scale: float,
    temperature: float,
    top_k: int,
    top_p: float,
    sample_logits: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    step_id = _build_step_id(height, width, device)
    max_step = int(step_id.max().item())
    grid = torch.full(
        (1, height, width), wrapper.mask_token_id, device=device, dtype=torch.long
    )
    gate_map = torch.full((height * width,), 0.5, device=device, dtype=torch.float32)

    text_input_ids, text_attention_mask = wrapper._normalize_text_batch(
        text_input_ids,
        text_attention_mask,
    )
    if text_input_ids is None or text_attention_mask is None:
        raise ValueError("text_input_ids is required.")

    use_cfg = cfg_scale > 1.0 and unconditional_text_input_ids is not None
    if use_cfg:
        kv_text_ids, kv_text_mask = wrapper._build_cfg_text_batch(
            text_input_ids,
            text_attention_mask,
            unconditional_text_input_ids,
            unconditional_text_attention_mask,
        )
    else:
        kv_text_ids, kv_text_mask = text_input_ids, text_attention_mask

    (
        cond_horizontal_hidden,
        cond_vertical_hidden,
        backbone_cache,
        prefix_len,
        prefix_mask,
    ) = wrapper._prefill_generation_prefix(kv_text_ids, kv_text_mask)

    batch_size = int(kv_text_ids.size(0))
    vertical_cache = DynamicCache() if wrapper.vertical_block is not None else None
    past_image_len = 0
    prev_positions = torch.empty((0,), device=device, dtype=torch.long)
    prev_h_hidden = None
    prev_v_hidden = None

    for step in range(max_step + 1):
        step_positions = (step_id == step).nonzero(as_tuple=False).view(-1).to(
            device=device, dtype=torch.long
        )
        if step_positions.numel() == 0:
            continue

        rows = step_positions // width
        cols = step_positions % width
        left_mask = cols > 0
        up_mask = rows > 0
        both_mask = left_mask & up_mask
        corner_mask = ~left_mask & ~up_mask
        h_only = left_mask & ~up_mask
        v_only = up_mask & ~left_mask
        step_gate = torch.empty(
            (step_positions.numel(),), device=device, dtype=torch.float32
        )

        if h_only.any():
            step_gate[h_only] = 1.0
        if v_only.any():
            step_gate[v_only] = 0.0

        if both_mask.any():
            if prev_h_hidden is None or prev_v_hidden is None:
                raise RuntimeError("Missing previous hidden states for interior gates.")
            total = int(height * width)
            pos_to_idx = torch.full((total,), -1, device=device, dtype=torch.long)
            pos_to_idx[prev_positions] = torch.arange(
                prev_positions.numel(), device=device
            )
            left_idx = pos_to_idx[step_positions[both_mask] - 1]
            up_idx = pos_to_idx[step_positions[both_mask] - width]
            rw = wrapper._hv_gate_from_pair(
                prev_h_hidden[:, left_idx, :],
                prev_v_hidden[:, up_idx, :],
                out_dtype=prev_h_hidden.dtype,
            )
            rw = _select_visual_gate_batch(rw, use_cfg)
            step_gate[both_mask] = rw.squeeze(0).squeeze(-1).float()

        if corner_mask.any():
            rw_corner = wrapper._hv_gate_corner(
                cond_horizontal_hidden,
                out_dtype=cond_horizontal_hidden.dtype,
            )
            rw_corner = _select_visual_gate_batch(rw_corner, use_cfg)
            step_gate[corner_mask] = rw_corner.reshape(-1)[0].float()

        gate_map[step_positions] = step_gate

        branch_step_logits = wrapper._compute_step_logits_from_prev(
            cond_horizontal_hidden=cond_horizontal_hidden,
            cond_vertical_hidden=cond_vertical_hidden,
            prev_h_hidden=prev_h_hidden,
            prev_v_hidden=prev_v_hidden,
            step_positions=step_positions,
            prev_positions=prev_positions,
            height=height,
            width=width,
            device=device,
        )

        if use_cfg:
            u_step_logits, c_step_logits = branch_step_logits.chunk(2, dim=0)
            step_logits = u_step_logits + cfg_scale * (c_step_logits - u_step_logits)
        else:
            step_logits = branch_step_logits

        if wrapper.visual_token_offset is not None:
            step_logits = step_logits.clone()
            step_logits[:, :, : wrapper.visual_token_offset] = float("-inf")

        step_pred = _sample_logits(
            step_logits,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            sample_logits=sample_logits,
        )
        grid.view(1, -1)[:, step_positions] = step_pred

        step_token_ids = step_pred.expand(batch_size, -1).contiguous()
        current_h_hidden, current_v_input_hidden, backbone_cache = wrapper._append_backbone_kv_step(
            step_token_ids=step_token_ids,
            step_positions=step_positions,
            prefix_len=prefix_len,
            prefix_attention_mask=prefix_mask,
            past_key_values=backbone_cache,
            past_image_len=past_image_len,
        )
        current_v_hidden, vertical_cache = wrapper._append_vertical_kv_step(
            step_hidden=current_v_input_hidden,
            step_positions=step_positions,
            prefix_len=prefix_len,
            past_key_values=vertical_cache,
            past_image_len=past_image_len,
        )

        prev_positions = step_positions
        prev_h_hidden = current_h_hidden
        prev_v_hidden = current_v_hidden
        past_image_len += int(step_positions.numel())

    return grid[0], gate_map.view(height, width)


def horizontal_gate_colormap(gate: np.ndarray) -> np.ndarray:
    gate = np.clip(gate.astype(np.float32), 0.0, 1.0)
    low = np.array([38, 96, 204], dtype=np.float32)
    mid = np.array([247, 247, 245], dtype=np.float32)
    high = np.array([218, 48, 38], dtype=np.float32)
    t_low = np.clip(gate / 0.5, 0.0, 1.0)[..., None]
    t_high = np.clip((gate - 0.5) / 0.5, 0.0, 1.0)[..., None]
    low_half = low + (mid - low) * t_low
    high_half = mid + (high - mid) * t_high
    rgb = np.where((gate[..., None] < 0.5), low_half, high_half)
    return rgb.clip(0, 255).astype(np.uint8)


def save_heatmap(gate: np.ndarray, out_path: Path, cell_size: int) -> Image.Image:
    heat = Image.fromarray(horizontal_gate_colormap(gate), mode="RGB")
    resample = getattr(Image, "Resampling", Image).NEAREST
    heat = heat.resize((gate.shape[1] * cell_size, gate.shape[0] * cell_size), resample)
    draw = ImageDraw.Draw(heat)
    w, h = heat.size
    grid_color = (35, 35, 35)
    for x in range(0, w + 1, cell_size):
        draw.line((x, 0, x, h), fill=grid_color, width=1)
    for y in range(0, h + 1, cell_size):
        draw.line((0, y, w, y), fill=grid_color, width=1)
    heat.save(out_path)
    return heat


def save_overlay(image: Image.Image, gate: np.ndarray, out_path: Path) -> Image.Image:
    heat = Image.fromarray(horizontal_gate_colormap(gate), mode="RGB")
    resample = getattr(Image, "Resampling", Image).BILINEAR
    heat = heat.resize(image.size, resample)
    overlay = Image.blend(image.convert("RGB"), heat, alpha=0.42)
    overlay.save(out_path)
    return overlay


def save_comparison(image: Image.Image, heatmap: Image.Image, overlay: Image.Image, out_path: Path):
    target_h = max(image.height, heatmap.height, overlay.height)
    resample = getattr(Image, "Resampling", Image).BICUBIC

    def resize_to_h(im: Image.Image) -> Image.Image:
        if im.height == target_h:
            return im.convert("RGB")
        w = int(round(im.width * target_h / im.height))
        return im.convert("RGB").resize((w, target_h), resample)

    panels = [resize_to_h(image), resize_to_h(heatmap), resize_to_h(overlay)]
    gap = 16
    canvas = Image.new(
        "RGB",
        (sum(p.width for p in panels) + gap * (len(panels) - 1), target_h),
        (255, 255, 255),
    )
    x = 0
    for panel in panels:
        canvas.paste(panel, (x, 0))
        x += panel.width + gap
    canvas.save(out_path)


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    _, wrapper, tokenizer, vq, visual_token_offset, device = load_model(args)
    text_ids = make_prefix(
        tokenizer,
        args.prompt,
        args.text_template,
        args.height,
        args.width,
        device,
    )
    uncond_text_ids = None
    if float(args.cfg_scale) > 1.0:
        uncond_text_ids = make_prefix(
            tokenizer,
            "",
            args.text_template,
            args.height,
            args.width,
            device,
        )

    print(
        "flashar gate-map generation: "
        f"height={args.height} width={args.width} cfg_scale={args.cfg_scale} "
        f"use_cfg={uncond_text_ids is not None} ckpt={args.ckpt_path}",
        flush=True,
    )
    tokens, gate_map = generate_with_gate_map(
        wrapper,
        height=args.height,
        width=args.width,
        device=device,
        text_input_ids=text_ids,
        text_attention_mask=None,
        unconditional_text_input_ids=uncond_text_ids,
        unconditional_text_attention_mask=None,
        cfg_scale=args.cfg_scale,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        sample_logits=not args.greedy,
    )

    image = decode_image(vq, tokens, visual_token_offset, args.height, args.width, device)
    gate_np = gate_map.detach().float().cpu().numpy()

    image_path = outdir / "generated.png"
    heatmap_path = outdir / "horizontal_gate_heatmap.png"
    overlay_path = outdir / "horizontal_gate_overlay.png"
    comparison_path = outdir / "comparison.png"
    npy_path = outdir / "horizontal_gate_map.npy"
    meta_path = outdir / "metadata.json"

    image.save(image_path)
    heatmap = save_heatmap(gate_np, heatmap_path, args.cell_size)
    overlay = save_overlay(image, gate_np, overlay_path)
    save_comparison(image, heatmap, overlay, comparison_path)
    np.save(npy_path, gate_np)
    meta_path.write_text(
        json.dumps(
            {
                "prompt": args.prompt,
                "ckpt_path": args.ckpt_path,
                "height": args.height,
                "width": args.width,
                "cfg_scale": args.cfg_scale,
                "seed": args.seed,
                "gate_mean": float(gate_np.mean()),
                "gate_min": float(gate_np.min()),
                "gate_max": float(gate_np.max()),
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"generated_image={image_path}", flush=True)
    print(f"gate_heatmap={heatmap_path}", flush=True)
    print(f"gate_overlay={overlay_path}", flush=True)
    print(f"comparison={comparison_path}", flush=True)
    print(
        "gate_stats="
        f"mean:{gate_np.mean():.4f} min:{gate_np.min():.4f} max:{gate_np.max():.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
