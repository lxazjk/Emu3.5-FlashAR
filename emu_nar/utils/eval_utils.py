from __future__ import annotations

import argparse
import os
from contextlib import nullcontext
from typing import Any, Optional, Tuple

import torch
import torch.distributed as dist

from emu_nar.model import EmuNAR, _sample_logits
from emu_nar.utils.text_utils import encode_text_ids
from src.vision_tokenizer import build_vision_tokenizer


def decode_neighbor_grid(
    *,
    wrapper: EmuNAR,
    device: torch.device,
    height: int,
    width: int,
    prompt_ids: torch.Tensor,
    prompt_attention: torch.Tensor,
    visual_token_offset: int,
    temperature: float,
    top_k: int,
    top_p: float,
    sample_logits: bool,
    cfg_scale: float = 1.0,
    uncond_prompt_ids: Optional[torch.Tensor] = None,
    uncond_prompt_attention: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    mask_ids = wrapper.module if hasattr(wrapper, "module") else wrapper
    mask_token_id = mask_ids.mask_token_id
    grid = torch.full((1, height, width), mask_token_id, device=device, dtype=torch.long)
    rows = torch.arange(height, device=device).unsqueeze(1).expand(height, width)
    cols = torch.arange(width, device=device).unsqueeze(0).expand(height, width)
    step_id = (rows + cols).reshape(-1)
    max_step = int(step_id.max().item())

    use_cfg = (
        float(cfg_scale) > 1.0
        and uncond_prompt_ids is not None
        and uncond_prompt_attention is not None
    )

    for step in range(0, max_step + 1):
        positions = (step_id == step).nonzero(as_tuple=False).view(-1)
        if positions.numel() == 0:
            continue
        image_ids = grid.view(1, -1)
        prev_positions = positions if step == 0 else (step_id == (step - 1)).nonzero(
            as_tuple=False
        ).view(-1)
        positions = positions.to(device=device, dtype=torch.long)
        prev_positions = prev_positions.to(device=device, dtype=torch.long)

        outputs = wrapper(
            input_ids=image_ids,
            height=height,
            width=width,
            text_input_ids=prompt_ids,
            text_attention_mask=prompt_attention,
            step_positions=positions,
            prev_positions=prev_positions,
        )
        step_logits = outputs["step_logits"]

        if use_cfg:
            uncond_outputs = wrapper(
                input_ids=image_ids,
                height=height,
                width=width,
                text_input_ids=uncond_prompt_ids,
                text_attention_mask=uncond_prompt_attention,
                step_positions=positions,
                prev_positions=prev_positions,
            )
            uncond_step_logits = uncond_outputs["step_logits"]
            step_logits = uncond_step_logits + float(cfg_scale) * (
                step_logits - uncond_step_logits
            )

        if visual_token_offset:
            step_logits = step_logits.clone()
            step_logits[:, :, :visual_token_offset] = float("-inf")

        step_pred = _sample_logits(
            step_logits,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            sample_logits=sample_logits,
        )
        if dist.is_initialized():
            dist.broadcast(step_pred, src=0)
        grid.view(1, -1)[:, positions] = step_pred

    return grid


def run_epoch_generate(
    args: argparse.Namespace,
    wrapper: EmuNAR,
    tokenizer,
    vq_model,
    vq_dtype,
    device: torch.device,
    visual_token_offset: int,
    rank: int,
    is_main: bool,
    epoch: int,
    run_tag: str = "",
) -> Tuple[Any, Any]:
    del rank
    if not args.eval_generate_prompt:
        return vq_model, vq_dtype
    height = int(args.eval_generate_height)
    width = int(args.eval_generate_width)
    if height <= 0 or width <= 0:
        if is_main:
            print("[WARN] eval_generate_height/width must be set to enable generation.")
        return vq_model, vq_dtype
    out_dir = args.eval_generate_outdir or args.save_dir
    if is_main:
        os.makedirs(out_dir, exist_ok=True)
    tag = run_tag if run_tag else f"epoch{epoch}"

    def _save_outputs(grid: torch.Tensor) -> None:
        if not is_main:
            return
        grid_cpu = grid.detach().cpu()
        pt_path = os.path.join(out_dir, f"gen_{tag}.pt")
        torch.save(grid_cpu, pt_path)
        print(f"[EVAL] saved grid: {pt_path}")
        if not args.eval_generate_decode:
            return
        nonlocal vq_model, vq_dtype
        if vq_model is None or str(next(vq_model.parameters()).device) != str(device):
            vq_model = build_vision_tokenizer(args.vq_type, args.vq_path, device=str(device))
            vq_model.eval()
            for p in vq_model.parameters():
                p.requires_grad = False
            vq_dtype = next(vq_model.parameters()).dtype
        codes = grid_cpu - visual_token_offset
        codes = codes.to(device=next(vq_model.parameters()).device, dtype=torch.long)
        with torch.no_grad():
            image = vq_model.decode_code(codes[None], shape=(1, height, width, 256)).float()
        image = image[0].permute(1, 2, 0)
        try:
            from PIL import Image
            import numpy as np

            img = Image.fromarray(
                ((image + 1.0) * 127.5).clamp(0, 255).cpu().numpy().astype(np.uint8)
            )
            png_path = os.path.join(out_dir, f"gen_{tag}.png")
            img.save(png_path)
            print(f"[EVAL] saved image: {png_path}")
        except Exception as exc:
            print(f"[WARN] decode image failed: {exc}")

    was_training = wrapper.training
    wrapper.eval()

    prompt_ids = encode_text_ids(
        tokenizer=tokenizer,
        text_template=args.text_template,
        text=args.eval_generate_prompt,
        text_max_length=args.text_max_length,
        add_boi=args.add_boi,
        height=height,
        width=width,
    ).to(device)
    if prompt_ids.dim() == 1:
        prompt_ids = prompt_ids.unsqueeze(0)
    prompt_attention = torch.ones((prompt_ids.size(0), prompt_ids.size(1)), dtype=torch.long, device=device)

    uncond_prompt_ids = None
    uncond_prompt_attention = None
    if float(args.eval_generate_cfg_scale) > 1.0:
        uncond_prompt_ids = encode_text_ids(
            tokenizer=tokenizer,
            text_template=args.text_template,
            text="",
            text_max_length=args.text_max_length,
            add_boi=args.add_boi,
            height=height,
            width=width,
        ).to(device)
        if uncond_prompt_ids.dim() == 1:
            uncond_prompt_ids = uncond_prompt_ids.unsqueeze(0)
        uncond_prompt_attention = torch.ones(
            (uncond_prompt_ids.size(0), uncond_prompt_ids.size(1)),
            dtype=torch.long,
            device=device,
        )

    rng_cpu = torch.get_rng_state()
    rng_cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    torch.manual_seed(args.eval_generate_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.eval_generate_seed)

    prev_fastpath = None
    if hasattr(torch.backends, "mha"):
        prev_fastpath = torch.backends.mha.get_fastpath_enabled()
        torch.backends.mha.set_fastpath_enabled(False)
    sdp_ctx = nullcontext()
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "sdp_kernel"):
        sdp_ctx = torch.backends.cuda.sdp_kernel(
            enable_flash=False, enable_mem_efficient=False, enable_math=True
        )
    # FSDP orig-param writeback can fail after running the wrapped module inside
    # inference_mode(), so keep eval generation on no_grad() instead.
    with sdp_ctx, torch.no_grad():
        grid = decode_neighbor_grid(
            wrapper=wrapper,
            device=device,
            height=height,
            width=width,
            prompt_ids=prompt_ids,
            prompt_attention=prompt_attention,
            visual_token_offset=visual_token_offset,
            temperature=args.eval_generate_temperature,
            top_k=args.eval_generate_top_k,
            top_p=args.eval_generate_top_p,
            sample_logits=args.eval_generate_sample,
            cfg_scale=float(args.eval_generate_cfg_scale),
            uncond_prompt_ids=uncond_prompt_ids,
            uncond_prompt_attention=uncond_prompt_attention,
        )

    if prev_fastpath is not None:
        torch.backends.mha.set_fastpath_enabled(prev_fastpath)
    _save_outputs(grid)
    if args.fsdp and dist.is_initialized():
        dist.barrier()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    torch.set_rng_state(rng_cpu)
    if rng_cuda is not None:
        torch.cuda.set_rng_state_all(rng_cuda)

    if was_training:
        wrapper.train()
    return vq_model, vq_dtype
