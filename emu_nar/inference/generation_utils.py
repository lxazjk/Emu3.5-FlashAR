# -*- coding: utf-8 -*-
# NAR-aware interleaved generation for Emu3.5.

import re
from typing import Generator, List, Any

import numpy as np
import torch

from emu_nar.model import EmuNAR
from emu_nar.inference.token_utils import (
    collect_hw_tokens as _collect_hw_tokens,
    get_digit_token_ids as _get_digit_token_ids,
    get_special_ids as _get_special_ids,
    parse_hw_from_tokens as _parse_hw_from_tokens,
    sample_next_token as _sample_next_token,
)
from src.utils.generation_utils import multimodal_decode
from src.utils.nar_checkpoint_utils import (
    infer_vertical_from_state as _infer_vertical_from_state,
    load_state_with_fuse_compat as _load_state_with_fuse_compat,
    load_nar_metadata as _load_nar_metadata,
    resolve_nar_ckpt_path as _resolve_nar_ckpt_path,
    safe_torch_load as _safe_torch_load,
)


@torch.no_grad()
def generate(
    cfg,
    model,
    tokenizer,
    input_ids,
    unconditional_ids,
    full_unconditional_ids=None,
    force_same_image_size=True,
) -> Generator[Any, None, None]:
    if getattr(cfg, "streaming", False):
        yield from streaming_generate(
            cfg,
            model,
            tokenizer,
            input_ids,
            unconditional_ids,
            force_same_image_size=force_same_image_size,
        )
    else:
        yield non_streaming_generate(
            cfg,
            model,
            tokenizer,
            input_ids,
            unconditional_ids,
            force_same_image_size=force_same_image_size,
        )

def _get_nar_wrapper(cfg, model) -> EmuNAR:
    if hasattr(model, "nar_wrapper"):
        return model.nar_wrapper
    nar_ckpt_path = _resolve_nar_ckpt_path(
        nar_ckpt_path=getattr(cfg, "nar_ckpt_path", ""),
        model_path=cfg.model_path,
        merge_dtype=getattr(cfg, "nar_merge_dtype", "bf16"),
        fsdp_wrap_policy=getattr(cfg, "nar_fsdp_wrap_policy", "transformer"),
        fsdp_min_params=int(getattr(cfg, "nar_fsdp_min_params", 1_000_000)),
        use_vertical_block=getattr(cfg, "nar_use_vertical_block", None),
        vertical_layers=int(getattr(cfg, "nar_vertical_layers", 0)),
    )

    state = _safe_torch_load(nar_ckpt_path)
    nar_metadata = _load_nar_metadata(nar_ckpt_path)
    model_config = model.config
    visual_token_offset = int(model_config.eoi_token_id) + 1
    inferred_use_vertical, inferred_vertical_layers = _infer_vertical_from_state(state)
    nar_use_vertical = getattr(cfg, "nar_use_vertical_block", None)
    if nar_use_vertical is None:
        use_vertical_block = inferred_use_vertical
    else:
        use_vertical_block = bool(nar_use_vertical)

    cfg_vertical_layers = int(getattr(cfg, "nar_vertical_layers", 0))
    if cfg_vertical_layers > 0:
        vertical_layers = cfg_vertical_layers
    elif inferred_vertical_layers > 0:
        vertical_layers = inferred_vertical_layers
    else:
        vertical_layers = int(getattr(model_config, "nar_vertical_layers", 1))
    if use_vertical_block and vertical_layers <= 0:
        vertical_layers = 1
    cfg_vertical_start_layer = int(getattr(cfg, "nar_vertical_start_layer", -1))
    if cfg_vertical_start_layer >= 0:
        vertical_start_layer = cfg_vertical_start_layer
    elif "vertical_start_layer" in nar_metadata:
        vertical_start_layer = int(nar_metadata["vertical_start_layer"])
    else:
        vertical_start_layer = int(getattr(model_config, "num_hidden_layers", 0))

    wrapper = EmuNAR(
        pretrained_backbone=model.model,
        vocab_size=model_config.vocab_size,
        hidden_size=model_config.hidden_size,
        pad_token_id=-100,
        mask_token_id=model_config.pad_token_id,
        visual_token_offset=visual_token_offset,
        use_vertical_block=use_vertical_block,
        vertical_layers=vertical_layers,
        vertical_start_layer=vertical_start_layer,
        lm_head=model.lm_head,
        split_backbone=bool(getattr(cfg, "nar_split_backbone", False)),
    )
    _load_state_with_fuse_compat(wrapper, state, load_desc="load NAR ckpt")
    wrapper = wrapper.to(next(model.parameters()).device)
    wrapper.eval()
    model.nar_wrapper = wrapper
    return wrapper


@torch.no_grad()
def non_streaming_generate(
    cfg,
    model,
    tokenizer,
    input_ids,
    unconditional_ids,
    force_same_image_size=True,
):
    special = _get_special_ids(cfg, tokenizer)
    boi_id = special["BOI"]
    img_id = special["IMG"]
    eoi_id = special["EOI"]
    eol_id = special["EOL"]
    eos_id = special["EOS"]

    visual_token_offset = int(model.config.eoi_token_id) + 1
    digit_token_ids = _get_digit_token_ids(tokenizer)

    max_new_tokens = cfg.sampling_params.get("max_new_tokens", 0)
    text_top_k = cfg.sampling_params.get("text_top_k", 0)
    text_top_p = cfg.sampling_params.get("text_top_p", 1.0)
    text_temp = cfg.sampling_params.get("text_temperature", 1.0)
    img_top_k = cfg.sampling_params.get("image_top_k", 0)
    img_top_p = cfg.sampling_params.get("image_top_p", 1.0)
    img_temp = cfg.sampling_params.get("image_temperature", 1.0)
    do_sample = bool(cfg.sampling_params.get("do_sample", True))

    cfg_scale = float(getattr(cfg, "classifier_free_guidance", 1.0))

    device = input_ids.device
    cond_ids = input_ids[0].tolist()
    uncond_ids = unconditional_ids[0].tolist()
    generated: List[int] = []

    nar_wrapper = _get_nar_wrapper(cfg, model)

    fixed_hw = None
    while max_new_tokens <= 0 or len(generated) < max_new_tokens:
        cond_tensor = torch.tensor([cond_ids], device=device, dtype=torch.long)
        logits = model(input_ids=cond_tensor, use_cache=False).logits[:, -1, :]
        logits[:, visual_token_offset:] = float("-inf")
        if do_sample:
            next_id = _sample_next_token(logits, text_temp, text_top_k, text_top_p)
            next_id = int(next_id.item())
        else:
            next_id = int(logits.argmax(dim=-1).item())
        cond_ids.append(next_id)
        uncond_ids.append(next_id)
        generated.append(next_id)

        if next_id == eos_id:
            break

        if next_id != boi_id:
            continue

        # Generate size tokens and IMG token autoregressively.
        size_tokens: List[int] = []
        if force_same_image_size and fixed_hw is not None:
            h_fix, w_fix = fixed_hw
            size_tokens = tokenizer.encode(f"{h_fix}*{w_fix}", add_special_tokens=False)
            for tok in size_tokens:
                cond_ids.append(tok)
                uncond_ids.append(tok)
                generated.append(tok)
            cond_ids.append(img_id)
            uncond_ids.append(img_id)
            generated.append(img_id)
        else:
            while True:
                cond_tensor = torch.tensor([cond_ids], device=device, dtype=torch.long)
                logits = model(input_ids=cond_tensor, use_cache=False).logits[:, -1, :]
                allowed_mask = torch.full_like(logits, float("-inf"))
                for tid in digit_token_ids:
                    allowed_mask[:, tid] = 0.0
                allowed_mask[:, img_id] = 0.0
                logits = logits + allowed_mask
                if do_sample:
                    next_id = _sample_next_token(logits, text_temp, text_top_k, text_top_p)
                    next_id = int(next_id.item())
                else:
                    next_id = int(logits.argmax(dim=-1).item())
                cond_ids.append(next_id)
                uncond_ids.append(next_id)
                generated.append(next_id)
                if next_id == img_id:
                    break
                size_tokens.append(next_id)
                if len(size_tokens) > 16:
                    cond_ids.append(img_id)
                    uncond_ids.append(img_id)
                    generated.append(img_id)
                    break

        # Parse height/width for NAR grid.
        try:
            if not size_tokens:
                size_tokens = _collect_hw_tokens(cond_ids, boi_id, img_id)
            height, width = _parse_hw_from_tokens(tokenizer, size_tokens)
        except Exception:
            tgt_h = getattr(cfg, "target_height", None)
            tgt_w = getattr(cfg, "target_width", None)
            if tgt_h is None or tgt_w is None:
                raise
            height, width = int(tgt_h), int(tgt_w)
        if force_same_image_size and fixed_hw is None:
            fixed_hw = (height, width)

        # NAR decode for visual tokens.
        prefix_ids = torch.tensor([cond_ids], device=device, dtype=torch.long)
        uncond_prefix = torch.tensor([uncond_ids], device=device, dtype=torch.long)
        grid = nar_wrapper.generate(
            height=height,
            width=width,
            device=device,
            text_input_ids=prefix_ids,
            unconditional_text_input_ids=uncond_prefix,
            cfg_scale=cfg_scale,
            temperature=img_temp,
            top_k=img_top_k,
            top_p=img_top_p,
            sample_logits=do_sample,
        )

        # Append visual tokens + row separators + EOI.
        for r in range(height):
            for c in range(width):
                tok = int(grid[r, c].item())
                cond_ids.append(tok)
                uncond_ids.append(tok)
                generated.append(tok)
            if r < height - 1:
                cond_ids.append(eol_id)
                uncond_ids.append(eol_id)
                generated.append(eol_id)
        cond_ids.append(eoi_id)
        uncond_ids.append(eoi_id)
        generated.append(eoi_id)

    return np.array(generated, dtype=np.int64)


@torch.no_grad()
def streaming_generate(
    cfg,
    model,
    tokenizer,
    input_ids,
    unconditional_ids,
    force_same_image_size=True,
):
    gen_ids = non_streaming_generate(
        cfg,
        model,
        tokenizer,
        input_ids,
        unconditional_ids,
        force_same_image_size=force_same_image_size,
    )
    full_ids = torch.cat([input_ids, torch.tensor([gen_ids], device=input_ids.device)], dim=1)
    decoded = tokenizer.batch_decode(full_ids, skip_special_tokens=False)[0]
    pattern = re.compile(
        rf"({re.escape(tokenizer.boi_token)}.*?{re.escape(tokenizer.eoi_token)})",
        re.DOTALL,
    )
    chunks = re.split(pattern, decoded)
    for c in chunks:
        if not c or not c.strip():
            continue
        if tokenizer.boi_token in c and tokenizer.eoi_token in c:
            yield {"type": "image", "image": c}
        else:
            yield {"type": "text", "text": c}
