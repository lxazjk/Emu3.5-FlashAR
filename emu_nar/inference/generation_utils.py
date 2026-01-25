# -*- coding: utf-8 -*-
# NAR-aware interleaved generation for Emu3.5.

import re
from typing import Generator, List, Dict, Any, Optional

from PIL import Image
import numpy as np
import torch

from .neighbor_ar_wrapper import NeighborARWrapper


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


def _get_special_ids(cfg, tokenizer):
    if hasattr(cfg, "special_token_ids"):
        return cfg.special_token_ids
    return {k: tokenizer.encode(v)[0] for k, v in cfg.special_tokens.items()}


def _get_digit_token_ids(tokenizer) -> set[int]:
    digits = [str(i) for i in range(10)] + ["*"]
    ids = set()
    for d in digits:
        ids.add(tokenizer.encode(d, add_special_tokens=False)[0])
    return ids


def _sample_next_token(logits, temperature, top_k, top_p):
    logits = logits / max(temperature, 1e-5)
    if top_k > 0:
        top_k = min(top_k, logits.size(-1))
        cutoff = torch.topk(logits, top_k)[0][..., -1, None]
        logits = torch.where(logits < cutoff, torch.full_like(logits, -float("inf")), logits)
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits = logits.masked_fill(indices_to_remove, -float("inf"))
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


def _parse_hw_from_tokens(tokenizer, hw_tokens: List[int]) -> tuple[int, int]:
    hw_str = tokenizer.decode(hw_tokens)
    h_str, w_str = hw_str.split("*")
    return int(h_str), int(w_str)


def _collect_hw_tokens(seq: List[int], boi_id: int, img_id: int) -> List[int]:
    last_boi = len(seq) - 1 - seq[::-1].index(boi_id)
    last_img = len(seq) - 1 - seq[::-1].index(img_id)
    return seq[last_boi + 1 : last_img]


def _get_nar_wrapper(cfg, model) -> NeighborARWrapper:
    if hasattr(model, "nar_wrapper"):
        return model.nar_wrapper
    nar_ckpt_path = getattr(cfg, "nar_ckpt_path", "")
    if not nar_ckpt_path:
        raise ValueError("nar_ckpt_path is required for NAR generation.")

    model_config = model.config
    visual_token_offset = int(model_config.eoi_token_id) + 1
    vertical_layers = getattr(cfg, "nar_vertical_layers", getattr(model_config, "nar_vertical_layers", 1))
    wrapper = NeighborARWrapper(
        pretrained_backbone=model.model,
        vocab_size=model_config.vocab_size,
        hidden_size=model_config.hidden_size,
        num_heads=model_config.num_attention_heads,
        pad_token_id=-100,
        mask_token_id=model_config.pad_token_id,
        visual_token_offset=visual_token_offset,
        img_token_id=model_config.img_token_id,
        eol_token_id=model_config.eol_token_id,
        eoi_token_id=model_config.eoi_token_id,
        use_vertical_block=getattr(cfg, "nar_use_vertical_block", False),
        vertical_layers=vertical_layers,
        lm_head=model.lm_head,
    )
    state = torch.load(nar_ckpt_path, map_location="cpu")
    wrapper.load_state_dict(state, strict=True)
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


@torch.no_grad()
def multimodal_decode(
    outputs,
    tokenizer,
    vision_tokenizer,
):
    outputs = outputs.replace("<|extra_101|>", "").replace("<|extra_204|>", "")
    pattern = re.compile(
        rf"({re.escape(tokenizer.bog_token)}.*?{re.escape(tokenizer.eog_token)}|"
        rf"{re.escape(tokenizer.boc_token)}.*?{re.escape(tokenizer.eoc_token)}|"
        rf"{re.escape(tokenizer.boi_token)}.*?{re.escape(tokenizer.eoi_token)})",
        re.DOTALL,
    )
    multimodal_output = []
    chunks = re.split(pattern, outputs)
    for c in chunks:
        if len(c) == 0:
            continue
        if tokenizer.boi_token in c and tokenizer.eoi_token in c:
            image = decode_image(c, tokenizer, vision_tokenizer)
            if image is not None:
                multimodal_output.append(("image", image))
        elif tokenizer.bog_token in c and tokenizer.eog_token in c:
            multimodal_output.append(
                ("global_cot", c.replace(tokenizer.bog_token, "").replace(tokenizer.eog_token, ""))
            )
        elif tokenizer.boc_token in c and tokenizer.eoc_token in c:
            multimodal_output.append(
                ("image_cot", c.replace(tokenizer.boc_token, "").replace(tokenizer.eoc_token, ""))
            )
        elif tokenizer.boi_token not in c and len(c.strip()) > 0:
            multimodal_output.append(("text", c))
    return multimodal_output


def decode_image(image_string, tokenizer, vision_tokenizer):
    image: List[List[int]] = []
    image_rows = re.split(re.escape(tokenizer.eol_token), image_string)
    for r in image_rows:
        token_ids = re.findall(r"<\|visual token (\d+)\|>", r)
        if len(token_ids) > 0:
            row_token = [int(m) for m in token_ids]
            image.append(row_token)
    try:
        image = torch.tensor(
            image, dtype=torch.long, device=next(iter(vision_tokenizer.parameters())).device
        )
        h, w = image.shape
        image = vision_tokenizer.decode_code(image[None], shape=(1, h, w, 256)).float()
        image = image[0].permute(1, 2, 0)
        image = Image.fromarray(
            ((image + 1.0) * 127.5).clamp(0, 255).detach().cpu().numpy().astype(np.uint8)
        )
        return image
    except Exception as ex:
        print(f"decode image failed {ex}")
        return None
