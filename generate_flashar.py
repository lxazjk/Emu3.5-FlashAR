# -*- coding: utf-8 -*-

import argparse

import torch
from PIL import Image

from flashar.model import Emuflashar
from flashar.utils.text_utils import (
    build_image_prefix_tokens as _build_image_prefix_tokens,
    build_text_tokenizer as _build_text_tokenizer,
)
from src.emu3p5 import Emu3Config, Emu3ForCausalLM
from src.utils.flashar_checkpoint_utils import (
    infer_vertical_from_state,
    load_flashar_metadata,
    safe_torch_load,
)
from src.vision_tokenizer import build_vision_tokenizer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--tokenizer_path", required=True)
    p.add_argument("--vq_path", required=True)
    p.add_argument("--ckpt_path", required=True)
    p.add_argument("--height", type=int, required=True)
    p.add_argument("--width", type=int, required=True)
    p.add_argument("--prompt", type=str, default="")
    p.add_argument("--text_template", type=str, default="{text}")
    p.add_argument("--add_boi", action="store_true")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_k", type=int, default=0)
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument("--greedy", action="store_true")
    p.add_argument("--cfg_scale", type=float, default=1.0)
    p.add_argument("--out", type=str, default="flashar_out.png")
    p.add_argument("--visual_token_offset", type=int, default=-1)
    p.add_argument(
        "--use_vertical_block",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    p.add_argument("--vertical_layers", type=int, default=0)
    p.add_argument(
        "--vertical_start_layer",
        type=int,
        default=-1,
        help="Backbone layer index where the vertical branch starts. <0 means auto.",
    )
    p.add_argument("--split_backbone", action="store_true",
                   help="Split backbone at vertical_start_layer: shared layers first, then parallel H/V branches.")
    return p.parse_args()


def main():
    args = parse_args()
    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    torch_dtype = dtype_map[args.dtype]
    device = torch.device(args.device)

    cfg = Emu3Config.from_pretrained(args.model_path, trust_remote_code=True)
    visual_token_offset = args.visual_token_offset
    if visual_token_offset < 0:
        visual_token_offset = int(cfg.eoi_token_id) + 1
    backbone = Emu3ForCausalLM.from_pretrained(
        args.model_path, config=cfg, torch_dtype=torch_dtype, attn_implementation="eager"
    ).to(device)

    state = safe_torch_load(args.ckpt_path)
    flashar_metadata = load_flashar_metadata(args.ckpt_path)
    inferred_use_vertical_block, inferred_vertical_layers = infer_vertical_from_state(state)
    use_vertical_block = (
        bool(args.use_vertical_block)
        if args.use_vertical_block is not None
        else bool(flashar_metadata.get("use_vertical_block", inferred_use_vertical_block))
    )
    vertical_layers = (
        int(args.vertical_layers)
        if int(args.vertical_layers) > 0
        else int(
            flashar_metadata.get(
                "vertical_layers",
                inferred_vertical_layers if inferred_vertical_layers > 0 else (1 if use_vertical_block else 0),
            )
        )
    )
    if use_vertical_block and vertical_layers <= 0:
        vertical_layers = 1
    vertical_start_layer = (
        int(args.vertical_start_layer)
        if int(args.vertical_start_layer) >= 0
        else (
            int(flashar_metadata["vertical_start_layer"])
            if "vertical_start_layer" in flashar_metadata
            else int(cfg.num_hidden_layers)
        )
    )

    wrapper = Emuflashar(
        pretrained_backbone=backbone.model,
        vocab_size=cfg.vocab_size,
        hidden_size=cfg.hidden_size,
        pad_token_id=-100,
        mask_token_id=cfg.pad_token_id,
        visual_token_offset=visual_token_offset,
        use_vertical_block=use_vertical_block,
        vertical_layers=vertical_layers,
        vertical_start_layer=vertical_start_layer,
        split_backbone=args.split_backbone,
    ).to(device=device, dtype=torch_dtype)

    wrapper.load_state_dict(state, strict=True)
    wrapper.eval()

    tokenizer = None
    text_ids = None
    prompt = args.text_template.replace("{text}", args.prompt)
    should_build_prefix = bool(prompt) or args.add_boi
    if should_build_prefix:
        tokenizer = _build_text_tokenizer(args.tokenizer_path)
        prefix_ids = tokenizer.encode(prompt, add_special_tokens=False)
        if args.add_boi and hasattr(tokenizer, "boi_token"):
            prefix_ids = list(prefix_ids) + _build_image_prefix_tokens(tokenizer, args.height, args.width)
        if prefix_ids:
            text_ids = torch.tensor(prefix_ids, dtype=torch.long, device=device).unsqueeze(0)
    if text_ids is None:
        raise ValueError(
            "flashar generation now requires a non-empty prefix for KV-cache decoding. "
            "Provide --prompt, a non-empty --text_template, or --add_boi."
        )
    
    with torch.no_grad():
        tokens = wrapper.generate(
            height=args.height,
            width=args.width,
            device=device,
            text_input_ids=text_ids,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            sample_logits=not args.greedy,
            cfg_scale=args.cfg_scale,
        )

    vq = build_vision_tokenizer("ibq", args.vq_path, device=device)
    embed_dim = getattr(vq.quantize, "e_dim", 256)
    with torch.no_grad():
        vq_tokens = (tokens - visual_token_offset).clamp_min(0)
        img = vq.decode_code(vq_tokens[None].to(device), shape=(1, args.height, args.width, embed_dim)).float()

    img = img[0].permute(1, 2, 0)
    img = ((img + 1.0) * 127.5).clamp(0, 255).cpu().numpy().astype("uint8")
    Image.fromarray(img).save(args.out)
    print("saved", args.out)


if __name__ == "__main__":
    main()
