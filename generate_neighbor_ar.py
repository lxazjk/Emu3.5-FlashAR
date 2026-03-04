# -*- coding: utf-8 -*-

import argparse
import os.path as osp

import torch
from PIL import Image
from transformers import AutoTokenizer

from src.emu3p5 import Emu3Config, Emu3ForCausalLM
from src.vision_tokenizer import build_vision_tokenizer
from emu_nar.modeling_emu_nar import EmuNAR
from emu_nar.lora import apply_lora_to_backbone, apply_progressive_lora_to_backbone


def _build_text_tokenizer(tokenizer_path: str):
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        special_tokens_file=osp.join(tokenizer_path, "emu3_vision_tokens.txt"),
        trust_remote_code=True,
    )
    tokenizer.bos_token = "<|extra_203|>"
    tokenizer.eos_token = "<|extra_204|>"
    tokenizer.pad_token = "<|endoftext|>"
    tokenizer.eol_token = "<|extra_200|>"
    tokenizer.eof_token = "<|extra_201|>"
    tokenizer.tms_token = "<|extra_202|>"
    tokenizer.img_token = "<|image token|>"
    tokenizer.boi_token = "<|image start|>"
    tokenizer.eoi_token = "<|image end|>"
    tokenizer.bss_token = "<|extra_100|>"
    tokenizer.ess_token = "<|extra_101|>"
    tokenizer.bog_token = "<|extra_60|>"
    tokenizer.eog_token = "<|extra_61|>"
    tokenizer.boc_token = "<|extra_50|>"
    tokenizer.eoc_token = "<|extra_51|>"
    return tokenizer


def _build_image_prefix_tokens(tokenizer, height: int, width: int) -> list[int]:
    boi_id = tokenizer.encode(tokenizer.boi_token, add_special_tokens=False)[0]
    img_id = tokenizer.encode(tokenizer.img_token, add_special_tokens=False)[0]
    hw_ids = tokenizer.encode(f"{height}*{width}", add_special_tokens=False)
    return [boi_id, *hw_ids, img_id]


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
    p.add_argument("--out", type=str, default="nar_out.png")
    p.add_argument("--visual_token_offset", type=int, default=-1)
    p.add_argument(
        "--use_vertical_block",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    p.add_argument("--vertical_layers", type=int, default=1)
    p.add_argument("--lora_layers", type=int, default=0)
    p.add_argument("--lora_r", type=int, default=0)
    p.add_argument("--lora_r_min", type=int, default=0)
    p.add_argument("--lora_alpha", type=float, default=0.0)
    p.add_argument("--lora_dropout", type=float, default=0.0)
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

    if args.lora_r > 0 and args.lora_layers > 0:
        if args.lora_r_min > 0:
            alpha_scale = (args.lora_alpha if args.lora_alpha > 0 else float(args.lora_r)) / float(
                args.lora_r
            )
            apply_progressive_lora_to_backbone(
                backbone,
                num_layers=args.lora_layers,
                r_min=args.lora_r_min,
                r_max=args.lora_r,
                alpha_scale=alpha_scale,
                dropout=args.lora_dropout,
            )
        else:
            lora_alpha = args.lora_alpha if args.lora_alpha > 0 else float(args.lora_r)
            apply_lora_to_backbone(
                backbone,
                num_layers=args.lora_layers,
                r=args.lora_r,
                alpha=lora_alpha,
                dropout=args.lora_dropout,
            )

    wrapper = EmuNAR(
        pretrained_backbone=backbone.model,
        vocab_size=cfg.vocab_size,
        hidden_size=cfg.hidden_size,
        num_heads=cfg.num_attention_heads,
        pad_token_id=-100,
        mask_token_id=cfg.pad_token_id,
        visual_token_offset=visual_token_offset,
        img_token_id=cfg.img_token_id,
        eol_token_id=cfg.eol_token_id,
        eoi_token_id=cfg.eoi_token_id,
        use_vertical_block=args.use_vertical_block,
        vertical_layers=args.vertical_layers,
    ).to(device=device, dtype=torch_dtype)

    state = torch.load(args.ckpt_path, map_location="cpu")
    wrapper.load_state_dict(state, strict=True)
    wrapper.eval()

    text_ids = None
    if args.prompt:
        tokenizer = _build_text_tokenizer(args.tokenizer_path)
        prompt = args.text_template.replace("{text}", args.prompt)
        text_ids = tokenizer.encode(prompt, add_special_tokens=False)
        if args.add_boi and hasattr(tokenizer, "boi_token"):
            text_ids = list(text_ids) + _build_image_prefix_tokens(tokenizer, args.height, args.width)
        text_ids = torch.tensor(text_ids, dtype=torch.long, device=device).unsqueeze(0)
    
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
