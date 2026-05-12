#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

from flashar.model import Emuflashar
from flashar.utils.text_utils import (
    build_image_prefix_tokens as build_image_prefix_tokens,
    build_text_tokenizer as build_text_tokenizer,
)
from src.emu3p5 import Emu3Config, Emu3ForCausalLM
from src.utils.flashar_checkpoint_utils import (
    infer_vertical_from_state,
    load_flashar_metadata,
    safe_torch_load,
)
from src.vision_tokenizer import build_vision_tokenizer


def parse_args():
    p = argparse.ArgumentParser(
        description="Generate images in the directory layout expected by GenEval."
    )
    p.add_argument("--metadata", default="datasets/geneval/prompts/evaluation_metadata.jsonl")
    p.add_argument("--outdir", default="outputs/geneval_flashar/images")
    p.add_argument("--model_path", default="./weights/Emu3.5-Image")
    p.add_argument("--tokenizer_path", default="./src/tokenizer_emu3_ibq")
    p.add_argument("--vq_path", default="./weights/Emu3.5-VisionTokenizer")
    p.add_argument("--ckpt_path", default="./outputs/flashar_finetune/flashar_final")
    p.add_argument("--height", type=int, default=32)
    p.add_argument("--width", type=int, default=32)
    p.add_argument(
        "--text_template",
        default="<|extra_203|>You are a helpful assistant for t2i task. USER: {text} ASSISTANT: <|extra_100|>",
    )
    p.add_argument("--samples_per_prompt", type=int, default=4)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, default=-1)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_k", type=int, default=0)
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument("--cfg_scale", type=float, default=2.5)
    p.add_argument("--greedy", action="store_true")
    p.add_argument("--seed", type=int, default=6666)
    p.add_argument("--visual_token_offset", type=int, default=-1)
    p.add_argument("--use_vertical_block", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--vertical_layers", type=int, default=0)
    p.add_argument("--vertical_start_layer", type=int, default=-1)
    p.add_argument("--split_backbone", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def read_metadata(path):
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def load_model(args):
    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    torch_dtype = dtype_map[args.dtype]
    device = torch.device(args.device)

    cfg = Emu3Config.from_pretrained(args.model_path, trust_remote_code=True)
    visual_token_offset = args.visual_token_offset
    if visual_token_offset < 0:
        visual_token_offset = int(cfg.eoi_token_id) + 1

    backbone = Emu3ForCausalLM.from_pretrained(
        args.model_path,
        config=cfg,
        torch_dtype=torch_dtype,
        attn_implementation="eager",
    ).to(device)
    backbone.eval()

    state = safe_torch_load(args.ckpt_path, mmap=True)
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

    tokenizer = build_text_tokenizer(args.tokenizer_path)
    vq = build_vision_tokenizer("ibq", args.vq_path, device=device)
    return cfg, wrapper, tokenizer, vq, visual_token_offset, device


def make_prefix(tokenizer, prompt, template, height, width, device):
    text = template.replace("{text}", prompt)
    prefix_ids = tokenizer.encode(text, add_special_tokens=False)
    if hasattr(tokenizer, "boi_token"):
        prefix_ids = list(prefix_ids) + build_image_prefix_tokens(tokenizer, height, width)
    if not prefix_ids:
        raise ValueError("empty generation prefix")
    return torch.tensor(prefix_ids, dtype=torch.long, device=device).unsqueeze(0)


@torch.no_grad()
def decode_image(vq, tokens, visual_token_offset, height, width, device):
    embed_dim = getattr(vq.quantize, "e_dim", 256)
    vq_tokens = (tokens - visual_token_offset).clamp_min(0)
    image = vq.decode_code(
        vq_tokens[None].to(device),
        shape=(1, height, width, embed_dim),
    ).float()
    image = image[0].permute(1, 2, 0)
    image = ((image + 1.0) * 127.5).clamp(0, 255).cpu().numpy().astype("uint8")
    return Image.fromarray(image)


def save_grid(sample_paths, out_path):
    images = [Image.open(p).convert("RGB") for p in sample_paths]
    if not images:
        return
    w, h = images[0].size
    grid = Image.new("RGB", (w * len(images), h))
    for idx, image in enumerate(images):
        grid.paste(image, (idx * w, 0))
    grid.save(out_path)


def main():
    args = parse_args()
    metadata = read_metadata(args.metadata)
    end = len(metadata) if args.end < 0 else min(args.end, len(metadata))
    selected = metadata[args.start:end]

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    _, wrapper, tokenizer, vq, visual_token_offset, device = load_model(args)
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
        "GenEval generation args: "
        f"height={args.height} width={args.width} "
        f"cfg_scale={args.cfg_scale} use_cfg={uncond_text_ids is not None} "
        f"samples_per_prompt={args.samples_per_prompt} "
        f"start={args.start} end={end} ckpt_path={args.ckpt_path}",
        flush=True,
    )

    pbar = tqdm(list(enumerate(selected, start=args.start)), desc="geneval generate", dynamic_ncols=True)
    for idx, item in pbar:
        prompt = item["prompt"]
        prompt_dir = outdir / f"{idx:05d}"
        samples_dir = prompt_dir / "samples"
        metadata_path = prompt_dir / "metadata.jsonl"
        samples_dir.mkdir(parents=True, exist_ok=True)

        with open(metadata_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

        sample_paths = []
        for sample_idx in range(args.samples_per_prompt):
            sample_path = samples_dir / f"{sample_idx:04d}.png"
            sample_paths.append(sample_path)
            if sample_path.exists() and not args.overwrite:
                continue
            seed = args.seed + idx * args.samples_per_prompt + sample_idx
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            text_ids = make_prefix(
                tokenizer,
                prompt,
                args.text_template,
                args.height,
                args.width,
                device,
            )
            tokens = wrapper.generate(
                height=args.height,
                width=args.width,
                device=device,
                text_input_ids=text_ids,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                sample_logits=not args.greedy,
                unconditional_text_input_ids=uncond_text_ids,
                cfg_scale=args.cfg_scale,
            )
            image = decode_image(vq, tokens, visual_token_offset, args.height, args.width, device)
            image.save(sample_path)

        save_grid(sample_paths, prompt_dir / "grid.png")


if __name__ == "__main__":
    main()
