# -*- coding: utf-8 -*-
# Check whether AR self-attention in visual token segment prefers structured positions
# such as the previous row same column (i-1, j).

import argparse
import json
import os
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image
from transformers import AutoTokenizer

from src.emu3p5 import Emu3ForCausalLM, Emu3Config
from src.vision_tokenizer import build_vision_tokenizer
from src.utils.input_utils import smart_resize


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--tokenizer_path", type=str, required=True)
    parser.add_argument("--vq_path", type=str, required=True)
    parser.add_argument("--vq_type", type=str, default="ibq")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--image", type=str, default="")
    parser.add_argument("--prompt", type=str, default="Generate an image.")
    parser.add_argument("--image_area", type=int, default=1048576)
    parser.add_argument("--max_height", type=int, default=16)
    parser.add_argument("--max_width", type=int, default=16)
    parser.add_argument("--random_height", type=int, default=16)
    parser.add_argument("--random_width", type=int, default=16)
    parser.add_argument("--save_dir", type=str, default="./outputs/nar_attention_check")
    parser.add_argument("--bias_threshold", type=float, default=1.2)
    return parser.parse_args()


def encode_image_tokens(image_path: str, vq_model, image_area: int) -> torch.Tensor:
    image = Image.open(image_path).convert("RGB")
    image = smart_resize(image, image_area)
    w, h = image.size
    device = next(vq_model.parameters()).device
    dtype = next(vq_model.parameters()).dtype
    image_t = torch.tensor((np.array(image) / 127.5 - 1.0)).to(device, dtype).permute(2, 0, 1)
    _, _, token = vq_model.encode(image_t[None])
    token = token[-1].view(h // 16, w // 16)
    return token


def build_visual_sequence(grid: torch.Tensor, eol_id: int) -> List[int]:
    h, w = grid.shape
    seq: List[int] = []
    for r in range(h):
        seq.extend(grid[r].tolist())
        if r < h - 1:
            seq.append(eol_id)
    return seq


def build_sequence(
    tokenizer,
    prompt: str,
    grid: torch.Tensor,
    special_ids: Dict[str, int],
) -> Tuple[torch.Tensor, int, torch.Tensor]:
    h, w = grid.shape
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    if not prompt_ids or prompt_ids[0] != special_ids["BOS"]:
        prompt_ids = [special_ids["BOS"]] + prompt_ids

    hw_tokens = tokenizer.encode(f"{h}*{w}", add_special_tokens=False)
    prefix = [special_ids["BOI"]] + hw_tokens + [special_ids["IMG"]]

    visual_seq = build_visual_sequence(grid, special_ids["EOL"])

    full_ids = torch.tensor(
        [prompt_ids + prefix + visual_seq + [special_ids["EOI"]]],
        dtype=torch.long,
    )
    visual_offset = len(prompt_ids) + len(prefix)
    return full_ids, visual_offset, torch.tensor(visual_seq, dtype=torch.long)


def build_visual_index_map(height: int, width: int, visual_offset: int) -> Tuple[torch.Tensor, torch.Tensor]:
    idx_map = torch.full((height, width), -1, dtype=torch.long)
    visual_indices: List[int] = []
    for r in range(height):
        for c in range(width):
            idx = visual_offset + r * (width + 1) + c
            idx_map[r, c] = idx
            visual_indices.append(idx)
    return idx_map, torch.tensor(visual_indices, dtype=torch.long)


def compute_bias_metrics(attn_mean: torch.Tensor, idx_map: torch.Tensor, visual_indices: torch.Tensor) -> Dict[str, float]:
    height, width = idx_map.shape
    idx_curr_above = []
    idx_above = []
    idx_curr_left = []
    idx_left = []

    for r in range(height):
        for c in range(width):
            idx = idx_map[r, c].item()
            if r > 0:
                idx_curr_above.append(idx)
                idx_above.append(idx_map[r - 1, c].item())
            if c > 0:
                idx_curr_left.append(idx)
                idx_left.append(idx_map[r, c - 1].item())

    metrics = {}
    if idx_curr_above:
        idx_curr_above_t = torch.tensor(idx_curr_above, device=attn_mean.device, dtype=torch.long)
        idx_above_t = torch.tensor(idx_above, device=attn_mean.device, dtype=torch.long)
        above_mean = attn_mean[idx_curr_above_t, idx_above_t].mean().item()

        prev_means = []
        for idx in idx_curr_above_t.tolist():
            prev_mask = visual_indices < idx
            if prev_mask.any():
                prev_mean = attn_mean[idx, visual_indices[prev_mask]].mean().item()
                prev_means.append(prev_mean)
        prev_visual_mean = float(np.mean(prev_means)) if prev_means else 0.0
        metrics["above_mean"] = above_mean
        metrics["prev_visual_mean"] = prev_visual_mean
        metrics["above_ratio"] = above_mean / (prev_visual_mean + 1e-9)
    else:
        metrics["above_mean"] = 0.0
        metrics["prev_visual_mean"] = 0.0
        metrics["above_ratio"] = 0.0

    if idx_curr_left:
        idx_curr_left_t = torch.tensor(idx_curr_left, device=attn_mean.device, dtype=torch.long)
        idx_left_t = torch.tensor(idx_left, device=attn_mean.device, dtype=torch.long)
        left_mean = attn_mean[idx_curr_left_t, idx_left_t].mean().item()
        metrics["left_mean"] = left_mean
        if metrics["prev_visual_mean"] > 0:
            metrics["left_ratio"] = left_mean / (metrics["prev_visual_mean"] + 1e-9)
        else:
            metrics["left_ratio"] = 0.0
    else:
        metrics["left_mean"] = 0.0
        metrics["left_ratio"] = 0.0

    return metrics


def save_heatmap(attn_mean: torch.Tensor, idx_map: torch.Tensor, out_path: str, title: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("[WARN] matplotlib not available, skipping heatmap.")
        return

    height, width = idx_map.shape
    heatmap = np.full((height, width), np.nan, dtype=np.float32)
    for r in range(1, height):
        for c in range(width):
            idx = idx_map[r, c].item()
            idx_above = idx_map[r - 1, c].item()
            heatmap[r, c] = float(attn_mean[idx, idx_above].item())

    plt.figure(figsize=(6, 5))
    plt.imshow(heatmap, cmap="viridis")
    plt.colorbar()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main() -> None:
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    torch_dtype = dtype_map[args.dtype]
    device = torch.device(args.device)

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_path,
        special_tokens_file=f"{args.tokenizer_path}/emu3_vision_tokens.txt",
        trust_remote_code=True,
    )
    tokenizer.bos_token = "<|extra_203|>"
    tokenizer.eos_token = "<|extra_204|>"
    tokenizer.pad_token = "<|endoftext|>"
    tokenizer.eol_token = "<|extra_200|>"
    tokenizer.img_token = "<|image token|>"
    tokenizer.boi_token = "<|image start|>"
    tokenizer.eoi_token = "<|image end|>"

    special_ids = {
        "BOS": tokenizer.encode(tokenizer.bos_token)[0],
        "EOS": tokenizer.encode(tokenizer.eos_token)[0],
        "PAD": tokenizer.encode(tokenizer.pad_token)[0],
        "EOL": tokenizer.encode(tokenizer.eol_token)[0],
        "IMG": tokenizer.encode(tokenizer.img_token)[0],
        "BOI": tokenizer.encode(tokenizer.boi_token)[0],
        "EOI": tokenizer.encode(tokenizer.eoi_token)[0],
    }

    model_config = Emu3Config.from_pretrained(args.model_path, trust_remote_code=True)
    model = Emu3ForCausalLM.from_pretrained(
        args.model_path,
        config=model_config,
        torch_dtype=torch_dtype,
        attn_implementation="eager",
    ).to(device)
    model.eval()

    vq_model = build_vision_tokenizer(args.vq_type, args.vq_path, device=args.device)

    if args.image:
        grid = encode_image_tokens(args.image, vq_model, args.image_area)
    else:
        visual_start = tokenizer.encode("<|visual token 000000|>", add_special_tokens=False)[0]
        grid = torch.randint(
            visual_start,
            model.config.vocab_size,
            (args.random_height, args.random_width),
            dtype=torch.long,
            device=device,
        )

    if args.max_height > 0:
        grid = grid[: args.max_height]
    if args.max_width > 0:
        grid = grid[:, : args.max_width]

    input_ids, visual_offset, _ = build_sequence(tokenizer, args.prompt, grid, special_ids)
    input_ids = input_ids.to(device)

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids, dtype=torch.long),
            output_attentions=True,
            use_cache=False,
            return_dict=True,
        )

    if outputs.attentions is None:
        raise RuntimeError("Model did not return attentions. Try attn_implementation='eager'.")

    attn_stack = torch.stack(outputs.attentions, dim=0)  # L x B x H x S x S
    attn_stack = attn_stack[:, 0]  # L x H x S x S
    attn_mean = attn_stack.mean(dim=(0, 1)).float()  # S x S

    height, width = grid.shape
    idx_map, visual_indices = build_visual_index_map(height, width, visual_offset)
    idx_map = idx_map.to(attn_mean.device)
    visual_indices = visual_indices.to(attn_mean.device)

    metrics = compute_bias_metrics(attn_mean, idx_map, visual_indices)
    metrics["height"] = int(height)
    metrics["width"] = int(width)
    metrics["bias_threshold"] = float(args.bias_threshold)
    metrics["bias_toward_above"] = metrics["above_ratio"] > args.bias_threshold

    per_layer = []
    for layer_idx in range(attn_stack.shape[0]):
        layer_attn = attn_stack[layer_idx].mean(dim=0).float()
        layer_metrics = compute_bias_metrics(layer_attn, idx_map, visual_indices)
        layer_metrics["layer"] = int(layer_idx)
        per_layer.append(layer_metrics)
    metrics["per_layer"] = per_layer

    summary_path = os.path.join(args.save_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=True)

    heatmap_path = os.path.join(args.save_dir, "above_attention_heatmap.png")
    save_heatmap(attn_mean, idx_map, heatmap_path, "Attention to Previous Row Same Column")

    print("[INFO] Summary saved:", summary_path)
    print("[INFO] Heatmap saved:", heatmap_path)
    print(
        "[RESULT] above_mean={:.6f} prev_visual_mean={:.6f} left_mean={:.6f} above_ratio={:.3f} left_ratio={:.3f}".format(
            metrics["above_mean"],
            metrics["prev_visual_mean"],
            metrics["left_mean"],
            metrics["above_ratio"],
            metrics["left_ratio"],
        )
    )
    print("[RESULT] bias_toward_above:", metrics["bias_toward_above"])


if __name__ == "__main__":
    main()
