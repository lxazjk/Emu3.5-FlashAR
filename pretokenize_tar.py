# -*- coding: utf-8 -*-

import argparse
import glob
import io
import os
import tarfile
from typing import Dict

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from tqdm import tqdm

from src.vision_tokenizer import build_vision_tokenizer
from src.utils.input_utils import smart_resize


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--split", type=str, default="train")
    p.add_argument("--vq_path", required=True)
    p.add_argument("--vq_type", type=str, default="ibq")
    p.add_argument("--vq_device", type=str, default="auto")
    p.add_argument("--image_area", type=int, default=1048576)
    p.add_argument("--grid_height", type=int, default=0)
    p.add_argument("--grid_width", type=int, default=0)
    p.add_argument("--max_height", type=int, default=0)
    p.add_argument("--max_width", type=int, default=0)
    return p.parse_args()


def _encode_image_to_tokens(
    image: Image.Image,
    vq_model,
    image_area: int,
    grid_height: int,
    grid_width: int,
    max_height: int,
    max_width: int,
) -> torch.Tensor:
    if grid_height > 0 and grid_width > 0:
        image = image.resize((grid_width * 16, grid_height * 16), Image.BICUBIC)
    else:
        image = smart_resize(image, image_area)

    w, h = image.size
    device = next(vq_model.parameters()).device
    dtype = next(vq_model.parameters()).dtype
    image_t = torch.tensor((np.array(image) / 127.5 - 1.0)).to(device, dtype).permute(2, 0, 1)
    with torch.no_grad():
        _, _, token = vq_model.encode(image_t[None])
        token = token[-1].view(h // 16, w // 16)

    if max_height > 0:
        token = token[:max_height]
    if max_width > 0:
        token = token[:, :max_width]

    return token.cpu()


def _write_member(tar_out: tarfile.TarFile, name: str, data: bytes) -> None:
    info = tarfile.TarInfo(name=name)
    info.size = len(data)
    tar_out.addfile(info, io.BytesIO(data))


def main():
    args = parse_args()
    rank = 0
    world_size = 1
    local_rank = 0
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        if not dist.is_initialized():
            dist.init_process_group("nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))

    in_split = os.path.join(args.input_dir, args.split)
    out_split = os.path.join(args.output_dir, args.split)
    os.makedirs(out_split, exist_ok=True)

    shards = sorted(glob.glob(os.path.join(in_split, "*.tar")))
    if not shards:
        raise FileNotFoundError(f"No tar shards found under {in_split}")
    if world_size > 1:
        shards = shards[rank::world_size]

    vq_device = args.vq_device
    if vq_device in ("auto", "cuda"):
        vq_device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    vq_model = build_vision_tokenizer(args.vq_type, args.vq_path, device=vq_device)
    vq_model.eval()

    it = tqdm(shards, desc=f"rank {rank}") if rank == 0 else shards
    for shard in it:
        out_path = os.path.join(out_split, os.path.basename(shard))
        if os.path.exists(out_path):
            print(f"[INFO] skip existing {out_path}")
            continue
        print(f"[INFO] processing {shard} -> {out_path}")
        with tarfile.open(shard, "r:*") as tf, tarfile.open(out_path, "w") as out:
            bucket: Dict[str, Dict[str, bytes]] = {}
            for member in tf:
                if not member.isfile():
                    continue
                name = os.path.basename(member.name)
                stem, ext = os.path.splitext(name)
                ext = ext.lower()
                fobj = tf.extractfile(member)
                if fobj is None:
                    continue
                if ext in (".jpg", ".jpeg", ".png"):
                    bucket.setdefault(stem, {})["image"] = fobj.read()
                elif ext in (".txt", ".caption", ".caption.txt"):
                    bucket.setdefault(stem, {})["text"] = fobj.read()
                else:
                    continue

                entry = bucket.get(stem)
                if entry is None:
                    continue
                if "image" not in entry or "text" not in entry:
                    continue

                image = Image.open(io.BytesIO(entry["image"])).convert("RGB")
                tokens = _encode_image_to_tokens(
                    image,
                    vq_model,
                    args.image_area,
                    args.grid_height,
                    args.grid_width,
                    args.max_height,
                    args.max_width,
                )
                buf = io.BytesIO()
                torch.save(tokens, buf)
                _write_member(out, f"{stem}.pt", buf.getvalue())
                _write_member(out, f"{stem}.txt", entry["text"])
                del bucket[stem]


if __name__ == "__main__":
    main()
