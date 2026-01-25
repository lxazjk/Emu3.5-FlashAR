# -*- coding: utf-8 -*-
# Infinity-MM webdataset parsing and image preprocessing utilities.

from __future__ import annotations

import io
import json
import os.path as osp
import tarfile
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
from PIL import Image, ImageFile
import torch

from src.utils.input_utils import smart_resize

ImageFile.LOAD_TRUNCATED_IMAGES = True

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def clean_text(text: str) -> str:
    return text.replace("<image>", "").replace("<img>", "").strip()


def extract_text_from_entry(entry: Dict[str, Any], source: str) -> str:
    conversations = entry.get("conversations") or []

    def role_name(turn: Dict[str, Any]) -> str:
        return (turn.get("from") or turn.get("role") or "").lower()

    def collect(role_set: set[str]) -> List[str]:
        texts = []
        for turn in conversations:
            if role_name(turn) in role_set and turn.get("value"):
                texts.append(str(turn.get("value")))
        return texts

    assistant_roles = {"gpt", "assistant"}
    human_roles = {"human", "user"}
    text = ""
    if conversations:
        if source == "assistant":
            texts = collect(assistant_roles)
            text = texts[-1] if texts else ""
        elif source == "human":
            texts = collect(human_roles)
            text = texts[-1] if texts else ""
        elif source == "both":
            parts = []
            for turn in conversations:
                if turn.get("value"):
                    role = role_name(turn)
                    if role in assistant_roles or role in human_roles:
                        parts.append(str(turn.get("value")))
            text = "\n".join(parts)

    if not text:
        for key in ("caption", "text", "summary", "title"):
            if entry.get(key):
                text = str(entry.get(key))
                break

    return clean_text(text)


def round_to_multiple(value: int, multiple: int) -> int:
    return int((value + multiple // 2) // multiple * multiple)


def resize_image(image: Image.Image, image_area: int, image_size: int) -> Image.Image:
    if image_size > 0:
        size = round_to_multiple(image_size, 16)
        return image.resize((size, size), Image.BICUBIC)
    area = image_area if image_area > 0 else 512 * 512
    return smart_resize(image, area=area, ds_factor=16)


def image_to_tensor(image: Image.Image) -> torch.Tensor:
    array = np.array(image, dtype=np.float32)
    if array.ndim == 2:
        array = np.stack([array, array, array], axis=-1)
    if array.shape[2] > 3:
        array = array[:, :, :3]
    return torch.from_numpy(array).permute(2, 0, 1) / 127.5 - 1.0


@torch.no_grad()
def encode_images(
    vq_model,
    images: List[Image.Image],
    image_area: int,
    image_size: int,
    device: torch.device,
    dtype: torch.dtype,
    visual_token_offset: int,
) -> Tuple[torch.Tensor, int, int]:
    processed = []
    heights = []
    widths = []
    for image in images:
        image = image.convert("RGB")
        image = resize_image(image, image_area=image_area, image_size=image_size)
        processed.append(image)
        heights.append(image.height)
        widths.append(image.width)

    if len(set(heights)) != 1 or len(set(widths)) != 1:
        if len(processed) > 1:
            raise ValueError(
                "All images in a batch must share the same size. Use --image_size or batch_size=1."
            )

    batch = torch.stack([image_to_tensor(img) for img in processed], dim=0)
    batch = batch.to(device=device, dtype=dtype)
    _, _, info = vq_model.encode(batch)
    tokens = info[-1]

    token_h = batch.size(2) // 16
    token_w = batch.size(3) // 16
    tokens = tokens.view(batch.size(0), token_h, token_w).long()

    if visual_token_offset:
        tokens = tokens + visual_token_offset

    return tokens, token_h, token_w


class InfinityMMShardDataset(torch.utils.data.IterableDataset):
    def __init__(
        self,
        shard_paths: Iterable[str],
        text_source: str,
        rank: int = 0,
        world_size: int = 1,
    ) -> None:
        super().__init__()
        self.shard_paths = list(shard_paths)
        self.text_source = text_source
        self.rank = rank
        self.world_size = world_size

    def _iter_shards(self) -> List[str]:
        shards = sorted(self.shard_paths)
        if self.world_size > 1:
            shards = shards[self.rank :: self.world_size]
        worker = torch.utils.data.get_worker_info()
        if worker is not None and worker.num_workers > 1:
            shards = shards[worker.id :: worker.num_workers]
        return shards

    def __iter__(self):
        for shard in self._iter_shards():
            with tarfile.open(shard, "r:*") as tf:
                bucket: Dict[str, Dict[str, bytes]] = {}
                for member in tf:
                    if not member.isfile():
                        continue
                    name = osp.basename(member.name)
                    stem, ext = osp.splitext(name)
                    ext = ext.lower()
                    fobj = tf.extractfile(member)
                    if fobj is None:
                        continue
                    if ext == ".json":
                        bucket.setdefault(stem, {})["json"] = fobj.read()
                    elif ext in IMAGE_EXTS:
                        bucket.setdefault(stem, {})["image"] = fobj.read()
                    else:
                        continue

                    entry = bucket.get(stem)
                    if entry is None or "json" not in entry or "image" not in entry:
                        continue

                    try:
                        meta = json.loads(entry["json"].decode("utf-8"))
                    except Exception:
                        del bucket[stem]
                        continue

                    text = extract_text_from_entry(meta, self.text_source)
                    if not text:
                        del bucket[stem]
                        continue

                    try:
                        image = Image.open(io.BytesIO(entry["image"])).convert("RGB")
                    except Exception:
                        del bucket[stem]
                        continue

                    del bucket[stem]
                    yield {"image": image, "text": text}


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    images = [b["image"] for b in batch]
    texts = [b["text"] for b in batch]
    return {"images": images, "texts": texts}


__all__ = [
    "IMAGE_EXTS",
    "InfinityMMShardDataset",
    "collate_fn",
    "clean_text",
    "encode_images",
    "extract_text_from_entry",
    "image_to_tensor",
    "resize_image",
    "round_to_multiple",
]
