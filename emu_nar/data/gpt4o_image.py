# -*- coding: utf-8 -*-
# GPT4o-Image (T2I) dataset and image encoding utilities.

from __future__ import annotations

import json
import os.path as osp
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
from PIL import Image, ImageFile
import torch

from src.utils.input_utils import smart_resize

ImageFile.LOAD_TRUNCATED_IMAGES = True


# ---------------------------------------------------------------------------
# Text / image helpers
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    return text.replace("<image>", "").replace("<img>", "").strip()


def _round_to_multiple(value: int, multiple: int) -> int:
    return int((value + multiple // 2) // multiple * multiple)


def resize_image(image: Image.Image, image_area: int, image_size: int) -> Image.Image:
    if image_size > 0:
        size = _round_to_multiple(image_size, 16)
        return image.resize((size, size), Image.BICUBIC)
    area = image_area if image_area > 0 else 512 * 512
    return smart_resize(image, area=area, ds_factor=16)


def _image_to_tensor(image: Image.Image) -> torch.Tensor:
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
    heights, widths = [], []
    for image in images:
        image = image.convert("RGB")
        image = resize_image(image, image_area=image_area, image_size=image_size)
        processed.append(image)
        heights.append(image.height)
        widths.append(image.width)

    if len(set(heights)) != 1 or len(set(widths)) != 1:
        if len(processed) > 1:
            raise ValueError(
                "All images in a batch must share the same size. "
                "Use --image_size or batch_size=1."
            )

    batch = torch.stack([_image_to_tensor(img) for img in processed], dim=0)
    batch = batch.to(device=device, dtype=dtype)
    _, _, info = vq_model.encode(batch)
    tokens = info[-1]

    token_h = batch.size(2) // 16
    token_w = batch.size(3) // 16
    tokens = tokens.view(batch.size(0), token_h, token_w).long()

    if visual_token_offset:
        tokens = tokens + visual_token_offset

    return tokens, token_h, token_w


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {"images": [b["image"] for b in batch], "texts": [b["text"] for b in batch]}


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class GPT4oImageDataset(torch.utils.data.IterableDataset):
    def __init__(
        self,
        json_path: str,
        image_root: str,
        rank: int = 0,
        world_size: int = 1,
    ) -> None:
        super().__init__()
        with open(json_path, "r", encoding="utf-8") as f:
            self.items = json.load(f)
        self.image_root = image_root
        self.rank = rank
        self.world_size = world_size

    def _iter_items(self) -> Iterable[dict]:
        items = self.items
        if self.world_size > 1:
            items = items[self.rank :: self.world_size]
        return items

    def __iter__(self):
        for item in self._iter_items():
            if not isinstance(item, dict):
                continue
            prompt = clean_text(str(item.get("input_prompt") or ""))
            output_image = item.get("output_image")
            if not prompt or not output_image:
                continue
            image_path = output_image
            if not osp.isabs(image_path):
                image_path = osp.join(self.image_root, output_image)
            try:
                image = Image.open(image_path).convert("RGB")
            except Exception:
                continue
            yield {"image": image, "text": prompt}


__all__ = [
    "GPT4oImageDataset",
    "clean_text",
    "collate_fn",
    "encode_images",
    "resize_image",
]
