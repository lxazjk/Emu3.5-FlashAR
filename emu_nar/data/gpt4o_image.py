# -*- coding: utf-8 -*-
# ShareGPT-4o-Image (T2I) dataset utilities.

from __future__ import annotations

import json
import os.path as osp
from typing import Iterable, List

from PIL import Image
import torch

from .infinity_mm import clean_text


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


__all__ = ["GPT4oImageDataset"]
