# -*- coding: utf-8 -*-
# Pretokenized tar dataset: .pt tokens + .txt prompts.

from __future__ import annotations

import io
import os.path as osp
import tarfile
from typing import Iterable, List

import torch


class PretokShardDataset(torch.utils.data.IterableDataset):
    def __init__(
        self,
        shard_paths: Iterable[str],
        rank: int = 0,
        world_size: int = 1,
    ) -> None:
        super().__init__()
        self.shard_paths = list(shard_paths)
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
                bucket: dict[str, dict[str, bytes]] = {}
                for member in tf:
                    if not member.isfile():
                        continue
                    name = osp.basename(member.name)
                    stem, ext = osp.splitext(name)
                    ext = ext.lower()
                    fobj = tf.extractfile(member)
                    if fobj is None:
                        continue
                    if ext == ".pt":
                        bucket.setdefault(stem, {})["pt"] = fobj.read()
                    elif ext == ".txt":
                        bucket.setdefault(stem, {})["text"] = fobj.read()
                    else:
                        continue

                    entry = bucket.get(stem)
                    if entry is None:
                        continue
                    if "pt" not in entry or "text" not in entry:
                        continue

                    tokens = torch.load(io.BytesIO(entry["pt"]), map_location="cpu")
                    text = entry["text"].decode("utf-8", errors="ignore")
                    del bucket[stem]
                    yield {"tokens": tokens, "text": text}


def collate_pretok(batch: List[dict]) -> dict:
    tokens = [b["tokens"] for b in batch]
    texts = [b["text"] for b in batch]
    shapes = {tuple(t.shape) for t in tokens}
    if len(shapes) != 1:
        raise ValueError("All token grids in a batch must share the same shape.")
    return {"tokens": torch.stack(tokens, dim=0), "texts": texts}


__all__ = ["PretokShardDataset", "collate_pretok"]
