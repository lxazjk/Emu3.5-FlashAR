#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import io
import json
import os
import os.path as osp
import shutil
import tarfile
import tempfile
from dataclasses import dataclass
from typing import Dict, Optional

from PIL import Image
from tqdm import tqdm
import torch

from emu_nar.data.pretokenize_tar import (
    _encode_image_to_tokens,
    _write_member,
    extract_text_from_entry,
)
from src.vision_tokenizer import build_vision_tokenizer


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
TEXT_EXTS = {".txt", ".caption", ".caption.txt"}
PRETOK_TOKEN = "_pretok_"


@dataclass
class ShardProcessResult:
    samples_written: int
    skipped_invalid: int


@dataclass
class DatasetRunResult:
    processed_shards: int
    reclaimed_bytes: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Pretokenize raw tar shards one-by-one. After each shard is successfully "
            "converted and verified, remove the corresponding raw shard to reclaim space."
        )
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--input_split_dir",
        default="",
        help="Directory containing raw .tar shards for a single dataset.",
    )
    input_group.add_argument(
        "--input_root",
        default="",
        help="Root directory containing multiple raw dataset directories.",
    )
    parser.add_argument(
        "--output_root",
        default="",
        help=(
            "Output root for a single dataset. Final output is <output_root>/<split_name>/*.tar. "
            "If omitted, defaults to a sibling directory named <split_name>_pretok_<size>."
        ),
    )
    parser.add_argument(
        "--output_base_root",
        default="",
        help=(
            "Base root used with --input_root. Each dataset outputs to "
            "<output_base_root>/<dataset>_pretok_<size>/<dataset>/*.tar. "
            "Defaults to input_root."
        ),
    )
    parser.add_argument(
        "--split_name",
        default="",
        help="Output split subdirectory name. Defaults to basename(input_split_dir).",
    )
    parser.add_argument(
        "--dataset_pattern",
        default="*",
        help="Dataset directory glob used with --input_root.",
    )
    parser.add_argument("--pattern", default="*.tar", help="Shard glob pattern inside input_split_dir.")
    parser.add_argument(
        "--start_after",
        default="",
        help="Start processing after this shard basename (exclusive).",
    )
    parser.add_argument("--limit", type=int, default=0, help="Optional max number of shards to handle.")
    parser.add_argument(
        "--delete_mode",
        choices=["delete", "trash", "keep"],
        default="delete",
        help="How to handle raw shards after successful conversion/verification.",
    )
    parser.add_argument(
        "--trash_dir",
        default="",
        help="Used only when delete_mode=trash. Defaults to <input_split_dir>/.trash_done.",
    )
    parser.add_argument(
        "--cleanup_existing_done",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "If a valid pretokenized output already exists for a raw shard, "
            "treat it as done and remove the raw shard too."
        ),
    )
    parser.add_argument(
        "--require_cuda",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fail fast if CUDA is unavailable, to avoid silently running on CPU.",
    )
    parser.add_argument("--vq_path", required=True)
    parser.add_argument("--vq_type", default="ibq")
    parser.add_argument("--vq_device", default="cuda:0")
    parser.add_argument("--text_source", default="assistant", choices=["assistant", "human", "both", "caption"])
    parser.add_argument("--image_area", type=int, default=1048576)
    parser.add_argument("--grid_height", type=int, default=32)
    parser.add_argument("--grid_width", type=int, default=32)
    parser.add_argument("--max_height", type=int, default=0)
    parser.add_argument("--max_width", type=int, default=0)
    parser.add_argument(
        "--log_path",
        default="",
        help="Optional jsonl status log. Defaults to <output_root>/<split_name>.rolling_pretok.jsonl",
    )
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def ensure_runtime(args: argparse.Namespace) -> None:
    if args.require_cuda and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable. Refusing to pretokenize on CPU because it will be extremely slow. "
            "Re-run with --no-require_cuda only if CPU fallback is intentional."
        )


def normalize_device(device: str) -> str:
    if device == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    return device


def pretok_suffix(args: argparse.Namespace) -> str:
    if args.grid_height > 0 and args.grid_width > 0 and args.grid_height == args.grid_width:
        return f"{PRETOK_TOKEN}{int(args.grid_height)}"
    if args.grid_height > 0 and args.grid_width > 0:
        return f"{PRETOK_TOKEN}{int(args.grid_height)}x{int(args.grid_width)}"
    return PRETOK_TOKEN.rstrip("_")


def default_output_root_for_split(input_split_dir: str, args: argparse.Namespace) -> str:
    parent = osp.dirname(osp.abspath(input_split_dir))
    split_name = osp.basename(osp.normpath(input_split_dir))
    return osp.join(parent, f"{split_name}{pretok_suffix(args)}")


def discover_input_split_dirs(input_root: str, dataset_pattern: str) -> list[str]:
    import fnmatch

    result = []
    for name in sorted(os.listdir(input_root)):
        full = osp.join(input_root, name)
        if not osp.isdir(full):
            continue
        if name.startswith("."):
            continue
        if PRETOK_TOKEN in name:
            continue
        if not fnmatch.fnmatch(name, dataset_pattern):
            continue
        try:
            has_tar = any(entry.name.endswith(".tar") and entry.is_file() for entry in os.scandir(full))
        except FileNotFoundError:
            continue
        if not has_tar:
            continue
        result.append(full)
    return result


def list_input_shards(input_split_dir: str, pattern: str, start_after: str, limit: int) -> list[str]:
    names = sorted(
        [
            osp.join(input_split_dir, name)
            for name in os.listdir(input_split_dir)
            if name.endswith(".tar") and _matches_pattern(name, pattern)
        ]
    )
    if start_after:
        start_after = osp.basename(start_after)
        names = [p for p in names if osp.basename(p) > start_after]
    if limit > 0:
        names = names[:limit]
    return names


def _matches_pattern(name: str, pattern: str) -> bool:
    import fnmatch

    return fnmatch.fnmatch(name, pattern)


def validate_pretok_tar(path: str) -> bool:
    if not osp.exists(path) or osp.getsize(path) <= 0:
        return False
    stem_counts: Dict[str, Dict[str, int]] = {}
    try:
        with tarfile.open(path, "r:*") as tf:
            for member in tf:
                if not member.isfile():
                    continue
                base = osp.basename(member.name)
                stem, ext = osp.splitext(base)
                ext = ext.lower()
                if ext not in {".pt", ".txt"}:
                    continue
                counts = stem_counts.setdefault(stem, {".pt": 0, ".txt": 0})
                counts[ext] += 1
    except Exception:
        return False
    if not stem_counts:
        return False
    return all(counts[".pt"] == 1 and counts[".txt"] == 1 for counts in stem_counts.values())


def process_one_shard(
    input_path: str,
    output_path: str,
    *,
    vq_model,
    args: argparse.Namespace,
) -> ShardProcessResult:
    tmp_dir = tempfile.mkdtemp(prefix="rolling_pretok_", dir=osp.dirname(output_path))
    tmp_output = osp.join(tmp_dir, osp.basename(output_path))
    bucket: Dict[str, Dict[str, bytes]] = {}
    written = 0
    skipped_invalid = 0
    try:
        with tarfile.open(input_path, "r:*") as tf, tarfile.open(tmp_output, "w") as out:
            for member in tf:
                if not member.isfile():
                    continue
                name = osp.basename(member.name)
                stem, ext = osp.splitext(name)
                ext = ext.lower()
                fobj = tf.extractfile(member)
                if fobj is None:
                    continue
                if ext in IMAGE_EXTS:
                    bucket.setdefault(stem, {})["image"] = fobj.read()
                elif ext in TEXT_EXTS:
                    bucket.setdefault(stem, {})["text"] = fobj.read()
                elif ext == ".json":
                    try:
                        meta = json.loads(fobj.read().decode("utf-8"))
                    except Exception:
                        continue
                    text = extract_text_from_entry(meta, args.text_source)
                    if text:
                        bucket.setdefault(stem, {})["text"] = text.encode("utf-8")
                else:
                    continue

                entry = bucket.get(stem)
                if entry is None or "image" not in entry or "text" not in entry:
                    continue
                try:
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
                except Exception:
                    skipped_invalid += 1
                    del bucket[stem]
                    continue

                buf = io.BytesIO()
                torch.save(tokens, buf)
                _write_member(out, f"{stem}.pt", buf.getvalue())
                _write_member(out, f"{stem}.txt", entry["text"])
                written += 1
                del bucket[stem]

        if written <= 0:
            raise RuntimeError(f"No samples were written from {input_path}")
        if not validate_pretok_tar(tmp_output):
            raise RuntimeError(f"Output verification failed for {tmp_output}")

        os.makedirs(osp.dirname(output_path), exist_ok=True)
        os.replace(tmp_output, output_path)
        return ShardProcessResult(samples_written=written, skipped_invalid=skipped_invalid)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def cleanup_raw_shard(input_path: str, *, input_split_dir: str, args: argparse.Namespace) -> str:
    if args.delete_mode == "keep":
        return "kept"
    if args.delete_mode == "trash":
        trash_dir = args.trash_dir or osp.join(input_split_dir, ".trash_done")
        os.makedirs(trash_dir, exist_ok=True)
        target = osp.join(trash_dir, osp.basename(input_path))
        if osp.exists(target):
            os.remove(target)
        shutil.move(input_path, target)
        return target
    os.remove(input_path)
    return "deleted"


def append_log(log_path: str, payload: Dict[str, object]) -> None:
    os.makedirs(osp.dirname(log_path), exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def run_one_dataset(
    *,
    input_split_dir: str,
    output_root: str,
    split_name: str,
    vq_model,
    args: argparse.Namespace,
) -> DatasetRunResult:
    output_split_dir = osp.join(output_root, split_name)
    log_path = args.log_path or osp.join(output_root, f"{split_name}.rolling_pretok.jsonl")

    if not osp.isdir(input_split_dir):
        raise NotADirectoryError(input_split_dir)
    os.makedirs(output_split_dir, exist_ok=True)

    shards = list_input_shards(input_split_dir, args.pattern, args.start_after, args.limit)
    if not shards:
        raise FileNotFoundError(f"No shards matched {args.pattern} under {input_split_dir}")

    pbar = tqdm(shards, desc="rolling pretokenize", unit="shard")
    processed = 0
    reclaimed_bytes = 0
    for input_path in pbar:
        shard_name = osp.basename(input_path)
        output_path = osp.join(output_split_dir, shard_name)
        input_size = osp.getsize(input_path)

        if validate_pretok_tar(output_path):
            cleanup_result = "skipped_existing"
            if args.cleanup_existing_done:
                if not args.dry_run:
                    cleanup_result = cleanup_raw_shard(
                        input_path,
                        input_split_dir=input_split_dir,
                        args=args,
                    )
                reclaimed_bytes += input_size
            append_log(
                log_path,
                {
                    "status": "existing_output",
                    "input": input_path,
                    "output": output_path,
                    "cleanup": cleanup_result,
                    "input_size": input_size,
                },
            )
            processed += 1
            pbar.set_postfix(done=processed, reclaimed_gb=f"{reclaimed_bytes / 1024**3:.1f}")
            continue

        if args.dry_run:
            append_log(
                log_path,
                {
                    "status": "dry_run",
                    "input": input_path,
                    "output": output_path,
                    "input_size": input_size,
                },
            )
            processed += 1
            pbar.set_postfix(done=processed, reclaimed_gb=f"{reclaimed_bytes / 1024**3:.1f}")
            continue

        result = process_one_shard(input_path, output_path, vq_model=vq_model, args=args)
        cleanup_result = cleanup_raw_shard(
            input_path,
            input_split_dir=input_split_dir,
            args=args,
        )
        reclaimed_bytes += input_size
        append_log(
            log_path,
            {
                "status": "processed",
                "input": input_path,
                "output": output_path,
                "cleanup": cleanup_result,
                "samples_written": result.samples_written,
                "samples_skipped_invalid": result.skipped_invalid,
                "input_size": input_size,
            },
        )
        processed += 1
        pbar.set_postfix(done=processed, reclaimed_gb=f"{reclaimed_bytes / 1024**3:.1f}")

    print(
        "[INFO] rolling pretokenize finished:",
        f"processed={processed}",
        f"reclaimed_gb={reclaimed_bytes / 1024**3:.2f}",
        f"output_split_dir={output_split_dir}",
        f"log_path={log_path}",
    )
    return DatasetRunResult(processed_shards=processed, reclaimed_bytes=reclaimed_bytes)


def main() -> None:
    args = parse_args()
    ensure_runtime(args)

    vq_device = normalize_device(args.vq_device)
    vq_model = build_vision_tokenizer(args.vq_type, args.vq_path, device=vq_device)
    vq_model.eval()

    if args.input_split_dir:
        input_split_dir = osp.abspath(args.input_split_dir)
        split_name = args.split_name or osp.basename(osp.normpath(input_split_dir))
        output_root = (
            osp.abspath(args.output_root)
            if args.output_root
            else default_output_root_for_split(input_split_dir, args)
        )
        run_one_dataset(
            input_split_dir=input_split_dir,
            output_root=output_root,
            split_name=split_name,
            vq_model=vq_model,
            args=args,
        )
        return

    input_root = osp.abspath(args.input_root)
    if not osp.isdir(input_root):
        raise NotADirectoryError(input_root)
    output_base_root = osp.abspath(args.output_base_root) if args.output_base_root else input_root
    input_split_dirs = discover_input_split_dirs(input_root, args.dataset_pattern)
    if not input_split_dirs:
        raise FileNotFoundError(f"No raw dataset dirs found under {input_root}")

    total_processed = 0
    total_reclaimed = 0
    for input_split_dir in input_split_dirs:
        split_name = osp.basename(osp.normpath(input_split_dir))
        output_root = osp.join(output_base_root, f"{split_name}{pretok_suffix(args)}")
        result = run_one_dataset(
            input_split_dir=input_split_dir,
            output_root=output_root,
            split_name=split_name,
            vq_model=vq_model,
            args=args,
        )
        total_processed += result.processed_shards
        total_reclaimed += result.reclaimed_bytes

    print(
        "[INFO] all datasets finished:",
        f"datasets={len(input_split_dirs)}",
        f"processed_shards={total_processed}",
        f"reclaimed_gb={total_reclaimed / 1024**3:.2f}",
    )


if __name__ == "__main__":
    main()
