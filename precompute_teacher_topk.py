# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import io
import os
import os.path as osp
import tarfile
from typing import Dict, Iterable, List, Tuple

import torch
from tqdm import tqdm
from transformers import AutoTokenizer

from src.emu3p5 import Emu3Config, Emu3ForCausalLM


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


def _build_image_prefix_tokens(tokenizer, height: int, width: int) -> List[int]:
    boi_id = tokenizer.encode(tokenizer.boi_token, add_special_tokens=False)[0]
    img_id = tokenizer.encode(tokenizer.img_token, add_special_tokens=False)[0]
    hw_ids = tokenizer.encode(f"{height}*{width}", add_special_tokens=False)
    return [boi_id, *hw_ids, img_id]


def _encode_text_ids(
    tokenizer,
    text_template: str,
    text: str,
    text_max_length: int,
    add_boi: bool,
    height: int,
    width: int,
) -> torch.Tensor:
    prompt = text_template.replace("{text}", text)
    text_ids = tokenizer.encode(prompt, add_special_tokens=False)
    if text_max_length > 0:
        text_ids = text_ids[: text_max_length]
    if add_boi and hasattr(tokenizer, "boi_token"):
        text_ids = list(text_ids) + _build_image_prefix_tokens(tokenizer, height, width)
    return torch.tensor(text_ids, dtype=torch.long)


def _pad_text_ids(text_ids: List[torch.Tensor], pad_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
    max_len = max(ids.numel() for ids in text_ids)
    padded = []
    mask = []
    for ids in text_ids:
        pad_len = max_len - ids.numel()
        if pad_len > 0:
            pad = torch.full((pad_len,), pad_id, dtype=ids.dtype)
            ids = torch.cat([ids, pad], dim=0)
        padded.append(ids)
        attn = torch.zeros((max_len,), dtype=torch.long)
        attn[: ids.numel() - pad_len] = 1
        mask.append(attn)
    return torch.stack(padded, dim=0), torch.stack(mask, dim=0)


def _write_member(tar_out: tarfile.TarFile, name: str, data: bytes) -> None:
    info = tarfile.TarInfo(name=name)
    info.size = len(data)
    tar_out.addfile(info, io.BytesIO(data))


def _load_emu3(
    model_path: str, config: Emu3Config, torch_dtype: torch.dtype
) -> Emu3ForCausalLM:
    return Emu3ForCausalLM.from_pretrained(
        model_path,
        config=config,
        torch_dtype=torch_dtype,
        attn_implementation="eager",
    )


def _iter_pretok_samples(tar_path: str) -> Iterable[Tuple[str, bytes, bytes]]:
    with tarfile.open(tar_path, "r:*") as tf:
        bucket: Dict[str, Dict[str, bytes]] = {}
        for member in tf:
            if not member.isfile():
                continue
            name = osp.basename(member.name)
            stem, ext = osp.splitext(name)
            ext = ext.lower()
            if ext not in (".pt", ".txt"):
                continue
            fobj = tf.extractfile(member)
            if fobj is None:
                continue
            data = fobj.read()
            key = "pt" if ext == ".pt" else "text"
            bucket.setdefault(stem, {})[key] = data
            entry = bucket.get(stem)
            if entry is None:
                continue
            if "pt" in entry and "text" in entry:
                yield stem, entry["pt"], entry["text"]
                del bucket[stem]


def _save_topk(
    tar_out: tarfile.TarFile,
    stem: str,
    pt_bytes: bytes,
    txt_bytes: bytes,
    topk_indices: torch.Tensor,
    topk_logits: torch.Tensor,
    save_dtype: torch.dtype,
) -> None:
    _write_member(tar_out, f"{stem}.pt", pt_bytes)
    _write_member(tar_out, f"{stem}.txt", txt_bytes)
    payload = {
        "indices": topk_indices.to(dtype=torch.int32, device="cpu"),
        "logits": topk_logits.to(dtype=save_dtype, device="cpu"),
    }
    buf = io.BytesIO()
    torch.save(payload, buf)
    _write_member(tar_out, f"{stem}.logits.pt", buf.getvalue())


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--pretok_glob", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--model_path", type=str, required=True)
    p.add_argument("--tokenizer_path", type=str, required=True)
    p.add_argument(
        "--text_template",
        type=str,
        default="<|extra_203|>You are a helpful assistant for t2i task. USER: {text} ASSISTANT: <|extra_100|>",
    )
    p.add_argument("--text_max_length", type=int, default=128)
    p.add_argument("--add_boi", action="store_true")
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    p.add_argument("--save_dtype", type=str, default="fp16", choices=["bf16", "fp16", "fp32"])
    p.add_argument("--topk", type=int, default=256)
    p.add_argument("--visual_token_offset", type=int, default=-1)
    p.add_argument("--visual_vocab_size", type=int, default=131072)
    p.add_argument("--mask_visual_only", action="store_true")
    p.add_argument("--shard_rank", type=int, default=0)
    p.add_argument("--shard_world_size", type=int, default=1)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    import glob as _glob

    shard_paths = sorted(_glob.glob(args.pretok_glob))
    if not shard_paths:
        raise FileNotFoundError(f"No pretokenized tar shards for glob: {args.pretok_glob}")
    if args.shard_world_size > 1:
        shard_paths = shard_paths[args.shard_rank :: args.shard_world_size]

    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    torch_dtype = dtype_map[args.dtype]
    save_dtype = dtype_map[args.save_dtype]

    device = torch.device(args.device)
    tokenizer = _build_text_tokenizer(args.tokenizer_path)
    model_config = Emu3Config.from_pretrained(args.model_path, trust_remote_code=True)
    teacher = _load_emu3(args.model_path, model_config, torch_dtype).to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    visual_token_offset = args.visual_token_offset
    if visual_token_offset < 0:
        visual_token_offset = int(model_config.eoi_token_id) + 1
    max_visual_id = visual_token_offset + args.visual_vocab_size - 1
    if max_visual_id >= model_config.vocab_size:
        raise ValueError(
            f"visual_token_offset {visual_token_offset} + visual_vocab_size {args.visual_vocab_size} "
            f"exceeds vocab_size {model_config.vocab_size}."
        )

    for tar_path in shard_paths:
        out_path = osp.join(args.output_dir, osp.basename(tar_path))
        if osp.abspath(out_path) == osp.abspath(tar_path):
            raise ValueError("output_dir must be different from input shard directory.")
        if osp.exists(out_path) and not args.overwrite:
            print(f"[SKIP] {out_path} exists")
            continue

        pending: List[Dict[str, object]] = []

        with tarfile.open(out_path, "w") as tar_out:
            pbar = tqdm(
                _iter_pretok_samples(tar_path),
                desc=f"topk {osp.basename(tar_path)}",
                unit="sample",
            )
            for stem, pt_bytes, txt_bytes in pbar:
                tokens = torch.load(io.BytesIO(pt_bytes), map_location="cpu")
                text = txt_bytes.decode("utf-8", errors="ignore")
                pending.append(
                    {
                        "stem": stem,
                        "pt_bytes": pt_bytes,
                        "txt_bytes": txt_bytes,
                        "tokens": tokens,
                        "text": text,
                    }
                )
                if len(pending) >= args.batch_size:
                    _process_batch(
                        pending,
                        tokenizer,
                        teacher,
                        device,
                        visual_token_offset,
                        args.text_template,
                        args.text_max_length,
                        args.add_boi,
                        args.topk,
                        args.mask_visual_only,
                        args.visual_vocab_size,
                        save_dtype,
                        tar_out,
                    )
                    pbar.set_postfix(batch=len(pending))
                    pending = []

            if pending:
                _process_batch(
                    pending,
                    tokenizer,
                    teacher,
                    device,
                    visual_token_offset,
                    args.text_template,
                    args.text_max_length,
                    args.add_boi,
                    args.topk,
                    args.mask_visual_only,
                    args.visual_vocab_size,
                    save_dtype,
                    tar_out,
                )

        print(f"[OK] wrote {out_path}")


def _process_batch(
    batch: List[Dict[str, object]],
    tokenizer,
    teacher: Emu3ForCausalLM,
    device: torch.device,
    visual_token_offset: int,
    text_template: str,
    text_max_length: int,
    add_boi: bool,
    topk: int,
    mask_visual_only: bool,
    visual_vocab_size: int,
    save_dtype: torch.dtype,
    tar_out: tarfile.TarFile,
) -> None:
    tokens = torch.stack([b["tokens"] for b in batch], dim=0).long()
    height = int(tokens.size(1))
    width = int(tokens.size(2))
    if visual_token_offset:
        tokens = tokens + visual_token_offset
    input_ids = tokens.view(tokens.size(0), -1).to(device)

    text_ids = [
        _encode_text_ids(
            tokenizer=tokenizer,
            text_template=text_template,
            text=b["text"],
            text_max_length=text_max_length,
            add_boi=add_boi,
            height=height,
            width=width,
        )
        for b in batch
    ]
    text_ids, text_attention_mask = _pad_text_ids(text_ids, tokenizer.pad_token_id)
    text_ids = text_ids.to(device)
    text_attention_mask = text_attention_mask.to(device)

    full_input_ids = torch.cat([text_ids, input_ids], dim=1)
    full_attention = torch.ones_like(full_input_ids, dtype=torch.long)
    full_attention[:, : text_attention_mask.size(1)] = text_attention_mask

    with torch.no_grad():
        teacher_logits = teacher(
            input_ids=full_input_ids,
            attention_mask=full_attention,
            use_cache=False,
            return_dict=True,
        ).logits

    text_len = text_ids.size(1)
    token_len = input_ids.size(1)
    if text_len > 0:
        teacher_logits = teacher_logits[:, text_len - 1 : text_len - 1 + token_len, :]
    else:
        teacher_logits = teacher_logits[:, :token_len, :]

    if mask_visual_only:
        teacher_logits = teacher_logits.clone()
        if visual_token_offset > 0:
            teacher_logits[:, :, :visual_token_offset] = float("-inf")
        tail = visual_token_offset + visual_vocab_size
        if tail < teacher_logits.size(-1):
            teacher_logits[:, :, tail:] = float("-inf")

    topk_logits, topk_idx = torch.topk(teacher_logits, k=topk, dim=-1)

    for i, b in enumerate(batch):
        _save_topk(
            tar_out=tar_out,
            stem=str(b["stem"]),
            pt_bytes=b["pt_bytes"],
            txt_bytes=b["txt_bytes"],
            topk_indices=topk_idx[i],
            topk_logits=topk_logits[i],
            save_dtype=save_dtype,
        )


if __name__ == "__main__":
    main()
