# -*- coding: utf-8 -*-
# NAR fine-tuning entry for Infinity-MM webdataset shards (json + image).

from __future__ import annotations

import argparse
import glob
import os
import os.path as osp
import random
from contextlib import nullcontext
from functools import partial
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from src.emu3p5 import Emu3Config, Emu3ForCausalLM
from emu_nar.inference.neighbor_ar_wrapper import NeighborARWrapper
from src.vision_tokenizer import build_vision_tokenizer

from emu_nar.data.infinity_mm import (
    InfinityMMShardDataset,
    collate_fn,
    encode_images,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--tokenizer_path", type=str, required=True)
    parser.add_argument("--vq_path", type=str, required=True)
    parser.add_argument("--vq_type", type=str, default="ibq")
    parser.add_argument("--dataset_glob", type=str, required=True, help="Glob for Infinity-MM tar shards.")
    parser.add_argument(
        "--text_source",
        type=str,
        default="assistant",
        choices=["assistant", "human", "both", "caption"],
    )
    parser.add_argument("--text_template", type=str, default="{text}")
    parser.add_argument("--text_max_length", type=int, default=0)
    parser.add_argument("--add_boi", action="store_true")
    parser.add_argument("--image_area", type=int, default=0)
    parser.add_argument("--image_size", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=0.0)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--vq_device", type=str, default="auto")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--train_backbone", action="store_true")
    parser.add_argument("--fsdp", action="store_true")
    parser.add_argument("--fsdp_min_params", type=int, default=1_000_000)
    parser.add_argument("--fsdp_wrap_policy", type=str, default="transformer", choices=["size", "transformer"])
    parser.add_argument("--fsdp_cpu_offload", action="store_true")
    parser.add_argument("--save_dir", type=str, default="./outputs/nar_finetune")
    parser.add_argument("--save_every_steps", type=int, default=0)
    parser.add_argument("--resume_path", type=str, default="")
    parser.add_argument("--visual_token_offset", type=int, default=-1)
    parser.add_argument("--visual_vocab_size", type=int, default=131072)
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--persistent_workers", action="store_true")
    parser.add_argument("--pin_memory", action="store_true")
    parser.add_argument("--use_vertical_block", action="store_true")
    return parser.parse_args()


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


def _safe_torch_load(path: str) -> Dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _seed_worker(worker_id: int, base_seed: int, rank: int) -> None:
    seed = base_seed + worker_id + rank * 1000
    random.seed(seed)
    np.random.seed(seed)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    if args.fsdp:
        if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
            raise ValueError("FSDP requires torchrun/torch.distributed with RANK/WORLD_SIZE set.")
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        rank = 0
        world_size = 1
        device = torch.device(args.device)
        local_rank = 0

    is_main = rank == 0
    if args.fsdp or is_main:
        os.makedirs(args.save_dir, exist_ok=True)
    if is_main and args.train_backbone:
        print("[WARN] training full backbone; consider unfreezing after NAR heads converge.")

    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    torch_dtype = dtype_map[args.dtype]

    model_config = Emu3Config.from_pretrained(args.model_path, trust_remote_code=True)
    backbone = Emu3ForCausalLM.from_pretrained(
        args.model_path,
        config=model_config,
        torch_dtype=torch_dtype,
        attn_implementation="eager",
    )
    if not args.fsdp:
        backbone = backbone.to(device)

    visual_token_offset = args.visual_token_offset
    if visual_token_offset < 0:
        visual_token_offset = int(model_config.eoi_token_id) + 1
    max_visual_id = visual_token_offset + args.visual_vocab_size - 1
    if max_visual_id >= model_config.vocab_size:
        raise ValueError(
            f"visual_token_offset {visual_token_offset} + visual_vocab_size {args.visual_vocab_size} "
            f"exceeds vocab_size {model_config.vocab_size}."
        )

    wrapper = NeighborARWrapper(
        pretrained_backbone=backbone.model,
        vocab_size=model_config.vocab_size,
        hidden_size=model_config.hidden_size,
        num_heads=model_config.num_attention_heads,
        pad_token_id=-100,
        mask_token_id=model_config.pad_token_id,
        visual_token_offset=visual_token_offset,
        img_token_id=model_config.img_token_id,
        eol_token_id=model_config.eol_token_id,
        eoi_token_id=model_config.eoi_token_id,
        use_vertical_block=args.use_vertical_block,
    )
    if args.fsdp:
        wrapper = wrapper.to(dtype=torch_dtype)
    else:
        wrapper = wrapper.to(device=device, dtype=torch_dtype)

    if args.fsdp:
        from torch.distributed.fsdp import CPUOffload, FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy, transformer_auto_wrap_policy
        from src.emu3p5.modeling_emu3 import Emu3DecoderLayer

        if args.fsdp_wrap_policy == "transformer":
            auto_wrap_policy = partial(
                transformer_auto_wrap_policy, transformer_layer_cls={Emu3DecoderLayer}
            )
        else:
            auto_wrap_policy = partial(size_based_auto_wrap_policy, min_num_params=args.fsdp_min_params)
        cpu_offload = CPUOffload(offload_params=True) if args.fsdp_cpu_offload else None
        wrapper = FSDP(
            wrapper,
            auto_wrap_policy=auto_wrap_policy,
            cpu_offload=cpu_offload,
            device_id=device,
        )

    if args.resume_path:
        if args.fsdp:
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
            from torch.distributed.fsdp import ShardedStateDictConfig, StateDictType

            resume_path = args.resume_path
            rank_path = resume_path
            if not resume_path.endswith(f".rank{rank}.pt"):
                candidate = resume_path + f".rank{rank}.pt"
                if os.path.exists(candidate):
                    rank_path = candidate
            if not os.path.exists(rank_path):
                raise FileNotFoundError(f"Missing sharded checkpoint for rank {rank}: {rank_path}")
            state_cfg = ShardedStateDictConfig(offload_to_cpu=True)
            with FSDP.state_dict_type(wrapper, StateDictType.SHARDED_STATE_DICT, state_cfg):
                state_dict = _safe_torch_load(rank_path)
                wrapper.load_state_dict(state_dict, strict=True)
            if rank == 0:
                print("[INFO] loaded sharded ckpt:", resume_path)
            dist.barrier()
        else:
            state_dict = _safe_torch_load(args.resume_path)
            wrapper.load_state_dict(state_dict, strict=True)
            if rank == 0:
                print("[INFO] loaded ckpt:", args.resume_path)

    if not args.train_backbone:
        target = wrapper.module if args.fsdp else wrapper
        for p in target.backbone.parameters():
            p.requires_grad = False

    optimizer = torch.optim.AdamW(wrapper.parameters(), lr=args.lr)

    tokenizer = _build_text_tokenizer(args.tokenizer_path)

    vq_device = device if args.vq_device == "auto" else torch.device(args.vq_device)
    vq_model = build_vision_tokenizer(args.vq_type, args.vq_path, device=str(vq_device))
    vq_model.eval()
    for p in vq_model.parameters():
        p.requires_grad = False
    vq_dtype = next(vq_model.parameters()).dtype

    shard_paths = sorted(glob.glob(args.dataset_glob))
    if not shard_paths:
        raise FileNotFoundError(f"No tar shards found for glob: {args.dataset_glob}")

    ds = InfinityMMShardDataset(
        shard_paths=shard_paths,
        text_source=args.text_source,
        rank=rank,
        world_size=world_size,
    )

    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        worker_init_fn=partial(_seed_worker, base_seed=args.seed, rank=rank),
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers and args.num_workers > 0,
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
    )

    wrapper.train()
    global_step = 0
    accum_steps = max(1, args.grad_accum_steps)
    image_area = args.image_area or int(getattr(model_config, "image_area", 512 * 512))
    for epoch in range(args.epochs):
        if is_main:
            progress = tqdm(loader, desc=f"epoch {epoch}")
        else:
            progress = loader
        last_step = None
        for step, batch in enumerate(progress):
            last_step = step
            images = batch["images"]
            texts = batch["texts"]

            tokens, height, width = encode_images(
                vq_model=vq_model,
                images=images,
                image_area=image_area,
                image_size=args.image_size,
                device=vq_device,
                dtype=vq_dtype,
                visual_token_offset=visual_token_offset,
            )
            input_ids = tokens.view(tokens.size(0), -1).to(device)

            text_ids = [
                _encode_text_ids(
                    tokenizer=tokenizer,
                    text_template=args.text_template,
                    text=text,
                    text_max_length=args.text_max_length,
                    add_boi=args.add_boi,
                    height=height,
                    width=width,
                )
                for text in texts
            ]
            text_ids, text_attention_mask = _pad_text_ids(text_ids, tokenizer.pad_token_id)
            text_ids = text_ids.to(device)
            text_attention_mask = text_attention_mask.to(device)

            is_sync_step = ((step + 1) % accum_steps == 0)
            if args.fsdp and not is_sync_step and accum_steps > 1:
                sync_ctx = wrapper.no_sync()
            else:
                sync_ctx = nullcontext()

            with sync_ctx:
                outputs = wrapper(
                    input_ids=input_ids,
                    height=height,
                    width=width,
                    text_input_ids=text_ids,
                    text_attention_mask=text_attention_mask,
                )
                raw_loss = outputs["loss"]
                loss = raw_loss / accum_steps
                loss.backward()

            if is_sync_step:
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(wrapper.parameters(), args.grad_clip)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            if is_main and step % 10 == 0:
                progress.set_postfix(loss=f"{raw_loss.item():.4f}")

            global_step += 1
            if args.save_every_steps > 0 and global_step % args.save_every_steps == 0:
                step_path = os.path.join(args.save_dir, f"nar_step{global_step}.pt")
                if args.fsdp:
                    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
                    from torch.distributed.fsdp import ShardedStateDictConfig, StateDictType

                    state_cfg = ShardedStateDictConfig(offload_to_cpu=True)
                    with FSDP.state_dict_type(wrapper, StateDictType.SHARDED_STATE_DICT, state_cfg):
                        state_dict = wrapper.state_dict()
                    torch.save(state_dict, step_path + f".rank{rank}.pt")
                    if is_main:
                        print("[INFO] saved step sharded:", step_path)
                    dist.barrier()
                else:
                    if is_main:
                        torch.save(wrapper.state_dict(), step_path)
                        print("[INFO] saved step:", step_path)

        if accum_steps > 1 and last_step is not None and (last_step + 1) % accum_steps != 0:
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(wrapper.parameters(), args.grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        save_path = os.path.join(args.save_dir, f"nar_epoch{epoch}.pt")
        if args.fsdp:
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
            from torch.distributed.fsdp import ShardedStateDictConfig, StateDictType

            state_cfg = ShardedStateDictConfig(offload_to_cpu=True)
            with FSDP.state_dict_type(wrapper, StateDictType.SHARDED_STATE_DICT, state_cfg):
                state_dict = wrapper.state_dict()
            torch.save(state_dict, save_path + f".rank{rank}.pt")
            if is_main:
                print("[INFO] saved sharded:", save_path)
            dist.barrier()
        else:
            if is_main:
                torch.save(wrapper.state_dict(), save_path)
                print("[INFO] saved:", save_path)


if __name__ == "__main__":
    main()
