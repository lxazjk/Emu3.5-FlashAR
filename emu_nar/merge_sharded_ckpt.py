# -*- coding: utf-8 -*-
# Merge FSDP sharded checkpoints into a single full state dict.

from __future__ import annotations

import argparse
import os

import torch
import torch.distributed as dist

from src.emu3p5 import Emu3Config, Emu3ForCausalLM
from emu_nar.modeling_emu_nar import EmuNAR
from emu_nar.lora import apply_lora_to_backbone


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", type=str, required=True)
    p.add_argument("--ckpt_base", type=str, required=True)
    p.add_argument("--output_path", type=str, required=True)
    p.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    p.add_argument("--fsdp_wrap_policy", type=str, default="transformer", choices=["size", "transformer"])
    p.add_argument("--fsdp_min_params", type=int, default=1_000_000)
    p.add_argument("--use_vertical_block", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--vertical_layers", type=int, default=1)
    p.add_argument("--lora_layers", type=int, default=0)
    p.add_argument("--lora_r", type=int, default=0)
    p.add_argument("--lora_alpha", type=float, default=0.0)
    p.add_argument("--lora_dropout", type=float, default=0.0)
    return p.parse_args()


def _safe_torch_load(path: str):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _detect_state_dict_type(state_dict: dict) -> str:
    for val in state_dict.values():
        name = type(val).__name__
        if name in ("ShardedTensor", "DTensor"):
            return "sharded"
        if hasattr(val, "_local_shards") or hasattr(val, "_metadata"):
            return "sharded"
    for key in state_dict.keys():
        if "flat_param" in key or "_flat_param" in key:
            return "local"
    return "full"


def main() -> None:
    args = parse_args()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if not torch.cuda.is_available():
        raise RuntimeError("merge_sharded_ckpt requires CUDA; no accelerator is available.")
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    torch_dtype = dtype_map[args.dtype]

    shard_path = f"{args.ckpt_base}.rank{rank}.pt"
    if not os.path.exists(shard_path):
        raise FileNotFoundError(f"Missing shard for rank {rank}: {shard_path}")
    state_dict = _safe_torch_load(shard_path)
    state_type = _detect_state_dict_type(state_dict)
    if state_type == "full":
        if rank == 0:
            torch.save(state_dict, args.output_path)
            print(f"[INFO] saved full ckpt (direct): {args.output_path}")
        dist.barrier()
        return

    config = Emu3Config.from_pretrained(args.model_path, trust_remote_code=True)
    backbone = Emu3ForCausalLM(config=config)
    backbone = backbone.to(device=device, dtype=torch_dtype)
    if args.lora_r > 0 and args.lora_layers > 0:
        lora_alpha = args.lora_alpha if args.lora_alpha > 0 else float(args.lora_r)
        apply_lora_to_backbone(
            backbone,
            num_layers=args.lora_layers,
            r=args.lora_r,
            alpha=lora_alpha,
            dropout=args.lora_dropout,
        )

    visual_token_offset = int(config.eoi_token_id) + 1
    wrapper = EmuNAR(
        pretrained_backbone=backbone.model,
        vocab_size=config.vocab_size,
        hidden_size=config.hidden_size,
        num_heads=config.num_attention_heads,
        pad_token_id=-100,
        mask_token_id=config.pad_token_id,
        visual_token_offset=visual_token_offset,
        use_vertical_block=args.use_vertical_block,
        vertical_layers=args.vertical_layers,
        lm_head=backbone.lm_head,
    ).to(device=device, dtype=torch_dtype)

    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import FullStateDictConfig, ShardedStateDictConfig, StateDictType
    from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy, transformer_auto_wrap_policy
    from src.emu3p5.modeling_emu3 import Emu3DecoderLayer
    from functools import partial

    if args.fsdp_wrap_policy == "transformer":
        auto_wrap_policy = partial(
            transformer_auto_wrap_policy, transformer_layer_cls={Emu3DecoderLayer}
        )
    else:
        auto_wrap_policy = partial(size_based_auto_wrap_policy, min_num_params=args.fsdp_min_params)

    wrapper = FSDP(
        wrapper,
        auto_wrap_policy=auto_wrap_policy,
        cpu_offload=None,
        use_orig_params=False,
        device_id=device,
    )

    load_error = None
    if state_type == "sharded":
        try:
            shard_cfg = ShardedStateDictConfig(offload_to_cpu=True)
            with FSDP.state_dict_type(wrapper, StateDictType.SHARDED_STATE_DICT, shard_cfg):
                wrapper.load_state_dict(state_dict, strict=True)
        except Exception as exc:
            load_error = exc
    elif state_type == "local":
        from torch.distributed.fsdp import LocalStateDictConfig

        try:
            local_cfg = LocalStateDictConfig(offload_to_cpu=True)
            with FSDP.state_dict_type(wrapper, StateDictType.LOCAL_STATE_DICT, local_cfg):
                wrapper.load_state_dict(state_dict, strict=True)
        except Exception as exc:
            load_error = exc

    if load_error is not None:
        raise load_error

    full_cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(wrapper, StateDictType.FULL_STATE_DICT, full_cfg):
        full_state = wrapper.state_dict()
    if rank == 0:
        torch.save(full_state, args.output_path)
        print(f"[INFO] saved full ckpt: {args.output_path}")
    dist.barrier()


if __name__ == "__main__":
    main()
