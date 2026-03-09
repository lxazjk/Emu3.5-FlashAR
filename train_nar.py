# -*- coding: utf-8 -*-

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
from emu_nar.modeling_emu_nar import EmuNAR, _sample_logits
from src.vision_tokenizer import build_vision_tokenizer

from emu_nar.data.pretokenized import PretokShardDataset, collate_pretok


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--tokenizer_path", type=str, required=True)
    parser.add_argument("--vq_path", type=str, required=True)
    parser.add_argument("--vq_type", type=str, default="ibq")
    parser.add_argument("--pretok_glob", type=str, default="", help="Glob for pretokenized .tar shards.")
    parser.add_argument("--text_template", type=str, default="{text}")
    parser.add_argument("--text_max_length", type=int, default=0)
    parser.add_argument("--add_boi", action="store_true")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--shuffle", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-6)
    parser.add_argument("--grad_clip", type=float, default=0.0)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--low_cpu_mem_usage", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--train_backbone", action="store_true")
    parser.add_argument("--fsdp_use_orig_params", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fsdp_min_params", type=int, default=1_000_000)
    parser.add_argument("--fsdp_wrap_policy", type=str, default="transformer", choices=["size", "transformer"])
    parser.add_argument("--fsdp_cpu_offload", action="store_true")
    parser.add_argument("--fsdp_no_sync", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--save_dir", type=str, default="./outputs/nar_finetune")
    parser.add_argument("--save_every_steps", type=int, default=0)
    parser.add_argument("--max_steps", type=int, default=0)
    parser.add_argument("--log_every_steps", type=int, default=10)
    parser.add_argument("--resume_path", type=str, default="")
    parser.add_argument(
        "--save_full_state",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When using FSDP, save a full (merged) state dict at the end of training.",
    )
    parser.add_argument(
        "--save_epoch",
        type=str,
        default="auto",
        choices=["auto", "none", "sharded", "full"],
        help="Per-epoch checkpoint mode.",
    )
    parser.add_argument("--eval_generate_prompt", type=str, default="")
    parser.add_argument("--eval_generate_height", type=int, default=0)
    parser.add_argument("--eval_generate_width", type=int, default=0)
    parser.add_argument("--eval_generate_outdir", type=str, default="")
    parser.add_argument("--eval_generate_decode", action="store_true")
    parser.add_argument("--eval_generate_temperature", type=float, default=1.0)
    parser.add_argument("--eval_generate_top_k", type=int, default=0)
    parser.add_argument("--eval_generate_top_p", type=float, default=1.0)
    parser.add_argument("--eval_generate_cfg_scale", type=float, default=1.0)
    parser.add_argument("--eval_generate_sample", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--eval_generate_device", type=str, default="cuda:0")
    parser.add_argument("--eval_generate_every_steps", type=int, default=0)
    parser.add_argument(
        "--eval_generate_timing",
        type=str,
        default="end",
        choices=["start", "end", "both", "none"],
    )
    parser.add_argument("--visual_token_offset", type=int, default=-1)
    parser.add_argument("--visual_vocab_size", type=int, default=131072)
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--persistent_workers", action="store_true")
    parser.add_argument("--pin_memory", action="store_true")
    parser.add_argument("--use_vertical_block", action="store_true")
    parser.add_argument("--vertical_layers", type=int, default=0)
    parser.add_argument(
        "--learnable_fuse",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use learnable h/v fusion weights instead of fixed 0.5 average.",
    )
    parser.add_argument(
        "--fuse_h_init",
        type=float,
        default=0.5,
        help="Initial horizontal fusion weight for interior positions (vertical=1-h).",
    )
    parser.add_argument(
        "--fuse_corner_h_init",
        type=float,
        default=-1.0,
        help="Initial horizontal fusion weight for corner; <0 means use fuse_h_init.",
    )
    parser.add_argument(
        "--vertical_head_warmup_steps",
        type=int,
        default=0,
        help="Train only vertical_head for the first N optimizer steps, then restore normal trainable params.",
    )
    parser.add_argument(
        "--ar_distill_weight",
        type=float,
        default=0.0,
        help="Extra AR distillation weight applied only during vertical_head warmup.",
    )
    parser.add_argument(
        "--ar_distill_temperature",
        type=float,
        default=2.0,
        help="Temperature for AR distillation.",
    )
    parser.add_argument(
        "--phase2_lr_factor",
        type=float,
        default=0.1,
        help=(
            "When warmup ends and normal training resumes, multiply optimizer LR by this factor. "
            "Set <=0 to disable."
        ),
    )
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--wandb_project", type=str, default="Emu3.5-NAR")
    parser.add_argument("--wandb_entity", type=str, default="")
    parser.add_argument("--wandb_name", type=str, default="")
    parser.add_argument(
        "--wandb_mode",
        type=str,
        default="online",
        choices=["online", "offline", "disabled"],
    )
    parser.add_argument("--wandb_log_interval", type=int, default=10)
    parser.add_argument("--aux_loss_h_weight", type=float, default=0.0)
    parser.add_argument("--aux_loss_v_weight", type=float, default=0.0)
    parser.add_argument("--eval_generate_seed", type=int, default=42,
                        help="Fixed RNG seed for eval generation (for fair comparison across steps).")
    parser.add_argument("--lr_scheduler", type=str, default="cosine", choices=["none", "cosine"],
                        help="LR scheduler for phase-2 training.")
    parser.add_argument("--lr_min_factor", type=float, default=0.05,
                        help="Cosine decay: min_lr = phase2_lr * lr_min_factor.")
    parser.add_argument("--phase1_flat_steps", type=int, default=200,
                        help="Phase-1: keep LR flat for this many steps before cosine decay starts.")
    parser.add_argument("--phase2_flat_steps", type=int, default=500,
                        help="Phase-2: keep LR flat for this many steps before cosine decay starts.")
    args = parser.parse_args()
    args.fsdp = True
    return args


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


def _detect_fsdp_state_type(path: str, state_dict: Dict[str, Any]) -> str:
    if path.endswith(".full.pt"):
        return "full"
    for key in state_dict.keys():
        if "flat_param" in key or "_flat_param" in key:
            return "local"
    for val in state_dict.values():
        name = type(val).__name__
        if name in ("ShardedTensor", "DTensor"):
            return "sharded"
        if hasattr(val, "_local_shards") or hasattr(val, "_metadata"):
            return "sharded"
    return "full"


def _seed_worker(worker_id: int, base_seed: int, rank: int) -> None:
    seed = base_seed + worker_id + rank * 1000
    random.seed(seed)
    np.random.seed(seed)


def _setup_distributed(args: argparse.Namespace) -> Tuple[int, int, int, torch.device]:
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
        local_rank = 0
        device = torch.device(args.device)
    return rank, world_size, local_rank, device


def _prepare_output_dir(args: argparse.Namespace, is_main: bool) -> None:
    if args.fsdp or is_main:
        os.makedirs(args.save_dir, exist_ok=True)


def _build_backbone(
    args: argparse.Namespace,
    model_config: Emu3Config,
    torch_dtype: torch.dtype,
    device: torch.device,
    is_main: bool,
) -> Emu3ForCausalLM:
    pretrained_kwargs = dict(
        config=model_config,
        torch_dtype=torch_dtype,
        attn_implementation="eager",
    )
    if args.low_cpu_mem_usage:
        pretrained_kwargs["low_cpu_mem_usage"] = True
    try:
        backbone = Emu3ForCausalLM.from_pretrained(args.model_path, **pretrained_kwargs)
    except TypeError:
        pretrained_kwargs.pop("low_cpu_mem_usage", None)
        backbone = Emu3ForCausalLM.from_pretrained(args.model_path, **pretrained_kwargs)
    if args.gradient_checkpointing:
        backbone.gradient_checkpointing_enable()
        backbone.config.use_cache = False
    if not args.fsdp:
        backbone = backbone.to(device)
    return backbone


def _resolve_visual_offset(
    args: argparse.Namespace, model_config: Emu3Config
) -> int:
    visual_token_offset = args.visual_token_offset
    if visual_token_offset < 0:
        visual_token_offset = int(model_config.eoi_token_id) + 1
    max_visual_id = visual_token_offset + args.visual_vocab_size - 1
    if max_visual_id >= model_config.vocab_size:
        raise ValueError(
            f"visual_token_offset {visual_token_offset} + visual_vocab_size {args.visual_vocab_size} "
            f"exceeds vocab_size {model_config.vocab_size}."
        )
    return visual_token_offset


def _build_wrapper(
    args: argparse.Namespace,
    backbone: Emu3ForCausalLM,
    model_config: Emu3Config,
    torch_dtype: torch.dtype,
    device: torch.device,
) -> Tuple[EmuNAR, int]:
    visual_token_offset = _resolve_visual_offset(args, model_config)
    vertical_layers = args.vertical_layers
    if vertical_layers <= 0:
        vertical_layers = int(getattr(model_config, "nar_vertical_layers", 1))
    wrapper = EmuNAR(
        pretrained_backbone=backbone.model,
        vocab_size=model_config.vocab_size,
        hidden_size=model_config.hidden_size,
        num_heads=model_config.num_attention_heads,
        pad_token_id=-100,
        mask_token_id=model_config.pad_token_id,
        visual_token_offset=visual_token_offset,
        use_vertical_block=args.use_vertical_block,
        vertical_layers=vertical_layers,
        learnable_fuse=args.learnable_fuse,
        fuse_h_init=args.fuse_h_init,
        fuse_corner_h_init=args.fuse_corner_h_init,
        lm_head=backbone.lm_head,
    )
    if args.fsdp:
        wrapper = wrapper.to(dtype=torch_dtype)
    else:
        wrapper = wrapper.to(device=device, dtype=torch_dtype)
    if args.learnable_fuse and (not args.fsdp):
        # FSDP flatten requires uniform dtype; keep fp32 gates only in non-FSDP mode.
        if getattr(wrapper, "fuse_h_logit", None) is not None:
            wrapper.fuse_h_logit.data = wrapper.fuse_h_logit.data.float()
        if getattr(wrapper, "fuse_corner_h_logit", None) is not None:
            wrapper.fuse_corner_h_logit.data = wrapper.fuse_corner_h_logit.data.float()
    return wrapper, visual_token_offset


def _wrap_fsdp_if_needed(
    args: argparse.Namespace,
    wrapper: EmuNAR,
    device: torch.device,
) -> EmuNAR:
    if not args.fsdp:
        return wrapper
    from torch.distributed.fsdp import CPUOffload, FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy, transformer_auto_wrap_policy
    from src.emu3p5.modeling_emu3 import Emu3DecoderLayer

    if args.fsdp_wrap_policy == "transformer":
        auto_wrap_policy = partial(transformer_auto_wrap_policy, transformer_layer_cls={Emu3DecoderLayer})
    else:
        auto_wrap_policy = partial(size_based_auto_wrap_policy, min_num_params=args.fsdp_min_params)
    cpu_offload = CPUOffload(offload_params=True) if args.fsdp_cpu_offload else None
    fsdp_kwargs = dict(
        auto_wrap_policy=auto_wrap_policy,
        cpu_offload=cpu_offload,
        device_id=device,
        use_orig_params=args.fsdp_use_orig_params,
    )
    wrapper = FSDP(wrapper, **fsdp_kwargs)
    return wrapper


def _is_allowed_resume_missing(key: str) -> bool:
    return key.endswith("fuse_h_logit") or key.endswith("fuse_corner_h_logit")


def _load_state_with_fuse_compat(
    module: torch.nn.Module,
    state_dict: Dict[str, Any],
    strict: bool,
    load_desc: str,
) -> None:
    if not strict:
        module.load_state_dict(state_dict, strict=False)
        return
    try:
        module.load_state_dict(state_dict, strict=True)
        return
    except RuntimeError as exc:
        msg = str(exc)
        if "fuse_h_logit" not in msg and "fuse_corner_h_logit" not in msg:
            raise
    incompat = module.load_state_dict(state_dict, strict=False)
    missing_keys = list(getattr(incompat, "missing_keys", []))
    unexpected_keys = list(getattr(incompat, "unexpected_keys", []))
    bad_missing = [k for k in missing_keys if not _is_allowed_resume_missing(k)]
    if bad_missing or unexpected_keys:
        raise RuntimeError(
            f"{load_desc}: incompatible state_dict. "
            f"missing={bad_missing} unexpected={unexpected_keys}"
        )
    if (not dist.is_initialized()) or dist.get_rank() == 0:
        print(
            f"[WARN] {load_desc}: initialized missing fuse params from defaults: {missing_keys}"
        )


def _resume_if_needed(
    args: argparse.Namespace,
    wrapper: EmuNAR,
    rank: int,
    is_main: bool,
) -> None:
    if not args.resume_path:
        return
    if args.fsdp:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp import (
            FullStateDictConfig,
            LocalStateDictConfig,
            ShardedStateDictConfig,
            StateDictType,
        )

        resume_path = args.resume_path
        rank_path = resume_path
        if not resume_path.endswith(f".rank{rank}.pt"):
            candidate = resume_path + f".rank{rank}.pt"
            if os.path.exists(candidate):
                rank_path = candidate
        if not os.path.exists(rank_path):
            raise FileNotFoundError(f"Missing sharded checkpoint for rank {rank}: {rank_path}")
        state_dict = _safe_torch_load(rank_path)
        state_type = _detect_fsdp_state_type(rank_path, state_dict)
        if state_type == "sharded":
            state_cfg = ShardedStateDictConfig(offload_to_cpu=True)
            with FSDP.state_dict_type(wrapper, StateDictType.SHARDED_STATE_DICT, state_cfg):
                _load_state_with_fuse_compat(
                    wrapper, state_dict, strict=True, load_desc="load sharded ckpt"
                )
            if rank == 0:
                print("[INFO] loaded sharded ckpt:", resume_path)
        elif state_type == "local":
            state_cfg = LocalStateDictConfig(offload_to_cpu=True)
            with FSDP.state_dict_type(wrapper, StateDictType.LOCAL_STATE_DICT, state_cfg):
                _load_state_with_fuse_compat(
                    wrapper, state_dict, strict=True, load_desc="load local ckpt"
                )
            if rank == 0:
                print("[INFO] loaded local ckpt:", resume_path)
        else:
            state_cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=False)
            with FSDP.state_dict_type(wrapper, StateDictType.FULL_STATE_DICT, state_cfg):
                _load_state_with_fuse_compat(
                    wrapper, state_dict, strict=True, load_desc="load full ckpt"
                )
            if rank == 0:
                print("[INFO] loaded full ckpt:", resume_path)
        dist.barrier()
    else:
        state_dict = _safe_torch_load(args.resume_path)
        _load_state_with_fuse_compat(
            wrapper, state_dict, strict=True, load_desc="load ckpt"
        )
        if is_main:
            print("[INFO] loaded ckpt:", args.resume_path)


def _freeze_backbone_if_needed(args: argparse.Namespace, wrapper: EmuNAR) -> None:
    if args.train_backbone:
        return
    target = wrapper.module if args.fsdp else wrapper
    for p in target.backbone.parameters():
        p.requires_grad = False


def _snapshot_requires_grad(module: torch.nn.Module) -> Dict[str, bool]:
    return {name: p.requires_grad for name, p in module.named_parameters()}


def _restore_requires_grad(module: torch.nn.Module, state: Dict[str, bool]) -> None:
    for name, p in module.named_parameters():
        if name in state:
            p.requires_grad = state[name]


def _set_vertical_branch_trainable(module: torch.nn.Module) -> None:
    """
    Warmup stage trainable set:
      - vertical_block.*
      - vertical_norm.*
      - vertical_head.*
      - learnable fusion gates (if enabled)
    """
    for name, p in module.named_parameters():
        keep = (
            name.startswith("vertical_block.")
            or name.startswith("vertical_norm.")
            or name.startswith("vertical_head.")
            or name.startswith("horizontal_head.")
            or name in ("fuse_h_logit", "fuse_corner_h_logit")
        )
        p.requires_grad = keep


def _build_optimizer(
    args: argparse.Namespace, wrapper: EmuNAR
) -> torch.optim.Optimizer:
    trainable_params = [p for p in wrapper.parameters() if p.requires_grad and p.numel() > 0]
    if not trainable_params:
        raise ValueError("No trainable parameters found. Check freeze settings.")
    return torch.optim.AdamW(trainable_params, lr=args.lr, foreach=True)


def _build_vq(
    args: argparse.Namespace, device: torch.device
) -> Tuple[Any, Any, torch.device]:
    # train.sh path uses pretokenized image tokens only; keep VQ lazy-loaded in eval.
    return None, None, device


def _init_wandb(args: argparse.Namespace, is_main: bool):
    if not args.wandb or not is_main:
        return None
    try:
        import wandb
    except Exception as exc:
        print(f"[WARN] wandb disabled: {exc}")
        return None
    config = {
        k: v for k, v in vars(args).items() if isinstance(v, (str, int, float, bool))
    }
    init_kwargs: Dict[str, Any] = {
        "project": args.wandb_project,
        "mode": args.wandb_mode,
        "config": config,
    }
    if args.wandb_entity:
        init_kwargs["entity"] = args.wandb_entity
    if args.wandb_name:
        init_kwargs["name"] = args.wandb_name
    return wandb.init(**init_kwargs)


def _build_dataset(
    args: argparse.Namespace, rank: int, world_size: int
) -> Tuple[Any, Any, bool]:
    shard_paths = sorted(glob.glob(args.pretok_glob))
    if not shard_paths:
        raise FileNotFoundError(f"No pretokenized tar shards for glob: {args.pretok_glob}")
    ds = PretokShardDataset(
        shard_paths=shard_paths,
        rank=rank,
        world_size=world_size,
        seed=args.seed,
        shuffle=args.shuffle,
    )
    return ds, collate_pretok, True


def _build_loader(
    args: argparse.Namespace, ds, data_collate, rank: int
) -> DataLoader:
    return DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=data_collate,
        worker_init_fn=partial(_seed_worker, base_seed=args.seed, rank=rank),
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers and args.num_workers > 0,
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
    )


def _save_step_checkpoint(
    args: argparse.Namespace,
    wrapper: EmuNAR,
    rank: int,
    is_main: bool,
    global_step: int,
) -> None:
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


def _save_epoch_checkpoint(
    args: argparse.Namespace,
    wrapper: EmuNAR,
    rank: int,
    is_main: bool,
    epoch: int,
    save_epoch_mode: str,
) -> None:
    if save_epoch_mode == "none":
        return
    save_path = os.path.join(args.save_dir, f"nar_epoch{epoch}.pt")
    if args.fsdp:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp import FullStateDictConfig, ShardedStateDictConfig, StateDictType

        if save_epoch_mode == "full":
            full_path = os.path.join(args.save_dir, f"nar_epoch{epoch}.full.pt")
            state_cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            with FSDP.state_dict_type(wrapper, StateDictType.FULL_STATE_DICT, state_cfg):
                state_dict = wrapper.state_dict()
            if rank == 0:
                torch.save(state_dict, full_path)
                print("[INFO] saved full:", full_path)
            dist.barrier()
        else:
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


def _save_final_checkpoint(args: argparse.Namespace, wrapper: EmuNAR, rank: int) -> None:
    if not args.fsdp or not args.save_full_state:
        return
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import FullStateDictConfig, StateDictType

    final_path = os.path.join(args.save_dir, "nar_final.pt")
    state_cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(wrapper, StateDictType.FULL_STATE_DICT, state_cfg):
        state_dict = wrapper.state_dict()
    if rank == 0:
        torch.save(state_dict, final_path)
        print("[INFO] saved full model:", final_path)
    dist.barrier()


def _run_epoch_generate(
    args: argparse.Namespace,
    wrapper: EmuNAR,
    tokenizer,
    vq_model,
    vq_dtype,
    device: torch.device,
    visual_token_offset: int,
    rank: int,
    is_main: bool,
    epoch: int,
    run_tag: str = "",
) -> Tuple[Any, Any]:
    if not args.eval_generate_prompt:
        return vq_model, vq_dtype
    height = int(args.eval_generate_height)
    width = int(args.eval_generate_width)
    if height <= 0 or width <= 0:
        if is_main:
            print("[WARN] eval_generate_height/width must be set to enable generation.")
        return vq_model, vq_dtype
    out_dir = args.eval_generate_outdir or args.save_dir
    if is_main:
        os.makedirs(out_dir, exist_ok=True)
    tag = run_tag if run_tag else f"epoch{epoch}"

    def _save_outputs(grid: torch.Tensor) -> None:
        if not is_main:
            return
        grid_cpu = grid.detach().cpu()
        pt_path = os.path.join(out_dir, f"gen_{tag}.pt")
        torch.save(grid_cpu, pt_path)
        print(f"[EVAL] saved grid: {pt_path}")
        if not args.eval_generate_decode:
            return
        nonlocal vq_model, vq_dtype
        if vq_model is None or str(next(vq_model.parameters()).device) != str(device):
            vq_model = build_vision_tokenizer(args.vq_type, args.vq_path, device=str(device))
            vq_model.eval()
            for p in vq_model.parameters():
                p.requires_grad = False
            vq_dtype = next(vq_model.parameters()).dtype
        codes = grid_cpu - visual_token_offset
        codes = codes.to(device=next(vq_model.parameters()).device, dtype=torch.long)
        with torch.no_grad():
            image = vq_model.decode_code(codes[None], shape=(1, height, width, 256)).float()
        image = image[0].permute(1, 2, 0)
        try:
            from PIL import Image
            import numpy as np

            img = Image.fromarray(((image + 1.0) * 127.5).clamp(0, 255).cpu().numpy().astype(np.uint8))
            png_path = os.path.join(out_dir, f"gen_{tag}.png")
            img.save(png_path)
            print(f"[EVAL] saved image: {png_path}")
        except Exception as exc:
            print(f"[WARN] decode image failed: {exc}")

    was_training = wrapper.training
    wrapper.eval()

    prompt_ids = _encode_text_ids(
        tokenizer=tokenizer,
        text_template=args.text_template,
        text=args.eval_generate_prompt,
        text_max_length=args.text_max_length,
        add_boi=args.add_boi,
        height=height,
        width=width,
    ).to(device)
    if prompt_ids.dim() == 1:
        prompt_ids = prompt_ids.unsqueeze(0)
    prompt_attention = torch.ones((prompt_ids.size(0), prompt_ids.size(1)), dtype=torch.long, device=device)

    mask_ids = wrapper.module if hasattr(wrapper, "module") else wrapper
    mask_token_id = mask_ids.mask_token_id
    grid = torch.full((1, height, width), mask_token_id, device=device, dtype=torch.long)
    rows = torch.arange(height, device=device).unsqueeze(1).expand(height, width)
    cols = torch.arange(width, device=device).unsqueeze(0).expand(height, width)
    step_id = (rows + cols).reshape(-1)
    max_step = int(step_id.max().item())

    # Fix RNG for reproducible generation across training steps.
    _rng_cpu = torch.get_rng_state()
    _rng_cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    torch.manual_seed(args.eval_generate_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.eval_generate_seed)

    prev_fastpath = None
    if hasattr(torch.backends, "mha"):
        prev_fastpath = torch.backends.mha.get_fastpath_enabled()
        torch.backends.mha.set_fastpath_enabled(False)
    sdp_ctx = nullcontext()
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "sdp_kernel"):
        sdp_ctx = torch.backends.cuda.sdp_kernel(
            enable_flash=False, enable_mem_efficient=False, enable_math=True
        )
    with sdp_ctx, torch.inference_mode():
        for step in range(0, max_step + 1):
            positions = (step_id == step).nonzero(as_tuple=False).view(-1)
            if positions.numel() == 0:
                continue
            image_ids = grid.view(1, -1)
            prev_positions = positions if step == 0 else (step_id == (step - 1)).nonzero(
                as_tuple=False
            ).view(-1)
            positions = positions.to(device=device, dtype=torch.long)
            prev_positions = prev_positions.to(device=device, dtype=torch.long)
            outputs = wrapper(
                input_ids=image_ids,
                height=height,
                width=width,
                text_input_ids=prompt_ids,
                text_attention_mask=prompt_attention,
                step_positions=positions,
                prev_positions=prev_positions,
            )
            step_logits = outputs["step_logits"]
            if visual_token_offset:
                step_logits = step_logits.clone()
                step_logits[:, :, :visual_token_offset] = float("-inf")
            step_pred = _sample_logits(
                step_logits,
                temperature=args.eval_generate_temperature,
                top_k=args.eval_generate_top_k,
                top_p=args.eval_generate_top_p,
                sample_logits=args.eval_generate_sample,
            )
            if args.fsdp and dist.is_initialized():
                dist.broadcast(step_pred, src=0)
            grid.view(1, -1)[:, positions] = step_pred

    if prev_fastpath is not None:
        torch.backends.mha.set_fastpath_enabled(prev_fastpath)
    _save_outputs(grid)
    if args.fsdp and dist.is_initialized():
        dist.barrier()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # Restore RNG so training randomness is unaffected by eval.
    torch.set_rng_state(_rng_cpu)
    if _rng_cuda is not None:
        torch.cuda.set_rng_state_all(_rng_cuda)

    if was_training:
        wrapper.train()
    return vq_model, vq_dtype


def main() -> None:
    args = parse_args()
    if not args.fsdp:
        raise ValueError("This training entry requires FSDP and hard-enables it in parse_args().")
    if not args.pretok_glob:
        raise ValueError("train.sh flow requires --pretok_glob with pretokenized tar shards.")
    if args.max_steps > 0 and args.vertical_head_warmup_steps >= args.max_steps:
        adjusted = max(0, args.max_steps - 1)
        if dist.is_initialized():
            if dist.get_rank() == 0:
                print(
                    f"[WARN] vertical_head_warmup_steps={args.vertical_head_warmup_steps} "
                    f">= max_steps={args.max_steps}; clamped to {adjusted} "
                    "to ensure non-warmup training steps exist."
                )
        else:
            print(
                f"[WARN] vertical_head_warmup_steps={args.vertical_head_warmup_steps} "
                f">= max_steps={args.max_steps}; clamped to {adjusted} "
                "to ensure non-warmup training steps exist."
            )
        args.vertical_head_warmup_steps = adjusted
    random.seed(args.seed)
    rank, world_size, local_rank, device = _setup_distributed(args)

    is_main = rank == 0
    _prepare_output_dir(args, is_main)
    if is_main and args.train_backbone:
        print("[WARN] training full backbone from AR init; use conservative LR.")
    if is_main and args.vertical_head_warmup_steps > 0:
        print(
            f"[INFO] warmup enabled: train vertical branch "
            f"(vertical_block/vertical_norm/vertical_head + fuse gates) only for first "
            f"{args.vertical_head_warmup_steps} steps."
        )

    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    torch_dtype = dtype_map[args.dtype]

    model_config = Emu3Config.from_pretrained(args.model_path, trust_remote_code=True)
    backbone = _build_backbone(args, model_config, torch_dtype, device, is_main)
    wrapper, visual_token_offset = _build_wrapper(args, backbone, model_config, torch_dtype, device)
    del backbone
    wrapper = _wrap_fsdp_if_needed(args, wrapper, device)
    _resume_if_needed(args, wrapper, rank, is_main)
    _freeze_backbone_if_needed(args, wrapper)
    warmup_target = wrapper.module if args.fsdp else wrapper
    trainable_state = _snapshot_requires_grad(warmup_target)
    vertical_warmup_steps = max(0, int(args.vertical_head_warmup_steps))
    vertical_only_active = vertical_warmup_steps > 0
    phase2_start_step = 0  # set when phase 2 begins
    optimizer = _build_optimizer(args, wrapper)
    initial_lr = float(optimizer.param_groups[0]["lr"])  # remember for phase2 reset

    # Build cosine scheduler immediately if there is no warmup phase.
    scheduler: "Optional[torch.optim.lr_scheduler.LRScheduler]" = None
    if args.lr_scheduler == "cosine" and args.max_steps > 0 and not vertical_only_active:
        phase2_lr = float(optimizer.param_groups[0]["lr"])
        lr_min = phase2_lr * args.lr_min_factor
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.max_steps, eta_min=lr_min
        )
        if is_main:
            print(f"[INFO] cosine scheduler: T_max={args.max_steps} lr {phase2_lr:.2e} -> {lr_min:.2e}")

    if vertical_only_active:
        _set_vertical_branch_trainable(warmup_target)
        if is_main:
            print(
                f"[INFO] vertical-branch warmup active: "
                f"steps 1..{vertical_warmup_steps}."
            )

    tokenizer = _build_text_tokenizer(args.tokenizer_path)

    ds, data_collate, _ = _build_dataset(args, rank, world_size)
    loader = _build_loader(args, ds, data_collate, rank)
    vq_model, vq_dtype, _ = _build_vq(args, device)

    save_epoch_mode = args.save_epoch
    if save_epoch_mode == "auto":
        save_epoch_mode = "sharded" if args.fsdp else "full"
    wandb_run = _init_wandb(args, is_main)

    wrapper.train()
    global_step = 0
    accum_steps = max(1, args.grad_accum_steps)
    stop_training = False

    for epoch in range(args.epochs):
        if hasattr(ds, "set_epoch"):
            ds.set_epoch(epoch)
        if args.eval_generate_timing in ("start", "both"):
            vq_model, vq_dtype = _run_epoch_generate(
                args,
                wrapper,
                tokenizer,
                vq_model,
                vq_dtype,
                device,
                visual_token_offset,
                rank,
                is_main,
                epoch,
                run_tag=f"epoch{epoch}_start",
            )
        if is_main:
            progress = tqdm(desc=f"epoch {epoch}")
        else:
            progress = None
        data_iter = iter(loader)
        last_step = None
        step = 0
        while True:
            has_batch = True
            try:
                batch = next(data_iter)
            except StopIteration:
                has_batch = False
                batch = None

            if args.fsdp and world_size > 1:
                flag = torch.tensor([1 if has_batch else 0], device=device)
                dist.all_reduce(flag, op=dist.ReduceOp.MIN)
                if flag.item() == 0:
                    if is_main:
                        print("[WARN] some rank ran out of data; stopping epoch early.")
                    break
            elif not has_batch:
                break

            last_step = step
            tokens = batch["tokens"].long()
            texts = batch["texts"]
            height = int(tokens.size(1))
            width = int(tokens.size(2))
            if visual_token_offset:
                tokens = tokens + visual_token_offset
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
            if args.fsdp and args.fsdp_no_sync and not is_sync_step and accum_steps > 1:
                sync_ctx = wrapper.no_sync()
            else:
                sync_ctx = nullcontext()

            with sync_ctx:
                distill_active = vertical_only_active and args.ar_distill_weight > 0.0
                outputs = wrapper(
                    input_ids=input_ids,
                    height=height,
                    width=width,
                    text_input_ids=text_ids,
                    text_attention_mask=text_attention_mask,
                    ar_distill=distill_active,
                    ar_distill_temperature=args.ar_distill_temperature,
                )
                raw_loss = outputs["loss"]
                raw_loss_h = outputs.get("loss_h", raw_loss.detach().new_zeros(()))
                raw_loss_v = outputs.get("loss_v", raw_loss.detach().new_zeros(()))
                raw_loss_distill = outputs.get(
                    "loss_distill", raw_loss.detach().new_zeros(())
                )
                raw_fuse_w_h = outputs.get(
                    "fuse_w_h", raw_loss.detach().new_tensor(float(args.fuse_h_init))
                )
                raw_fuse_w_v = outputs.get(
                    "fuse_w_v", raw_loss.detach().new_tensor(float(1.0 - args.fuse_h_init))
                )
                raw_total_loss = (
                    raw_loss
                    + args.aux_loss_h_weight * raw_loss_h
                    + args.aux_loss_v_weight * raw_loss_v
                    + args.ar_distill_weight * raw_loss_distill
                )
                loss = raw_total_loss / accum_steps
                loss.backward()

            if is_sync_step:
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(wrapper.parameters(), args.grad_clip)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if scheduler is not None:
                    scheduler.step()

            global_step += 1
            if progress is not None:
                progress.update(1)
            # Phase-2 cosine: start decaying after flat steps
            if (not vertical_only_active
                    and args.phase2_flat_steps > 0
                    and args.lr_scheduler == "cosine"
                    and args.max_steps > 0
                    and global_step == phase2_start_step + args.phase2_flat_steps):
                phase2_cosine_steps = max(1, args.max_steps - global_step)
                phase2_lr = float(optimizer.param_groups[0]["lr"])
                lr_min = phase2_lr * args.lr_min_factor
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=phase2_cosine_steps, eta_min=lr_min
                )
                if is_main:
                    print(
                        f"[INFO] phase2 cosine scheduler started at step={global_step}: "
                        f"T_max={phase2_cosine_steps} lr {phase2_lr:.2e} -> {lr_min:.2e}"
                    )
            # Phase-1 cosine: start decaying after flat warmup steps
            if (vertical_only_active
                    and args.phase1_flat_steps > 0
                    and global_step == args.phase1_flat_steps
                    and args.lr_scheduler == "cosine"):
                phase1_remaining = max(1, vertical_warmup_steps - global_step)
                phase1_lr = float(optimizer.param_groups[0]["lr"])
                phase1_lr_min = phase1_lr * args.lr_min_factor
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=phase1_remaining, eta_min=phase1_lr_min
                )
                if is_main:
                    print(
                        f"[INFO] phase1 cosine scheduler started at step={global_step}: "
                        f"T_max={phase1_remaining} lr {phase1_lr:.2e} -> {phase1_lr_min:.2e}"
                    )
            if vertical_only_active and global_step >= vertical_warmup_steps:
                _restore_requires_grad(warmup_target, trainable_state)
                vertical_only_active = False
                if args.phase2_lr_factor > 0:
                    old_lrs = [float(pg["lr"]) for pg in optimizer.param_groups]
                    # Reset from initial_lr (not cosine-end LR) to avoid inheriting tiny cosine value
                    phase2_lr_val = initial_lr * float(args.phase2_lr_factor)
                    for pg in optimizer.param_groups:
                        pg["lr"] = phase2_lr_val
                    new_lrs = [float(pg["lr"]) for pg in optimizer.param_groups]
                else:
                    old_lrs = [float(pg["lr"]) for pg in optimizer.param_groups]
                    new_lrs = old_lrs
                if is_main:
                    print(
                        f"[INFO] vertical-branch warmup finished at step={global_step}; "
                        "restored normal trainable parameters."
                    )
                    if args.phase2_lr_factor > 0:
                        print(
                            f"[INFO] phase2 lr scale applied: factor={args.phase2_lr_factor:.6f} "
                            f"lr0 {old_lrs[0]:.8e} -> {new_lrs[0]:.8e}"
                        )
                    else:
                        print("[INFO] phase2 lr scale disabled.")
                # Phase-2 cosine: delay start until after flat steps.
                scheduler = None  # reset any phase-1 scheduler
                phase2_start_step = global_step
                if args.lr_scheduler == "cosine" and args.phase2_flat_steps == 0 and args.max_steps > 0:
                    phase2_steps = max(1, args.max_steps - global_step)
                    phase2_lr = float(optimizer.param_groups[0]["lr"])
                    lr_min = phase2_lr * args.lr_min_factor
                    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                        optimizer, T_max=phase2_steps, eta_min=lr_min
                    )
                    if is_main:
                        print(
                            f"[INFO] phase2 cosine scheduler (immediate): T_max={phase2_steps} "
                            f"lr {phase2_lr:.2e} -> {lr_min:.2e}"
                        )
                elif args.lr_scheduler == "cosine" and args.phase2_flat_steps > 0 and is_main:
                    print(
                        f"[INFO] phase2 flat LR for {args.phase2_flat_steps} steps, "
                        f"then cosine decay from step {global_step + args.phase2_flat_steps}"
                    )

            current_step = global_step
            should_log_progress = (
                args.log_every_steps > 0 and current_step % args.log_every_steps == 0
            )
            should_log_wandb = (
                args.wandb and args.wandb_log_interval > 0 and current_step % args.wandb_log_interval == 0
            )
            if should_log_progress or should_log_wandb:
                log_metrics = torch.stack(
                    [
                        raw_loss.detach(),
                        raw_loss_h.detach(),
                        raw_loss_v.detach(),
                        raw_loss_distill.detach(),
                        raw_fuse_w_h.detach(),
                        raw_fuse_w_v.detach(),
                    ]
                )
                if args.fsdp and world_size > 1:
                    dist.all_reduce(log_metrics, op=dist.ReduceOp.SUM)
                    log_metrics /= float(world_size)
                if is_main and should_log_progress and progress is not None:
                    progress.set_postfix(
                        loss=f"{log_metrics[0].item():.4f}",
                        loss_h=f"{log_metrics[1].item():.4f}",
                        loss_v=f"{log_metrics[2].item():.4f}",
                        loss_distill=f"{log_metrics[3].item():.4f}",
                        fuse_h=f"{log_metrics[4].item():.3f}",
                        fuse_v=f"{log_metrics[5].item():.3f}",
                    )
                    print(
                        f"[METRIC] step={current_step} "
                        f"loss={log_metrics[0].item():.6f} "
                        f"loss_h={log_metrics[1].item():.6f} "
                        f"loss_v={log_metrics[2].item():.6f} "
                        f"loss_distill={log_metrics[3].item():.6f} "
                        f"fuse_h={log_metrics[4].item():.4f} "
                        f"fuse_v={log_metrics[5].item():.4f} "
                        f"lr={optimizer.param_groups[0]['lr']:.2e} "
                        f"aux_h_w={args.aux_loss_h_weight:.3f} "
                        f"aux_v_w={args.aux_loss_v_weight:.3f} "
                        f"distill_w={args.ar_distill_weight:.3f} "
                        f"distill_active={int(distill_active)}",
                        flush=True,
                    )
                if is_main and should_log_wandb and wandb_run is not None:
                    wandb_run.log(
                        {
                            "loss": float(log_metrics[0].item()),
                            "loss_h": float(log_metrics[1].item()),
                            "loss_v": float(log_metrics[2].item()),
                            "loss_distill": float(log_metrics[3].item()),
                            "fuse_w_h": float(log_metrics[4].item()),
                            "fuse_w_v": float(log_metrics[5].item()),
                            "aux_loss_h_weight": float(args.aux_loss_h_weight),
                            "aux_loss_v_weight": float(args.aux_loss_v_weight),
                            "ar_distill_weight": float(args.ar_distill_weight),
                            "ar_distill_active": int(distill_active),
                            "epoch": epoch,
                        },
                        step=current_step,
                    )
            if args.save_every_steps > 0 and global_step % args.save_every_steps == 0:
                _save_step_checkpoint(args, wrapper, rank, is_main, global_step)
            if args.eval_generate_every_steps > 0 and global_step % args.eval_generate_every_steps == 0:
                vq_model, vq_dtype = _run_epoch_generate(
                    args,
                    wrapper,
                    tokenizer,
                    vq_model,
                    vq_dtype,
                    device,
                    visual_token_offset,
                    rank,
                    is_main,
                    epoch,
                    run_tag=f"step{global_step}",
                )
            if args.max_steps > 0 and global_step >= args.max_steps:
                stop_training = True
                if is_main:
                    print(f"[INFO] reached max_steps={args.max_steps}; stopping training.")
                break

            step += 1

        if progress is not None:
            progress.close()

        if accum_steps > 1 and last_step is not None and (last_step + 1) % accum_steps != 0:
            if args.fsdp:
                if is_main:
                    print(
                        "[WARN] Dropping last incomplete grad accumulation for FSDP; "
                        "set grad_accum_steps=1 or make epoch length divisible to avoid this."
                    )
                optimizer.zero_grad(set_to_none=True)
            else:
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(wrapper.parameters(), args.grad_clip)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

        if stop_training:
            if is_main:
                print("[INFO] max_steps reached, skip epoch-end checkpoint/generation.")
            break

        _save_epoch_checkpoint(args, wrapper, rank, is_main, epoch, save_epoch_mode)

        if args.eval_generate_timing in ("end", "both"):
            vq_model, vq_dtype = _run_epoch_generate(
                args,
                wrapper,
                tokenizer,
                vq_model,
                vq_dtype,
                device,
                visual_token_offset,
                rank,
                is_main,
                epoch,
                run_tag=f"epoch{epoch}_end",
            )

    _save_final_checkpoint(args, wrapper, rank)
    if is_main and wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
