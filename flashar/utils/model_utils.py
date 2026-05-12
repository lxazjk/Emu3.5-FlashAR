from __future__ import annotations

import argparse
import glob
import json
import os
import os.path as osp
import random
from functools import partial
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

from flashar.data.pretokenized import PretokShardDataset, collate_pretok
from flashar.model import Emuflashar
from src.emu3p5 import Emu3Config, Emu3ForCausalLM
from src.vision_tokenizer import build_vision_tokenizer


def seed_worker(worker_id: int, base_seed: int, rank: int) -> None:
    seed = base_seed + worker_id + rank * 1000
    random.seed(seed)
    np.random.seed(seed)


def setup_distributed(args: argparse.Namespace) -> Tuple[int, int, int, torch.device]:
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


def validate_model_path(model_path: str) -> None:
    if not osp.isdir(model_path):
        return
    cfg_path = osp.join(model_path, "config.json")
    if not osp.exists(cfg_path):
        return
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            raw_cfg = json.load(f)
    except Exception:
        return
    model_type = str(raw_cfg.get("model_type", "")).lower()
    arch = " ".join(str(x) for x in raw_cfg.get("architectures", [])).lower()
    if "visionvq" in model_type or "visionvq" in arch:
        raise ValueError(
            f"model_path={model_path} points to VisionVQ weights (model_type={raw_cfg.get('model_type')}). "
            "train_flashar.py expects Emu3ForCausalLM weights. "
            "Use the LLM checkpoint (e.g. BAAI/Emu3.5-Image) for --model_path, "
            "and keep VisionTokenizer for --vq_path."
        )


def build_backbone(
    args: argparse.Namespace,
    model_config: Emu3Config,
    torch_dtype: torch.dtype,
    device: torch.device,
    is_main: bool,
) -> Emu3ForCausalLM:
    del is_main
    pretrained_kwargs = dict(
        config=model_config,
        torch_dtype=torch_dtype,
        attn_implementation="eager",
    )
    if args.low_cpu_mem_usage:
        pretrained_kwargs["low_cpu_mem_usage"] = True
    pretrained_kwargs["output_loading_info"] = True
    try:
        backbone, loading_info = Emu3ForCausalLM.from_pretrained(args.model_path, **pretrained_kwargs)
    except TypeError:
        pretrained_kwargs.pop("low_cpu_mem_usage", None)
        backbone, loading_info = Emu3ForCausalLM.from_pretrained(args.model_path, **pretrained_kwargs)
    missing = list(loading_info.get("missing_keys", []))
    unexpected = list(loading_info.get("unexpected_keys", []))
    if len(unexpected) > 128 and len(missing) > 128:
        sample_unexpected = ", ".join(unexpected[:5])
        raise ValueError(
            "Loaded checkpoint appears incompatible with Emu3ForCausalLM "
            f"(missing_keys={len(missing)}, unexpected_keys={len(unexpected)}). "
            f"Sample unexpected keys: {sample_unexpected}. "
            "Check --model_path points to Emu3.5/Emu3.5-Image language model weights, not VisionVQ weights."
        )
    if args.gradient_checkpointing:
        backbone.gradient_checkpointing_enable()
        backbone.config.use_cache = False
    if not args.fsdp:
        backbone = backbone.to(device)
    return backbone


def resolve_visual_offset(args: argparse.Namespace, model_config: Emu3Config) -> int:
    visual_token_offset = args.visual_token_offset
    if visual_token_offset < 0:
        visual_token_offset = int(model_config.eoi_token_id) + 1
    vocab_size = int(model_config.vocab_size)
    if visual_token_offset >= vocab_size:
        raise ValueError(
            f"visual_token_offset {visual_token_offset} must be < vocab_size {vocab_size}."
        )
    requested_visual = int(args.visual_vocab_size)
    available_visual = vocab_size - visual_token_offset
    if requested_visual <= 0:
        raise ValueError("visual_vocab_size must be positive.")
    if requested_visual > available_visual:
        codebook_size = int(getattr(model_config, "codebook_size", 0) or 0)
        raise ValueError(
            f"visual_token_offset {visual_token_offset} + visual_vocab_size {requested_visual} "
            f"exceeds vocab_size {vocab_size} (available_visual={available_visual}, "
            f"model_config.codebook_size={codebook_size}). "
            "Likely wrong --model_path/config (e.g. VisionVQ config used as LLM config)."
        )
    return visual_token_offset


def build_wrapper(
    args: argparse.Namespace,
    backbone: Emu3ForCausalLM,
    model_config: Emu3Config,
    torch_dtype: torch.dtype,
    device: torch.device,
) -> Tuple[Emuflashar, int]:
    visual_token_offset = resolve_visual_offset(args, model_config)
    vertical_layers = args.vertical_layers
    if vertical_layers <= 0:
        vertical_layers = int(getattr(model_config, "flashar_vertical_layers", 1))
    vertical_start_layer = int(getattr(args, "vertical_start_layer", -1))
    if vertical_start_layer < 0:
        vertical_start_layer = int(getattr(model_config, "flashar_vertical_start_layer", -1))
    wrapper = Emuflashar(
        pretrained_backbone=backbone.model,
        vocab_size=model_config.vocab_size,
        hidden_size=model_config.hidden_size,
        pad_token_id=-100,
        mask_token_id=model_config.pad_token_id,
        visual_token_offset=visual_token_offset,
        use_vertical_block=args.use_vertical_block,
        vertical_layers=vertical_layers,
        vertical_start_layer=vertical_start_layer,
        lm_head=backbone.lm_head,
        split_backbone=getattr(args, "split_backbone", False),
    )
    if args.fsdp:
        wrapper = wrapper.to(dtype=torch_dtype)
    else:
        wrapper = wrapper.to(device=device, dtype=torch_dtype)
    return wrapper, visual_token_offset


def wrap_fsdp_if_needed(
    args: argparse.Namespace,
    wrapper: Emuflashar,
    device: torch.device,
    sync_module_states: bool = False,
) -> Emuflashar:
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
        sync_module_states=sync_module_states,
        use_orig_params=args.fsdp_use_orig_params,
    )
    return FSDP(wrapper, **fsdp_kwargs)


def freeze_backbone_if_needed(args: argparse.Namespace, wrapper: Emuflashar) -> None:
    if args.train_backbone:
        return
    target = wrapper.module if args.fsdp else wrapper
    for p in target.backbone.parameters():
        p.requires_grad = False


def snapshot_requires_grad(module: torch.nn.Module) -> Dict[str, bool]:
    return {name: p.requires_grad for name, p in module.named_parameters()}


def restore_requires_grad(module: torch.nn.Module, state: Dict[str, bool]) -> None:
    for name, p in module.named_parameters():
        if name in state:
            p.requires_grad = state[name]


def set_vertical_branch_trainable(module: torch.nn.Module) -> None:
    for name, p in module.named_parameters():
        keep = (
            name.startswith("vertical_block.")
            or name.startswith("vertical_norm.")
            or name.startswith("vertical_head.")
            or name.startswith("hv_gate_mlp.")
            or name.startswith("hv_gate_corner.")
        )
        p.requires_grad = keep


def set_backbone_gradient_checkpointing(
    module: torch.nn.Module,
    enabled: bool,
) -> None:
    backbone = getattr(module, "backbone", None)
    if backbone is None:
        return
    if enabled:
        if hasattr(backbone, "gradient_checkpointing_enable"):
            backbone.gradient_checkpointing_enable()
        elif hasattr(backbone, "gradient_checkpointing"):
            backbone.gradient_checkpointing = True
        if getattr(backbone, "config", None) is not None:
            backbone.config.use_cache = False
        return
    if hasattr(backbone, "gradient_checkpointing_disable"):
        backbone.gradient_checkpointing_disable()
    elif hasattr(backbone, "gradient_checkpointing"):
        backbone.gradient_checkpointing = False


def is_backbone_param_name(name: str) -> bool:
    return (
        name.startswith("backbone.")
        or ".backbone." in name
        or name.startswith("_fsdp_wrapped_module.backbone.")
        or "._fsdp_wrapped_module.backbone." in name
    )


def format_group_lrs(optimizer: torch.optim.Optimizer) -> str:
    items = []
    for idx, group in enumerate(optimizer.param_groups):
        group_name = group.get("name", f"group{idx}")
        items.append(f"{group_name}={float(group['lr']):.2e}")
    return ", ".join(items)


def build_optimizer(args: argparse.Namespace, wrapper: Emuflashar) -> torch.optim.Optimizer:
    backbone_params: List[torch.nn.Parameter] = []
    other_params: List[torch.nn.Parameter] = []
    for name, p in wrapper.named_parameters():
        if not (p.requires_grad and p.numel() > 0):
            continue
        if args.train_backbone and is_backbone_param_name(name):
            backbone_params.append(p)
        else:
            other_params.append(p)

    if not backbone_params and not other_params:
        raise ValueError("No trainable parameters found. Check freeze settings.")

    if args.train_backbone and backbone_params:
        backbone_lr = float(args.backbone_lr)
        if backbone_lr <= 0:
            backbone_lr = float(args.lr) * float(args.backbone_lr_factor)
        if backbone_lr <= 0:
            backbone_lr = float(args.lr)

        param_groups: List[Dict[str, Any]] = []
        if other_params:
            param_groups.append(
                {"params": other_params, "lr": float(args.lr), "name": "non_backbone"}
            )
        param_groups.append(
            {"params": backbone_params, "lr": backbone_lr, "name": "backbone"}
        )
        return torch.optim.AdamW(param_groups, lr=float(args.lr), foreach=True)

    trainable_params = other_params if other_params else backbone_params
    return torch.optim.AdamW(trainable_params, lr=float(args.lr), foreach=True)


def build_vq(args: argparse.Namespace, device: torch.device) -> Tuple[Any, Any, torch.device]:
    del args
    return None, None, device


def init_wandb(args: argparse.Namespace, is_main: bool):
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


def build_dataset(
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


def build_loader(args: argparse.Namespace, ds, data_collate, rank: int) -> DataLoader:
    return DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=data_collate,
        worker_init_fn=partial(seed_worker, base_seed=args.seed, rank=rank),
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers and args.num_workers > 0,
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
    )
