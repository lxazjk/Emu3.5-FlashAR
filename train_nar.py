# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import random
from typing import Optional

import torch
import torch.distributed as dist

from src.emu3p5 import Emu3Config
from emu_nar.utils.checkpoint_utils import (
    infer_resume_step as _infer_resume_step,
    preload_single_file_resume_if_needed as _preload_single_file_resume_if_needed,
    prepare_output_dir as _prepare_output_dir,
    resolve_resume_path as _resolve_resume_path,
    resume_if_needed as _resume_if_needed,
    save_checkpoint as _save_checkpoint,
)
from emu_nar.utils.config_utils import load_train_arg_defaults
from emu_nar.utils.model_utils import (
    build_backbone as _build_backbone,
    build_dataset as _build_dataset,
    build_loader as _build_loader,
    build_optimizer as _build_optimizer,
    build_vq as _build_vq,
    build_wrapper as _build_wrapper,
    format_group_lrs as _format_group_lrs,
    freeze_backbone_if_needed as _freeze_backbone_if_needed,
    init_wandb as _init_wandb,
    set_backbone_gradient_checkpointing as _set_backbone_gradient_checkpointing,
    set_vertical_branch_trainable as _set_vertical_branch_trainable,
    setup_distributed as _setup_distributed,
    snapshot_requires_grad as _snapshot_requires_grad,
    validate_model_path as _validate_model_path,
    wrap_fsdp_if_needed as _wrap_fsdp_if_needed,
)
from emu_nar.utils.text_utils import build_text_tokenizer as _build_text_tokenizer
from emu_nar.utils.training_utils import (
    TrainingLoopState,
    run_training_epochs as _run_training_epochs,
)


def parse_args() -> argparse.Namespace:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config_json", type=str, default="")
    pre_args, _ = pre_parser.parse_known_args()

    parser = argparse.ArgumentParser(parents=[pre_parser])
    parser.add_argument("--model_path", type=str, default="")
    parser.add_argument("--tokenizer_path", type=str, default="")
    parser.add_argument("--vq_path", type=str, default="")
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
    parser.add_argument(
        "--backbone_lr",
        type=float,
        default=0.0,
        help=(
            "Backbone LR when train_backbone is enabled. "
            "If <=0, uses lr * backbone_lr_factor."
        ),
    )
    parser.add_argument(
        "--backbone_lr_factor",
        type=float,
        default=0.1,
        help="Fallback backbone LR factor: backbone_lr = lr * backbone_lr_factor.",
    )
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
    parser.add_argument(
        "--save_latest_only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Overwrite a fixed latest checkpoint path instead of creating "
            "step/epoch-indexed files."
        ),
    )
    parser.add_argument("--save_every_steps", type=int, default=0)
    parser.add_argument("--max_steps", type=int, default=0)
    parser.add_argument("--log_every_steps", type=int, default=10)
    parser.add_argument("--resume_path", type=str, default="")
    parser.add_argument("--fresh_run", action=argparse.BooleanOptionalAction, default=False)
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
    parser.add_argument(
        "--hf_max_shard_size",
        type=str,
        default="5GB",
        help="HF safetensors max shard size for saved checkpoints, e.g. 5GB.",
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
    parser.add_argument(
        "--visual_vocab_size",
        type=int,
        default=131072,
        help="Visual vocabulary size used for bounds checking.",
    )
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--persistent_workers", action="store_true")
    parser.add_argument("--pin_memory", action="store_true")
    parser.add_argument("--use_vertical_block", action="store_true")
    parser.add_argument("--vertical_layers", type=int, default=0)
    parser.add_argument(
        "--vertical_start_layer",
        type=int,
        default=-1,
        help=(
            "Backbone layer index where the vertical branch starts. "
            "Set <0 to auto-use num_hidden_layers - vertical_layers."
        ),
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
    parser.add_argument("--split_backbone", action="store_true",
                        help="Split backbone at vertical_start_layer for parallel H/V branches.")
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
    parser.add_argument(
        "--gate_collapse_weight",
        type=float,
        default=0.0,
        help=(
            "Weight for hv-gate anti-collapse regularization. "
            "Minimizes 1 - normalized binary entropy of the interior gate."
        ),
    )
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
    if pre_args.config_json:
        config_defaults = load_train_arg_defaults(pre_args.config_json)
        valid_dests = {action.dest for action in parser._actions}
        unknown_keys = sorted(set(config_defaults) - valid_dests)
        if unknown_keys:
            raise ValueError(
                f"Unknown keys in config_json={pre_args.config_json}: {unknown_keys}"
            )
        parser.set_defaults(**config_defaults)
    args = parser.parse_args()
    args.fsdp = True
    return args


def main() -> None:
    args = parse_args()
    required_paths = {
        "model_path": "--model_path",
        "tokenizer_path": "--tokenizer_path",
        "vq_path": "--vq_path",
    }
    for field, flag in required_paths.items():
        if not getattr(args, field):
            raise ValueError(
                f"Missing required setting {field}. "
                f"Provide it via {flag} or --config_json."
            )
    _validate_model_path(args.model_path)
    if not args.fsdp:
        raise ValueError("This training entry requires FSDP and hard-enables it in parse_args().")
    if not args.pretok_glob:
        raise ValueError("train.sh flow requires --pretok_glob with pretokenized tar shards.")
    if args.grad_accum_steps < 1:
        raise ValueError(f"grad_accum_steps must be >= 1, got {args.grad_accum_steps}.")
    if args.fsdp_cpu_offload and args.grad_accum_steps > 1 and not args.fsdp_no_sync:
        raise ValueError(
            "Unsupported FSDP config: fsdp_cpu_offload=true with grad_accum_steps>1 "
            "requires fsdp_no_sync=true. PyTorch documents that gradient accumulation "
            "outside no_sync() is not supported with CPU offloading."
        )
    if args.fsdp_cpu_offload and args.grad_accum_steps > 1 and args.fsdp_use_orig_params:
        raise ValueError(
            "Unsupported FSDP config for this training entry: "
            "fsdp_use_orig_params=true + fsdp_cpu_offload=true + grad_accum_steps>1 "
            "can trigger upstream FSDP backward-hook failures (for example "
            "`AssertionError: check _post_backward_hook`) even when fsdp_no_sync=true. "
            "Use one of these instead: set fsdp_cpu_offload=false, or set "
            "grad_accum_steps=1. Keeping use_orig_params=true is still recommended "
            "for this repo's trainable-parameter switching logic."
        )
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
    _resolve_resume_path(args, is_main)
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
    if is_main and getattr(wrapper, "use_vertical_block", False):
        print(
            "[INFO] vertical branch layout: "
            f"start_layer={int(getattr(wrapper, 'vertical_start_layer', -1))} "
            f"depth={int(getattr(wrapper, 'vertical_layers', 0))} "
            f"backbone_layers={int(getattr(wrapper, 'backbone_num_layers', 0))}"
        )
    preloaded_single_file_resume = _preload_single_file_resume_if_needed(
        args,
        wrapper,
        rank,
        is_main,
    )
    del backbone
    wrapper = _wrap_fsdp_if_needed(
        args,
        wrapper,
        device,
        sync_module_states=preloaded_single_file_resume,
    )
    if not preloaded_single_file_resume:
        _resume_if_needed(args, wrapper, rank, is_main)
    resume_step = _infer_resume_step(args, is_main)
    _freeze_backbone_if_needed(args, wrapper)
    warmup_target = wrapper.module if args.fsdp else wrapper
    trainable_state = _snapshot_requires_grad(warmup_target)
    vertical_warmup_steps = max(0, int(args.vertical_head_warmup_steps))
    global_step = max(0, int(resume_step))
    vertical_only_active = vertical_warmup_steps > global_step
    phase2_start_step = vertical_warmup_steps if not vertical_only_active else 0
    if is_main and global_step > 0:
        print(f"[INFO] resume global_step={global_step}")
    optimizer = _build_optimizer(args, wrapper)
    initial_lrs = [float(pg["lr"]) for pg in optimizer.param_groups]  # remember for phase2 reset
    if is_main:
        print(f"[INFO] optimizer LR groups: {_format_group_lrs(optimizer)}")

    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None
    resumed_after_warmup = vertical_warmup_steps > 0 and global_step >= vertical_warmup_steps
    if resumed_after_warmup:
        if args.phase2_lr_factor > 0:
            for group_idx, pg in enumerate(optimizer.param_groups):
                pg["lr"] = initial_lrs[group_idx] * float(args.phase2_lr_factor)
            if is_main:
                print(
                    f"[INFO] resumed after warmup (step={global_step}); "
                    f"applied phase2 lr factor={args.phase2_lr_factor:.6f}. "
                    f"LR groups: {_format_group_lrs(optimizer)}"
                )
        elif is_main:
            print("[INFO] resumed after warmup; phase2 lr scale disabled.")

    if args.lr_scheduler == "cosine" and args.max_steps > 0:
        if vertical_only_active:
            # Resume inside phase-1 warmup: recreate phase1 cosine scheduler if already past flat steps.
            if global_step >= args.phase1_flat_steps:
                phase1_remaining = max(1, vertical_warmup_steps - args.phase1_flat_steps)
                phase1_lr = float(optimizer.param_groups[0]["lr"])
                phase1_lr_min = phase1_lr * args.lr_min_factor
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=phase1_remaining, eta_min=phase1_lr_min
                )
                progressed = min(phase1_remaining, global_step - args.phase1_flat_steps)
                if progressed > 0:
                    scheduler.step(progressed)
                if is_main:
                    print(
                        f"[INFO] restored phase1 cosine scheduler at step={global_step}: "
                        f"progress={progressed}/{phase1_remaining}, "
                        f"lr {phase1_lr:.2e} -> {phase1_lr_min:.2e}"
                    )
        else:
            phase2_cosine_start = phase2_start_step + max(0, int(args.phase2_flat_steps))
            if global_step >= phase2_cosine_start:
                phase2_steps_total = max(1, args.max_steps - phase2_cosine_start)
                phase2_lr = float(optimizer.param_groups[0]["lr"])
                lr_min = phase2_lr * args.lr_min_factor
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=phase2_steps_total, eta_min=lr_min
                )
                progressed = min(phase2_steps_total, global_step - phase2_cosine_start)
                if progressed > 0:
                    scheduler.step(progressed)
                if is_main:
                    print(
                        f"[INFO] restored phase2 cosine scheduler at step={global_step}: "
                        f"progress={progressed}/{phase2_steps_total}, "
                        f"lr {phase2_lr:.2e} -> {lr_min:.2e}"
                    )
            elif is_main and args.phase2_flat_steps > 0 and vertical_warmup_steps > 0:
                remain = phase2_cosine_start - global_step
                print(f"[INFO] resumed in phase2 flat-LR stage; remaining flat steps: {remain}")

    if vertical_only_active:
        _set_vertical_branch_trainable(warmup_target)
        if args.gradient_checkpointing:
            _set_backbone_gradient_checkpointing(warmup_target, enabled=False)
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
        save_epoch_mode = "full"
    wandb_run = _init_wandb(args, is_main)
    loop_state = TrainingLoopState(
        optimizer=optimizer,
        scheduler=scheduler,
        initial_lrs=initial_lrs,
        trainable_state=trainable_state,
        global_step=global_step,
        vertical_only_active=vertical_only_active,
        vertical_warmup_steps=vertical_warmup_steps,
        phase2_start_step=phase2_start_step,
        save_epoch_mode=save_epoch_mode,
        vq_model=vq_model,
        vq_dtype=vq_dtype,
        wandb_run=wandb_run,
    )
    loop_state = _run_training_epochs(
        args,
        wrapper=wrapper,
        tokenizer=tokenizer,
        ds=ds,
        loader=loader,
        device=device,
        visual_token_offset=visual_token_offset,
        rank=rank,
        world_size=world_size,
        is_main=is_main,
        warmup_target=warmup_target,
        state=loop_state,
    )

    _save_checkpoint(
        args,
        wrapper,
        rank,
        is_main,
        loop_state.global_step,
        save_reason="final",
    )
    if is_main and wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
