from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.distributed as dist
from tqdm import tqdm

from emu_nar.utils.checkpoint_utils import (
    save_checkpoint,
)
from emu_nar.utils.eval_utils import run_epoch_generate
from emu_nar.utils.model_utils import (
    restore_requires_grad,
    set_backbone_gradient_checkpointing,
)
from emu_nar.utils.text_utils import encode_text_ids, pad_text_ids


@dataclass
class TrainingLoopState:
    optimizer: torch.optim.Optimizer
    scheduler: Optional[Any]
    initial_lrs: list[float]
    trainable_state: dict[str, bool]
    global_step: int
    vertical_only_active: bool
    vertical_warmup_steps: int
    phase2_start_step: int
    save_epoch_mode: str
    vq_model: Any
    vq_dtype: Any
    wandb_run: Any = None


@dataclass
class StepMetrics:
    raw_loss: torch.Tensor
    raw_loss_h: torch.Tensor
    raw_loss_v: torch.Tensor
    raw_loss_distill: torch.Tensor
    raw_loss_gate_collapse: torch.Tensor
    raw_hv_gate_entropy: torch.Tensor
    raw_hv_gate_h: torch.Tensor
    raw_hv_gate_v: torch.Tensor
    effective_aux_h_weight: float
    effective_aux_v_weight: float
    effective_distill_weight: float
    effective_gate_collapse_weight: float
    distill_active: bool


def _encode_batch_texts(
    args: argparse.Namespace,
    tokenizer: Any,
    texts: list[str],
    *,
    height: int,
    width: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    text_ids = [
        encode_text_ids(
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
    text_ids, text_attention_mask = pad_text_ids(text_ids, tokenizer.pad_token_id)
    return text_ids.to(device), text_attention_mask.to(device)


def _all_ranks_have_batch(
    args: argparse.Namespace,
    *,
    world_size: int,
    has_batch: bool,
    device: torch.device,
    is_main: bool,
) -> bool:
    if args.fsdp and world_size > 1:
        flag = torch.tensor([1 if has_batch else 0], device=device)
        dist.all_reduce(flag, op=dist.ReduceOp.MIN)
        if flag.item() == 0:
            if is_main:
                print("[WARN] some rank ran out of data; stopping epoch early.")
            return False
        return True
    return has_batch


def _forward_backward_step(
    args: argparse.Namespace,
    *,
    wrapper: Any,
    input_ids: torch.Tensor,
    height: int,
    width: int,
    text_ids: torch.Tensor,
    text_attention_mask: torch.Tensor,
    accum_steps: int,
    vertical_only_active: bool,
) -> StepMetrics:
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
    raw_loss_distill = outputs.get("loss_distill", raw_loss.detach().new_zeros(()))
    raw_loss_gate_collapse = outputs.get(
        "loss_gate_collapse", raw_loss.detach().new_zeros(())
    )
    raw_hv_gate_h = outputs.get("hv_gate_h", raw_loss.detach().new_tensor(0.5))
    raw_hv_gate_v = outputs.get("hv_gate_v", raw_loss.detach().new_tensor(0.5))
    raw_hv_gate_entropy = outputs.get(
        "hv_gate_entropy", raw_loss.detach().new_tensor(1.0)
    )
    effective_aux_h_weight = 0.0 if vertical_only_active else float(args.aux_loss_h_weight)
    effective_aux_v_weight = float(args.aux_loss_v_weight)
    effective_distill_weight = float(args.ar_distill_weight)
    effective_gate_collapse_weight = float(args.gate_collapse_weight)
    total_loss = (
        raw_loss
        + effective_aux_h_weight * raw_loss_h
        + effective_aux_v_weight * raw_loss_v
        + effective_distill_weight * raw_loss_distill
        + effective_gate_collapse_weight * raw_loss_gate_collapse
    )
    (total_loss / accum_steps).backward()
    return StepMetrics(
        raw_loss=raw_loss,
        raw_loss_h=raw_loss_h,
        raw_loss_v=raw_loss_v,
        raw_loss_distill=raw_loss_distill,
        raw_loss_gate_collapse=raw_loss_gate_collapse,
        raw_hv_gate_entropy=raw_hv_gate_entropy,
        raw_hv_gate_h=raw_hv_gate_h,
        raw_hv_gate_v=raw_hv_gate_v,
        effective_aux_h_weight=effective_aux_h_weight,
        effective_aux_v_weight=effective_aux_v_weight,
        effective_distill_weight=effective_distill_weight,
        effective_gate_collapse_weight=effective_gate_collapse_weight,
        distill_active=distill_active,
    )


def _maybe_start_cosine_scheduler(
    args: argparse.Namespace,
    state: TrainingLoopState,
    *,
    is_main: bool,
) -> None:
    if (
        not state.vertical_only_active
        and args.phase2_flat_steps > 0
        and args.lr_scheduler == "cosine"
        and args.max_steps > 0
        and state.global_step == state.phase2_start_step + args.phase2_flat_steps
    ):
        phase2_cosine_steps = max(1, args.max_steps - state.global_step)
        phase2_lr = float(state.optimizer.param_groups[0]["lr"])
        lr_min = phase2_lr * args.lr_min_factor
        state.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            state.optimizer,
            T_max=phase2_cosine_steps,
            eta_min=lr_min,
        )
        if is_main:
            print(
                f"[INFO] phase2 cosine scheduler started at step={state.global_step}: "
                f"T_max={phase2_cosine_steps} lr {phase2_lr:.2e} -> {lr_min:.2e}"
            )

    if (
        state.vertical_only_active
        and args.phase1_flat_steps > 0
        and state.global_step == args.phase1_flat_steps
        and args.lr_scheduler == "cosine"
    ):
        phase1_remaining = max(1, state.vertical_warmup_steps - state.global_step)
        phase1_lr = float(state.optimizer.param_groups[0]["lr"])
        phase1_lr_min = phase1_lr * args.lr_min_factor
        state.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            state.optimizer,
            T_max=phase1_remaining,
            eta_min=phase1_lr_min,
        )
        if is_main:
            print(
                f"[INFO] phase1 cosine scheduler started at step={state.global_step}: "
                f"T_max={phase1_remaining} lr {phase1_lr:.2e} -> {phase1_lr_min:.2e}"
            )


def _maybe_finish_vertical_warmup(
    args: argparse.Namespace,
    state: TrainingLoopState,
    *,
    warmup_target: torch.nn.Module,
    is_main: bool,
) -> None:
    if not state.vertical_only_active:
        return
    if state.global_step < state.vertical_warmup_steps:
        return

    restore_requires_grad(warmup_target, state.trainable_state)
    if args.gradient_checkpointing:
        set_backbone_gradient_checkpointing(warmup_target, enabled=True)
    state.vertical_only_active = False

    old_lrs = [float(pg["lr"]) for pg in state.optimizer.param_groups]
    if args.phase2_lr_factor > 0:
        for group_idx, pg in enumerate(state.optimizer.param_groups):
            pg["lr"] = state.initial_lrs[group_idx] * float(args.phase2_lr_factor)
        new_lrs = [float(pg["lr"]) for pg in state.optimizer.param_groups]
    else:
        new_lrs = old_lrs

    if is_main:
        print(
            f"[INFO] vertical-branch warmup finished at step={state.global_step}; "
            "restored normal trainable parameters."
        )
        if args.phase2_lr_factor > 0:
            lr_changes = ", ".join(
                f"g{i}: {old_lrs[i]:.8e}->{new_lrs[i]:.8e}"
                for i in range(len(new_lrs))
            )
            print(
                f"[INFO] phase2 lr scale applied: factor={args.phase2_lr_factor:.6f} "
                f"{lr_changes}"
            )
        else:
            print("[INFO] phase2 lr scale disabled.")

    state.scheduler = None
    state.phase2_start_step = state.global_step
    if args.lr_scheduler == "cosine" and args.phase2_flat_steps == 0 and args.max_steps > 0:
        phase2_steps = max(1, args.max_steps - state.global_step)
        phase2_lr = float(state.optimizer.param_groups[0]["lr"])
        lr_min = phase2_lr * args.lr_min_factor
        state.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            state.optimizer,
            T_max=phase2_steps,
            eta_min=lr_min,
        )
        if is_main:
            print(
                f"[INFO] phase2 cosine scheduler (immediate): T_max={phase2_steps} "
                f"lr {phase2_lr:.2e} -> {lr_min:.2e}"
            )
    elif args.lr_scheduler == "cosine" and args.phase2_flat_steps > 0 and is_main:
        print(
            f"[INFO] phase2 flat LR for {args.phase2_flat_steps} steps, "
            f"then cosine decay from step {state.global_step + args.phase2_flat_steps}"
        )


def _should_run_on_step(global_step: int, interval: int) -> bool:
    return interval > 0 and global_step > 0 and global_step % interval == 0


def _commit_optimizer_step(
    args: argparse.Namespace,
    *,
    wrapper: Any,
    state: TrainingLoopState,
    warmup_target: torch.nn.Module,
    progress: Any,
    is_main: bool,
) -> None:
    if args.grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(wrapper.parameters(), args.grad_clip)
    state.optimizer.step()
    state.optimizer.zero_grad(set_to_none=True)
    if state.scheduler is not None:
        state.scheduler.step()

    state.global_step += 1
    if progress is not None:
        progress.update(1)

    _maybe_start_cosine_scheduler(args, state, is_main=is_main)
    _maybe_finish_vertical_warmup(
        args,
        state,
        warmup_target=warmup_target,
        is_main=is_main,
    )


def _log_step_metrics(
    args: argparse.Namespace,
    state: TrainingLoopState,
    *,
    metrics: StepMetrics,
    progress: Any,
    world_size: int,
    is_main: bool,
    epoch: int,
) -> None:
    if state.global_step <= 0:
        return

    should_log_progress = _should_run_on_step(state.global_step, args.log_every_steps)
    should_log_wandb = (
        args.wandb
        and _should_run_on_step(state.global_step, args.wandb_log_interval)
    )
    if not (should_log_progress or should_log_wandb):
        return

    log_metrics = torch.stack(
        [
            metrics.raw_loss.detach(),
            metrics.raw_loss_h.detach(),
            metrics.raw_loss_v.detach(),
            metrics.raw_loss_distill.detach(),
            metrics.raw_loss_gate_collapse.detach(),
            metrics.raw_hv_gate_entropy.detach(),
            metrics.raw_hv_gate_h.detach(),
            metrics.raw_hv_gate_v.detach(),
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
            gate_collapse=f"{log_metrics[4].item():.4f}",
            gate_entropy=f"{log_metrics[5].item():.3f}",
            hv_gate_h=f"{log_metrics[6].item():.3f}",
            hv_gate_v=f"{log_metrics[7].item():.3f}",
        )
        print(
            f"[METRIC] step={state.global_step} "
            f"loss={log_metrics[0].item():.6f} "
            f"loss_h={log_metrics[1].item():.6f} "
            f"loss_v={log_metrics[2].item():.6f} "
            f"loss_distill={log_metrics[3].item():.6f} "
            f"loss_gate_collapse={log_metrics[4].item():.6f} "
            f"hv_gate_entropy={log_metrics[5].item():.4f} "
            f"hv_gate_h={log_metrics[6].item():.4f} "
            f"hv_gate_v={log_metrics[7].item():.4f} "
            f"lr={state.optimizer.param_groups[0]['lr']:.2e} "
            f"aux_h_w={metrics.effective_aux_h_weight:.3f} "
            f"aux_v_w={metrics.effective_aux_v_weight:.3f} "
            f"distill_w={metrics.effective_distill_weight:.3f} "
            f"gate_collapse_w={metrics.effective_gate_collapse_weight:.3f} "
            f"distill_active={int(metrics.distill_active)}",
            flush=True,
        )

    if is_main and should_log_wandb and state.wandb_run is not None:
        state.wandb_run.log(
            {
                "loss": float(log_metrics[0].item()),
                "loss_h": float(log_metrics[1].item()),
                "loss_v": float(log_metrics[2].item()),
                "loss_distill": float(log_metrics[3].item()),
                "loss_gate_collapse": float(log_metrics[4].item()),
                "hv_gate_entropy": float(log_metrics[5].item()),
                "hv_gate_h": float(log_metrics[6].item()),
                "hv_gate_v": float(log_metrics[7].item()),
                "aux_loss_h_weight": float(metrics.effective_aux_h_weight),
                "aux_loss_v_weight": float(metrics.effective_aux_v_weight),
                "ar_distill_weight": float(metrics.effective_distill_weight),
                "gate_collapse_weight": float(metrics.effective_gate_collapse_weight),
                "ar_distill_active": int(metrics.distill_active),
                "epoch": epoch,
            },
            step=state.global_step,
        )


def _run_post_optimizer_step(
    args: argparse.Namespace,
    state: TrainingLoopState,
    *,
    metrics: StepMetrics,
    progress: Any,
    world_size: int,
    is_main: bool,
    epoch: int,
    wrapper: Any,
    tokenizer: Any,
    device: torch.device,
    visual_token_offset: int,
    rank: int,
) -> bool:
    _log_step_metrics(
        args,
        state,
        metrics=metrics,
        progress=progress,
        world_size=world_size,
        is_main=is_main,
        epoch=epoch,
    )

    if _should_run_on_step(state.global_step, args.save_every_steps):
        save_checkpoint(
            args,
            wrapper,
            rank,
            is_main,
            state.global_step,
            epoch=epoch,
            save_reason="step",
        )
    if _should_run_on_step(state.global_step, args.eval_generate_every_steps):
        state.vq_model, state.vq_dtype = run_epoch_generate(
            args,
            wrapper,
            tokenizer,
            state.vq_model,
            state.vq_dtype,
            device,
            visual_token_offset,
            rank,
            is_main,
            epoch,
            run_tag=f"step{state.global_step}",
        )
    if args.max_steps > 0 and state.global_step >= args.max_steps:
        if is_main:
            print(f"[INFO] reached max_steps={args.max_steps}; stopping training.")
        return True
    return False


def _finalize_partial_accumulation(
    args: argparse.Namespace,
    *,
    wrapper: Any,
    state: TrainingLoopState,
    warmup_target: torch.nn.Module,
    last_step: Optional[int],
    accum_steps: int,
    progress: Any,
    is_main: bool,
) -> bool:
    if accum_steps <= 1 or last_step is None or (last_step + 1) % accum_steps == 0:
        return False
    if args.fsdp:
        if is_main:
            print(
                "[WARN] Dropping last incomplete grad accumulation for FSDP; "
                "set grad_accum_steps=1 or make epoch length divisible to avoid this."
            )
        state.optimizer.zero_grad(set_to_none=True)
        return False
    _commit_optimizer_step(
        args,
        wrapper=wrapper,
        state=state,
        warmup_target=warmup_target,
        progress=progress,
        is_main=is_main,
    )
    return True


def run_training_epochs(
    args: argparse.Namespace,
    *,
    wrapper: Any,
    tokenizer: Any,
    ds: Any,
    loader: Any,
    device: torch.device,
    visual_token_offset: int,
    rank: int,
    world_size: int,
    is_main: bool,
    warmup_target: torch.nn.Module,
    state: TrainingLoopState,
) -> TrainingLoopState:
    wrapper.train()
    accum_steps = max(1, args.grad_accum_steps)
    stop_training = False

    for epoch in range(args.epochs):
        if hasattr(ds, "set_epoch"):
            ds.set_epoch(epoch)
        if args.eval_generate_timing in ("start", "both"):
            state.vq_model, state.vq_dtype = run_epoch_generate(
                args,
                wrapper,
                tokenizer,
                state.vq_model,
                state.vq_dtype,
                device,
                visual_token_offset,
                rank,
                is_main,
                epoch,
                run_tag=f"epoch{epoch}_start",
            )

        progress = tqdm(desc=f"epoch {epoch}") if is_main else None
        data_iter = iter(loader)
        last_step = None
        last_metrics: Optional[StepMetrics] = None
        step = 0
        while True:
            has_batch = True
            try:
                batch = next(data_iter)
            except StopIteration:
                has_batch = False
                batch = None

            if not _all_ranks_have_batch(
                args,
                world_size=world_size,
                has_batch=has_batch,
                device=device,
                is_main=is_main,
            ):
                break

            last_step = step
            tokens = batch["tokens"].long()
            texts = batch["texts"]
            height = int(tokens.size(1))
            width = int(tokens.size(2))
            if visual_token_offset:
                tokens = tokens + visual_token_offset
            input_ids = tokens.view(tokens.size(0), -1).to(device)
            text_ids, text_attention_mask = _encode_batch_texts(
                args,
                tokenizer,
                texts,
                height=height,
                width=width,
                device=device,
            )

            is_sync_step = (step + 1) % accum_steps == 0
            if args.fsdp and args.fsdp_no_sync and not is_sync_step and accum_steps > 1:
                sync_ctx = wrapper.no_sync()
            else:
                sync_ctx = nullcontext()

            with sync_ctx:
                metrics = _forward_backward_step(
                    args,
                    wrapper=wrapper,
                    input_ids=input_ids,
                    height=height,
                    width=width,
                    text_ids=text_ids,
                    text_attention_mask=text_attention_mask,
                    accum_steps=accum_steps,
                    vertical_only_active=state.vertical_only_active,
                )
            last_metrics = metrics

            if is_sync_step:
                _commit_optimizer_step(
                    args,
                    wrapper=wrapper,
                    state=state,
                    warmup_target=warmup_target,
                    progress=progress,
                    is_main=is_main,
                )
                stop_training = _run_post_optimizer_step(
                    args,
                    state,
                    metrics=metrics,
                    progress=progress,
                    world_size=world_size,
                    is_main=is_main,
                    epoch=epoch,
                    wrapper=wrapper,
                    tokenizer=tokenizer,
                    device=device,
                    visual_token_offset=visual_token_offset,
                    rank=rank,
                )
                if stop_training:
                    break

            step += 1

        if not stop_training and _finalize_partial_accumulation(
            args,
            wrapper=wrapper,
            state=state,
            warmup_target=warmup_target,
            last_step=last_step,
            accum_steps=accum_steps,
            progress=progress,
            is_main=is_main,
        ):
            if last_metrics is None:
                raise RuntimeError("partial accumulation flushed without any recorded step metrics")
            stop_training = _run_post_optimizer_step(
                args,
                state,
                metrics=last_metrics,
                progress=progress,
                world_size=world_size,
                is_main=is_main,
                epoch=epoch,
                wrapper=wrapper,
                tokenizer=tokenizer,
                device=device,
                visual_token_offset=visual_token_offset,
                rank=rank,
            )

        if progress is not None:
            progress.close()

        if stop_training:
            if is_main:
                print("[INFO] max_steps reached, skip epoch-end checkpoint/generation.")
            break

        if state.save_epoch_mode != "none":
            save_checkpoint(
                args,
                wrapper,
                rank,
                is_main,
                state.global_step,
                epoch=epoch,
                save_reason="epoch",
            )

        if args.eval_generate_timing in ("end", "both"):
            state.vq_model, state.vq_dtype = run_epoch_generate(
                args,
                wrapper,
                tokenizer,
                state.vq_model,
                state.vq_dtype,
                device,
                visual_token_offset,
                rank,
                is_main,
                epoch,
                run_tag=f"epoch{epoch}_end",
            )

    return state
