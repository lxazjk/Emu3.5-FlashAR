from __future__ import annotations

import argparse
import json
import os
import os.path as osp
import re
from typing import Any, Dict, Optional

import torch
import torch.distributed as dist

from emu_nar.model import EmuNAR


def safe_torch_load(path: str) -> Dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def infer_resume_step(args: argparse.Namespace, is_main: bool) -> int:
    resume_path = str(args.resume_path or "").strip()
    if not resume_path:
        return 0

    def _read_json_step(json_path: str) -> Optional[int]:
        if not osp.exists(json_path):
            return None
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception as exc:
            if is_main:
                print(f"[WARN] failed reading {json_path}: {exc}")
            return None
        for key in ("step", "global_step"):
            if key in payload:
                try:
                    step = int(payload[key])
                except Exception as exc:
                    if is_main:
                        print(f"[WARN] invalid {key} in {json_path}: {exc}")
                    return None
                if is_main:
                    print(f"[INFO] inferred resume step from {json_path} ({key}): {step}")
                return max(0, step)
        if is_main:
            print(
                f"[WARN] {json_path} has no step/global_step field; "
                f"available keys: {sorted(payload.keys())}"
            )
        return None

    resume_dir = osp.dirname(resume_path)
    resume_name = osp.basename(resume_path)
    step_json = osp.join(resume_dir, "nar_step_latest.json")
    epoch_json = osp.join(resume_dir, "nar_epoch_latest.json")

    step_match = re.search(r"nar_step(\d+)", resume_name)
    if step_match:
        step = int(step_match.group(1))
        if is_main:
            print(f"[INFO] inferred resume step from checkpoint name: {step}")
        return step

    if "nar_step_latest" in resume_name:
        step = _read_json_step(step_json)
        if step is not None:
            return step

    if "nar_epoch" in resume_name:
        step = _read_json_step(epoch_json)
        if step is not None:
            return step
        step = _read_json_step(step_json)
        if step is not None:
            if is_main:
                print(
                    f"[WARN] {epoch_json} has no usable step; "
                    f"fallback to {step_json} step={step}."
                )
            return step

    if is_main:
        print(
            f"[WARN] cannot infer resume step from {resume_path}; "
            "fallback to step=0 (warmup/distill may restart)."
        )
    return 0


def detect_fsdp_state_type(path: str, state_dict: Dict[str, Any]) -> str:
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


def resolve_rank_resume_path(resume_path: str, rank: int) -> str:
    resume_path = str(resume_path or "").strip()
    if not resume_path:
        return ""
    if resume_path.endswith(f".rank{rank}.pt"):
        return resume_path
    candidate = resume_path + f".rank{rank}.pt"
    if osp.exists(candidate):
        return candidate
    return resume_path


def should_preload_single_file_resume(args: argparse.Namespace, rank: int) -> bool:
    if not getattr(args, "fsdp", False):
        return False
    resume_path = str(getattr(args, "resume_path", "") or "").strip()
    if not resume_path or not osp.isfile(resume_path):
        return False
    return resolve_rank_resume_path(resume_path, rank) == resume_path


def prepare_output_dir(args: argparse.Namespace, is_main: bool) -> None:
    if args.fsdp or is_main:
        os.makedirs(args.save_dir, exist_ok=True)


def resolve_resume_path(args: argparse.Namespace, is_main: bool) -> None:
    if getattr(args, "fresh_run", False):
        args.resume_path = ""
        if is_main:
            print("[INFO] fresh run requested; start from base weights")
        return

    if args.resume_path:
        if is_main:
            print("[INFO] resume from:", args.resume_path)
        return

    candidates = [
        os.path.join(args.save_dir, "nar_step_latest.pt"),
        os.path.join(args.save_dir, "nar_epoch_latest.full.pt"),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            args.resume_path = candidate
            if is_main:
                print("[INFO] resume from:", args.resume_path)
            return

    epoch_candidates = sorted(
        [
            os.path.join(args.save_dir, name)
            for name in os.listdir(args.save_dir)
            if name.startswith("nar_epoch") and name.endswith(".full.pt")
        ],
        reverse=True,
    ) if os.path.isdir(args.save_dir) else []
    if epoch_candidates:
        args.resume_path = epoch_candidates[0]
        if is_main:
            print("[INFO] resume from:", args.resume_path)
        return

    if is_main:
        print("[WARN] no resume checkpoint found; start from base weights")


def preload_single_file_resume_if_needed(
    args: argparse.Namespace,
    wrapper: EmuNAR,
    rank: int,
    is_main: bool,
) -> bool:
    if not should_preload_single_file_resume(args, rank):
        return False

    if is_main:
        print(
            "[INFO] single-file FSDP resume detected; "
            "pre-loading checkpoint on rank0 before FSDP wrap:",
            args.resume_path,
        )

    if rank == 0:
        state_dict = safe_torch_load(args.resume_path)
        state_type = detect_fsdp_state_type(args.resume_path, state_dict)
        if state_type != "full":
            raise RuntimeError(
                "single-file FSDP resume expected a full state dict, "
                f"but detected state_type={state_type} for {args.resume_path}"
            )
        load_state_with_fuse_compat(
            wrapper,
            state_dict,
            strict=True,
            load_desc="pre-load full ckpt before FSDP wrap",
        )
        print("[INFO] pre-loaded full ckpt on rank0:", args.resume_path)
        del state_dict

    if dist.is_initialized():
        dist.barrier()
    return True


def is_allowed_resume_missing(key: str) -> bool:
    compat_tokens = (
        "fuse_h_logit",
        "fuse_corner_h_logit",
        "hv_gate_mlp.",
        "hv_gate_corner.",
        "horizontal_head.bias",
        "vertical_head.bias",
    )
    return any(tok in key for tok in compat_tokens)


def load_state_with_fuse_compat(
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
        if (
            "fuse_h_logit" not in msg
            and "fuse_corner_h_logit" not in msg
            and "hv_gate_mlp" not in msg
            and "hv_gate_corner" not in msg
            and "horizontal_head.bias" not in msg
            and "vertical_head.bias" not in msg
        ):
            raise
    incompat = module.load_state_dict(state_dict, strict=False)
    missing_keys = list(getattr(incompat, "missing_keys", []))
    unexpected_keys = list(getattr(incompat, "unexpected_keys", []))
    bad_missing = [k for k in missing_keys if not is_allowed_resume_missing(k)]
    bad_unexpected = [k for k in unexpected_keys if not is_allowed_resume_missing(k)]
    if bad_missing or bad_unexpected:
        raise RuntimeError(
            f"{load_desc}: incompatible state_dict. "
            f"missing={bad_missing} unexpected={bad_unexpected}"
        )
    if (not dist.is_initialized()) or dist.get_rank() == 0:
        print(
            f"[WARN] {load_desc}: tolerated compat param mismatch. "
            f"missing={missing_keys} unexpected={unexpected_keys}"
        )


def resume_if_needed(
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
        rank_path = resolve_rank_resume_path(resume_path, rank)
        if not os.path.exists(rank_path):
            raise FileNotFoundError(f"Missing sharded checkpoint for rank {rank}: {rank_path}")
        state_dict = safe_torch_load(rank_path)
        state_type = detect_fsdp_state_type(rank_path, state_dict)
        if state_type == "sharded":
            state_cfg = ShardedStateDictConfig(offload_to_cpu=True)
            with FSDP.state_dict_type(wrapper, StateDictType.SHARDED_STATE_DICT, state_cfg):
                load_state_with_fuse_compat(
                    wrapper, state_dict, strict=True, load_desc="load sharded ckpt"
                )
            if rank == 0:
                print("[INFO] loaded sharded ckpt:", resume_path)
        elif state_type == "local":
            state_cfg = LocalStateDictConfig(offload_to_cpu=True)
            with FSDP.state_dict_type(wrapper, StateDictType.LOCAL_STATE_DICT, state_cfg):
                load_state_with_fuse_compat(
                    wrapper, state_dict, strict=True, load_desc="load local ckpt"
                )
            if rank == 0:
                print("[INFO] loaded local ckpt:", resume_path)
        else:
            state_cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=False)
            with FSDP.state_dict_type(wrapper, StateDictType.FULL_STATE_DICT, state_cfg):
                load_state_with_fuse_compat(
                    wrapper, state_dict, strict=True, load_desc="load full ckpt"
                )
            if rank == 0:
                print("[INFO] loaded full ckpt:", resume_path)
        dist.barrier()
    else:
        state_dict = safe_torch_load(args.resume_path)
        load_state_with_fuse_compat(wrapper, state_dict, strict=True, load_desc="load ckpt")
        if is_main:
            print("[INFO] loaded ckpt:", args.resume_path)


def save_step_checkpoint(
    args: argparse.Namespace,
    wrapper: EmuNAR,
    rank: int,
    is_main: bool,
    global_step: int,
) -> None:
    if args.save_latest_only:
        step_path = os.path.join(args.save_dir, "nar_step_latest.pt")
    else:
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
            if args.save_latest_only:
                with open(
                    os.path.join(args.save_dir, "nar_step_latest.json"),
                    "w",
                    encoding="utf-8",
                ) as f:
                    json.dump({"step": int(global_step)}, f)
        dist.barrier()
    else:
        if is_main:
            torch.save(wrapper.state_dict(), step_path)
            print("[INFO] saved step:", step_path)
            if args.save_latest_only:
                with open(
                    os.path.join(args.save_dir, "nar_step_latest.json"),
                    "w",
                    encoding="utf-8",
                ) as f:
                    json.dump({"step": int(global_step)}, f)


def save_epoch_checkpoint(
    args: argparse.Namespace,
    wrapper: EmuNAR,
    rank: int,
    is_main: bool,
    epoch: int,
    global_step: int,
    save_epoch_mode: str,
) -> None:
    if save_epoch_mode == "none":
        return
    if args.save_latest_only:
        save_path = os.path.join(args.save_dir, "nar_epoch_latest.pt")
        full_path = os.path.join(args.save_dir, "nar_epoch_latest.full.pt")
    else:
        save_path = os.path.join(args.save_dir, f"nar_epoch{epoch}.pt")
        full_path = os.path.join(args.save_dir, f"nar_epoch{epoch}.full.pt")
    if args.fsdp:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp import FullStateDictConfig, ShardedStateDictConfig, StateDictType

        if save_epoch_mode == "full":
            state_cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            with FSDP.state_dict_type(wrapper, StateDictType.FULL_STATE_DICT, state_cfg):
                state_dict = wrapper.state_dict()
            if rank == 0:
                torch.save(state_dict, full_path)
                print("[INFO] saved full:", full_path)
                if args.save_latest_only:
                    with open(
                        os.path.join(args.save_dir, "nar_epoch_latest.json"),
                        "w",
                        encoding="utf-8",
                    ) as f:
                        json.dump({"epoch": int(epoch), "global_step": int(global_step)}, f)
            dist.barrier()
        else:
            state_cfg = ShardedStateDictConfig(offload_to_cpu=True)
            with FSDP.state_dict_type(wrapper, StateDictType.SHARDED_STATE_DICT, state_cfg):
                state_dict = wrapper.state_dict()
            torch.save(state_dict, save_path + f".rank{rank}.pt")
            if is_main:
                print("[INFO] saved sharded:", save_path)
                if args.save_latest_only:
                    with open(
                        os.path.join(args.save_dir, "nar_epoch_latest.json"),
                        "w",
                        encoding="utf-8",
                    ) as f:
                        json.dump({"epoch": int(epoch), "global_step": int(global_step)}, f)
            dist.barrier()
    else:
        if is_main:
            torch.save(wrapper.state_dict(), save_path)
            print("[INFO] saved:", save_path)
            if args.save_latest_only:
                with open(
                    os.path.join(args.save_dir, "nar_epoch_latest.json"),
                    "w",
                    encoding="utf-8",
                ) as f:
                    json.dump({"epoch": int(epoch), "global_step": int(global_step)}, f)


def save_final_checkpoint(args: argparse.Namespace, wrapper: EmuNAR, rank: int) -> None:
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
