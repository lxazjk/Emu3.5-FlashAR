from __future__ import annotations

import argparse
import json
import os
import os.path as osp
import re
import shutil
from typing import Any, Dict, Optional

import torch
import torch.distributed as dist
from huggingface_hub import save_torch_state_dict
from safetensors.torch import load_file as safe_load_file

from flashar.model import Emuflashar


HF_CHECKPOINT_DIRNAME = "flashar_final"
HF_MODEL_FILENAMES = ("model.safetensors", "pytorch_model.bin")
HF_MODEL_INDEX_FILENAMES = ("model.safetensors.index.json", "pytorch_model.bin.index.json")
HF_SHARD_FILENAME_RE = re.compile(r"^(?:model|pytorch_model)-\d{5}-of-\d{5}\.(?:safetensors|bin)$")
CHECKPOINT_META_FILENAME = "checkpoint_meta.json"


def get_checkpoint_save_path(args: argparse.Namespace) -> str:
    return os.path.join(args.save_dir, HF_CHECKPOINT_DIRNAME)


def normalize_checkpoint_path(path: str) -> str:
    path = str(path or "").strip()
    if not path:
        return ""
    if (not osp.exists(path)) and path.endswith(".pt") and osp.isdir(path[:-3]):
        return path[:-3]
    return path


def resolve_hf_checkpoint_dir(path: str) -> Optional[str]:
    path = normalize_checkpoint_path(path)
    if not path:
        return None
    if osp.isdir(path):
        return path
    basename = osp.basename(path)
    if (
        basename in HF_MODEL_FILENAMES
        or basename in HF_MODEL_INDEX_FILENAMES
        or HF_SHARD_FILENAME_RE.fullmatch(basename) is not None
    ):
        parent = osp.dirname(path)
        meta_path = osp.join(parent, CHECKPOINT_META_FILENAME)
        cfg_path = osp.join(parent, "config.json")
        has_hf_weights = any(
            osp.exists(osp.join(parent, name))
            for name in HF_MODEL_FILENAMES + HF_MODEL_INDEX_FILENAMES
        )
        if osp.exists(meta_path) or osp.exists(cfg_path) or has_hf_weights:
            return parent
    return None


def _checkpoint_meta_path(path: str) -> str:
    return osp.join(path, CHECKPOINT_META_FILENAME)


def _read_json_payload(json_path: str, *, is_main: bool) -> Optional[Dict[str, Any]]:
    if not osp.exists(json_path):
        return None
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        if is_main:
            print(f"[WARN] failed reading {json_path}: {exc}")
        return None


def _extract_step_from_payload(
    payload: Dict[str, Any],
    *,
    json_path: str,
    is_main: bool,
) -> Optional[int]:
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


def _write_hf_checkpoint_dir(
    path: str,
    *,
    state_dict: Dict[str, Any],
    target: Emuflashar,
    args: argparse.Namespace,
    global_step: int,
    epoch: Optional[int],
    save_reason: str,
) -> None:
    tmp_path = path + ".tmp"
    if osp.isdir(tmp_path):
        shutil.rmtree(tmp_path)
    elif osp.exists(tmp_path):
        os.remove(tmp_path)
    os.makedirs(tmp_path, exist_ok=True)

    hf_state_dict = {}
    for key, value in state_dict.items():
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"Expected tensor state for key={key}, got {type(value)!r}")
        hf_state_dict[key] = value.detach().cpu().contiguous()
    save_torch_state_dict(
        hf_state_dict,
        tmp_path,
        filename_pattern="model{suffix}.safetensors",
        max_shard_size=(getattr(args, "hf_max_shard_size", None) or "5GB"),
        safe_serialization=True,
        is_main_process=True,
    )

    backbone_config = {}
    if getattr(getattr(target, "backbone", None), "config", None) is not None:
        backbone_config = dict(target.backbone.config.to_dict())
    config_payload = {
        "model_type": "emu3_flashar",
        "architectures": ["Emuflashar"],
        "format": "hf_safetensors",
        "format_version": 1,
        "base_model_name_or_path": str(getattr(args, "model_path", "")),
        "tokenizer_name_or_path": str(getattr(args, "tokenizer_path", "")),
        "vq_model_name_or_path": str(getattr(args, "vq_path", "")),
        "torch_dtype": str(getattr(args, "dtype", "")),
        "vocab_size": int(getattr(target, "vocab_size", 0)),
        "hidden_size": int(getattr(target, "hidden_size", 0)),
        "visual_token_offset": getattr(target, "visual_token_offset", None),
        "use_vertical_block": bool(getattr(target, "use_vertical_block", False)),
        "vertical_layers": (
            len(target.vertical_block)
            if getattr(target, "vertical_block", None) is not None
            else 0
        ),
        "vertical_start_layer": int(getattr(target, "vertical_start_layer", -1)),
        "backbone_config": backbone_config,
    }
    with open(osp.join(tmp_path, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config_payload, f, ensure_ascii=False, indent=2)

    meta_payload = {
        "format": "hf_safetensors",
        "global_step": int(global_step),
        "epoch": None if epoch is None else int(epoch),
        "save_reason": str(save_reason),
    }
    with open(_checkpoint_meta_path(tmp_path), "w", encoding="utf-8") as f:
        json.dump(meta_payload, f, ensure_ascii=False, indent=2)

    if osp.isdir(path):
        shutil.rmtree(path)
    elif osp.exists(path):
        os.remove(path)
    os.replace(tmp_path, path)


def _torch_load_cpu(path: str) -> Dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _load_sharded_hf_state(index_path: str) -> Dict[str, Any]:
    with open(index_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    weight_map = payload.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError(f"Invalid HF shard index at {index_path}: missing weight_map")
    base_dir = osp.dirname(index_path)
    state_dict: Dict[str, Any] = {}
    for shard_name in sorted(set(str(name) for name in weight_map.values())):
        shard_path = osp.join(base_dir, shard_name)
        if not osp.exists(shard_path):
            raise FileNotFoundError(f"Missing shard referenced by {index_path}: {shard_path}")
        if shard_name.endswith(".safetensors"):
            shard_state = safe_load_file(shard_path, device="cpu")
        else:
            shard_state = _torch_load_cpu(shard_path)
        state_dict.update(shard_state)
    return state_dict


def safe_torch_load(path: str) -> Dict[str, Any]:
    path = normalize_checkpoint_path(path)
    if path.endswith(".index.json"):
        return _load_sharded_hf_state(path)
    if HF_SHARD_FILENAME_RE.fullmatch(osp.basename(path)) is not None:
        for index_name in HF_MODEL_INDEX_FILENAMES:
            index_path = osp.join(osp.dirname(path), index_name)
            if osp.exists(index_path):
                return _load_sharded_hf_state(index_path)
    if path.endswith(".safetensors"):
        return safe_load_file(path, device="cpu")
    if osp.isdir(path):
        safetensor_index_path = osp.join(path, "model.safetensors.index.json")
        if osp.exists(safetensor_index_path):
            return _load_sharded_hf_state(safetensor_index_path)
        bin_index_path = osp.join(path, "pytorch_model.bin.index.json")
        if osp.exists(bin_index_path):
            return _load_sharded_hf_state(bin_index_path)
        safetensor_path = osp.join(path, "model.safetensors")
        if osp.exists(safetensor_path):
            return safe_load_file(safetensor_path, device="cpu")
        bin_path = osp.join(path, "pytorch_model.bin")
        if osp.exists(bin_path):
            path = bin_path
        else:
            raise FileNotFoundError(
                f"Unsupported HF checkpoint directory: {path}. "
                "Expected model.safetensors, model.safetensors.index.json, "
                "or pytorch_model.bin."
            )
    return _torch_load_cpu(path)


def infer_resume_step(args: argparse.Namespace, is_main: bool) -> int:
    resume_path = normalize_checkpoint_path(args.resume_path)
    if not resume_path:
        return 0

    hf_dir = resolve_hf_checkpoint_dir(resume_path)
    if hf_dir is not None:
        payload = _read_json_payload(_checkpoint_meta_path(hf_dir), is_main=is_main)
        if payload is not None:
            step = _extract_step_from_payload(
                payload,
                json_path=_checkpoint_meta_path(hf_dir),
                is_main=is_main,
            )
            if step is not None:
                return step

    resume_dir = osp.dirname(resume_path)
    resume_name = osp.basename(resume_path)
    step_json = osp.join(resume_dir, "flashar_step_latest.json")
    epoch_json = osp.join(resume_dir, "flashar_epoch_latest.json")

    step_match = re.search(r"flashar_step(\d+)", resume_name)
    if step_match:
        step = int(step_match.group(1))
        if is_main:
            print(f"[INFO] inferred resume step from checkpoint name: {step}")
        return step

    if "flashar_step_latest" in resume_name:
        payload = _read_json_payload(step_json, is_main=is_main)
        step = (
            _extract_step_from_payload(payload, json_path=step_json, is_main=is_main)
            if payload is not None
            else None
        )
        if step is not None:
            return step

    if "flashar_epoch" in resume_name:
        epoch_payload = _read_json_payload(epoch_json, is_main=is_main)
        step = (
            _extract_step_from_payload(epoch_payload, json_path=epoch_json, is_main=is_main)
            if epoch_payload is not None
            else None
        )
        if step is not None:
            return step
        step_payload = _read_json_payload(step_json, is_main=is_main)
        step = (
            _extract_step_from_payload(step_payload, json_path=step_json, is_main=is_main)
            if step_payload is not None
            else None
        )
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
    if osp.isdir(path):
        return "full"
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
    resume_path = normalize_checkpoint_path(resume_path)
    if not resume_path:
        return ""
    if osp.isdir(resume_path):
        return resume_path
    if resume_path.endswith(f".rank{rank}.pt"):
        return resume_path
    candidate = resume_path + f".rank{rank}.pt"
    if osp.exists(candidate):
        return candidate
    return resume_path


def should_preload_single_file_resume(args: argparse.Namespace, rank: int) -> bool:
    if not getattr(args, "fsdp", False):
        return False
    resume_path = normalize_checkpoint_path(getattr(args, "resume_path", "") or "")
    if not resume_path or not osp.exists(resume_path):
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
        args.resume_path = normalize_checkpoint_path(args.resume_path)
        if is_main:
            print("[INFO] resume from:", args.resume_path)
        return

    candidates = [
        get_checkpoint_save_path(args),
        os.path.join(args.save_dir, "flashar_final.pt"),
        os.path.join(args.save_dir, "flashar_step_latest.pt"),
        os.path.join(args.save_dir, "flashar_epoch_latest.full.pt"),
    ]
    for candidate in candidates:
        candidate = normalize_checkpoint_path(candidate)
        if os.path.exists(candidate):
            args.resume_path = candidate
            if is_main:
                print("[INFO] resume from:", args.resume_path)
            return

    epoch_candidates = sorted(
        [
            os.path.join(args.save_dir, name)
            for name in os.listdir(args.save_dir)
            if name.startswith("flashar_epoch") and name.endswith(".full.pt")
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
    wrapper: Emuflashar,
    rank: int,
    is_main: bool,
) -> bool:
    args.resume_path = normalize_checkpoint_path(args.resume_path)
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
    wrapper: Emuflashar,
    rank: int,
    is_main: bool,
) -> None:
    args.resume_path = normalize_checkpoint_path(args.resume_path)
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


def save_checkpoint(
    args: argparse.Namespace,
    wrapper: Emuflashar,
    rank: int,
    is_main: bool,
    global_step: int,
    epoch: Optional[int] = None,
    save_reason: str = "manual",
) -> None:
    save_path = get_checkpoint_save_path(args)
    target = wrapper.module if hasattr(wrapper, "module") else wrapper

    if args.fsdp:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp import FullStateDictConfig, StateDictType

        state_cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(wrapper, StateDictType.FULL_STATE_DICT, state_cfg):
            state_dict = wrapper.state_dict()
        if rank == 0:
            _write_hf_checkpoint_dir(
                save_path,
                state_dict=state_dict,
                target=target,
                args=args,
                global_step=global_step,
                epoch=epoch,
                save_reason=save_reason,
            )
            print(
                f"[INFO] saved HF checkpoint: {save_path} "
                f"(reason={save_reason} step={int(global_step)})"
            )
        dist.barrier()
        return

    if not is_main:
        return
    state_dict = wrapper.state_dict()
    _write_hf_checkpoint_dir(
        save_path,
        state_dict=state_dict,
        target=target,
        args=args,
        global_step=global_step,
        epoch=epoch,
        save_reason=save_reason,
    )
    print(
        f"[INFO] saved HF checkpoint: {save_path} "
        f"(reason={save_reason} step={int(global_step)})"
    )
