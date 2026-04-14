from __future__ import annotations

import glob
import json
import os.path as osp
import re
from typing import Any, Dict, Tuple

import torch
from safetensors.torch import load_file as safe_load_file


_FUSE_COMPAT_TOKENS = (
    "fuse_h_logit",
    "fuse_corner_h_logit",
    "hv_gate_mlp.",
    "hv_gate_corner.",
    "horizontal_head.bias",
    "vertical_head.bias",
)
_HF_MODEL_FILENAMES = ("model.safetensors", "pytorch_model.bin")
_HF_MODEL_INDEX_FILENAMES = ("model.safetensors.index.json", "pytorch_model.bin.index.json")
_HF_SHARD_FILENAME_RE = re.compile(r"^(?:model|pytorch_model)-\d{5}-of-\d{5}\.(?:safetensors|bin)$")


def _torch_load_cpu(path: str, *, mmap: bool = False):
    load_kwargs = {
        "map_location": "cpu",
        "weights_only": False,
    }
    if mmap:
        load_kwargs["mmap"] = True
    try:
        return torch.load(path, **load_kwargs)
    except TypeError:
        load_kwargs.pop("mmap", None)
        try:
            return torch.load(path, **load_kwargs)
        except TypeError:
            load_kwargs.pop("weights_only", None)
            return torch.load(path, **load_kwargs)


def _load_sharded_hf_state(index_path: str, *, mmap: bool = False):
    with open(index_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    weight_map = payload.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError(f"Invalid HF shard index at {index_path}: missing weight_map")
    base_dir = osp.dirname(index_path)
    state_dict = {}
    for shard_name in sorted(set(str(name) for name in weight_map.values())):
        shard_path = osp.join(base_dir, shard_name)
        if not osp.exists(shard_path):
            raise FileNotFoundError(f"Missing shard referenced by {index_path}: {shard_path}")
        if shard_name.endswith(".safetensors"):
            shard_state = safe_load_file(shard_path, device="cpu")
        else:
            shard_state = _torch_load_cpu(shard_path, mmap=mmap)
        state_dict.update(shard_state)
    return state_dict


def safe_torch_load(path: str, *, mmap: bool = False):
    if (not osp.exists(path)) and path.endswith(".pt") and osp.isdir(path[:-3]):
        path = path[:-3]
    if path.endswith(".index.json"):
        return _load_sharded_hf_state(path, mmap=mmap)
    if _HF_SHARD_FILENAME_RE.fullmatch(osp.basename(path)) is not None:
        for index_name in _HF_MODEL_INDEX_FILENAMES:
            index_path = osp.join(osp.dirname(path), index_name)
            if osp.exists(index_path):
                return _load_sharded_hf_state(index_path, mmap=mmap)
    if path.endswith(".safetensors"):
        return safe_load_file(path, device="cpu")
    if osp.isdir(path):
        safetensor_index_path = osp.join(path, "model.safetensors.index.json")
        if osp.exists(safetensor_index_path):
            return _load_sharded_hf_state(safetensor_index_path, mmap=mmap)
        bin_index_path = osp.join(path, "pytorch_model.bin.index.json")
        if osp.exists(bin_index_path):
            return _load_sharded_hf_state(bin_index_path, mmap=mmap)
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
    return _torch_load_cpu(path, mmap=mmap)


def strip_shard_suffix(path: str) -> str:
    match = re.match(r"^(.*)\.rank\d+\.pt$", path)
    return match.group(1) if match else path


def is_fuse_key(key: str) -> bool:
    return any(token in key for token in _FUSE_COMPAT_TOKENS)


def infer_vertical_from_state(state_dict: Dict[str, Any]) -> Tuple[bool, int]:
    pattern = re.compile(r"^(?:module\.)?vertical_block\.(\d+)\.")
    max_layer = -1
    has_vertical_norm = False
    for key in state_dict.keys():
        match = pattern.match(key)
        if match is not None:
            max_layer = max(max_layer, int(match.group(1)))
        if key.startswith("vertical_norm.") or key.startswith("module.vertical_norm."):
            has_vertical_norm = True
    if max_layer >= 0:
        return True, max_layer + 1
    if has_vertical_norm:
        return True, 1
    return False, 0


def load_nar_metadata(path: str) -> Dict[str, Any]:
    path = strip_shard_suffix(str(path))
    if (not osp.exists(path)) and path.endswith(".pt") and osp.isdir(path[:-3]):
        path = path[:-3]
    if not osp.isdir(path):
        return {}
    cfg_path = osp.join(path, "config.json")
    if not osp.exists(cfg_path):
        return {}
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def load_state_with_allowed_missing(
    module: torch.nn.Module,
    state_dict: Dict[str, Any],
    *,
    load_desc: str,
    allowed_missing_prefixes: Tuple[str, ...] = (),
) -> None:
    incompat = module.load_state_dict(state_dict, strict=False)
    missing_keys = list(getattr(incompat, "missing_keys", []))
    unexpected_keys = list(getattr(incompat, "unexpected_keys", []))
    bad_missing = [
        key
        for key in missing_keys
        if not key.startswith(allowed_missing_prefixes) and not is_fuse_key(key)
    ]
    bad_unexpected = [key for key in unexpected_keys if not is_fuse_key(key)]
    if bad_missing or bad_unexpected:
        raise RuntimeError(
            f"{load_desc}: incompatible state_dict. missing={bad_missing} unexpected={bad_unexpected}"
        )
    if missing_keys or unexpected_keys:
        print(
            f"[WARN] {load_desc}: partial load tolerated. "
            f"missing={missing_keys} unexpected={unexpected_keys}"
        )


def load_state_with_fuse_compat(
    module: torch.nn.Module,
    state_dict: Dict[str, Any],
    *,
    load_desc: str,
) -> None:
    try:
        module.load_state_dict(state_dict, strict=True)
        return
    except RuntimeError as exc:
        message = str(exc)
        if not any(token in message for token in _FUSE_COMPAT_TOKENS):
            raise
    incompat = module.load_state_dict(state_dict, strict=False)
    missing_keys = list(getattr(incompat, "missing_keys", []))
    unexpected_keys = list(getattr(incompat, "unexpected_keys", []))
    bad_missing = [key for key in missing_keys if not is_fuse_key(key)]
    bad_unexpected = [key for key in unexpected_keys if not is_fuse_key(key)]
    if bad_missing or bad_unexpected:
        raise RuntimeError(
            f"{load_desc}: incompatible state_dict. missing={bad_missing} unexpected={bad_unexpected}"
        )
    print(
        f"[WARN] {load_desc}: fuse parameters mismatch tolerated. "
        f"missing={missing_keys} unexpected={unexpected_keys}"
    )


def resolve_nar_ckpt_path(
    *,
    nar_ckpt_path: str,
    model_path: str,
    merge_dtype: str = "bf16",
    fsdp_wrap_policy: str = "transformer",
    fsdp_min_params: int = 1_000_000,
    use_vertical_block: bool | None = None,
    vertical_layers: int = 0,
) -> str:
    del model_path
    del merge_dtype
    del fsdp_wrap_policy
    del fsdp_min_params
    del use_vertical_block
    del vertical_layers
    if not nar_ckpt_path:
        raise ValueError("nar_ckpt_path is required for NAR inference loading.")
    nar_ckpt_path = strip_shard_suffix(nar_ckpt_path)

    if osp.exists(nar_ckpt_path):
        return nar_ckpt_path
    if nar_ckpt_path.endswith(".pt") and osp.isdir(nar_ckpt_path[:-3]):
        return nar_ckpt_path[:-3]

    if nar_ckpt_path.endswith(".pt"):
        full_candidate = nar_ckpt_path[:-3] + ".full.pt"
    else:
        full_candidate = nar_ckpt_path + ".full.pt"
    if osp.exists(full_candidate):
        return full_candidate

    shard_paths = sorted(glob.glob(nar_ckpt_path + ".rank*.pt"))
    if not shard_paths and nar_ckpt_path.endswith(".pt"):
        base = nar_ckpt_path[:-3]
        shard_paths = sorted(glob.glob(base + ".rank*.pt"))
        if shard_paths:
            nar_ckpt_path = base

    if not shard_paths:
        raise FileNotFoundError(f"NAR ckpt not found: {nar_ckpt_path}")
    raise FileNotFoundError(
        "Sharded NAR checkpoints are no longer supported in the streamlined mainline. "
        f"Found {len(shard_paths)} shard files for base path: {nar_ckpt_path}. "
        "Please provide a merged `.full.pt` checkpoint path instead."
    )
