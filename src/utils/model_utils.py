# -*- coding: utf-8 -*-
# Copyright 2025 BAAI. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import json
import os.path as osp
from typing import Any

import torch

from flashar.model import Emuflashar
from flashar.utils.text_utils import build_text_tokenizer as _build_text_tokenizer

from ..emu3p5 import Emu3Config, Emu3ForCausalLM
from ..emu3p5.modeling_emu3 import Emu3Model
from ..vision_tokenizer import build_vision_tokenizer
from .flashar_checkpoint_utils import (
    infer_vertical_from_state as _infer_vertical_from_state,
    load_flashar_metadata as _load_flashar_metadata,
    load_state_with_allowed_missing as _load_state_with_allowed_missing,
    resolve_flashar_ckpt_path as _resolve_flashar_ckpt_path,
    safe_torch_load as _safe_torch_load,
)


def _validate_model_path(model_path):
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
            "Please set model_path to Emu3.5/Emu3.5-Image language model weights."
        )

def build_emu3p5(
    model_path,
    tokenizer_path,
    vq_path,
    vq_type="ibq",
    model_device="auto",
    vq_device="cuda:0",
    **kwargs,
):
    _validate_model_path(model_path)
    if isinstance(model_device, int):
        device_map = f"cuda:{model_device}"
    else:
        device_map = model_device

    print(device_map)

    # MLLM
    model_config = Emu3Config.from_pretrained(
        model_path,
        trust_remote_code=True,
    )
    flashar_ckpt_path = kwargs.pop("flashar_ckpt_path", "")
    flashar_enabled = bool(flashar_ckpt_path)
    attn_impl = kwargs.pop("attn_implementation", None)
    if attn_impl is None:
        attn_impl = "eager" if flashar_enabled else "flash_attention_2"
    model, loading_info = Emu3ForCausalLM.from_pretrained(
        model_path,
        config=model_config,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
        attn_implementation=attn_impl,
        output_loading_info=True,
        # attn_implementation=\"eager\", # if you cann't install flash_attention
    )
    missing = list(loading_info.get("missing_keys", []))
    unexpected = list(loading_info.get("unexpected_keys", []))
    if len(unexpected) > 128 and len(missing) > 128:
        sample_unexpected = ", ".join(unexpected[:5])
        raise ValueError(
            "Loaded checkpoint appears incompatible with Emu3ForCausalLM "
            f"(missing_keys={len(missing)}, unexpected_keys={len(unexpected)}). "
            f"Sample unexpected keys: {sample_unexpected}."
        )
    model.eval()
    
    # text tokenizer
    tokenizer = _build_text_tokenizer(tokenizer_path)

    # vq tokenizer
    vq_model = build_vision_tokenizer(vq_type, vq_path, device=vq_device, **kwargs)

    return model, tokenizer, vq_model


def build_emu3p5_flashar_inference(
    model_path,
    tokenizer_path,
    vq_path,
    flashar_ckpt_path,
    vq_type="ibq",
    model_device="cuda:0",
    vq_device="cuda:0",
    flashar_use_vertical_block=None,
    flashar_vertical_layers=0,
    flashar_vertical_start_layer=-1,
    flashar_attn_implementation="eager",
    flashar_merge_dtype="bf16",
    flashar_fsdp_wrap_policy="transformer",
    flashar_fsdp_min_params=1_000_000,
    flashar_split_backbone=False,
    **kwargs,
):
    _validate_model_path(model_path)
    if isinstance(model_device, int):
        device_map = f"cuda:{model_device}"
    else:
        device_map = model_device

    print(device_map)
    print("[flashar-LOAD] resolving checkpoint path", flush=True)
    resolved_ckpt_path = _resolve_flashar_ckpt_path(
        flashar_ckpt_path=flashar_ckpt_path,
        model_path=model_path,
        merge_dtype=flashar_merge_dtype,
        fsdp_wrap_policy=flashar_fsdp_wrap_policy,
        fsdp_min_params=flashar_fsdp_min_params,
        use_vertical_block=flashar_use_vertical_block,
        vertical_layers=flashar_vertical_layers,
    )
    print(f"[flashar-LOAD] loading checkpoint from {resolved_ckpt_path}", flush=True)
    state = _safe_torch_load(resolved_ckpt_path, mmap=True)
    flashar_metadata = _load_flashar_metadata(resolved_ckpt_path)
    print("[flashar-LOAD] checkpoint deserialized on CPU", flush=True)

    inferred_use_vertical, inferred_vertical_layers = _infer_vertical_from_state(state)
    if flashar_use_vertical_block is None:
        use_vertical_block = inferred_use_vertical
    else:
        use_vertical_block = bool(flashar_use_vertical_block)
    if int(flashar_vertical_layers) > 0:
        vertical_layers = int(flashar_vertical_layers)
    elif inferred_vertical_layers > 0:
        vertical_layers = inferred_vertical_layers
    else:
        vertical_layers = 1 if use_vertical_block else 0
    backbone_state = {}
    flashar_head_state = {}
    for key, value in state.items():
        if key.startswith("backbone."):
            backbone_state[key[len("backbone."):]] = value
        else:
            flashar_head_state[key] = value
    print(
        f"[flashar-LOAD] split state dict: backbone={len(backbone_state)} flashar={len(flashar_head_state)}",
        flush=True,
    )

    if flashar_attn_implementation == "flash_attention_2":
        print(
            "[flashar-LOAD] flash_attention_2 is temporarily disabled for flashar inference; falling back to eager.",
            flush=True,
        )
        flashar_attn_implementation = "eager"

    model_config = Emu3Config.from_pretrained(
        model_path,
        trust_remote_code=True,
    )
    if int(flashar_vertical_start_layer) >= 0:
        vertical_start_layer = int(flashar_vertical_start_layer)
    elif "vertical_start_layer" in flashar_metadata:
        vertical_start_layer = int(flashar_metadata["vertical_start_layer"])
    else:
        vertical_start_layer = int(model_config.num_hidden_layers)
    model_config._attn_implementation = flashar_attn_implementation
    # Avoid Transformers' flex-attention architecture allowlist check by
    # constructing the model directly and loading the already-extracted
    # backbone state from the dedicated flashar inference checkpoint.
    if device_map == "auto":
        target_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        target_device = torch.device(device_map)

    print(
        f"[flashar-LOAD] instantiating backbone on {target_device} with dtype={torch.bfloat16}",
        flush=True,
    )
    backbone = Emu3Model(model_config)
    backbone = backbone.to(device=target_device, dtype=torch.bfloat16)
    print("[flashar-LOAD] loading backbone weights", flush=True)
    incompat = backbone.load_state_dict(backbone_state, strict=True)
    missing = list(getattr(incompat, "missing_keys", []))
    unexpected = list(getattr(incompat, "unexpected_keys", []))
    if missing or unexpected:
        raise RuntimeError(
            "flashar backbone load got unexpected mismatch. "
            f"missing_keys={missing} unexpected_keys={unexpected}"
        )
    backbone.eval()
    print("[flashar-LOAD] backbone ready", flush=True)

    print("[flashar-LOAD] building flashar wrapper", flush=True)
    wrapper = Emuflashar(
        pretrained_backbone=backbone,
        vocab_size=model_config.vocab_size,
        hidden_size=model_config.hidden_size,
        pad_token_id=-100,
        mask_token_id=model_config.pad_token_id,
        visual_token_offset=int(model_config.eoi_token_id) + 1,
        use_vertical_block=use_vertical_block,
        vertical_layers=max(1, vertical_layers) if use_vertical_block else 0,
        vertical_start_layer=vertical_start_layer,
        lm_head=None,
        split_backbone=bool(flashar_split_backbone),
    )
    print("[flashar-LOAD] loading flashar head weights", flush=True)
    _load_state_with_allowed_missing(
        wrapper,
        flashar_head_state,
        load_desc="load flashar head state",
        allowed_missing_prefixes=("backbone.",),
    )
    wrapper = wrapper.to(next(backbone.parameters()).device)
    wrapper.eval()
    print("[flashar-LOAD] wrapper ready", flush=True)

    tokenizer = _build_text_tokenizer(tokenizer_path)
    vq_model = build_vision_tokenizer(vq_type, vq_path, device=vq_device, **kwargs)
    return wrapper, tokenizer, vq_model, resolved_ckpt_path
