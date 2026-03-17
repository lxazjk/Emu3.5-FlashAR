#!/usr/bin/env python3

import argparse
import json
import os
import os.path as osp
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import torch
from transformers import GenerationConfig
from transformers.generation import LogitsProcessorList

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.benchmark_utils import (
    build_prompt_tensor,
    build_special_token_ids,
    build_unconditional_tensor,
    cleanup_cuda,
    decode_ar_tokens_to_image,
    decode_nar_grid_to_image,
    load_benchmark_cfg,
    prepare_benchmark_cfg,
    save_image,
    sync_cuda,
)
from src.utils.generation_utils import build_logits_processor
from src.utils.model_utils import build_emu3p5, build_emu3p5_nar_inference


def _log_stage(message: str) -> None:
    stamp = datetime.now().strftime("%H:%M:%S")
    print(f"[STAGE {stamp}] {message}", flush=True)


def _run_ar_t2i(
    *,
    cfg: Any,
    model: Any,
    tokenizer: Any,
    vq_model: Any,
    prompt_text: str,
    out_dir: str,
) -> Dict[str, Any]:
    device = next(model.parameters()).device
    _log_stage("ar: building single-image prompt tensors")
    input_ids = build_prompt_tensor(
        cfg, tokenizer, prompt_text, add_image_prefix=True, device=device
    )
    unconditional_ids = build_unconditional_tensor(
        cfg, tokenizer, add_image_prefix=True, device=device
    )

    logits_processor = build_logits_processor(
        cfg,
        unconditional_ids,
        model,
        tokenizer,
        full_unconditional_ids=None,
        force_same_image_size=True,
    )
    # Force pure single-image T2I mode: start directly from `... BOI H*W IMG`
    # and constrain the remaining decoding to visual/EOL/EOI tokens only.
    logits_processor.in_image = True
    logits_processor.in_visual = True
    logits_processor.first_in_image = False
    logits_processor.image_nums = 1

    image_token_budget = int(cfg.target_height) * int(cfg.target_width) + int(cfg.target_height) - 1 + 1
    _log_stage(
        f"ar: prompt_len={input_ids.shape[1]} image_budget={image_token_budget} size={cfg.target_height}x{cfg.target_width}"
    )
    generation_kwargs = dict(cfg.sampling_params)
    generation_kwargs["max_new_tokens"] = image_token_budget
    generation_config = GenerationConfig(
        **generation_kwargs,
        pad_token_id=cfg.special_token_ids["PAD"],
        eos_token_id=cfg.special_token_ids["EOS"],
    )

    _log_stage("ar: starting generate()")
    infer_start = time.perf_counter()
    token_ids = model.generate(
        input_ids,
        generation_config,
        logits_processor=LogitsProcessorList([logits_processor]),
    )
    sync_cuda()
    infer_seconds = time.perf_counter() - infer_start
    _log_stage(f"ar: generate() finished in {infer_seconds:.2f}s")

    _log_stage("ar: decoding final image")
    decode_start = time.perf_counter()
    image = decode_ar_tokens_to_image(
        tokenizer=tokenizer,
        vq_model=vq_model,
        token_ids=token_ids,
    )
    decode_seconds = time.perf_counter() - decode_start
    _log_stage(f"ar: image decode finished in {decode_seconds:.2f}s")

    image_path = osp.join(out_dir, "final.png")
    save_image(image, image_path)
    return {
        "infer_seconds": infer_seconds,
        "decode_seconds": decode_seconds,
        "final_image": osp.abspath(image_path),
    }


def _run_nar_t2i(
    *,
    cfg: Any,
    nar_wrapper: Any,
    tokenizer: Any,
    vq_model: Any,
    prompt_text: str,
    out_dir: str,
) -> Dict[str, Any]:
    device = next(nar_wrapper.parameters()).device
    _log_stage("nar: building single-image prompt tensors")
    prompt_ids = build_prompt_tensor(
        cfg, tokenizer, prompt_text, add_image_prefix=True, device=device
    )
    uncond_prompt_ids = None
    if float(cfg.classifier_free_guidance) > 1.0:
        uncond_prompt_ids = build_unconditional_tensor(
            cfg, tokenizer, add_image_prefix=True, device=device
        )

    visual_token_offset = int(nar_wrapper.visual_token_offset)
    diag_steps = int(cfg.target_height) + int(cfg.target_width) - 1
    _log_stage(
        f"nar: prompt_len={prompt_ids.shape[1]} diag_steps={diag_steps} size={cfg.target_height}x{cfg.target_width} cfg={cfg.classifier_free_guidance}"
    )

    _log_stage("nar: starting wrapper.generate()")
    infer_start = time.perf_counter()
    grid = nar_wrapper.generate(
        height=int(cfg.target_height),
        width=int(cfg.target_width),
        device=device,
        text_input_ids=prompt_ids,
        unconditional_text_input_ids=uncond_prompt_ids,
        cfg_scale=float(cfg.classifier_free_guidance),
        temperature=float(cfg.sampling_params.get("image_temperature", 1.0)),
        top_k=int(cfg.sampling_params.get("image_top_k", 0)),
        top_p=float(cfg.sampling_params.get("image_top_p", 1.0)),
        sample_logits=bool(cfg.sampling_params.get("do_sample", True)),
        use_kv_cache=bool(getattr(cfg, "nar_use_kv_cache", True)),
    )
    sync_cuda()
    infer_seconds = time.perf_counter() - infer_start
    _log_stage(f"nar: wrapper.generate() finished in {infer_seconds:.2f}s")

    _log_stage("nar: decoding final image")
    decode_start = time.perf_counter()
    image = decode_nar_grid_to_image(
        vq_model=vq_model,
        grid=grid,
        visual_token_offset=visual_token_offset,
        height=int(cfg.target_height),
        width=int(cfg.target_width),
    )
    decode_seconds = time.perf_counter() - decode_start
    _log_stage(f"nar: image decode finished in {decode_seconds:.2f}s")

    image_path = osp.join(out_dir, "final.png")
    save_image(image, image_path)
    return {
        "infer_seconds": infer_seconds,
        "decode_seconds": decode_seconds,
        "final_image": osp.abspath(image_path),
    }

def _run_single_mode(mode_name: str, cfg_path: str, out_root: str) -> Dict[str, Any]:
    save_path = osp.join(out_root, mode_name)
    os.makedirs(save_path, exist_ok=True)

    old_save_path_env = os.environ.get("EMU35_BENCH_SAVE_PATH")
    os.environ["EMU35_BENCH_SAVE_PATH"] = save_path
    try:
        cfg = prepare_benchmark_cfg(load_benchmark_cfg(cfg_path), save_path)
    finally:
        if old_save_path_env is None:
            os.environ.pop("EMU35_BENCH_SAVE_PATH", None)
        else:
            os.environ["EMU35_BENCH_SAVE_PATH"] = old_save_path_env
    _log_stage(f"{mode_name}: loading model/tokenizer/vq")
    load_start = time.perf_counter()
    resolved_nar_ckpt_path = ""
    load_mode = "base_model"
    if mode_name == "nar":
        load_mode = "nar_direct_inference_ckpt"
        nar_wrapper, tokenizer, vq_model, resolved_nar_ckpt_path = build_emu3p5_nar_inference(
            cfg.model_path,
            cfg.tokenizer_path,
            cfg.vq_path,
            nar_ckpt_path=getattr(cfg, "nar_ckpt_path", ""),
            vq_type=cfg.vq_type,
            model_device=cfg.hf_device,
            vq_device=cfg.vq_device,
            nar_use_vertical_block=getattr(cfg, "nar_use_vertical_block", None),
            nar_vertical_layers=int(getattr(cfg, "nar_vertical_layers", 0)),
            nar_attn_implementation=str(getattr(cfg, "nar_attn_implementation", "eager")),
            nar_merge_dtype=str(getattr(cfg, "nar_merge_dtype", "bf16")),
            nar_fsdp_wrap_policy=str(getattr(cfg, "nar_fsdp_wrap_policy", "transformer")),
            nar_fsdp_min_params=int(getattr(cfg, "nar_fsdp_min_params", 1_000_000)),
            **getattr(cfg, "diffusion_decoder_kwargs", {}),
        )
        model = None
    else:
        model, tokenizer, vq_model = build_emu3p5(
            cfg.model_path,
            cfg.tokenizer_path,
            cfg.vq_path,
            vq_type=cfg.vq_type,
            model_device=cfg.hf_device,
            vq_device=cfg.vq_device,
            nar_ckpt_path="",
            **getattr(cfg, "diffusion_decoder_kwargs", {}),
        )
        nar_wrapper = None
    sync_cuda()
    load_seconds = time.perf_counter() - load_start
    _log_stage(f"{mode_name}: load finished in {load_seconds:.2f}s")

    cfg.special_token_ids = build_special_token_ids(cfg, tokenizer)
    prompt_text = cfg.prompts[0][1]
    if not isinstance(prompt_text, str):
        raise ValueError("benchmark_ar_vs_nar.py only supports pure t2i string prompts.")
    if getattr(cfg, "use_image", False):
        raise ValueError("benchmark_ar_vs_nar.py only supports use_image=False.")
    if int(getattr(cfg, "target_height", 0) or 0) <= 0 or int(getattr(cfg, "target_width", 0) or 0) <= 0:
        raise ValueError("target_height/target_width must be set for single-image t2i benchmark.")
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    if mode_name == "ar":
        _log_stage("ar: entering single-image benchmark path")
        run_metrics = _run_ar_t2i(
            cfg=cfg,
            model=model,
            tokenizer=tokenizer,
            vq_model=vq_model,
            prompt_text=prompt_text,
            out_dir=save_path,
        )
    elif mode_name == "nar":
        _log_stage("nar: entering single-image benchmark path")
        run_metrics = _run_nar_t2i(
            cfg=cfg,
            nar_wrapper=nar_wrapper,
            tokenizer=tokenizer,
            vq_model=vq_model,
            prompt_text=prompt_text,
            out_dir=save_path,
        )
    else:
        raise ValueError(f"Unsupported mode {mode_name}")

    metrics = {
        "mode": mode_name,
        "cfg_path": osp.abspath(cfg_path),
        "save_path": osp.abspath(save_path),
        "prompt_count": 1,
        "load_seconds": load_seconds,
        "infer_seconds": run_metrics["infer_seconds"],
        "decode_seconds": run_metrics["decode_seconds"],
        "total_seconds": load_seconds + run_metrics["infer_seconds"] + run_metrics["decode_seconds"],
        "final_image": run_metrics["final_image"],
        "nar_ckpt_path": resolved_nar_ckpt_path or getattr(cfg, "nar_ckpt_path", ""),
        "load_mode": load_mode,
        "nar_attn_implementation": str(getattr(cfg, "nar_attn_implementation", "")),
        "nar_use_kv_cache": bool(getattr(cfg, "nar_use_kv_cache", True)),
        "hf_device": str(cfg.hf_device),
        "vq_device": str(cfg.vq_device),
        "height": int(cfg.target_height),
        "width": int(cfg.target_width),
    }

    cleanup_cuda(model, nar_wrapper, tokenizer, vq_model, cfg)
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ar_cfg",
        type=str,
        default="configs/benchmark_t2i_ar.py",
        help="AR benchmark config path.",
    )
    parser.add_argument(
        "--nar_cfg",
        type=str,
        default="configs/benchmark_t2i_nar.py",
        help="NAR benchmark config path.",
    )
    parser.add_argument(
        "--out_root",
        type=str,
        default="",
        help="Output directory. Defaults to outputs/bench_ar_vs_nar/<timestamp>.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.out_root:
        out_root = osp.abspath(args.out_root)
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_root = osp.abspath(osp.join("outputs", "bench_ar_vs_nar", stamp))
    os.makedirs(out_root, exist_ok=True)

    summaries = []
    for mode_name, cfg_path in (("ar", args.ar_cfg), ("nar", args.nar_cfg)):
        print(f"[INFO] running {mode_name} benchmark with {cfg_path}", flush=True)
        metrics = _run_single_mode(mode_name, cfg_path, out_root)
        summaries.append(metrics)
        print(
            "[RESULT] "
            f"mode={metrics['mode']} "
            f"load_mode={metrics['load_mode']} "
            f"attn={metrics['nar_attn_implementation'] or 'default'} "
            f"kv_cache={metrics.get('nar_use_kv_cache', False)} "
            f"load={metrics['load_seconds']:.2f}s "
            f"infer={metrics['infer_seconds']:.2f}s "
            f"decode={metrics['decode_seconds']:.2f}s "
            f"total={metrics['total_seconds']:.2f}s "
            f"final_image={metrics['final_image']}",
            flush=True,
        )

    if len(summaries) == 2:
        ar_metrics, nar_metrics = summaries
        if nar_metrics["infer_seconds"] > 0:
            speedup = ar_metrics["infer_seconds"] / nar_metrics["infer_seconds"]
        else:
            speedup = 0.0
        comparison = {
            "ar_infer_seconds": ar_metrics["infer_seconds"],
            "nar_infer_seconds": nar_metrics["infer_seconds"],
            "ar_over_nar_infer_speedup": speedup,
        }
    else:
        comparison = {}

    payload = {
        "out_root": out_root,
        "runs": summaries,
        "comparison": comparison,
    }
    summary_path = osp.join(out_root, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"[INFO] summary saved to {summary_path}", flush=True)


if __name__ == "__main__":
    main()
