# -*- coding: utf-8 -*-
# NAR fine-tuning entry for Infinity-MM webdataset shards (json + image).

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
from emu_nar.inference.neighbor_ar_wrapper import NeighborARWrapper, _sample_logits
from src.vision_tokenizer import build_vision_tokenizer
from emu_nar.lora import (
    apply_lora_to_backbone,
    apply_progressive_lora_to_backbone,
    collect_lora_modules,
    iter_lora_parameters,
)

from emu_nar.data.infinity_mm import (
    InfinityMMShardDataset,
    collate_fn,
    encode_images,
)
from emu_nar.data.gpt4o_image import GPT4oImageDataset
from emu_nar.data.pretokenized import PretokShardDataset, collate_pretok


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--tokenizer_path", type=str, required=True)
    parser.add_argument("--vq_path", type=str, required=True)
    parser.add_argument("--vq_type", type=str, default="ibq")
    parser.add_argument("--dataset_glob", type=str, default="", help="Glob for Infinity-MM tar shards.")
    parser.add_argument("--dataset_json", type=str, default="", help="JSON list for GPT4o-Image T2I.")
    parser.add_argument("--image_root", type=str, default="", help="Image root for dataset_json.")
    parser.add_argument("--pretok_glob", type=str, default="", help="Glob for pretokenized .tar shards.")
    parser.add_argument(
        "--text_source",
        type=str,
        default="assistant",
        choices=["assistant", "human", "both", "caption"],
    )
    parser.add_argument("--text_template", type=str, default="{text}")
    parser.add_argument("--text_max_length", type=int, default=0)
    parser.add_argument("--add_boi", action="store_true")
    parser.add_argument("--image_area", type=int, default=0)
    parser.add_argument("--image_size", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-6)
    parser.add_argument("--grad_clip", type=float, default=0.0)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--vq_device", type=str, default="auto")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--train_backbone", action="store_true")
    parser.add_argument("--fsdp", action="store_true")
    parser.add_argument("--fsdp_min_params", type=int, default=1_000_000)
    parser.add_argument("--fsdp_wrap_policy", type=str, default="transformer", choices=["size", "transformer"])
    parser.add_argument("--fsdp_cpu_offload", action="store_true")
    parser.add_argument("--save_dir", type=str, default="./outputs/nar_finetune")
    parser.add_argument("--save_every_steps", type=int, default=0)
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
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument(
        "--chunked_loss",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Compute loss in row chunks to save memory (slower).",
    )
    parser.add_argument("--lora_r", type=int, default=0)
    parser.add_argument(
        "--lora_r_min",
        type=int,
        default=0,
        help="Minimum LoRA rank for progressive schedule (front layers).",
    )
    parser.add_argument("--lora_alpha", type=float, default=0.0)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument("--lora_layers", type=int, default=0)
    return parser.parse_args()


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


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

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
        device = torch.device(args.device)
        local_rank = 0

    is_main = rank == 0
    if args.fsdp or is_main:
        os.makedirs(args.save_dir, exist_ok=True)
    if is_main and args.train_backbone:
        print("[WARN] training full backbone; consider unfreezing after NAR heads converge.")

    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    torch_dtype = dtype_map[args.dtype]

    model_config = Emu3Config.from_pretrained(args.model_path, trust_remote_code=True)
    backbone = Emu3ForCausalLM.from_pretrained(
        args.model_path,
        config=model_config,
        torch_dtype=torch_dtype,
        attn_implementation="eager",
    )
    if args.gradient_checkpointing:
        backbone.gradient_checkpointing_enable()
        backbone.config.use_cache = False
    if args.lora_r > 0 and args.lora_layers > 0:
        if args.lora_r_min > 0:
            alpha_scale = (args.lora_alpha if args.lora_alpha > 0 else float(args.lora_r)) / float(
                args.lora_r
            )
            lora_count = apply_progressive_lora_to_backbone(
                backbone,
                num_layers=args.lora_layers,
                r_min=args.lora_r_min,
                r_max=args.lora_r,
                alpha_scale=alpha_scale,
                dropout=args.lora_dropout,
            )
        else:
            lora_alpha = args.lora_alpha if args.lora_alpha > 0 else float(args.lora_r)
            lora_count = apply_lora_to_backbone(
                backbone,
                num_layers=args.lora_layers,
                r=args.lora_r,
                alpha=lora_alpha,
                dropout=args.lora_dropout,
            )
        if is_main:
            print(f"[INFO] applied LoRA to {lora_count} linear layers.")
    if not args.fsdp:
        backbone = backbone.to(device)

    visual_token_offset = args.visual_token_offset
    if visual_token_offset < 0:
        visual_token_offset = int(model_config.eoi_token_id) + 1
    max_visual_id = visual_token_offset + args.visual_vocab_size - 1
    if max_visual_id >= model_config.vocab_size:
        raise ValueError(
            f"visual_token_offset {visual_token_offset} + visual_vocab_size {args.visual_vocab_size} "
            f"exceeds vocab_size {model_config.vocab_size}."
        )

    vertical_layers = args.vertical_layers
    if vertical_layers <= 0:
        vertical_layers = int(getattr(model_config, "nar_vertical_layers", 1))
    wrapper = NeighborARWrapper(
        pretrained_backbone=backbone.model,
        vocab_size=model_config.vocab_size,
        hidden_size=model_config.hidden_size,
        num_heads=model_config.num_attention_heads,
        pad_token_id=-100,
        mask_token_id=model_config.pad_token_id,
        visual_token_offset=visual_token_offset,
        img_token_id=model_config.img_token_id,
        eol_token_id=model_config.eol_token_id,
        eoi_token_id=model_config.eoi_token_id,
        use_vertical_block=args.use_vertical_block,
        vertical_layers=vertical_layers,
        lm_head=backbone.lm_head,
    )
    if args.fsdp:
        wrapper = wrapper.to(dtype=torch_dtype)
    else:
        wrapper = wrapper.to(device=device, dtype=torch_dtype)

    if args.fsdp:
        from torch.distributed.fsdp import CPUOffload, FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy, transformer_auto_wrap_policy
        from src.emu3p5.modeling_emu3 import Emu3DecoderLayer

        if args.fsdp_wrap_policy == "transformer":
            auto_wrap_policy = partial(
                transformer_auto_wrap_policy, transformer_layer_cls={Emu3DecoderLayer}
            )
        else:
            auto_wrap_policy = partial(size_based_auto_wrap_policy, min_num_params=args.fsdp_min_params)
        cpu_offload = CPUOffload(offload_params=True) if args.fsdp_cpu_offload else None
        lora_enabled = args.lora_r > 0 and args.lora_layers > 0
        ignored_modules = collect_lora_modules(wrapper) if lora_enabled else None
        fsdp_kwargs = dict(
            auto_wrap_policy=auto_wrap_policy,
            cpu_offload=cpu_offload,
            device_id=device,
            use_orig_params=False,
        )
        if ignored_modules:
            import inspect

            if "ignored_modules" in inspect.signature(FSDP).parameters:
                fsdp_kwargs["ignored_modules"] = ignored_modules
        wrapper = FSDP(wrapper, **fsdp_kwargs)
        if ignored_modules:
            for mod in ignored_modules:
                mod.to(device=device, dtype=torch_dtype)

    if args.resume_path:
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
                    wrapper.load_state_dict(state_dict, strict=True)
                if rank == 0:
                    print("[INFO] loaded sharded ckpt:", resume_path)
            elif state_type == "local":
                state_cfg = LocalStateDictConfig(offload_to_cpu=True)
                with FSDP.state_dict_type(wrapper, StateDictType.LOCAL_STATE_DICT, state_cfg):
                    wrapper.load_state_dict(state_dict, strict=True)
                if rank == 0:
                    print("[INFO] loaded local ckpt:", resume_path)
            else:
                # Full state dict needs to be loaded on all ranks for FSDP.
                state_cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=False)
                with FSDP.state_dict_type(wrapper, StateDictType.FULL_STATE_DICT, state_cfg):
                    wrapper.load_state_dict(state_dict, strict=True)
                if rank == 0:
                    print("[INFO] loaded full ckpt:", resume_path)
            dist.barrier()
        else:
            state_dict = _safe_torch_load(args.resume_path)
            wrapper.load_state_dict(state_dict, strict=True)
            if rank == 0:
                print("[INFO] loaded ckpt:", args.resume_path)

    if not args.train_backbone:
        target = wrapper.module if args.fsdp else wrapper
        for p in target.backbone.parameters():
            p.requires_grad = False
        if args.lora_r > 0 and args.lora_layers > 0:
            for p in iter_lora_parameters(target.backbone):
                p.requires_grad = True

    trainable_params = [p for p in wrapper.parameters() if p.requires_grad and p.numel() > 0]
    use_foreach = not (args.fsdp and args.lora_r > 0 and args.lora_layers > 0)
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, foreach=use_foreach)

    tokenizer = _build_text_tokenizer(args.tokenizer_path)

    use_pretok = bool(args.pretok_glob)
    vq_device = device if args.vq_device == "auto" else torch.device(args.vq_device)
    vq_model = None
    vq_dtype = None
    if not use_pretok:
        vq_model = build_vision_tokenizer(args.vq_type, args.vq_path, device=str(vq_device))
        vq_model.eval()
        for p in vq_model.parameters():
            p.requires_grad = False
        vq_dtype = next(vq_model.parameters()).dtype

    if use_pretok:
        shard_paths = sorted(glob.glob(args.pretok_glob))
        if not shard_paths:
            raise FileNotFoundError(f"No pretokenized tar shards for glob: {args.pretok_glob}")
        ds = PretokShardDataset(
            shard_paths=shard_paths,
            rank=rank,
            world_size=world_size,
        )
        data_collate = collate_pretok
    elif args.dataset_json:
        image_root = args.image_root or osp.dirname(args.dataset_json)
        ds = GPT4oImageDataset(
            json_path=args.dataset_json,
            image_root=image_root,
            rank=rank,
            world_size=world_size,
        )
        data_collate = collate_fn
    else:
        if not args.dataset_glob:
            raise ValueError("Provide --dataset_glob, --dataset_json, or --pretok_glob.")
        shard_paths = sorted(glob.glob(args.dataset_glob))
        if not shard_paths:
            raise FileNotFoundError(f"No tar shards found for glob: {args.dataset_glob}")
        ds = InfinityMMShardDataset(
            shard_paths=shard_paths,
            text_source=args.text_source,
            rank=rank,
            world_size=world_size,
        )
        data_collate = collate_fn

    loader = DataLoader(
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

    save_epoch_mode = args.save_epoch
    if save_epoch_mode == "auto":
        save_epoch_mode = "sharded" if args.fsdp else "full"

    wrapper.train()
    global_step = 0
    accum_steps = max(1, args.grad_accum_steps)
    image_area = args.image_area or int(getattr(model_config, "image_area", 512 * 512))

    def _run_epoch_generate(epoch: int) -> None:
        if not args.eval_generate_prompt:
            return
        if not args.fsdp:
            raise ValueError("eval_generate requires --fsdp for multi-GPU generation.")
        height = int(args.eval_generate_height)
        width = int(args.eval_generate_width)
        if height <= 0 or width <= 0:
            if is_main:
                print("[WARN] eval_generate_height/width must be set to enable generation.")
            return
        out_dir = args.eval_generate_outdir or args.save_dir
        if is_main:
            os.makedirs(out_dir, exist_ok=True)

        def _save_outputs(grid: torch.Tensor) -> None:
            if not is_main:
                return
            grid_cpu = grid.detach().cpu()
            torch.save(grid_cpu, os.path.join(out_dir, f"gen_epoch{epoch}.pt"))
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

                img = Image.fromarray(
                    ((image + 1.0) * 127.5).clamp(0, 255).cpu().numpy().astype(np.uint8)
                )
                img.save(os.path.join(out_dir, f"gen_epoch{epoch}.png"))
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
        prompt_attention = torch.ones(
            (prompt_ids.size(0), prompt_ids.size(1)), dtype=torch.long, device=device
        )

        mask_token_id = wrapper.module.mask_token_id
        grid = torch.full((1, height, width), mask_token_id, device=device, dtype=torch.long)
        rows = torch.arange(height, device=device).unsqueeze(1).expand(height, width)
        cols = torch.arange(width, device=device).unsqueeze(0).expand(height, width)
        step_id = (rows + cols).reshape(-1)
        max_step = int(step_id.max().item())

        prev_fastpath = None
        if hasattr(torch.backends, "mha"):
            prev_fastpath = torch.backends.mha.get_fastpath_enabled()
            torch.backends.mha.set_fastpath_enabled(False)
        sdp_ctx = nullcontext()
        if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "sdp_kernel"):
            sdp_ctx = torch.backends.cuda.sdp_kernel(
                enable_flash=False, enable_mem_efficient=False, enable_math=True
            )
        with sdp_ctx, torch.no_grad():
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
                if rank == 0:
                    step_pred = _sample_logits(
                        step_logits,
                        temperature=args.eval_generate_temperature,
                        top_k=args.eval_generate_top_k,
                        top_p=args.eval_generate_top_p,
                        sample_logits=args.eval_generate_sample,
                    )
                else:
                    step_pred = torch.zeros(
                        (1, positions.numel()), device=device, dtype=torch.long
                    )
                dist.broadcast(step_pred, src=0)
                grid.view(1, -1)[:, positions] = step_pred

        if prev_fastpath is not None:
            torch.backends.mha.set_fastpath_enabled(prev_fastpath)
        _save_outputs(grid)
        dist.barrier()
        if was_training:
            wrapper.train()
    for epoch in range(args.epochs):
        if args.eval_generate_timing in ("start", "both"):
            _run_epoch_generate(epoch)
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
            if use_pretok:
                tokens = batch["tokens"].long()
                texts = batch["texts"]
                height = int(tokens.size(1))
                width = int(tokens.size(2))
                if visual_token_offset:
                    tokens = tokens + visual_token_offset
                input_ids = tokens.view(tokens.size(0), -1).to(device)
            else:
                images = batch["images"]
                texts = batch["texts"]
                tokens, height, width = encode_images(
                    vq_model=vq_model,
                    images=images,
                    image_area=image_area,
                    image_size=args.image_size,
                    device=vq_device,
                    dtype=vq_dtype,
                    visual_token_offset=visual_token_offset,
                )
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
            if args.fsdp and not is_sync_step and accum_steps > 1:
                sync_ctx = wrapper.no_sync()
            else:
                sync_ctx = nullcontext()

            with sync_ctx:
                outputs = wrapper(
                    input_ids=input_ids,
                    height=height,
                    width=width,
                    text_input_ids=text_ids,
                    text_attention_mask=text_attention_mask,
                    chunked_loss=args.chunked_loss,
                )
                raw_loss = outputs["loss"]
                loss = raw_loss / accum_steps
                loss.backward()

            if is_sync_step:
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(wrapper.parameters(), args.grad_clip)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            if is_main and step % 10 == 0:
                progress.set_postfix(loss=f"{raw_loss.item():.4f}")

            global_step += 1
            if progress is not None:
                progress.update(1)
            if args.save_every_steps > 0 and global_step % args.save_every_steps == 0:
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

        save_path = os.path.join(args.save_dir, f"nar_epoch{epoch}.pt")
        if save_epoch_mode != "none":
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

        if args.eval_generate_timing in ("end", "both"):
            _run_epoch_generate(epoch)

    if args.fsdp and args.save_full_state:
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


if __name__ == "__main__":
    main()
