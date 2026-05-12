<div align="center">

# FlashAR

### Efficient Post-Training Acceleration for Autoregressive Image Generation

[![arXiv](https://img.shields.io/badge/arXiv-2605.09430-b31b1b.svg)](https://arxiv.org/abs/2605.09430)
[![Project](https://img.shields.io/badge/Project-Page-blue)](https://lxazjk.github.io/FlashAR/)
[![Hugging Face](https://img.shields.io/badge/HuggingFace-Checkpoint-yellow)](https://huggingface.co/lxazjk/Emu3.5-Image-FlashAR)
[![License](https://img.shields.io/badge/License-Apache--2.0-green.svg)](LICENSE)

</div>

This repository contains the Emu3.5-Image implementation of FlashAR. It provides
the code needed to:

- run inference with an existing FlashAR checkpoint;
- pretokenize image-text data into training shards;
- post-train FlashAR on top of an Emu3.5-Image backbone;
- generate and evaluate samples with GenEval-style metadata;
- benchmark standard AR decoding against FlashAR diagonal decoding.

<p align="center">
  <img src="assets/teaser.png" width="95%" alt="Generated samples from FlashAR">
</p>

## Overview

FlashAR accelerates a pretrained raster-scan autoregressive image generator by
adding a vertical prediction branch and a learnable fusion gate. Decoding then
proceeds by anti-diagonal steps, reducing the serial image-token decoding length
from `H * W` to `H + W - 1`.

<p align="center">
  <img src="assets/overview.png" width="86%" alt="Overview of FlashAR">
</p>

## Repository Layout

```text
.
├── flashar/                         # FlashAR model, data, training utilities
│   ├── data/                        # Pretokenized dataset and preprocessing
│   ├── inference/                   # Sampling and token-format helpers
│   ├── model/                       # Emu3.5 FlashAR wrapper and decoding logic
│   └── utils/                       # Config, checkpoint, optimizer, training helpers
├── src/                             # Emu3.5 model, tokenizer, VQ tokenizer, runtime utils
├── configs/                         # Training and benchmark configs
├── tools/                           # GenEval, benchmarking, visualization scripts
├── requirements/                    # Dependency sets
├── train_flashar.py                 # Main FSDP training entry
├── generate_flashar.py              # Single-prompt generation entry
├── train.sh                         # Config-driven torchrun launcher
├── generate.sh                      # Generation launcher
├── tokenization.sh                  # Pretokenization launcher
└── README.md
```

## Installation

Create a clean Python environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements/transformers.txt
pip install flash_attn==2.8.3 --no-build-isolation
```

## Download Weights

Download the base Emu3.5-Image model, the Emu3.5 vision tokenizer, and the
FlashAR checkpoint. The examples below assume this layout:

The FlashAR checkpoint is available on Hugging Face:
[lxazjk/Emu3.5-Image-FlashAR](https://huggingface.co/lxazjk/Emu3.5-Image-FlashAR).

```text
weights/
├── Emu3.5-Image/
└── Emu3.5-VisionTokenizer/

checkpoints/
└── Emu3.5-Image-Flash/
```

The default scripts expect:

```text
MODEL_PATH       ./weights/Emu3.5-Image
TOKENIZER_PATH   ./src/tokenizer_emu3_ibq
VQ_PATH          ./weights/Emu3.5-VisionTokenizer
CKPT_PATH        ./checkpoints/Emu3.5-Image-Flash
```

You can use different locations by setting environment variables or editing the
JSON configs under `configs/`.

## Quick Start: Inference With Existing Weights

Use this path if you already have:

- the base Emu3.5-Image model;
- the Emu3.5 vision tokenizer;
- a trained FlashAR checkpoint.

Recommended local layout:

```text
weights/Emu3.5-Image/
weights/Emu3.5-VisionTokenizer/
checkpoints/Emu3.5-Image-Flash/
```

Generate one image with the shell launcher:

```bash
MODEL_PATH=./weights/Emu3.5-Image \
TOKENIZER_PATH=./src/tokenizer_emu3_ibq \
VQ_PATH=./weights/Emu3.5-VisionTokenizer \
CKPT_PATH=./checkpoints/Emu3.5-Image-Flash \
PROMPT="a red car parked next to a blue mailbox" \
CFG_SCALE=5.0 \
OUT_PATH=./outputs/sample.png \
bash generate.sh
```

The launcher uses the default 32 x 32 visual-token grid.

Important generation options:

| Option | Description |
| --- | --- |
| `CFG_SCALE` | Classifier-free guidance scale. |
| `PROMPT` | Text prompt. |
| `OUT_PATH` | Output image path. |
| `USE_VERTICAL_BLOCK` | `auto` by default; set `1` or `0` to force the branch on or off. |
| `SPLIT_BACKBONE` | Set `1` if the checkpoint was trained with split backbone inference. |

## Full Training Pipeline

The training workflow is:

```text
raw image-text data -> pretokenize -> train FlashAR -> generate/evaluate
```

### Step 1: Pretokenize Data

Training reads tar shards containing visual tokens and captions. Each sample in
a shard contains:

```text
<sample>.pt   # visual token tensor and shape metadata
<sample>.txt  # text prompt or caption
```

Run pretokenization with `tokenization.sh`:

```bash
JSON_PATH=./data/GPT4o-Image/text_to_image.json \
IMAGE_ROOT=./data/GPT4o-Image \
OUTPUT_DIR=./data/GPT4o-Image_pretok_32 \
VQ_PATH=./weights/Emu3.5-VisionTokenizer \
SHARD_SIZE=5000 \
NPROC_PER_NODE=8 \
bash tokenization.sh
```

Key variables:

| Variable | Description |
| --- | --- |
| `JSON_PATH` | Input metadata JSON containing image paths and text. |
| `IMAGE_ROOT` | Root directory for source images. |
| `OUTPUT_DIR` | Directory where tar shards are written. |
| `SPLIT` | Split name under `OUTPUT_DIR`; default is `text_to_image`. |
| `VQ_PATH` | Emu3.5 vision tokenizer path. |
| `SHARD_SIZE` | Number of samples per tar shard. |
| `NPROC_PER_NODE` | Number of distributed pretokenization workers. |

After this step, point `data.pretok_glob` in the training config to the produced
tar files, for example:

```text
./data/GPT4o-Image_pretok_32/text_to_image_partfirst/*.tar
```

### Step 2: Train FlashAR

Edit `configs/train_flashar.default.json` so that the paths match your machine:

```json
{
  "paths": {
    "model_path": "./weights/Emu3.5-Image",
    "tokenizer_path": "./src/tokenizer_emu3_ibq",
    "vq_path": "./weights/Emu3.5-VisionTokenizer",
    "save_dir": "./outputs/flashar_finetune",
    "resume_path": ""
  },
  "data": {
    "pretok_glob": "./data/GPT4o-Image_pretok_32/text_to_image_partfirst/*.tar"
  }
}
```

Start training:

```bash
TRAIN_CONFIG_JSON=./configs/train_flashar.default.json bash train.sh
```

Override common options without editing the config:

```bash
TRAIN_CONFIG_JSON=./configs/train_flashar.default.json \
EXTRA_ARGS="--max_steps 1000 --save_every_steps 100 --lr 1e-5" \
bash train.sh
```

Resume or continue from a FlashAR checkpoint:

```bash
TRAIN_CONFIG_JSON=./configs/train_flashar.default.json \
EXTRA_ARGS="--resume_path ./checkpoints/Emu3.5-Image-Flash --save_dir ./outputs/continue_lr1e6 --lr 1e-6 --lr_scheduler none" \
bash train.sh
```

Main training config fields:

| Field | Description |
| --- | --- |
| `launcher.cuda_visible_devices` | CUDA devices used by `torchrun`. |
| `launcher.nproc_per_node` | Number of distributed processes. |
| `paths.model_path` | Base Emu3.5-Image model. |
| `paths.vq_path` | Emu3.5 vision tokenizer. |
| `paths.save_dir` | Output directory for checkpoints. |
| `paths.resume_path` | Optional checkpoint used to resume. |
| `data.pretok_glob` | Glob for pretokenized tar shards. |
| `optimization.lr` | Main learning rate. |
| `optimization.max_steps` | Maximum optimizer steps. |
| `optimization.train_backbone` | Whether to update the backbone. |
| `model.use_vertical_block` | Enable the vertical branch. |
| `model.vertical_layers` | Number of vertical branch layers. |

The training script saves the final FlashAR checkpoint to:

```text
<save_dir>/flashar_final/
```

### Step 3: Evaluate

First generate images in GenEval format:

```bash
CUDA_VISIBLE_DEVICES=0 python tools/generate_geneval_flashar.py \
  --model_path ./weights/Emu3.5-Image \
  --tokenizer_path ./src/tokenizer_emu3_ibq \
  --vq_path ./weights/Emu3.5-VisionTokenizer \
  --ckpt_path ./outputs/flashar_finetune/flashar_final \
  --metadata ./datasets/geneval/prompts/evaluation_metadata.jsonl \
  --outdir ./outputs/geneval_flashar/images \
  --samples_per_prompt 4 \
  --cfg_scale 5.0 \
  --dtype bf16
```

Expected generated layout:

```text
outputs/geneval_flashar/images/<prompt_id>/samples/0000.png
outputs/geneval_flashar/images/<prompt_id>/samples/0001.png
outputs/geneval_flashar/images/<prompt_id>/metadata.jsonl
```

Download the official GenEval repository to `geneval/`, install its evaluation
dependencies, and run the scorer:

```bash
CUDA_VISIBLE_DEVICES=0 python geneval/evaluation/evaluate_images.py \
  ./outputs/geneval_flashar/images \
  --outfile ./outputs/geneval_flashar/results.jsonl \
  --model-path ./weights/geneval_detectors

python geneval/evaluation/summary_scores.py \
  ./outputs/geneval_flashar/results.jsonl
```

## Benchmark AR vs FlashAR

Compare standard AR decoding and FlashAR decoding:

```bash
EMU35_BENCH_FLASHAR_CKPT=./checkpoints/Emu3.5-Image-Flash \
python tools/benchmark_ar_vs_flashar.py \
  --ar_cfg configs/benchmark_t2i_ar.py \
  --flashar_cfg configs/benchmark_t2i_flashar.py \
  --out_dir outputs/bench_ar_vs_flashar
```

## Results

**Emu3.5-Image-34B inference efficiency at 512 x 512.**

| Method | Type | Steps | Data | Latency (s) | Decoding steps |
| --- | --- | ---: | ---: | ---: | ---: |
| Emu3.5-Image | From scratch | 940K | 150B | 130.10 | 1024 |
| BlockDiffusion | Post-training | 50K | 80M | 6.17 | 64 |
| **FlashAR** | Post-training | 50K | 80M | **5.68** | **63** |

**Emu3.5-Image-34B GenEval at 512 x 512.**

| Method | Overall | Single Obj | Two Obj | Counting | Colors | Position | Color Attr |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Emu3.5-Image | 80.48 | 100.00 | 94.95 | 53.75 | 90.96 | 73.00 | 70.25 |
| BlockDiffusion | 73.83 | 96.88 | 88.89 | 47.50 | 85.64 | 68.00 | 58.44 |
| **FlashAR** | **80.29** | 98.75 | 91.92 | 53.75 | **92.55** | **80.00** | 64.00 |

<p align="center">
  <img src="assets/apendix_emu.png" width="95%" alt="Text-guided generation samples">
</p>

## Citation

```bibtex
@article{zhou2026flashar,
  title={FlashAR: Efficient Post-Training Acceleration for Autoregressive Image Generation},
  author={Zhou, Junkang and He, Yefei and Chen, Feng and Wang, Weijie and Zhuang, Bohan},
  journal={arXiv preprint arXiv:2605.09430},
  year={2026}
}
```
