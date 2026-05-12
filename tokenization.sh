#!/usr/bin/env bash
set -euo pipefail

: "${NPROC_PER_NODE:=8}"
: "${JSON_PATH:=./data/GPT4o-Image/text_to_image.json}"
: "${IMAGE_ROOT:=./data/GPT4o-Image}"
: "${OUTPUT_DIR:=./data/GPT4o-Image_pretok_32}"
: "${SPLIT:=text_to_image}"
: "${VQ_PATH:=./weights/Emu3.5-VisionTokenizer}"
: "${VQ_TYPE:=ibq}"
: "${VQ_DEVICE:=cuda}"
: "${GRID_HEIGHT:=32}"
: "${GRID_WIDTH:=32}"
: "${OUTPUT_PREFIX:=gpt4o_t2i_32}"
: "${SHARD_SIZE:=5000}"

torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" flashar/data/pretokenize_tar.py \
  --json_path "${JSON_PATH}" \
  --image_root "${IMAGE_ROOT}" \
  --output_dir "${OUTPUT_DIR}" \
  --split "${SPLIT}" \
  --vq_path "${VQ_PATH}" \
  --vq_type "${VQ_TYPE}" \
  --vq_device "${VQ_DEVICE}" \
  --grid_height "${GRID_HEIGHT}" \
  --grid_width "${GRID_WIDTH}" \
  --output_prefix "${OUTPUT_PREFIX}" \
  --shard_size "${SHARD_SIZE}"
