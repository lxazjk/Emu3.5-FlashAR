#!/usr/bin/env bash
set -euo pipefail

: "${PYTHON_BIN:=python3}"
: "${MODEL_PATH:=./weights/Emu3.5-Image}"
: "${TOKENIZER_PATH:=./src/tokenizer_emu3_ibq}"
: "${VQ_PATH:=./weights/Emu3.5-VisionTokenizer}"
: "${CKPT_PATH:=./outputs/flashar_finetune/flashar_final}"
: "${HEIGHT:=32}"
: "${WIDTH:=32}"
: "${PROMPT:=a pig}"
: "${TEXT_TEMPLATE:=<|extra_203|>You are a helpful assistant for t2i task. USER: {text} ASSISTANT: <|extra_100|>}"
: "${OUT_PATH:=./flashar_out.png}"
: "${DTYPE:=bf16}"
: "${TEMPERATURE:=1.0}"
: "${TOP_K:=0}"
: "${TOP_P:=1.0}"
: "${CFG_SCALE:=1.0}"
: "${USE_VERTICAL_BLOCK:=auto}"
: "${VERTICAL_LAYERS:=0}"
: "${VERTICAL_START_LAYER:=-1}"
: "${SPLIT_BACKBONE:=0}"

CMD=(
  "${PYTHON_BIN}" generate_flashar.py
  --model_path "${MODEL_PATH}"
  --tokenizer_path "${TOKENIZER_PATH}"
  --vq_path "${VQ_PATH}"
  --ckpt_path "${CKPT_PATH}"
  --height "${HEIGHT}"
  --width "${WIDTH}"
  --prompt "${PROMPT}"
  --text_template "${TEXT_TEMPLATE}"
  --dtype "${DTYPE}"
  --temperature "${TEMPERATURE}"
  --top_k "${TOP_K}"
  --top_p "${TOP_P}"
  --cfg_scale "${CFG_SCALE}"
  --out "${OUT_PATH}"
  --vertical_layers "${VERTICAL_LAYERS}"
  --vertical_start_layer "${VERTICAL_START_LAYER}"
  --add_boi
)

if [ "${USE_VERTICAL_BLOCK}" = "1" ]; then
  CMD+=(--use_vertical_block)
elif [ "${USE_VERTICAL_BLOCK}" = "0" ]; then
  CMD+=(--no-use_vertical_block)
fi

if [ "${SPLIT_BACKBONE}" = "1" ]; then
  CMD+=(--split_backbone)
fi

"${CMD[@]}"
