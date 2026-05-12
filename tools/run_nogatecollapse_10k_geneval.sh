#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

RUN_NAME="${RUN_NAME:-emu3.5-nar_nogatecollapse_10k}"
START_CKPT="${START_CKPT:-./weights/flashar_step74000_geneval077798_cfg5}"
SAVE_DIR="${SAVE_DIR:-./outputs/${RUN_NAME}}"
TRAIN_LOG="${TRAIN_LOG:-./logs/train_${RUN_NAME}.log}"
TORCHRUN_LOG_DIR="${TORCHRUN_LOG_DIR:-./logs/torchrun_${RUN_NAME}}"
GENEVAL_IMAGE_DIR="${GENEVAL_IMAGE_DIR:-./outputs/geneval_${RUN_NAME}_cfg5/images}"
GENEVAL_EVAL_DIR="${GENEVAL_EVAL_DIR:-./outputs/geneval_${RUN_NAME}_cfg5_officialdet}"
GENEVAL_DRIVER_LOG="${GENEVAL_DRIVER_LOG:-./logs/geneval_${RUN_NAME}_cfg5_generate_driver.log}"
GENEVAL_EVAL_LOG="${GENEVAL_EVAL_LOG:-./logs/geneval_${RUN_NAME}_cfg5_officialdet_eval.log}"

CUDA_VISIBLE_DEVICES_TRAIN="${CUDA_VISIBLE_DEVICES_TRAIN:-0,1,2,3,4,5,6,7}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
TARGET_MAX_STEPS="${TARGET_MAX_STEPS:-84000}"
LR="${LR:-1e-6}"
CFG_SCALE="${CFG_SCALE:-5.0}"
HEIGHT="${HEIGHT:-32}"
WIDTH="${WIDTH:-32}"
SAMPLES_PER_PROMPT="${SAMPLES_PER_PROMPT:-4}"
NUM_GEN_GPUS="${NUM_GEN_GPUS:-8}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
GENEVAL_PYTHON="${GENEVAL_PYTHON:-./.venv-geneval3/bin/python}"

mkdir -p logs "${SAVE_DIR}" "${GENEVAL_IMAGE_DIR}" "${GENEVAL_EVAL_DIR}" "${TORCHRUN_LOG_DIR}"

echo "[$(date '+%F %T')] train start: run=${RUN_NAME} start_ckpt=${START_CKPT} max_steps=${TARGET_MAX_STEPS} lr=${LR} gate_collapse_weight=0.0" | tee "${TRAIN_LOG}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_TRAIN}" \
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
PYTHONUNBUFFERED=1 \
PYTHONFAULTHANDLER=1 \
torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" --tee 3 --log-dir "${TORCHRUN_LOG_DIR}" \
  train_flashar.py \
  --config_json ./configs/train_flashar_resume_opengpt4o_gen.json \
  --resume_path "${START_CKPT}" \
  --save_dir "${SAVE_DIR}" \
  --pretok_glob '/opt/tiger/Emu3.5-NAR/OpenGPT4o-Image_pretok_32/gen/*.tar' \
  --lr "${LR}" \
  --lr_scheduler none \
  --max_steps "${TARGET_MAX_STEPS}" \
  --gate_collapse_weight 0.0 \
  --save_every_steps 2000 \
  --num_workers 0 \
  --wandb_name "${RUN_NAME}" \
  2>&1 | tee -a "${TRAIN_LOG}"

CKPT_PATH="${SAVE_DIR}/flashar_final"
echo "[$(date '+%F %T')] train done: ckpt=${CKPT_PATH}" | tee -a "${TRAIN_LOG}"

echo "[$(date '+%F %T')] geneval generation start: image_dir=${GENEVAL_IMAGE_DIR} cfg=${CFG_SCALE} HW=${HEIGHT}x${WIDTH}" | tee "${GENEVAL_DRIVER_LOG}"
TOTAL_PROMPTS="$(${PYTHON_BIN} - <<'PY'
from pathlib import Path
path = Path("datasets/geneval/prompts/evaluation_metadata.jsonl")
if not path.exists():
    path = Path("geneval/prompts/evaluation_metadata.jsonl")
print(sum(1 for line in path.open("r", encoding="utf-8") if line.strip()))
PY
)"
CHUNK=$(( (TOTAL_PROMPTS + NUM_GEN_GPUS - 1) / NUM_GEN_GPUS ))

for ((rank=0; rank<NUM_GEN_GPUS; rank++)); do
  start=$((rank * CHUNK))
  end=$((start + CHUNK))
  if (( end > TOTAL_PROMPTS )); then
    end="${TOTAL_PROMPTS}"
  fi
  if (( start >= TOTAL_PROMPTS )); then
    continue
  fi
  log_path="./logs/geneval_${RUN_NAME}_cfg5_rank${rank}_${start}_${end}.log"
  echo "[$(date '+%F %T')] launch rank=${rank} gpu=${rank} range=${start}-${end}" | tee -a "${GENEVAL_DRIVER_LOG}"
  CUDA_VISIBLE_DEVICES="${rank}" "${PYTHON_BIN}" tools/generate_geneval_flashar.py \
    --metadata datasets/geneval/prompts/evaluation_metadata.jsonl \
    --outdir "${GENEVAL_IMAGE_DIR}" \
    --ckpt_path "${CKPT_PATH}" \
    --height "${HEIGHT}" \
    --width "${WIDTH}" \
    --cfg_scale "${CFG_SCALE}" \
    --samples_per_prompt "${SAMPLES_PER_PROMPT}" \
    --start "${start}" \
    --end "${end}" \
    --device cuda:0 \
    --overwrite \
    > "${log_path}" 2>&1 &
done
wait

image_count="$(find "${GENEVAL_IMAGE_DIR}" -path '*/samples/*.png' -type f | wc -l)"
expected_count=$((TOTAL_PROMPTS * SAMPLES_PER_PROMPT))
echo "[$(date '+%F %T')] geneval generation done: images=${image_count} expected=${expected_count}" | tee -a "${GENEVAL_DRIVER_LOG}"
if [[ "${image_count}" -ne "${expected_count}" ]]; then
  echo "Expected ${expected_count} generated images, found ${image_count}" >&2
  exit 1
fi

if [[ ! -x "${GENEVAL_PYTHON}" ]]; then
  GENEVAL_PYTHON="${PYTHON_BIN}"
fi

echo "[$(date '+%F %T')] geneval official-det eval start" | tee "${GENEVAL_EVAL_LOG}"
CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" "${GENEVAL_PYTHON}" geneval/evaluation/evaluate_images.py \
  "${GENEVAL_IMAGE_DIR}" \
  --outfile "${GENEVAL_EVAL_DIR}/results.jsonl" \
  --model-path weights/geneval_detectors \
  --options \
    model=mask2former_swin-s-p4-w7-224_8xb2-lsj-50e_coco_official_mmdet3 \
    num_gpus=8 \
    gpu_ids=0,1,2,3,4,5,6,7 \
    clip_num_workers=0 \
  2>&1 | tee -a "${GENEVAL_EVAL_LOG}"

"${GENEVAL_PYTHON}" geneval/evaluation/summary_scores.py \
  "${GENEVAL_EVAL_DIR}/results.jsonl" \
  2>&1 | tee "${GENEVAL_EVAL_DIR}/summary.txt" | tee -a "${GENEVAL_EVAL_LOG}"

echo "[$(date '+%F %T')] all done: summary=${GENEVAL_EVAL_DIR}/summary.txt" | tee -a "${GENEVAL_EVAL_LOG}"
