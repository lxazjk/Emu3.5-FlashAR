set -euo pipefail

RESUME_PATH="${RESUME_PATH:-}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
MAX_STEPS="${MAX_STEPS:-5000}"
EPOCHS="${EPOCHS:-20}"
LR="${LR:-1e-4}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-1}"
TRAIN_BACKBONE="${TRAIN_BACKBONE:-1}"
VERTICAL_HEAD_WARMUP_STEPS="${VERTICAL_HEAD_WARMUP_STEPS:-1000}"
AR_DISTILL_WEIGHT="${AR_DISTILL_WEIGHT:-1.0}"
AR_DISTILL_TEMPERATURE="${AR_DISTILL_TEMPERATURE:-2.0}"
PHASE2_LR_FACTOR="${PHASE2_LR_FACTOR:-0.1}"
EVAL_EVERY_STEPS="${EVAL_EVERY_STEPS:-200}"
EVAL_GENERATE_SEED="${EVAL_GENERATE_SEED:-42}"
LR_SCHEDULER="${LR_SCHEDULER:-cosine}"
LR_MIN_FACTOR="${LR_MIN_FACTOR:-0.05}"
PHASE2_FLAT_STEPS="${PHASE2_FLAT_STEPS:-1000}"
AUX_LOSS_H_WEIGHT="${AUX_LOSS_H_WEIGHT:-0.1}"
AUX_LOSS_V_WEIGHT="${AUX_LOSS_V_WEIGHT:-0.5}"
LEARNABLE_FUSE="${LEARNABLE_FUSE:-1}"
FUSE_H_INIT="${FUSE_H_INIT:-0.3}"
FUSE_CORNER_H_INIT="${FUSE_CORNER_H_INIT:--1}"
if [ -z "${RESUME_PATH}" ]; then
  RESUME_PATH="$(ls -1t ./outputs/nar_finetune/nar_epoch*.full.pt 2>/dev/null | head -n 1 || true)"
fi
RESUME_ARGS=()
if [ -n "${RESUME_PATH}" ]; then
  echo "[INFO] resume from: ${RESUME_PATH}"
  RESUME_ARGS=(--resume_path "${RESUME_PATH}")
else
  echo "[WARN] no resume checkpoint found; start from base weights"
fi
BACKBONE_ARGS=()
if [ "${TRAIN_BACKBONE}" = "1" ]; then
  BACKBONE_ARGS=(--train_backbone)
fi
FUSE_ARGS=(--no-learnable_fuse)
if [ "${LEARNABLE_FUSE}" = "1" ]; then
  FUSE_ARGS=(
    --learnable_fuse
    --fuse_h_init "${FUSE_H_INIT}"
    --fuse_corner_h_init "${FUSE_CORNER_H_INIT}"
  )
fi

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
PYTORCH_ALLOC_CONF=expandable_segments:True \
torchrun --standalone --nproc_per_node=8 train_nar.py \
  --model_path "./weights/Emu3.5-Image" \
  --tokenizer_path "./src/tokenizer_emu3_ibq" \
  --vq_path "./weights/Emu3.5-VisionTokenizer" \
  --text_template "<|extra_203|>You are a helpful assistant for t2i task. USER: {text} ASSISTANT: <|extra_100|>" \
  --pretok_glob "/opt/tiger/Emu3.5-NAR/GPT4o-Image_pretok_32/text_to_image_partfirst/*.tar" \
  --add_boi \
  --fsdp_wrap_policy transformer \
  --lr "${LR}" \
  --text_max_length 128 \
  --use_vertical_block \
  --vertical_layers 4 \
  --epochs "${EPOCHS}" \
  --max_steps "${MAX_STEPS}" \
  --vertical_head_warmup_steps "${VERTICAL_HEAD_WARMUP_STEPS}" \
  --ar_distill_weight "${AR_DISTILL_WEIGHT}" \
  --ar_distill_temperature "${AR_DISTILL_TEMPERATURE}" \
  --phase2_lr_factor "${PHASE2_LR_FACTOR}" \
  "${BACKBONE_ARGS[@]}" \
  --gradient_checkpointing \
  --save_epoch full \
  --eval_generate_prompt "A futuristic cyberpunk city street during heavy rain at night. Neon signs in pink and cyan reflect on the wet pavement. A lone samurai with a glowing katana stands in the foreground, wearing a high-tech armored trench coat. Towering skyscrapers disappear into the mist above. Cinematic composition, dramatic lighting, ray tracing, highly detailed, photorealistic, 8k resolution, cyberpunk aesthetic, Blade Runner style, sharp focus, volumetric lighting." \
  --eval_generate_height 32 \
  --eval_generate_width 32 \
  --eval_generate_decode \
  --eval_generate_timing none \
  --eval_generate_every_steps "${EVAL_EVERY_STEPS}" \
  --aux_loss_h_weight "${AUX_LOSS_H_WEIGHT}" \
  --aux_loss_v_weight "${AUX_LOSS_V_WEIGHT}" \
  "${FUSE_ARGS[@]}" \
  --eval_generate_outdir ./outputs/nar_finetune \
  "${RESUME_ARGS[@]}" \
  --prefetch_factor=4 --persistent_workers --pin_memory \
  --grad_accum_steps "${GRAD_ACCUM_STEPS}" \
  --eval_generate_seed "${EVAL_GENERATE_SEED}" \
  --lr_scheduler "${LR_SCHEDULER}" \
  --lr_min_factor "${LR_MIN_FACTOR}" \
  --phase2_flat_steps "${PHASE2_FLAT_STEPS}" \
  ${EXTRA_ARGS}
