  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  PYTORCH_ALLOC_CONF=expandable_segments:True \
  torchrun --standalone --nproc_per_node=8 emu_nar/finetune.py \
    --model_path "./weights/Emu3.5-Image" \
    --tokenizer_path "./src/tokenizer_emu3_ibq" \
    --vq_path "./weights/Emu3.5-VisionTokenizer" \
    --text_template "<|extra_203|>You are a helpful assistant for t2i task. USER: {text} ASSISTANT: <|extra_100|>" \
    --pretok_glob "/opt/tiger/Emu3.5-NAR/GPT4o-Image_pretok_32/text_to_image_partfirst/*.tar" \
    --add_boi \
    --fsdp --fsdp_wrap_policy transformer \
    --text_max_length 128 \
    --use_vertical_block \
    --vertical_layers 4 \
    --epochs 5 \
    --lora_layers 32 --lora_r_min 2 --lora_r 8 --lora_alpha 12 \
    --lora_dropout 0.05 \
    --gradient_checkpointing \
    --save_epoch full \
    --eval_generate_prompt "A futuristic cyberpunk city street during heavy rain at night. Neon signs in pink and cyan reflect on the wet pavement. A lone samurai with a glowing katana stands in the foreground, wearing a high-tech armored trench coat. Towering skyscrapers disappear into the mist above. Cinematic composition, dramatic lighting, ray tracing, highly detailed, photorealistic, 8k resolution, cyberpunk aesthetic, Blade Runner style, sharp focus, volumetric lighting." \
    --eval_generate_height 32 \
    --eval_generate_width 32 \
    --eval_generate_decode \
    --eval_generate_timing start \
    --eval_generate_outdir ./outputs/nar_finetune \
    --resume_path ./outputs/nar_finetune/nar_epoch3.full.pt \
    --prefetch_factor=4 --persistent_workers --pin_memory \
    --grad_accum_steps 2