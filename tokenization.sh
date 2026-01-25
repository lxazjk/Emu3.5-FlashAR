  torchrun --standalone --nproc_per_node=8 pretokenize_tar.py \
    --input_dir "./dataset/Infinity-MM/stage1" \
    --output_dir "./dataset/Infinity-MM/stage1_pretok" \
    --split "merge_priority_sample_10M_1" \
    --text_source assistant \
    --vq_path "./weights/Emu3.5-VisionTokenizer" \
    --vq_type ibq \
    --vq_device cpu \
    --image_area 1048576