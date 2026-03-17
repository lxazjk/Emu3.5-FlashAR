import os
from pathlib import Path

from src.utils.logging_utils import setup_logger

cfg_name = Path(__file__).stem


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value is not None and value != "" else default


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return float(value) if value is not None and value != "" else default


def _env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return value.lower() not in {"0", "false", "no", "off"}


model_path = os.getenv("EMU35_IMAGE_MODEL_PATH", "./weights/Emu3.5-Image")
vq_path = os.getenv("EMU35_VQ_PATH", "./weights/Emu3.5-VisionTokenizer")
nar_ckpt_path = os.getenv(
    "EMU35_BENCH_NAR_CKPT",
    "./outputs/nar_finetune_w1200_p235_gatefix_v010_gc010/nar_epoch_latest.full.pt",
)
nar_use_vertical_block = True
nar_vertical_layers = _env_int("EMU35_NAR_VERTICAL_LAYERS", 2)
nar_attn_implementation = os.getenv("EMU35_NAR_ATTN_IMPL", "eager")
nar_merge_dtype = os.getenv("EMU35_NAR_MERGE_DTYPE", "bf16")
nar_fsdp_wrap_policy = os.getenv("EMU35_NAR_FSDP_WRAP_POLICY", "transformer")
nar_fsdp_min_params = _env_int("EMU35_NAR_FSDP_MIN_PARAMS", 1_000_000)
nar_use_kv_cache = _env_flag("EMU35_NAR_USE_KV_CACHE", True)

tokenizer_path = "./src/tokenizer_emu3_ibq"
vq_type = "ibq"

task_type = "t2i"
use_image = False

exp_name = os.getenv("EMU35_BENCH_EXP_NAME", "emu3p5_benchmark_nar")
save_path = os.getenv("EMU35_BENCH_SAVE_PATH", f"./outputs/{exp_name}")
save_to_proto = True
setup_logger(save_path)

hf_device = os.getenv("EMU35_HF_DEVICE", "cuda:0")
vq_device = os.getenv("EMU35_VQ_DEVICE", "cuda:0")
streaming = False
unconditional_type = "no_text"
classifier_free_guidance = _env_float("EMU35_CFG_SCALE", 2.5)
max_new_tokens = _env_int("EMU35_MAX_NEW_TOKENS", 5120)
image_area = 1048576

aspect_ratios = {
    "4:3": "55*73",
    "21:9": "41*97",
    "16:9": "47*85",
    "3:2": "52*78",
    "1:1": "64*64",
    "3:4": "73*55",
    "9:16": "85*47",
    "2:3": "78*52",
    "default": "32*32",
    "auto": None,
}


def get_target_size(aspect_ratio: str):
    value = aspect_ratios.get(aspect_ratio, None)
    if value is None:
        return None, None
    h, w = map(int, value.split("*"))
    return h, w


aspect_ratio = os.getenv("EMU35_ASPECT_RATIO", "default")
target_height, target_width = get_target_size(aspect_ratio)


def build_unc_and_template(task: str, with_image: bool):
    task_str = task.lower()
    if with_image:
        unc_p = "<|extra_203|>You are a helpful assistant. USER: <|IMAGE|> ASSISTANT: <|extra_100|>"
        tmpl = "<|extra_203|>You are a helpful assistant for %s task. USER: {question}<|IMAGE|> ASSISTANT: <|extra_100|>" % task_str
    else:
        unc_p = "<|extra_203|>You are a helpful assistant. USER:  ASSISTANT: <|extra_100|>"
        tmpl = "<|extra_203|>You are a helpful assistant for %s task. USER: {question} ASSISTANT: <|extra_100|>" % task_str
    return unc_p, tmpl


unc_prompt, template = build_unc_and_template(task_type, use_image)

sampling_params = dict(
    use_cache=True,
    text_top_k=1024,
    text_top_p=0.9,
    text_temperature=1.0,
    image_top_k=5120,
    image_top_p=1.0,
    image_temperature=1.0,
    top_k=131072,
    top_p=1.0,
    temperature=1.0,
    num_beams_per_group=1,
    num_beam_groups=1,
    diversity_penalty=0.0,
    max_new_tokens=max_new_tokens,
    guidance_scale=1.0,
    use_differential_sampling=True,
)

sampling_params["do_sample"] = sampling_params["num_beam_groups"] <= 1
sampling_params["num_beams"] = sampling_params["num_beams_per_group"] * sampling_params["num_beam_groups"]


special_tokens = dict(
    BOS="<|extra_203|>",
    EOS="<|extra_204|>",
    PAD="<|endoftext|>",
    EOL="<|extra_200|>",
    EOF="<|extra_201|>",
    TMS="<|extra_202|>",
    IMG="<|image token|>",
    BOI="<|image start|>",
    EOI="<|image end|>",
    BSS="<|extra_100|>",
    ESS="<|extra_101|>",
    BOG="<|extra_60|>",
    EOG="<|extra_61|>",
    BOC="<|extra_50|>",
    EOC="<|extra_51|>",
)

seed = _env_int("EMU35_SEED", 6666)

_prompts_base = [
    {
        "prompt": os.getenv(
            "EMU35_BENCH_PROMPT",
            "A futuristic cyberpunk city street during heavy rain at night. Neon signs in pink and cyan reflect on the wet pavement. A lone samurai with a glowing katana stands in the foreground, wearing a high-tech armored trench coat. Towering skyscrapers disappear into the mist above. Cinematic composition, dramatic lighting, ray tracing, highly detailed, photorealistic, 8k resolution, cyberpunk aesthetic, Blade Runner style, sharp focus, volumetric lighting.",
        ),
    },
]

if use_image:
    prompts = _prompts_base
else:
    prompts = [p["prompt"] for p in _prompts_base]
