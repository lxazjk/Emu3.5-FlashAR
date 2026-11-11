import os.path as osp

from omegaconf import OmegaConf
import torch
from transformers import AutoModel

from .ibq import IBQ


def build_vision_tokenizer(
    type,
    model_path,
    device="cuda:0",
    config_file="config.yaml",
    ckpt_file="model.ckpt",
    **kwargs,
):
    match type:
        case "ibq":
            if osp.exists(model_path):
                cfg_path = osp.join(model_path, config_file)
                ckpt_path = osp.join(model_path, ckpt_file)
                safetensors_path = osp.join(model_path, "model.safetensors")
                if not osp.exists(cfg_path):
                    raise FileNotFoundError(
                        f"config not found under {model_path}: expected {config_file}"
                    )
                cfg = OmegaConf.load(cfg_path)
                tokenizer = IBQ(**cfg)
                if osp.exists(ckpt_path):
                    ckpt = torch.load(ckpt_path, map_location="cpu")
                    if isinstance(ckpt, dict) and "state_dict" in ckpt:
                        ckpt = ckpt["state_dict"]
                elif osp.exists(safetensors_path):
                    from safetensors.torch import load_file
                    ckpt = load_file(safetensors_path)
                else:
                    raise FileNotFoundError(
                        f"ckpt not found under {model_path}: expected {ckpt_file} or model.safetensors"
                    )
                tokenizer.load_state_dict(ckpt)
                tokenizer.eval().to(device)
                return tokenizer

            model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
            model = model.to(device)
            model.eval()
            return model
        case _:
            raise NotImplementedError(f"Unsupported vision tokenizer type: {type}")
