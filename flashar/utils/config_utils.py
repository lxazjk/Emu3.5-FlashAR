from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


_COMMENT_KEYS = {"_comment", "__comment__", "comment"}


def _load_json(path: str) -> Dict[str, Any]:
    json_path = Path(path)
    if not json_path.exists():
        raise FileNotFoundError(f"Config JSON not found: {path}")
    with json_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Config JSON must be an object: {path}")
    return payload


def _flatten_dict(node: Dict[str, Any], out: Dict[str, Any]) -> None:
    for key, value in node.items():
        if key in _COMMENT_KEYS:
            continue
        if isinstance(value, dict):
            _flatten_dict(value, out)
            continue
        if key in out:
            raise ValueError(f"Duplicate config key after flattening: {key}")
        out[key] = value


def load_train_arg_defaults(path: str) -> Dict[str, Any]:
    payload = dict(_load_json(path))
    payload.pop("launcher", None)
    flat: Dict[str, Any] = {}
    _flatten_dict(payload, flat)
    return flat


def load_launcher_config(path: str) -> Dict[str, Any]:
    payload = _load_json(path)
    launcher = payload.get("launcher", {})
    if launcher is None:
        launcher = {}
    if not isinstance(launcher, dict):
        raise ValueError("launcher section must be an object")
    return {
        "cuda_visible_devices": str(launcher.get("cuda_visible_devices", "0,1,2,3,4,5,6,7")),
        "pytorch_alloc_conf": str(launcher.get("pytorch_alloc_conf", "expandable_segments:True")),
        "nproc_per_node": int(launcher.get("nproc_per_node", 8)),
        "standalone": bool(launcher.get("standalone", True)),
    }
