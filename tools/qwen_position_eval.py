#!/usr/bin/env python3
import argparse
import base64
import json
import mimetypes
from pathlib import Path
from typing import Any, Dict, List

from openai import OpenAI


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--image_dir", required=True)
    p.add_argument("--steps", required=True, help="comma-separated steps")
    p.add_argument("--api_key", required=True)
    p.add_argument("--base_url", default="https://dashscope.aliyuncs.com/compatible-mode/v1")
    p.add_argument("--model", default="qwen3.5-plus")
    p.add_argument("--out_json", required=True)
    return p.parse_args()


def image_to_data_url(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    b64 = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def parse_json(text: str) -> Dict[str, Any]:
    s = (text or "").strip()
    if s.startswith("```"):
        s = s.replace("```json", "").replace("```", "").strip()
    return json.loads(s)


def main() -> None:
    args = parse_args()
    steps: List[int] = [int(x.strip()) for x in args.steps.split(",") if x.strip()]
    client = OpenAI(api_key=args.api_key, base_url=args.base_url)

    prompt = (
        "你是图像构图分析器。判断图中最主要人物(主角)的水平位置。"
        "严格输出JSON："
        '{"position":"left|center|right","x_ratio":0.0,"confidence":0.0,"reason":"..."}'
        "其中x_ratio是主角中心点在图像宽度上的相对位置(0最左,1最右)。"
        "只输出JSON。"
    )

    out: List[Dict[str, Any]] = []
    for step in steps:
        path = Path(args.image_dir) / f"gen_step{step}.png"
        msg = [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": image_to_data_url(path)}},
        ]
        rsp = client.chat.completions.create(
            model=args.model,
            messages=[{"role": "user", "content": msg}],
        )
        raw = rsp.choices[0].message.content or ""
        rec = parse_json(raw)
        rec["step"] = step
        rec["image"] = str(path)
        out.append(rec)
        print(f"[OK] step={step} position={rec.get('position')} x={rec.get('x_ratio')}", flush=True)

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] wrote {out_path}")


if __name__ == "__main__":
    main()
