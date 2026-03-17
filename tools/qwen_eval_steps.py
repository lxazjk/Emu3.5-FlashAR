#!/usr/bin/env python3
import argparse
import base64
import json
import mimetypes
import os
import re
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List

from openai import OpenAI


EVAL_RUBRIC = """
你是图像生成模型评测器。请根据给定 prompt 对单张图片评分，输出严格 JSON：
{
  "overall_score": 0-10 (float),
  "prompt_alignment": 0-10 (float),
  "cyberpunk_scene": 0-10 (float),
  "samurai_katana_foreground": 0-10 (float),
  "neon_rain_reflection": 0-10 (float),
  "composition_lighting": 0-10 (float),
  "photorealism_detail": 0-10 (float),
  "major_issues": ["..."],
  "brief_reason": "..."
}
仅返回 JSON，不要 markdown，不要额外文字。
"""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate generated images with Qwen VL.")
    p.add_argument("--image_dir", required=True)
    p.add_argument("--pattern", default="gen_step*.png")
    p.add_argument("--steps", default="", help="comma-separated, e.g. 10000,10200,10400")
    p.add_argument("--prompt", required=True)
    p.add_argument("--model", default="qwen3.5-plus")
    p.add_argument("--base_url", default="https://dashscope.aliyuncs.com/compatible-mode/v1")
    p.add_argument("--api_key", default=os.getenv("QWEN_API_KEY", ""))
    p.add_argument("--out_json", required=True)
    return p.parse_args()


def image_to_data_url(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    b64 = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def parse_step(path: Path) -> int:
    m = re.search(r"step(\d+)", path.stem)
    return int(m.group(1)) if m else -1


def safe_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return 0.0


def extract_json(text: str) -> Dict[str, Any]:
    s = text.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?", "", s).strip()
        if s.endswith("```"):
            s = s[:-3].strip()
    return json.loads(s)


def call_eval(client: OpenAI, model: str, prompt: str, image_path: Path) -> Dict[str, Any]:
    data_url = image_to_data_url(image_path)
    msg = [
        {"type": "text", "text": f"目标 prompt:\n{prompt}\n\n{EVAL_RUBRIC}"},
        {"type": "image_url", "image_url": {"url": data_url}},
    ]
    resp = client.chat.completions.create(model=model, messages=[{"role": "user", "content": msg}])
    raw = resp.choices[0].message.content or ""
    obj = extract_json(raw)
    obj["raw"] = raw
    return obj


def summarize(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    valid = [x for x in items if "error" not in x]
    if not valid:
        return {"num_selected": len(items), "num_valid": 0}
    by_overall = sorted(valid, key=lambda x: safe_float(x.get("overall_score")), reverse=True)
    keys = [
        "overall_score",
        "prompt_alignment",
        "cyberpunk_scene",
        "samurai_katana_foreground",
        "neon_rain_reflection",
        "composition_lighting",
        "photorealism_detail",
    ]
    averages = {k: round(mean(safe_float(x.get(k, 0.0)) for x in valid), 4) for k in keys}
    return {
        "num_selected": len(items),
        "num_valid": len(valid),
        "avg": averages,
        "top3": by_overall[:3],
        "bottom3": by_overall[-3:],
    }


def main() -> None:
    args = parse_args()
    if not args.api_key:
        raise RuntimeError("Missing API key. Set QWEN_API_KEY or pass --api_key.")

    image_dir = Path(args.image_dir)
    files = sorted(image_dir.glob(args.pattern), key=parse_step)
    if args.steps.strip():
        keep = {int(x.strip()) for x in args.steps.split(",") if x.strip()}
        files = [f for f in files if parse_step(f) in keep]
    if not files:
        raise FileNotFoundError("No images selected.")

    client = OpenAI(api_key=args.api_key, base_url=args.base_url)
    out: List[Dict[str, Any]] = []
    for f in files:
        step = parse_step(f)
        print(f"[EVAL] step={step} file={f.name}", flush=True)
        rec: Dict[str, Any] = {"step": step, "image": str(f)}
        try:
            rec.update(call_eval(client, args.model, args.prompt, f))
        except Exception as e:
            rec["error"] = str(e)
        out.append(rec)

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = summarize(out)
    summary_path = out_path.with_name(out_path.stem + "_summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] wrote {out_path}")
    print(f"[OK] wrote {summary_path}")


if __name__ == "__main__":
    main()
