#!/usr/bin/env python3
"""Generate justification only from a frozen VILLAIN submission."""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict

import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agents.base_agent import AgentConfig, SharedModels


def load_yaml_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r") as f:
        return yaml.safe_load(f) or {}


def get_nested(d: Dict, *keys, default=None):
    for key in keys:
        if isinstance(d, dict):
            d = d.get(key, default)
        else:
            return default
    return d if d is not None else default


def parse_json_object(text: str):
    if "</think>" in text:
        text = text.split("</think>")[-1].strip()
    m = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
    if m:
        text = m.group(1)
    else:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            text = m.group(0)
    return json.loads(re.sub(r",\s*([}\]])", r"\1", text))


def build_agent_config(cfg, device, strength, layers):
    return AgentConfig(
        device=device,
        knowledge_store_path=get_nested(cfg, "stores", "knowledge_store_path", default="dataset/AVerImaTeC_Shared_Task/Knowledge_Store/val"),
        text_related_store_path=get_nested(cfg, "stores", "text_related_store_path", default=""),
        image_related_store_path=get_nested(cfg, "stores", "image_related_store_path", default=""),
        image_embedding_store_path=get_nested(cfg, "stores", "image_embedding_store_path", default=""),
        image_dir=get_nested(cfg, "data", "image_dir", default="dataset/AVerImaTeC/images"),
        target=get_nested(cfg, "data", "target", default="val"),
        text_model=get_nested(cfg, "models", "text_model", default="Qwen/Qwen3-Embedding-8B"),
        text_model_type=get_nested(cfg, "models", "text_model_type", default="qwen"),
        image_model=get_nested(cfg, "models", "image_model", default="OpenSearch-AI/Ops-MM-embedding-v1-7B"),
        vlm_model=get_nested(cfg, "models", "vlm_model", default="Qwen/Qwen2.5-VL-7B-Instruct"),
        vlm_suppression_strength=float(strength),
        vlm_suppression_layers=layers,
        reranker_model=get_nested(cfg, "models", "reranker_model", default="Qwen/Qwen3-Reranker-8B"),
    )


def main():
    ap = argparse.ArgumentParser(description="Run justification-only VILLAIN ablation")
    ap.add_argument("--config", required=True)
    ap.add_argument("--source_submission", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--start_idx", type=int, default=0)
    ap.add_argument("--end_idx", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--suppression_strength", type=float, default=0.5)
    ap.add_argument("--suppression_layers", default="12")
    args = ap.parse_args()

    cfg = load_yaml_config(args.config)
    data_path = get_nested(cfg, "data", "data_path", default="dataset/AVerImaTeC/val.json")
    image_dir = Path(get_nested(cfg, "data", "image_dir", default="dataset/AVerImaTeC/images"))
    device = args.device or get_nested(cfg, "models", "device", default="cuda:0")
    layers = [int(x) for x in args.suppression_layers.split(",") if x.strip()]

    samples = json.loads(Path(data_path).read_text())
    source = json.loads(Path(args.source_submission).read_text())
    source_by_id = {int(r["id"]): r for r in source}
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "submission").mkdir(exist_ok=True)

    agent_config = build_agent_config(cfg, device, args.suppression_strength, layers)
    shared = SharedModels(agent_config)
    start = max(0, args.start_idx)
    end = len(samples) if args.end_idx is None else min(len(samples), args.end_idx)

    schema = {
        "type": "object",
        "properties": {"justification": {"type": "string"}},
        "required": ["justification"],
        "additionalProperties": False,
    }

    print("=" * 50)
    print("VILLAIN justification-only runner")
    print("=" * 50)
    print(f"source_submission={args.source_submission}")
    print(f"output_dir={out}")
    print(f"range=[{start}:{end}]")
    print(f"suppression strength={args.suppression_strength}, layers={layers}")

    processed = skipped = 0
    for idx in range(start, end):
        sample = samples[idx]
        cid = int(sample.get("claim_id", sample.get("id", idx)))
        claim_dir = out / "submission" / str(cid)
        if (claim_dir / "submission.json").exists():
            skipped += 1
            continue
        frozen = dict(source_by_id[cid])
        qa_lines = []
        questions = frozen.get("questions", []) or []
        evidence = frozen.get("evidence", []) or []
        for i, q in enumerate(questions):
            ans = evidence[i].get("text", "") if i < len(evidence) and isinstance(evidence[i], dict) else ""
            qa_lines.append(f"{i+1}. Q: {q}\n   A: {ans}")
        prompt = f"""You are an expert fact-checker. Generate only a concise justification for the fixed verdict using the fixed evidence below.

Claim: {sample['claim_text']}
Fixed verdict: {frozen.get('verdict', 'Not Enough Evidence')}
Fixed question-answer evidence:
{chr(10).join(qa_lines) if qa_lines else 'No evidence.'}

Do not change the verdict. Do not add or remove evidence. Output only JSON:
{{"justification": "..."}}
"""
        content = [{"type": "text", "text": prompt}]
        for img in sample.get("claim_images", []) or []:
            img_path = image_dir / img
            if img_path.exists():
                content.append({"type": "image", "image": str(img_path)})
        print(f"[J-only] Claim {cid}: fixed_q={len(questions)} verdict={frozen.get('verdict')}")
        text = shared.generate_with_vlm(
            [{"role": "user", "content": content}],
            max_new_tokens=1024,
            do_sample=False,
            repetition_penalty=1.05,
            no_repeat_ngram_size=8,
            json_schema=schema,
        )
        try:
            justification = parse_json_object(text).get("justification", "").strip()
        except Exception as exc:
            print(f"[J-only] Parse error claim {cid}: {exc}")
            justification = frozen.get("justification", "")
        frozen["justification"] = justification or frozen.get("justification", "")
        claim_dir.mkdir(parents=True, exist_ok=True)
        (claim_dir / "submission.json").write_text(json.dumps(frozen, indent=2))
        (claim_dir / "justification_only.json").write_text(json.dumps({"id": cid, "raw_response": text, "justification": frozen["justification"]}, indent=2))
        processed += 1

    rows = []
    for d in sorted((out / "submission").iterdir(), key=lambda x: int(x.name) if x.name.isdigit() else 10**9):
        p = d / "submission.json"
        if p.exists():
            rows.append(json.loads(p.read_text()))
    if rows:
        (out / "submission.json").write_text(json.dumps(rows, indent=2))
    print(f"processed={processed} skipped={skipped} total_written={len(rows)}")


if __name__ == "__main__":
    main()
