#!/usr/bin/env python3
"""Run VILLAIN Agent 5 from saved Agent 4 QA outputs.

This supports VJ-supp and E-supp partial cells by reusing a completed
Agents 1-4 evidence-side run and rerunning only Agent 5 with the configured VLM.
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List

import yaml
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agents.base_agent import AgentConfig, SharedModels  # noqa: E402
from agents.pipeline import PipelineResult  # noqa: E402
from agents.qa_generation_agent import QAPair  # noqa: E402
from agents.verdict_agent import VerdictAgent  # noqa: E402
from run_multi_agent import save_per_claim_outputs  # noqa: E402


def load_yaml_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r") as f:
        return yaml.safe_load(f) or {}


def get_nested(d: Dict[str, Any], *keys: str, default=None):
    cur = d
    for key in keys:
        if isinstance(cur, dict):
            cur = cur.get(key, default)
        else:
            return default
    return cur


def qa_pairs_from_agent4(agent4_data: Dict[str, Any]) -> List[QAPair]:
    metadata = agent4_data.get("metadata") or {}
    raw_pairs = metadata.get("qa_pairs") or agent4_data.get("all_qa_pairs") or []
    if not raw_pairs:
        raw_pairs = [
            {"question": q, "answer": a}
            for q, a in zip(agent4_data.get("questions", []), agent4_data.get("answers", []))
        ]
    pairs = []
    for item in raw_pairs:
        q = str(item.get("question", "")).strip()
        a = str(item.get("answer", "")).strip()
        if q and a:
            pairs.append(QAPair(question=q, answer=a))
    return pairs


def main():
    parser = argparse.ArgumentParser(description="Run VILLAIN Agent 5 from saved Agent 4 QA outputs")
    parser.add_argument("--config", required=True)
    parser.add_argument("--input_dir", required=True, help="Run directory containing submission/<id>/agent4_qa_generation.json")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--start_idx", type=int, default=None)
    parser.add_argument("--end_idx", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--vlm_model", default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    cfg = load_yaml_config(args.config)
    data_path = get_nested(cfg, "data", "data_path", default="dataset/AVerImaTeC/val.json")
    image_dir = get_nested(cfg, "data", "image_dir", default="dataset/AVerImaTeC/images")
    target = get_nested(cfg, "data", "target", default="val")
    text_model = get_nested(cfg, "models", "text_model", default="/home/aied_test/models/mxbai-embed-large-v1")
    text_model_type = get_nested(cfg, "models", "text_model_type", default="mxbai")
    image_model = get_nested(cfg, "models", "image_model", default="/home/aied_test/models/Ops-MM-embedding-v1-7B")
    vlm_model = args.vlm_model or get_nested(cfg, "models", "vlm_model", default="/home/aied_test/models/Qwen2.5-VL-7B-Instruct")
    reranker_model = get_nested(cfg, "models", "reranker_model", default="/home/aied_test/models/mxbai-rerank-large-v1")
    device = args.device or get_nested(cfg, "models", "device", default="cuda:0")
    num_qa_to_select = get_nested(cfg, "qa_generation", "num_qa_to_select", default=2)

    with open(data_path, "r") as f:
        all_samples = json.load(f)
    start = 0 if args.start_idx is None else max(0, args.start_idx)
    end = len(all_samples) if args.end_idx is None else min(len(all_samples), args.end_idx)
    samples = all_samples[start:end]

    agent_config = AgentConfig(
        device=device,
        image_dir=image_dir,
        target=target,
        text_model=text_model,
        text_model_type=text_model_type,
        image_model=image_model,
        vlm_model=vlm_model,
        reranker_model=reranker_model,
    )
    print("=" * 50)
    print("VILLAIN Agent5-only Partial Runner")
    print("=" * 50)
    print(f"config={args.config}")
    print(f"input_dir={args.input_dir}")
    print(f"output_dir={args.output_dir}")
    print(f"samples=[{start}:{end}] n={len(samples)}")
    print(f"vlm_model={vlm_model}")
    print(f"device={device}")
    print(f"num_qa_to_select={num_qa_to_select}")

    shared_models = SharedModels(agent_config)
    verdict_agent = VerdictAgent(agent_config, shared_models=shared_models, num_qa_to_select=num_qa_to_select)

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    processed = 0
    skipped = 0
    parse_empty = 0

    for offset, sample in enumerate(tqdm(samples, desc="Agent5 partial claims")):
        claim_id = sample.get("claim_id", sample.get("id", start + offset))
        out_submission = output_dir / "submission" / str(claim_id) / "submission.json"
        if out_submission.exists() and not args.overwrite:
            skipped += 1
            continue

        agent4_path = input_dir / "submission" / str(claim_id) / "agent4_qa_generation.json"
        if not agent4_path.exists():
            raise FileNotFoundError(f"Missing Agent 4 output for claim {claim_id}: {agent4_path}")
        agent4_data = json.load(open(agent4_path))
        qa_pairs = qa_pairs_from_agent4(agent4_data)
        if not qa_pairs:
            parse_empty += 1

        claim_text = sample["claim_text"]
        claim_images = sample.get("claim_images", [])
        speaker = (sample.get("metadata") or {}).get("speaker", "Unknown")
        date = sample.get("date", "Not Specified")
        label = sample.get("label", "")

        verdict_analysis = verdict_agent.analyze(
            claim_id=claim_id,
            claim_text=claim_text,
            claim_images=claim_images,
            qa_pairs=qa_pairs,
            speaker=speaker,
            date=date,
        )
        metadata = verdict_analysis.metadata or {}

        result = PipelineResult(
            claim_id=claim_id,
            claim_text=claim_text,
            claim_images=claim_images,
            label=label,
            speaker=speaker,
            date=date,
        )
        result.all_qa_pairs = [{"question": qa.question, "answer": qa.answer} for qa in qa_pairs]
        result.verdict_analysis = verdict_analysis
        result.questions = verdict_analysis.questions
        result.answers = verdict_analysis.answers
        result.veracity_verdict = metadata.get("veracity_verdict", "")
        result.justification = metadata.get("justification", "")

        save_per_claim_outputs(result, str(output_dir), save_intermediate=True)
        out_dir = output_dir / "submission" / str(claim_id)
        out_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(agent4_path, out_dir / "agent4_qa_generation.json")
        processed += 1
        print(
            "done",
            claim_id,
            "qa_in",
            len(qa_pairs),
            "selected",
            len(result.questions),
            "verdict",
            result.veracity_verdict,
            flush=True,
        )

    print("=" * 50)
    print("Agent5 partial complete")
    print(f"processed={processed} skipped={skipped} empty_agent4_qa={parse_empty}")
    print(f"outputs={output_dir}/submission")


if __name__ == "__main__":
    main()
