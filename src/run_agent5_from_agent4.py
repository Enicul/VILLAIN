#!/usr/bin/env python3
"""Run VILLAIN Agent 5 from saved Agent 4 QA checkpoints."""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict

import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agents.base_agent import AgentConfig, SharedModels
from agents.qa_generation_agent import QAPair
from agents.verdict_agent import VerdictAgent
from run_multi_agent import agent_analysis_to_dict


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


def load_agent4_pairs(path: Path):
    data = json.loads(path.read_text())
    candidates = []
    if isinstance(data.get("metadata"), dict):
        candidates.extend([
            data["metadata"].get("qa_pairs"),
            data["metadata"].get("all_qa_pairs"),
        ])
    candidates.extend([data.get("all_qa_pairs"), data.get("qa_pairs")])
    for value in candidates:
        if isinstance(value, list):
            pairs = []
            for item in value:
                if not isinstance(item, dict):
                    continue
                q = item.get("question", "")
                a = item.get("answer", "")
                if q and a:
                    pairs.append(QAPair(question=q, answer=a))
            return pairs
    return []


def save_outputs(out_dir: Path, claim_id: int, source_agent4: Path, analysis, qa_pairs):
    claim_dir = out_dir / "submission" / str(claim_id)
    claim_dir.mkdir(parents=True, exist_ok=True)
    if source_agent4.exists():
        shutil.copy2(source_agent4, claim_dir / "agent4_qa_generation.json")

    metadata = analysis.metadata or {}
    questions = analysis.questions or []
    answers = analysis.answers or []
    verdict = metadata.get("veracity_verdict", "Not Enough Evidence")
    justification = metadata.get("justification", "")

    submission = {
        "id": claim_id,
        "questions": questions,
        "justification": justification,
        "verdict": verdict,
        "evidence": [{"text": a, "images": []} for a in answers],
    }
    (claim_dir / "submission.json").write_text(json.dumps(submission, indent=2))

    agent5_output = agent_analysis_to_dict(analysis)
    agent5_output["veracity_verdict"] = verdict
    agent5_output["justification"] = justification
    agent5_output["selected_questions"] = questions
    agent5_output["selected_answers"] = answers
    agent5_output["source_qa_count"] = len(qa_pairs)
    (claim_dir / "agent5_verdict.json").write_text(json.dumps(agent5_output, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Run Agent5 from saved Agent4 QA outputs")
    parser.add_argument("--config", required=True)
    parser.add_argument("--source_output_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--start_idx", type=int, default=None)
    parser.add_argument("--end_idx", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--suppression_strength", type=float, default=None)
    parser.add_argument("--suppression_layers", default=None, help="Comma-separated layer ids")
    args = parser.parse_args()

    cfg = load_yaml_config(args.config)
    data_path = get_nested(cfg, "data", "data_path", default="dataset/AVerImaTeC/val.json")
    image_dir = get_nested(cfg, "data", "image_dir", default="dataset/AVerImaTeC/images")
    target = get_nested(cfg, "data", "target", default="val")
    device = args.device or get_nested(cfg, "models", "device", default="cuda:0")
    strength = args.suppression_strength
    if strength is None:
        strength = get_nested(cfg, "models", "vlm_suppression_strength", default=1.0)
    if args.suppression_layers is not None:
        layers = [int(x) for x in args.suppression_layers.split(",") if x.strip()]
    else:
        layers = get_nested(cfg, "models", "vlm_suppression_layers", default=[])

    agent_config = AgentConfig(
        device=device,
        knowledge_store_path=get_nested(cfg, "stores", "knowledge_store_path", default="dataset/AVerImaTeC_Shared_Task/Knowledge_Store/val"),
        text_related_store_path=get_nested(cfg, "stores", "text_related_store_path", default=""),
        image_related_store_path=get_nested(cfg, "stores", "image_related_store_path", default=""),
        image_embedding_store_path=get_nested(cfg, "stores", "image_embedding_store_path", default=""),
        image_dir=image_dir,
        target=target,
        text_model=get_nested(cfg, "models", "text_model", default="Qwen/Qwen3-Embedding-8B"),
        text_model_type=get_nested(cfg, "models", "text_model_type", default="qwen"),
        image_model=get_nested(cfg, "models", "image_model", default="OpenSearch-AI/Ops-MM-embedding-v1-7B"),
        vlm_model=get_nested(cfg, "models", "vlm_model", default="Qwen/Qwen2.5-VL-7B-Instruct"),
        vlm_suppression_strength=float(strength),
        vlm_suppression_layers=layers,
        reranker_model=get_nested(cfg, "models", "reranker_model", default="Qwen/Qwen3-Reranker-8B"),
    )
    num_qa_to_select = get_nested(cfg, "qa_generation", "num_qa_to_select", default=10)

    samples = json.loads(Path(data_path).read_text())
    start = 0 if args.start_idx is None else max(0, args.start_idx)
    end = len(samples) if args.end_idx is None else min(len(samples), args.end_idx)
    source_dir = Path(args.source_output_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 50)
    print("VILLAIN Agent5-only runner")
    print("=" * 50)
    print(f"source_output_dir={source_dir}")
    print(f"output_dir={out_dir}")
    print(f"range=[{start}:{end}]")
    print(f"vlm_model={agent_config.vlm_model}")
    print(f"suppression strength={agent_config.vlm_suppression_strength}, layers={agent_config.vlm_suppression_layers}")
    print(f"device={agent_config.device}")

    shared = SharedModels(agent_config)
    agent = VerdictAgent(agent_config, shared_models=shared, num_qa_to_select=num_qa_to_select)
    processed = 0
    skipped = 0
    missing = []

    for idx in range(start, end):
        sample = samples[idx]
        claim_id = sample.get("claim_id", sample.get("id", idx))
        claim_dir = out_dir / "submission" / str(claim_id)
        if (claim_dir / "submission.json").exists():
            skipped += 1
            continue
        source_agent4 = source_dir / "submission" / str(claim_id) / "agent4_qa_generation.json"
        if not source_agent4.exists():
            missing.append(claim_id)
            print(f"[Agent5-only] Missing Agent4 checkpoint for claim {claim_id}: {source_agent4}")
            continue
        qa_pairs = load_agent4_pairs(source_agent4)
        print(f"[Agent5-only] Claim {claim_id}: {len(qa_pairs)} QA pairs")
        metadata = sample.get("metadata", {}) or {}
        analysis = agent.analyze(
            claim_id=claim_id,
            claim_text=sample["claim_text"],
            claim_images=sample.get("claim_images", []),
            qa_pairs=qa_pairs,
            speaker=metadata.get("speaker", sample.get("speaker", "Unknown")),
            date=sample.get("date", "Not Specified"),
        )
        save_outputs(out_dir, claim_id, source_agent4, analysis, qa_pairs)
        processed += 1

    print(f"processed={processed} skipped={skipped} missing={len(missing)}")
    if missing:
        print("missing_claim_ids=" + ",".join(map(str, missing)))


if __name__ == "__main__":
    main()
