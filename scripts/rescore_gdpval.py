#!/usr/bin/env python3
# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Re-score GDPVal results offline with a different judge model.

Reads an existing results JSONL, extracts deliverable text + rubric from
each sample, and re-scores sequentially (one at a time) to avoid rate limits.

Usage:
    python scripts/rescore_gdpval.py \
        --input results/.../0_selfjudge.jsonl \
        --output results/.../0_qwen397b_rescore.jsonl \
        --judge-model gpt-4.1-2025-04-14 \
        --judge-base-url https://api.openai.com/v1 \
        --judge-api-key <your-key> \
        --judge-prompt-template responses_api_agents/stirrup_agent/prompts/judge_prompt.j2 \
        --concurrency 4
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

# Add repo root to path and import scoring module directly to avoid Ray dependency
repo_root = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, repo_root)

import importlib.util

_spec = importlib.util.spec_from_file_location(
    "_gdpval_scoring",
    Path(repo_root) / "responses_api_agents" / "stirrup_agent" / "tasks" / "_gdpval_scoring.py",
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
score_with_rubric = _mod.score_with_rubric


def extract_deliverable_from_output(sample: dict) -> str:
    """Reconstruct deliverable text from output items."""
    parts: list[str] = []
    output_items = sample["response"]["output"]

    # Find the finish function call reason
    for item in output_items:
        if item.get("type") == "function_call" and item.get("name") == "finish":
            try:
                args = json.loads(item["arguments"])
                reason = args.get("reason", "")
                if reason:
                    parts.append(reason)
            except (json.JSONDecodeError, TypeError):
                pass

    # Find last assistant message
    for item in reversed(output_items):
        if item.get("role") == "assistant" and item.get("type") == "message":
            content = item.get("content", [])
            if isinstance(content, list):
                for c in content:
                    if isinstance(c, dict) and c.get("type") == "output_text":
                        text = c.get("text", "").strip()
                        if text and text not in parts:
                            parts.append(text)
                            break
            break

    return "\n\n".join(parts) if parts else ""


async def rescore_sample(
    sample: dict,
    judge_model: str,
    judge_base_url: str,
    judge_api_key: str,
    judge_prompt_template: str,
    semaphore: asyncio.Semaphore,
) -> dict:
    """Re-score a single sample and return updated copy."""
    meta = sample["responses_create_params"].get("metadata", {})
    task_id = meta.get("task_id", "unknown")

    deliverable_text = extract_deliverable_from_output(sample)
    if not deliverable_text:
        print(f"[{task_id}] No deliverable text found, score=0.0", flush=True)
        result = dict(sample)
        result["reward"] = 0.0
        return result

    rubric_json = meta.get("rubric_json", "[]")
    rubric_pretty = meta.get("rubric_pretty", "")
    task_prompt = meta.get("prompt", "")

    async with semaphore:
        score = await score_with_rubric(
            deliverable_text=deliverable_text,
            rubric_json=rubric_json,
            rubric_pretty=rubric_pretty,
            task_prompt=task_prompt,
            judge_prompt_template=judge_prompt_template,
            model_base_url=judge_base_url,
            model_name=judge_model,
            api_key=judge_api_key,
        )

    print(f"[{task_id}] score={score:.4f}", flush=True)
    result = dict(sample)
    result["reward"] = score
    return result


async def main():
    parser = argparse.ArgumentParser(description="Re-score GDPVal results with a different judge")
    parser.add_argument("--input", required=True, help="Input JSONL file")
    parser.add_argument("--output", required=True, help="Output JSONL file")
    parser.add_argument("--judge-model", required=True)
    parser.add_argument("--judge-base-url", required=True)
    parser.add_argument("--judge-api-key", required=True)
    parser.add_argument("--judge-prompt-template", required=True)
    parser.add_argument("--concurrency", type=int, default=4, help="Max concurrent scoring requests")
    args = parser.parse_args()

    samples = []
    with open(args.input) as f:
        for line in f:
            samples.append(json.loads(line))
    print(f"Loaded {len(samples)} samples from {args.input}", flush=True)

    # Resume from partial output if it exists
    done_ids = set()
    output_path = Path(args.output)
    if output_path.exists():
        with open(output_path) as f:
            for line in f:
                d = json.loads(line)
                tid = d.get("responses_create_params", {}).get("metadata", {}).get("task_id")
                if tid:
                    done_ids.add(tid)
        print(f"Resuming: {len(done_ids)} already scored, {len(samples) - len(done_ids)} remaining", flush=True)

    remaining = []
    for s in samples:
        tid = s.get("responses_create_params", {}).get("metadata", {}).get("task_id")
        if tid not in done_ids:
            remaining.append(s)

    # Write results incrementally
    semaphore = asyncio.Semaphore(args.concurrency)
    lock = asyncio.Lock()
    rewards_all = []

    async def score_and_write(sample):
        result = await rescore_sample(
            sample, args.judge_model, args.judge_base_url, args.judge_api_key, args.judge_prompt_template, semaphore
        )
        async with lock:
            with open(args.output, "a") as f:
                f.write(json.dumps(result) + "\n")
            rewards_all.append(result["reward"])
        return result

    tasks = [score_and_write(s) for s in remaining]
    await asyncio.gather(*tasks)

    # Final stats (read full file for accurate count including resumed)
    all_rewards = []
    with open(args.output) as f:
        for line in f:
            all_rewards.append(json.loads(line)["reward"])
    zeros = sum(1 for r in all_rewards if r == 0.0)
    print(f"\nDone! {len(all_rewards)} samples scored.", flush=True)
    print(f"Mean reward: {sum(all_rewards)/len(all_rewards):.4f}", flush=True)
    print(f"Zeros: {zeros}/{len(all_rewards)}", flush=True)
    print(f"Written to {args.output}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
