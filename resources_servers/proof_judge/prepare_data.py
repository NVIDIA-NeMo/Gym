#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Convert raw proof JSONL ({"problem": "..."}) into Gym-compatible format.

The output JSONL has the structure expected by NeMo Gym / nemo-rl:
    {
        "agent_ref": {"name": "<agent>"},
        "responses_create_params": {"input": [{"role": "user", "content": "<prompt>"}]},
        "problem": "<raw problem text>"
    }

agent_ref is required by both Gym's rollout_collection (for routing to the correct agent)
and nemo-rl's rollouts.py (for per-agent metrics). The name must match the top-level
YAML key of the agent config in proof_judge.yaml.

Usage:
    python prepare_data.py \
        --input /path/to/raw_problems.jsonl \
        --output data/train.jsonl

    python prepare_data.py \
        --input /path/to/raw_problems_val.jsonl \
        --output data/validation.jsonl
"""
import argparse
import json
from pathlib import Path

import yaml

PROMPT_TEMPLATES_DIR = Path(__file__).parent / "prompt_templates"


def _load_prompt_template(filename: str) -> str:
    """Load the 'user' field from a prompt YAML file in prompt_templates/."""
    with open(PROMPT_TEMPLATES_DIR / filename) as f:
        return yaml.safe_load(f)["user"]


PROVER_PROMPT_TEMPLATE = _load_prompt_template("prover.yaml")


DEFAULT_AGENT_NAME = "proof_simple_agent"


def convert_proof_jsonl(
    input_path: str,
    output_path: str,
    problem_field: str = "problem",
    agent_name: str = DEFAULT_AGENT_NAME,
) -> int:
    """Convert raw proof JSONL to Gym-compatible format.

    Returns the number of examples written.
    """
    count = 0
    with open(input_path) as fin, open(output_path, "w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            problem = row[problem_field]
            user_content = PROVER_PROMPT_TEMPLATE.format(problem=problem)
            gym_example = {
                "agent_ref": {"name": agent_name},
                "responses_create_params": {
                    "input": [{"role": "user", "content": user_content}],
                },
                "problem": problem,
            }
            fout.write(json.dumps(gym_example, ensure_ascii=False) + "\n")
            count += 1
    return count


def main():
    parser = argparse.ArgumentParser(description="Convert raw proof JSONL to Gym-compatible format")
    parser.add_argument("--input", required=True, help="Path to input JSONL (raw proof problems)")
    parser.add_argument("--output", required=True, help="Path to output JSONL (Gym-compatible)")
    parser.add_argument(
        "--problem-field", default="problem",
        help="JSON field name containing the problem text (default: 'problem')",
    )
    parser.add_argument(
        "--agent-name", default=DEFAULT_AGENT_NAME,
        help=f"Agent name for agent_ref routing (default: '{DEFAULT_AGENT_NAME}'). "
        "Must match the top-level YAML key in proof_judge.yaml.",
    )
    args = parser.parse_args()

    count = convert_proof_jsonl(args.input, args.output, args.problem_field, args.agent_name)
    print(f"Converted {count} examples: {args.input} -> {args.output}")
    print(f"Agent ref: {args.agent_name}")


if __name__ == "__main__":
    main()
