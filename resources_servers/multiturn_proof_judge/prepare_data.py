#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Convert raw proof JSONL ({"problem": "..."}) into Gym-compatible format.

The output JSONL has the structure expected by NeMo Gym / nemo-rl:
  {
    "agent_ref": {"name": "<agent_name>"},
    "responses_create_params": {"input": [{"role": "user", "content": "<raw problem>"}]},
    "problem": "<original problem text>"
  }

The initial prover prompt template is applied at runtime by the judge's
seed_session endpoint, keeping all prompt logic in one place.

Usage:
    python prepare_data.py \
        --input /path/to/raw_problems.jsonl \
        --output data/train.jsonl
"""
import argparse
import json

DEFAULT_AGENT_NAME = "multiturn_proof_agent"


def convert_proof_jsonl(
    input_path: str,
    output_path: str,
    problem_field: str = "problem",
    agent_name: str = DEFAULT_AGENT_NAME,
) -> int:
    count = 0
    with open(input_path) as fin, open(output_path, "w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            problem = row[problem_field]
            gym_example = {
                "agent_ref": {"name": agent_name},
                "responses_create_params": {
                    "input": [{"role": "user", "content": problem}],
                },
                "problem": problem,
            }
            fout.write(json.dumps(gym_example, ensure_ascii=False) + "\n")
            count += 1
    return count


def main():
    parser = argparse.ArgumentParser(description="Convert raw proof JSONL to Gym-compatible format")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--problem-field", default="problem")
    parser.add_argument("--agent-name", default=DEFAULT_AGENT_NAME)
    args = parser.parse_args()

    count = convert_proof_jsonl(args.input, args.output, args.problem_field, args.agent_name)
    print(f"Converted {count} examples: {args.input} -> {args.output}")


if __name__ == "__main__":
    main()
