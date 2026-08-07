# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
from pathlib import Path

import reasoning_gym


DATA_DIR = Path(__file__).resolve().parent / "data"
ANSWER_FORMAT = "Put only the final answer inside <answer></answer> tags or \\boxed{...}."


def format_entry(entry: dict) -> dict:
    return {
        "responses_create_params": {"input": [{"role": "user", "content": f"{entry['question']}\n\n{ANSWER_FORMAT}"}]},
        **entry,
        "agent_ref": {"type": "responses_api_agents", "name": "kiro_reasoning_gym_agent"},
    }


def prepare(task: str, size: int, seed: int, config: dict, output: Path) -> int:
    dataset = reasoning_gym.create_dataset(task, size=size, seed=seed, **config)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as output_file:
        for index in range(size):
            output_file.write(json.dumps(format_entry(dataset[index])) + "\n")
    return size


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="knights_knaves")
    parser.add_argument("--size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--config", type=json.loads, default={})
    parser.add_argument("--output", type=Path, default=DATA_DIR / "train.jsonl")
    args = parser.parse_args()
    if args.size < 1:
        parser.error("--size must be positive")
    count = prepare(args.task, args.size, args.seed, args.config, args.output)
    print(f"Wrote {count} rows to {args.output}")


if __name__ == "__main__":
    main()
