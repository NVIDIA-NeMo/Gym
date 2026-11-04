#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate a Reasoning Gym dataset for the OpenClaw agent environment."""

import argparse
import json
from pathlib import Path

import reasoning_gym


AGENT_NAME = "openclaw_reasoning_gym_agent"
ANSWER_FORMAT_INSTRUCTION = (
    "Put only your final answer inside <answer></answer> tags or \\boxed{...}. "
    "Do not mention either delimiter elsewhere in your response."
)


def format_entry_to_nemo_gym(entry: dict) -> dict:
    return {
        "responses_create_params": {
            "input": [
                {
                    "role": "user",
                    "content": f"{entry['question']}\n\n{ANSWER_FORMAT_INSTRUCTION}",
                }
            ]
        },
        **entry,
        "agent_ref": {"type": "responses_api_agents", "name": AGENT_NAME},
    }


def prepare(task: str, size: int, seed: int, config: dict, output: Path) -> None:
    dataset = reasoning_gym.create_dataset(task, size=size, seed=seed, **config)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as stream:
        for index in range(size):
            stream.write(json.dumps(format_entry_to_nemo_gym(dataset[index])) + "\n")
    print(f"Wrote {size} {task} rows to {output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="knights_knaves")
    parser.add_argument("--size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--config", default="{}", help="Task-specific JSON configuration")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent / "data" / "train_knights_knaves.jsonl",
    )
    args = parser.parse_args()
    prepare(args.task, args.size, args.seed, json.loads(args.config), args.output)


if __name__ == "__main__":
    main()
