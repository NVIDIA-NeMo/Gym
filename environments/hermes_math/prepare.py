#!/usr/bin/env python3
"""Prepare DAPO-Math-17k for the Hermes agent environment."""

import argparse
import json
from pathlib import Path


AGENT_NAME = "hermes_math_agent"
DATASET_NAME = "YouJiacheng/DAPO-Math-17k-dedup"


def format_entry_to_nemo_gym(example: dict) -> dict:
    return {
        "responses_create_params": {"input": example["prompt"]},
        "question": example["prompt"][0]["content"],
        "expected_answer": example["reward_model"]["ground_truth"],
        "agent_ref": {"type": "responses_api_agents", "name": AGENT_NAME},
    }


def prepare(output: Path, limit: int | None) -> None:
    from datasets import load_dataset

    dataset = load_dataset(DATASET_NAME, split="train")
    count = min(limit, len(dataset)) if limit is not None else len(dataset)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as stream:
        for index in range(count):
            stream.write(json.dumps(format_entry_to_nemo_gym(dataset[index])) + "\n")
    print(f"Wrote {count} DAPO17K rows to {output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent / "data" / "dapo17k_train.jsonl",
    )
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    prepare(args.output, args.limit)


if __name__ == "__main__":
    main()
