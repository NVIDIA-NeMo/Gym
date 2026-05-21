"""Download reasoning_gym training data from HuggingFace.

Usage:
    python environments/reasoning_gym/prepare.py
    python environments/reasoning_gym/prepare.py --split train
"""

import argparse
import json
from pathlib import Path


def prepare(split: str) -> None:
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("pip install datasets")

    output_path = Path(__file__).parent / "data" / f"{split}.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset("nvidia/Nemotron-RL-ReasoningGym-v1", split=split)

    with output_path.open("w") as f:
        for row in dataset:
            f.write(json.dumps(row) + "\n")

    print(f"Wrote {len(dataset)} rows to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="train", choices=["train"])
    args = parser.parse_args()
    prepare(args.split)


if __name__ == "__main__":
    main()
