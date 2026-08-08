# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download


REPO_ID = "nvidia/Nemotron-RL-math-OpenMathReasoning"
DATA_DIR = Path(__file__).resolve().parent / "data"


def prepare(split: str, output: Path, limit: int | None) -> int:
    source = Path(hf_hub_download(repo_id=REPO_ID, filename=f"{split}.jsonl", repo_type="dataset"))
    output.parent.mkdir(parents=True, exist_ok=True)
    if limit is None:
        shutil.copyfile(source, output)
        with output.open() as output_file:
            return sum(1 for _ in output_file)

    count = 0
    with source.open() as input_file, output.open("w") as output_file:
        for line in input_file:
            if count == limit:
                break
            output_file.write(line)
            count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["train", "validation"], default="train")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    output = args.output or DATA_DIR / f"{args.split}.jsonl"
    count = prepare(args.split, output, args.limit)
    print(f"Wrote {count} rows to {output}")


if __name__ == "__main__":
    main()
