# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""BigCodeBench dataset prep.

Direct port of NeMo-Skills' ``nemo_skills/dataset/bigcodebench/prepare.py``
with two differences:

  1. We emit Gym-shaped JSONL (``question`` + ``verifier_metadata``) instead
     of Skills' (``question`` + ``task_id`` + ``split``); the test code,
     entry_point, and code_prompt that Skills pulls back from the
     ``bigcodebench`` HF dataset at eval time are written directly into
     ``verifier_metadata`` so the Gym resource server is self-sufficient.
  2. The output filename is ``bigcodebench_benchmark.jsonl`` (Gym
     convention), not ``<split>.jsonl``.

The ``question``-construction logic (extract instruct prefix, wrap
``code_prompt`` in a Python code fence, replace 4-space indents with
tabs) is character-for-character with Skills.
"""

import argparse
from pathlib import Path

import datasets
import orjson


BIGCODEBENCH_VERSION = "v0.1.4"
DATA_DIR = Path(__file__).parent / "data"
DEFAULT_SPLIT = "hard"


def parse_data(split: str = "hard") -> "datasets.Dataset":
    dataset_name = "bigcode/bigcodebench" if split == "full" else "bigcode/bigcodebench-hard"
    return datasets.load_dataset(dataset_name, split=BIGCODEBENCH_VERSION)


def _extract_prefix(text: str, delimiter: str) -> str:
    index = text.find(delimiter)
    assert index != -1, f"delimiter {delimiter!r} not found in instruct_prompt"
    return text[:index].strip()


def _wrap_in_code_tag(text: str) -> str:
    if "```" not in text or "```python" not in text:
        return f"```python\n{text}\n```"
    return text


def _build_question(instruct_prompt: str, code_prompt: str) -> str:
    prefix = _extract_prefix(instruct_prompt, "You should write self-contained code starting with:")
    wrapped = _wrap_in_code_tag(code_prompt)
    return prefix + "\n\n" + "You should write self-contained code starting with:" + "\n" + wrapped


def prepare(output_path: Path = DATA_DIR / "bigcodebench_benchmark.jsonl", split: str = DEFAULT_SPLIT) -> Path:
    dataset = parse_data(split=split)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_written = 0
    with open(output_path, "wb") as f:
        for row in dataset:
            question = _build_question(row["instruct_prompt"], row["code_prompt"])
            # Skills swaps 4-space indents for tabs after question construction
            # because Nemotron-style models prefer tabs. Replicate verbatim.
            question = question.replace("    ", "\t")

            out = {
                "question": question,
                "verifier_metadata": {
                    "task_id": row["task_id"],
                    "test": row["test"],
                    "entry_point": row["entry_point"],
                    "code_prompt": row["code_prompt"],
                    "split": split,
                },
            }
            f.write(orjson.dumps(out) + b"\n")
            n_written += 1

    print(f"Wrote {n_written} problems to {output_path}")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split",
        default=DEFAULT_SPLIT,
        choices=["full", "hard"],
        help="bigcodebench split to download",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DATA_DIR / "bigcodebench_benchmark.jsonl",
    )
    args = parser.parse_args()
    prepare(output_path=args.output_path, split=args.split)
