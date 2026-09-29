# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepare SuperGPQA benchmark data for NeMo Gym."""

from __future__ import annotations

import argparse
import json
import random
import uuid
from pathlib import Path

from datasets import load_dataset


BENCHMARK_DIR = Path(__file__).parent
DATA_DIR = BENCHMARK_DIR / "data"
OUTPUT_FPATH = DATA_DIR / "supergpqa_benchmark.jsonl"


def _preprocess(text):
    if text is None:
        return " "
    text = text.strip()
    return text.replace("  ", " ")


def _format_problem(question: str, choices: list[str]) -> str:
    options_text = "\n".join(f"{chr(ord('A') + idx)}: {choice}" for idx, choice in enumerate(choices))
    return f"{question}\n\n{options_text}"


def _to_row(entry: dict, rng: random.Random) -> dict:
    choices = [_preprocess(option) for option in entry["options"]]
    if len(choices) != len(set(choices)):
        raise ValueError(f"Choices are not unique: {choices}")

    answer_letter = entry["answer_letter"].strip().upper()
    answer_idx = ord(answer_letter) - ord("A")
    correct_answer = choices[answer_idx]

    shuffled_choices = list(choices)
    rng.shuffle(shuffled_choices)
    correct_idx = shuffled_choices.index(correct_answer)

    seed = json.dumps(
        {"uuid": entry["uuid"], "split": "test", "choices": shuffled_choices},
        sort_keys=True,
        ensure_ascii=False,
    )
    return {
        "question": entry["question"],
        "problem": _format_problem(entry["question"], shuffled_choices),
        "options": [{chr(ord("A") + idx): choice} for idx, choice in enumerate(shuffled_choices)],
        "expected_answer": chr(ord("A") + correct_idx),
        "uuid": str(uuid.uuid5(uuid.NAMESPACE_URL, seed)),
        "subset_for_metrics": entry["discipline"],
        "discipline": entry["discipline"],
        "field": entry["field"],
        "subfield": entry["subfield"],
        "difficulty": entry["difficulty"],
        "is_calculation": entry["is_calculation"],
    }


def prepare(split: str = "test", random_seed: int = 42) -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if split != "test":
        raise ValueError(f"Invalid split: {split}")

    ds = load_dataset("m-a-p/SuperGPQA")["train"]
    rng = random.Random(random_seed)

    with OUTPUT_FPATH.open("w", encoding="utf-8") as fout:
        for entry in ds:
            fout.write(json.dumps(_to_row(entry, rng), ensure_ascii=False) + "\n")

    print(f"Wrote {len(ds)} problems to {OUTPUT_FPATH}")
    return OUTPUT_FPATH


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test", choices=("test",))
    parser.add_argument("--random_seed", type=int, default=42)
    args = parser.parse_args()
    prepare(split=args.split, random_seed=args.random_seed)
