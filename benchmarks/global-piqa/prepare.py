# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepare Global-PIQA benchmark data for NeMo Gym."""

from __future__ import annotations

import argparse
import json
import uuid
from pathlib import Path

from datasets import get_dataset_config_names, load_dataset


BENCHMARK_DIR = Path(__file__).parent
DATA_DIR = BENCHMARK_DIR / "data"
OUTPUT_FPATH = DATA_DIR / "global-piqa_benchmark.jsonl"
HF_REPO_ID = "mrlbenchmarks/global-piqa-nonparallel"

EXTRACT_REGEX = [
    r"(?i)[Tt]he (?:[Bb]est [Aa]nswer|[Ff]inal [Aa]nswer|[Aa]nswer)[^A-B]*([A-B])",
    r"(?i)[Aa]nswer\s*:[^A-B]*([A-B])",
    r"(?i)\\boxed\{([A-B])\}",
    r"[\s\S]*\b\(?\s*([A-B])\s*\)?\.?\b",
]


def supported_languages() -> list[str]:
    return list(get_dataset_config_names(HF_REPO_ID))


def _digit_to_letter(digit: int) -> str:
    return chr(ord("A") + digit)


def _to_row(entry: dict, language: str) -> dict:
    question = entry["prompt"].strip()
    option_a = entry["solution0"].strip()
    option_b = entry["solution1"].strip()
    seed = json.dumps(
        {
            "language": language,
            "prompt": question,
            "solution0": option_a,
            "solution1": option_b,
            "label": entry["label"],
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    return {
        "question": question,
        "problem": question,
        "A": option_a,
        "B": option_b,
        "options": [{"A": option_a}, {"B": option_b}],
        "expected_answer": _digit_to_letter(int(entry["label"])),
        "template_metadata": {"output_regex": EXTRACT_REGEX},
        "subset_for_metrics": language,
        "target_language": language,
        "uuid": str(uuid.uuid5(uuid.NAMESPACE_URL, seed)),
    }


def prepare(languages: list[str] | None = None) -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if languages is None:
        languages = supported_languages()

    count = 0
    with OUTPUT_FPATH.open("w", encoding="utf-8") as fout:
        for language in languages:
            ds = load_dataset(HF_REPO_ID, language, split="test")
            for entry in ds:
                fout.write(json.dumps(_to_row(entry, language), ensure_ascii=False) + "\n")
                count += 1

    print(f"Wrote {count} problems to {OUTPUT_FPATH}")
    return OUTPUT_FPATH


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--languages", nargs="+", default=supported_languages())
    args = parser.parse_args()
    prepare(languages=args.languages)
