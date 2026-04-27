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

QUERY_TEMPLATE = """
Given the following situation, which option is more likely to be correct?

Situation:
{prompt} ...

Option A: {solution0}

Option B: {solution1}

Your response should end with "The best answer is: [answer_letter]" where [answer_letter] is one of A or B.
""".strip()

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


def _question_text(entry: dict) -> str:
    return QUERY_TEMPLATE.format(**entry)


def _to_row(entry: dict, language: str) -> dict:
    question = _question_text(entry)
    seed = json.dumps(
        {
            "language": language,
            "prompt": entry["prompt"],
            "solution0": entry["solution0"],
            "solution1": entry["solution1"],
            "label": entry["label"],
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    return {
        "question": question,
        "problem": question,
        "options": [{"A": entry["solution0"]}, {"B": entry["solution1"]}],
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
