# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare MMMLU benchmark data for NeMo Gym."""

import argparse
import json
import sys
import uuid
from pathlib import Path


BENCHMARK_DIR = Path(__file__).parent
if str(BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_DIR))

import mmmlu_utils as utils  # noqa: E402


DATA_DIR = BENCHMARK_DIR / "data"
OUTPUT_FPATH = DATA_DIR / "mmmlu_benchmark.jsonl"
LETTER_REGEX = r"\b\(?\s*([A-D]|[أ-د]|[অ]|[ব]|[ড]|[ঢ]|[Ａ]|[Ｂ]|[Ｃ]|[Ｄ])\s*\)?\.?\b"
GREEDY_REGEX = r"[\s\S]*" + LETTER_REGEX


def _output_regexes() -> list[str]:
    regexes = [
        utils.MULTILINGUAL_ANSWER_PATTERN_TEMPLATE.format(answer_regex)
        for answer_regex in utils.MULTILINGUAL_ANSWER_REGEXES
    ]
    regexes.append(GREEDY_REGEX)
    return regexes


def _to_gym_row(entry: dict, language: str) -> dict:
    expected_answer = str(entry[utils.Schema.ANSWER]).strip().upper()
    category = utils.SUBJECT_TO_CATEGORY.get(entry[utils.Schema.SUBJECT], "other")
    mcq_fields = utils.get_mcq_fields(entry)
    question = mcq_fields["question"]
    seed = json.dumps(
        {"language": language, "question": question, "answer": expected_answer},
        sort_keys=True,
        ensure_ascii=False,
    )
    return {
        "question": question,
        "options": [{letter: mcq_fields[letter]} for letter in utils.Schema.OPTIONS],
        "expected_answer": expected_answer,
        "template_metadata": {"output_regex": _output_regexes()},
        "subset_for_metrics": language,
        "category": category,
        "uuid": str(uuid.uuid5(uuid.NAMESPACE_URL, seed)),
    }


def prepare(languages: list[str] | None = None, include_english: bool = False) -> Path:
    if languages is None:
        languages = list(utils.SUPPORTED_LANGUAGES)

    selected_languages = [language for language in languages if language != "EN-US"]
    valid_languages = set(utils.SUPPORTED_LANGUAGES)
    if include_english:
        valid_languages.add("EN-US")
        selected_languages.append("EN-US")

    invalid_languages = set(selected_languages) - valid_languages
    if invalid_languages:
        raise ValueError(f"Unsupported languages: {invalid_languages}")

    datasets = utils.download_mmmlu_datasets(selected_languages)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    count = 0
    with OUTPUT_FPATH.open("w", encoding="utf-8") as fout:
        for language, examples in datasets.items():
            for entry in examples:
                fout.write(json.dumps(_to_gym_row(entry, language), ensure_ascii=False) + "\n")
                count += 1

    print(f"Wrote {count} problems to {OUTPUT_FPATH}")
    return OUTPUT_FPATH


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--languages", default=utils.SUPPORTED_LANGUAGES, nargs="+")
    parser.add_argument("--include_english", action="store_true")
    args = parser.parse_args()
    prepare(languages=args.languages, include_english=args.include_english)
