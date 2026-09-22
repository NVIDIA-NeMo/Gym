# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare the 5-shot Indic GSM8K text-completion benchmark."""

from __future__ import annotations

import argparse
import json
import random
import uuid
from pathlib import Path

from datasets import load_dataset


BENCHMARK_DIR = Path(__file__).parent
OUTPUT_FPATH = BENCHMARK_DIR / "data" / "indic_gsm8k_benchmark.jsonl"
SOURCE_ID = "anushakamathofficial/indic_GSM8K_v5"
SOURCE_REVISION = "f778a10d5d7f4d976574c7296b2e8ce89c30a38d"
SOURCE_SPLIT = "test"
FEWSHOT_SEED = 42
NUM_FEWSHOT = 5
MAX_OUTPUT_TOKENS = 4096

LANGUAGES = {
    "hi": "Hindi",
    "bn": "Bengali",
    "ur": "Urdu",
    "ta": "Tamil",
    "ne": "Nepali",
    "mr": "Marathi",
    "ml": "Malayalam",
    "te": "Telugu",
    "pa": "Punjabi",
    "gu": "Gujarati",
    "kn": "Kannada",
    "or": "Odia",
    "as": "Assamese",
    "sa": "Sanskrit",
}


def _question(row: dict) -> str:
    question = row.get("question")
    if not isinstance(question, str):
        raise ValueError("Every source row must contain a string question")
    return question.strip()


def _answer(row: dict) -> str:
    answer = row.get("answer")
    if not isinstance(answer, str):
        raise ValueError("Every source row must contain a string answer")
    return answer


def render_prompt(rows: list[dict], row_index: int, rng: random.Random) -> str:
    """Render lm-eval's default same-split 5-shot context for one row."""
    current = rows[row_index]
    sampled = [rows[index] for index in rng.sample(range(len(rows)), NUM_FEWSHOT + 1)]
    demos = [row for row in sampled if row != current][:NUM_FEWSHOT]

    context = "".join(f"Question: {_question(row)}\nAnswer: {_answer(row)}\n\n" for row in demos)
    return f"{context}Question: {_question(current)}\nAnswer:"


def _to_row(source_row: dict, language_code: str, row_index: int, prompt: str) -> dict:
    source_key = f"{SOURCE_ID}@{SOURCE_REVISION}:{language_code}:{SOURCE_SPLIT}:{row_index}"
    return {
        "uuid": str(uuid.uuid5(uuid.NAMESPACE_URL, source_key)),
        "question": _question(source_row),
        "prompt": prompt,
        "expected_answer": _answer(source_row),
        "language": LANGUAGES[language_code],
        "language_code": language_code,
        "subset_for_metrics": language_code,
        "judge_pass_stage": source_row.get("judge_pass_stage"),
        "source_index": row_index,
        "responses_create_params": {
            "max_output_tokens": MAX_OUTPUT_TOKENS,
            "temperature": 0.0,
        },
    }


def prepare(
    languages: list[str] | None = None,
    output_fpath: Path = OUTPUT_FPATH,
) -> Path:
    selected_languages = list(LANGUAGES) if languages is None else languages
    unknown = sorted(set(selected_languages) - LANGUAGES.keys())
    if unknown:
        raise ValueError(f"Unsupported languages: {', '.join(unknown)}")

    output_fpath.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_fpath.open("w", encoding="utf-8") as output_file:
        for language_code in selected_languages:
            dataset = load_dataset(
                SOURCE_ID,
                language_code,
                split=SOURCE_SPLIT,
                revision=SOURCE_REVISION,
                token=True,
            )
            rows = [dict(row) for row in dataset]
            if len(rows) < NUM_FEWSHOT + 1:
                raise ValueError(f"Language {language_code} has fewer than {NUM_FEWSHOT + 1} rows")

            rng = random.Random(FEWSHOT_SEED)
            for row_index, source_row in enumerate(rows):
                prompt = render_prompt(rows, row_index, rng)
                prepared = _to_row(source_row, language_code, row_index, prompt)
                output_file.write(json.dumps(prepared, ensure_ascii=False) + "\n")
                count += 1

    print(f"Wrote {count} problems to {output_fpath}")
    return output_fpath


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--languages", nargs="+", choices=LANGUAGES, default=list(LANGUAGES))
    args = parser.parse_args()
    prepare(args.languages)
