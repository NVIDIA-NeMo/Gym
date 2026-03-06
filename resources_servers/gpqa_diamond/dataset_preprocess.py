# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

BOXED_INSTRUCTION = (
    "You should output your final response letter inside \\boxed{} and nothing else "
    "You can first think step-by-step."
)
CHOICE_LETTERS = ("A", "B", "C", "D")
OPTION_BLOCK_RE = re.compile(r"\n\s*\nA\)\s", re.MULTILINE)
DATA_DIR = Path(__file__).resolve().parent / "data"
RAW_OUTPUT_PATH = DATA_DIR / "diamond_raw.jsonl"
TRAIN_OUTPUT_PATH = DATA_DIR / "train.jsonl"


def _extract_question(problem: str, options_block: Optional[str]) -> str:
    """Extract plain question text from raw GPQA `problem` field."""
    problem = (problem or "").strip()
    options_block = (options_block or "").strip()

    # Preferred path: remove known options suffix if present.
    if options_block and problem.endswith(options_block):
        return problem[: -len(options_block)].rstrip()

    # Fallback path: split at first option marker in the problem text.
    m = OPTION_BLOCK_RE.search(problem)
    if m:
        return problem[: m.start()].rstrip()

    return problem


def _build_row(raw: dict[str, Any], *, uuid: str | None = None) -> dict[str, Any]:
    question = _extract_question(raw.get("problem", ""), raw.get("options"))
    choices = [str(raw.get(letter, "")).strip() for letter in CHOICE_LETTERS]

    if not question:
        raise ValueError("Row missing non-empty `problem` field.")
    if any(not c for c in choices):
        raise ValueError("Row missing one or more answer choices in A/B/C/D fields.")

    prompt = f"{BOXED_INSTRUCTION} {question}\n" + "\n".join(
        f"{letter}: {choice}" for letter, choice in zip(CHOICE_LETTERS, choices)
    )
    options = [{letter: choice} for letter, choice in zip(CHOICE_LETTERS, choices)]
    expected_answer = str(raw.get("expected_answer", "")).strip().upper()

    if expected_answer not in CHOICE_LETTERS:
        raise ValueError(f"Invalid `expected_answer` {expected_answer!r}.")

    metadata = {
        "explanation": raw.get("explanation"),
        "subset_for_metrics": raw.get("subset_for_metrics"),
        "difficulty": raw.get("difficulty"),
    }

    return {
        "responses_create_params": {
            "input": [
                {"role": "user", "content": prompt},
            ]
        },
        "options": options,
        "expected_answer": expected_answer,
        "grading_mode": "strict_single_letter_boxed",
        "metadata": metadata,
        "uuid": uuid or str(uuid4()),
    }


def _normalize_text(text: Optional[str]) -> str:
    if text is None:
        return ""
    return text.strip()


def _build_raw_row_from_hf(entry: dict[str, Any], rng: random.Random) -> dict[str, Any]:
    """Build raw GPQA-style row from Idavidrein/gpqa entry."""
    choices = [
        _normalize_text(entry.get("Incorrect Answer 1")),
        _normalize_text(entry.get("Incorrect Answer 2")),
        _normalize_text(entry.get("Incorrect Answer 3")),
        _normalize_text(entry.get("Correct Answer")),
    ]
    rng.shuffle(choices)

    correct_choice = _normalize_text(entry.get("Correct Answer"))
    correct_index = choices.index(correct_choice)
    letters = CHOICE_LETTERS
    options_text = "\n".join(f"{letters[i]}) {choices[i]}" for i in range(len(letters)))
    difficulty_raw = entry.get("Writer's Difficulty Estimate")
    difficulty = None
    if difficulty_raw is not None:
        difficulty = re.split(r"\s*\(", str(difficulty_raw))[0]

    return {
        "expected_answer": letters[correct_index],
        "explanation": _normalize_text(entry.get("Explanation")),
        "subset_for_metrics": entry.get("Subdomain"),
        "difficulty": difficulty,
        "problem": f"{_normalize_text(entry.get('Question'))}\n\n{options_text}",
        "options": options_text,
        "A": choices[0],
        "B": choices[1],
        "C": choices[2],
        "D": choices[3],
    }


def _download_raw_rows(split: str, random_seed: int) -> list[dict[str, Any]]:
    from datasets import load_dataset

    hf_split = f"gpqa_{split}"
    dataset = load_dataset("Idavidrein/gpqa", hf_split)["train"]
    rng = random.Random(random_seed)
    return [_build_raw_row_from_hf(dict(entry), rng) for entry in dataset]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    raw_rows = _download_raw_rows(split="diamond", random_seed=42)
    _write_jsonl(RAW_OUTPUT_PATH, raw_rows)
    print(f"Downloaded {len(raw_rows)} raw rows to {RAW_OUTPUT_PATH}")
    gym_rows = [_build_row(raw) for raw in raw_rows]
    _write_jsonl(TRAIN_OUTPUT_PATH, gym_rows)
    print(f"Wrote {len(gym_rows)} rows to {TRAIN_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
