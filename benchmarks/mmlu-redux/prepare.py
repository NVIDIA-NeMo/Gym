# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare MMLU-Redux 2.0 benchmark data for NeMo Gym (mcqa).

Ports NeMo-Skills' ``mmlu-redux`` benchmark (``edinburgh-dawg/mmlu-redux-2.0``).
Strategy adapted from ZeroEval / NeMo-Skills ``format_entry`` logic.
"""

import json
import uuid
from pathlib import Path

from datasets import load_dataset
from tqdm.auto import tqdm

from benchmarks.mmlu_subject_categories import MMLU_SUBJECT_TO_CATEGORY, MMLU_SUBJECTS


BENCHMARK_DIR = Path(__file__).parent
DATA_DIR = BENCHMARK_DIR / "data"
OUTPUT_FPATH = DATA_DIR / "mmlu-redux_benchmark.jsonl"

HF_REPO = "edinburgh-dawg/mmlu-redux-2.0"


def format_entry(entry: dict, category: str) -> dict | None:
    if entry["error_type"] == "ok":
        final_answer = chr(65 + int(entry["answer"]))
    elif entry["error_type"] == "wrong_groundtruth" and entry["correct_answer"] in list("ABCD"):
        # NeMo-Skills had a typo ("correct_answer" literal); use the dataset's correct letter.
        final_answer = str(entry["correct_answer"]).strip().upper()
    else:
        return None

    choices = entry["choices"]
    if len(choices) != 4:
        return None
    letters = ["A", "B", "C", "D"]
    options = [{letters[i]: str(choices[i])} for i in range(4)]
    options_text = "\n".join(f"{letters[i]}) {choices[i]}" for i in range(4))
    stem = (entry.get("question") or "").strip()
    seed = json.dumps({"category": category, "question": stem, "answer": final_answer}, sort_keys=True)
    row_uuid = str(uuid.uuid5(uuid.NAMESPACE_URL, seed))
    subset = MMLU_SUBJECT_TO_CATEGORY[category][0]
    return {
        "question": stem,
        "options_text": options_text,
        "options": options,
        "expected_answer": final_answer,
        "subset_for_metrics": subset,
        "subject": category,
        "uuid": row_uuid,
    }


def prepare() -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    count = 0
    with OUTPUT_FPATH.open("w", encoding="utf-8") as fout:
        for category in tqdm(MMLU_SUBJECTS, desc="MMLU-Redux categories"):
            dataset = load_dataset(HF_REPO, name=category, split="test")
            for entry in dataset:
                row = format_entry(entry, category)
                if row is None:
                    continue
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                count += 1

    print(f"Wrote {count} problems to {OUTPUT_FPATH}")
    return OUTPUT_FPATH


if __name__ == "__main__":
    prepare()
