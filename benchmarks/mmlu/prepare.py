# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare MMLU (Hendrycks) benchmark data for NeMo Gym (mcqa).

Ports NeMo-Skills' ``mmlu`` benchmark: loads per-subject test splits from HuggingFace
and writes Gym JSONL rows compatible with ``mcqa`` + ``lenient_answer_colon_md`` grading.
"""

import json
import uuid
from pathlib import Path

from datasets import load_dataset
from tqdm.auto import tqdm

from benchmarks.mmlu_subject_categories import MMLU_SUBJECT_TO_CATEGORY, MMLU_SUBJECTS


BENCHMARK_DIR = Path(__file__).parent
DATA_DIR = BENCHMARK_DIR / "data"
OUTPUT_FPATH = DATA_DIR / "mmlu_benchmark.jsonl"

HF_CANDIDATES = ("lukaemon/mmlu", "cais/mmlu")


def _load_mmlu_subject(repo_id: str, subject: str):
    return load_dataset(repo_id, subject, split="test")


def _find_hf_repo() -> str:
    last_err: Exception | None = None
    for repo in HF_CANDIDATES:
        try:
            _load_mmlu_subject(repo, MMLU_SUBJECTS[0])
            return repo
        except Exception as e:  # noqa: BLE001 — try next mirror
            last_err = e
    raise RuntimeError(f"Could not load MMLU from {HF_CANDIDATES}: {last_err}")


def _row_from_example(subject: str, ex: dict, letter: str) -> dict:
    choices = ex["choices"]
    if len(choices) != 4:
        raise ValueError(f"Expected 4 choices for {subject}, got {len(choices)}")
    letters = ["A", "B", "C", "D"]
    options = [{letters[i]: choices[i]} for i in range(4)]
    options_text = "\n".join(f"{letters[i]}) {choices[i]}" for i in range(4))
    stem = (ex.get("question") or "").strip()
    seed = json.dumps({"subject": subject, "question": stem, "answer": letter}, sort_keys=True)
    row_uuid = str(uuid.uuid5(uuid.NAMESPACE_URL, seed))
    subset = MMLU_SUBJECT_TO_CATEGORY[subject][0]
    return {
        "question": stem,
        "options_text": options_text,
        "options": options,
        "expected_answer": letter,
        "subset_for_metrics": subset,
        "subject": subject,
        "uuid": row_uuid,
    }


def prepare() -> Path:
    """Download MMLU test splits and write Gym JSONL."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    repo = _find_hf_repo()
    print(f"Using HuggingFace dataset: {repo}")

    count = 0
    with OUTPUT_FPATH.open("w", encoding="utf-8") as fout:
        for subject in tqdm(MMLU_SUBJECTS, desc="MMLU subjects"):
            ds = _load_mmlu_subject(repo, subject)
            for ex in ds:
                ans_idx = ex["answer"]
                letter = chr(ord("A") + int(ans_idx))
                row = _row_from_example(subject, ex, letter)
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                count += 1

    print(f"Wrote {count} problems to {OUTPUT_FPATH}")
    return OUTPUT_FPATH


if __name__ == "__main__":
    prepare()
