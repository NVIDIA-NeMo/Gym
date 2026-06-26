# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare MMLU-Redux 2.0 benchmark data for NeMo Gym."""

import argparse
import json
import uuid
from pathlib import Path

from datasets import load_dataset
from tqdm.auto import tqdm


# MMLU subcategories from https://github.com/hendrycks/test/blob/master/categories.py
SUBCATEGORIES = {
    "abstract_algebra": ["math"],
    "anatomy": ["health"],
    "astronomy": ["physics"],
    "business_ethics": ["business"],
    "clinical_knowledge": ["health"],
    "college_biology": ["biology"],
    "college_chemistry": ["chemistry"],
    "college_computer_science": ["computer science"],
    "college_mathematics": ["math"],
    "college_medicine": ["health"],
    "college_physics": ["physics"],
    "computer_security": ["computer science"],
    "conceptual_physics": ["physics"],
    "econometrics": ["economics"],
    "electrical_engineering": ["engineering"],
    "elementary_mathematics": ["math"],
    "formal_logic": ["philosophy"],
    "global_facts": ["other"],
    "high_school_biology": ["biology"],
    "high_school_chemistry": ["chemistry"],
    "high_school_computer_science": ["computer science"],
    "high_school_european_history": ["history"],
    "high_school_geography": ["geography"],
    "high_school_government_and_politics": ["politics"],
    "high_school_macroeconomics": ["economics"],
    "high_school_mathematics": ["math"],
    "high_school_microeconomics": ["economics"],
    "high_school_physics": ["physics"],
    "high_school_psychology": ["psychology"],
    "high_school_statistics": ["math"],
    "high_school_us_history": ["history"],
    "high_school_world_history": ["history"],
    "human_aging": ["health"],
    "human_sexuality": ["culture"],
    "international_law": ["law"],
    "jurisprudence": ["law"],
    "logical_fallacies": ["philosophy"],
    "machine_learning": ["computer science"],
    "management": ["business"],
    "marketing": ["business"],
    "medical_genetics": ["health"],
    "miscellaneous": ["other"],
    "moral_disputes": ["philosophy"],
    "moral_scenarios": ["philosophy"],
    "nutrition": ["health"],
    "philosophy": ["philosophy"],
    "prehistory": ["history"],
    "professional_accounting": ["other"],
    "professional_law": ["law"],
    "professional_medicine": ["health"],
    "professional_psychology": ["psychology"],
    "public_relations": ["politics"],
    "security_studies": ["politics"],
    "sociology": ["culture"],
    "us_foreign_policy": ["politics"],
    "virology": ["health"],
    "world_religions": ["philosophy"],
}

BENCHMARK_DIR = Path(__file__).parent
DATA_DIR = BENCHMARK_DIR / "data"
OUTPUT_FPATH = DATA_DIR / "mmlu-redux_benchmark.jsonl"
LETTERS = ["A", "B", "C", "D"]


def _options_text(choices: list[str]) -> str:
    return "\n".join(f"{letter}) {choice}" for letter, choice in zip(LETTERS, choices, strict=True))


def _expected_answer(entry: dict) -> str | None:
    if entry["error_type"] == "ok":
        return chr(ord("A") + int(entry["answer"]))
    if entry["error_type"] == "wrong_groundtruth" and entry["correct_answer"] in LETTERS:
        return entry["correct_answer"]
    return None


def _to_gym_row(entry: dict, subject: str) -> dict | None:
    expected_answer = _expected_answer(entry)
    if expected_answer is None:
        return None

    choices = [str(choice) for choice in entry["choices"]]
    if len(choices) != len(LETTERS):
        raise ValueError(f"Expected 4 choices for {subject}, found {len(choices)}")

    question = entry["question"].strip()
    seed = json.dumps(
        {"subject": subject, "question": question, "answer": expected_answer},
        sort_keys=True,
        ensure_ascii=False,
    )
    return {
        "question": question,
        "problem": f"{question}\n\n{_options_text(choices)}",
        "options": [{letter: choice} for letter, choice in zip(LETTERS, choices, strict=True)],
        "expected_answer": expected_answer,
        "subset_for_metrics": SUBCATEGORIES[subject][0],
        "subcategory": subject,
        "source": entry["source"],
        "uuid": str(uuid.uuid5(uuid.NAMESPACE_URL, seed)),
    }


def prepare(split: str = "test") -> Path:
    if split != "test":
        raise ValueError("MMLU-Redux only supports split='test'.")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    count = 0
    with OUTPUT_FPATH.open("w", encoding="utf-8") as fout:
        for subject in tqdm(SUBCATEGORIES, desc="MMLU-Redux subjects"):
            dataset = load_dataset("edinburgh-dawg/mmlu-redux-2.0", name=subject, split="test")
            for entry in dataset:
                row = _to_gym_row(entry, subject)
                if row is None:
                    continue
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                count += 1

    print(f"Wrote {count} problems to {OUTPUT_FPATH}")
    return OUTPUT_FPATH


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test", choices=("test",))
    args = parser.parse_args()
    prepare(split=args.split)
