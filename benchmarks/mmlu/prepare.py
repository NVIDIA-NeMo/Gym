# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare MMLU benchmark data for NeMo Gym."""

import argparse
import csv
import io
import json
import os
import tarfile
import urllib.request
import uuid
from pathlib import Path


URL = "https://people.eecs.berkeley.edu/~hendrycks/data.tar"

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
OUTPUT_FPATH = DATA_DIR / "mmlu_benchmark.jsonl"
LETTERS = ["A", "B", "C", "D"]


def read_csv_files_from_tar(tar_file_path: Path, split: str) -> dict[str, list[dict[str, str]]]:
    column_names = ["question", "A", "B", "C", "D", "expected_answer"]
    result = {}

    with tarfile.open(tar_file_path, "r") as tar:
        csv_files = [
            member
            for member in tar.getmembers()
            if member.name.startswith(f"data/{split}/") and member.name.endswith(".csv")
        ]

        for csv_file in csv_files:
            file_name = os.path.basename(csv_file.name)
            file_content = tar.extractfile(csv_file)
            if file_content is None:
                continue

            content_str = io.TextIOWrapper(file_content, encoding="utf-8")
            csv_reader = csv.reader(content_str)
            rows = []
            for row in csv_reader:
                if len(row) == len(column_names):
                    rows.append(dict(zip(column_names, row, strict=True)))
                else:
                    print(f"Warning: skipping row in {file_name} due to incorrect number of columns")

            result[file_name.rsplit("_", 1)[0]] = rows

    return result


def _options_text(row: dict[str, str]) -> str:
    return "\n".join(f"{letter}) {row[letter]}" for letter in LETTERS)


def _to_gym_row(subject: str, row: dict[str, str]) -> dict:
    expected_answer = row["expected_answer"].strip().upper()
    if expected_answer not in LETTERS:
        raise ValueError(f"Unexpected answer {expected_answer!r} for subject {subject}")

    question = row["question"].strip()
    seed = json.dumps(
        {"subject": subject, "question": question, "answer": expected_answer},
        sort_keys=True,
        ensure_ascii=False,
    )
    return {
        "question": question,
        "problem": f"{question}\n\n{_options_text(row)}",
        "options": [{letter: row[letter]} for letter in LETTERS],
        "expected_answer": expected_answer,
        "subset_for_metrics": SUBCATEGORIES[subject][0],
        "subject": subject,
        "uuid": str(uuid.uuid5(uuid.NAMESPACE_URL, seed)),
    }


def _output_fpath_for_split(split: str) -> Path:
    if split == "test":
        return OUTPUT_FPATH
    return DATA_DIR / f"mmlu_{split}.jsonl"


def prepare(split: str = "test") -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    data_file = DATA_DIR / "data.tar"
    output_fpath = _output_fpath_for_split(split)

    print(f"Downloading MMLU from {URL}")
    urllib.request.urlretrieve(URL, data_file)

    original_data = read_csv_files_from_tar(data_file, split)
    count = 0
    with output_fpath.open("w", encoding="utf-8") as fout:
        for subject, rows in original_data.items():
            for row in rows:
                fout.write(json.dumps(_to_gym_row(subject, row), ensure_ascii=False) + "\n")
                count += 1

    data_file.unlink()
    print(f"Wrote {count} problems to {output_fpath}")
    return output_fpath


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test", choices=("dev", "test", "val"))
    args = parser.parse_args()
    prepare(split=args.split)
