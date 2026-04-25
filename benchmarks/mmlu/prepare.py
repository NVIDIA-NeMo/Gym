# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare MMLU (Hendrycks) benchmark data for NeMo Gym (mcqa).

Data preparation matches NeMo-Skills ``nemo_skills/dataset/mmlu/prepare.py``:
Berkeley ``data.tar``, CSV layout ``question``, ``A``–``D``, ``expected_answer``, and the
same ``subcategories`` mapping. Output is Gym ``mcqa`` JSONL (plus ``options_text``
for the prompt template).
"""

import argparse
import csv
import io
import json
import os
import tarfile
import urllib.request
import uuid
from pathlib import Path


# mmlu subcategories from https://github.com/hendrycks/test/blob/master/categories.py
# (same dict as NeMo-Skills ``nemo_skills/dataset/mmlu/prepare.py``).
subcategories = {
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

URL = "https://people.eecs.berkeley.edu/~hendrycks/data.tar"

BENCHMARK_DIR = Path(__file__).parent
DATA_DIR = BENCHMARK_DIR / "data"
OUTPUT_FPATH = DATA_DIR / "mmlu_benchmark.jsonl"


def read_csv_files_from_tar(tar_file_path: str, split: str) -> dict:
    """Same logic as NeMo-Skills ``mmlu/prepare.read_csv_files_from_tar``."""
    result: dict[str, list] = {}
    column_names = ["question", "A", "B", "C", "D", "expected_answer"]

    with tarfile.open(tar_file_path, "r") as tar:
        members = tar.getmembers()
        csv_files = [
            member for member in members if member.name.startswith(f"data/{split}/") and member.name.endswith(".csv")
        ]

        for csv_file in csv_files:
            file_name = os.path.basename(csv_file.name)
            file_content = tar.extractfile(csv_file)
            if file_content is not None:
                content_str = io.TextIOWrapper(file_content, encoding="utf-8")
                csv_reader = csv.reader(content_str)
                csv_data = []
                for row in csv_reader:
                    if len(row) == len(column_names):
                        csv_data.append(dict(zip(column_names, row, strict=True)))
                    else:
                        print(f"Warning: Skipping row in {file_name} due to incorrect number of columns")

                result[file_name.rsplit("_", 1)[0]] = csv_data

    return result


def _gym_row(subject: str, question: dict) -> dict:
    """NeMo CSV row -> Gym mcqa JSONL row."""
    letters = ["A", "B", "C", "D"]
    choices = [question[k] for k in letters]
    stem = question["question"].strip()
    letter = str(question["expected_answer"]).strip().upper()
    if len(letter) != 1 or letter not in letters:
        raise ValueError(f"Bad expected_answer {letter!r} for subject={subject}")
    options = [{letters[i]: choices[i]} for i in range(4)]
    options_text = "\n".join(f"{letters[i]}) {choices[i]}" for i in range(4))
    seed = json.dumps({"subject": subject, "question": stem, "answer": letter}, sort_keys=True)
    row_uuid = str(uuid.uuid5(uuid.NAMESPACE_URL, seed))
    return {
        "question": stem,
        "options_text": options_text,
        "options": options,
        "expected_answer": letter,
        "subset_for_metrics": subcategories[subject][0],
        "subject": subject,
        "uuid": row_uuid,
    }


def _output_fpath_for_split(split: str) -> Path:
    """Benchmark config pins ``test`` to ``mmlu_benchmark.jsonl``; other splits match NeMo filenames."""
    if split == "test":
        return OUTPUT_FPATH
    return DATA_DIR / f"mmlu_{split}.jsonl"


def prepare(split: str = "test") -> Path:
    """Download Hendrycks ``data.tar`` and write Gym JSONL (NeMo-Skills-equivalent ingestion)."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    data_file = DATA_DIR / "data.tar"
    out_path = _output_fpath_for_split(split)

    print(f"Downloading {URL} ...")
    urllib.request.urlretrieve(URL, data_file)

    original_data = read_csv_files_from_tar(str(data_file), split)
    count = 0
    with out_path.open("w", encoding="utf-8") as fout:
        for subject, questions in original_data.items():
            for q in questions:
                fout.write(json.dumps(_gym_row(subject, q), ensure_ascii=False) + "\n")
                count += 1

    os.remove(data_file)
    print(f"Wrote {count} problems to {out_path}")
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split",
        default="test",
        choices=("dev", "test", "val"),
    )
    args = parser.parse_args()
    prepare(split=args.split)
