# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare Indic JEEBench with the authors' zero-shot CoT prompt."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path

import yaml
from datasets import load_dataset


BENCHMARK_DIR = Path(__file__).parent
OUTPUT_FPATH = BENCHMARK_DIR / "data/jee_bench_benchmark.jsonl"
SOURCE_ID = "ai4bharat/indic-jee-bench"
# Public dataset revision, not a credential.
SOURCE_REVISION = "5ce209f9900a434ead4f040a52004f6fae8aa1f7"  # pragma: allowlist secret
SOURCE_SPLIT = "test"
LANGUAGES = {
    "en": "English",
    "as": "Assamese",
    "bn": "Bengali",
    "gu": "Gujarati",
    "hi": "Hindi",
    "kn": "Kannada",
    "ml": "Malayalam",
    "mr": "Marathi",
    "ne": "Nepali",
    "or": "Odia",
    "pa": "Punjabi",
    "sa": "Sanskrit",
    "ta": "Tamil",
    "te": "Telugu",
    "ur": "Urdu",
}
DEFAULT_LANGUAGES = tuple(code for code in LANGUAGES if code != "en")
QUESTION_TYPES = {"MCQ", "MCQ(multiple)", "Integer", "Numeric"}
ALIGNMENT_FIELDS = ("description", "index", "subject", "type", "gold")


def question_id(row: dict) -> str:
    """Question numbers repeat between papers; identify them with the paper name."""
    return f"{row['description']}/{row['index']}"


def _index_rows(rows: list[dict], language: str) -> dict[str, dict]:
    if not rows:
        raise ValueError(f"Empty JEEBench source for {language}")
    indexed = {}
    for row in rows:
        for field in ("description", "subject", "type", "gold", "question"):
            if not isinstance(row.get(field), str) or not row[field].strip():
                raise ValueError(f"Nonempty {field} required for {language}; no English fallback")
        if type(row.get("index")) is not int or row["index"] < 1 or "/" in row["description"]:
            raise ValueError("Expected a positive question index and unambiguous paper description")
        if row["type"] not in QUESTION_TYPES or row["subject"] not in {"phy", "chem", "math"}:
            raise ValueError("Unknown JEEBench question type or subject")
        if row.get("language_code") != language or row.get("language") != LANGUAGES[language]:
            raise ValueError(f"Language metadata mismatch for {language}")
        gold = row["gold"]
        if row["type"] in {"MCQ", "MCQ(multiple)"}:
            if not re.fullmatch(r"[ABCD]+", gold) or gold != "".join(sorted(set(gold))):
                raise ValueError(f"Invalid option gold: {gold!r}")
            if row["type"] == "MCQ" and len(gold) != 1:
                raise ValueError("Single-choice questions require exactly one gold option")
        else:
            value = float(gold)
            if not math.isfinite(value) or (row["type"] == "Integer" and (value < 0 or not value.is_integer())):
                raise ValueError(f"Invalid numeric gold: {gold!r}")
        identity = question_id(row)
        if identity in indexed:
            raise ValueError(f"Duplicate question: {language}/{identity}")
        indexed[identity] = row
    return indexed


def build_rows(records: list[dict], *, language: str, revision: str) -> list[dict]:
    """Preserve options/golds and reproduce inference.py's CoT prompt exactly."""
    prefixes = yaml.safe_load((BENCHMARK_DIR / "prompts/prefixes.yaml").read_text(encoding="utf-8"))
    rows = []
    for record in records:
        identity = question_id(record)
        task_id = f"indic/jee_bench/{revision}/{language}/{identity}"
        question = record["question"].replace("\n\n", "\n").strip()
        rows.append(
            {
                "uuid": task_id,
                "question_id": identity,
                "question": record["question"],
                "instruction": f"{prefixes[record['type']]}\n\nProblem: {question}\nSolution: Let's think step by step.",
                "language": language,
                "language_name": LANGUAGES[language],
                "subset_for_metrics": language,
                "judge_pass_stage": record.get("judge_pass_stage"),
                "source_id": SOURCE_ID,
                "source_revision": revision,
                # Defaults belong to JEEBench rows; explicit run-level overrides still win.
                "responses_create_params": {"temperature": 0.0, "max_output_tokens": 2048},
                "verifier_metadata": {
                    "task_id": task_id,
                    "language": language,
                    "subject": record["subject"],
                    "question_type": record["type"],
                    "gold": record["gold"],
                    "description": record["description"],
                    "index": record["index"],
                },
            }
        )
    return rows


def prepare(
    *,
    languages: list[str] | None = None,
    question_ids: list[str] | None = None,
    revision: str = SOURCE_REVISION,
    source_dir: str | None = None,
    output_fpath: str | None = None,
) -> Path:
    """Load the wide HF test split or a local snapshot containing test.parquet.

    Translations share the original English answers and identities on each row.
    Missing translations are reported; they are never replaced by English rows.
    """
    selected = list(DEFAULT_LANGUAGES) if languages is None else languages
    if isinstance(selected, str) or not selected or len(set(selected)) != len(selected):
        raise ValueError("languages must contain unique language codes")
    # Gym passes Hydra ListConfig values; persist plain JSON-compatible lists.
    selected = list(selected)
    if set(selected) - LANGUAGES.keys():
        raise ValueError(f"Unknown languages: {sorted(set(selected) - LANGUAGES.keys())}")
    if question_ids is not None and (
        isinstance(question_ids, str) or not question_ids or len(set(question_ids)) != len(question_ids)
    ):
        raise ValueError("question_ids must contain unique description/index IDs")
    source_files = {}
    if source_dir is None:
        dataset = load_dataset(SOURCE_ID, split=SOURCE_SPLIT, revision=revision)
    else:
        path = Path(source_dir) / f"{SOURCE_SPLIT}.parquet"
        source_files[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        dataset = load_dataset("parquet", data_files={SOURCE_SPLIT: str(path)}, split=SOURCE_SPLIT)
    records = list(dataset)

    def load(language: str) -> dict[str, dict]:
        column = "question" if language == "en" else f"question_{LANGUAGES[language]}_translation"
        if any(column not in row for row in records):
            raise ValueError(f"Missing source column: {column}")
        return _index_rows(
            [
                {**row, "question": row[column], "language": LANGUAGES[language], "language_code": language}
                for row in records
                if language == "en" or row[column] is not None
            ],
            language,
        )

    english = load("en")
    if len(english) != 515:
        raise ValueError(f"Expected 515 English JEEBench questions, got {len(english)}")
    wanted = set(english) if question_ids is None else set(question_ids)
    if wanted - english.keys():
        raise ValueError(f"Unknown question IDs: {sorted(wanted - english.keys())}; use description/index")
    rows, coverage = [], {}
    for language in selected:
        indexed = english if language == "en" else load(language)
        if indexed.keys() - english.keys():
            raise ValueError(f"Unknown translated question identities for {language}")
        for identity, row in indexed.items():
            if any(row[field] != english[identity][field] for field in ALIGNMENT_FIELDS):
                raise ValueError(f"English metadata/gold mismatch: {language}/{identity}")
        missing = wanted - indexed.keys()
        if question_ids is not None and missing:
            raise ValueError(f"Requested translations missing for {language}: {sorted(missing)}")
        selected_records = [indexed[identity] for identity in english if identity in wanted and identity in indexed]
        if not selected_records:
            raise ValueError(f"No selected questions for {language}")
        coverage[language] = {
            "published_rows": len(indexed),
            "selected_rows": len(selected_records),
            "missing_english_ids": sorted(english.keys() - indexed.keys()),
            "judge_pass_stage": dict(Counter(record.get("judge_pass_stage") for record in selected_records)),
        }
        rows.extend(build_rows(selected_records, language=language, revision=revision))

    path = Path(output_fpath) if output_fpath else OUTPUT_FPATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    manifest = {
        "source_id": SOURCE_ID,
        "source_revision": revision,
        "source_split": SOURCE_SPLIT,
        "source_dir": source_dir,
        "source_file_sha256": source_files,
        "languages": selected,
        "question_ids": list(question_ids) if question_ids is not None else None,
        "coverage": coverage,
        "prepared_rows": len(rows),
        "prepared_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "adapter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "prompt_sha256": hashlib.sha256((BENCHMARK_DIR / "prompts/prefixes.yaml").read_bytes()).hexdigest(),
    }
    path.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(rows)} JEEBench questions to {path}")
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--languages", nargs="+", choices=LANGUAGES)
    parser.add_argument("--question-ids", nargs="+")
    parser.add_argument("--revision", default=SOURCE_REVISION)
    parser.add_argument("--source-dir")
    parser.add_argument("--output-fpath")
    prepare(**vars(parser.parse_args()))
