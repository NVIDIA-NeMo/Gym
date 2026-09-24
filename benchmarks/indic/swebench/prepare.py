# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare Indic SWE-bench using the English Verified prompt and task schema."""

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

from datasets import load_dataset


BENCHMARK_DIR = Path(__file__).parent
OUTPUT_FPATH = BENCHMARK_DIR / "data" / "swebench_benchmark.jsonl"
PROMPT_PATH = BENCHMARK_DIR.parents[1] / "swebench/minimax_prompt.txt"
SOURCE_ID = "ai4bharat/indic-swe-bench"
EXPECTED_INSTANCES = 500
LANGUAGE_NAMES = {
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
DEFAULT_LANGUAGES = tuple(LANGUAGE_NAMES)
INSTANCE_FIELDS = (
    "repo",
    "instance_id",
    "base_commit",
    "patch",
    "test_patch",
    "problem_statement",
    "hints_text",
    "created_at",
    "version",
    "FAIL_TO_PASS",
    "PASS_TO_PASS",
    "environment_setup_commit",
    "difficulty",
)


def build_rows(
    records: Sequence[Mapping[str, str]],
    *,
    languages: Sequence[str] = DEFAULT_LANGUAGES,
) -> list[dict]:
    """Select translated issues while preserving the original test and sandbox inputs."""
    if (
        isinstance(languages, str)
        or not languages
        or any(code not in (*DEFAULT_LANGUAGES, "en") for code in languages)
    ):
        raise ValueError(f"languages must be a nonempty sequence from {(*DEFAULT_LANGUAGES, 'en')}")
    if len(set(languages)) != len(languages):
        raise ValueError("languages must be unique")
    if len(records) != EXPECTED_INSTANCES:
        raise ValueError(f"Expected {EXPECTED_INSTANCES} SWE-bench Verified instances, got {len(records)}")
    if len({record["instance_id"] for record in records}) != len(records):
        raise ValueError("Duplicate instance_id in SWE-bench Verified")
    template = PROMPT_PATH.read_text()
    rows = []
    for language in languages:
        column = (
            "problem_statement" if language == "en" else f"problem_statement_{LANGUAGE_NAMES[language]}_translation"
        )
        for record in records:
            identity = record["instance_id"]
            problem = record.get(column)
            if not isinstance(problem, str) or not problem.strip():
                raise ValueError(f"Missing problem statement: {language}/{identity}")
            prompt = (
                template.replace("{{ workspace_path }}", "/testbed")
                .replace("{{ instance.problem_statement }}", problem)
                .replace("{{ instance.repo_language ~ ' ' if instance.repo_language else '' }}", "")
            )
            rows.append(
                {
                    **{key: record[key] for key in INSTANCE_FIELDS},
                    "problem_statement": problem,
                    "question": prompt,
                    "subset": "verified",
                    "split": "test",
                    "language": language,
                    "uuid": f"{SOURCE_ID}/{language}/{identity}",
                }
            )
    return rows


def prepare(
    *,
    languages: Sequence[str] = DEFAULT_LANGUAGES,
    output_fpath: str | None = None,
) -> Path:
    """Load the test split and write rows for the existing SWE-bench verifier."""
    dataset = load_dataset(SOURCE_ID, split="test")
    rows = build_rows(list(dataset), languages=languages)
    output = Path(output_fpath) if output_fpath else OUTPUT_FPATH
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows)} problems to {output}")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--languages", nargs="+", choices=(*DEFAULT_LANGUAGES, "en"), default=DEFAULT_LANGUAGES)
    parser.add_argument("--output-fpath")
    prepare(**vars(parser.parse_args()))
