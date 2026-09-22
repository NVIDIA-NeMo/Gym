# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare Indic SWE-bench using the English Verified prompt and task schema."""

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import get_token, hf_hub_download

from nemo_gym.global_config import get_hf_token


BENCHMARK_DIR = Path(__file__).parent
OUTPUT_FPATH = BENCHMARK_DIR / "data" / "swebench_benchmark.jsonl"
PROMPT_PATH = BENCHMARK_DIR.parents[1] / "swebench/minimax_prompt.txt"
SOURCE_ID = "ai4bharat/indic-swe-bench"
SOURCE_REVISION = "f03e95b7749c7f06b8aa1a2ba75360d9fcb26a43"
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
    instance_ids: Sequence[str] | None = None,
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
    indexed = {}
    for record in records:
        if any(not isinstance(record.get(key), str) for key in INSTANCE_FIELDS):
            raise ValueError("Missing or invalid SWE-bench instance fields")
        identity = record["instance_id"]
        if not identity.strip() or identity in indexed:
            raise ValueError(f"Empty or duplicate instance_id: {identity!r}")
        indexed[identity] = record
    if instance_ids is not None:
        if isinstance(instance_ids, str) or not instance_ids or len(set(instance_ids)) != len(instance_ids):
            raise ValueError("instance_ids must be a nonempty sequence of unique IDs")
        if set(instance_ids) - indexed.keys():
            raise ValueError(f"Unknown instance_ids: {sorted(set(instance_ids) - indexed.keys())}")
    selected = set(indexed if instance_ids is None else instance_ids)
    template = PROMPT_PATH.read_text()
    rows = []
    for language in languages:
        column = (
            "problem_statement" if language == "en" else f"problem_statement_{LANGUAGE_NAMES[language]}_translation"
        )
        for identity, record in indexed.items():
            problem = record.get(column)
            if not isinstance(problem, str) or not problem.strip():
                raise ValueError(f"Missing problem statement: {language}/{identity}")
            if identity not in selected:
                continue
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
                    "uuid": f"{SOURCE_ID}/{SOURCE_REVISION}/{language}/{identity}",
                }
            )
    return rows


def prepare(
    *,
    languages: Sequence[str] = DEFAULT_LANGUAGES,
    instance_ids: Sequence[str] | None = None,
    output_fpath: str | None = None,
) -> Path:
    """Download the pinned dataset and write rows for the existing SWE-bench verifier."""
    source = hf_hub_download(
        repo_id=SOURCE_ID,
        filename="test.parquet",
        repo_type="dataset",
        revision=SOURCE_REVISION,
        token=get_hf_token() or get_token(),
    )
    dataset = load_dataset("parquet", data_files={"test": source}, split="test")
    rows = build_rows(dataset.to_list(), languages=languages, instance_ids=instance_ids)
    output = Path(output_fpath) if output_fpath else OUTPUT_FPATH
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows)} problems to {output}")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--languages", nargs="+", default=DEFAULT_LANGUAGES)
    parser.add_argument("--instance-ids", nargs="+")
    parser.add_argument("--output-fpath")
    prepare(**vars(parser.parse_args()))
