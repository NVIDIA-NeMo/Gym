# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare pinned Indic AIME 2026 with exact canonical problem and answer alignment."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from benchmarks.indic._shared.source import (
    LANGUAGE_NAMES,
    QUALITY_STAGES,
    SourceSpec,
    download_hf_file,
    sha256,
    write_jsonl_with_manifest,
)


BENCHMARK_DIR = Path(__file__).parent
OUTPUT_FPATH = BENCHMARK_DIR / "data" / "aime_2026_benchmark.jsonl"
DEFAULT_LANGUAGES = ("bn", "gu", "hi", "kn", "ml", "mr", "ne", "or", "pa", "ta", "te", "ur")
SOURCE = SourceSpec(
    name="indic/aime_2026",
    repo_id="anushakamathofficial/indic_aime_2026",
    revision="938c1c90c23b25ca0f43d1bfc5103b332e8c033e",
    split="train",
    text_field="problem",
    license="CC-BY-NC-SA-4.0",
    expected_english_rows=30,
)
CANONICAL_REPO = "MathArena/aime_2026"
CANONICAL_REVISION = "d2de22f3c656b4f56cf8981212186377d1e23bc3"
CANONICAL_FILE = "data/train-00000-of-00001.parquet"
UPSTREAM_REVISION = "b89f2f0ad64ced464d2944f08c3c0aaeaa0df64b"
CANONICAL_COLUMNS = {"problem_idx", "answer", "problem"}
SOURCE_COLUMNS = CANONICAL_COLUMNS | {"language", "language_code", "judge_pass_stage"}


def _index_rows(records: Sequence[Mapping[str, Any]], *, language: str | None) -> dict[str, Mapping[str, Any]]:
    if not records:
        raise ValueError(f"Empty AIME 2026 configuration: {language or 'canonical'}")
    indexed = {}
    for row in records:
        if not isinstance(row, Mapping) or set(row) != (SOURCE_COLUMNS if language else CANONICAL_COLUMNS):
            raise ValueError(f"Unexpected AIME 2026 source columns for {language or 'canonical'}")
        if type(row["problem_idx"]) is not int or row["problem_idx"] < 1:
            raise ValueError("AIME 2026 requires positive integer problem_idx values")
        if type(row["answer"]) is not int or not 0 <= row["answer"] <= 999:
            raise ValueError("AIME 2026 requires integer answers between 0 and 999 inclusive")
        if not isinstance(row["problem"], str) or not row["problem"].strip():
            raise ValueError("Nonempty translated problem required; English fallback is forbidden")
        identity = str(row["problem_idx"])
        if identity in indexed:
            raise ValueError(f"Duplicate AIME 2026 problem identity: {language}/{identity}")
        if language:
            if row["language_code"] != language or row["language"] != LANGUAGE_NAMES[language]:
                raise ValueError(f"Language metadata mismatch: {language}/{identity}")
            stage = row["judge_pass_stage"]
            if (
                not isinstance(stage, str)
                or stage not in QUALITY_STAGES
                or (stage == "english_source") != (language == "en")
            ):
                raise ValueError(f"Unexpected translation quality stage: {language}/{identity}")
        indexed[identity] = row
    return indexed


def _selection(
    languages: Sequence[str] | None, config_name: str | None, question_ids: Sequence[str | int] | None
) -> tuple[list[str], list[str] | None]:
    if isinstance(languages, str):
        raise ValueError("languages must be a sequence, not a string")
    if config_name is not None:
        if languages is not None and list(languages) != [config_name]:
            raise ValueError("config_name conflicts with languages")
        languages = [config_name]
    selected = list(DEFAULT_LANGUAGES if languages is None else languages)
    if not selected or any(not isinstance(code, str) for code in selected) or len(set(selected)) != len(selected):
        raise ValueError("languages must contain unique configuration codes")
    unsupported = set(selected) - {"en", *DEFAULT_LANGUAGES}
    if unsupported:
        raise ValueError(f"Unsupported AIME 2026 language configurations: {sorted(unsupported)}")
    if question_ids is None:
        return selected, None
    if isinstance(question_ids, str) or not question_ids:
        raise ValueError("question_ids must contain unique positive decimal IDs")
    normalized = []
    for value in question_ids:
        if type(value) is int and value > 0:
            identity = str(value)
        elif (
            isinstance(value, str)
            and value.isascii()
            and value.isdecimal()
            and str(int(value)) == value
            and value != "0"
        ):
            identity = value
        else:
            raise ValueError("question_ids must contain positive decimal IDs, as strings or integers, without padding")
        if identity in normalized:
            raise ValueError("question_ids must contain unique positive decimal IDs")
        normalized.append(identity)
    return selected, normalized


def load_source(
    *,
    languages: Sequence[str] | None = None,
    config_name: str | None = None,
    question_ids: Sequence[str | int] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Validate every selected configuration before filtering published translations.

    English text and answers must match the pinned MathArena source exactly. Missing
    translated questions are reported, not replaced with English or silently hidden.
    """
    import pyarrow.parquet as pq

    selected, question_ids = _selection(languages, config_name, question_ids)
    canonical_path = download_hf_file(CANONICAL_REPO, CANONICAL_REVISION, CANONICAL_FILE)
    canonical = _index_rows(pq.read_table(canonical_path).to_pylist(), language=None)
    if set(canonical) != {str(index) for index in range(1, SOURCE.expected_english_rows + 1)}:
        raise ValueError(f"Expected canonical AIME 2026 problem IDs 1 through {SOURCE.expected_english_rows}")
    wanted = set(canonical) if question_ids is None else set(question_ids)
    if wanted - canonical.keys():
        raise ValueError(f"Unknown question IDs: {sorted(wanted - canonical.keys(), key=int)}")
    configs, files = {}, {}
    for language in dict.fromkeys(["en", *selected]):
        relative = f"data/{language}/{SOURCE.split}.parquet"
        path = download_hf_file(SOURCE.repo_id, SOURCE.revision, relative)
        indexed = _index_rows(pq.read_table(path).to_pylist(), language=language)
        if indexed.keys() - canonical.keys():
            raise ValueError(f"Translated IDs absent from canonical source: {language}")
        if language == "en" and indexed.keys() != canonical.keys():
            raise ValueError("English AIME 2026 configuration must contain every canonical problem")
        fields = CANONICAL_COLUMNS if language == "en" else CANONICAL_COLUMNS - {"problem"}
        for identity, row in indexed.items():
            if any(row[key] != canonical[identity][key] for key in fields):
                raise ValueError(f"Canonical field mismatch: {language}/{identity}")
        configs[language] = indexed
        files[relative] = {"sha256": sha256(path), "rows": len(indexed)}
    records, coverage = [], {}
    for language in selected:
        indexed = configs[language]
        identities = sorted(wanted & indexed.keys(), key=int)
        if not identities:
            raise ValueError(f"Requested selection has no published translated rows for {language}")
        for identity in identities:
            records.append(
                {
                    **indexed[identity],
                    "question_id": identity,
                    "language": language,
                    "language_name": LANGUAGE_NAMES[language],
                }
            )
        stages = Counter(indexed[identity]["judge_pass_stage"] for identity in identities)
        coverage[language] = {
            "published_rows": len(indexed),
            "selected_rows": len(identities),
            "missing_english_ids": sorted(canonical.keys() - indexed.keys(), key=int),
            "missing_selected_ids": sorted(wanted - indexed.keys(), key=int),
            "judge_pass_stage": dict(sorted(stages.items())),
            "human_evaluation_pending": sum(
                count for stage, count in stages.items() if stage.endswith("review_needed")
            ),
        }
    return records, {
        "source_id": SOURCE.repo_id,
        "source_revision": SOURCE.revision,
        "source_split": SOURCE.split,
        "source_license": SOURCE.license,
        "source_configs": selected,
        "source_files": files,
        "canonical_source_id": CANONICAL_REPO,
        "canonical_revision": CANONICAL_REVISION,
        "canonical_file_sha256": sha256(canonical_path),
        "english_rows": len(canonical),
        "coverage": coverage,
        "question_ids": question_ids,
        "canonical_join": "exact_problem_idx_all_three_English_fields_translated_answer_unchanged",
        "translation_quality_policy": "include_all_published_and_report_flags",
    }


def build_rows(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Keep answer keys in verifier metadata, separate from the official policy prompt."""
    rows = []
    for row in records:
        task_id = f"{SOURCE.name}/{SOURCE.revision}/{row['language']}/{row['question_id']}"
        metadata = {
            "task_id": task_id,
            "language": row["language"],
            "problem_idx": row["problem_idx"],
            "expected_answer": row["answer"],
        }
        rows.append(
            {
                **row,
                **metadata,
                "uuid": task_id,
                "verifier_metadata": metadata,
                "benchmark_id": SOURCE.name,
                "source_id": SOURCE.repo_id,
                "source_revision": SOURCE.revision,
                "subset_for_metrics": row["language"],
                "human_evaluation_pending": row["judge_pass_stage"].endswith("review_needed"),
            }
        )
    groups = {}
    for language in sorted({row["language"] for row in rows}):
        ids = sorted(row["task_id"] for row in rows if row["language"] == language)
        groups[language] = {"questions": len(ids), "ids_sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest()}
    return [{**row, "expected_groups": groups} for row in rows]


def prepare(
    *,
    languages: Sequence[str] | None = None,
    config_name: str | None = None,
    question_ids: Sequence[str | int] | None = None,
    output_fpath: str | None = None,
) -> Path:
    """Write native Gym rows and a hash-backed manifest after all alignment checks."""
    records, metadata = load_source(languages=languages, config_name=config_name, question_ids=question_ids)
    rows = build_rows(records)
    metadata.update(
        {
            "benchmark_id": SOURCE.name,
            "official_protocol_revision": UPSTREAM_REVISION,
            "adapter_sha256": sha256(Path(__file__)),
            "prompt_sha256": sha256(BENCHMARK_DIR / "prompts/default.yaml"),
            "prompt_mode": "official_MathArena_English_instructions_translated_problem",
            "max_output_tokens": 120000,
            "thinking_enabled_by_default": True,
            "expected_groups": rows[0]["expected_groups"],
        }
    )
    return write_jsonl_with_manifest(Path(output_fpath) if output_fpath else OUTPUT_FPATH, rows, metadata)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--languages", nargs="+")
    parser.add_argument("--config-name")
    parser.add_argument("--question-ids", nargs="+")
    parser.add_argument("--output-fpath")
    print(prepare(**vars(parser.parse_args())))


if __name__ == "__main__":
    main()
