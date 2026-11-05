# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepare AI4Bharat's translated FrontierMath public sample for Gym."""

import argparse
import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

from nemo_gym.data import apply_prompt_to_row, load_prompt_config


BENCHMARK_DIR = Path(__file__).resolve().parent
PROMPT_PATH = BENCHMARK_DIR / "prompts/default.yaml"
SOURCE_ID = "ai4bharat/indic-frontier-math"
# Dataset commit recorded in the supplied Hugging Face cache, not a credential.
SOURCE_REVISION = "dcae6631c88d3ae1a26c188eb8aa13ad4c7f4ad7"  # pragma: allowlist secret
SOURCE_FILENAME = "sample.parquet"
LANGUAGE_NAMES = {
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
LANGUAGES = tuple(LANGUAGE_NAMES)


def _canonical_answer(value: str) -> str:
    value = value.strip()
    if value.startswith(r"\(") and value.endswith(r"\)"):
        value = value[2:-2]
    return "".join(value.split())


def prepare(
    *, dataset_dir: str | None = None, languages: list[str] | None = None, output_dir: str | None = None
) -> Path:
    """Prepare full Gym requests; the reference answer is never part of model input."""
    selected = list(LANGUAGES) if languages is None else list(languages)
    if not selected or len(set(selected)) != len(selected) or set(selected) - set(LANGUAGES):
        raise ValueError(f"languages must be a nonempty, unique subset of {LANGUAGES}")
    source = (
        Path(dataset_dir) / SOURCE_FILENAME
        if dataset_dir is not None
        else Path(
            hf_hub_download(repo_id=SOURCE_ID, filename=SOURCE_FILENAME, repo_type="dataset", revision=SOURCE_REVISION)
        )
    )
    manifest = json.loads((BENCHMARK_DIR / "answers.json").read_text())
    prompt_config = load_prompt_config(str(PROMPT_PATH))
    answers = {row["row_id"]: row for row in manifest["answers"]}
    table = pq.read_table(source)
    required_columns = {"id", "tier", "problem", "answer"} | {
        f"problem_{LANGUAGE_NAMES[code]}_translation" for code in selected if code != "en"
    }
    missing = required_columns - set(table.column_names)
    if missing:
        raise ValueError(f"Missing source columns: {sorted(missing)}")
    records = table.to_pylist()
    if len(records) != len(answers) or {row["id"] for row in records} != set(answers):
        raise ValueError("Source problem IDs must be unique and match the answer key")
    for row in records:
        answer = answers[row["id"]]
        if not isinstance(row["problem"], str) or not row["problem"].strip():
            raise ValueError(f"Empty English problem for {row['id']}")
        if hashlib.sha256(row["problem"].encode()).hexdigest() != answer["english_problem_sha256"]:
            raise ValueError(f"English statement changed for {row['id']}; review the answer key before evaluation")
        if type(row["tier"]) is not int or row["tier"] != answer["tier"]:
            raise ValueError(f"Tier mismatch for {row['id']}")
        if not isinstance(row["answer"], str) or _canonical_answer(row["answer"]) != _canonical_answer(
            answer["expected_answer"]
        ):
            raise ValueError(f"Source answer changed for {row['id']}; review the answer key before evaluation")

    rows = []
    row_indexes = {row_id: index for index, row_id in enumerate(answers)}
    for code in selected:
        for row in records:
            problem = row["problem" if code == "en" else f"problem_{LANGUAGE_NAMES[code]}_translation"]
            if not isinstance(problem, str) or not problem.strip():
                raise ValueError(f"Empty problem for {code}/{row['id']}; English fallback is forbidden")
            answer = answers[row["id"]]
            rows.append(
                apply_prompt_to_row(
                    {
                        "row_id": row["id"],
                        "row_index": row_indexes[row["id"]],
                        "problem": problem,
                        "language": LANGUAGE_NAMES[code],
                        "language_code": code,
                        "judge_pass_stage": "english_source" if code == "en" else "not_provided",
                        "human_evaluation_pending": code != "en",
                        "human_review_reason": ""
                        if code == "en"
                        else "Per-item review status is not provided by the source.",
                        "task_id": f"{code}/{row['id']}",
                        "question": problem,
                        "expected_answer": answer["expected_answer"],
                        "answer_type": answer["answer_type"],
                        "tier": answer["tier"],
                        "answer_key_version": manifest["version"],
                    },
                    prompt_config,
                )
            )

    # Validate every requested file before publishing any output.
    destination = Path(output_dir) if output_dir else BENCHMARK_DIR / "data"
    destination.mkdir(parents=True, exist_ok=True)
    combined = destination / "indic_frontiermath_benchmark.jsonl"
    for path, content in [(combined, rows)] + [
        (destination / f"{code}.jsonl", [row for row in rows if row["language_code"] == code]) for code in selected
    ]:
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in content), encoding="utf-8")
    # Gym applies the configured prompt to raw rows. Keep the rendered requests
    # above for standalone runners and comparison with existing evaluations.
    raw_rows = [{key: value for key, value in row.items() if key != "responses_create_params"} for row in rows]
    raw_path = destination / "indic_frontiermath_raw.jsonl"
    raw_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in raw_rows), encoding="utf-8")
    audit = {
        "source_id": SOURCE_ID,
        "source_revision": SOURCE_REVISION if dataset_dir is None else None,
        "source_file": str(source.resolve()),
        "local_override": dataset_dir is not None,
        "answer_key_version": manifest["version"],
        "protocol": "exact_boxed_text_no_tools",
        "rows": len(rows),
        "unique_problems": len(answers),
        "languages": selected,
        "human_evaluation_pending": sum(row["human_evaluation_pending"] for row in rows),
        "failed_translation_review": sum(row["judge_pass_stage"].startswith("failed_") for row in rows),
        "translation_review_status": "not_provided",
        "parquet_sha256": {SOURCE_FILENAME: hashlib.sha256(source.read_bytes()).hexdigest()},
    }
    (destination / "preparation.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(f"Prepared {len(rows)} rows ({len(answers)} problems, {len(selected)} languages): {combined}")
    return raw_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", help="Local directory containing sample.parquet; skips the Hub download.")
    parser.add_argument("--languages", nargs="+", choices=LANGUAGES)
    parser.add_argument("--output-dir")
    prepare(**vars(parser.parse_args()))
