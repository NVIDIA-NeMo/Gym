# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Join the local translated public sample to a versioned reference answer key."""

import argparse
import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq

from nemo_gym.prompt import apply_prompt_to_row, load_prompt_config


BENCHMARK_DIR = Path(__file__).resolve().parent
PROMPT_PATH = BENCHMARK_DIR / "prompts/default.yaml"
LANGUAGES = ("en", "as", "bn", "gu", "hi", "kn", "ml", "mr", "ne", "or", "pa", "sa", "ta", "te", "ur")
DATASET_RELATIVE_PATH = Path("huggingface_datasets/anushakamathofficial/indic_frontiermath")


def find_dataset_dir() -> Path:
    """Locate the user's sibling dataset snapshot without a machine-specific path."""
    for parent in BENCHMARK_DIR.parents:
        candidate = parent / DATASET_RELATIVE_PATH
        if (candidate / "data/en/sample.parquet").is_file():
            return candidate
    raise FileNotFoundError(
        "Dataset snapshot not found. Pass --dataset-dir (Gym: +prepare_script_args.dataset_dir=...)."
    )


def prepare(
    *, dataset_dir: str | None = None, languages: list[str] | None = None, output_dir: str | None = None
) -> Path:
    """Prepare full Gym requests; the reference answer is never part of model input."""
    source = Path(dataset_dir) if dataset_dir else find_dataset_dir()
    selected = list(LANGUAGES) if languages is None else list(languages)
    if not selected or len(set(selected)) != len(selected) or set(selected) - set(LANGUAGES):
        raise ValueError(f"languages must be a nonempty, unique subset of {LANGUAGES}")
    manifest = json.loads((BENCHMARK_DIR / "answers.json").read_text())
    prompt_config = load_prompt_config(str(PROMPT_PATH))
    answers = {row["row_id"]: row for row in manifest["answers"]}
    english = pq.read_table(source / "data/en/sample.parquet").to_pylist()
    if len(english) != len(answers) or {row["row_id"] for row in english} != set(answers):
        raise ValueError("English problem IDs do not match the answer key")
    english_by_id = {row["row_id"]: row for row in english}
    for row in english:
        digest = hashlib.sha256(row["problem"].encode()).hexdigest()
        if digest != answers[row["row_id"]]["english_problem_sha256"]:
            raise ValueError(f"English statement changed for {row['row_id']}; review the answer key before evaluation")

    rows = []
    for code in selected:
        translated = pq.read_table(source / "data" / code / "sample.parquet").to_pylist()
        if len(translated) != len(answers) or {row["row_id"] for row in translated} != set(answers):
            raise ValueError(f"Problem IDs for {code} do not match the answer key")
        for row in translated:
            if row["language_code"] != code or row["row_index"] != english_by_id[row["row_id"]]["row_index"]:
                raise ValueError(f"Language/index mismatch for {code}/{row['row_id']}")
            if not row["problem"].strip():
                raise ValueError(f"Empty problem for {code}/{row['row_id']}")
            answer = answers[row["row_id"]]
            rows.append(
                apply_prompt_to_row(
                    {
                        **row,
                        "task_id": f"{code}/{row['row_id']}",
                        "question": row["problem"],
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
        "dataset_dir": str(source.resolve()),
        "answer_key_version": manifest["version"],
        "protocol": "exact_boxed_text_no_tools",
        "rows": len(rows),
        "unique_problems": len(answers),
        "languages": selected,
        "human_evaluation_pending": sum(row["human_evaluation_pending"] for row in rows),
        "failed_translation_review": sum(row["judge_pass_stage"].startswith("failed_") for row in rows),
        "parquet_sha256": {
            code: hashlib.sha256((source / "data" / code / "sample.parquet").read_bytes()).hexdigest()
            for code in dict.fromkeys(["en", *selected])
        },
    }
    (destination / "preparation.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(f"Prepared {len(rows)} rows ({len(answers)} problems, {len(selected)} languages): {combined}")
    return raw_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir")
    parser.add_argument("--languages", nargs="+", choices=LANGUAGES)
    parser.add_argument("--output-dir")
    prepare(**vars(parser.parse_args()))
