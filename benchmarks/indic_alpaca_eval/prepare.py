# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare aligned Indic AlpacaEval Parquets as native Gym single-turn tasks."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq

from resources_servers.alpaca_pointwise_judge.task_data import LANGUAGE_ALIASES, LANGUAGES, AlpacaTask


SOURCE_REPO = "ai4bharat/indic-alpaca-eval"
SOURCE_REVISION = "94f35aa27454ddcfbf1128b8c8b61093ab284163"  # pragma: allowlist secret
DEFAULT_OUTPUT = Path(__file__).parent / "data"
RUBRIC_PATH = (
    Path(__file__).resolve().parents[2] / "resources_servers/alpaca_pointwise_judge/prompts/alpaca_pointwise_v1.yaml"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def convert_rows(rows: dict[str, list[dict]], *, dataset_hashes: dict[str, str]) -> list[dict]:
    """Preserve language, stable IDs, translation flags, and expected coverage."""
    counts = {code: len(language_rows) for code, language_rows in rows.items()}
    set_id = hashlib.sha256(
        json.dumps(
            {
                "hashes": dataset_hashes,
                "selection": {code: [r["row_id"] for r in items] for code, items in rows.items()},
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()
    output = []
    for code, language_rows in rows.items():
        for row in language_rows:
            metadata = AlpacaTask(
                question=row["instruction"],
                row_id=row["row_id"],
                row_index=row["row_index"],
                language_code=code,
                language_name=LANGUAGES[code],
                expected_counts=counts,
                evaluation_set_id=set_id,
                judge_pass_stage=row["judge_pass_stage"],
                human_evaluation_pending=row["human_evaluation_pending"],
                human_review_reason=row.get("human_review_reason", ""),
                source_sha256=dataset_hashes[code],
            )
            output.append(
                {
                    "id": f"{code}/{row['row_id']}",
                    # Native aggregation groups by this key across language jobs.
                    "_ng_task_index": list(LANGUAGES).index(code) * 805 + row["row_index"],
                    # Independently owned row defaults preserve other benchmarks' metadata
                    # and allow explicit run-level CLI overrides.
                    "responses_create_params": {
                        "input": [{"role": "user", "content": row["instruction"]}],
                        "temperature": 1.0,
                        "top_p": 0.95,
                        "max_output_tokens": 120000,
                        "metadata": {
                            "extra_body": json.dumps({"seed": 42}),
                            "chat_template_kwargs": json.dumps({"enable_thinking": True}),
                        },
                    },
                    "verifier_metadata": metadata.model_dump(),
                }
            )
    return output


def prepare(
    dataset_dir: Path | None = None,
    *,
    output_dir: Path = DEFAULT_OUTPUT,
    languages: list[str] | None = None,
    limit_per_language: int | None = None,
) -> Path:
    """Use local Parquets or the pinned HF snapshot; never vendor benchmark text."""
    if isinstance(languages, str):
        raise ValueError("languages must be a list of language codes")
    languages = [LANGUAGE_ALIASES.get(code, code) for code in (languages if languages is not None else LANGUAGES)]
    if not languages or len(languages) != len(set(languages)) or not set(languages) <= LANGUAGES.keys():
        raise ValueError("Select unique supported language codes")
    if limit_per_language is not None and (type(limit_per_language) is not int or not 1 <= limit_per_language <= 805):
        raise ValueError("limit_per_language must be between 1 and 805")
    local_source = dataset_dir is not None
    if dataset_dir is None:
        from huggingface_hub import snapshot_download

        dataset_dir = Path(
            snapshot_download(
                repo_id=SOURCE_REPO,
                repo_type="dataset",
                revision=SOURCE_REVISION,
                allow_patterns=["eval.parquet"],
            )
        )
    dataset_dir = Path(dataset_dir)
    output_dir = Path(output_dir)
    wide_path = dataset_dir / "eval.parquet"
    if wide_path.exists():
        table = pq.read_table(wide_path)
        columns = {code: f"instruction_{LANGUAGES[code]}_translation" for code in languages}
        missing = {"id", "position", "instruction", *columns.values()} - set(table.column_names)
        if missing:
            raise ValueError(f"Missing source columns: {sorted(missing)}")
        source = table.to_pylist()
        english_path = wide_path
        english = [{"row_id": row["id"], "row_index": row["position"]} for row in source]
        selected = {
            code: [
                {
                    "row_id": row["id"],
                    "row_index": row["position"],
                    "instruction": row[column],
                    "language_code": code,
                    # The published table does not expose per-row review status.
                    "judge_pass_stage": "not_provided",
                    "human_evaluation_pending": None,
                }
                for row in source
            ]
            for code, column in columns.items()
        }
        source_paths = {code: wide_path for code in languages}
    else:
        english_path = dataset_dir / "data/en/eval.parquet"
        english = pq.read_table(english_path).to_pylist()
        source_paths = {code: dataset_dir / "data" / code / "eval.parquet" for code in languages}
        selected = {code: pq.read_table(path).to_pylist() for code, path in source_paths.items()}
    expected = [(row["row_id"], row["row_index"]) for row in english]
    if (
        len(english) != 805
        or any(not isinstance(row["row_id"], str) or not row["row_id"].strip() for row in english)
        or len({row["row_id"] for row in english}) != 805
        or [row["row_index"] for row in english] != list(range(805))
    ):
        raise ValueError("Expected 805 unique English source rows indexed 0–804")
    hashes = {}
    for code in languages:
        path = source_paths[code]
        rows = selected[code]
        if [(row["row_id"], row["row_index"]) for row in rows] != expected:
            raise ValueError(f"{code}: IDs or order do not align with the English source")
        if any(row["language_code"] != code for row in rows):
            raise ValueError(f"{code}: incorrect source language metadata")
        if any(not isinstance(row["instruction"], str) or not row["instruction"].strip() for row in rows):
            raise ValueError(f"{code}: empty instruction")
        selected[code] = rows[:limit_per_language] if limit_per_language else rows
        hashes[code] = sha256(path)
    converted = convert_rows(selected, dataset_hashes=hashes)
    output_path = output_dir / "indic_alpaca_eval_benchmark.jsonl"
    write_rows(output_path, converted)
    for code in languages:
        # Keep a common coverage contract so language-job outputs can be pooled.
        write_rows(
            output_dir / f"{code}.jsonl",
            [row for row in converted if row["verifier_metadata"]["language_code"] == code],
        )
    write_rows(
        output_dir / "smoke.jsonl",
        convert_rows(
            {code: rows[:1] for code, rows in selected.items()},
            dataset_hashes=hashes,
        ),
    )
    manifest = {
        "source_repo": SOURCE_REPO,
        "source_revision": None if local_source else SOURCE_REVISION,
        "local_source": str(dataset_dir.resolve()),
        "english_sha256": sha256(english_path),
        "dataset_sha256": hashes,
        "languages": languages,
        "expected_total": len(converted),
        "expected_counts": {code: len(rows) for code, rows in selected.items()},
        "evaluation_set_id": converted[0]["verifier_metadata"]["evaluation_set_id"],
        "protocol": "indic_alpaca_pointwise_v1",
        "adapter_sha256": sha256(Path(__file__)),
        "rubric_sha256": sha256(RUBRIC_PATH),
        "prepared_sha256": sha256(output_path),
        "translation_review_filter": "none",
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Prepared {len(converted)} tasks across {len(languages)} languages: {output_path}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--languages", nargs="+")
    parser.add_argument("--limit-per-language", type=int)
    args = parser.parse_args()
    prepare(
        args.dataset_dir,
        output_dir=args.output_dir,
        languages=args.languages,
        limit_per_language=args.limit_per_language,
    )


if __name__ == "__main__":
    main()
