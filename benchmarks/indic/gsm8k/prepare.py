# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare ai4bharat/indic-gsm8k for zero-shot chat evaluation in NeMo Gym."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import uuid
from pathlib import Path

from huggingface_hub import hf_hub_download


SOURCE_ID = "ai4bharat/indic-gsm8k"
SOURCE_REVISION = "e21f3ff7e9fe703a7f4faeefdba27801be278eb5"
SOURCE_FILE = "test.parquet"
SOURCE_SHA256 = "c3ee9883b5afac206456a1ac19d568536e3b6c920f8ea0c57457dda08c50c101"
LANGUAGES = {
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
DEFAULT_LANGUAGES = ("bn", "gu", "hi", "kn", "mr", "ml", "ne", "or", "pa", "ta", "te", "ur")
OUTPUT_FPATH = Path(__file__).parent / "data/indic_gsm8k_benchmark.jsonl"
OUTPUT_TOKENS = 120000
FINAL_ANSWER = re.compile(r"####\s+(-?[0-9][0-9,.]*)\s*$")


def build_rows(source: list[dict], *, languages: list[str] | None = None, source_sha256: str) -> list[dict]:
    """Expand translated questions while retaining original numeric answer keys."""
    selected = list(DEFAULT_LANGUAGES) if languages is None else languages
    if not isinstance(selected, list) or not selected or len(set(selected)) != len(selected):
        raise ValueError("languages must be a nonempty list of unique language codes")
    if set(selected) - LANGUAGES.keys():
        raise ValueError("Unsupported language code")
    if not source:
        raise ValueError("Source dataset must not be empty")
    output, seen = [], set()
    for index, row in enumerate(source):
        source_id = row.get("id")
        if not isinstance(source_id, str) or not source_id or source_id in seen:
            raise ValueError(f"Missing or duplicate source id at row {index}")
        seen.add(source_id)
        reference = row.get("answer")
        match = FINAL_ANSWER.search(reference) if isinstance(reference, str) else None
        if not match:
            raise ValueError(f"Missing original numeric answer at row {index}")
        for code in selected:
            question = row.get(f"question_{LANGUAGES[code]}_translation")
            if not isinstance(question, str) or not question.strip():
                raise ValueError(f"Missing translated question: row={index}, language={code}")
            output.append(
                {
                    "uuid": str(uuid.uuid5(uuid.NAMESPACE_URL, f"indic-gsm8k:{source_sha256}:{source_id}:{code}")),
                    "source_id": source_id,
                    "source_index": index,
                    "source_sha256": source_sha256,
                    "source_repository": SOURCE_ID,
                    "source_revision": SOURCE_REVISION,
                    "question": question,
                    # Original answer keys are authoritative; translated worked solutions can differ.
                    "expected_answer": reference,
                    "language": LANGUAGES[code],
                    "language_code": code,
                    "subset_for_metrics": code,
                    "responses_create_params": {"max_output_tokens": OUTPUT_TOKENS},
                }
            )
    return output


def prepare(
    *, source: Path | str | None = None, output: Path | str = OUTPUT_FPATH, languages: list[str] | None = None
) -> Path:
    """Download the pinned HF test split or use an identical local Parquet file."""
    import pyarrow.parquet as pq

    if source is None:
        source = hf_hub_download(
            repo_id=SOURCE_ID, filename=SOURCE_FILE, repo_type="dataset", revision=SOURCE_REVISION
        )
    source, output = Path(source), Path(output)
    checksum = hashlib.sha256(source.read_bytes()).hexdigest()
    if checksum != SOURCE_SHA256:
        raise ValueError(f"Source checksum differs from {SOURCE_ID}@{SOURCE_REVISION}: {source}")
    rows = build_rows(pq.read_table(source).to_pylist(), languages=languages, source_sha256=checksum)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp")
    temporary.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    temporary.replace(output)
    print(f"Wrote {len(rows)} zero-shot chat tasks from {source} (sha256={checksum})")
    return output


def is_prepared_data_current(fpath: Path) -> bool:
    """Let Gym refresh a cached dataset when its source or generation limit differs."""
    try:
        with Path(fpath).open(encoding="utf-8") as stream:
            row = json.loads(next(stream))
        return (
            row.get("source_repository") == SOURCE_ID
            and row.get("source_revision") == SOURCE_REVISION
            and row.get("source_sha256") == SOURCE_SHA256
            and row.get("responses_create_params", {}).get("max_output_tokens") == OUTPUT_TOKENS
        )
    except (OSError, StopIteration, json.JSONDecodeError, AttributeError):
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, help="Local copy of the pinned HF test.parquet; otherwise download it")
    parser.add_argument("--output", type=Path, default=OUTPUT_FPATH)
    parser.add_argument("--languages", nargs="+", choices=LANGUAGES)
    args = parser.parse_args()
    prepare(source=args.source, output=args.output, languages=args.languages)
