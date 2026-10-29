# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare zero-shot chat evaluation from the local, wide Indic GSM8K corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import uuid
from pathlib import Path


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
DEFAULT_SOURCE = Path("data/indic-gsm8k/test.parquet")
OUTPUT = Path(__file__).parent / "data/indic_gsm8k_posttrain.jsonl"
OUTPUT_TOKENS = 120000
FINAL_ANSWER = re.compile(r"####\s+(-?[0-9][0-9,.]*)\s*$")


def build_rows(source: list[dict], *, languages: list[str] | None = None, source_sha256: str) -> list[dict]:
    """Expand translated questions while retaining original numeric answer keys."""
    selected = list(LANGUAGES) if languages is None else languages
    if not isinstance(selected, list) or not selected or len(set(selected)) != len(selected):
        raise ValueError("languages must be a nonempty list of unique language codes")
    if set(selected) - LANGUAGES.keys():
        raise ValueError("Unsupported language code")
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
    *, source: Path | str = DEFAULT_SOURCE, output: Path | str = OUTPUT, languages: list[str] | None = None
) -> Path:
    """Write chat tasks from a local Parquet source, recording its checksum."""
    import pyarrow.parquet as pq

    source, output = Path(source), Path(output)
    checksum = hashlib.sha256(source.read_bytes()).hexdigest()
    rows = build_rows(pq.read_table(source).to_pylist(), languages=languages, source_sha256=checksum)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp")
    temporary.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    temporary.replace(output)
    print(f"Wrote {len(rows)} zero-shot chat tasks from {source} (sha256={checksum})")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--languages", nargs="+", choices=LANGUAGES)
    args = parser.parse_args()
    prepare(source=args.source, output=args.output, languages=args.languages)
