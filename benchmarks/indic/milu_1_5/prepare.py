# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare MILU-1.5 for zero-shot, generated-answer post-training evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import yaml
from huggingface_hub import hf_hub_download


DIRECTORY = Path(__file__).resolve().parent
SOURCE_ID = "ai4bharat/MILU-1.5"
# Public Hugging Face dataset commit, not a credential.
SOURCE_REVISION = "d8a3cc2b4223989399adef5c2685915099e371a5"  # pragma: allowlist secret
LANGUAGES = {
    "en": "English",
    "bn": "Bengali",
    "hi": "Hindi",
    "ta": "Tamil",
    "te": "Telugu",
    "ml": "Malayalam",
    "kn": "Kannada",
    "mr": "Marathi",
    "gu": "Gujarati",
    "pa": "Punjabi",
    "or": "Odia",
}
OPTION_FIELDS = ("option1", "option2", "option3", "option4")
COUNTS = {
    "en": 13432,
    "bn": 6343,
    "hi": 14681,
    "ta": 6098,
    "te": 7031,
    "ml": 3935,
    "kn": 5967,
    "mr": 6652,
    "gu": 4498,
    "pa": 3768,
    "or": 4170,
}


def validate_rows(rows: Sequence[Mapping[str, object]], *, language: str) -> None:
    """Validate MILU source fields before preparing prompts or labels."""
    for index, row in enumerate(rows):
        for field in ("question", *OPTION_FIELDS, "target", "domain", "subject", "language"):
            value = row.get(field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"MILU-1.5 {language}/{index}: missing or empty {field}")
        if row["target"] not in OPTION_FIELDS or row["language"] != LANGUAGES[language]:
            raise ValueError(f"MILU-1.5 {language}/{index}: invalid target or language")
        if type(row.get("is_translated")) is not bool:
            raise ValueError(f"MILU-1.5 {language}/{index}: invalid is_translated flag")


def build_rows(
    records: Sequence[Mapping[str, Any]],
    *,
    language: str,
    question_ids: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """Keep source option order and identifiers when selecting a test subset."""
    if language not in LANGUAGES:
        raise ValueError(f"Unknown MILU-1.5 language code: {language}")
    validate_rows(records, language=language)
    known = {str(i) for i in range(len(records))}
    if question_ids is not None and (
        isinstance(question_ids, str)
        or not question_ids
        or any(not isinstance(index, str) for index in question_ids)
        or len(set(question_ids)) != len(question_ids)
        or set(question_ids) - known
    ):
        raise ValueError("question_ids must contain unique existing string row indices")
    wanted = set(question_ids) if question_ids is not None else known
    rows = []
    for index, row in enumerate(records):
        if str(index) not in wanted:
            continue
        rows.append(
            {
                "question": row["question"].strip(),
                **{field: row[field] for field in OPTION_FIELDS},
                # Responses has no stop field; the chat adapter forwards extra_body.
                "responses_create_params": {
                    "metadata": {"extra_body": json.dumps({"stop": ["</s>", "<|im_end|>", "<eos>"]})}
                },
                "options": [{label: row[field]} for label, field in zip("ABCD", OPTION_FIELDS, strict=True)],
                "expected_answer": "ABCD"[OPTION_FIELDS.index(row["target"])],
                "uuid": f"milu-1.5-{SOURCE_REVISION}-{language}-{index}",
                "metadata": {
                    "language": language,
                    "domain": row["domain"],
                    "subject": row["subject"],
                    "is_translated": row["is_translated"],
                    "source_dataset": SOURCE_ID,
                    "source_revision": SOURCE_REVISION,
                    "source_split": "test",
                    "source_row_index": index,
                    "num_fewshot": 0,
                    "prompt_protocol": "milu_v2",
                },
            }
        )
    return rows


def prepare(
    output_fpath: str | Path | None = None,
    *,
    dataset_dir: str | Path | None = None,
    languages: Sequence[str] | None = None,
    question_ids: Sequence[str] | None = None,
) -> Path:
    """Read a local MILU-1.5 snapshot or download its pinned test Parquet files.

    Check and serialize one language at a time, then atomically replace the output.
    Any error leaves the previous prepared dataset intact. Validation data is unused.
    """
    if isinstance(languages, str):
        raise ValueError("languages must be a list of language codes")
    selected = list(LANGUAGES if languages is None else languages)
    if not selected or len(set(selected)) != len(selected) or set(selected) - set(LANGUAGES):
        raise ValueError(f"Select unique language codes from {list(LANGUAGES)}")
    import pyarrow.parquet as pq

    checksums = yaml.safe_load((DIRECTORY / "checksums.yaml").read_text(encoding="utf-8"))
    output_path = Path(output_fpath) if output_fpath else DIRECTORY / "data/milu_1_5_benchmark.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            for language in selected:
                filename = f"{LANGUAGES[language]}/test-00000-of-00001.parquet"
                path = (
                    Path(dataset_dir) / filename
                    if dataset_dir is not None
                    else Path(
                        hf_hub_download(
                            repo_id=SOURCE_ID, filename=filename, repo_type="dataset", revision=SOURCE_REVISION
                        )
                    )
                )
                with path.open("rb") as source:
                    digest = hashlib.file_digest(source, "sha256").hexdigest()
                if digest != checksums[filename]:
                    raise ValueError(f"MILU-1.5 {path}: SHA-256 differs from release {SOURCE_REVISION}")
                records = pq.read_table(path).to_pylist()
                if len(records) != COUNTS[language]:
                    raise ValueError(f"MILU-1.5 {filename}: expected {COUNTS[language]} rows, received {len(records)}")
                for row in build_rows(records, language=language, question_ids=question_ids):
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        temporary.replace(output_path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return output_path


def main() -> None:
    """Prepare all languages or a reproducible subset from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path)
    parser.add_argument("--languages", nargs="+", choices=list(LANGUAGES))
    parser.add_argument("--question-ids", nargs="+")
    parser.add_argument("--output-path", type=Path)
    args = parser.parse_args()
    print(
        prepare(
            args.output_path,
            dataset_dir=args.dataset_dir,
            languages=args.languages,
            question_ids=args.question_ids,
        )
    )


if __name__ == "__main__":  # pragma: no cover
    main()
