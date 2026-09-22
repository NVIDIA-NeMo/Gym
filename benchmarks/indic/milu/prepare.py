# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare MILU tasks for answer-letter likelihood evaluation."""

from __future__ import annotations

import argparse
import json
import random
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from huggingface_hub import hf_hub_download


DIRECTORY = Path(__file__).resolve().parent
SOURCE_ID = "ai4bharat/MILU"
SOURCE_REVISION = "846d7016558b2004386f30e59a2158731262492f"
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
COUNTS = {
    "en": (13535, 813),
    "bn": (6637, 812),
    "hi": (14831, 812),
    "ta": (6372, 813),
    "te": (7304, 812),
    "ml": (4321, 813),
    "kn": (6234, 812),
    "mr": (6924, 810),
    "gu": (4826, 813),
    "pa": (4099, 811),
    "or": (4525, 812),
}
OPTION_FIELDS = ("option1", "option2", "option3", "option4")


def render_question(row: Mapping[str, Any]) -> str:
    """Render a MILU question and its labeled options."""
    return (
        row["question"].strip()
        + r"\n"
        + "".join(rf"{label}. {row[field]}\n" for label, field in zip("ABCD", OPTION_FIELDS, strict=True))
        + "Answer:"
    )


def validate_rows(rows: Sequence[Mapping[str, Any]], *, language: str) -> None:
    """Validate required MILU source fields."""
    for index, row in enumerate(rows):
        for field in ("question", *OPTION_FIELDS, "target", "domain", "subject", "language"):
            if not isinstance(row.get(field), str) or not row[field].strip():
                raise ValueError(f"MILU {language}/{index}: missing or empty {field}")
        if row["target"] not in OPTION_FIELDS or row["language"] != LANGUAGES[language]:
            raise ValueError(f"MILU {language}/{index}: invalid target or language")
        if type(row.get("is_translated")) is not bool:
            raise ValueError(f"MILU {language}/{index}: invalid is_translated flag")


def build_rows(
    records: Sequence[Mapping[str, Any]],
    validation: Sequence[Mapping[str, Any]],
    *,
    language: str,
    num_fewshot: int = 5,
    fewshot_seed: int = 42,
    question_ids: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """Build MCQA rows with seeded random validation demonstrations."""
    if type(num_fewshot) is not int or num_fewshot < 0 or num_fewshot > len(validation):
        raise ValueError("num_fewshot must be nonnegative and fit in the validation split")
    if type(fewshot_seed) is not int:
        raise ValueError("fewshot_seed must be an integer")
    validate_rows(records, language=language)
    validate_rows(validation, language=language)
    known = {str(i) for i in range(len(records))}
    if question_ids is not None and (
        isinstance(question_ids, str)
        or not question_ids
        or len(set(question_ids)) != len(question_ids)
        or set(question_ids) - known
    ):
        raise ValueError("question_ids must contain unique existing string row indices")
    wanted = set(question_ids) if question_ids is not None else known
    result = []
    sampler = random.Random(fewshot_seed)
    for index, row in enumerate(records):
        sampled_ids = sampler.sample(range(len(validation)), num_fewshot)
        if str(index) not in wanted:
            continue
        demonstrations = [(i, validation[i]) for i in sampled_ids if validation[i] != row]
        prompt = "".join(
            render_question(doc) + " " + "ABCD"[OPTION_FIELDS.index(doc["target"])] + "\n\n"
            for _, doc in demonstrations
        )
        prompt += render_question(row)
        continuations = [" " + label for label in "ABCD"]
        result.append(
            {
                "prompt": prompt,
                "choices": continuations,
                "responses_create_params": {},
                "options": [{label: row[field]} for label, field in zip("ABCD", OPTION_FIELDS, strict=True)],
                "expected_answer": "ABCD"[OPTION_FIELDS.index(row["target"])],
                "uuid": f"milu-{SOURCE_REVISION}-{language}-{index}",
                "metadata": {
                    "language": language,
                    "domain": row["domain"],
                    "subject": row["subject"],
                    "is_translated": row["is_translated"],
                    "source_revision": SOURCE_REVISION,
                    "num_fewshot": num_fewshot,
                    "fewshot_sampler": "default",
                    "fewshot_seed": fewshot_seed,
                    "fewshot_ids": [str(i) for i, _ in demonstrations],
                },
            }
        )
    return result


def prepare(
    output_fpath: str | Path | None = None,
    *,
    languages: Sequence[str] | None = None,
    num_fewshot: int = 5,
    fewshot_seed: int = 42,
    question_ids: Sequence[str] | None = None,
) -> Path:
    """Prepare the selected languages using a pinned dataset revision."""
    if isinstance(languages, str):
        raise ValueError("languages must be a list of language codes")
    selected = list(LANGUAGES if languages is None else languages)
    if not selected or len(set(selected)) != len(selected) or set(selected) - set(LANGUAGES):
        raise ValueError(f"Select unique language codes from {list(LANGUAGES)}")
    if type(num_fewshot) is not int or num_fewshot < 0:
        raise ValueError("num_fewshot must be a nonnegative integer")
    import pyarrow.parquet as pq

    rows = []
    for language in selected:
        splits = {}
        for split, count in zip(("test", "validation"), COUNTS[language], strict=True):
            filename = f"{LANGUAGES[language]}/{split}-00000-of-00001.parquet"
            path = hf_hub_download(repo_id=SOURCE_ID, filename=filename, repo_type="dataset", revision=SOURCE_REVISION)
            records = pq.read_table(path).to_pylist()
            if len(records) != count:
                raise ValueError(f"MILU {filename}: expected {count} rows, received {len(records)}")
            splits[split] = records
        rows.extend(
            build_rows(
                splits["test"],
                splits["validation"],
                language=language,
                num_fewshot=num_fewshot,
                fewshot_seed=fewshot_seed,
                question_ids=question_ids,
            )
        )
    output_path = Path(output_fpath) if output_fpath else DIRECTORY / "data/milu_benchmark.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return output_path


def main() -> None:
    """Prepare all official languages or an explicitly selected subset."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--languages", nargs="+", choices=list(LANGUAGES))
    parser.add_argument("--num-fewshot", type=int, default=5)
    parser.add_argument("--fewshot-seed", type=int, default=42)
    parser.add_argument("--question-ids", nargs="+")
    parser.add_argument("--output-path", type=Path)
    args = parser.parse_args()
    print(
        prepare(
            args.output_path,
            languages=args.languages,
            num_fewshot=args.num_fewshot,
            fewshot_seed=args.fewshot_seed,
            question_ids=args.question_ids,
        )
    )


if __name__ == "__main__":  # pragma: no cover
    main()
