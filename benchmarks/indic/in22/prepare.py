# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare pinned IN22-Gen or IN22-Conv translation pairs for NeMo Gym."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


DIRECTORY = Path(__file__).resolve().parent
ENGLISH_COLUMN = "eng_Latn"
LANGUAGES = {
    "assamese": ("asm_Beng", "Assamese", "as"),
    "bengali": ("ben_Beng", "Bengali", "bn"),
    "gujarati": ("guj_Gujr", "Gujarati", "gu"),
    "hindi": ("hin_Deva", "Hindi", "hi"),
    "kannada": ("kan_Knda", "Kannada", "kn"),
    "malayalam": ("mal_Mlym", "Malayalam", "ml"),
    "marathi": ("mar_Deva", "Marathi", "mr"),
    "nepali": ("npi_Deva", "Nepali", "ne"),
    "odiya": ("ory_Orya", "Odia", "or"),
    "punjabi": ("pan_Guru", "Punjabi", "pa"),
    "tamil": ("tam_Taml", "Tamil", "ta"),
    "telugu": ("tel_Telu", "Telugu", "te"),
    "urdu": ("urd_Arab", "Urdu", "ur"),
}
DIRECTIONS = ("en_xx", "xx_en")
ALL_TRANSLATION_COLUMNS = {
    "asm_Beng",
    "ben_Beng",
    "brx_Deva",
    "doi_Deva",
    "eng_Latn",
    "gom_Deva",
    "guj_Gujr",
    "hin_Deva",
    "kan_Knda",
    "kas_Arab",
    "mai_Deva",
    "mal_Mlym",
    "mar_Deva",
    "mni_Mtei",
    "npi_Deva",
    "ory_Orya",
    "pan_Guru",
    "san_Deva",
    "sat_Olck",
    "snd_Deva",
    "tam_Taml",
    "tel_Telu",
    "urd_Arab",
}
SUBSETS = {
    "gen": {
        "repo_id": "ai4bharat/IN22-Gen",
        "revision": "e042ab3d3063110b1a85efa0a59bdbf8553bb928",
        "rows": 1024,
        "metadata_columns": {"context", "source", "url", "domain", "num_words", "bucket"},
    },
    "conv": {
        "repo_id": "ai4bharat/IN22-Conv",
        "revision": "18cd45870ff0a9e65df9b80dbbcc615eec0e4899",
        "rows": 1503,
        "metadata_columns": {"doc_id", "sent_id", "topic", "domain", "prompt", "scenario", "speaker", "turn"},
    },
}


def _validate_records(records: Sequence[Mapping[str, Any]], *, subset: str) -> None:
    spec = SUBSETS[subset]
    if len(records) != spec["rows"]:
        raise ValueError(f"IN22-{subset}: expected {spec['rows']} rows, received {len(records)}")
    expected_columns = ALL_TRANSLATION_COLUMNS | spec["metadata_columns"]
    required_columns = {ENGLISH_COLUMN, *(details[0] for details in LANGUAGES.values())}
    for index, row in enumerate(records):
        if set(row) != expected_columns:
            raise ValueError(f"IN22-{subset}/{index}: unexpected source schema")
        if any(not isinstance(row[column], str) or not row[column].strip() for column in required_columns):
            raise ValueError(f"IN22-{subset}/{index}: selected translations must be nonempty strings")


def build_rows(
    records: Sequence[Mapping[str, Any]],
    *,
    subset: str,
    languages: Sequence[str] | None = None,
    directions: Sequence[str] | None = None,
    question_ids: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """Build directed pairs in the same task/language order as the reference runner."""
    if subset not in SUBSETS:
        raise ValueError(f"subset must be one of {list(SUBSETS)}")
    selected_languages = list(LANGUAGES if languages is None else languages)
    selected_directions = list(DIRECTIONS if directions is None else directions)
    if (
        not selected_languages
        or len(set(selected_languages)) != len(selected_languages)
        or set(selected_languages) - set(LANGUAGES)
    ):
        raise ValueError(f"Select unique languages from {list(LANGUAGES)}")
    if (
        not selected_directions
        or len(set(selected_directions)) != len(selected_directions)
        or set(selected_directions) - set(DIRECTIONS)
    ):
        raise ValueError(f"Select unique directions from {list(DIRECTIONS)}")
    _validate_records(records, subset=subset)
    known_ids = {str(index) for index in range(len(records))}
    if question_ids is not None and (
        isinstance(question_ids, str)
        or not question_ids
        or any(not isinstance(value, str) for value in question_ids)
        or len(set(question_ids)) != len(question_ids)
        or set(question_ids) - known_ids
    ):
        raise ValueError("question_ids must contain unique existing string row indices")
    wanted = set(question_ids) if question_ids is not None else known_ids

    rows = []
    for direction in selected_directions:
        for language in selected_languages:
            column, display_name, indic_code = LANGUAGES[language]
            if direction == "en_xx":
                source_column, target_column = ENGLISH_COLUMN, column
                source_code, target_code = "en", indic_code
                source_name, target_name = "English", display_name
            else:
                source_column, target_column = column, ENGLISH_COLUMN
                source_code, target_code = indic_code, "en"
                source_name, target_name = display_name, "English"
            for index, record in enumerate(records):
                if str(index) not in wanted:
                    continue
                rows.append(
                    {
                        "text": record[source_column].strip(),
                        "translation": record[target_column].strip(),
                        "source_language": source_code,
                        "target_language": target_code,
                        "source_lang_name": source_name,
                        "target_lang_name": target_name,
                    }
                )
    return rows


def _load_records(*, subset: str) -> list[dict[str, Any]]:
    from datasets import load_dataset

    spec = SUBSETS[subset]
    return list(
        load_dataset(
            spec["repo_id"],
            "default",
            split="test",
            revision=spec["revision"],
        )
    )


def prepare(
    output_fpath: str | Path | None = None,
    *,
    subset: str = "gen",
    languages: Sequence[str] | None = None,
    directions: Sequence[str] | None = None,
    question_ids: Sequence[str] | None = None,
) -> Path:
    """Download one pinned IN22 subset and write both translation directions."""
    if subset not in SUBSETS:
        raise ValueError(f"subset must be one of {list(SUBSETS)}")
    if isinstance(languages, str) or isinstance(directions, str):
        raise ValueError("languages and directions must be lists")
    rows = build_rows(
        _load_records(subset=subset),
        subset=subset,
        languages=languages,
        directions=directions,
        question_ids=question_ids,
    )
    output_path = Path(output_fpath) if output_fpath else DIRECTORY / f"data/in22_{subset}_benchmark.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset", choices=list(SUBSETS), default="gen")
    parser.add_argument("--languages", nargs="+", choices=list(LANGUAGES))
    parser.add_argument("--directions", nargs="+", choices=list(DIRECTIONS))
    parser.add_argument("--question-ids", nargs="+")
    parser.add_argument("--output-path", type=Path)
    args = parser.parse_args()
    print(
        prepare(
            args.output_path,
            subset=args.subset,
            languages=args.languages,
            directions=args.directions,
            question_ids=args.question_ids,
        )
    )


if __name__ == "__main__":  # pragma: no cover
    main()
