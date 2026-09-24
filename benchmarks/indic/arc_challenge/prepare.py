# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare Indic ARC-Challenge for five-shot answer-letter likelihood scoring."""

from __future__ import annotations

import argparse
import json
import random
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from huggingface_hub import hf_hub_download


DIRECTORY = Path(__file__).resolve().parent
SOURCE_ID = "anushakamathofficial/indic_ARC-Challenge"
SOURCE_REVISION = "7d8d96ef635a28a51809deeb6490257eecfe1611"
CANONICAL_SOURCE_ID = "allenai/ai2_arc"
CANONICAL_SOURCE_REVISION = "210d026faf9955653af8916fad021475a3f00453"
CANONICAL_TEST_FILE = "ARC-Challenge/test-00000-of-00001.parquet"
CANONICAL_TRAIN_FILE = "ARC-Challenge/train-00000-of-00001.parquet"
LANGUAGES = {
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
DEFAULT_LANGUAGES = tuple(language for language in LANGUAGES if language != "en")
TEST_COUNT = 1172
TRAIN_COUNT = 1119
NUM_FEWSHOT = 5
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
QUALITY_STAGES = {
    "english_source",
    "first_judge_pass",
    "passed_after_correction",
    "failed_after_correction_review_needed",
    "failed_first_judge_review_needed",
}


def _choices(row: Mapping[str, Any]) -> tuple[list[str], list[str], str]:
    """Return positional letters, stripped option text, and the positional gold letter."""
    choices = row.get("choices")
    if not isinstance(choices, Mapping):
        raise ValueError("ARC choices must contain text and label lists")
    texts = choices.get("text")
    source_labels = choices.get("label")
    if (
        isinstance(texts, (str, bytes))
        or isinstance(source_labels, (str, bytes))
        or not isinstance(texts, Sequence)
        or not isinstance(source_labels, Sequence)
        or not 2 <= len(texts) <= len(LETTERS)
        or len(texts) != len(source_labels)
        or any(not isinstance(label, str) or not label for label in source_labels)
        or len(set(source_labels)) != len(source_labels)
    ):
        raise ValueError("ARC requires two to 26 uniquely labeled choices")
    if any(not isinstance(text, str) or not text.strip() for text in texts):
        raise ValueError("ARC choice text must be nonempty")
    answer = row.get("answerKey")
    if answer not in source_labels:
        raise ValueError("ARC answerKey must identify one source choice")
    labels = list(LETTERS[: len(texts)])
    return labels, [text.strip() for text in texts], labels[source_labels.index(answer)]


def render_question(row: Mapping[str, Any]) -> str:
    """Render the raw-completion prompt used by the reference task YAML."""
    question = row.get("question")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("ARC question must be nonempty")
    labels, texts, _ = _choices(row)
    options = "\n".join(f"{label}. {text}" for label, text in zip(labels, texts, strict=True))
    return f"Question: {question.strip()}\n{options}\nAnswer:"


def _validate_test_rows(
    records: Sequence[Mapping[str, Any]],
    english: Sequence[Mapping[str, Any]],
    *,
    language: str,
) -> None:
    if len(records) != len(english):
        raise ValueError(f"ARC {language}: translated and English row counts differ")
    seen: set[str] = set()
    expected_fields = {
        "id",
        "question",
        "choices",
        "answerKey",
        "language",
        "language_code",
        "judge_pass_stage",
    }
    for index, (row, original) in enumerate(zip(records, english, strict=True)):
        if set(row) != expected_fields:
            raise ValueError(f"ARC {language}/{index}: unexpected source schema")
        task_id = row.get("id")
        if not isinstance(task_id, str) or not task_id or task_id in seen:
            raise ValueError(f"ARC {language}/{index}: missing or duplicate id")
        seen.add(task_id)
        if task_id != original.get("id"):
            raise ValueError(f"ARC {language}/{task_id}: source order differs from English")
        if row.get("language_code") != language or row.get("language") != LANGUAGES[language]:
            raise ValueError(f"ARC {language}/{task_id}: incorrect language metadata")
        stage = row.get("judge_pass_stage")
        if stage not in QUALITY_STAGES or (stage == "english_source") != (language == "en"):
            raise ValueError(f"ARC {language}/{task_id}: incorrect translation status")
        _choices(row)
        source_choices = row["choices"]
        original_choices = original.get("choices")
        if (
            not isinstance(original_choices, Mapping)
            or row.get("answerKey") != original.get("answerKey")
            or source_choices["label"] != original_choices.get("label")
        ):
            raise ValueError(f"ARC {language}/{task_id}: labels or gold answer differ from English")
        render_question(row)


def _validate_training(rows: Sequence[Mapping[str, Any]]) -> None:
    seen: set[str] = set()
    for index, row in enumerate(rows):
        if set(row) != {"id", "question", "choices", "answerKey"}:
            raise ValueError(f"ARC train/{index}: unexpected source schema")
        task_id = row.get("id")
        if not isinstance(task_id, str) or not task_id or task_id in seen:
            raise ValueError(f"ARC train/{index}: missing or duplicate id")
        seen.add(task_id)
        _choices(row)
        render_question(row)


def build_rows(
    records: Sequence[Mapping[str, Any]],
    english: Sequence[Mapping[str, Any]],
    training: Sequence[Mapping[str, Any]],
    *,
    language: str,
    fewshot_seed: int = 42,
    question_ids: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """Build rows while advancing the seeded sampler in full test-set order."""
    if language not in LANGUAGES:
        raise ValueError(f"Unsupported ARC language: {language}")
    if type(fewshot_seed) is not int:
        raise ValueError("fewshot_seed must be an integer")
    if len(training) < NUM_FEWSHOT:
        raise ValueError(f"ARC needs at least {NUM_FEWSHOT} training examples")
    _validate_test_rows(records, english, language=language)
    _validate_training(training)
    test_ids = {row["id"] for row in english}
    if test_ids & {row["id"] for row in training}:
        raise ValueError("ARC train and test identities overlap")
    known_ids = {row["id"] for row in records}
    if question_ids is not None and (
        isinstance(question_ids, str)
        or not question_ids
        or len(set(question_ids)) != len(question_ids)
        or set(question_ids) - known_ids
    ):
        raise ValueError("question_ids must contain unique existing ARC ids")
    wanted = set(question_ids) if question_ids is not None else known_ids

    sampler = random.Random(fewshot_seed)
    result = []
    for row in records:
        sampled = sampler.sample(range(len(training)), NUM_FEWSHOT)
        task_id = row["id"]
        if task_id not in wanted:
            continue
        labels, texts, gold = _choices(row)
        demonstrations = [render_question(training[index]) + " " + _choices(training[index])[2] for index in sampled]
        prompt = "\n\n".join([*demonstrations, render_question(row)])
        result.append(
            {
                "prompt": prompt,
                "choices": [" " + label for label in labels],
                "responses_create_params": {},
                "options": [{label: text} for label, text in zip(labels, texts, strict=True)],
                "expected_answer": gold,
                "subset_for_metrics": language,
                "uuid": f"indic-arc-challenge-{SOURCE_REVISION}-{language}-{task_id}",
                "metadata": {
                    "language": language,
                    "task_id": task_id,
                    "judge_pass_stage": row["judge_pass_stage"],
                    "source_revision": SOURCE_REVISION,
                    "canonical_source_revision": CANONICAL_SOURCE_REVISION,
                    "source_choice_labels": list(row["choices"]["label"]),
                    "num_fewshot": NUM_FEWSHOT,
                    "fewshot_sampler": "default",
                    "fewshot_seed": fewshot_seed,
                    "fewshot_ids": [training[index]["id"] for index in sampled],
                    "fewshot_language": "en",
                    "fewshot_split": "train",
                },
            }
        )
    return result


def _read_parquet(*, repo_id: str, revision: str, filename: str) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq

    path = hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset", revision=revision)
    return pq.read_table(path).to_pylist()


def prepare(
    output_fpath: str | Path | None = None,
    *,
    languages: Sequence[str] | None = None,
    fewshot_seed: int = 42,
    question_ids: Sequence[str] | None = None,
) -> Path:
    """Prepare pinned multilingual test rows and canonical English demonstrations."""
    if isinstance(languages, str):
        raise ValueError("languages must be a list of language codes")
    selected = list(DEFAULT_LANGUAGES if languages is None else languages)
    if not selected or len(set(selected)) != len(selected) or set(selected) - set(LANGUAGES):
        raise ValueError(f"Select unique language codes from {list(LANGUAGES)}")
    if type(fewshot_seed) is not int:
        raise ValueError("fewshot_seed must be an integer")

    canonical_test = _read_parquet(
        repo_id=CANONICAL_SOURCE_ID,
        revision=CANONICAL_SOURCE_REVISION,
        filename=CANONICAL_TEST_FILE,
    )
    english = _read_parquet(
        repo_id=SOURCE_ID,
        revision=SOURCE_REVISION,
        filename="data/en/test.parquet",
    )
    training = _read_parquet(
        repo_id=CANONICAL_SOURCE_ID,
        revision=CANONICAL_SOURCE_REVISION,
        filename=CANONICAL_TRAIN_FILE,
    )
    if len(canonical_test) != TEST_COUNT or len(english) != TEST_COUNT or len(training) != TRAIN_COUNT:
        raise ValueError(
            f"ARC expected {TEST_COUNT} test and {TRAIN_COUNT} train rows; received "
            f"{len(canonical_test)}, {len(english)}, and {len(training)}"
        )
    for index, (published, canonical) in enumerate(zip(english, canonical_test, strict=True)):
        if any(published.get(field) != canonical.get(field) for field in ("id", "question", "choices", "answerKey")):
            raise ValueError(f"ARC English row {index} differs from the pinned canonical test split")
    _validate_test_rows(english, english, language="en")

    rows = []
    for language in selected:
        records = (
            english
            if language == "en"
            else _read_parquet(
                repo_id=SOURCE_ID,
                revision=SOURCE_REVISION,
                filename=f"data/{language}/test.parquet",
            )
        )
        if len(records) != TEST_COUNT:
            raise ValueError(f"ARC {language}: expected {TEST_COUNT} rows, received {len(records)}")
        rows.extend(
            build_rows(
                records,
                english,
                training,
                language=language,
                fewshot_seed=fewshot_seed,
                question_ids=question_ids,
            )
        )

    output_path = Path(output_fpath) if output_fpath else DIRECTORY / "data/arc_challenge_benchmark.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return output_path


def main() -> None:
    """Prepare all Indic languages or an explicitly selected subset."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--languages", nargs="+", choices=list(LANGUAGES))
    parser.add_argument("--fewshot-seed", type=int, default=42)
    parser.add_argument("--question-ids", nargs="+")
    parser.add_argument("--output-path", type=Path)
    args = parser.parse_args()
    print(
        prepare(
            args.output_path,
            languages=args.languages,
            fewshot_seed=args.fewshot_seed,
            question_ids=args.question_ids,
        )
    )


if __name__ == "__main__":  # pragma: no cover
    main()
