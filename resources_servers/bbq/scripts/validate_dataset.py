#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Validate an arbitrary bbq_two_judge JSONL dataset against docs/DATA_SPEC.md.

Row-level checks enforce the Gym shape and the ExpectedValues contract.
Dataset-level checks enforce the anti-shortcut gates from DATA_SPEC section 5
that are decidable from the JSONL alone. Mix targets (section 3) are reported,
not failed, because tolerances are only meaningful at scale.

Exit status is nonzero when any check fails.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from util import ExpectedValues  # noqa: E402

LEAKAGE_FIELDS = {
    "thinking",
    "response_output",
    "response_output_thinking",
    "input_output",
}
MCQ_OPTION_PATTERN = re.compile(r"(?im)^\s*(?:\(?[A-E]\)|[A-E][.)])\s+\S")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            raise SystemExit(f"{path}:{line_number}: blank JSONL line")
        value = json.loads(line)
        if not isinstance(value, dict):
            raise SystemExit(f"{path}:{line_number}: expected a JSON object")
        rows.append(value)
    return rows


def all_keys(value: Any) -> set[str]:
    if isinstance(value, dict):
        return set(value) | set().union(*(all_keys(child) for child in value.values()))
    if isinstance(value, list):
        return set().union(*(all_keys(child) for child in value)) if value else set()
    return set()


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().casefold()


def normalize_label(value: str) -> str:
    normalized = normalize_text(value)
    return re.sub(r"^the\s+", "", normalized)


def content_words(value: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", normalize_text(value))


def ngrams(words: list[str], size: int) -> set[tuple[str, ...]]:
    return {tuple(words[i : i + size]) for i in range(len(words) - size + 1)}


def policy_visible_body(row: dict[str, Any]) -> str:
    """The row input minus the fixed developer instruction (boilerplate)."""

    body = row["input"]
    params_input = (row.get("responses_create_params") or {}).get("input")
    if isinstance(params_input, list):
        for item in params_input:
            if isinstance(item, dict) and item.get("role") == "developer":
                content = item.get("content")
                if isinstance(content, str) and content:
                    body = body.replace(content, " ")
    return body


class Failures:
    def __init__(self) -> None:
        self.messages: list[str] = []

    def check(self, condition: bool, message: str) -> None:
        if not condition:
            self.messages.append(message)


def validate_rows(rows: list[dict[str, Any]], failures: Failures) -> list[ExpectedValues]:
    parsed: list[ExpectedValues] = []
    for line_number, row in enumerate(rows, 1):
        where = f"row {line_number}"
        failures.check(row.get("task_name") == "bbq_two_judge", f"{where}: wrong task_name")
        failures.check(
            isinstance(row.get("input"), str) and bool(row["input"].strip()),
            f"{where}: input must be a nonempty string",
        )
        failures.check(row.get("output") == "", f"{where}: output must be empty")
        failures.check(
            not (all_keys(row) & LEAKAGE_FIELDS),
            f"{where}: SFT-target leakage field present",
        )
        if isinstance(row.get("input"), str):
            failures.check(
                not MCQ_OPTION_PATTERN.search(row["input"]),
                f"{where}: input contains lettered answer options (DATA_SPEC section 2)",
            )
        try:
            expected = ExpectedValues.model_validate(row.get("expected_values"))
        except Exception as exc:
            failures.check(False, f"{where}: expected_values invalid: {exc}")
            continue
        if isinstance(row.get("input"), str):
            failures.check(
                expected.question in row["input"],
                f"{where}: visible question missing from input",
            )
        parsed.append(expected)
    return parsed


def validate_dataset_gates(
    rows: list[dict[str, Any]],
    parsed: list[ExpectedValues],
    failures: Failures,
    *,
    max_actor_label_rows: int,
    max_question_opening_rows: int,
    max_five_gram_rows: int,
    question_opening_words: int,
) -> None:
    failures.check(
        len({item.provenance_id for item in parsed}) == len(parsed),
        "provenance_id values must be unique",
    )

    label_rows: dict[tuple[str, str], int] = Counter()
    actor_sets: dict[str, Counter] = defaultdict(Counter)
    question_openings: dict[tuple[str, tuple[str, ...]], int] = Counter()
    shingle_rows: dict[tuple[str, ...], set[int]] = defaultdict(set)
    five_gram_rows: dict[tuple[str, ...], set[int]] = defaultdict(set)

    for index, (row, expected) in enumerate(zip(rows, parsed, strict=False), 1):
        row_labels = {
            normalize_label(aliases[0]) for aliases in expected.actor_aliases.values() if aliases
        }
        for label in row_labels:
            label_rows[(expected.category, label)] += 1
        actor_sets[expected.category][tuple(sorted(row_labels))] += 1

        opening = tuple(content_words(expected.question)[:question_opening_words])
        if opening:
            question_openings[(expected.category, opening)] += 1

        words = content_words(policy_visible_body(row))
        for shingle in ngrams(words, 8):
            shingle_rows[shingle].add(index)
        for gram in ngrams(words, 5):
            five_gram_rows[gram].add(index)

    for (category, label), count in sorted(label_rows.items()):
        failures.check(
            count <= max_actor_label_rows,
            f"actor label {label!r} appears in {count} {category} rows "
            f"(cap {max_actor_label_rows})",
        )
    for category, sets in sorted(actor_sets.items()):
        for actor_set, count in sorted(sets.items()):
            failures.check(
                count <= 1,
                f"actor set {actor_set} appears in {count} {category} rows (must be unique)",
            )
    for (category, opening), count in sorted(question_openings.items()):
        failures.check(
            count <= max_question_opening_rows,
            f"question opening {' '.join(opening)!r} appears in {count} {category} rows "
            f"(cap {max_question_opening_rows})",
        )
    for shingle, indices in sorted(shingle_rows.items()):
        failures.check(
            len(indices) <= 1,
            f"8-word shingle {' '.join(shingle)!r} repeats across rows {sorted(indices)}",
        )
    for gram, indices in sorted(five_gram_rows.items()):
        failures.check(
            len(indices) <= max_five_gram_rows,
            f"5-gram {' '.join(gram)!r} appears in rows {sorted(indices)} "
            f"(cap {max_five_gram_rows})",
        )


def report_mix(parsed: list[ExpectedValues]) -> None:
    total = len(parsed)
    ambig = sum(1 for item in parsed if item.context_condition == "ambig")
    disambig = total - ambig
    stereo = sum(1 for item in parsed if item.disambig_direction == "stereo_consistent")
    neg = sum(1 for item in parsed if item.question_polarity == "neg")
    print(f"rows: {total}")
    print(f"categories: {dict(sorted(Counter(item.category for item in parsed).items()))}")
    print(f"ambig: {ambig} ({ambig / total:.1%})  disambig: {disambig} ({disambig / total:.1%})")
    if disambig:
        print(
            f"disambig stereo_consistent: {stereo} ({stereo / disambig:.1%} of disambig); "
            f"counter_stereo: {disambig - stereo}"
        )
    print(f"polarity neg: {neg} ({neg / total:.1%})")
    print(f"actor counts: {dict(sorted(Counter(item.actor_count for item in parsed).items()))}")
    print(
        f"render modes: {dict(sorted(Counter(item.source_render_mode for item in parsed).items()))}"
    )
    print("mix targets are reported only; see docs/DATA_SPEC.md section 3")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Gym-ready bbq_two_judge JSONL file")
    parser.add_argument("--max-actor-label-rows", type=int, default=2)
    parser.add_argument("--max-question-opening-rows", type=int, default=3)
    parser.add_argument("--max-five-gram-rows", type=int, default=2)
    parser.add_argument("--question-opening-words", type=int, default=4)
    args = parser.parse_args()

    rows = load_jsonl(args.input)
    if not rows:
        print("FAIL: dataset is empty")
        return 1

    failures = Failures()
    parsed = validate_rows(rows, failures)
    if len(parsed) == len(rows):
        validate_dataset_gates(
            rows,
            parsed,
            failures,
            max_actor_label_rows=args.max_actor_label_rows,
            max_question_opening_rows=args.max_question_opening_rows,
            max_five_gram_rows=args.max_five_gram_rows,
            question_opening_words=args.question_opening_words,
        )

    if parsed:
        report_mix(parsed)
    if failures.messages:
        print(f"\nFAIL: {len(failures.messages)} violation(s):")
        for message in failures.messages:
            print(f"- {message}")
        return 1
    versions = sorted({item.schema_version for item in parsed})
    print(f"\nValidated {len(rows)} rows against the {'/'.join(versions)} contract and gates.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
