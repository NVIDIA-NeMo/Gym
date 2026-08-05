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
"""Validate the five curated prototype fixtures and their reference rollouts."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from util import (  # noqa: E402
    AnswerJudgment,
    ExpectedValues,
    ExplanationJudgment,
    answer_reward,
    explanation_reward,
)

LEAKAGE_FIELDS = {
    "thinking",
    "response_output",
    "response_output_thinking",
    "input_output",
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            raise AssertionError(f"{path}:{line_number}: blank JSONL line")
        value = json.loads(line)
        if not isinstance(value, dict):
            raise AssertionError(f"{path}:{line_number}: expected a JSON object")
        rows.append(value)
    return rows


def canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def all_keys(value: Any) -> set[str]:
    if isinstance(value, dict):
        return set(value) | set().union(*(all_keys(child) for child in value.values()))
    if isinstance(value, list):
        return set().union(*(all_keys(child) for child in value)) if value else set()
    return set()


def contains_absolute_path(value: Any) -> bool:
    if isinstance(value, str):
        return value.startswith("/")
    if isinstance(value, dict):
        return any(contains_absolute_path(child) for child in value.values())
    if isinstance(value, list):
        return any(contains_absolute_path(child) for child in value)
    return False


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_upstream_sources(
    provenance: dict[str, Any],
    examples: list[dict[str, Any]],
    *,
    bbq_sft_root: Path,
    bbq_rlvr_root: Path,
) -> None:
    gym_info = provenance["gym_source"]
    gym_path = bbq_rlvr_root / gym_info["file"]
    assert file_hash(gym_path) == gym_info["source_file_sha256"]
    gym_lines = gym_path.read_text(encoding="utf-8").splitlines()

    upstream_rows: dict[str, list[dict[str, Any]]] = {}
    for relative_path, expected_hash in provenance["upstream_source_files"].items():
        source_path = bbq_sft_root / relative_path
        assert file_hash(source_path) == expected_hash
        payload = json.loads(source_path.read_text(encoding="utf-8"))
        assert isinstance(payload, list)
        upstream_rows[relative_path] = payload

    for example, item in zip(examples, provenance["fixtures"], strict=True):
        source_row = upstream_rows[item["upstream_source_file"]][item["upstream_source_index_0"]]
        assert canonical_hash(source_row) == item["source_record_sha256"]
        assert source_row["question"] == example["expected_values"]["question"]

        raw_gym_line = gym_lines[item["gym_line_1"] - 1]
        assert (
            hashlib.sha256(raw_gym_line.encode("utf-8")).hexdigest()
            == item["raw_jsonl_line_sha256"]
        )
        gym_row = json.loads(raw_gym_line)
        assert canonical_hash(gym_row) == item["gym_record_sha256"]
        assert gym_row["input"] == example["input"]


def main(
    *,
    bbq_sft_root: Path | None = None,
    bbq_rlvr_root: Path | None = None,
) -> None:
    examples = load_jsonl(ROOT / "data/example.jsonl")
    assert len(examples) == 5, "data/example.jsonl must contain exactly five rows"

    parsed: list[ExpectedValues] = []
    for line_number, row in enumerate(examples, 1):
        assert row.get("task_name") == "bbq_two_judge", f"row {line_number}: wrong task_name"
        assert isinstance(row.get("input"), str) and row["input"].strip()
        assert row.get("output") == ""
        assert not (all_keys(row) & LEAKAGE_FIELDS), f"row {line_number}: SFT-target leakage"
        expected = ExpectedValues.model_validate(row.get("expected_values"))
        assert expected.question in row["input"], f"row {line_number}: visible question missing"
        parsed.append(expected)

    assert {item.category for item in parsed} == {
        "Age",
        "PhysicalAppearance",
        "DisabilityStatus",
    }
    assert {item.context_condition for item in parsed} == {"ambig", "disambig"}
    assert {item.disambig_direction for item in parsed if item.context_condition == "disambig"} == {
        "stereo_consistent",
        "counter_stereo",
    }
    assert any(len(item.biased_actor_labels) > 1 for item in parsed)
    assert any(item.actor_count >= 3 and item.wrong_neutral_actor_labels for item in parsed)

    # Provenance records (source indices + content hashes for each fixture)
    # are an internal audit artifact and optional in distribution copies;
    # when present they are fully verified.
    provenance_path = ROOT / "data/example_provenance.json"
    provenance = None
    if provenance_path.exists():
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        assert not contains_absolute_path(provenance), "provenance must not contain absolute paths"
        fixture_provenance = provenance.get("fixtures", [])
        assert len(fixture_provenance) == len(examples)
        for index, (row, item) in enumerate(zip(examples, fixture_provenance, strict=True), 1):
            assert item["example_line_1"] == index
            assert item["builder_source_id"] == row["expected_values"]["source_id"]
            assert item["fixture_id"] == row["expected_values"]["provenance_id"]
            assert item["protected_axis"] == row["expected_values"]["protected_axis"]
            assert item["input_sha256"] == hashlib.sha256(row["input"].encode("utf-8")).hexdigest()
            assert item["fixture_record_sha256"] == canonical_hash(row)

    if (bbq_sft_root is None) != (bbq_rlvr_root is None):
        raise AssertionError("provide both bbq_sft_root and bbq_rlvr_root, or neither")
    if bbq_sft_root is not None and bbq_rlvr_root is not None:
        assert provenance is not None, "upstream audit requires data/example_provenance.json"
        validate_upstream_sources(
            provenance,
            examples,
            bbq_sft_root=bbq_sft_root,
            bbq_rlvr_root=bbq_rlvr_root,
        )

    rollouts = load_jsonl(ROOT / "data/example_rollouts.jsonl")
    assert len(rollouts) == len(examples)
    expected_by_provenance = {item.provenance_id: item for item in parsed}
    for rollout in rollouts:
        expected = expected_by_provenance[rollout["provenance_id"]]
        assert rollout["source_id"] == expected.source_id
        answer = AnswerJudgment.model_validate(rollout["answer_judgment"])
        explanation = ExplanationJudgment.model_validate(rollout["explanation_judgment"])
        reward_1 = answer_reward(answer, expected)
        reward_2 = explanation_reward(explanation)
        assert rollout["reward_answer"] == reward_1
        assert rollout["reward_explanation_quality"] == reward_2
        assert rollout["reward"] == reward_1 * reward_2
        assert rollout["rollout_kind"] == "live_reference"

    metrics = json.loads((ROOT / "data/example_metrics.json").read_text(encoding="utf-8"))
    assert metrics["Number of examples"] == len(examples)
    mean_reward = sum(item["reward"] for item in rollouts) / len(rollouts)
    assert metrics["Reference mean reward"] == mean_reward

    checked = "5 fixtures, provenance records, reference rollouts, and metrics" \
        if provenance is not None else "5 fixtures, reference rollouts, and metrics"
    print(f"Validated {checked}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bbq-sft-root", type=Path)
    parser.add_argument("--bbq-rlvr-root", type=Path)
    args = parser.parse_args()
    main(bbq_sft_root=args.bbq_sft_root, bbq_rlvr_root=args.bbq_rlvr_root)
