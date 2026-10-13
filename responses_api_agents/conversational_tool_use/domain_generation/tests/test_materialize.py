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

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from responses_api_agents.conversational_tool_use.domain_generation.materialize import (
    POLICY_TOOL_AGENT_REF,
    main,
    materialize_policy_tool_rows,
    read_jsonl,
)
from responses_api_agents.conversational_tool_use.policy_tool_generation.models import (
    PolicyToolGenerationRunRequest,
)


def rollout(*candidates: dict) -> dict:
    return {"result": {"candidates": list(candidates)}}


def test_materializer_preserves_objects_and_uses_casefold_only_first_wins(tmp_path: Path) -> None:
    source = tmp_path / "domains.jsonl"
    original = [
        {"name": "Home Services", "applications": [{"function": "Schedule a visit"}], "extra": {"keep": 1}},
        {"name": "HOME SERVICES", "applications": [{"function": "Duplicate"}]},
        {"name": " Home Services ", "applications": [{"function": "Whitespace is significant"}]},
        {"name": "A-B", "applications": []},
        {"name": "a b", "applications": []},
    ]
    rows = materialize_policy_tool_rows(
        [(1, rollout(*original[:2])), (2, rollout(*original[2:]))],
        source=source,
        profile="general",
    )

    assert [row["domain"] for row in rows] == [original[0], original[2], original[3], original[4]]
    assert all(row["responses_create_params"] == {"input": []} for row in rows)
    assert all(row["profile"] == "general" for row in rows)
    assert all(row["agent_ref"] == POLICY_TOOL_AGENT_REF for row in rows)
    assert [row["source_artifacts"]["domain_generation"]["candidate_index"] for row in rows] == [0, 0, 1, 2]
    assert all(PolicyToolGenerationRunRequest.model_validate(row) for row in rows)


def test_materializer_preserves_parent_lineage_without_paths(tmp_path: Path) -> None:
    source = tmp_path / "domains.jsonl"
    domain_rollout = rollout({"name": "Home Services"})
    domain_rollout.update(
        {
            "id": "domain-run",
            "_ng_task_index": 7,
            "_ng_rollout_index": 3,
            "_ng_attempt_index": 2,
            "source_artifacts": {"seed": {"profile": "general"}},
        }
    )

    [row] = materialize_policy_tool_rows(
        [(1, domain_rollout)],
        source=source,
        profile="general",
    )

    assert row["id"] == "domain-run_ng_t7_r3_a2_candidate_000000"
    assert row["source_artifacts"] == {
        "seed": {"profile": "general"},
        "domain_generation": {
            "id": "domain-run",
            "_ng_task_index": 7,
            "_ng_rollout_index": 3,
            "_ng_attempt_index": 2,
            "candidate_index": 0,
        },
    }
    assert str(source) not in json.dumps(row)


def test_materializer_shuffle_is_explicit_and_seeded(tmp_path: Path) -> None:
    source = tmp_path / "domains.jsonl"
    candidates = [{"name": f"Domain {index}", "value": index} for index in range(8)]
    expected = candidates.copy()
    random.Random(17).shuffle(expected)

    rows = materialize_policy_tool_rows(
        [(1, rollout(*candidates))],
        source=source,
        profile="proactive",
        shuffle_seed=17,
    )

    assert [row["domain"] for row in rows] == expected


def test_cli_requires_named_paths_and_writes_jsonl(tmp_path: Path) -> None:
    input_path = tmp_path / "domain-rollouts.jsonl"
    output_path = tmp_path / "policy-inputs.jsonl"
    candidates = [
        {"name": "Home Services", "nested": {"preserved": [1, 2]}},
        {"name": "home services", "nested": {"dropped": True}},
        {"name": "Event Support", "nested": {"preserved": [3]}},
    ]
    input_path.write_text(json.dumps(rollout(*candidates)) + "\n", encoding="utf-8")

    assert (
        main(
            [
                "--input-file",
                str(input_path),
                "--output-file",
                str(output_path),
                "--profile",
                "general",
            ]
        )
        == 0
    )

    written = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert [row["domain"] for row in written] == [candidates[0], candidates[2]]
    assert [row["profile"] for row in written] == ["general", "general"]

    with pytest.raises(SystemExit):
        main(
            [
                "--input-file",
                str(input_path),
                "--output-file",
                str(output_path),
            ]
        )
    with pytest.raises(SystemExit):
        main([])


def test_reader_rejects_rollouts_without_typed_result(tmp_path: Path) -> None:
    input_path = tmp_path / "bad.jsonl"
    input_path.write_text('{"reward": 1.0}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="result.candidates"):
        materialize_policy_tool_rows(read_jsonl(input_path), source=input_path, profile="general")


def test_materializer_rejects_duplicate_output_ids(tmp_path: Path) -> None:
    source = tmp_path / "domains.jsonl"
    duplicated_identity = [
        (1, {"id": "same", "result": {"candidates": [{"name": "First"}]}}),
        (2, {"id": "same", "result": {"candidates": [{"name": "Second"}]}}),
    ]

    with pytest.raises(ValueError, match="duplicate materialized row id"):
        materialize_policy_tool_rows(
            duplicated_identity,
            source=source,
            profile="general",
        )


def test_cli_rejects_same_input_and_output_path(tmp_path: Path) -> None:
    path = tmp_path / "domains.jsonl"
    path.write_text(json.dumps(rollout({"name": "Home Services"})) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="must differ"):
        main(
            [
                "--input-file",
                str(path),
                "--output-file",
                str(path),
                "--profile",
                "general",
            ]
        )
