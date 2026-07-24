# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from responses_api_agents.conversational_tool_use_domain_generation.materialize import (
    POLICY_TOOL_AGENT_REF,
    main,
    materialize_policy_tool_rows,
    read_jsonl,
)
from responses_api_agents.conversational_tool_use_policy_tool_generation.models import (
    PolicyToolGenerationRunRequest,
)


def rollout(*candidates: dict) -> dict:
    return {"result": {"candidates": list(candidates)}}


def test_materializer_preserves_objects_and_uses_casefold_only_first_wins(tmp_path: Path) -> None:
    source = tmp_path / "domains.jsonl"
    original = [
        {"name": "Retail", "applications": [{"function": "Order status"}], "extra": {"keep": 1}},
        {"name": "RETAIL", "applications": [{"function": "Duplicate"}]},
        {"name": " Retail ", "applications": [{"function": "Whitespace is significant"}]},
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
    assert all(PolicyToolGenerationRunRequest.model_validate(row) for row in rows)


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
        {"name": "Retail", "nested": {"preserved": [1, 2]}},
        {"name": "retail", "nested": {"dropped": True}},
        {"name": "Travel", "nested": {"preserved": [3]}},
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
