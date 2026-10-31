# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Producer and stdlib collector bind exactly the same saved grading result."""

import ast
import hashlib
import json
import math

import pytest

from nemo_gym.core.episode import rollout_context
from nemo_gym.orchestration.recovery import COLLECT_SCRIPT
from nemo_gym.server_runtime.verifier_observability import observe_programmatic_grading, observe_verifier
from nemo_gym.trajectory.trajectory_record import grading_result_binding


def collector_binding(value):
    node = next(
        n
        for n in ast.parse(COLLECT_SCRIPT).body
        if isinstance(n, ast.FunctionDef) and n.name == "grading_result_binding"
    )
    namespace = {"hashlib": hashlib, "json": json, "math": math}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "<actual collector binding>", "exec"), namespace)
    return namespace["grading_result_binding"](value)


def result(reward=1):
    return {
        "response": {"id": "synthetic-response", "output": [{"type": "message", "content": "saved input"}]},
        "reward": reward,
    }


@pytest.mark.parametrize("reward", [0, 0.0, 1, 1.0, 0.5, -0.0])
def test_producer_collector_numeric_canonicalization(reward):
    row = result(reward)
    row["reward_components"] = {"part": reward}
    assert grading_result_binding(row) == collector_binding(row)
    if reward in (0, 1):
        equivalent = result(float(reward))
        equivalent["reward_components"] = {"part": float(reward)}
        assert grading_result_binding(row) == grading_result_binding(equivalent)


@pytest.mark.parametrize("value", [None, True, "1", float("nan"), float("inf"), float("-inf")])
def test_unusable_numeric_reward_is_unknown(value):
    assert grading_result_binding(result(value)) is None
    assert collector_binding(result(value)) is None


@pytest.mark.parametrize(
    "change",
    [
        "missing_reward",
        "missing_response",
        "missing_output",
        "bool_component",
        "nonfinite_component",
        "nonfinite_response",
    ],
)
def test_missing_or_nonfinite_evidence_never_binds(change):
    row = result()
    if change == "missing_reward":
        row.pop("reward")
    elif change == "missing_response":
        row.pop("response")
    elif change == "missing_output":
        row["response"].pop("output")
    elif change == "bool_component":
        row["reward_components"] = {"part": True}
    elif change == "nonfinite_component":
        row["reward_components"] = {"part": float("nan")}
    else:
        row["response"]["extra"] = float("inf")
    assert grading_result_binding(row) is None and collector_binding(row) is None


def test_unencodable_saved_text_leaves_grade_unbound():
    row = result()
    row["response"]["output"][0]["content"] = chr(0xD83D)
    assert grading_result_binding(row) is None and collector_binding(row) is None


@pytest.mark.asyncio
async def test_canonical_export_retains_grading_facts_without_raw_payload():
    from nemo_gym.artifacts.rollout_rows import _rollout_for_export
    from nemo_gym.rollout_collection import _build_trajectory_record

    async def verify():
        return observe_programmatic_grading(lambda: result(1))

    with rollout_context("0-0"):
        actual = await observe_verifier(verify, enabled=True)()
    trajectory = _build_trajectory_record({"_ng_task_index": 0, "_ng_rollout_index": 0}, actual)
    exported = _rollout_for_export({"ng_trajectory": trajectory.model_dump(mode="json")})
    operations = exported["ng_trajectory"]["verifier_invocations"][0]["grading_operations"]
    assert operations[0]["result_binding_sha256"] == grading_result_binding(actual)
    assert "saved input" not in json.dumps(exported)
