# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import copy

import pytest

from nemo_gym.rollout_collection import _build_trajectory_record


@pytest.mark.parametrize(
    "field,facts",
    [
        (
            "grading_operations",
            [{"operation_id": "grade", "kind": "llm_judge", "outcome": "unknown", "dependencies": []}],
        ),
        (
            "sandbox_observations",
            [
                {
                    "role": "verifier",
                    "sandbox_id": "sandbox",
                    "outcome": "unknown",
                    "teardown_confirmed": True,
                    "cleanup_errors": ["earlier-close-failure"],
                }
            ],
        ),
    ],
)
def test_returned_facts_survive_a_capture_created_verifier(field, facts):
    returned = {"invocation_id": "verify", "status": "completed", field: facts}
    capture = {"invocation_id": "verify", "model_calls": [{"model_call_id": "judge-call"}]}
    result = {"ng_model_call_capture": {"verifier_invocations": [capture]}, "ng_verifier_invocations": [returned]}
    actual = _build_trajectory_record({"_ng_task_index": 0, "_ng_rollout_index": 0}, result)
    verifier = actual.verifier_invocations[0]
    observed = [item.model_dump(mode="json", exclude_unset=True) for item in getattr(verifier, field)]
    assert observed == facts
    assert verifier.model_calls[0].model_call_id == "judge-call"
    assert verifier.status == "completed"
    # Two incompatible returned observations cannot establish one authoritative grading/lifecycle.
    conflict = copy.deepcopy(returned)
    conflict[field] = []
    result["ng_verifier_invocations"].append(conflict)
    actual = _build_trajectory_record({"_ng_task_index": 0, "_ng_rollout_index": 0}, result)
    assert getattr(actual.verifier_invocations[0], field) == []
    assert any(gap.code == f"verifier_{field}_conflict" for gap in actual.gaps)
