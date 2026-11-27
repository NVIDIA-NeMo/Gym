# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from nemo_gym.rollouts.rollout_collection import (
    _collected_outcome,
    _CollectedOutcome,
    _CompletedRollout,
    _episode_record,
)


def _reply(**overrides: object) -> dict:
    return {
        "episode_id": {"rollout_id": "0-0", "attempt": 0},
        "task_id": {"taskset": "gdpval", "task_id": "workbook"},
        "result": {
            "reward": 0.0,
            "response": {"output": []},
            "mask_sample": True,
            "failure_kind": "gdpval:permanent_judge_error",
            "_ng_failure_class": "permanent_judge_error",
            "_ng_failure_terminal": True,
            "deliverables_dir": "/saved/workbook",
            "judge_response": {"error": "permanent_judge_error", "detail": "judge context exceeded"},
            **overrides,
        },
    }


def test_terminal_verifier_failure_retains_masking_and_artifacts() -> None:
    reply = _reply()
    record = _episode_record(reply)
    completed = _CompletedRollout(
        row={"_ng_task_index": 0, "_ng_rollout_index": 0, "_ng_run_id": "run"},
        result=record,
        rollout_latency_ms=None,
    )
    saved: _CollectedOutcome = _collected_outcome(completed, record)

    assert saved.structured_failure
    assert saved.record["_ng_failure_class"] == "gdpval:permanent_judge_error"
    assert saved.record["_ng_failure_legacy_class"] == "permanent_judge_error"
    assert saved.record["_ng_failure_record"]["failure"]["terminal"] is True
    assert saved.record["mask_sample"] is True
    assert saved.record["deliverables_dir"] == "/saved/workbook"
    assert saved.record["judge_response"] == reply["result"]["judge_response"]
    assert "reward" not in saved.record


def test_terminal_failure_of_any_verifier_namespace_is_accepted() -> None:
    record = _episode_record(_reply(failure_kind="other_verifier:permanent_judge_error"))

    assert record["_ng_failure_class"] == "permanent_judge_error"
    assert record["_ng_failure_terminal"] is True


@pytest.mark.parametrize(
    "overrides",
    [
        {"mask_sample": False},
        {"mask_sample": 1},
        {"_ng_failure_terminal": False},
        {"_ng_failure_terminal": 1},
        {"failure_kind": "permanent_judge_error"},
        {"failure_kind": ":permanent_judge_error"},
        {"failure_kind": "gdpval:"},
        {"failure_kind": "gdpval task:permanent_judge_error"},
        {"failure_kind": "gdpval:permanent_judge_error:detail", "_ng_failure_class": "permanent_judge_error:detail"},
        {"failure_kind": "GDPVal:permanent_judge_error"},
        {"_ng_failure_class": "reference_missing"},
        {"_ng_task_index": 99},
        {"_ng_failure_record": {}},
        {"_ng_failure_counts_as_zero": True, "_ng_task_index": 99},
    ],
)
def test_incomplete_failure_form_and_unrelated_collector_keys_are_rejected(overrides: dict) -> None:
    with pytest.raises(ValueError, match="reserved for rollout collection"):
        _episode_record(_reply(**overrides))
