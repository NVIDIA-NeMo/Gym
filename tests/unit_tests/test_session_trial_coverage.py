# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Strict consumer boundary controls; fixture metadata is not a new runtime acceptance claim."""

import json
from pathlib import Path

import pytest

from nemo_gym.orchestration.recovery import (
    CollectedEvidence,
    ManagedRecoveryFact,
    decide_from_collected,
    trial_health,
)


def session_fixture(case="healthy"):
    value = json.loads((Path(__file__).parent / f"fixtures/cpu-grading-mcqa-{case}.json").read_text())
    c = CollectedEvidence.model_validate(value)
    row = c.results[0]
    environment = c.quality["provenance"]["servers"][row["_ng_environment_server"]]
    environment["implementation"] = "single_agent_turn_legacy"
    row["_ng_result_type"] = "single_agent_turn_legacy"
    row["materialized_task_id"] = "dataset-task"
    t = row["ng_trial_observations"]
    t["task_id"] = "dataset-task"
    t["turns"][0]["task_id"] = "dataset-task"
    t["invocations"][0].update(in_session=True, verification_skipped=None, invocation_id="root")
    t["turns"][0]["invocation_id"] = "root"
    c.checkpoint_enabled = False
    c.managed_selection = [
        ManagedRecoveryFact(identity=(0, 0), attempts=0, max_attempts=3, pending=False, reserved=False)
    ]
    return c


def health(c):
    d = decide_from_collected(c, max_infra_retries=3, max_walltime_s=None, live_job=False)
    return trial_health(d, c)[0]


@pytest.mark.parametrize("case,reward", [("healthy", 1), ("wrong", 0)])
def test_session_consumer_boundary_keeps_ordinary_wrong_zero(case, reward):
    c = session_fixture(case)
    assert c.results[0]["reward"] == reward
    assert health(c) is True


@pytest.mark.parametrize(
    "mutation",
    [
        "checkpoint",
        "unknown_checkpoint",
        "missing_id",
        "wrong_id",
        "foreign_invocation",
        "multiple_invocations",
        "missing_verifier",
        "failed_verifier",
        "tool",
        "wrong_turn",
        "wrong_response",
        "recovered",
        "no_reader",
        "stale",
        "wrong_session",
        "skip",
    ],
)
def test_session_trial_rejects_incomplete_or_contradictory_metadata(mutation):
    c = session_fixture()
    assert health(c) is True
    row = c.results[0]
    t = row["ng_trial_observations"]
    if mutation == "checkpoint":
        c.checkpoint_enabled = True
    elif mutation == "unknown_checkpoint":
        c.checkpoint_enabled = None
    elif mutation == "missing_id":
        row.pop("materialized_task_id")
    elif mutation == "wrong_id":
        row["materialized_task_id"] = "another-task"
    elif mutation == "foreign_invocation":
        t["invocations"][0]["invocation_id"] = "activation-2"
    elif mutation == "multiple_invocations":
        t["invocations"].append(dict(t["invocations"][0]))
    elif mutation == "missing_verifier":
        t["verifier_invocations"] = []
    elif mutation == "failed_verifier":
        t["verifier_invocations"][0]["status"] = "failed"
    elif mutation == "tool":
        t["tool_call_count"] = 1
    elif mutation == "wrong_turn":
        t["turns"][0]["rollout_id"] = "foreign"
    elif mutation == "wrong_response":
        t["invocations"][0]["model_calls"][0]["response_id"] = "foreign"
    elif mutation == "recovered":
        row["_ng_failure_events"] = {"unselected": {"boundary": "model"}}
    elif mutation == "no_reader":
        c.managed_selection = None
    elif mutation == "stale":
        c.quality["source_matches"] = False
    elif mutation == "wrong_session":
        t["invocations"][0]["in_session"] = False
    elif mutation == "skip":
        t["invocations"][0]["verification_skipped"] = True
    assert health(c) is None


@pytest.mark.parametrize(
    "config,expected",
    [
        (None, None),
        ("[]", None),
        ("checkpoint: [bad]", None),
        ("{}", False),
        ("checkpoint: {enabled: false}", False),
        ("checkpoint: {enabled: true, control_auth_token: fixture}", True),
    ],
)
def test_reader_checkpoint_fact_comes_from_bound_saved_config(tmp_path, config, expected):
    from nemo_gym.orchestration.recovery import select_managed_evidence
    from tests.unit_tests.test_managed_recovery_selection import collect, prepare_managed

    prepare_managed(tmp_path, "healthy")
    path = tmp_path / "artifacts/rollouts_resolved_config.yaml"
    if config is not None:
        path.write_text(config)
    selected = select_managed_evidence(collect(tmp_path), tmp_path)
    assert selected.managed_reader_error is None
    assert selected.checkpoint_enabled is expected
