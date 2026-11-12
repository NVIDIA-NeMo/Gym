# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The planner and resumed Store obey one saved policy, including unrouted failures."""

import json
import subprocess
import sys

import pytest

from nemo_gym.core.failure_kinds import AGENT_RUN_ERROR, KILL_SHAPED
from nemo_gym.core.outcome_policy import OutcomeAction
from nemo_gym.orchestration.recovery import (
    COLLECT_SCRIPT,
    CollectedEvidence,
    decide_from_collected,
    select_managed_evidence,
)
from nemo_gym.rollouts.recovery_policy import RecoveryOutcomePolicyConfig, episode_recovery_action
from nemo_gym.rollouts.rollout_recovery import RunManifest
from nemo_gym.rollouts.rollout_store import RolloutStore


@pytest.mark.parametrize("diagnostic", [False, True])
@pytest.mark.parametrize("action", ["retry", "fail"])
@pytest.mark.parametrize("delivery", ["delivered", "possibly_delivered"])
def test_saved_policy_controls_unknown_failure_in_planner_and_resume(tmp_path, diagnostic, action, delivery):
    job = tmp_path / "job" / "bench"
    output = job / "artifacts" / "rollouts.jsonl"
    output.parent.mkdir(parents=True)
    policy = RecoveryOutcomePolicyConfig(default_action=action)
    # Use the launcher's real saved-policy layout; no injected Store policy or pending mock.
    (job.parent / "resolved-config.yaml").write_text(
        json.dumps(
            {
                "driver": {
                    "benchmarks": {"bench": {"resumable": {"recovery_outcome_policy": policy.model_dump(mode="json")}}}
                }
            }
        )
    )
    inputs = [{"_ng_task_index": 0, "_ng_rollout_index": 0}]
    source = job / "input.jsonl"
    source.write_text(json.dumps(inputs[0]) + "\n")

    def prepare():
        return inputs, RunManifest.create(source, inputs, {}, {}).model_copy(update={"max_rollout_attempts": 2})

    with RolloutStore.start_or_resume(output, prepare, resume=False) as store:
        row = dict(inputs[0])
        store.allocate_attempt(row)
        failure = row | {
            "_ng_failure_class": AGENT_RUN_ERROR,
            "_ng_unrouted_run_error": True,
            "_ng_failure_http_status": 500,
            "_ng_failure_record": {
                "source": "collector",
                "delivery": delivery,
                "http_status": 500,
                "run_id": store.manifest.run_id,
                "episode_id": {"rollout_id": "0-0", "attempt": row["_ng_attempt_index"]},
                "failure": {
                    "terminal": False,
                    "failure_reason": "Unknown execution failure",
                    "failure_kind": AGENT_RUN_ERROR,
                },
            },
        }
        if diagnostic:
            store.record_unstarted(failure | {"_ng_omitted_rollout": True})
        else:
            store.record_outcome(failure)
    before = output.with_name("rollouts_failures.jsonl").read_bytes()
    raw = subprocess.run(
        [sys.executable, "-I", "-S", "-c", COLLECT_SCRIPT, str(job)],
        capture_output=True,
        text=True,
        check=True,
    )
    collected = select_managed_evidence(CollectedEvidence.model_validate_json(raw.stdout), job)
    assert collected.managed_reader_error is None
    decision = decide_from_collected(
        collected, max_infra_retries=3, max_walltime_s=None, live_job=False, outcome_policy=policy
    )
    assert decision.action.value == ("resubmit" if action == "retry" else "stop"), decision.reason
    assert bool(RolloutStore.read(output).pending()) is (action == "retry")
    with RolloutStore.start_or_resume(output, prepare, resume=True) as resumed:
        pending = resumed.pending()
        assert bool(pending) is (action == "retry")
        if pending:
            resumed.record_dispatch(pending[0])
            resumed.record_outcome(pending[0] | {"reward": 1.0})
    assert output.with_name("rollouts_failures.jsonl").read_bytes() == before
    if action == "retry":
        assert RolloutStore.read(output).coverage()["successful"] == 1
        assert RolloutStore.read(output).pending() == []


def test_interruption_action_is_a_policy_rule_not_a_store_veto():
    row = {"_ng_failure_class": KILL_SHAPED}
    assert episode_recovery_action(row, policy=RecoveryOutcomePolicyConfig()) == OutcomeAction.FAIL
    assert (
        episode_recovery_action(row, policy=RecoveryOutcomePolicyConfig(by_failure_kind={KILL_SHAPED: "retry"}))
        == OutcomeAction.RETRY
    )
