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
"""A result masked by infrastructure (a lost session, a failed provider or verifier) is retried, not kept as complete.

Real Store, real stdlib collector and the real recovery decision, read from saved files as an existing run is.
"""

import json

import pytest

from nemo_gym.agents.agent_outcome import OutcomeAction
from nemo_gym.orchestration.recovery import RecoveryOutcomePolicyConfig
from nemo_gym.rollouts.rollout_recovery import RunManifest
from nemo_gym.rollouts.rollout_store import RolloutStore
from tests.unit_tests.orchestration.test_allocation_cut_recovery import _decide
from tests.unit_tests.orchestration.test_eval_resubmit import slurm_bin as slurm_bin


def _masked(kind):
    return {"reward": 0.0, "mask_sample": True, "failure_kind": kind, "failure_reason": "sandbox gone"}


def _run(job, outcomes, *, cap=3):
    """One scored rollout, then one rollout per entry of ``outcomes``: the saved result of each of its attempts."""
    artifacts = job / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    rows = [{"_ng_task_index": i, "_ng_rollout_index": 0} for i in range(1 + len(outcomes))]
    source = job / "source.jsonl"
    source.write_text("".join(json.dumps(row) + "\n" for row in rows))

    def prepare():
        inputs = json.loads(json.dumps(rows))
        return inputs, RunManifest.create(source, inputs, {}, {}).model_copy(update={"max_rollout_attempts": cap})

    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("SLURM_JOB_ID", "100")
        with RolloutStore.start_or_resume(artifacts / "rollouts.jsonl", prepare, resume=False) as store:
            scored = dict(rows[0])
            store.record_dispatch(scored)
            store.record_outcome(scored | {"reward": 1.0})
            for row, attempts in zip(rows[1:], outcomes):
                for fields in attempts:
                    dispatched = dict(row)
                    store.allocate_attempt(dispatched)
                    store.record_outcome(dispatched | fields)
    (job / ".gym_job_chain").write_text("100\n")
    (job / ".gym_accumulated_walltime").write_text("14400\n")
    return artifacts / "rollouts.jsonl", prepare


def _evidence(decision, identity):
    return next(item for item in decision.evidence if item.identity == identity)


def test_a_masked_session_lost_result_is_retryable_against_the_attempt_cap(tmp_path):
    _run(tmp_path, [[_masked("session_lost")]])
    _, decision = _decide(tmp_path, allocation_cut=False)
    assert decision.action.value == "resubmit", decision.reason
    assert decision.counts == {"scored": 1, "retryable": 1}
    item = _evidence(decision, (1, 0))
    assert (item.failure_kind, item.attempts, item.max_attempts) == ("session_lost", 1, 3)
    assert not decision.complete


def test_a_masked_infrastructure_retry_spends_the_infra_retry_budget(tmp_path):
    _run(tmp_path, [[_masked("session_lost")]])
    (tmp_path / ".gym_infra_retries").write_text("2\n")
    _, decision = _decide(tmp_path, allocation_cut=False)
    assert decision.action.value == "stop"
    assert decision.reason == "the stored infra-retry budget is used (2 of 2)"


@pytest.mark.parametrize(
    "kind", ["session_lost", "provider_unavailable", "infrastructure_error", "verifier_error", "session_expired"]
)
def test_each_infrastructure_kind_is_retried(tmp_path, kind):
    _run(tmp_path, [[_masked(kind)]])
    _, decision = _decide(tmp_path, allocation_cut=False)
    assert decision.counts == {"scored": 1, "retryable": 1}, decision.reason


def test_the_resume_runs_a_masked_infrastructure_result_again_and_the_new_attempt_supersedes_it(tmp_path):
    output, prepare = _run(tmp_path, [[_masked("session_lost")]])
    with RolloutStore.start_or_resume(output, prepare, resume=True) as store:
        assert store.coverage()["complete"] is False
        pending = store.pending()
        assert [(row["_ng_task_index"], row["_ng_attempt_index"]) for row in pending] == [(1, 1)]
        store.record_dispatch(pending[0])
        store.record_outcome(pending[0] | {"reward": 1.0})
    store = RolloutStore.read(output, import_legacy=False)
    selected = sorted(store.selected("success"), key=lambda row: row["_ng_task_index"])
    assert [(row["_ng_task_index"], row["_ng_attempt_index"], row.get("mask_sample")) for row in selected] == [
        (0, 0, None),
        (1, 1, None),
    ]
    coverage = store.coverage()
    assert (coverage["complete"], coverage["masked"], coverage["masked_retryable"]) == (True, 0, 0)
    _, decision = _decide(tmp_path, allocation_cut=False)
    assert (decision.action.value, decision.counts) == ("noop", {"scored": 2})


def test_a_masked_infrastructure_result_at_the_attempt_cap_is_exhausted_and_not_run(tmp_path):
    output, _ = _run(tmp_path, [[_masked("session_lost")] * 3])
    store = RolloutStore.read(output, import_legacy=False)
    assert store.pending() == []
    coverage = store.coverage()
    assert (coverage["failure_budget_exhausted"], coverage["retryable"], coverage["complete"]) == (1, 0, True)
    _, decision = _decide(tmp_path, allocation_cut=False)
    assert decision.action.value == "stop"
    assert decision.counts == {"scored": 1, "exhausted": 1}
    item = _evidence(decision, (1, 0))
    assert (item.failure_kind, item.attempts, item.max_attempts) == ("session_lost", 3, 3)
    assert "used every attempt since the last resubmit" in decision.reason


@pytest.mark.parametrize("kind", [None, "agent_timeout", "model_context_limit", "env:own_kind", "not_registered"])
def test_a_masked_result_with_a_model_kind_or_no_infrastructure_kind_is_kept(tmp_path, kind):
    fields = {"reward": 0.0, "mask_sample": True} | ({"failure_kind": kind} if kind else {})
    output, _ = _run(tmp_path, [[fields]])
    store = RolloutStore.read(output, import_legacy=False)
    assert store.pending() == []
    assert store.coverage()["complete"] is True
    _, decision = _decide(tmp_path, allocation_cut=False)
    assert decision.action.value == "stop"
    assert decision.counts == {"scored": 1, "masked": 1}


def test_a_masked_judge_failure_stays_with_re_verification(tmp_path):
    answer = {
        "response": {
            "object": "response",
            "id": "r1",
            "status": "completed",
            "output": [{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "4"}]}],
        }
    }
    output, _ = _run(tmp_path, [[_masked("judge_failed") | answer]])
    store = RolloutStore.read(output, import_legacy=False)
    assert store.pending() == []
    _, decision = _decide(tmp_path, allocation_cut=False)
    assert decision.action.value == "reverify"
    assert decision.counts == {"scored": 1, "judge_recoverable": 1}


def test_a_policy_that_fails_session_lost_keeps_it_final(tmp_path):
    from nemo_gym.orchestration.recovery import decide_from_collected

    _run(tmp_path, [[_masked("session_lost")]])
    collected, _ = _decide(tmp_path, allocation_cut=False)
    policy = RecoveryOutcomePolicyConfig(by_failure_kind={"session_lost": OutcomeAction.FAIL})
    decision = decide_from_collected(
        collected, max_infra_retries=2, max_walltime_s=None, live_job=False, outcome_policy=policy
    )
    assert decision.action.value == "stop"
    assert decision.counts == {"scored": 1, "terminal": 1}


def test_inspect_of_a_saved_run_with_one_masked_session_lost_result_resubmits(tmp_path, slurm_bin):
    from nemo_gym.orchestration.submit import inspect_recovery
    from tests.unit_tests.orchestration.test_eval_resubmit import _handle, _resumable_run

    job, _, _ = _resumable_run(tmp_path)
    for name in ("rollouts.jsonl", "rollouts_materialized_inputs.jsonl"):
        (job / "artifacts" / name).unlink()
    _run(job, [[_masked("session_lost")]])
    # The chain's walltime is read from the scheduler's accounting, as on the operator's login host.
    (job / ".gym_accumulated_walltime").unlink()
    (job / ".gym_job_chain").write_text("100\n101\n")
    (slurm_bin / "accounting").write_text("100|TIMEOUT|7200|0\n101|FAILED|60|0\n")
    outcome = inspect_recovery(*_handle(job))
    assert outcome.dispatched is None
    assert outcome.decision.action.value == "resubmit", outcome.decision.reason
    assert outcome.decision.counts == {"scored": 1, "retryable": 1}


def test_unmanaged_masked_results_are_still_kept(tmp_path):
    # Without a Store selection the legacy resume reuses every result row, so recovery keeps the masked row.
    from nemo_gym.orchestration.recovery import decide_recovery

    decision = decide_recovery(
        materialized=[{"_ng_task_index": 0, "_ng_rollout_index": 0}],
        results=[{"_ng_task_index": 0, "_ng_rollout_index": 0} | _masked("session_lost")],
        failures=[],
        budget=_budget(),
    )
    assert decision.counts == {"masked": 1}


def _budget():
    from nemo_gym.orchestration.recovery import RecoveryBudget

    return RecoveryBudget(live_job=False, max_infra_retries=2, infra_retries_used=0)


_ANSWER = {
    "response": {
        "object": "response",
        "id": "r1",
        "status": "completed",
        "output": [{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "4"}]}],
    }
}


@pytest.mark.parametrize("kind", ["session_lost", "verifier_error", "infrastructure_error"])
def test_a_masked_infrastructure_result_that_kept_a_saved_answer_is_not_retried(tmp_path, kind):
    # A saved model answer is never sampled again: the row keeps the masked-and-kept behaviour.
    output, _ = _run(tmp_path, [[_masked(kind) | _ANSWER]])
    store = RolloutStore.read(output, import_legacy=False)
    assert store.pending() == []
    coverage = store.coverage()
    assert (coverage["complete"], coverage["masked_retryable"], coverage["retryable"]) == (True, 0, 0)
    assert store.attempt_count({"_ng_task_index": 1, "_ng_rollout_index": 0}) == 0
    _, decision = _decide(tmp_path, allocation_cut=False)
    assert decision.action.value == "stop"
    assert decision.counts == {"scored": 1, "masked": 1}


def test_a_masked_infrastructure_result_with_only_a_no_generation_placeholder_is_still_retried(tmp_path):
    from nemo_gym.artifacts.rollout_rows import NG_NO_GENERATION_METADATA

    placeholder = {
        "response": {
            "object": "response",
            "id": "r1",
            "output": [],
            "metadata": dict(NG_NO_GENERATION_METADATA),
        }
    }
    _run(tmp_path, [[_masked("session_lost") | placeholder]])
    _, decision = _decide(tmp_path, allocation_cut=False)
    assert decision.counts == {"scored": 1, "retryable": 1}, decision.reason


# A masked row whose saved transcript holds no gradeable artifact has no answer for the contract: it is an omission,
# and running it again is the ordinary infrastructure retry. "Never regenerate" applies to a gradeable answer only.


def _agent_sandbox_verified():
    """The verifier graded the agent's own sandbox (a terminal task's container), which is gone after the rollout."""
    return {
        "ng_trajectory": {
            "verifier_invocations": [
                {
                    "invocation_id": "v1",
                    "status": "completed",
                    "sandbox_observations": [
                        {"role": "agent", "provider": "p"},
                        {"role": "verifier", "provider": "p"},
                    ],
                }
            ]
        }
    }


def _patch(intact):
    import hashlib

    patch = "diff --git a/x b/x\n"
    return {"model_patch": patch, "model_patch_sha256": hashlib.sha256(patch.encode()).hexdigest() if intact else "0"}


@pytest.mark.parametrize(
    "kind,fields",
    [
        ("verifier_error", {}),
        ("session_lost", {}),
        ("verifier_error", _patch(intact=True)),
        ("judge_failed", _agent_sandbox_verified()),
        # A saved answer is never regenerated, also when its verifier had no intact workspace to grade.
        ("session_lost", {"model_patch": None}),
        ("verifier_error", {"model_patch": None}),
        ("verifier_error", _patch(intact=False)),
        ("verifier_error", _agent_sandbox_verified()),
        ("verifier_error", {"session_id": "s-1", "sandbox_resource_limit": None}),
    ],
    ids=[
        "text-answer",
        "text-answer-session-lost",
        "intact-patch",
        "judge-failure",
        "patch-never-collected",
        "verifier-patch-never-collected",
        "patch-not-intact",
        "agent-sandbox-graded",
        "sandboxed-session",
    ],
)
def test_a_masked_result_with_a_saved_answer_is_not_sampled_again(tmp_path, kind, fields):
    output, _ = _run(tmp_path, [[_masked(kind) | _ANSWER | fields]])
    store = RolloutStore.read(output, import_legacy=False)
    assert store.pending() == []


def test_the_store_keeps_a_sandboxed_session_row_it_reads_without_its_trajectory(tmp_path):
    # The recovery reader reads the Store with the trajectory skipped: a saved answer stays the result there too.
    output, _ = _run(
        tmp_path, [[_masked("verifier_error") | _ANSWER | {"session_id": "s-1", "sandbox_resource_limit": None}]]
    )
    store = RolloutStore.read(
        output, import_legacy=False, skip_keys=frozenset({"ng_trajectory", "ng_agent_observations"})
    )
    assert store.pending() == []


@pytest.mark.parametrize(
    "stage,expected", [("seed", "retryable"), ("verification", "retryable"), ("agent", "masked"), (None, "masked")]
)
def test_masked_oom_recovery_uses_shared_origin_stage(tmp_path, stage, expected):
    row = _masked("provider_oom_killed") | {
        "_ng_failure_id": "oom",
        "_ng_failure_events": {
            "oom": {
                "event_id": "oom",
                "boundary": "sandbox",
                "failure_kind": "provider_oom_killed",
                "stage": stage,
            }
        },
    }
    output, _ = _run(tmp_path, [[row]])
    store = RolloutStore.read(output, import_legacy=False)
    assert len(store.pending()) == (1 if expected == "retryable" else 0)
    _, decision = _decide(tmp_path, allocation_cut=False)
    assert decision.counts == {"scored": 1, expected: 1}
    if expected == "retryable":
        assert _evidence(decision, (1, 0)).attempts == 1


@pytest.mark.parametrize("attribution", ["evaluation", "model", "unknown"])
def test_masked_oom_recovery_honors_the_recorded_retry_action_after_reload(tmp_path, attribution):
    expected = "retryable"
    row = _masked("provider_oom_killed") | {
        "failure_outcome_action": "retry",
        "failure_outcome_attribution": attribution,
    }
    _run(tmp_path, [[row]])
    _, decision = _decide(tmp_path, allocation_cut=False)
    assert decision.counts == {"scored": 1, expected: 1}


def test_masked_seed_oom_saved_fail_never_becomes_a_retry(tmp_path):
    row = _masked("provider_oom_killed") | {
        "failure_outcome_action": "fail",
        "failure_outcome_attribution": "evaluation",
        "_ng_failure_stage": "seed",
    }
    output, _ = _run(tmp_path, [[row]])
    assert RolloutStore.read(output, import_legacy=False).pending() == []
    _, decision = _decide(tmp_path, allocation_cut=False)
    assert decision.counts == {"scored": 1, "terminal": 1}


def test_saved_launcher_classification_drives_store_resume_and_planner(tmp_path):
    from nemo_gym.core.failure_kinds import FailureClassificationConfig

    job = tmp_path / "benchmark"
    output, prepare = _run(job, [[_masked("provider_oom_killed") | {"_ng_failure_stage": "agent"}]])
    assert RolloutStore.read(output).pending() == []
    classification = FailureClassificationConfig(
        by_stage={"agent": {"by_failure_kind": {"provider_oom_killed": "evaluation"}}}
    )
    (tmp_path / "resolved-config.yaml").write_text(
        json.dumps(
            {
                "driver": {
                    "benchmarks": {
                        "benchmark": {
                            "resumable": {
                                "recovery_outcome_policy": {"classification": classification.model_dump(mode="json")}
                            }
                        }
                    }
                }
            }
        )
    )
    assert len(RolloutStore.read(output).pending()) == 1
    from nemo_gym.orchestration.recovery import decide_from_collected

    collected, _ = _decide(job, allocation_cut=False)
    decision = decide_from_collected(
        collected,
        max_infra_retries=2,
        max_walltime_s=None,
        live_job=False,
        outcome_policy=RecoveryOutcomePolicyConfig(classification=classification),
    )
    assert decision.counts == {"scored": 1, "retryable": 1}
    with RolloutStore.start_or_resume(output, prepare, resume=True) as store:
        pending = store.pending()
        assert [(row["_ng_task_index"], row["_ng_attempt_index"]) for row in pending] == [(1, 1)]
        store.record_dispatch(pending[0])
        store.record_outcome(pending[0] | {"reward": 1.0})
    assert RolloutStore.read(output).pending() == []


@pytest.mark.parametrize("classification", [{"by_failure_kind": {"provider_oom_killed": "typo"}}, [], False])
def test_store_refuses_malformed_saved_classification(tmp_path, classification):
    from nemo_gym.config.config_types import ConfigError

    output, _ = _run(tmp_path / "benchmark", [[_masked("session_lost")]])
    (tmp_path / "resolved-config.yaml").write_text(
        json.dumps(
            {
                "driver": {
                    "benchmarks": {
                        "benchmark": {"resumable": {"recovery_outcome_policy": {"classification": classification}}}
                    }
                }
            }
        )
    )
    with pytest.raises(ConfigError, match="saved failure classification"):
        RolloutStore.read(output)


@pytest.mark.parametrize("section", [{"resumable": []}, {"resumable": {"recovery_outcome_policy": False}}])
def test_store_refuses_malformed_saved_policy_sections(tmp_path, section):
    from nemo_gym.config.config_types import ConfigError

    output, _ = _run(tmp_path / "benchmark", [[_masked("session_lost")]])
    (tmp_path / "resolved-config.yaml").write_text(json.dumps({"driver": {"benchmarks": {"benchmark": section}}}))
    with pytest.raises(ConfigError, match="must be a mapping"):
        RolloutStore.read(output)
