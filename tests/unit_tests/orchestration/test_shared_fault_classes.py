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
"""The fault classes are one shared default in the failure vocabulary; a policy timeout's class is a run setting."""

import pytest

from nemo_gym.core import failure_kinds
from nemo_gym.orchestration import recovery
from nemo_gym.orchestration.recovery import (
    GateCriteria,
    fault_report,
    gate_summary_from_collected,
    trial_verdict,
)
from tests.unit_tests.orchestration.test_eval_resubmit_decision import (  # noqa: F401
    complete_job,
    fake_executor,
    run_cli,
)
from tests.unit_tests.orchestration.test_evaluation_health import _gate, _run


def test_the_gate_uses_the_shared_classifier():
    from collections import Counter

    from nemo_gym.core.failure_kinds import classify_failure

    for kind in failure_kinds.FAILURE_KINDS | failure_kinds.GATE_DERIVED_KINDS:
        category = classify_failure(failure_kind=kind)
        assert fault_report(Counter({kind: 1}), 1)[category]["kinds"] == {kind: 1}
    assert fault_report(Counter({"provider_not_registered": 1}), 1)["unknown"]["count"] == 1


def test_a_command_the_harness_refused_is_the_models_fault():
    from collections import Counter

    report = fault_report(Counter({"agent_command_not_allowed": 1}), 100)
    assert report["model"]["kinds"] == {"agent_command_not_allowed": 1}
    assert report["evaluation"]["count"] == 0


@pytest.mark.parametrize(
    "kind",
    [
        "truncated_answer",
        "empty_answer",
        "malformed_answer",
        "model_server_error",
        "non_policy_model_rejection",
        "invalid_output_budget",
        "invalid_request_parameter",
    ],
)
def test_the_gate_derived_classes_are_registered_but_no_producer_may_emit_them(kind):
    assert kind in failure_kinds.GATE_DERIVED_KINDS and failure_kinds.is_gate_derived(kind)
    assert kind not in failure_kinds.FAILURE_KINDS
    assert not failure_kinds.is_registered(kind)
    with pytest.raises(ValueError, match="derived by the acceptance gate"):
        failure_kinds.require_failure_kind(kind)


def _timed_out_run():
    """One trial's policy request timed out on every attempt; 99 others are graded."""
    return _run(99, settled={"model_policy_response_timeout": 1})


def test_a_policy_timeout_is_the_models_fault_by_default():
    collected = _timed_out_run()
    _, (passed, reason, report), task = _gate(collected)
    assert report["model"]["kinds"] == {"model_policy_response_timeout": 1}, reason
    assert report["evaluation"]["server"] == 0
    assert task["criteria"]["evaluation"]["policy_timeout_counts_as_model"] is True


def test_a_run_that_owns_its_serving_can_count_a_policy_timeout_as_a_server_fault():
    from nemo_gym.orchestration.recovery import decide_from_collected

    collected = _timed_out_run()
    decision = decide_from_collected(collected, max_infra_retries=1, max_walltime_s=None, live_job=False)
    passed, reason, report = trial_verdict(decision, collected, policy_timeout_counts_as_model=False)
    assert report["evaluation"]["kinds"] == {"model_policy_response_timeout": 1}, reason
    assert report["evaluation"]["server"] == 1 and report["model"]["count"] == 0
    assert "(server 1: model_policy_response_timeout 1)" in reason
    task = gate_summary_from_collected(
        decision,
        collected,
        max_unhealthy_fraction=None,
        criteria=GateCriteria(policy_timeout_counts_as_model=False),
    )
    assert task["evaluation_health"]["evaluation"]["server"] == 1
    assert task["criteria"]["evaluation"]["policy_timeout_counts_as_model"] is False


def test_the_setting_is_a_strict_boolean():
    with pytest.raises(ValueError):
        GateCriteria(policy_timeout_counts_as_model="no")


@pytest.mark.parametrize("value,expected", [("false", False), ("true", True)])
def test_the_recover_command_sets_it_on_the_gate(tmp_path, fake_executor, capsys, value, expected):
    fake_executor.job_dir = complete_job(tmp_path)
    _, out = run_cli(
        ["eval", "resubmit", "gym-job-k", "bench", "--recover", "--policy-timeout-counts-as-model", value], capsys
    )
    assert out["gate_task"]["criteria"]["evaluation"]["policy_timeout_counts_as_model"] is expected


def test_ungraded_harness_failure_is_evaluation_side_not_model_invalid_output():
    from collections import Counter

    report = fault_report(Counter({"agent_route_failed": 1, "agent_invalid_output": 1}), 2)
    assert report["evaluation"]["kinds"] == {"agent_route_failed": 1}
    assert report["model"]["kinds"] == {"agent_invalid_output": 1}


def _policy_zero(*, kind="provider_oom_killed"):
    """A policy-owned explicit zero, requiring no downstream reconstruction of failure evidence."""
    from tests.unit_tests.orchestration.test_evaluation_health import _empty

    collected = _run(1)
    row = collected.results[0]
    _empty(row)
    row.update(
        failure_kind=kind,
        failure_outcome_action="count_as_zero",
        evaluation_completed=False,
        mask_sample=False,
    )
    return collected


@pytest.mark.parametrize("kind", ["provider_oom_killed", "provider_unavailable", "verifier_error"])
@pytest.mark.parametrize("omitted_report", [False, True])
def test_policy_counted_zero_stays_measured_incomplete_without_reinspecting_evidence(kind, omitted_report):
    from nemo_gym.orchestration.recovery import metric_coverage

    collected = _policy_zero(kind=kind)
    row = collected.results[0]
    if omitted_report:
        row["_ng_failure_events_omitted"] = True
    decision, (passed, reason, report), task = _gate(collected)
    assert decision.counts == {"scored": 1}
    assert decision.complete
    assert metric_coverage(decision, collected)["trials"] == 1
    assert row["evaluation_completed"] is False
    # Counting is the saved policy decision; it does not turn an infrastructure fact into model fault.
    assert report["model"]["count"] == 0
    if kind == "provider_oom_killed":
        assert report["unknown"]["kinds"] == {kind: 1}
        assert passed is None and report["status"] == "UNVERIFIED", reason
    else:
        assert report["evaluation"]["kinds"] == {kind: 1}
        assert passed is False and report["status"] == "FAIL", reason
    assert recovery._scored_kind(row) == kind
    facts = recovery.failure_facts(decision, collected)
    assert facts["response_failures"]["count"] == 0
    assert facts["trial_failures"]["count"] == 1
    assert task["failures"]["metric_coverage"]["trials"] == 1


@pytest.mark.parametrize(
    "overrides",
    [
        {"failure_outcome_action": None},
        {"failure_outcome_action": "fail"},
        {"failure_outcome_action": "verify"},
        {"failure_outcome_action": "retry"},
        {"failure_outcome_action": "invalid"},
        {"reward": 1.0},
        {"reward": False},
        {"mask_sample": True},
        {"mask_sample": None},
        {"mask_sample": 0},
        {"mask_sample": 1},
        {"mask_sample": "false"},
        {"mask_sample": "true"},
    ],
)
def test_incomplete_rows_do_not_infer_a_counted_zero_without_a_consistent_decision(overrides):
    from nemo_gym.core.outcome_policy import is_policy_counted_zero

    collected = _policy_zero()
    row = collected.results[0]
    row.update(overrides)
    assert not is_policy_counted_zero(row)
    decision, _, _ = _gate(collected)
    expected = "terminal" if overrides.get("failure_outcome_action") == "fail" else "masked"
    assert decision.counts == {expected: 1}


def test_old_incomplete_rows_remain_readable_without_inventing_a_decision():
    collected = _policy_zero()
    collected.results[0].pop("failure_outcome_action")
    decision, _, _ = _gate(collected)
    assert decision.counts == {"masked": 1}


def test_explicit_counted_zero_without_mask_uses_the_result_default():
    from nemo_gym.core.outcome_policy import is_policy_counted_zero

    collected = _policy_zero()
    collected.results[0].pop("mask_sample")
    assert is_policy_counted_zero(collected.results[0])
    decision, _, _ = _gate(collected)
    assert decision.counts == {"scored": 1}


def test_collector_preserves_policy_decision_with_omitted_failure_evidence(tmp_path):
    import json

    from tests.unit_tests.orchestration.test_eval_resubmit_decision import collect

    source = _policy_zero(kind="provider_unavailable")
    row = source.results[0]
    row["failure_outcome_attribution"] = "evaluation"
    row["_ng_failure_events_omitted"] = True
    row["response"]["metadata"] = {"unrelated_metadata": "private-metadata-marker"}
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    for name, rows in (
        ("rollouts_materialized_inputs.jsonl", source.materialized),
        ("rollouts.jsonl", source.results),
        ("rollouts_failures.jsonl", []),
    ):
        (artifacts / name).write_text("".join(json.dumps(item) + "\n" for item in rows))
    collected = collect(tmp_path)
    projected = collected.results[0]
    assert projected["failure_outcome_action"] == "count_as_zero"
    assert projected["failure_outcome_attribution"] == "evaluation"
    assert "private-metadata-marker" not in json.dumps(projected)
    decision, (passed, reason, report), task = _gate(collected)
    assert decision.counts == {"scored": 1}
    assert projected["evaluation_completed"] is False
    assert report["model"]["count"] == 0
    assert report["evaluation"]["kinds"] == {"provider_unavailable": 1}
    assert passed is False and report["status"] == "FAIL", reason
    assert task["failures"]["metric_coverage"]["trials"] == 1


@pytest.mark.parametrize("kind", ["provider_oom_killed", "provider_unavailable", "verifier_error"])
@pytest.mark.parametrize("attribution", ["model", "evaluation", "unknown"])
def test_trial_health_uses_saved_attribution_without_failure_evidence(kind, attribution):
    collected = _policy_zero(kind=kind)
    row = collected.results[0]
    row["failure_outcome_attribution"] = attribution
    row["_ng_failure_events_omitted"] = True
    decision, (passed, reason, report), task = _gate(collected)
    assert decision.counts == {"scored": 1}
    assert report[attribution]["kinds"] == {kind: 1}, reason
    assert all(report[side]["count"] == 0 for side in ("model", "evaluation", "unknown") if side != attribution)
    assert task["failures"]["metric_coverage"]["trials"] == 1
    assert passed is {"model": True, "evaluation": False, "unknown": None}[attribution], reason
    assert row["evaluation_completed"] is False


def test_same_failure_kind_can_have_different_saved_attribution_per_trial():
    collected = _run(3)
    for row, attribution in zip(collected.results, ["model", "evaluation", None], strict=True):
        row.update(
            reward=0.0,
            failure_kind="provider_oom_killed",
            failure_outcome_action="count_as_zero",
            failure_outcome_attribution=attribution,
            evaluation_completed=False,
            mask_sample=False,
        )
    decision, (passed, reason, report), _ = _gate(collected)
    assert decision.counts == {"scored": 3}
    assert report["model"]["kinds"] == {"provider_oom_killed": 1}
    assert report["evaluation"]["kinds"] == {"provider_oom_killed": 1}
    assert report["unknown"]["kinds"] == {"provider_oom_killed": 1}
    assert report["evaluation"]["server"] == 1
    assert passed is False, reason


@pytest.mark.parametrize("attribution", [None, "invalid", 0, False, {}])
def test_invalid_or_missing_saved_attribution_keeps_existing_health_classification(attribution):
    collected = _policy_zero()
    collected.results[0]["failure_outcome_attribution"] = attribution
    _, (_, _, report), _ = _gate(collected)
    assert report["unknown"]["kinds"] == {"provider_oom_killed": 1}
    assert report["model"]["count"] == 0


def test_explicit_unknown_attribution_does_not_fall_back_to_model_kind():
    collected = _policy_zero(kind="agent_timeout")
    collected.results[0]["failure_outcome_attribution"] = "unknown"
    _, (_, _, report), _ = _gate(collected)
    assert report["model"]["count"] == 0
    assert report["unknown"]["kinds"] == {"agent_timeout": 1}
    assert report["evaluation"]["count"] == 0


def test_bad_judge_verdict_preserves_other_trials_saved_attribution():
    from tests.unit_tests.orchestration.test_llm_judge_trial_coverage import _dep, _judge_call, _own

    collected = _run(100)
    collected.results[0].update(
        reward=0.0,
        failure_kind="provider_oom_killed",
        failure_outcome_action="count_as_zero",
        failure_outcome_attribution="model",
        evaluation_completed=False,
        mask_sample=False,
    )
    observed = collected.results[1]["ng_trial_observations"]
    grade = observed["verifier_invocations"][0]["grading_operations"][0]
    grade["kind"] = "llm_judge"
    call = _judge_call(observed)
    _own(observed, call)
    grade["dependencies"] = [_dep(call)]
    call["response_metadata"].update(dialect="chat", response_status=None, finish_reason="length")
    _, (passed, reason, report), _ = _gate(collected)
    assert report["model"]["kinds"] == {"provider_oom_killed": 1}, reason
    assert report["evaluation"]["kinds"] == {"grading_without_verdict": 1}, reason
    assert passed is True, reason  # One evaluation fault in 100 remains below the 2% gate.


@pytest.mark.parametrize("kind", ["provider_oom_killed", "verifier_error", "session_lost"])
def test_saved_fail_stops_masked_recovery_even_when_its_default_policy_would_retry(tmp_path, kind):
    from tests.unit_tests.orchestration.test_masked_infrastructure_recovery import _decide, _masked, _run

    output, _ = _run(
        tmp_path, [[_masked(kind) | {"failure_outcome_action": "fail", "failure_outcome_attribution": "evaluation"}]]
    )
    from nemo_gym.rollouts.rollout_store import RolloutStore

    assert RolloutStore.read(output, import_legacy=False).pending() == []
    _, decision = _decide(tmp_path, allocation_cut=False)
    assert decision.action.value == "stop"
    assert decision.counts == {"scored": 1, "terminal": 1}
    assert not decision.complete


def test_unfinished_run_health_preserves_saved_policy_attribution(tmp_path, monkeypatch):
    import json

    from nemo_gym.artifacts.run_artifacts import RunArtifacts
    from nemo_gym.cli.eval import _prepare_run_health_metrics

    collected = _policy_zero()
    collected.results[0]["failure_outcome_attribution"] = "model"
    collected.materialized.append({"_ng_task_index": 1, "_ng_rollout_index": 0, "_ng_collected_task_id": "1"})
    directory = tmp_path / "artifacts"
    directory.mkdir()
    output = directory / "rollouts.jsonl"
    output.write_text(json.dumps(collected.results[0]) + "\n")
    artifacts = RunArtifacts(output)
    artifacts.aggregate_metrics_json_fpath.write_text(
        json.dumps([{"agent_ref": {"type": "responses_api_agents", "name": "agent"}, "agent_metrics": {"score": 0.0}}])
    )
    monkeypatch.setattr(recovery, "select_managed_evidence", lambda *_: collected)
    values = _prepare_run_health_metrics(output)
    assert values["health/trial/expected"] == 2
    assert values["health/trial/failed_lower_bound"] == 0
    assert values["health/trial/failure_rate_lower_bound"] == 0


def _stage_zero(stage):
    collected = _policy_zero()
    collected.results[0].update(
        _ng_failure_id="final",
        _ng_failure_events={
            "final": {
                "event_id": "final",
                "boundary": "sandbox",
                "failure_kind": "provider_oom_killed",
                "stage": stage,
            }
        },
    )
    return collected


@pytest.mark.parametrize("stage,expected", [("agent", "model"), ("verification", "evaluation"), (None, "unknown")])
def test_trial_health_uses_the_final_event_stage(stage, expected):
    collected = _stage_zero(stage)
    _, (_, _, report), _ = _gate(collected)
    assert report[expected]["kinds"] == {"provider_oom_killed": 1}


def test_gate_configuration_roundtrip_changes_class_without_changing_saved_facts():
    from nemo_gym.orchestration.recovery import RecoveryDecision, RecoveryOutcomePolicyConfig, decide_from_collected

    collected = _stage_zero("agent")
    policy = RecoveryOutcomePolicyConfig.model_validate(
        {
            "classification": {
                "by_stage": {"agent": {"by_failure_kind": {"provider_oom_killed": "evaluation"}}},
            }
        }
    )
    policy = RecoveryOutcomePolicyConfig.model_validate_json(policy.model_dump_json())
    decision = decide_from_collected(
        collected, max_infra_retries=1, max_walltime_s=None, live_job=False, outcome_policy=policy
    )
    decision = RecoveryDecision.model_validate_json(decision.model_dump_json())
    gate = gate_summary_from_collected(decision, collected, max_unhealthy_fraction=None)
    assert gate["evaluation_health"]["evaluation"]["kinds"] == {"provider_oom_killed": 1}
    assert gate["evaluation_health"]["status"] == "FAIL"
    assert (
        gate["criteria"]["evaluation"]["classification"]["by_stage"]["agent"]["by_failure_kind"]["provider_oom_killed"]
        == "evaluation"
    )
    assert collected.results[0]["_ng_failure_events"]["final"]["stage"] == "agent"
    assert decision.counts == {"scored": 1}


def test_saved_unknown_class_wins_over_new_gate_override():
    from nemo_gym.core.failure_kinds import FailureClassificationConfig
    from nemo_gym.orchestration.recovery import decide_from_collected

    collected = _stage_zero("agent")
    collected.results[0]["failure_outcome_attribution"] = "unknown"
    decision = decide_from_collected(collected, max_infra_retries=1, max_walltime_s=None, live_job=False)
    passed, _, report = trial_verdict(
        decision,
        collected,
        classification=FailureClassificationConfig(by_failure_kind={"provider_oom_killed": "evaluation"}),
    )
    assert report["unknown"]["kinds"] == {"provider_oom_killed": 1}
    assert report["model"]["count"] == report["evaluation"]["count"] == 0
    assert passed is None


def test_recovered_stage_failure_does_not_turn_success_into_a_final_fault():
    collected = _run(1)
    before = _gate(collected)[1]
    row = collected.results[0]
    row["_ng_failure_events"] = {
        "recovered": {
            "event_id": "recovered",
            "boundary": "sandbox",
            "failure_kind": "provider_oom_killed",
            "stage": "verification",
        }
    }
    from nemo_gym.orchestration.recovery import CollectedEvidence

    collected = CollectedEvidence.model_validate_json(collected.model_dump_json())
    decision, (passed, reason, report), _ = _gate(collected)
    assert collected.results[0]["reward"] == 1.0
    assert decision.counts == {"scored": 1}
    assert decision.action.value == "noop"
    assert passed is before[0], reason
    assert reason == before[1]
    assert all(report[side]["count"] == 0 for side in ("model", "evaluation", "unknown"))


def test_collector_preserves_final_failure_stage_for_shared_health_classification(tmp_path):
    import json

    from tests.unit_tests.orchestration.test_eval_resubmit_decision import collect

    source = _stage_zero("agent")
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    for name, rows in (
        ("rollouts_materialized_inputs.jsonl", source.materialized),
        ("rollouts.jsonl", source.results),
    ):
        (artifacts / name).write_text("".join(json.dumps(row) + "\n" for row in rows))
    collected = collect(tmp_path)
    _, (passed, reason, report), _ = _gate(collected)
    assert report["model"]["kinds"] == {"provider_oom_killed": 1}
    assert passed is not False, reason  # Collection has no source-bound quality artifact in this fixture.


def test_recovery_uses_configured_class_after_policy_save_reload(tmp_path):
    from nemo_gym.orchestration.recovery import RecoveryOutcomePolicyConfig
    from tests.unit_tests.orchestration.test_eval_resubmit_decision import INFRA, _decide, ident, write_job_dir

    row = ident(0, **INFRA)
    row.update(
        _ng_failure_id="final",
        _ng_failure_events={
            "final": {
                "event_id": "final",
                "boundary": "sandbox",
                "failure_kind": "provider_unavailable",
                "stage": "seed",
            }
        },
    )
    job = write_job_dir(tmp_path / "job", n=1, results=[], failures=[row])
    policy = RecoveryOutcomePolicyConfig.model_validate(
        {
            "by_failure_class": {"model": "fail", "evaluation": "retry"},
            "classification": {"by_stage": {"seed": {"by_failure_kind": {"provider_unavailable": "model"}}}},
        }
    )
    policy = RecoveryOutcomePolicyConfig.model_validate_json(policy.model_dump_json())
    default = _decide(job)
    configured = _decide(job, policy)
    assert default.identities["retryable"] == [(0, 0)]
    assert configured.identities["terminal"] == [(0, 0)]
    assert configured.evidence[0].attempts == default.evidence[0].attempts
    assert configured.evidence[0].failure_kind == "provider_unavailable"
    assert configured.evidence[0].stage == "seed"


def test_recovery_rejects_zero_scoring_through_a_class_rule():
    from nemo_gym.orchestration.recovery import RecoveryOutcomePolicyConfig

    with pytest.raises(ValueError, match="does not support count_as_zero"):
        RecoveryOutcomePolicyConfig(by_failure_class={"model": "count_as_zero"})


@pytest.mark.parametrize("saved_class, expected", [(None, 1), ("unknown", 0)])
def test_cli_health_uses_saved_launcher_classification_and_exports_unknown(
    tmp_path, monkeypatch, saved_class, expected
):
    import json

    from nemo_gym.artifacts.run_artifacts import RunArtifacts
    from nemo_gym.cli.eval import _prepare_run_health_metrics

    collected = _stage_zero("agent")
    if saved_class is not None:
        collected.results[0]["failure_outcome_attribution"] = saved_class
    output = tmp_path / "benchmark" / "artifacts" / "rollouts.jsonl"
    output.parent.mkdir(parents=True)
    output.write_text(json.dumps(collected.results[0]) + "\n")
    RunArtifacts(output).aggregate_metrics_json_fpath.write_text(
        json.dumps([{"agent_ref": {"type": "responses_api_agents", "name": "agent"}, "agent_metrics": {"score": 0.0}}])
    )
    (tmp_path / "resolved-config.yaml").write_text(
        json.dumps(
            {
                "driver": {
                    "benchmarks": {
                        "benchmark": {
                            "resumable": {
                                "recovery_outcome_policy": {
                                    "classification": {
                                        "by_stage": {
                                            "agent": {"by_failure_kind": {"provider_oom_killed": "evaluation"}}
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        )
    )
    monkeypatch.setattr(recovery, "select_managed_evidence", lambda *_: collected)
    metrics = _prepare_run_health_metrics(output)
    assert metrics["health/trial/unclassified"] == int(saved_class == "unknown")
    assert metrics["health/trial/failed_lower_bound"] == expected
