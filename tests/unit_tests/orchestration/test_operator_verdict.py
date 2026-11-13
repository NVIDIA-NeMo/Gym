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
"""The operator verdict: accept, needs_action or unknown, and an UNVERIFIED that names its cause."""

import pytest

from nemo_gym.orchestration.recovery import (
    CollectedEvidence,
    IdentityEvidence,
    IdentityState,
    RecoveryAction,
    RecoveryDecision,
    RecoveryOutcome,
    operator_verdict,
    operator_verdict_line,
    with_operator_verdict,
)
from tests.unit_tests.orchestration.test_evaluation_health import _gate, _run


def _ready(collected):
    """A read score and no MLflow requirement, so only the run's own facts decide the gate."""
    collected.score = 0.9
    collected.quality["provenance"]["mlflow"]["tracking_required"] = False
    return collected


def _verdict(collected):
    decision, _, task = _gate(collected)
    return operator_verdict(task, decision), task, decision


def _decision(action, reason, counts, *, expected=10, **extra):
    identities = {state.value: [(i, 0) for i in range(n)] for state, n in counts.items()}
    return RecoveryDecision(
        action=action, reason=reason, expected=expected, counts=counts, identities=identities, evidence=[], **extra
    )


def _terminal(kind, n):
    return [IdentityEvidence(identity=(i, 0), state=IdentityState.TERMINAL, failure_kind=kind) for i in range(n)]


def _failed_task(**extra):
    return {"status": "FAILED", "expected_samples": 10, "errors": ["x"], "warnings": []} | extra


def test_a_clean_complete_run_is_accepted_with_nothing_to_do():
    verdict, task, _ = _verdict(_ready(_run(100)))
    assert task["status"] == "SUCCESS"
    assert verdict == {
        "state": "accept",
        "reason": "100 of 100 rollouts scored, run health PASS, nothing to recover",
        "action": "none",
    }
    assert operator_verdict_line(verdict) == (
        "operator verdict: accept: 100 of 100 rollouts scored, run health PASS, nothing to recover; action: none"
    )


def test_a_final_model_failure_is_still_accepted():
    verdict, task, _ = _verdict(_ready(_run(90, settled={"model_context_limit": 10})))
    assert task["status"] == "SUCCESS" and verdict["state"] == "accept" and verdict["action"] == "none"


def test_a_run_whose_every_rollout_ended_on_a_failed_model_call_is_not_accepted():
    collected = _ready(_run(32))
    collected.quality["issues"]["rollout_ended_on_failed_model_call"] = 32
    verdict, task, _ = _verdict(collected)
    assert task["status"] == "SUCCESS"
    assert verdict["state"] == "needs_action"
    assert verdict["reason"].startswith("every rollout ended on a failed model call, so the score most likely")
    assert "do not accept or report the score yet" in verdict["action"]
    assert verdict["action"].endswith(
        "read rollout quality (32 of 32 rollouts flagged rollout_ended_on_failed_model_call)"
    )


def test_a_run_where_no_task_had_a_successful_model_call_is_not_accepted():
    collected = _ready(_run(8))
    collected.quality["issues"]["task_no_successful_model_calls"] = 8
    verdict, _, _ = _verdict(collected)
    assert verdict["state"] == "needs_action"
    assert verdict["reason"].startswith("no task had a successful model call, so the score most likely")
    assert "8 of 8 tasks flagged task_no_successful_model_calls" in verdict["action"]


def test_a_partly_flagged_run_is_still_accepted():
    collected = _ready(_run(32))
    collected.quality["issues"]["rollout_ended_on_failed_model_call"] = 31
    verdict, _, _ = _verdict(collected)
    assert verdict["state"] == "accept" and verdict["action"] == "none"


def test_a_task_level_finding_counts_tasks_not_rollouts():
    identities = {IdentityState.SCORED.value: [(task, rollout) for task in range(4) for rollout in range(2)]}
    decision = RecoveryDecision(
        action=RecoveryAction.NOOP,
        reason="every identity is scored",
        expected=8,
        counts={IdentityState.SCORED: 8},
        identities=identities,
        evidence=[],
    )
    task = {
        "status": "SUCCESS",
        "expected_samples": 8,
        "errors": [],
        "evaluation_health": {"status": "PASS"},
        "rollout_quality": {"verdicts": None, "issues": {"task_no_successful_model_calls": 4}},
    }
    verdict = operator_verdict(task, decision)
    assert verdict["state"] == "needs_action"
    assert "read rollout quality (4 of 4 tasks flagged task_no_successful_model_calls)" in verdict["action"]


def test_some_flagged_rollouts_are_a_signal_not_a_reason_to_change_the_verdict():
    collected = _ready(_run(100))
    collected.quality["issues"]["rollout_ended_on_failed_model_call"] = 5
    verdict, _, _ = _verdict(collected)
    assert verdict["state"] == "accept" and verdict["action"] == "none"


def test_evaluation_side_failures_count_over_the_trials_not_over_accepted_omissions():
    from nemo_gym.artifacts.run_artifacts import ACCEPTED_STAGE_OMISSION

    run = _ready(_run(97, settled={"model_transport_error": 3})).model_dump()
    for index in range(100, 105):
        omitted = {"_ng_task_index": index, "_ng_rollout_index": 0, "_ng_collected_task_id": str(index)}
        omitted[ACCEPTED_STAGE_OMISSION] = {"stage_index": 1, "tolerate_unresolved": True}
        run["materialized"].append(omitted)
        run["managed_selection"].append(
            {"identity": [index, 0], "attempts": 0, "max_attempts": 3, "pending": False, "reserved": True}
        )
    run["quality"]["artifacts"]["expected"] = 105
    verdict, task, decision = _verdict(CollectedEvidence.model_validate(run))
    assert decision.counts[IdentityState.OMITTED] == 5 and task["expected_samples"] == 105
    assert task["evaluation_health"]["status"] == "FAIL" and task["evaluation_health"]["trials"] == 100
    assert verdict["reason"] == "3 of 100 trials failed on the evaluation side (model_transport_error 3)"


def test_evaluation_side_failures_over_the_limit_need_action():
    verdict, task, _ = _verdict(_ready(_run(97, settled={"model_transport_error": 3})))
    assert task["status"] == "FAILED"
    assert verdict["state"] == "needs_action"
    assert verdict["reason"] == "3 of 100 trials failed on the evaluation side (model_transport_error 3)"
    assert verdict["action"] == (
        "none: report the run as failing run health; no recovery reopens settled trials, so check the "
        "evaluation-side faults evaluation_health names (their kinds and the services' logs)"
    )


def test_recoverable_work_says_recover_now():
    decision = _decision(
        RecoveryAction.RESUBMIT, "2 failures can retry", {IdentityState.SCORED: 8, IdentityState.RETRYABLE: 2}
    )
    verdict = operator_verdict({"status": "PLANNED", "expected_samples": 10, "errors": []}, decision)
    assert verdict == {
        "state": "needs_action",
        "reason": "8 of 10 rollouts are settled; 2 retryable remain",
        "action": "`gym eval resubmit GYM_JOB_ID BENCHMARK --recover` now",
    }


def test_a_quota_stop_waits_for_the_reset_then_recovers():
    decision = _decision(
        RecoveryAction.STOP,
        "6 saved answer(s) wait on the grading provider's quota (provider_quota_exhausted), which resets at "
        "2026-01-02 03:04 UTC; nothing can be graded before then",
        {IdentityState.SCORED: 4, IdentityState.TERMINAL: 6},
    )
    decision.evidence = _terminal("provider_quota_exhausted", 6)
    verdict = operator_verdict(_failed_task(), decision)
    assert verdict["state"] == "needs_action"
    assert verdict["reason"] == "6 of 10 saved answers wait on the grading provider quota"
    assert verdict["action"] == (
        "wait until 2026-01-02 03:04 UTC, when the grading provider's quota resets, then run "
        "`gym eval resubmit GYM_JOB_ID BENCHMARK --recover`"
    )


def test_a_credentials_stop_names_the_fix_then_the_recovery_as_gyms_own_stop_text_does():
    decision = _decision(
        RecoveryAction.STOP,
        "10 failure(s) typed provider_credentials_rejected: the sandbox provider answered 401",
        {IdentityState.TERMINAL: 10},
    )
    decision.evidence = _terminal("provider_credentials_rejected", 10)
    verdict = operator_verdict(_failed_task(), decision)
    assert verdict["state"] == "needs_action"
    assert verdict["action"] == "fix the credentials, then run `gym eval resubmit GYM_JOB_ID BENCHMARK --recover`"


def test_a_stop_no_rule_knows_names_no_command_and_never_a_fresh_submit():
    decision = _decision(RecoveryAction.STOP, "96 failure(s) have no registered kind", {IdentityState.UNKNOWN: 10})
    decision.evidence = [
        IdentityEvidence(identity=(i, 0), state=IdentityState.UNKNOWN, failure_kind="collection_dispatch_failed")
        for i in range(10)
    ]
    verdict = operator_verdict(_failed_task(), decision)
    assert verdict["state"] == "needs_action"
    assert verdict["action"] == "read the stop reason; no recovery command is known for it"


def test_incomplete_coverage_needs_action():
    decision = _decision(RecoveryAction.NOOP, "x", {IdentityState.SCORED: 8, IdentityState.MISSING: 2})
    verdict = operator_verdict({"status": "SUCCESS", "expected_samples": 10, "errors": []}, decision)
    assert verdict["state"] == "needs_action"
    assert verdict["reason"] == "coverage is incomplete: 2 missing"
    assert verdict["action"] == "`gym eval resubmit GYM_JOB_ID BENCHMARK --recover` now"


def test_a_live_job_and_missing_evidence_are_unknown():
    live = _decision(RecoveryAction.NOOP, "x", {IdentityState.SCORED: 5}, scheduler_live=True)
    assert operator_verdict({"status": "RUNNING", "errors": []}, live)["state"] == "unknown"
    nothing = operator_verdict(None, None)
    assert nothing == {
        "state": "unknown",
        "reason": "no evidence was collected",
        "action": "inspect again once the reader runs",
    }
    unread = _decision(RecoveryAction.UNKNOWN, "nothing was read or decided", {})
    assert operator_verdict(_failed_task(), unread)["state"] == "unknown"


def test_an_unread_score_is_unknown_not_a_failed_run():
    collected = _run(100)
    verdict, task, _ = _verdict(collected)
    assert task["status"] == "FAILED"
    assert verdict["state"] == "unknown" and verdict["reason"] == "no score was read"


def _uncovered(collected, rows):
    for row in collected.results[:rows]:
        row["ng_trial_observations"]["verifier_invocations"][0]["grading_operations"] = None
    return collected


def _judge_failure(task, status):
    return {
        "_ng_task_index": task,
        "_ng_rollout_index": 0,
        "failure_kind": "judge_failed",
        "_ng_failure_http_status": status,
        "_ng_failure_terminal": False,
        "mask_sample": True,
    }


def test_unverified_after_a_judge_outage_names_the_cause_with_counts():
    collected = _ready(_uncovered(_run(64), 59))
    collected.failures = [_judge_failure(task, 401) for task in range(59) for _ in range(2)]
    verdict, task, _ = _verdict(collected)
    assert task["status"] == "UNVERIFIED"
    cause = "59 of 64 rows were re-judged after a judge failure (401 from the judge model server); 59 rows have no recorded grading"
    assert task["unverified_cause"] == {"kind": "judge_rejudged", "reason": cause}
    [warning] = [w for w in task["warnings"] if w.startswith("response/trial health unknown")]
    assert warning == f"response/trial health unknown: ['{cause}']"
    assert "trial: UNOBSERVED 59 of 64" in task["evaluation_health"]["detail"]
    # The judge answered the re-judgment; a fresh submit would sample every policy rollout again.
    assert verdict == {
        "state": "unknown",
        "reason": f"unverified: {cause}",
        "action": "none: report the run as unverified: the rows were re-judged and no grading was recorded, so "
        "whether they were graded is unknown; no recovery command records it",
    }


def test_unverified_without_a_cause_is_unknown():
    collected = _ready(_run(100, settled={}))
    collected.quality["artifacts"]["healthy"] = True
    decision, _, task = _gate(collected)
    task = {**task, "status": "UNVERIFIED", "unverified_cause": {"kind": "zero_rewards_unprovable", "reason": "r"}}
    verdict = operator_verdict(task, decision)
    assert verdict["state"] == "unknown"
    assert verdict["reason"] == "unverified without a known cause: r"
    assert verdict["action"] == "none: report the run as unverified; no operator action is known"


def test_unverified_unobserved_grading_is_reported_with_its_producer_never_resubmitted():
    verdict, task, _ = _verdict(_ready(_uncovered(_run(100), 10)))
    assert task["status"] == "UNVERIFIED"
    assert task["unverified_cause"]["kind"] == "grading_unobserved"
    # UNVERIFIED by construction: a resubmit cannot change it, and no flag does (benchmark-notes).
    assert verdict["state"] == "unknown"
    assert verdict["reason"].startswith("unverified: 10 of 100 rows have no recorded grading")
    assert verdict["action"] == (
        "none: report the run as unverified with this reason and do not present it as verified; a resubmit cannot "
        "change it, and no flag does: the fix is the verifier's grading record, in Gym"
    )


def test_the_outcome_carries_the_verdict_in_both_places():
    decision, _, task = _gate(_ready(_run(10)))
    outcome = with_operator_verdict(RecoveryOutcome(decision=decision, ok=True, dispatched=None, gate_task=task))
    assert outcome.operator_verdict == outcome.gate_task["operator_verdict"]
    assert outcome.operator_verdict["state"] == "accept"
    assert "operator_verdict" in outcome.model_dump(mode="json")
    assert (
        with_operator_verdict(RecoveryOutcome(decision=decision, ok=False, dispatched=None)).operator_verdict["state"]
        == "unknown"
    )


def test_a_proven_health_failure_is_kept_when_the_score_is_also_unread():
    verdict, task, _ = _verdict(_run(97, settled={"model_transport_error": 3}))
    assert any(e.startswith("score unavailable") for e in task["errors"])
    assert task["evaluation_health"]["status"] == "FAIL"
    assert verdict == {
        "state": "needs_action",
        "reason": "3 of 100 trials failed on the evaluation side (model_transport_error 3)",
        "action": "none: report the run as failing run health; no recovery reopens settled trials, so check the "
        "evaluation-side faults evaluation_health names (their kinds and the services' logs)",
    }


def test_an_unread_score_with_a_failed_summary_validity_is_not_score_only_unknown():
    collected = _run(100)
    collected.quality["artifacts"]["healthy"] = False
    verdict, task, _ = _verdict(collected)
    assert task["errors"][0].startswith("score unavailable") and task["errors"][1] == "summary validity failed"
    # The known summary failure is the dominant cause, not the unread score listed before it.
    assert verdict == {
        "state": "needs_action",
        "reason": "summary validity failed; the score was not read either",
        "action": "read the error; no recovery command is known for it",
    }


def _spent(reason, counts):
    return _decision(RecoveryAction.STOP, reason, counts)


def test_exhausted_attempts_name_the_extension_not_a_fresh_submit():
    decision = _spent(
        "2 identity(ies) used every attempt since the last resubmit, so a resume would run nothing",
        {IdentityState.SCORED: 8, IdentityState.EXHAUSTED: 2},
    )
    verdict = operator_verdict(_failed_task(), decision)
    assert verdict["state"] == "needs_action"
    assert "`gym eval resubmit GYM_JOB_ID BENCHMARK --recover --extend-attempts N`" in verdict["action"]
    assert "completed work is kept" in verdict["action"] and "submit" not in verdict["action"].replace("resubmit", "")


def test_spent_infra_retries_and_walltime_name_their_options():
    retries = _spent(
        "the stored infra-retry budget is used (3 of 3)", {IdentityState.SCORED: 8, IdentityState.RETRYABLE: 2}
    )
    action = operator_verdict(_failed_task(), retries)["action"]
    assert "--extend-retries N" in action and "--max-walltime" not in action and "--extend-attempts" not in action
    walltime = _spent(
        "the stored walltime budget is used (3600 of 3600 s)", {IdentityState.SCORED: 8, IdentityState.RETRYABLE: 2}
    )
    action = operator_verdict(_failed_task(), walltime)["action"]
    assert "--max-walltime HH:MM:SS --walltime-reason TEXT" in action and "--extend-retries" not in action
    both = _spent(
        "the stored infra-retry budget is used (3 of 3), and the stored walltime budget is used (3600 of 3600 s; "
        "less than a minute is left): raise both in one call",
        {IdentityState.SCORED: 8, IdentityState.RETRYABLE: 2},
    )
    action = operator_verdict(_failed_task(), both)["action"]
    assert "--extend-retries N --max-walltime" in action and "same call" in action
    assert "fresh" not in action and "`gym eval submit`" not in action


def test_non_budget_stops_keep_their_typed_remedy():
    decision = _decision(
        RecoveryAction.STOP, "10 failure(s) typed provider_credentials_rejected", {IdentityState.TERMINAL: 10}
    )
    decision.evidence = _terminal("provider_credentials_rejected", 10)
    assert operator_verdict(_failed_task(), decision)["action"].startswith("fix the credentials, then")


def _refused_outcome(refusal):
    decision = _decision(
        RecoveryAction.RESUBMIT, "2 failures can retry", {IdentityState.SCORED: 8, IdentityState.RETRYABLE: 2}
    )
    task = {"status": "FAILED", "expected_samples": 10, "errors": [], "warnings": []}
    outcome = RecoveryOutcome(decision=decision, ok=False, dispatched=None, gate_task=task, scheduler_refusal=refusal)
    return with_operator_verdict(outcome)


def test_a_definite_scheduler_refusal_is_not_a_recover_now_plan():
    refusal = {"scheduler_text": "sbatch: error: Invalid time limit", "job_dir_restored": True, "chain_marker": None}
    outcome = _refused_outcome(refusal)
    verdict = outcome.operator_verdict
    assert verdict["state"] == "needs_action"
    assert "sbatch: error: Invalid time limit" in verdict["reason"]
    assert "recover` now" not in verdict["action"] and "correct" in verdict["action"]
    assert outcome.gate_task["operator_verdict"] == verdict
    assert outcome.decision.action == RecoveryAction.RESUBMIT and outcome.scheduler_refusal == refusal


def test_an_ambiguous_scheduler_refusal_asks_for_reconciliation_first():
    refusal = {
        "scheduler_text": "sbatch: error: Socket timed out",
        "job_dir_restored": False,
        "chain_marker": "accounting unknown",
    }
    verdict = _refused_outcome(refusal).operator_verdict
    assert "unknown" in verdict["reason"] and "queued" in verdict["reason"]
    assert "reconcile" in verdict["action"] and "recover` now" not in verdict["action"]


def test_a_walltime_refusal_before_the_evidence_read_is_not_a_reader_failure():
    decision = _decision(RecoveryAction.STOP, "walltime raise refused: partition batch MaxTime 2:00:00 is low", {})
    plan = {"refused": "partition batch MaxTime 2:00:00 is below the run's segment walltime 04:00:00"}
    outcome = with_operator_verdict(RecoveryOutcome(decision=decision, ok=False, dispatched=None, walltime_plan=plan))
    verdict = outcome.operator_verdict
    assert outcome.gate_task is None
    assert verdict["state"] == "needs_action" and plan["refused"] in verdict["reason"]
    assert "reader" not in verdict["action"] and "inspect again" not in verdict["action"]
    # A genuinely missing reader stays unknown.
    unread = with_operator_verdict(RecoveryOutcome(decision=decision, ok=False, dispatched=None))
    assert unread.operator_verdict["state"] == "unknown"


def _two_rows():
    collected = _ready(_run(2))
    collected.quality["source_matches"] = False
    return collected


def test_an_explicit_zero_unobserved_count_is_not_a_missing_grading_count():
    collected = _two_rows()
    collected.failures = [_judge_failure(0, 401)]
    verdict, task, _ = _verdict(collected)
    assert task["trial_counts"]["unobserved"] == 0
    assert "known failures are a lower bound" in task["evaluation_health"]["detail"]
    assert task["unverified_cause"] == {
        "kind": "zero_rewards_unprovable",
        "reason": "failures recorded only as zero rewards cannot be ruled out over 2 rows",
    }
    assert verdict == {
        "state": "unknown",
        "reason": "unverified without a known cause: failures recorded only as zero rewards cannot be ruled out "
        "over 2 rows",
        "action": "none: report the run as unverified; no operator action is known",
    }


def test_recovered_judge_history_does_not_change_the_lower_bound_cause_or_action():
    # Both rows have bound grading, so nothing is unobserved: the judge failure on task 0 has recovered.
    without = _verdict(_two_rows())
    collected = _two_rows()
    collected.failures = [_judge_failure(0, 401)]
    with_history = _verdict(collected)
    assert with_history[1]["unverified_cause"] == without[1]["unverified_cause"]
    assert with_history[1]["unverified_cause"]["kind"] == "zero_rewards_unprovable"
    assert with_history[0] == without[0] and with_history[0]["state"] == "unknown"
    assert "judge" not in with_history[0]["action"]


def test_recovered_judge_history_does_not_displace_an_unrelated_current_cause():
    collected = _two_rows()
    collected.results[1]["ng_trial_observations"]["verifier_invocations"][0]["grading_operations"] = None
    collected.failures = [_judge_failure(0, 401)]
    _, _, task = _gate(collected)
    assert task["unverified_cause"]["kind"] == "grading_unobserved"
    assert task["unverified_cause"]["reason"].startswith("1 of 2 rows have no recorded grading")
    # The same history on the identity whose grading is unresolved is the judge cause, counted once.
    collected = _two_rows()
    collected.results[0]["ng_trial_observations"]["verifier_invocations"][0]["grading_operations"] = None
    collected.failures = [_judge_failure(0, 401), _judge_failure(0, 401)]
    _, _, task = _gate(collected)
    assert task["unverified_cause"] == {
        "kind": "judge_rejudged",
        "reason": "1 of 2 rows were re-judged after a judge failure (401 from the judge model server); "
        "1 rows have no recorded grading",
    }


def _spent_attempts_and_retries(tmp_path):
    """One scored and one exhausted rollout of a Store run whose infra-retry budget is used (3 of 3)."""
    from nemo_gym.orchestration.recovery import select_managed_evidence
    from tests.unit_tests.orchestration.test_managed_recovery_selection import collect, prepare_managed

    prepare_managed(tmp_path, "exhausted", n=2)
    (tmp_path / ".gym_infra_retries").write_text("3\n")
    return select_managed_evidence(collect(tmp_path), tmp_path)


def _budget_verdict(collected, **options):
    from nemo_gym.orchestration.recovery import decide_from_collected, gate_summary_from_collected

    decision = decide_from_collected(collected, max_infra_retries=3, max_walltime_s=None, live_job=False, **options)
    task = gate_summary_from_collected(decision, collected, max_unhealthy_fraction=None, mlflow_status="none")
    return decision, operator_verdict(task, decision)


def test_spent_attempts_and_infra_retries_name_both_extensions_at_every_stop(tmp_path):
    collected = _spent_attempts_and_retries(tmp_path)
    both = "`gym eval resubmit GYM_JOB_ID BENCHMARK --recover --extend-attempts N --extend-retries N`"
    # The first stop already names both budgets; following its advice with either flag alone stops again, and that
    # stop keeps naming the flag the call asked for, which nothing recorded.
    for options in ({}, {"extend_attempts": 1}, {"extend_retries": 1}):
        decision, verdict = _budget_verdict(collected, **options)
        assert decision.action == RecoveryAction.STOP, options
        assert verdict["state"] == "needs_action" and both in verdict["action"], (options, verdict)
        assert "same call" in verdict["action"] and "walltime" not in verdict["action"]
        assert "`gym eval submit`" not in verdict["action"]
    # Both in one call continue the saved run: the scored row is kept and only the exhausted one runs again.
    decision, verdict = _budget_verdict(collected, extend_attempts=1, extend_retries=1)
    assert decision.action == RecoveryAction.RESUBMIT and decision.counts == {"scored": 1, "retryable": 1}
    assert verdict["action"] == "`gym eval resubmit GYM_JOB_ID BENCHMARK --recover` now"


def test_spent_attempts_alone_still_name_only_the_attempt_extension(tmp_path):
    from nemo_gym.orchestration.recovery import select_managed_evidence
    from tests.unit_tests.orchestration.test_managed_recovery_selection import collect, prepare_managed

    prepare_managed(tmp_path, "exhausted", n=2)
    decision, verdict = _budget_verdict(select_managed_evidence(collect(tmp_path), tmp_path))
    assert decision.action == RecoveryAction.STOP
    assert verdict["action"] == (
        "your decision: continue the saved run (completed work is kept): "
        "`gym eval resubmit GYM_JOB_ID BENCHMARK --recover --extend-attempts N`; N is yours to choose"
    )


def test_reservations_without_an_outcome_wait_for_the_allocation_not_a_fresh_submit(tmp_path):
    from nemo_gym.orchestration.recovery import decide_from_collected, select_managed_evidence
    from tests.unit_tests.orchestration.test_managed_recovery_selection import collect, prepare_managed

    prepare_managed(tmp_path, "prior_reservation")
    collected = select_managed_evidence(collect(tmp_path), tmp_path)
    decision = decide_from_collected(collected, max_infra_retries=3, max_walltime_s=None, live_job=False)
    assert decision.action == RecoveryAction.STOP and "whether they ran is unknown" in decision.reason

    verdict = operator_verdict({"status": "FAILED", "expected_samples": 1, "errors": []}, decision)

    assert verdict["state"] == "needs_action" and "`gym eval submit`" not in verdict["action"]
    assert verdict["action"] == (
        "confirm with the scheduler that the allocation that made the reservations has ended, then inspect again: "
        "a recovery resumes them once that is known (completed work is kept)"
    )


@pytest.mark.parametrize(
    "reason,action",
    [
        (
            "admitted re-judgment outcome unknown for [(0, 0)]: the scheduler query failed, so the pass that "
            "admitted them may still run",
            "wait until the scheduler answers and inspect again (completed work is kept)",
        ),
        (
            "saved rows are unreadable (rollouts.jsonl), so what they hold is unknown",
            "make the saved rows readable again and inspect again (completed work is kept)",
        ),
        (
            "the run was written by a rollout store (an attempt manifest, an attempt journal or run-tagged rows); an "
            "authoritative Store reader is unavailable or could not verify these artifacts; recovery stops",
            "inspect again once the run's Store reader can verify its artifacts (completed work is kept)",
        ),
        (
            "saved-answer rejudge admission changed; collect the current evidence",
            "inspect again: the evidence changed after the recovery was planned (completed work is kept)",
        ),
        (
            "saved source or accounting snapshot is unavailable; collect the evidence again",
            "inspect again to collect the evidence the judge-only job needs (completed work is kept)",
        ),
    ],
)
def test_a_stop_whose_evidence_is_unknown_waits_and_inspects_again_not_a_fresh_submit(reason, action):
    decision = _decision(RecoveryAction.STOP, reason, {IdentityState.UNKNOWN: 10})
    verdict = operator_verdict(_failed_task(), decision)
    assert verdict == {
        "state": "unknown",
        "reason": f"recovery stopped with 10 unknown: {reason[:160]}",
        "action": action,
    }


COMMIT = "b" * 40


@pytest.mark.parametrize(
    "reason,command",
    [
        (
            "saved judge driver and installed recovery source must be the same immutable revision; --repin <this "
            "launcher's full commit> --repin-reason TEXT runs the judge-only job at it",
            "gym eval resubmit GYM_JOB_ID BENCHMARK --recover --judge-partition cpu --repin COMMIT --repin-reason TEXT",
        ),
        (
            f"planned on {COMMIT} for a run pinned at {'a' * 40}: its judge-only job runs the installed recovery "
            f"source; the pin was kept (--no-auto-repin): recover with --repin {COMMIT} --repin-reason TEXT to move it",
            f"gym eval resubmit GYM_JOB_ID BENCHMARK --recover --judge-partition cpu --repin {COMMIT} --repin-reason TEXT",
        ),
    ],
)
def test_a_stop_that_needs_the_launchers_revision_names_the_repin_not_a_fresh_submit(reason, command):
    # The command keeps the call's other options; --no-auto-repin goes, as the CLI refuses it beside --repin.
    decision = _decision(
        RecoveryAction.STOP,
        reason,
        {IdentityState.JUDGE_RECOVERABLE: 1},
        requested=["--no-auto-repin", "--judge-partition cpu"],
    )
    verdict = operator_verdict(_failed_task(), decision)
    assert verdict["state"] == "needs_action"
    assert verdict["action"].split("`")[1] == command and "`gym eval submit`" not in verdict["action"]


def test_an_accepted_run_whose_mlflow_delivery_is_pending_names_the_export():
    collected = _run(100)
    collected.score = 0.9  # tracking is required and the run has no MLflow evidence yet
    # Its saved provenance says MLflow is configured, so a recovery exports it.
    collected.quality["provenance"]["mlflow"]["configured"] = True
    collected.mlflow = {"live_run_id": None, "receipt": None}
    verdict, task, _ = _verdict(collected)
    assert task["status"] == "SUCCESS" and task["mlflow_delivery"]["state"] == "pending"
    assert task["mlflow_export_due"] is True
    assert verdict["state"] == "accept" and "MLflow delivery pending" in verdict["reason"]
    assert verdict["action"] == (
        "export the results to MLflow: `gym eval resubmit GYM_JOB_ID BENCHMARK --recover` runs the export of a complete run without a "
        "receipt (no model calls)"
    )
