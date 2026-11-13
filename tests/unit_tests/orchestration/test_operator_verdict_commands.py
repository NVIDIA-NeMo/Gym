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
"""The operator verdict prints one complete command: `gym eval resubmit <job id> <benchmark> --recover`, every option
the call asked for, the typed stop's fix and what the stop needs. Where Gym decided nothing is left to run, it reports
instead of inventing a fresh submit."""

import json

import pytest

from nemo_gym.orchestration import submit
from nemo_gym.orchestration.executors import resume_script
from nemo_gym.orchestration.recovery import (
    IdentityEvidence,
    IdentityState,
    RecoveryAction,
    RecoveryDecision,
    decide_from_collected,
    operator_verdict,
)
from tests.unit_tests.orchestration.test_eval_resubmit import _handle
from tests.unit_tests.orchestration.test_eval_resubmit import _local_index as _local_index
from tests.unit_tests.orchestration.test_eval_resubmit import slurm_bin as slurm_bin
from tests.unit_tests.orchestration.test_eval_resubmit_decision import fake_executor as fake_executor
from tests.unit_tests.orchestration.test_eval_resubmit_decision import mlflow_job
from tests.unit_tests.orchestration.test_evaluation_health import _run
from tests.unit_tests.orchestration.test_fresh_restart import _cut_before_inventory
from tests.unit_tests.orchestration.test_resubmit_repin import launcher_at as launcher_at
from tests.unit_tests.orchestration.test_resubmit_walltime import (
    _launcher_at_the_pinned_revision as _launcher_at_the_pinned_revision,
)
from tests.unit_tests.orchestration.test_resubmit_walltime import _timed_run


RUN = "gym eval resubmit gym-job-x gsm8k --recover"


def _spent_after(tmp_path, reason: str):
    """A run with one rollout missing whose infra retries are spent (3 of 3) and whose chain stopped on ``reason``."""
    job_dir = _timed_run(tmp_path)
    (job_dir / ".gym_infra_retries").write_text("3\n")
    (job_dir / resume_script.INCOMPLETE_MARKER_NAME).write_text(f"Previous job 100 ended FAILED: {reason}\n")
    return job_dir


@pytest.mark.parametrize(
    ("reason", "options", "action"),
    [
        (
            resume_script.SPINUP_TIMEOUT_REASON,
            {},
            "raise the server start-up limit above the recorded one with "
            f"`{RUN} --server-spinup-timeout-seconds N --spinup-timeout-reason TEXT --extend-retries N`",
        ),
        (
            resume_script.SPINUP_TIMEOUT_REASON,
            {"server_spinup_timeout_seconds": 3600, "spinup_timeout_reason": "slow image pull"},
            f"`{RUN} --server-spinup-timeout-seconds 3600 --spinup-timeout-reason TEXT --extend-retries N`",
        ),
        (
            resume_script.CREDENTIALS_REJECTED_REASON,
            {"judge_partition": "cpu", "judge_qos": "normal"},
            f"fix the credentials, then run `{RUN} --judge-partition cpu --judge-qos normal --extend-retries N`",
        ),
    ],
)
def test_a_spent_budget_after_a_typed_chain_stop_names_the_fix_the_extension_and_the_calls_options(
    tmp_path, slurm_bin, reason, options, action
):
    job_dir = _spent_after(tmp_path, reason)

    out = submit.recover(*_handle(job_dir), **options) if options else submit.inspect_recovery(*_handle(job_dir))

    assert (out.decision.action.value, out.dispatched) == ("stop", None), out.decision.reason
    assert out.operator_verdict["state"] == "needs_action"
    assert out.operator_verdict["action"].startswith(
        f"your decision: continue the saved run (completed work is kept): {action}"
    ), out.operator_verdict


def test_a_quota_chain_note_waits_for_the_reset_and_keeps_the_extension(tmp_path, slurm_bin):
    job_dir = _spent_after(tmp_path, resume_script.QUOTA_EXHAUSTED_REASON)

    out = submit.inspect_recovery(*_handle(job_dir))

    verdict = out.operator_verdict
    assert out.decision.counts == {"scored": 1, "missing": 1}
    # The stop is the spent budget; the one open row is missing, not a saved answer waiting on the quota.
    assert verdict["reason"].startswith("recovery stopped with 1 missing: the stored infra-retry budget is used")
    assert verdict["action"] == (
        "your decision: continue the saved run (completed work is kept): wait until the grading provider's quota "
        f"resets, then run `{RUN} --extend-retries N`; N is yours to choose"
    )


def _decision(action, reason, counts, **extra):
    identities = {state.value: [(i, 0) for i in range(n)] for state, n in counts.items()}
    extra.setdefault("evidence", [])
    return RecoveryDecision(action=action, reason=reason, expected=10, counts=counts, identities=identities, **extra)


def _terminal(kind: str, n: int) -> list[IdentityEvidence]:
    return [IdentityEvidence(identity=(i, 0), state=IdentityState.TERMINAL, failure_kind=kind) for i in range(n)]


def test_the_quota_stop_is_read_from_the_typed_evidence_not_the_reason_text():
    task = {"status": "FAILED", "expected_samples": 10, "errors": ["x"], "warnings": []}
    counts = {IdentityState.SCORED: 7, IdentityState.TERMINAL: 3}
    owner = "3 terminal unscored failure(s) need an owner decision; the run is not complete"
    # Owner-decision failures stay an owner decision whatever the chain or the reason text says about a quota.
    for extra in ({"chain_stop_kind": "provider_quota_exhausted"}, {}):
        decision = _decision(
            RecoveryAction.STOP,
            f"{owner}; the chain recorded: {resume_script.QUOTA_EXHAUSTED_REASON}",
            counts,
            evidence=_terminal("benchmark_failed", 3),
            **extra,
        )
        verdict = operator_verdict(task, decision)
        assert verdict["reason"].startswith("recovery stopped with 3 terminal: 3 terminal unscored"), verdict
        assert verdict["action"] == "read the stop reason; no recovery command is known for it"
    # Gym's own quota stop: every terminal row records the quota kind.
    decision = _decision(
        RecoveryAction.STOP,
        "3 saved answer(s) wait on the grading provider's quota (provider_quota_exhausted), which resets at "
        "2026-01-02 03:04 UTC; nothing can be graded before then",
        counts,
        evidence=_terminal("provider_quota_exhausted", 3),
        requested=["--judge-partition cpu"],
    )
    assert operator_verdict(task, decision) == {
        "state": "needs_action",
        "reason": "3 of 10 saved answers wait on the grading provider quota",
        "action": "wait until 2026-01-02 03:04 UTC, when the grading provider's quota resets, then run "
        "`gym eval resubmit GYM_JOB_ID BENCHMARK --recover --judge-partition cpu`",
    }


def test_an_inspection_repeats_the_pin_choice_and_the_gate_options(tmp_path, slurm_bin):
    job_dir = _timed_run(tmp_path)

    out = submit.inspect_recovery(
        *_handle(job_dir),
        auto_repin=False,
        score_key=["agent", "mean/reward"],
        score_agent="judge agent",
        max_unhealthy_fraction=0.02,
        policy_timeout_counts_as_model=False,
        mlflow_run_id="r-1",
        mlflow_status="FINISHED",
    )

    assert out.decision.action.value == "resubmit", out.decision.reason
    assert out.operator_verdict["action"] == (
        f"`{RUN} --no-auto-repin --score-key '[\"agent\", \"mean/reward\"]' --score-agent 'judge agent' "
        "--max-unhealthy-fraction 0.02 --policy-timeout-counts-as-model false --mlflow-run-id r-1 "
        "--mlflow-status FINISHED` now"
    )


def test_a_refused_extension_on_a_resumable_run_is_dropped_and_the_run_resumes(tmp_path, slurm_bin):
    job_dir = _timed_run(tmp_path)  # one rollout missing, budget left, no rollout store

    out = submit.inspect_recovery(*_handle(job_dir), extend_attempts=2)

    assert out.decision.action.value == "stop" and out.decision.reason.startswith("--extend-attempts 2 is refused")
    assert out.operator_verdict["action"] == (
        "drop --extend-attempts, which this run cannot take; the saved run continues without it (completed work is "
        f"kept): `{RUN}`"
    )


def test_an_unverified_run_by_construction_is_reported_with_its_export_never_resubmitted(tmp_path, fake_executor):
    fake_executor.job_dir = mlflow_job(tmp_path, mode="none")

    out = submit.recover("gym-job-r", "bench", score_key=["agent", "mean/reward"])

    assert (
        out.gate_task["status"] == "UNVERIFIED" and out.gate_task["unverified_cause"]["kind"] == "grading_unobserved"
    )
    assert out.gate_task["mlflow_delivery"]["state"] == "pending"
    verdict = out.operator_verdict
    assert verdict["state"] == "unknown"
    assert verdict["action"] == (
        "none: report the run as unverified with this reason and do not present it as verified; a resubmit cannot "
        "change it, and no flag does: the fix is the verifier's grading record, in Gym; "
        "report the MLflow delivery as not evidenced: the run's saved provenance does not say MLflow was configured "
        "(provenance.mlflow.configured), so a recovery does not export it and its results stand without an export; "
        "if MLflow was meant to be used, export them with the tracking tooling and inspect again with "
        "--mlflow-run-id and --mlflow-status"
    )
    assert "gym eval submit" not in verdict["action"]


def test_mlflow_delivery_unverified_by_configuration_is_reported_not_exported(tmp_path, fake_executor):
    job = mlflow_job(tmp_path, mode="postrun")
    receipt_path = job / "artifacts" / "rollouts_mlflow_export_done"
    receipt_path.write_text(json.dumps(json.loads(receipt_path.read_text()) | {"uploaded_rollouts": False}))
    fake_executor.job_dir = job

    out = submit.recover(
        "gym-job-r", "bench", score_key=["agent", "mean/reward"], mlflow_status="FINISHED", mlflow_run_id="postrun-run"
    )

    assert out.gate_task["mlflow_delivery"]["state"] == "unverified"
    assert out.operator_verdict["action"].endswith(
        "; report the MLflow delivery as unverified by configuration (upload_rollouts: false); an export does not "
        "change it"
    ), out.operator_verdict


def _collected_as(monkeypatch, fake_executor, collected):
    """The fake executor reads ``collected`` as the run's evidence; the rest of the recovery is the real one."""

    def recovery_evidence(self, record, benchmark, **options):
        fake_executor.calls.append("decide")
        decision = decide_from_collected(collected, max_infra_retries=3, max_walltime_s=None, live_job=False)
        return decision, collected

    monkeypatch.setattr(fake_executor, "recovery_evidence", recovery_evidence)


def test_failed_run_health_is_reported_without_a_fresh_submit(monkeypatch, fake_executor):
    collected = _run(97, settled={"model_transport_error": 3})
    collected.score = 0.9
    collected.quality["provenance"]["mlflow"]["tracking_required"] = False
    _collected_as(monkeypatch, fake_executor, collected)

    out = submit.recover("gym-job-h", "bench")

    assert out.decision.action == RecoveryAction.NOOP and out.gate_task["evaluation_health"]["status"] == "FAIL"
    assert out.operator_verdict == {
        "state": "needs_action",
        "reason": "3 of 100 trials failed on the evaluation side (model_transport_error 3)",
        "action": "none: report the run as failing run health; no recovery reopens settled trials, so check the "
        "evaluation-side faults evaluation_health names (their kinds and the services' logs)",
    }


def test_a_rejudged_run_without_a_grading_record_is_not_said_to_be_graded(monkeypatch, fake_executor):
    collected = _run(64)
    for row in collected.results[:59]:
        row["ng_trial_observations"]["verifier_invocations"][0]["grading_operations"] = None
    collected.failures = [
        {
            "_ng_task_index": task,
            "_ng_rollout_index": 0,
            "failure_kind": "judge_failed",
            "_ng_failure_http_status": 401,
            "_ng_failure_terminal": False,
            "mask_sample": True,
        }
        for task in range(59)
    ]
    collected.score = 0.9
    collected.quality["provenance"]["mlflow"]["tracking_required"] = False
    _collected_as(monkeypatch, fake_executor, collected)

    out = submit.recover("gym-job-j", "bench")

    assert out.gate_task["unverified_cause"]["kind"] == "judge_rejudged"
    assert out.operator_verdict["state"] == "unknown"
    assert out.operator_verdict["action"] == (
        "none: report the run as unverified: the rows were re-judged and no grading was recorded, so whether they "
        "were graded is unknown; no recovery command records it"
    )


def test_the_unsafe_continuation_keeps_the_extension_that_made_the_plan_possible(tmp_path, slurm_bin):
    job_dir = _spent_after(tmp_path, resume_script.RESUME_REFUSED_REASON)

    out = submit.inspect_recovery(*_handle(job_dir), extend_retries=1)

    assert out.decision.action.value == "resubmit", out.decision.reason
    assert out.decision.requested == ["--extend-retries 1"]
    action = out.operator_verdict["action"]
    assert "start a fresh run as the stop says" in action
    assert f"`{RUN} --extend-retries 1 --allow-unsafe-resume --unsafe-resume-reason TEXT`" in action, action


def test_a_failed_driver_install_names_its_fix_before_the_longer_walltime_restart(tmp_path, slurm_bin):
    job_dir = _cut_before_inventory(tmp_path, slurm_bin)
    (job_dir / resume_script.INCOMPLETE_MARKER_NAME).write_text(
        "Previous job 101 ended FAILED: driver_install_failed: (error text omitted); "
        "the chain does not retry it; see logs/driver-101.log\n"
    )

    out = submit.inspect_recovery(*_handle(job_dir))

    assert out.decision.action.value == "fresh-restart", out.decision.reason
    assert "driver_install_failed" in out.decision.reason
    action = out.operator_verdict["action"]
    assert action.index("fix the driver install") < action.index("gym eval resubmit"), action
    assert "--max-walltime HH:MM:SS --walltime-reason TEXT" in action


def _pending_export(configured, *, flagged=False):
    """A complete, healthy run whose required MLflow export has no receipt; ``configured`` is what its saved
    provenance says about the exporter (absent when None)."""
    collected = _run(100)
    collected.score = 0.9
    facts = {"tracking_required": True} | ({} if configured is None else {"configured": configured})
    collected.quality["provenance"]["mlflow"] = facts
    collected.mlflow = {"live_run_id": None, "receipt": None}
    if flagged:
        collected.quality["issues"]["rollout_ended_on_failed_model_call"] = 100
    return collected


@pytest.mark.parametrize("configured", [False, None])
def test_the_export_command_is_not_offered_when_recovery_would_not_export(monkeypatch, fake_executor, configured):
    _collected_as(monkeypatch, fake_executor, _pending_export(configured))
    exports = []
    monkeypatch.setattr(fake_executor, "export_tracking", lambda *a: exports.append(a), raising=False)

    out = submit.recover("gym-job-m", "bench")

    assert out.decision.action == RecoveryAction.NOOP and exports == []
    assert out.gate_task["mlflow_delivery"]["state"] == "pending"
    action = out.operator_verdict["action"]
    assert "gym eval resubmit" not in action and "runs the export" not in action, action
    assert "does not export" in action


def test_the_export_command_is_offered_only_when_recovery_reaches_the_export(monkeypatch, fake_executor):
    _collected_as(monkeypatch, fake_executor, _pending_export(True))
    exports = []
    monkeypatch.setattr(
        fake_executor,
        "export_tracking",
        lambda self, record, benchmark, collected: exports.append(benchmark) or {"state": "unsupported", "error": "x"},
        raising=False,
    )

    inspected = submit.inspect_recovery("gym-job-m", "bench")

    assert exports == [] and inspected.gate_task["mlflow_export_due"] is True
    assert (
        f"export the results to MLflow: `{RUN.replace('gym-job-x gsm8k', 'gym-job-m bench')}`"
        in (inspected.operator_verdict["action"])
    )
    # The same plan, applied, reaches the export.
    submit.recover("gym-job-m", "bench")
    assert exports == ["bench"]


@pytest.mark.parametrize("configured", [True, False])
def test_a_run_with_every_rollout_flagged_keeps_its_delivery_state_in_the_verdict(
    monkeypatch, fake_executor, configured
):
    _collected_as(monkeypatch, fake_executor, _pending_export(configured, flagged=True))

    out = submit.inspect_recovery("gym-job-m", "bench")

    verdict = out.operator_verdict
    assert verdict["state"] == "accept" and out.gate_task["status"] == "SUCCESS"
    assert verdict["reason"].startswith("accepted by contract; every rollout ended on a failed model call")
    assert "MLflow delivery pending" in verdict["reason"]
    assert verdict["action"].startswith("read rollout quality: 100 of 100 rollouts flagged")
    assert ("export the results to MLflow" in verdict["action"]) is configured
