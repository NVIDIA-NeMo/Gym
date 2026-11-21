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
"""`gym eval resubmit` decides from the run's own evidence before it queues anything.

The job dirs here are injected fixtures in the layout rollout collection and the job chain write; none is provider
evidence. Integration cases run the real ``COLLECT_SCRIPT`` as on a login node. Tests of quality arithmetic or
presentation supply explicit ``unit_quality`` inputs at the gate boundary instead of forging persisted reports.
"""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from nemo_gym.eval.recovery.decision import (
    RecoveryAction,
    RecoveryOutcome,
)
from nemo_gym.eval.recovery.evidence import (
    COLLECT_SCRIPT,
    CollectedEvidence,
)
from nemo_gym.eval.recovery.gate import (
    GateCriteria,
)
from nemo_gym.eval.recovery.planner import decide_from_collected
from nemo_gym.orchestration import submit
from nemo_gym.orchestration.executors.base import BaseExecutor
from nemo_gym.orchestration.executors.capabilities import Resubmittable
from tests.helpers.recovery_evidence import (
    HEALTH_UNKNOWN,
    INFRA,
    SAVED_ANSWER,
    STOP_PROVIDER,
    _skipped_job,
    complete_job,
    final_model_zero,
    ident,
    produce_quality,
    write_job_dir,
    write_postrun_receipt,
)


# The saved answer a judge failure keeps beside its row; re-verification re-judges it.


class FakeExecutor(Resubmittable):
    """Records every dispatch; ``supports_reverify`` stands for an executor with a saved-answer re-judgment path."""

    name = "fake"
    calls: list[str] = []
    # The options each saved-answer re-judgment was asked with (a judge partition, a repin), in call order.
    options: list[dict] = []
    job_dir: Path
    live = False
    supports_reverify = True

    walltime_plan = BaseExecutor.walltime_plan
    gym_pin = BaseExecutor.gym_pin

    def recovery_evidence(
        self,
        record,
        benchmark,
        *,
        score_key=None,
        score_agent=None,
        score_environment=None,
        outcome_policy=None,
        **options,
    ):
        FakeExecutor.calls.append("decide")
        out = subprocess.run(
            [
                sys.executable,
                "-c",
                COLLECT_SCRIPT,
                str(FakeExecutor.job_dir),
                *(
                    [
                        json.dumps(score_key),
                        json.dumps(
                            {
                                k: v
                                for k, v in (("agent", score_agent), ("environment", score_environment))
                                if v is not None
                            }
                        ),
                    ]
                    if score_key
                    else []
                ),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        collected = CollectedEvidence.model_validate_json(out.stdout)
        decision = decide_from_collected(
            collected,
            max_infra_retries=3,
            max_walltime_s=None,
            live_job=FakeExecutor.live,
            outcome_policy=outcome_policy,
        )
        return decision, collected

    def reverify_judge_failures(self, record, benchmark, decision, **options):
        FakeExecutor.options.append(options)
        if not FakeExecutor.supports_reverify:
            return BaseExecutor.reverify_judge_failures(self, record, benchmark, decision, **options)
        FakeExecutor.calls.append("reverify")
        return RecoveryOutcome(decision=decision, ok=True, dispatched="reverify")

    def resubmit(self, record, benchmark, *, keep_budget=False, accounted_walltime_s=None, accounting_snapshot=None):
        FakeExecutor.calls.append("resubmit" if keep_budget else "resubmit-resetting-budget")
        return type("Job", (), {"job_id": "4242"})()


@pytest.fixture
def fake_executor(monkeypatch):
    FakeExecutor.calls = []
    FakeExecutor.options = []
    FakeExecutor.live = False
    FakeExecutor.supports_reverify = True
    monkeypatch.setitem(submit._EXECUTORS_BY_NAME, "fake", FakeExecutor)
    monkeypatch.setattr(submit, "load_submission_record", lambda job_id: type("R", (), {"executor": "fake"})())
    return FakeExecutor


def test_recover_complete_run_is_a_successful_noop_with_zero_dispatch(tmp_path, fake_executor) -> None:
    fake_executor.job_dir = write_job_dir(
        tmp_path / "job", n=2, results=[ident(0, reward=1.0), ident(1, reward=0.0)], failures=[]
    )
    out = submit.recover("gym-job-a", "bench")
    assert out.ok and out.dispatched is None and out.decision.action == RecoveryAction.NOOP
    assert fake_executor.calls == ["decide"]


def test_recover_judge_only_rejudges_saved_answers_and_never_queues_the_policy(tmp_path, fake_executor) -> None:
    fake_executor.job_dir = write_job_dir(
        tmp_path / "job",
        n=2,
        results=[ident(0, reward=1.0), ident(1, _ng_failure_class="judge_failed", response=SAVED_ANSWER, reward=0.0)],
        failures=[],
    )
    out = submit.recover("gym-job-b", "bench")
    assert out.ok and out.dispatched == "reverify"
    assert fake_executor.calls == ["decide", "reverify"]


def test_recover_reports_an_executor_without_saved_answer_rejudgment_as_unsupported(tmp_path, fake_executor) -> None:
    fake_executor.supports_reverify = False
    fake_executor.job_dir = write_job_dir(
        tmp_path / "job",
        n=1,
        results=[ident(0, _ng_failure_class="judge_failed", response=SAVED_ANSWER, reward=0.0)],
        failures=[],
    )
    out = submit.recover("gym-job-b2", "bench")
    assert not out.ok and out.dispatched is None and "gym eval reverify --judge-failed-only" in out.unsupported
    assert fake_executor.calls == ["decide"]


def test_recover_keeps_a_terminal_model_zero_done(tmp_path, fake_executor) -> None:
    final = ident(
        1,
        failure_kind="model_context_limit",
        _ng_failure_terminal=True,
        _ng_failure_counts_as_zero=True,
        _ng_failure_id="e",
        _ng_failure_events={
            "e": {
                "boundary": "model",
                "operation": "chat_completions",
                "operation_id": "op",
                "attempt_id": "a1",
                "cause_id": None,
            }
        },
    )
    fake_executor.job_dir = write_job_dir(tmp_path / "job", n=2, results=[ident(0, reward=1.0)], failures=[final])
    out = submit.recover("gym-job-c", "bench")
    assert out.ok and out.decision.action == RecoveryAction.NOOP and fake_executor.calls == ["decide"]


@pytest.mark.parametrize(
    "contradiction",
    [
        dict(failure_kind="model_context_limit", _ng_failure_terminal=False),
        dict(failure_kind="model_transport_error"),
    ],
)
def test_counts_as_zero_without_terminal_model_ownership_is_not_final(tmp_path, fake_executor, contradiction) -> None:
    row = ident(1, _ng_failure_counts_as_zero=True, **contradiction)
    fake_executor.job_dir = write_job_dir(tmp_path / "job", n=2, results=[ident(0, reward=1.0)], failures=[row])
    out = submit.recover("gym-job-c2", "bench")
    assert out.decision.identities.get("model_final") is None
    assert not out.ok and out.decision.action == RecoveryAction.STOP and fake_executor.calls == ["decide"]


@pytest.mark.parametrize("kind", ["model_context_limit", "model_transport_error", "model_policy_response_timeout"])
def test_a_terminal_counts_as_zero_row_is_settled_as_rollout_collection_settles_it(
    tmp_path, fake_executor, kind
) -> None:
    # Explicit finality and model ownership are both required; the companion negative controls reject other owners.
    row = ident(1, _ng_failure_counts_as_zero=True, _ng_failure_terminal=True, failure_kind=kind)
    fake_executor.job_dir = write_job_dir(tmp_path / "job", n=2, results=[ident(0, reward=1.0)], failures=[row])
    out = submit.recover("gym-job-c3", "bench")
    assert out.decision.identities["model_final"] == [(1, 0)] and fake_executor.calls == ["decide"]


def test_recover_queues_missing_and_retryable_work_once_through_the_existing_resubmit(tmp_path, fake_executor) -> None:
    fake_executor.job_dir = write_job_dir(
        tmp_path / "job", n=3, results=[ident(0, reward=1.0)], failures=[ident(1, **INFRA)], retries=1
    )
    out = submit.recover("gym-job-d", "bench")
    assert out.ok and out.dispatched == "resubmit" and out.job_id == "4242"
    assert fake_executor.calls == ["decide", "resubmit"]


def test_recover_with_judge_failures_and_missing_work_resubmits_with_the_saved_answers_reverified_first(
    tmp_path, fake_executor
) -> None:
    fake_executor.job_dir = write_job_dir(
        tmp_path / "job",
        n=3,
        results=[ident(0, reward=1.0), ident(1, _ng_failure_class="judge_failed", response=SAVED_ANSWER)],
        failures=[],
        retries=0,
    )
    out = submit.recover("gym-job-d2", "bench")
    # The resumed run re-verifies the saved answer first, while its servers are up; only the missing work is run.
    assert out.dispatched == "resubmit" and out.decision.action == RecoveryAction.RESUBMIT
    assert fake_executor.calls == ["decide", "resubmit"]
    assert out.decision.identities["judge_recoverable"] == [(1, 0)] and out.decision.identities["missing"] == [(2, 0)]
    assert out.decision.reverify_first == [(1, 0)]
    assert out.reverify.startswith(f"1 saved answer(s) are re-verified by resume job {out.job_id} first")


def test_recover_does_not_queue_beside_a_live_job(tmp_path, fake_executor) -> None:
    fake_executor.job_dir = write_job_dir(tmp_path / "job", n=2, results=[ident(0, reward=1.0)], failures=[])
    fake_executor.live = True
    out = submit.recover("gym-job-e", "bench")
    # Nothing to decide while a job of the run waits or runs: no error, and nothing is queued beside it.
    assert out.ok and out.dispatched is None and "still pending or running" in out.decision.reason
    assert out.decision.action == RecoveryAction.NOOP and fake_executor.calls == ["decide"]


def test_recover_stops_on_unknown_ownership(tmp_path, fake_executor) -> None:
    fake_executor.job_dir = write_job_dir(
        tmp_path / "job",
        n=2,
        results=[ident(0, reward=1.0)],
        failures=[ident(1, failure_kind="whatever happened")],
        retries=0,
    )
    out = submit.recover("gym-job-f", "bench")
    assert not out.ok and out.decision.action == RecoveryAction.STOP and fake_executor.calls == ["decide"]


def test_plain_resubmit_is_unchanged_and_does_not_decide(tmp_path, fake_executor) -> None:
    assert submit.resubmit("gym-job-g", "bench").job_id == "4242"
    assert fake_executor.calls == ["resubmit-resetting-budget"]


def test_final_model_zeros_stay_done_and_are_not_dispatched_again(tmp_path, fake_executor) -> None:
    results = [ident(t, reward=1.0) for t in range(4)]
    finals = [final_model_zero(4, 0), final_model_zero(5, 0), final_model_zero(6, 0)]
    fake_executor.job_dir = write_job_dir(tmp_path / "job", n=7, results=results, failures=finals)
    out = submit.recover("gym-job-h", "bench")
    assert out.ok and out.decision.action == RecoveryAction.NOOP and fake_executor.calls == ["decide"]
    assert out.decision.identities["model_final"] == [(4, 0), (5, 0), (6, 0)]
    assert out.gate_task["scored_samples"] == out.gate_task["expected_samples"] == 7
    # Unknown health is a warning, never missing work; the unread score fails the gate.
    assert out.gate_task["status"] == "FAILED" and any("unknown" in w for w in out.gate_task["warnings"])
    assert "missing" not in out.decision.counts
    assert (
        out.gate_task["errors"][0]
        == "score unavailable: no requested metric was read (pass --score-key, and --score-agent or --score-environment for a Gym aggregate)"
    )


@pytest.mark.parametrize(
    "row",
    [
        dict(_ng_failure_class="judge_failed", response=SAVED_ANSWER),
        dict(failure_kind="model_transport_error", _ng_failure_terminal=False, _ng_failure_counts_as_zero=False),
    ],
)
def test_judge_or_nonfinal_rows_are_never_relabelled_model_zeros(tmp_path, fake_executor, row) -> None:
    final = final_model_zero(1, 0)
    other = {"_ng_task_index": 2, "_ng_rollout_index": 0, **row}
    results = [ident(0, reward=1.0)] + ([other] if "_ng_failure_class" in row else [])
    failures = [final] + ([] if "_ng_failure_class" in row else [other])
    fake_executor.job_dir = write_job_dir(tmp_path / "job", n=3, results=results, failures=failures, retries=0)
    out = submit.recover("gym-job-i", "bench")
    assert out.decision.identities["model_final"] == [(1, 0)]


def run_cli(argv: list[str], capsys) -> tuple[int, dict]:
    from nemo_gym.cli.main import main

    code, saved = 0, sys.argv
    sys.argv = ["gym", *argv]
    try:
        main()
    except SystemExit as exc:
        code = exc.code or 0
    finally:
        sys.argv = saved
    return code, json.loads(capsys.readouterr().out.strip().splitlines()[-1])


def test_cli_recover_emits_the_gate_task_and_dispatches_nothing_for_a_complete_run(
    tmp_path, fake_executor, capsys
) -> None:
    job = write_job_dir(tmp_path / "job", n=2, results=[ident(0, reward=1.0), ident(1, reward=0.0)], failures=[])
    (job / "artifacts" / "rollouts_aggregate_metrics.json").write_text(json.dumps({"agent": {"mean/reward": 0.5}}))
    produce_quality(job)
    write_postrun_receipt(job)
    fake_executor.job_dir = job
    code, out = run_cli(
        [
            "eval",
            "resubmit",
            "gym-job-j",
            "bench",
            "--recover",
            "--score-key",
            '["agent", "mean/reward"]',
            "--max-unhealthy-fraction",
            "0.02",
            "--mlflow-run-id",
            "postrun-run",
            "--mlflow-status",
            "FINISHED",
        ],
        capsys,
    )
    # UNVERIFIED is not accepted: the requested gate still exits non-zero.
    assert code == 1 and out["dispatched"] is None and fake_executor.calls == ["decide"]
    assert out["gate_task"]["status"] == "UNVERIFIED" and out["gate_task"]["score"] == 0.5
    assert out["gate_task"]["errors"] == [] and out["gate_task"]["warnings"][0] == HEALTH_UNKNOWN
    assert out["gate_task"]["warnings"][1].startswith("rollout quality not judged against the configured criterion")
    assert "2 unobserved" in out["gate_task"]["warnings"][1]


def test_cli_recover_without_score_or_criterion_fails_closed(tmp_path, fake_executor, capsys) -> None:
    fake_executor.job_dir = write_job_dir(tmp_path / "job", n=1, results=[ident(0, reward=1.0)], failures=[])
    code, out = run_cli(["eval", "resubmit", "gym-job-k", "bench", "--recover"], capsys)
    assert code == 0 and out["gate_task"]["score"] is None and out["gate_task"]["status"] == "FAILED"


def test_cli_recover_queues_missing_work_through_the_existing_resubmit(tmp_path, fake_executor, capsys) -> None:
    fake_executor.job_dir = write_job_dir(
        tmp_path / "job", n=2, results=[ident(0, reward=1.0)], failures=[], retries=0
    )
    code, out = run_cli(["eval", "resubmit", "gym-job-l", "bench", "--recover"], capsys)
    assert code == 0 and out["dispatched"] == "resubmit" and fake_executor.calls == ["decide", "resubmit"]
    assert out["gate_task"]["status"] == "RESUMING"


def test_inspect_reports_a_planned_resubmit_as_planned_not_dispatched(tmp_path, fake_executor) -> None:
    fake_executor.job_dir = write_job_dir(
        tmp_path / "job", n=2, results=[ident(0, reward=1.0)], failures=[], retries=0
    )
    out = submit.inspect_recovery("gym-job-p", "bench")
    assert out.decision.action == RecoveryAction.RESUBMIT and out.dispatched is None
    assert fake_executor.calls == ["decide"]
    assert out.gate_task["status"] == "PLANNED"
    assert out.gate_task["warnings"][-1] == (
        "PLANNED (NOT DISPATCHED): --inspect reports the resubmit decision and runs nothing"
    )


def test_inspect_reports_a_planned_rejudgment_as_planned_and_recovery_dispatches_it(tmp_path, fake_executor) -> None:
    fake_executor.job_dir = write_job_dir(
        tmp_path / "job",
        n=2,
        results=[ident(0, reward=1.0), ident(1, _ng_failure_class="judge_failed", response=SAVED_ANSWER, reward=0.0)],
        failures=[],
    )
    inspected = submit.inspect_recovery("gym-job-q", "bench")
    assert inspected.decision.action == RecoveryAction.REVERIFY and inspected.dispatched is None
    assert inspected.gate_task["status"] == "PLANNED"
    assert "PLANNED (NOT DISPATCHED): --inspect reports the reverify decision" in inspected.gate_task["warnings"][-1]

    recovered = submit.recover("gym-job-q", "bench")
    assert recovered.dispatched == "reverify" and recovered.gate_task["status"] == "RESUMING"
    assert not any("NOT DISPATCHED" in warning for warning in recovered.gate_task["warnings"])
    assert fake_executor.calls == ["decide", "decide", "reverify"]


def test_cli_score_flags_need_recover(capsys, fake_executor) -> None:
    with pytest.raises(SystemExit):
        from nemo_gym.cli.main import main

        saved, sys.argv = (
            sys.argv,
            ["gym", "eval", "resubmit", "gym-job-m", "bench", "--max-unhealthy-fraction", "0.02"],
        )
        try:
            main()
        finally:
            sys.argv = saved
    assert fake_executor.calls == []


def test_live_scheduler_job_is_running_not_success_or_failure(tmp_path, fake_executor) -> None:
    fake_executor.job_dir = write_job_dir(
        tmp_path / "job", n=2, results=[ident(0, reward=1.0), ident(1, reward=1.0)], failures=[]
    )
    fake_executor.live = True
    out = submit.recover("gym-job-n", "bench")
    assert out.gate_task["status"] == "RUNNING" and fake_executor.calls == ["decide"]
    assert out.ok and out.dispatched is None


def test_a_live_mlflow_run_is_unverified_delivery_not_an_unfinished_export(tmp_path, fake_executor) -> None:
    fake_executor.job_dir = mlflow_job(tmp_path, mode="live")
    out = submit.recover(
        "gym-job-o", "bench", score_key=["agent", "mean/reward"], max_unhealthy_fraction=0.02, mlflow_status="FINISHED"
    )
    assert out.decision.action == RecoveryAction.NOOP and out.dispatched is None
    # Delivery is a fact beside the verdict: it never turns the gate's status into FAILED.
    assert not [e for e in out.gate_task["errors"] if "MLflow" in e] and out.gate_task["status"] != "FAILED"
    assert out.gate_task["mlflow_delivery"] == {
        "state": "unverified",
        "reasons": [
            "MLflow live run live-run: configured payload delivery is not verified (live uploads log failures and "
            "continue), so live acceptance is unsupported"
        ],
    }


def test_omitted_rollout_rows_are_unsupported_not_scored_or_retried(tmp_path, fake_executor) -> None:
    fake_executor.job_dir = write_job_dir(
        tmp_path / "job",
        n=2,
        results=[ident(0, reward=1.0), ident(1, reward=0.0, _ng_omitted_rollout=True)],
        failures=[],
        retries=0,
    )
    out = submit.recover("gym-job-p", "bench")
    assert out.decision.action == RecoveryAction.STOP and out.decision.identities["unknown"] == [(1, 0)]
    assert fake_executor.calls == ["decide"]


def test_stopped_incomplete_chain_is_named_in_the_gate_errors(tmp_path, fake_executor) -> None:
    job = write_job_dir(tmp_path / "job", n=2, results=[ident(0, reward=1.0)], failures=[], retries=3)
    (job / ".gym_chain_incomplete").write_text("")
    fake_executor.job_dir = job
    out = submit.recover("gym-job-q", "bench")
    assert out.decision.action == RecoveryAction.STOP and fake_executor.calls == ["decide"]
    assert any(
        "the job chain stopped with 1 of 2 rollouts unsettled (1 missing)" in e for e in out.gate_task["errors"]
    )


def mlflow_job(tmp_path: Path, *, mode: str) -> Path:
    """A finished run whose MLflow files are the ones each producer leaves (or a damaged version of them)."""
    job = write_job_dir(tmp_path / "job", n=1, results=[ident(0, reward=1.0)], failures=[])
    produce_quality(job)
    (job / "artifacts" / "rollouts_aggregate_metrics.json").write_text(json.dumps({"agent": {"mean/reward": 1.0}}))
    if mode in ("live", "both"):
        # MLflowExporter.setup records the run it opened; teardown sets its status and writes nothing else.
        (job / "artifacts" / "rollouts_mlflow_run_id").write_text("live-run")
    if mode in ("postrun", "both", "stale"):
        write_postrun_receipt(job)
    if mode == "stale":
        (job / "artifacts" / "rollouts.jsonl").write_text(json.dumps(ident(0, reward=0.0)) + "\n")
    if mode in ("version-true", "blank-run-id"):
        write_postrun_receipt(job, run_id=" " if mode == "blank-run-id" else "postrun-run")
        receipt_path = job / "artifacts" / "rollouts_mlflow_export_done"
        receipt = json.loads(receipt_path.read_text())
        if mode == "version-true":
            receipt["version"] = True
        receipt_path.write_text(json.dumps(receipt))
    if mode == "run-id-marker":
        (job / "artifacts" / "rollouts_mlflow_export_done").write_text("postrun-run")
    if mode == "empty-marker":
        (job / "artifacts" / "rollouts_mlflow_export_done").write_text("")
    return job


@pytest.mark.parametrize(
    ("mode", "status", "mlflow"),
    [
        ("postrun", "FINISHED", []),
        ("postrun", "FAILED", ["MLflow run status FAILED"]),
        ("postrun", None, ["status not queried"]),
        ("postrun", "RUNNING", ["MLflow run status RUNNING"]),
        (
            "stale",
            "FINISHED",
            ["MLflow post-run export receipt of run postrun-run does not match the current artifacts"],
        ),
        ("empty-marker", "FINISHED", ["MLflow post-run export receipt is malformed or not a version-1 receipt"]),
        ("version-true", "FINISHED", ["MLflow post-run export receipt is malformed or not a version-1 receipt"]),
        ("blank-run-id", "FINISHED", ["MLflow post-run export receipt is malformed or not a version-1 receipt"]),
        (
            "run-id-marker",
            "FINISHED",
            [
                "MLflow post-run export marker of run postrun-run has no artifact digests, so the exported content "
                "is not verified"
            ],
        ),
        # A current receipt with its own run observed is bound, whatever earlier live run left its id behind.
        ("both", "FINISHED", []),
        ("none", None, ["no MLflow run evidence: no live run id and no post-run export receipt"]),
    ],
)
def test_mlflow_delivery_follows_the_producer_of_its_evidence(tmp_path, fake_executor, mode, status, mlflow) -> None:
    fake_executor.job_dir = mlflow_job(tmp_path, mode=mode)
    out = submit.recover(
        "gym-job-r",
        "bench",
        score_key=["agent", "mean/reward"],
        max_unhealthy_fraction=0.02,
        mlflow_status=status,
        mlflow_run_id="postrun-run",
    )
    assert out.dispatched is None and fake_executor.calls == ["decide"]
    # Grade-only rows have unobserved response/trial health. Mutating a row also invalidates both content bindings.
    health_unknown = [HEALTH_UNKNOWN]
    if mode == "stale":
        health_unknown = [
            'summary validity unknown: ["quality summary is not bound to this run\'s sole rollouts.jsonl source"]',
            HEALTH_UNKNOWN,
        ]
    # Unknown health alone is UNVERIFIED; MLflow delivery is a separate fact and never changes the status.
    assert out.gate_task["status"] == "UNVERIFIED" and out.gate_task["errors"] == []
    state = (
        "verified" if not mlflow else "pending" if mode == "none" or mlflow == ["status not queried"] else "unverified"
    )
    assert out.gate_task["mlflow_delivery"] == {"state": state, "reasons": mlflow}
    assert (state in ("verified", "pending", "unverified")) and (
        any(w.startswith("MLflow delivery ") for w in out.gate_task["warnings"]) == (state != "verified")
    )
    assert out.gate_task["warnings"][: len(health_unknown)] == health_unknown


def test_mlflow_status_for_a_run_without_an_mlflow_id_is_contradictory(tmp_path, fake_executor) -> None:
    fake_executor.job_dir = mlflow_job(tmp_path, mode="none")
    out = submit.recover(
        "gym-job-s", "bench", score_key=["agent", "mean/reward"], max_unhealthy_fraction=0.02, mlflow_status="FINISHED"
    )
    delivery = out.gate_task["mlflow_delivery"]
    assert out.gate_task["status"] != "FAILED" and not [e for e in out.gate_task["errors"] if "MLflow" in e]
    assert delivery["state"] == "unverified" and any("no MLflow run id" in r for r in delivery["reasons"])


def test_cli_passes_the_observed_mlflow_status(tmp_path, fake_executor, capsys) -> None:
    fake_executor.job_dir = mlflow_job(tmp_path, mode="postrun")
    code, out = run_cli(
        [
            "eval",
            "resubmit",
            "gym-job-t",
            "bench",
            "--recover",
            "--score-key",
            '["agent", "mean/reward"]',
            "--max-unhealthy-fraction",
            "0.02",
            "--mlflow-status",
            "FAILED",
        ],
        capsys,
    )
    # The observed status is a delivery fact; the gate is still not accepted (health is unknown), but not FAILED.
    assert code == 1 and out["mlflow"]["observed_status"] == "FAILED" and out["gate_task"]["status"] == "UNVERIFIED"
    delivery = out["gate_task"]["mlflow_delivery"]
    assert delivery["state"] == "unverified" and "MLflow run status FAILED" in delivery["reasons"]


def judge_failed_sidecar(task: int) -> dict:
    return {
        "_ng_task_index": task,
        "_ng_rollout_index": 0,
        "_ng_failure_class": "judge_failed",
        "failure_kind": "judge_failed",
        "response": SAVED_ANSWER,
        "_ng_failure_terminal": False,
        "_ng_failure_max_attempts": 3,
    }


def unstarted(task: int) -> dict:
    return {"_ng_task_index": task, "_ng_rollout_index": 0, "_ng_omitted_rollout": True, "_ng_dispatch_drained": True}


def unrouted(task: int) -> dict:
    return {
        "_ng_task_index": task,
        "_ng_rollout_index": 0,
        "_ng_omitted_rollout": True,
        "_ng_unrouted_run_error": True,
    }


def interrupted(task: int) -> dict:
    return {
        "_ng_task_index": task,
        "_ng_rollout_index": 0,
        "_ng_omitted_rollout": True,
        "_ng_failure_class": "kill_shaped",
    }


def test_sidecar_judge_failure_is_reverified_never_resubmitted(tmp_path, fake_executor) -> None:
    fake_executor.supports_reverify = False
    fake_executor.job_dir = write_job_dir(
        tmp_path / "job", n=2, results=[ident(0, reward=1.0)], failures=[judge_failed_sidecar(1)], retries=0
    )
    out = submit.recover("gym-job-u", "bench")
    assert out.decision.action == RecoveryAction.REVERIFY and out.decision.identities["judge_recoverable"] == [(1, 0)]
    assert fake_executor.calls == ["decide"] and out.dispatched is None and out.unsupported


def test_a_final_model_zero_before_the_resubmit_offset_stays_done(tmp_path, fake_executor) -> None:
    fake_executor.job_dir = write_job_dir(
        tmp_path / "job", n=2, results=[ident(0, reward=1.0)], failures=[final_model_zero(1, 0)], offset=1
    )
    out = submit.recover("gym-job-v", "bench")
    assert out.decision.identities["model_final"] == [(1, 0)] and out.decision.action == RecoveryAction.NOOP
    assert fake_executor.calls == ["decide"]


@pytest.mark.parametrize("later", [unstarted, unrouted])
def test_a_later_non_superseding_row_does_not_erase_a_final_model_zero(tmp_path, fake_executor, later) -> None:
    fake_executor.job_dir = write_job_dir(
        tmp_path / "job", n=2, results=[ident(0, reward=1.0)], failures=[final_model_zero(1, 0), later(1)]
    )
    out = submit.recover("gym-job-w", "bench")
    assert out.decision.identities["model_final"] == [(1, 0)] and out.decision.action == RecoveryAction.NOOP


def test_a_later_interrupted_row_restarts_the_input(tmp_path, fake_executor) -> None:
    fake_executor.job_dir = write_job_dir(
        tmp_path / "job", n=2, results=[ident(0, reward=1.0)], failures=[ident(1, **INFRA), interrupted(1)], retries=0
    )
    out = submit.recover("gym-job-x", "bench")
    assert out.decision.identities["missing"] == [(1, 0)] and fake_executor.calls == ["decide", "resubmit"]


def test_attempts_before_the_resubmit_offset_do_not_count_towards_the_cap(tmp_path, fake_executor) -> None:
    rows = [ident(1, _ng_attempt_index=a, **INFRA) for a in range(4)]
    fake_executor.job_dir = write_job_dir(
        tmp_path / "job", n=2, results=[ident(0, reward=1.0)], failures=rows, offset=3, retries=0
    )
    out = submit.recover("gym-job-y", "bench")
    item = [e for e in out.decision.evidence if e.identity == (1, 0)][0]
    assert (item.state, item.attempts) == ("retryable", 1)


def test_an_omitted_agent_error_without_interruption_class_stops(tmp_path, fake_executor) -> None:
    # An omitted adapter outcome is not an interruption merely because it carries the omission flag.
    row = interrupted(1) | {
        "_ng_failure_class": "agent_run_error",
        "_ng_failure_id": "f",
        "_ng_failure_events": {"e": {"boundary": "agent", "operation": "run"}},
    }
    fake_executor.job_dir = write_job_dir(
        tmp_path / "job", n=2, results=[ident(0, reward=1.0)], failures=[row], retries=0
    )
    out = submit.recover("gym-job-z", "bench")
    assert out.decision.identities["unknown"] == [(1, 0)] and "missing" not in out.decision.identities
    assert out.decision.action == RecoveryAction.STOP and fake_executor.calls == ["decide"]


def aggregate_list_job(tmp_path: Path) -> Path:
    """A finished run whose aggregate metrics file has the producer's shape: a list of per-agent entries."""
    job = complete_job(tmp_path)
    entries = [{"agent_ref": {"name": "agent"}, "agent_metrics": {"mean/reward": 0.75}, "key_metrics": {}}]
    (job / "artifacts" / "rollouts_aggregate_metrics.json").write_text(json.dumps(entries))
    produce_quality(job)
    write_postrun_receipt(job)
    return job


@pytest.mark.parametrize(
    ("score_key", "score"),
    [
        ('[0, "agent_metrics", "mean/reward"]', None),
        ('[1, "agent_metrics", "mean/reward"]', None),
        ('["0", "agent_metrics", "mean/reward"]', None),
        ('[0, "agent_metrics", 0]', None),
    ],
)
def test_cli_recover_reads_a_score_path_into_the_aggregate_list(tmp_path, fake_executor, capsys, score_key, score):
    fake_executor.job_dir = aggregate_list_job(tmp_path)
    code, out = run_cli(
        [
            "eval",
            "resubmit",
            "gym-job-k",
            "bench",
            "--recover",
            "--score-key",
            score_key,
            "--max-unhealthy-fraction",
            "0.02",
            "--mlflow-status",
            "FINISHED",
            "--mlflow-run-id",
            "postrun-run",
        ],
        capsys,
    )
    assert code == 1 and out["dispatched"] is None and fake_executor.calls == ["decide"]
    assert out["gate_task"]["score"] == score and out["gate_task"]["status"] == "FAILED"
    if score is None:
        assert (
            "score unavailable: no requested metric was read (pass --score-key, and --score-agent or --score-environment for a Gym aggregate)"
            in out["gate_task"]["errors"]
        )


def test_cli_recover_still_reads_a_string_key_path(tmp_path, fake_executor, capsys) -> None:
    fake_executor.job_dir = complete_job(tmp_path)
    code, out = run_cli(
        ["eval", "resubmit", "gym-job-k", "bench", "--recover", "--score-key", '["agent", "mean/reward"]'], capsys
    )
    assert code == 1 and out["gate_task"]["score"] == 0.75 and fake_executor.calls == ["decide"]


@pytest.mark.parametrize("score_key", ["[true, 0]", '[-1, "agent_metrics"]', "[0.5]", "[]", '{"a": 0}'])
def test_cli_recover_refuses_a_malformed_score_path_before_reading_anything(fake_executor, score_key) -> None:
    from nemo_gym.cli.main import main

    saved = sys.argv
    sys.argv = ["gym", "eval", "resubmit", "gym-job-k", "bench", "--recover", "--score-key", score_key]
    try:
        with pytest.raises(SystemExit) as exc:
            main()
    finally:
        sys.argv = saved
    assert exc.value.code == 2 and fake_executor.calls == []


# These controls execute the published collector and dispatch boundary on synthetic saved directories.
def test_review_mixed_judge_sidecar_never_replays_the_answer(tmp_path, fake_executor):
    fake_executor.job_dir = write_job_dir(
        tmp_path / "job",
        n=2,
        results=[],
        failures=[ident(0, _ng_failure_class="judge_failed", response=SAVED_ANSWER, failure_kind="judge_failed")],
    )
    out = submit.recover("review", "bench")
    # The resume runs only the missing rollout: the saved answer is re-verified first, never replayed.
    assert out.decision.action == RecoveryAction.RESUBMIT and out.decision.reverify_first == [(0, 0)]
    assert fake_executor.calls == ["decide", "resubmit"]
    assert out.decision.identities["judge_recoverable"] == [(0, 0)]


@pytest.mark.parametrize("fraction", [float("nan"), float("inf"), True, -0.1, 1.1])
def test_review_invalid_criterion_cannot_dispatch(tmp_path, fake_executor, fraction):
    fake_executor.job_dir = write_job_dir(tmp_path / "job", n=1, results=[], failures=[])
    with pytest.raises(ValueError, match="criterion"):
        submit.recover("review", "bench", max_unhealthy_fraction=fraction)
    assert fake_executor.calls == []


def test_review_unsupported_reverify_reports_failed_not_resuming(tmp_path, fake_executor):
    fake_executor.job_dir = write_job_dir(
        tmp_path / "job", n=1, results=[ident(0, _ng_failure_class="judge_failed", response=SAVED_ANSWER)], failures=[]
    )
    fake_executor.supports_reverify = False
    out = submit.recover("review", "bench")
    assert out.dispatched is None and out.gate_task["status"] == "FAILED"


@pytest.mark.parametrize("order", [False, True])
@pytest.mark.parametrize(
    "selectors,expected",
    [
        (["--score-agent", "first"], 0.0),
        (["--score-environment", "env-b"], 0.75),
        (["--score-agent", "first", "--score-environment", "env-a"], 0.0),
        (["--score-agent", "absent"], None),
    ],
)
def test_review_cli_selects_stable_aggregate_identity(tmp_path, fake_executor, capsys, order, selectors, expected):
    fake_executor.job_dir = write_job_dir(
        tmp_path / "job", n=2, results=[ident(0, reward=0.0), ident(1, reward=1.0)], failures=[]
    )
    rows = [
        {"agent_ref": {"name": "first"}, "_ng_environment_server": "env-a", "agent_metrics": {"mean/reward": 0.0}},
        {"agent_ref": {"name": "second"}, "_ng_environment_server": "env-b", "agent_metrics": {"mean/reward": 0.75}},
    ]
    path = fake_executor.job_dir / "artifacts" / "rollouts_aggregate_metrics.json"
    path.write_text(json.dumps(list(reversed(rows)) if order else rows))
    before = path.read_bytes()
    code, out = run_cli(
        [
            "eval",
            "resubmit",
            "review",
            "bench",
            "--recover",
            *selectors,
            "--score-key",
            '["agent_metrics", "mean/reward"]',
        ],
        capsys,
    )
    assert code == 1 and out["gate_task"]["score"] == expected
    assert out["dispatched"] is None and fake_executor.calls == ["decide"]
    assert path.read_bytes() == before


@pytest.mark.parametrize("metric", [None, True, float("nan"), float("inf")])
def test_review_identity_does_not_make_invalid_metrics_valid(tmp_path, fake_executor, metric):
    fake_executor.job_dir = write_job_dir(tmp_path / "job", n=1, results=[ident(0, reward=0.0)], failures=[])
    (fake_executor.job_dir / "artifacts" / "rollouts_aggregate_metrics.json").write_text(
        json.dumps([{"agent_ref": {"name": "a"}, "agent_metrics": {"mean/reward": metric}}])
    )
    out = submit.recover("review", "bench", score_agent="a", score_key=["agent_metrics", "mean/reward"])
    assert out.gate_task["score"] is None and out.dispatched is None


def test_review_duplicate_agent_needs_environment_and_duplicate_pair_rejects(tmp_path, fake_executor):
    fake_executor.job_dir = write_job_dir(tmp_path / "job", n=1, results=[ident(0, reward=0.0)], failures=[])
    path = fake_executor.job_dir / "artifacts" / "rollouts_aggregate_metrics.json"
    rows = [
        {"agent_ref": {"name": "a"}, "_ng_environment_server": env, "agent_metrics": {"mean/reward": score}}
        for env, score in (("x", 0.0), ("y", 1.0))
    ]
    path.write_text(json.dumps(rows))
    kwargs = {"score_agent": "a", "score_key": ["agent_metrics", "mean/reward"]}
    assert submit.recover("review", "bench", **kwargs).gate_task["score"] is None
    assert submit.recover("review", "bench", score_environment="x", **kwargs).gate_task["score"] == 0.0
    path.write_text(json.dumps(rows + [rows[0]]))
    assert submit.recover("review", "bench", score_environment="x", **kwargs).gate_task["score"] is None


STRICT_DEFAULT = {
    "response": {"max_failure_fraction": 0.02, "passes": "strictly below, before rounding"},
    "trial": {"max_failure_fraction": 0.02, "passes": "strictly below, before rounding"},
}


@pytest.mark.parametrize("value", [True, 0, -0.01, 1.5, float("nan"), float("inf"), "0.02"])
@pytest.mark.parametrize("name", ["max_response_failure_fraction", "max_trial_failure_fraction"])
def test_review_invalid_gate_criteria_fail_closed(tmp_path, fake_executor, name, value):
    with pytest.raises(ValueError):
        GateCriteria(**{name: value})
    fake_executor.job_dir = write_job_dir(tmp_path / "job", n=1, results=[ident(0, reward=1.0)], failures=[])
    with pytest.raises(ValueError, match="invalid gate criteria"):
        submit.recover("gym-job-criteria", "bench", **{name: value})
    assert fake_executor.calls == []


def test_review_recover_passes_the_outcome_policy_to_the_decision(tmp_path, fake_executor):
    fake_executor.job_dir = write_job_dir(
        tmp_path / "job", n=2, results=[ident(0, reward=1.0)], failures=[ident(1, **INFRA)]
    )
    out = submit.recover("gym-job-policy", "bench", outcome_policy=STOP_PROVIDER)
    assert out.decision.identities["terminal"] == [(1, 0)] and out.decision.action == RecoveryAction.STOP
    assert out.dispatched is None


# A rollout store's attempt manifest as its producer writes it (selected fields; the guard reads none of them).
MANAGED_MANIFEST = {
    "schema_version": 2,
    "selection_policy": "latest_allocated",
    "next_attempt": {"0-0": 2},
    "max_rollout_attempts": 3,
    "retry_terminal_timeouts": False,
    "run_id": "controlled-managed-run",
    "source_digest": "0" * 64,
    "materialized_digest": "1" * 64,
    "config_digest": "2" * 64,
    "legacy_import": False,
    "identity_overridden": False,
}


@pytest.mark.parametrize(
    "marker", ["manifest", "empty_manifest", "malformed_manifest", "journal_only", "tagged_result", "tagged_failure"]
)
def test_a_rollout_store_run_stops_instead_of_selecting_a_saved_attempt(tmp_path, fake_executor, marker) -> None:
    """An earlier saved success must not make a store-managed run look complete: only the store selects attempts."""
    tagged = {"_ng_run_id": "controlled-managed-run"}
    results = [ident(0, reward=1.0, _ng_attempt_index=0, **(tagged if marker == "tagged_result" else {}))]
    failures = (
        [ident(0, _ng_attempt_index=1, failure_kind="provider_unavailable", **tagged)]
        if marker == "tagged_failure"
        else []
    )
    job = write_job_dir(tmp_path / "job", n=1, results=results, failures=failures)
    manifest = job / "artifacts" / "rollouts_manifest.json"
    if marker == "manifest":
        manifest.write_text(json.dumps(MANAGED_MANIFEST))
    elif marker == "empty_manifest":
        manifest.write_text("")
    elif marker == "malformed_manifest":
        manifest.write_text("{not json")
    elif marker == "journal_only":
        (job / "artifacts" / "rollouts_attempts.jsonl").write_text('{"reserved": "0-0"}\n')
    out = subprocess.run([sys.executable, "-c", COLLECT_SCRIPT, str(job)], capture_output=True, text=True, check=True)
    assert "controlled-managed-run" not in out.stdout
    fake_executor.job_dir = job
    recovered = submit.recover("gym-job-a", "bench")
    assert recovered.decision.action == RecoveryAction.STOP
    assert "rollout store" in recovered.decision.reason
    assert recovered.dispatched is None and fake_executor.calls == ["decide"]
    # The known inventory stays the denominator; no saved row is read as an outcome.
    assert recovered.decision.expected == 1 and recovered.decision.counts == {"unknown": 1}
    assert recovered.decision.identities == {"unknown": [(0, 0)]}
    # Nothing on the run host changes.
    assert (job / "artifacts" / "rollouts.jsonl").read_text() == "".join(json.dumps(r) + "\n" for r in results)


@pytest.mark.parametrize("reward", [1.0, 0.0])
def test_a_run_without_a_rollout_store_keeps_its_complete_noop(tmp_path, fake_executor, reward) -> None:
    fake_executor.job_dir = write_job_dir(
        tmp_path / "job", n=1, results=[ident(0, reward=reward, _ng_run_id=None)], failures=[]
    )
    recovered = submit.recover("gym-job-a", "bench")
    assert recovered.ok and recovered.decision.action == RecoveryAction.NOOP


# The sidecar record rollout collection writes for a failed /run with route_failures_to_sidecar off.
UNROUTED_DISPATCH_ERROR = {
    "_ng_failure_class": "agent_run_error",
    "_ng_failure_type": "ClientResponseError",
    "_ng_failure_http_status": 400,
    "_ng_elapsed_s": 8.475396316498518e-05,
    "_ng_task_index": 0,
    "_ng_rollout_index": 0,
    "_ng_environment_server": "my_environment_server",
    "_ng_result_type": "legacy_agent",
    "_ng_failure_events": {
        "failure-id-fixture": {
            "event_id": "7becf785f445465b9a6e92b7247153b6",
            "boundary": "collection",
            "failure_reason": "Rollout dispatch failed: ClientResponseError",
            "failure_kind": "collection_dispatch_failed",
            "operation": "collection.run",
            "foreign_error": {
                "exception_type": "aiohttp.client_exceptions.ClientResponseError",
                "message": "boom",
                "http_status": 400,
            },
        }
    },
    "_ng_failure_id": "7becf785f445465b9a6e92b7247153b6",
    "_ng_omitted_rollout": True,
    "_ng_unrouted_run_error": True,
}


def test_an_unrouted_dispatch_error_without_a_standing_outcome_is_unknown_not_missing(tmp_path, fake_executor) -> None:
    fake_executor.job_dir = write_job_dir(tmp_path / "job", n=1, results=[], failures=[UNROUTED_DISPATCH_ERROR])
    out = submit.recover("gym-job-a", "bench")
    assert out.decision.action == RecoveryAction.STOP and out.dispatched is None
    assert out.decision.identities.get("missing") is None and out.decision.identities["unknown"] == [(0, 0)]
    (evidence,) = out.decision.evidence
    assert evidence.failure_id == "7becf785f445465b9a6e92b7247153b6" and evidence.attempts == 0
    assert evidence.events == [("collection", "collection.run", None, None, None)]


def test_an_unrouted_dispatch_error_keeps_other_evidence_in_charge(tmp_path, fake_executor) -> None:
    # Never dispatched: still missing, and the policy runs it.
    fake_executor.job_dir = write_job_dir(tmp_path / "missing", n=1, results=[], failures=[])
    assert submit.recover("gym-job-a", "bench").decision.identities["missing"] == [(0, 0)]
    # A saved grade stands over a later unrouted error.
    fake_executor.job_dir = write_job_dir(
        tmp_path / "graded", n=1, results=[ident(0, reward=0.0)], failures=[UNROUTED_DISPATCH_ERROR]
    )
    assert submit.recover("gym-job-a", "bench").decision.action == RecoveryAction.NOOP
    # A standing typed failure keeps its own decision; the unrouted record spends no attempt.
    fake_executor.job_dir = write_job_dir(
        tmp_path / "typed", n=1, results=[], failures=[ident(0, **INFRA), UNROUTED_DISPATCH_ERROR]
    )
    decision = submit.recover("gym-job-a", "bench").decision
    assert decision.identities.get("unknown") is None
    (evidence,) = decision.evidence
    assert evidence.attempts == 1 and evidence.failure_kind == "provider_unavailable"


@pytest.mark.parametrize("kind", ["missing", "judge", "settled"])
def test_inspect_recovery_reads_once_and_never_dispatches(tmp_path, fake_executor, kind):
    rows = (
        []
        if kind == "missing"
        else [
            ident(0, _ng_failure_class="judge_failed", response=SAVED_ANSWER, reward=0.0)
            if kind == "judge"
            else ident(0, reward=0.0)
        ]
    )
    fake_executor.job_dir = write_job_dir(tmp_path / "job", n=1, results=rows, failures=[])
    before = {str(p): p.read_bytes() for p in fake_executor.job_dir.rglob("*") if p.is_file()}
    outcome = submit.inspect_recovery("gym-job-inspection", "bench")
    assert outcome.decision.action == {"missing": "resubmit", "judge": "reverify", "settled": "noop"}[kind]
    assert outcome.dispatched is None and fake_executor.calls == ["decide"]
    assert outcome.gate_task["status"] != "SUCCESS"
    assert before == {str(p): p.read_bytes() for p in fake_executor.job_dir.rglob("*") if p.is_file()}


def test_original_record_reader_preserves_bytes_and_checks_submission_identity(tmp_path, fake_executor):
    from nemo_gym.orchestration.jobs import BenchmarkJob, SubmissionRecord

    record = SubmissionRecord(
        gym_job_id="gym-job-original",
        gym_version="test",
        submitted_by="test",
        submitted_at="2026-10-31T00:00:00Z",
        run_dir=str(tmp_path),
        cluster="test",
        executor="fake",
        benchmarks=[BenchmarkJob(benchmark="bench", job_dir=str(tmp_path / "bench"), job_id="1")],
    )
    path = tmp_path / "gym-job.json"
    path.write_text(record.model_dump_json())
    before = path.read_bytes()
    loaded, digest = submit.load_original_submission_record(path)
    assert loaded == record and digest == hashlib.sha256(before).hexdigest()
    with pytest.raises(ValueError, match="another submission"):
        submit.inspect_recovery("wrong-submission", "bench", original_record=loaded)
    assert fake_executor.calls == [] and path.read_bytes() == before


@pytest.mark.parametrize("score_key", [[], {}, [True], [-1], [0.5]])
def test_inspection_rejects_invalid_score_paths_before_collection(fake_executor, score_key):
    with pytest.raises(ValueError, match="score_key"):
        submit.inspect_recovery("gym-job-invalid", "bench", score_key=score_key)
    assert fake_executor.calls == []


@pytest.mark.parametrize("kind", ["missing", "judge", "settled"])
@pytest.mark.parametrize("gate_flags", [[], ["--max-unhealthy-fraction", "0.02"]])
def test_cli_inspection_never_dispatches_even_when_recovery_would(tmp_path, fake_executor, capsys, kind, gate_flags):
    rows = (
        []
        if kind == "missing"
        else [
            ident(0, _ng_failure_class="judge_failed", response=SAVED_ANSWER, reward=0.0)
            if kind == "judge"
            else ident(0, reward=0.0)
        ]
    )
    fake_executor.job_dir = write_job_dir(tmp_path / "job", n=1, results=rows, failures=[])
    before = {str(p): p.read_bytes() for p in fake_executor.job_dir.rglob("*") if p.is_file()}
    code, out = run_cli(
        ["eval", "resubmit", "gym-job-inspect", "bench", "--recover", "--inspect", *gate_flags],
        capsys,
    )
    assert code == 1 and out["gate_task"]["status"] != "SUCCESS"
    assert out["decision"]["action"] == {"missing": "resubmit", "judge": "reverify", "settled": "noop"}[kind]
    assert out["dispatched"] is None and fake_executor.calls == ["decide"]
    assert before == {str(p): p.read_bytes() for p in fake_executor.job_dir.rglob("*") if p.is_file()}


@pytest.mark.parametrize("flags", [["--inspect"], ["--recover", "--record", "missing.json"]])
def test_cli_inspection_modes_require_safe_combinations(fake_executor, monkeypatch, flags):
    from nemo_gym.cli.main import main

    monkeypatch.setattr(sys, "argv", ["gym", "eval", "resubmit", "job", "bench", *flags])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2 and fake_executor.calls == []


def test_cli_inspection_reads_an_original_record_without_local_index(tmp_path, fake_executor, monkeypatch, capsys):
    import hashlib

    from nemo_gym.orchestration.jobs import BenchmarkJob, SubmissionRecord

    job = write_job_dir(tmp_path / "job", n=1, results=[ident(0, reward=0.0)], failures=[])
    fake_executor.job_dir = job
    record = SubmissionRecord(
        gym_job_id="gym-job-original",
        gym_version="test",
        submitted_by="test",
        submitted_at="2026-10-31T00:00:00Z",
        run_dir=str(tmp_path),
        cluster="test",
        executor="fake",
        benchmarks=[BenchmarkJob(benchmark="bench", job_dir=str(job), job_id="1")],
    )
    source = tmp_path / "original.json"
    source.write_text(record.model_dump_json())
    before = source.read_bytes()

    def no_index(*args):
        raise AssertionError("original-record inspection must not read the local index")

    monkeypatch.setattr(submit, "load_submission_record", no_index)
    code, out = run_cli(
        [
            "eval",
            "resubmit",
            record.gym_job_id,
            "bench",
            "--recover",
            "--inspect",
            "--record",
            str(source),
            "--max-unhealthy-fraction",
            "0.02",
        ],
        capsys,
    )
    assert code == 1 and out["decision"]["action"] == "noop" and out["dispatched"] is None
    assert out["original_record_sha256"] == hashlib.sha256(before).hexdigest()
    assert fake_executor.calls == ["decide"] and source.read_bytes() == before


def test_cli_outcome_names_the_reader_token(tmp_path, fake_executor, capsys, monkeypatch) -> None:
    from nemo_gym.orchestration.executors.slurm import reader as slurm_reader

    monkeypatch.setattr(slurm_reader, "_reader_token", None)
    monkeypatch.setenv("GYM_READER_TOKEN", "caller-token")
    fake_executor.job_dir = write_job_dir(tmp_path / "job", n=1, results=[ident(0, reward=1.0)], failures=[])
    code, out = run_cli(["eval", "resubmit", "gym-job-t", "bench", "--recover", "--inspect"], capsys)
    assert out["reader_token"] == "caller-token"


# --- the judge-only job of a model-served run: its option, and what inspect reports of it --------------------------


def _judge_job_dir(tmp_path: Path) -> Path:
    return write_job_dir(
        tmp_path / "job",
        n=2,
        results=[ident(0, reward=1.0), ident(1, _ng_failure_class="judge_failed", response=SAVED_ANSWER, reward=0.0)],
        failures=[],
    )


def test_recover_passes_the_judge_partition_to_the_executor_only_when_given(tmp_path, fake_executor) -> None:
    fake_executor.job_dir = _judge_job_dir(tmp_path)
    assert submit.recover("gym-job-jp1", "bench").dispatched == "reverify"
    assert fake_executor.options == [{}]  # an executor that predates the option is called as before
    assert submit.recover("gym-job-jp2", "bench", judge_partition="cpu").dispatched == "reverify"
    assert fake_executor.options == [{}, {"judge_partition": "cpu"}]


def test_a_repin_reaches_the_saved_answer_rejudgment(tmp_path, fake_executor) -> None:
    fake_executor.job_dir = _judge_job_dir(tmp_path)
    submit.recover("gym-job-jp3", "bench", repin="b" * 40, repin_reason="launcher revision")
    assert fake_executor.options == [{"repin": "b" * 40, "repin_reason": "launcher revision"}]


def _journal_judge_jobs(job: Path, entries: list[dict]) -> None:
    from nemo_gym.eval.recovery.evidence import JUDGE_JOB_JOURNAL_NAME

    (job / JUDGE_JOB_JOURNAL_NAME).write_text("".join(json.dumps(entry) + "\n" for entry in entries))


def test_inspect_reports_the_judge_only_jobs_of_the_run_and_their_outcomes(tmp_path, fake_executor) -> None:
    from nemo_gym.eval.rollout_reverification import write_settle_receipt

    job = _judge_job_dir(tmp_path)
    fake_executor.job_dir = job
    done = job / "reverification" / "judge-a" / "receipt.json"
    write_settle_receipt(
        done,
        state="completed",
        source=str(job / "artifacts" / "rollouts.jsonl"),
        job_id="77",
        settlement={"selected": 2, "unsupported": 0, "judged": 2, "settled": 1, "withdrawn": 1},
        started_at="2026-01-01T00:00:00Z",
        coverage={"masked": 0, "expected": 2},
        due_after=0,
        refreshed={"metrics_recomputed": True, "health_summary_recomputed": True},
    )
    _journal_judge_jobs(
        job,
        [
            {
                "at": "2026-01-01T00:00:00Z",
                "job_id": "77",
                "partition": "cpu",
                "walltime": "01:00:00",
                "rows": 2,
                "receipt": str(done),
                "nonce": "judge-a",
            },
            {
                "at": "2026-01-01T01:00:00Z",
                "job_id": "78",
                "partition": "cpu",
                "walltime": "01:00:00",
                "rows": 1,
                "receipt": str(job / "reverification" / "judge-b" / "receipt.json"),
                "nonce": "judge-b",
            },
        ],
    )

    out = submit.inspect_recovery("gym-job-jp4", "bench")

    assert out.dispatched is None and fake_executor.calls == ["decide"]
    first, second = out.gate_task["judge_jobs"]
    assert (
        first["job_id"] == "77" and first["state"] == "completed" and first["settled"] == 1 and first["withdrawn"] == 1
    )
    assert first["partition"] == "cpu" and first["rows"] == 2
    assert second["job_id"] == "78" and second["state"] == "queued" and "settled" not in second
    assert any("judge-only job 77" in w and "settled 1" in w for w in out.gate_task["warnings"])
    assert any("judge-only job 78" in w and "no receipt" in w for w in out.gate_task["warnings"])
    # The saved answer text never leaves the run host with the evidence.
    assert "saved answer" not in json.dumps(out.gate_task["judge_jobs"])


def test_the_cli_takes_a_judge_partition_with_recover_only(tmp_path, fake_executor, capsys) -> None:
    fake_executor.job_dir = _judge_job_dir(tmp_path)
    code, out = run_cli(["eval", "resubmit", "gym-job-jp5", "bench", "--recover", "--judge-partition", "cpu"], capsys)
    assert code == 0 and out["dispatched"] == "reverify" and fake_executor.options == [{"judge_partition": "cpu"}]
    for argv in (
        ["eval", "resubmit", "gym-job-jp6", "bench", "--recover", "--inspect", "--judge-partition", "cpu"],
        ["eval", "resubmit", "gym-job-jp7", "bench", "--judge-partition", "cpu"],
    ):
        assert _cli_refuses(argv, capsys) == 2
    assert fake_executor.options == [{"judge_partition": "cpu"}]  # the refusals decided and dispatched nothing


def _cli_refuses(argv: list[str], capsys) -> int:
    """The parser's exit code for ``argv`` (its message goes to stderr, so ``run_cli`` cannot read a result)."""
    from nemo_gym.cli.main import main

    saved = sys.argv
    sys.argv = ["gym", *argv]
    try:
        with pytest.raises(SystemExit) as refused:
            main()
    finally:
        sys.argv = saved
        capsys.readouterr()
    return refused.value.code


# ---------------------------------------------------------------------------
# Verification skipped by configuration (judge_skipped): complete, unverified by construction
# ---------------------------------------------------------------------------


def test_resubmit_recover_on_an_all_skipped_run_dispatches_nothing(tmp_path, fake_executor) -> None:
    fake_executor.job_dir = _skipped_job(tmp_path)
    out = submit.recover("gym-job-skip", "bench")
    assert out.ok and out.dispatched is None and out.decision.action == RecoveryAction.NOOP
    assert fake_executor.calls == ["decide"]


def test_inspect_prints_the_unverified_reading_of_an_all_skipped_run(tmp_path, fake_executor) -> None:
    fake_executor.job_dir = _skipped_job(tmp_path)
    out = submit.inspect_recovery("gym-job-skip2", "bench")
    assert out.gate_task["status"] == "UNVERIFIED" and out.gate_task["score"] is None
    assert any("not graded by configuration (judge_skipped)" in w for w in out.gate_task["warnings"])
    assert out.gate_task["errors"] == [] and out.dispatched is None


def test_the_verdict_of_an_all_skipped_run_reports_it_unverified_by_configuration_and_its_delivery_state(
    tmp_path, fake_executor
) -> None:
    fake_executor.job_dir = _skipped_job(tmp_path)
    out = submit.inspect_recovery("gym-job-skip3", "bench")
    assert out.gate_task["status"] == "UNVERIFIED"
    assert out.operator_verdict == {
        "state": "unknown",
        "reason": "unverified by configuration: 5 of 5 rows were not verified (verification skipped by configuration)",
        # Nothing to recover; its provenance does not say MLflow was configured, so a recovery would not export it.
        "action": "none: report the run as unverified by configuration; report the MLflow delivery as not evidenced: "
        "the run's saved provenance does not say MLflow was configured (provenance.mlflow.configured), so a recovery "
        "does not export it and its results stand without an export; if MLflow was meant to be used, export them "
        "with the tracking tooling and inspect again with --mlflow-run-id and --mlflow-status",
    }
