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
from types import SimpleNamespace

import pytest

from nemo_gym.core.responses_api_types import NeMoGymResponse
from nemo_gym.orchestration import submit
from nemo_gym.orchestration.executors.base import BaseExecutor
from nemo_gym.orchestration.recovery import (
    COLLECT_SCRIPT,
    CollectedEvidence,
    GateCriteria,
    RecoveryAction,
    RecoveryOutcome,
    RecoveryOutcomePolicyConfig,
    decide_from_collected,
    gate_task_summary,
)
from nemo_gym.rollouts.rollout_collection import RolloutCollectionConfig, RolloutCollectionHelper


class _StartsWith:
    def __init__(self, prefix: str) -> None:
        self.prefix = prefix

    def __eq__(self, other: object) -> bool:
        return isinstance(other, str) and other.startswith(self.prefix)

    def __repr__(self) -> str:
        return f"startswith({self.prefix!r})"


# A report without run.stats.responses leaves the response gate unknown, and the trial count is a lower bound.
HEALTH_UNKNOWN = _StartsWith("response/trial health unknown: ['")


# The saved answer a judge failure keeps beside its row; re-verification re-judges it.

# Use the response serialized by the actual typed producer/verifier contract, not an output-only placeholder.
SAVED_ANSWER = NeMoGymResponse(
    id="response-saved",
    object="response",
    created_at=0,
    model="synthetic",
    output=[
        {
            "id": "message-saved",
            "type": "message",
            "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": "B", "annotations": []}],
        }
    ],
    parallel_tool_calls=False,
    tool_choice="auto",
    tools=[],
).model_dump(mode="json")


def ident(task: int, **extra) -> dict:
    return {"_ng_task_index": task, "_ng_rollout_index": 0, **extra}


def write_job_dir(
    root: Path,
    *,
    n: int,
    results: list[dict],
    failures: list[dict],
    offset: int | None = None,
    retries: int | None = None,
    walltime: int | None = None,
    chain: list[str] = (),
) -> Path:
    art = root / "artifacts"
    art.mkdir(parents=True)
    (art / "rollouts_materialized_inputs.jsonl").write_text(
        "".join(json.dumps(ident(t, responses_create_params={"input": f"prompt {t}"})) + "\n" for t in range(n))
    )
    (art / "rollouts.jsonl").write_text("".join(json.dumps(r) + "\n" for r in results))
    (art / "rollouts_failures.jsonl").write_text("".join(json.dumps(r) + "\n" for r in failures))
    if offset is not None:
        (art / "rollouts_resubmit_offset").write_text(str(offset))
    if retries is not None:
        (root / ".gym_infra_retries").write_text(str(retries))
    if walltime is not None:
        (root / ".gym_accumulated_walltime").write_text(str(walltime))
    if chain:
        (root / ".gym_job_chain").write_text("\n".join(chain))
    return root


def collect(job_dir: Path) -> CollectedEvidence:
    out = subprocess.run(
        [sys.executable, "-c", COLLECT_SCRIPT, str(job_dir)], capture_output=True, text=True, check=True
    )
    return CollectedEvidence.model_validate_json(out.stdout)


INFRA = dict(
    failure_kind="provider_unavailable",
    _ng_failure_terminal=False,
    _ng_failure_max_attempts=3,
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


def test_collector_reads_identity_and_cause_fields_only_and_honours_the_resubmit_offset(tmp_path) -> None:
    job = write_job_dir(
        tmp_path / "job",
        n=2,
        results=[ident(0, reward=1.0, response={"output_text": "secret answer"})],
        failures=[ident(1, **INFRA), ident(1, _ng_attempt_index=1, **INFRA)],
        offset=1,
        retries=2,
        walltime=7200,
        chain=["101", "102"],
    )
    got = collect(job)
    assert got.results == [{"_ng_task_index": 0, "_ng_rollout_index": 0, "reward": 1.0, "response": {}}]
    assert "secret answer" not in got.model_dump_json() and "prompt" not in got.model_dump_json()
    assert got.resubmit_offset == 1 and [f["_line"] for f in got.failures] == [0, 1]
    assert got.failures[1]["_ng_attempt_index"] == 1
    assert got.failures[1]["_ng_failure_events"]["e"]["attempt_id"] == "a1"
    assert (got.infra_retries_used, got.walltime_used_s, got.job_chain) == (2.0, 7200.0, ["101", "102"])


@pytest.mark.parametrize(
    ("used", "action"),
    [(0, RecoveryAction.RESUBMIT), (2, RecoveryAction.RESUBMIT), (3, RecoveryAction.STOP)],
)
def test_stored_infra_retry_budget_is_consumed_not_guessed(tmp_path, used, action) -> None:
    job = write_job_dir(tmp_path / "job", n=2, results=[ident(0, reward=1.0)], failures=[], retries=used)
    d = decide_from_collected(collect(job), max_infra_retries=3, max_walltime_s=None, live_job=False)
    assert d.action == action


PROTOCOL_VIOLATION = dict(
    failure_kind="environment_protocol_violation",
    _ng_failure_terminal=True,
    _ng_failure_events={
        "e": {
            "boundary": "collection",
            "operation": "collection.run",
            "operation_id": "op",
            "attempt_id": "a1",
            "cause_id": None,
        }
    },
)


def test_only_protocol_violations_stop_with_the_producer_hint_and_never_recover(tmp_path) -> None:
    job = write_job_dir(
        tmp_path / "job",
        n=2,
        results=[],
        failures=[ident(0, **PROTOCOL_VIOLATION), ident(1, **PROTOCOL_VIOLATION)],
    )
    d = decide_from_collected(collect(job), max_infra_retries=3, max_walltime_s=None, live_job=False)
    assert d.action == RecoveryAction.STOP
    assert "environment_protocol_violation" in d.reason and "producer bug" in d.reason
    assert "gym eval submit" in d.reason and "--recover" not in d.reason


def test_walltime_cap_and_unknown_cap_both_stop(tmp_path) -> None:
    job = write_job_dir(tmp_path / "job", n=1, results=[], failures=[], walltime=57600)
    assert (
        decide_from_collected(collect(job), max_infra_retries=3, max_walltime_s=57600.0, live_job=False).action
        == "stop"
    )
    assert (
        decide_from_collected(collect(job), max_infra_retries=None, max_walltime_s=None, live_job=False).action
        == "stop"
    )


def test_missing_inventory_is_a_fresh_restart(tmp_path) -> None:
    job = tmp_path / "job"
    (job / "artifacts").mkdir(parents=True)
    d = decide_from_collected(collect(job), max_infra_retries=3, max_walltime_s=None, live_job=False)
    assert d.action == RecoveryAction.FRESH_RESTART and "inventory" in d.reason


class FakeExecutor:
    """Records every dispatch; ``supports_reverify`` stands for an executor with a saved-answer re-judgment path."""

    name = "fake"
    calls: list[str] = []
    # The options each saved-answer re-judgment was asked with (a judge partition, a repin), in call order.
    options: list[dict] = []
    job_dir: Path
    live = False
    supports_reverify = True

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


def test_gate_summary_never_equates_scored_with_expected_by_dropping_failures(tmp_path) -> None:
    job = write_job_dir(
        tmp_path / "job",
        n=4,
        results=[ident(0, reward=1.0), ident(1, _ng_failure_class="judge_failed", response=SAVED_ANSWER)],
        failures=[ident(2, **INFRA)],
        retries=0,
    )
    d = decide_from_collected(collect(job), max_infra_retries=3, max_walltime_s=None, live_job=False)
    t = gate_task_summary(d, score=0.25, summary_valid=True, summary_reasons=[], health_passed=True, health_reasons=[])
    assert (t["expected_samples"], t["scored_samples"]) == (4, 1) and t["status"] != "SUCCESS"
    assert any("judge failures" in e for e in t["errors"]) and any("no row" in e for e in t["errors"])


@pytest.mark.parametrize(("summary_valid", "health"), [(None, True), (True, None), (False, True), (True, False)])
def test_gate_summary_unknown_or_failed_validity_or_health_is_a_blocker(tmp_path, summary_valid, health) -> None:
    job = write_job_dir(tmp_path / "job", n=1, results=[ident(0, reward=1.0)], failures=[])
    d = decide_from_collected(collect(job), max_infra_retries=3, max_walltime_s=None, live_job=False)
    t = gate_task_summary(
        d, score=1.0, summary_valid=summary_valid, summary_reasons=[], health_passed=health, health_reasons=[]
    )
    # Unknown withholds the pass (UNVERIFIED, reason in warnings); only a failed part is FAILED with an error.
    failed = False in (summary_valid, health)
    assert t["status"] == ("FAILED" if failed else "UNVERIFIED") and bool(t["errors"]) is failed
    assert failed or any("unknown" in w for w in t["warnings"])


def test_gate_summary_complete_run_succeeds_and_keeps_model_zeros_as_warnings(tmp_path) -> None:
    final = ident(1, failure_kind="model_context_limit", _ng_failure_terminal=True, _ng_failure_counts_as_zero=True)
    job = write_job_dir(tmp_path / "job", n=2, results=[ident(0, reward=1.0)], failures=[final])
    d = decide_from_collected(collect(job), max_infra_retries=3, max_walltime_s=None, live_job=False)
    t = gate_task_summary(d, score=0.5, summary_valid=True, summary_reasons=[], health_passed=True, health_reasons=[])
    assert (t["status"], t["expected_samples"], t["scored_samples"], t["errors"]) == ("SUCCESS", 2, 2, [])
    assert t["warnings"] and "final evaluated-model" in t["warnings"][0]


def unit_quality(*, expected, healthy=True, verdicts=None, partial=False, reasons=(), **facts) -> dict:
    """Explicit input to the quality/gate unit boundary, never a persisted producer report."""
    return {
        "source_matches": True,
        "artifacts": {
            "expected": expected,
            "missing": 0,
            "healthy": healthy,
            "unhealthy_reasons": list(reasons),
            "partial": partial,
        },
        "verdicts": verdicts if verdicts is not None else {"healthy": expected, "unhealthy": 0, "unobserved": 0},
        **facts,
    }


def produce_quality(job: Path) -> None:
    """Use current raw saved rows; a grade-only fixture has unobserved response/trial health."""
    from nemo_gym.rollouts.rollout_health import run_health_checks

    run_health_checks(job / "artifacts" / "rollouts.jsonl", workers=1)


def collect_with_score(job_dir: Path, key: str) -> CollectedEvidence:
    out = subprocess.run(
        [sys.executable, "-c", COLLECT_SCRIPT, str(job_dir), key], capture_output=True, text=True, check=True
    )
    return CollectedEvidence.model_validate_json(out.stdout)


def complete_job(tmp_path: Path) -> Path:
    job = write_job_dir(tmp_path / "job", n=4, results=[ident(t, reward=1.0) for t in range(4)], failures=[])
    (job / "artifacts" / "rollouts_aggregate_metrics.json").write_text(json.dumps({"agent": {"mean/reward": 0.75}}))
    write_postrun_receipt(job)
    return job


RECEIPT_FILES = {
    "rollouts_jsonl_fpath": "rollouts.jsonl",
    "resolved_config_yaml_fpath": "rollouts_resolved_config.yaml",
    "aggregate_metrics_json_fpath": "rollouts_aggregate_metrics.json",
    "failures_jsonl_fpath": "rollouts_failures.jsonl",
    "materialized_inputs_jsonl_fpath": "rollouts_materialized_inputs.jsonl",
}


def artifact_digests(job: Path) -> dict:
    files = {key: job / "artifacts" / name for key, name in RECEIPT_FILES.items()}
    return {key: hashlib.sha256(f.read_bytes()).hexdigest() if f.exists() else None for key, f in files.items()}


def write_postrun_receipt(job: Path, run_id: str = "postrun-run") -> None:
    """The version-1 receipt a content-bound post-run export writes: its backend run id and the artifact digests."""
    receipt = {
        "version": 1,
        "run_id": run_id,
        "artifacts": artifact_digests(job),
        "uploaded_rollouts": True,
        "rollout_projection_version": 1,
    }
    (job / "artifacts" / "rollouts_mlflow_export_done").write_text(json.dumps(receipt))


GOOD_VERDICTS = {"healthy": 4, "unhealthy": 0, "unobserved": 0}


@pytest.mark.parametrize(
    ("quality", "criterion", "summary_valid", "health"),
    [
        (None, 0.02, None, None),
        (dict(expected=4, healthy=True, verdicts=GOOD_VERDICTS, partial=True), 0.02, None, True),
        (dict(expected=3, healthy=True, verdicts=GOOD_VERDICTS), 0.02, False, True),
        (dict(expected=4, healthy=None, verdicts=GOOD_VERDICTS), 0.02, None, True),
        (dict(expected=4, healthy=False, verdicts=GOOD_VERDICTS, reasons=["missing 1 of 4"]), 0.02, False, True),
        (dict(expected=4, healthy=True, verdicts=GOOD_VERDICTS), None, True, None),
        (dict(expected=4, healthy=True, verdicts={"healthy": 3, "unhealthy": 0, "unobserved": 1}), 0.02, True, None),
        (dict(expected=4, healthy=True, verdicts={"healthy": 3, "unhealthy": 1, "unobserved": 0}), 0.02, True, False),
        (dict(expected=4, healthy=True, verdicts=GOOD_VERDICTS), 0.02, True, True),
    ],
)
def test_quality_inputs_keep_validity_and_health_apart_and_fail_closed(
    tmp_path, quality, criterion, summary_valid, health
) -> None:
    from nemo_gym.orchestration.recovery import quality_inputs

    job = complete_job(tmp_path)
    collected = collect_with_score(job, json.dumps(["agent", "mean/reward"]))
    collected.quality = unit_quality(**quality) if quality is not None else None
    d = decide_from_collected(collected, max_infra_retries=3, max_walltime_s=None, live_job=False)
    got_summary, _, got_health, reasons = quality_inputs(d, collected, max_unhealthy_fraction=criterion)
    assert (got_summary, got_health) == (summary_valid, health) and reasons


def test_gate_summary_from_collected_reads_the_named_score_and_passes_only_when_everything_holds(tmp_path) -> None:
    from nemo_gym.orchestration.recovery import gate_summary_from_collected

    job = complete_job(tmp_path)
    produce_quality(job)
    collected = collect_with_score(job, json.dumps(["agent", "mean/reward"]))
    d = decide_from_collected(collected, max_infra_retries=3, max_walltime_s=None, live_job=False)
    task = gate_summary_from_collected(
        d, collected, max_unhealthy_fraction=0.02, mlflow_status="FINISHED", mlflow_run_id="postrun-run"
    )
    assert (task["status"], task["score"], task["expected_samples"], task["scored_samples"], task["errors"]) == (
        "UNVERIFIED",
        0.75,
        4,
        4,
        [],
    )
    assert HEALTH_UNKNOWN in task["warnings"]
    missing_key = collect_with_score(job, json.dumps(["agent", "no_such_metric"]))
    assert gate_summary_from_collected(d, missing_key, max_unhealthy_fraction=0.02)["score"] is None


def final_model_zero(task: int, rollout: int) -> dict:
    """A terminal counts-as-zero sidecar row whose root event is the agent's policy request, caused by the model."""
    return {
        "_ng_task_index": task,
        "_ng_rollout_index": rollout,
        "_ng_failure_class": "agent_run_error",
        "failure_kind": "model_transport_error",
        "_ng_failure_terminal": True,
        "_ng_failure_counts_as_zero": True,
        "_ng_failure_id": f"f-{task}-{rollout}",
        "_ng_failure_events": {
            f"f-{task}-{rollout}": {
                "boundary": "agent",
                "operation": "agent.policy_request",
                "failure_kind": "model_transport_error",
                "operation_id": "op-1",
                "attempt_id": "att-1",
                "cause_id": "cause",
            },
            "cause": {
                "boundary": "model",
                "operation": "model_request",
                "failure_kind": "model_request_error",
                "operation_id": "op-2",
                "attempt_id": "att-3",
                "cause_id": None,
            },
        },
    }


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


@pytest.mark.parametrize(
    ("files", "chain", "capped", "action"),
    [
        ({".gym_infra_retries": "two"}, ["11", "12"], False, RecoveryAction.STOP),
        ({".gym_infra_retries": "1.5"}, ["11", "12"], False, RecoveryAction.STOP),
        ({".gym_infra_retries": "nan"}, ["11", "12"], False, RecoveryAction.STOP),
        ({".gym_infra_retries": "inf"}, ["11", "12"], False, RecoveryAction.STOP),
        ({".gym_accumulated_walltime": "nan"}, ["11", "12"], True, RecoveryAction.STOP),
        ({".gym_accumulated_walltime": "-1"}, ["11", "12"], True, RecoveryAction.STOP),
        ({}, ["11", "12"], False, RecoveryAction.RESUBMIT),
        ({}, ["11", "12"], True, RecoveryAction.STOP),
        ({".gym_accumulated_walltime": "x"}, ["11", "12"], True, RecoveryAction.STOP),
        ({}, ["11"], True, RecoveryAction.STOP),
        ({".gym_accumulated_walltime": "3600"}, ["11", "12"], True, RecoveryAction.STOP),
    ],
)
def test_a_lost_or_malformed_budget_counter_stops_and_is_never_reset(tmp_path, files, chain, capped, action) -> None:
    job = write_job_dir(tmp_path / "job", n=2, results=[ident(0, reward=1.0)], failures=[], chain=chain)
    for name, text in files.items():
        (job / name).write_text(text)
    d = decide_from_collected(
        collect(job), max_infra_retries=3, max_walltime_s=57600.0 if capped else None, live_job=False
    )
    assert d.action == action, d.reason


@pytest.mark.parametrize("row", [dict(), dict(reward="1")])
def test_a_result_without_a_usable_grade_is_unknown_not_scored(tmp_path, row) -> None:
    job = write_job_dir(tmp_path / "job", n=2, results=[ident(0, reward=1.0), ident(1, **row)], failures=[], retries=0)
    d = decide_from_collected(collect(job), max_infra_retries=3, max_walltime_s=None, live_job=False)
    assert d.identities.get("unknown") == [(1, 0)] and d.action == RecoveryAction.STOP


@pytest.mark.parametrize("row", [dict(reward=1.0, mask_sample=True), dict(reward=0.0, evaluation_completed=False)])
def test_a_masked_result_is_retained_but_not_scored(tmp_path, row) -> None:
    job = write_job_dir(tmp_path / "job", n=2, results=[ident(0, reward=1.0), ident(1, **row)], failures=[], retries=0)
    d = decide_from_collected(collect(job), max_infra_retries=3, max_walltime_s=None, live_job=False)
    assert d.identities.get("masked") == [(1, 0)] and "unknown" not in d.counts
    assert d.action == RecoveryAction.STOP and not d.complete


def test_a_legitimate_model_zero_grade_stays_scored(tmp_path) -> None:
    job = write_job_dir(tmp_path / "job", n=2, results=[ident(0, reward=1.0), ident(1, reward=0.0)], failures=[])
    d = decide_from_collected(collect(job), max_infra_retries=3, max_walltime_s=None, live_job=False)
    assert d.counts == {"scored": 2} and d.action == RecoveryAction.NOOP


def test_an_unavailable_score_is_an_explicit_blocker(tmp_path) -> None:
    from nemo_gym.orchestration.recovery import gate_summary_from_collected

    job = complete_job(tmp_path)
    produce_quality(job)
    collected = collect(job)
    d = decide_from_collected(collected, max_infra_retries=3, max_walltime_s=None, live_job=False)
    task = gate_summary_from_collected(d, collected, max_unhealthy_fraction=0.02)
    assert (
        task["score"] is None and task["status"] == "FAILED" and any("score unavailable" in e for e in task["errors"])
    )


def test_verdict_counts_that_do_not_cover_the_inventory_make_health_unknown(tmp_path) -> None:
    from nemo_gym.orchestration.recovery import quality_inputs

    job = complete_job(tmp_path)
    collected = collect(job)
    collected.quality = unit_quality(expected=4, verdicts={"healthy": 9, "unhealthy": 0, "unobserved": 0})
    d = decide_from_collected(collected, max_infra_retries=3, max_walltime_s=None, live_job=False)
    _, _, health, reasons = quality_inputs(d, collected, max_unhealthy_fraction=0.02)
    assert health is None and "do not cover" in reasons[0]


@pytest.mark.parametrize("score", [float("nan"), float("inf"), float("-inf")])
def test_a_nonfinite_aggregate_score_cannot_be_accepted(tmp_path, score) -> None:
    from nemo_gym.orchestration.recovery import gate_summary_from_collected

    job = complete_job(tmp_path)
    produce_quality(job)
    (job / "artifacts" / "rollouts_aggregate_metrics.json").write_text(json.dumps({"agent": {"mean/reward": score}}))
    out = subprocess.run(
        [sys.executable, "-c", COLLECT_SCRIPT, str(job), json.dumps(["agent", "mean/reward"])],
        capture_output=True,
        text=True,
        check=True,
    )
    collected = CollectedEvidence.model_validate_json(out.stdout)
    assert collected.score is None
    decision = decide_from_collected(collected, max_infra_retries=3, max_walltime_s=None, live_job=False)
    task = gate_summary_from_collected(
        decision, collected.model_copy(update={"score": score}), max_unhealthy_fraction=0.02
    )
    assert task["status"] == "FAILED" and any("score unavailable" in error for error in task["errors"])


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


@pytest.mark.parametrize(
    ("verdicts", "criterion", "health"),
    [
        ({"healthy": 98, "unhealthy": 2, "unobserved": 0}, 0.02, False),
        ({"healthy": 99, "unhealthy": 1, "unobserved": 0}, 0.02, True),
        ({"healthy": 0, "unhealthy": 100, "unobserved": 0}, float("nan"), None),
        ({"healthy": 0, "unhealthy": 100, "unobserved": 0}, float("inf"), None),
        ({"healthy": 100, "unhealthy": 0, "unobserved": 0}, True, None),
        ({"healthy": 101, "unhealthy": -1, "unobserved": 0}, 0.02, None),
        ({"healthy": True, "unhealthy": 0, "unobserved": 99}, 0.02, None),
    ],
)
def test_health_criterion_and_counts_are_strict(tmp_path, verdicts, criterion, health) -> None:
    from nemo_gym.orchestration.recovery import quality_inputs

    job = write_job_dir(tmp_path / "job", n=100, results=[ident(t, reward=1.0) for t in range(100)], failures=[])
    collected = collect(job)
    collected.quality = unit_quality(expected=100, verdicts=verdicts)
    d = decide_from_collected(collected, max_infra_retries=3, max_walltime_s=None, live_job=False)
    assert quality_inputs(d, collected, max_unhealthy_fraction=criterion)[2] is health


def test_absent_mlflow_evidence_is_a_pending_delivery_fact_not_a_failure(tmp_path) -> None:
    from nemo_gym.orchestration.recovery import gate_summary_from_collected

    job = complete_job(tmp_path)
    produce_quality(job)
    (job / "artifacts" / "rollouts_mlflow_export_done").unlink()
    collected = collect_with_score(job, json.dumps(["agent", "mean/reward"]))
    d = decide_from_collected(collected, max_infra_retries=3, max_walltime_s=None, live_job=False)
    task = gate_summary_from_collected(d, collected, max_unhealthy_fraction=0.02)
    assert task["status"] != "FAILED" and not [e for e in task["errors"] if "MLflow" in e]
    assert task["mlflow_delivery"] == {
        "state": "pending",
        "reasons": ["no MLflow run evidence: no live run id and no post-run export receipt"],
    }


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


def test_named_health_facts_below_the_criterion_reach_the_gate_warnings(tmp_path) -> None:
    from nemo_gym.orchestration.recovery import gate_summary_from_collected

    job = complete_job(tmp_path)
    collected = collect_with_score(job, json.dumps(["agent", "mean/reward"]))
    collected.quality = unit_quality(
        expected=4,
        verdicts={"healthy": 3, "unhealthy": 1, "unobserved": 0},
        issues={"model_call_runaway_generation": 1, "task_consistently_unhealthy": 1, "agent_turn_hollow": 0},
        ignored_checks=[],
        coverage={
            "model_call_runaway_generation": {"evaluated": 4, "unobserved": 0, "ignored": 0},
            "run_excessive_failures": {"evaluated": 0, "unobserved": 1, "ignored": 0},
        },
    )
    d = decide_from_collected(collected, max_infra_retries=3, max_walltime_s=None, live_job=False)
    task = gate_summary_from_collected(
        d, collected, max_unhealthy_fraction=0.5, mlflow_status="FINISHED", mlflow_run_id="postrun-run"
    )
    assert task["status"] == "UNVERIFIED"
    assert task["warnings"][-4].startswith("rollout quality within the configured criterion")
    assert task["warnings"][-3:-1] == [
        "health-check issues (nonzero): model_call_runaway_generation 1, task_consistently_unhealthy 1",
        "health checks not fully evaluated: run_excessive_failures (evaluated 0, unobserved 1, ignored 0)",
    ]
    assert task["warnings"][-1] == task["health_verdict"]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("issues", None),
        ("issues", {"model_call_runaway_generation": -1}),
        ("issues", {"model_call_runaway_generation": True}),
        ("coverage", None),
        ("coverage", {"run_excessive_failures": {"evaluated": 1}}),
        ("ignored_checks", "run_excessive_failures"),
    ],
)
def test_malformed_health_facts_are_unknown_never_clean(field, value) -> None:
    from nemo_gym.orchestration.recovery import health_check_facts

    good = {
        "issues": {"a": 0},
        "coverage": {"a": {"evaluated": 1, "unobserved": 0, "ignored": 0}},
        "ignored_checks": [],
    }
    assert health_check_facts(good) == []
    facts = health_check_facts(good | {field: value})
    assert len(facts) == 1 and "unknown" in facts[0]


def test_the_receipt_fixture_matches_the_content_bound_producer(tmp_path) -> None:
    base = pytest.importorskip("nemo_gym.artifacts.exporters.base")
    if not hasattr(base, "_artifact_version"):
        pytest.skip("this producer writes a run-id-only post-run marker")
    from nemo_gym.artifacts.run_artifacts import RunArtifacts

    job = complete_job(tmp_path)
    artifacts = RunArtifacts(rollouts_jsonl_fpath=job / "artifacts" / "rollouts.jsonl")
    assert artifact_digests(job) == base._artifact_version(artifacts)


@pytest.mark.parametrize(
    "local",
    [
        {"live_run_id": None, "receipt": False},
        {"live_run_id": None, "receipt": "unexpected"},
        {"live_run_id": False, "receipt": None},
        {"live_run_id": None, "receipt": {"current": True, "run_id": ""}},
        {"live_run_id": None, "receipt": {"current": True}},
        {"live_run_id": 7, "receipt": None},
        {"live_run_id": None, "receipt": {"run_id": 5, "current": True}},
        {"live_run_id": None, "receipt": {"run_id": "postrun-run", "current": "yes"}},
        {"live_run_id": None, "receipt": {"run_id": "postrun-run"}},
        {"live_run_id": None, "receipt": ["postrun-run"]},
        {"live_run_id": "  ", "receipt": None},
    ],
)
def test_malformed_mlflow_evidence_never_passes(local) -> None:
    from nemo_gym.orchestration.recovery import mlflow_errors

    assert mlflow_errors(local, "FINISHED")


def test_a_current_post_run_receipt_with_finished_is_the_only_mlflow_pass() -> None:
    from nemo_gym.orchestration.recovery import mlflow_errors

    assert (
        mlflow_errors(
            {
                "live_run_id": None,
                "receipt": {
                    "run_id": "postrun-run",
                    "current": True,
                    "uploaded_rollouts": True,
                    "rollout_projection_version": 1,
                },
            },
            "FINISHED",
            observed_run_id="postrun-run",
        )
        == []
    )


def gate_job(tmp_path: Path, *, expected: int, final_zeros: int, missing: int = 0) -> Path:
    """Saved row/score fixture for gate unit tests; quality is supplied separately."""
    scored = expected - final_zeros - missing
    results = [ident(t, reward=float(t % 2)) for t in range(scored)]
    failures = [final_model_zero(t, 0) for t in range(scored, scored + final_zeros)]
    job = write_job_dir(tmp_path / "job", n=expected, results=results, failures=failures)
    (job / "artifacts" / "rollouts_aggregate_metrics.json").write_text(json.dumps({"agent": {"mean/reward": 0.5}}))
    write_postrun_receipt(job)
    return job


def gate_of(job: Path, *, quality: dict | None = None) -> tuple:
    from nemo_gym.orchestration.recovery import gate_summary_from_collected

    collected = collect_with_score(job, json.dumps(["agent", "mean/reward"]))
    d = decide_from_collected(collected, max_infra_retries=3, max_walltime_s=None, live_job=False)
    collected.quality = quality if quality is not None else unit_quality(expected=d.expected)
    return d, gate_summary_from_collected(
        d, collected, max_unhealthy_fraction=0.02, mlflow_status="FINISHED", mlflow_run_id="postrun-run"
    )


@pytest.mark.parametrize(("final_zeros", "failed"), [(0, False), (1, False), (2, True), (3, True)])
def test_known_trial_failures_fail_at_two_percent_and_are_only_a_lower_bound_below_it(
    tmp_path, final_zeros, failed
) -> None:
    job = gate_job(tmp_path, expected=100, final_zeros=final_zeros)
    rollouts = (job / "artifacts" / "rollouts.jsonl").read_bytes()
    d, task = gate_of(job)
    # Settled answers stand: the action is NOOP whatever the health, and nothing rewrites the main output.
    assert d.action == RecoveryAction.NOOP and (job / "artifacts" / "rollouts.jsonl").read_bytes() == rollouts
    assert task["status"] == ("FAILED" if failed else "UNVERIFIED")
    assert task["scored_samples"] == 100 and task["score"] == 0.5
    [health] = [e for e in task["errors"] + task["warnings"] if e.startswith("response/trial health")]
    health += " " + ((task["evaluation_health"] or {}).get("detail") or "")
    rate = f"evaluation-side faults: {final_zeros} of 100 ({final_zeros / 100:.2%}) is not below 2%"
    if failed:
        assert health.startswith("response/trial health failed") and f"trial: {rate}" in health
        assert f"(server {final_zeros}: model_transport_error {final_zeros})" in health
    else:
        # These rows carry no grading observation: the trial gate stays unknown and says why, with the count.
        assert health.startswith("response/trial health unknown")
        assert f"trial: UNOBSERVED {100 - final_zeros} of 100 expected trials" in health
    assert not any(w.startswith("health gate within") and "trial" in w for w in task["warnings"])


@pytest.mark.parametrize(
    ("responses", "state"),
    [
        ({"rollouts": 100, "unknown": 0, "selected": 250, "faulted": 4, "duplicated": 0}, "within"),
        ({"rollouts": 100, "unknown": 0, "selected": 250, "faulted": 5, "duplicated": 0}, "failed"),
        ({"rollouts": 100, "unknown": 0, "selected": 250, "faulted": 5, "duplicated": 1}, "unknown"),
        ({"rollouts": 100, "unknown": 1, "selected": 250, "faulted": 0, "duplicated": 0}, "unknown"),
        ({"rollouts": 99, "unknown": 0, "selected": 250, "faulted": 0, "duplicated": 0}, "unknown"),
        ({"rollouts": 100, "unknown": 0, "selected": 0, "faulted": 0, "duplicated": 0}, "unknown"),
        ({"rollouts": 100, "unknown": 0, "selected": True, "faulted": 0, "duplicated": 0}, "unknown"),
        ({"rollouts": 100, "unknown": 0, "selected": 250, "faulted": 251, "duplicated": 0}, "unknown"),
        ({"rollouts": 100, "unknown": 0, "selected": 250, "faulted": 0}, "unknown"),
    ],
)
def test_the_response_gate_reads_the_reported_population_strictly_below_two_percent(
    tmp_path, responses, state
) -> None:
    d, task = gate_of(
        gate_job(tmp_path, expected=100, final_zeros=0),
        quality=unit_quality(expected=100, responses=responses),
    )
    assert d.action == RecoveryAction.NOOP and task["status"] == "UNVERIFIED"
    [health] = [w for w in task["warnings"] if w.startswith("response/trial health")]
    health += " " + ((task["evaluation_health"] or {}).get("detail") or "")
    # Whatever the reported response population, it never decides health: only the unobserved trials do.
    assert health.startswith("response/trial health unknown") and "response:" not in health, state
    assert not any("response:" in w for w in task["warnings"])


def test_the_trial_gate_is_not_evaluated_on_an_incomplete_run(tmp_path) -> None:
    d, task = gate_of(gate_job(tmp_path, expected=100, final_zeros=0, missing=1))
    assert d.action == RecoveryAction.RESUBMIT and not d.complete
    [health] = [w for w in task["warnings"] if w.startswith("response/trial health")]
    health += " " + ((task["evaluation_health"] or {}).get("detail") or "")
    assert "trial: not evaluated, the run is not complete" in health


def test_the_trial_boundary_names_the_run_provenance_and_the_open_protocol_question(tmp_path) -> None:
    job = gate_job(tmp_path, expected=100, final_zeros=1)
    quality = unit_quality(
        expected=100,
        provenance={
            "resolved_config": "read",
            "route_failures_to_sidecar": None,
            "servers": {
                "a": {
                    "kind": "responses_api_agents",
                    "implementation": "scicode_agent",
                    "resources_server": "r",
                    "skip_verification": None,
                }
            },
        },
        ended_on_failed_model_call=0,
        verification_skipped={"true": 0, "false": 0, "absent": 100},
    )
    d, task = gate_of(job, quality=quality)
    [health] = [w for w in task["warnings"] if w.startswith("response/trial health")]
    health += " " + ((task["evaluation_health"] or {}).get("detail") or "")
    assert d.action == RecoveryAction.NOOP and health.startswith("response/trial health unknown")
    assert "trial: UNOBSERVED 99 of 100 expected trials" in health and "route_failures_to_sidecar None" in health
    assert "('scicode_agent', 'r', None)" in health and "rollouts ended on a failed model call at least 0" in health
    assert "open for the benchmark PIC" in health
    assert "verification_skipped {'true': 0, 'false': 0, 'absent': 100}" in health
    assert "in_session) runtime observations are absent or malformed, so it is unknown" in health


@pytest.mark.parametrize(
    "quality",
    [
        {"provenance": {"resolved_config": "read", "servers": [{"bad": True}]}},
        {"provenance": {"resolved_config": "read", "servers": "bad"}},
        {"provenance": {"resolved_config": "read", "servers": {"a": "bad"}}},
        {"provenance": {"resolved_config": True}},
        {"provenance": "bad", "ended_on_failed_model_call": True, "verification_skipped": {"true": -1}},
        {"ended_on_failed_model_call": -1, "verification_skipped": "bad"},
        "bad",
    ],
)
def test_a_malformed_persisted_trial_fact_is_named_never_read_or_raised(quality) -> None:
    from nemo_gym.orchestration.recovery import trial_boundary

    text = trial_boundary(quality)
    assert "malformed" in text or "not reported" in text
    assert "failed model call unknown" in text or "at least" not in text


@pytest.mark.parametrize(
    ("counts", "reported"),
    [
        ({"true": 1, "false": 99, "absent": 0}, "{'true': 1, 'false': 99, 'absent': 0}"),
        ({"true": 0, "false": 100, "absent": 0}, "{'true': 0, 'false': 100, 'absent': 0}"),
        ({"true": True, "false": 99, "absent": 0}, "not reported or malformed"),
        ({"true": -1, "false": 99, "absent": 0}, "not reported or malformed"),
        (None, "not reported or malformed"),
    ],
)
def test_a_context_window_signal_is_reported_and_stays_unknown_without_changing_the_score(
    tmp_path, counts, reported
) -> None:
    job = gate_job(tmp_path, expected=100, final_zeros=0)
    d, task = gate_of(job, quality=unit_quality(expected=100, context_window_exceeded=counts))
    [health] = [w for w in task["warnings"] if w.startswith("response/trial health")]
    health += " " + ((task["evaluation_health"] or {}).get("detail") or "")
    assert d.action == RecoveryAction.NOOP and task["score"] == 0.5 and task["status"] == "UNVERIFIED"
    # A context-window signal is never read as proven clipping: the trial gate stays unknown, not failed.
    assert health.startswith("response/trial health unknown")
    assert f"records context_window_exceeded {reported} (a signal" in health and "not proof" in health


@pytest.mark.parametrize(
    ("configured", "message"),
    [
        (True, "although the saved config snapshot names a tracking URI, experiment and run name"),
        (False, "does not name all of the tracking URI, experiment and run name; exporter activation and whether"),
        (None, "no MLflow run evidence: no live run id and no post-run export receipt"),
        ("yes", "no MLflow run evidence: no live run id and no post-run export receipt"),
    ],
)
def test_missing_mlflow_evidence_names_the_configuration_as_a_pending_fact(tmp_path, configured, message) -> None:
    job = gate_job(tmp_path, expected=100, final_zeros=0)
    (job / "artifacts" / "rollouts_mlflow_export_done").unlink()
    from nemo_gym.orchestration.recovery import gate_summary_from_collected

    collected = collect_with_score(job, json.dumps(["agent", "mean/reward"]))
    collected.quality = unit_quality(
        expected=100, provenance={"resolved_config": "read", "servers": {}, "mlflow": {"configured": configured}}
    )
    d = decide_from_collected(collected, max_infra_retries=3, max_walltime_s=None, live_job=False)
    task = gate_summary_from_collected(d, collected, max_unhealthy_fraction=0.02, mlflow_status=None)
    delivery = task["mlflow_delivery"]
    assert d.action == RecoveryAction.NOOP and not [e for e in task["errors"] if "MLflow" in e]
    assert delivery["state"] == "pending" and len(delivery["reasons"]) == 1
    assert message in delivery["reasons"][0]


@pytest.mark.parametrize(
    ("observed_run_id", "message"),
    [
        (None, "MLflow observed status names no run, so it is not bound to the receipt's run postrun-run"),
        ("  ", "MLflow observed status names no run, so it is not bound to the receipt's run postrun-run"),
        ("other-run", "MLflow observed status is for run other-run, not the receipt's run postrun-run"),
    ],
)
def test_a_finished_status_counts_only_for_the_receipt_s_own_run(observed_run_id, message) -> None:
    from nemo_gym.orchestration.recovery import mlflow_errors

    local = {
        "live_run_id": None,
        "receipt": {
            "run_id": "postrun-run",
            "current": True,
            "uploaded_rollouts": True,
            "rollout_projection_version": 1,
        },
    }
    assert mlflow_errors(local, "FINISHED", observed_run_id=observed_run_id) == [message]


@pytest.mark.parametrize(
    ("observed_run_id", "status", "accepted"),
    [
        ("postrun-run", "FINISHED", True),
        # A current post-run export receipt is the run of record beside a live run (both exist, either is named).
        ("live-old", "FINISHED", True),
        (None, "FINISHED", True),
        ("other", "FINISHED", False),
        ("postrun-run", "FAILED", False),
    ],
)
def test_a_current_post_run_receipt_beside_an_earlier_live_run_is_the_run_of_record(
    observed_run_id, status, accepted
) -> None:
    from nemo_gym.orchestration.recovery import mlflow_errors

    local = {
        "live_run_id": "live-old",
        "receipt": {
            "run_id": "postrun-run",
            "current": True,
            "uploaded_rollouts": True,
            "rollout_projection_version": 1,
        },
    }
    errors = mlflow_errors(local, status, observed_run_id=observed_run_id)
    assert (errors == []) is accepted
    stale = {"live_run_id": "live-old", "receipt": {"run_id": "postrun-run", "current": False}}
    assert mlflow_errors(stale, "FINISHED", observed_run_id="postrun-run")


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


@pytest.mark.parametrize(
    ("scope", "message"),
    [
        (False, "records that the rollouts were not uploaded"),
        (None, "does not record whether the rollouts were uploaded"),
        ("yes", "does not record whether the rollouts were uploaded"),
    ],
)
def test_a_receipt_without_an_uploaded_rollouts_record_is_not_delivery(scope, message) -> None:
    from nemo_gym.orchestration.recovery import mlflow_errors

    receipt = {"run_id": "postrun-run", "current": True, "uploaded_rollouts": scope}
    errors = mlflow_errors({"live_run_id": None, "receipt": receipt}, "FINISHED", observed_run_id="postrun-run")
    assert len(errors) == 1 and message in errors[0]


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


def test_review_absent_results_stops_with_inventory_and_final_zero_preserved(tmp_path):
    job = write_job_dir(
        tmp_path / "job",
        n=2,
        results=[],
        failures=[
            ident(0, failure_kind="model_context_limit", _ng_failure_terminal=True, _ng_failure_counts_as_zero=True)
        ],
    )
    (job / "artifacts" / "rollouts.jsonl").unlink()
    out = decide_from_collected(collect(job), max_infra_retries=3, max_walltime_s=None, live_job=False)
    assert out.action == RecoveryAction.STOP and out.expected == 2
    assert out.identities["model_final"] == [(0, 0)]


@pytest.mark.parametrize("stored", [None, 10])
def test_review_last_allocation_walltime_cannot_be_assumed_accounted(tmp_path, stored):
    job = write_job_dir(tmp_path / "job", n=1, results=[], failures=[], chain=["101"], walltime=stored)
    out = decide_from_collected(collect(job), max_infra_retries=3, max_walltime_s=100, live_job=False)
    assert out.action == RecoveryAction.STOP


@pytest.mark.parametrize("name", ["rollouts.jsonl", "rollouts_failures.jsonl"])
@pytest.mark.parametrize("tail", [b'{"_ng_task_index":', b'{"x":"\xe2\x82'])
def test_review_truncated_last_row_is_read_only(tmp_path, name, tail):
    job = write_job_dir(tmp_path / "job", n=2, results=[ident(0, reward=0.0)], failures=[ident(1, **INFRA)])
    path = job / "artifacts" / name
    path.write_bytes(path.read_bytes() + tail)
    before = path.read_bytes()
    got = collect(job)
    assert got.results == [ident(0, reward=0.0)]
    assert len(got.failures) == 1
    assert path.read_bytes() == before


@pytest.mark.parametrize("kind", [None, "unregistered", "provider_unavailable"])
def test_review_unknown_or_provider_zero_is_not_model_final(tmp_path, kind):
    job = write_job_dir(
        tmp_path / "job",
        n=1,
        results=[],
        failures=[ident(0, failure_kind=kind, _ng_failure_terminal=True, _ng_failure_counts_as_zero=True)],
    )
    out = decide_from_collected(collect(job), max_infra_retries=3, max_walltime_s=None, live_job=False)
    assert out.action == RecoveryAction.STOP and out.identities.get("model_final") is None


def test_review_unsupported_omitted_sidecar_is_not_missing_or_zero(tmp_path):
    job = write_job_dir(
        tmp_path / "job",
        n=1,
        results=[],
        failures=[
            ident(
                0,
                failure_kind="model_context_limit",
                _ng_omitted_rollout=True,
                _ng_failure_terminal=True,
                _ng_failure_counts_as_zero=True,
            )
        ],
    )
    out = decide_from_collected(collect(job), max_infra_retries=3, max_walltime_s=None, live_job=False)
    assert out.action == RecoveryAction.STOP and out.identities == {"unknown": [(0, 0)]}


@pytest.mark.parametrize("fraction", [float("nan"), float("inf"), True, -0.1, 1.1])
def test_review_invalid_criterion_cannot_dispatch(tmp_path, fake_executor, fraction):
    fake_executor.job_dir = write_job_dir(tmp_path / "job", n=1, results=[], failures=[])
    with pytest.raises(ValueError, match="criterion"):
        submit.recover("review", "bench", max_unhealthy_fraction=fraction)
    assert fake_executor.calls == []


def test_review_boolean_expected_count_is_not_one(tmp_path):
    from nemo_gym.orchestration.recovery import quality_inputs

    job = write_job_dir(tmp_path / "job", n=1, results=[ident(0, reward=1.0)], failures=[])
    got = collect(job)
    got.quality = unit_quality(expected=True, verdicts={"healthy": 1, "unhealthy": 0, "unobserved": 0})
    decision = decide_from_collected(got, max_infra_retries=3, max_walltime_s=None, live_job=False)
    assert quality_inputs(decision, got, max_unhealthy_fraction=0.02)[0] is not True


def test_review_unsupported_reverify_reports_failed_not_resuming(tmp_path, fake_executor):
    fake_executor.job_dir = write_job_dir(
        tmp_path / "job", n=1, results=[ident(0, _ng_failure_class="judge_failed", response=SAVED_ANSWER)], failures=[]
    )
    fake_executor.supports_reverify = False
    out = submit.recover("review", "bench")
    assert out.dispatched is None and out.gate_task["status"] == "FAILED"


@pytest.mark.parametrize("sources", [["other.jsonl"], ["rollouts.jsonl", "other.jsonl"], None])
def test_review_quality_from_another_rollout_is_not_accepted(tmp_path, sources):
    from nemo_gym.orchestration.recovery import quality_inputs

    job = write_job_dir(tmp_path / "job", n=1, results=[ident(0, reward=1.0)], failures=[])
    produce_quality(job)
    path = job / "artifacts" / "quality_summary.json"
    payload = json.loads(path.read_text())
    payload["run"]["rollouts"] = sources
    path.write_text(json.dumps(payload))
    got = collect(job)
    decision = decide_from_collected(got, max_infra_retries=3, max_walltime_s=None, live_job=False)
    summary_valid, _, health, _ = quality_inputs(decision, got, max_unhealthy_fraction=0.02)
    assert summary_valid is not True and health is not True


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


@pytest.mark.parametrize(
    "accounting,expected",
    [
        ("101|COMPLETED|10\n102|FAILED|20\n", 30.0),
        ("101|CANCELLED by 12|10\n102|TIMEOUT|20\n", 30.0),
        ("101|COMPLETED|10\n", None),
        ("101|COMPLETED|10\n102|RUNNING|20\n", None),
        ("101|COMPLETED|10\n102|FAILED|-1\n", None),
        ("101|COMPLETED|10\n101|FAILED|20\n", None),
        ("101|COMPLETED|10\n103|FAILED|20\n", None),
        ("101||10\n102|FAILED|20\n", None),
        ("", None),
    ],
)
def test_review_walltime_requires_all_terminal_allocations(accounting, expected):
    from nemo_gym.orchestration.recovery import accounted_chain_walltime

    assert accounted_chain_walltime(["101", "102"], accounting, stored_walltime_s=10) == expected


@pytest.mark.parametrize("elapsed,action", [(99, "resubmit"), (100, "stop"), (101, "stop")])
def test_review_verified_last_allocation_debits_the_same_cap(tmp_path, elapsed, action):
    job = write_job_dir(tmp_path / "job", n=1, results=[], failures=[], chain=["101"], walltime=10)
    got = collect(job).model_copy(update={"accounted_walltime_s": elapsed})
    before = (job / ".gym_accumulated_walltime").read_bytes()
    out = decide_from_collected(got, max_infra_retries=3, max_walltime_s=100, live_job=False)
    assert out.action == action and (job / ".gym_accumulated_walltime").read_bytes() == before


@pytest.mark.parametrize(
    "name,contents",
    [
        ("rollouts.jsonl", b'{"broken":\n'),
        ("rollouts.jsonl", b'{"broken":\n{"_ng_task_index":0,"_ng_rollout_index":0,"reward":1}\n'),
        ("rollouts_materialized_inputs.jsonl", b'{"broken":'),
    ],
)
def test_review_corruption_outside_an_unterminated_result_tail_is_rejected(tmp_path, name, contents):
    job = write_job_dir(tmp_path / "job", n=1, results=[ident(0, reward=1.0)], failures=[])
    path = job / "artifacts" / name
    path.write_bytes(contents)
    with pytest.raises(subprocess.CalledProcessError):
        collect(job)
    assert path.read_bytes() == contents


def test_review_valid_unterminated_row_is_retained(tmp_path):
    job = write_job_dir(tmp_path / "job", n=1, results=[ident(0, reward=0.0)], failures=[])
    path = job / "artifacts" / "rollouts.jsonl"
    path.write_bytes(path.read_bytes().rstrip(b"\n"))
    assert collect(job).results == [ident(0, reward=0.0)]


@pytest.mark.parametrize("boundary", ["judge", "provider", "infrastructure"])
def test_review_contradictory_zero_cause_never_settles_model_failure(tmp_path, boundary):
    row = ident(
        0,
        failure_kind="model_transport_error",
        _ng_failure_terminal=True,
        _ng_failure_counts_as_zero=True,
        _ng_failure_events={"cause": {"boundary": boundary}},
    )
    job = write_job_dir(tmp_path / "job", n=1, results=[], failures=[row])
    out = decide_from_collected(collect(job), max_infra_retries=3, max_walltime_s=None, live_job=False)
    assert out.action == "stop" and "model_final" not in out.identities


def test_review_an_ambiguous_manual_reset_epoch_stays_unknown():
    from nemo_gym.orchestration.recovery import accounted_chain_walltime

    accounting = "101|COMPLETED|10\n102|FAILED|20\n"
    assert accounted_chain_walltime(["101", "102"], accounting, stored_walltime_s=0) is None
    assert accounted_chain_walltime(["101", "102"], accounting, stored_walltime_s=30) == 30


KILLED = dict(_ng_omitted_rollout=True, _ng_failure_class="kill_shaped")


def test_review_interrupted_observations_do_not_consume_retry_allowance(tmp_path):
    job = write_job_dir(
        tmp_path / "job", n=1, results=[], failures=[ident(0, **KILLED), ident(0, **KILLED), ident(0, **INFRA)]
    )
    out = decide_from_collected(collect(job), max_infra_retries=3, max_walltime_s=None, live_job=False)
    art = job / "artifacts"
    config = RolloutCollectionConfig(
        input_jsonl_fpath=str(art / "rollouts_materialized_inputs.jsonl"),
        output_jsonl_fpath=str(art / "rollouts.jsonl"),
    )
    (retried,), *_ = RolloutCollectionHelper()._load_from_cache(config)
    # Collection retries the task as its second attempt; recovery agrees that one attempt of three is spent.
    assert retried["_ng_attempt_index"] == 1
    assert out.identities.get("exhausted") is None
    assert out.identities["retryable"] == [(0, 0)]


FINAL_TIMEOUT_ZERO = dict(
    failure_kind="model_policy_response_timeout",
    _ng_failure_terminal=True,
    _ng_failure_counts_as_zero=True,
    _ng_failure_id="final",
)
POLICY_REQUEST = {"boundary": "agent", "operation": "agent.policy_request", "operation_id": None, "attempt_id": None}
RECOVERED_VERIFY = {"boundary": "resources_server", "operation": "verify", "operation_id": "v", "attempt_id": None}


def _final_zero(events: dict) -> dict:
    return ident(0, **FINAL_TIMEOUT_ZERO, _ng_failure_events=events)


@pytest.mark.parametrize(
    "events",
    [
        {"final": {**POLICY_REQUEST, "cause_id": None}},
        {"final": {**POLICY_REQUEST, "cause_id": None}, "other": {**RECOVERED_VERIFY, "cause_id": None}},
    ],
    ids=["final-only", "with-unrelated-recovered-event"],
)
def test_review_unrelated_recovered_event_keeps_a_settled_final_zero(tmp_path, events):
    job = write_job_dir(tmp_path / "job", n=1, results=[], failures=[_final_zero(events)])
    out = decide_from_collected(collect(job), max_infra_retries=3, max_walltime_s=None, live_job=False)
    assert out.action == RecoveryAction.NOOP
    assert out.identities["model_final"] == [(0, 0)]


@pytest.mark.parametrize(
    "events,failure_id",
    [
        # The final failure's own cause chain reaches a resources server event.
        ({"final": {**POLICY_REQUEST, "cause_id": "other"}, "other": {**RECOVERED_VERIFY, "cause_id": None}}, "final"),
        # The failure_id names no saved event, so every saved event is checked, as before.
        ({"x": {**POLICY_REQUEST, "cause_id": None}, "other": {**RECOVERED_VERIFY, "cause_id": None}}, "final"),
        # A cause_id names no saved event: the chain cannot be checked.
        ({"final": {**POLICY_REQUEST, "cause_id": "gone"}}, "final"),
    ],
    ids=["cause-chain-contradicts", "unresolved-failure-id", "broken-cause-chain"],
)
def test_review_final_zero_whose_cause_chain_contradicts_or_is_unresolved_stops(tmp_path, events, failure_id):
    row = {**_final_zero(events), "_ng_failure_id": failure_id}
    job = write_job_dir(tmp_path / "job", n=1, results=[], failures=[row])
    out = decide_from_collected(collect(job), max_infra_retries=3, max_walltime_s=None, live_job=False)
    assert out.action == RecoveryAction.STOP
    assert out.identities.get("model_final") is None


STRICT_DEFAULT = {
    "response": {"max_failure_fraction": 0.02, "passes": "strictly below, before rounding"},
    "trial": {"max_failure_fraction": 0.02, "passes": "strictly below, before rounding"},
}


def test_review_gate_reports_its_effective_named_criteria(tmp_path):
    _, task = gate_of(gate_job(tmp_path, expected=100, final_zeros=0))
    assert task["criteria"] == {
        "evaluation": {
            "max_failure_fraction": 0.02,
            "passes": "strictly below, before rounding",
            "policy_timeout_counts_as_model": True,
            "classification": RecoveryOutcomePolicyConfig().classification.model_dump(mode="json"),
        },
        "response": {"max_failure_fraction": 0.02, "applied": False},
        "trial": {"max_failure_fraction": 0.02, "applied": False},
    }


def test_review_trial_criterion_is_configurable_without_changing_decision_or_rows(tmp_path):
    from nemo_gym.orchestration.recovery import gate_summary_from_collected

    job = gate_job(tmp_path, expected=100, final_zeros=1)
    rollouts = (job / "artifacts" / "rollouts.jsonl").read_bytes()
    collected = collect_with_score(job, json.dumps(["agent", "mean/reward"]))
    d = decide_from_collected(collected, max_infra_retries=3, max_walltime_s=None, live_job=False)
    tasks = {
        limit: gate_summary_from_collected(
            d,
            collected,
            max_unhealthy_fraction=0.02,
            mlflow_status="FINISHED",
            mlflow_run_id="postrun-run",
            criteria=GateCriteria(max_trial_failure_fraction=limit),
        )
        for limit in (0.02, 0.01)
    }
    health = {
        limit: [w for w in t["warnings"] if w.startswith("response/trial health")][0] for limit, t in tasks.items()
    }
    # The configured fraction is accepted and reported, never applied: 1 of 100 stays unverified under both.
    assert health[0.02] == health[0.01] and health[0.01].startswith("response/trial health unknown")
    assert tasks[0.01]["criteria"]["trial"] == {"max_failure_fraction": 0.01, "applied": False}
    assert d.action == RecoveryAction.NOOP and (job / "artifacts" / "rollouts.jsonl").read_bytes() == rollouts
    assert tasks[0.02]["score"] == tasks[0.01]["score"] == 0.5


@pytest.mark.parametrize("value", [True, 0, -0.01, 1.5, float("nan"), float("inf"), "0.02"])
@pytest.mark.parametrize("name", ["max_response_failure_fraction", "max_trial_failure_fraction"])
def test_review_invalid_gate_criteria_fail_closed(tmp_path, fake_executor, name, value):
    with pytest.raises(ValueError):
        GateCriteria(**{name: value})
    fake_executor.job_dir = write_job_dir(tmp_path / "job", n=1, results=[ident(0, reward=1.0)], failures=[])
    with pytest.raises(ValueError, match="invalid gate criteria"):
        submit.recover("gym-job-criteria", "bench", **{name: value})
    assert fake_executor.calls == []


def test_review_retryable_failures_are_not_labelled_infrastructure(tmp_path):
    job = write_job_dir(tmp_path / "job", n=2, results=[ident(0, reward=1.0)], failures=[ident(1, **INFRA)])
    d = decide_from_collected(collect(job), max_infra_retries=3, max_walltime_s=None, live_job=False)
    task = gate_task_summary(
        d, score=None, summary_valid=None, summary_reasons=[], health_passed=None, health_reasons=[]
    )
    assert any("retryable failures awaiting another attempt" in e for e in task["errors"])
    assert not any("infra" in e for e in task["errors"])


STOP_PROVIDER = RecoveryOutcomePolicyConfig(by_failure_kind={"provider_unavailable": "fail"})


def _decide(job, policy=None):
    return decide_from_collected(
        collect(job), max_infra_retries=3, max_walltime_s=None, live_job=False, outcome_policy=policy
    )


def test_review_same_eligible_failure_retries_or_stops_ungraded_by_configuration(tmp_path):
    job = write_job_dir(tmp_path / "job", n=2, results=[ident(0, reward=1.0)], failures=[ident(1, **INFRA)])
    before = {p.name: p.read_bytes() for p in (job / "artifacts").iterdir()}
    default, configured = _decide(job), _decide(job, STOP_PROVIDER)
    assert default.identities["retryable"] == [(1, 0)] and default.action == RecoveryAction.RESUBMIT
    assert configured.identities["terminal"] == [(1, 0)] and configured.action == RecoveryAction.STOP
    assert configured.identities["scored"] == default.identities["scored"] == [(0, 0)]
    assert {p.name: p.read_bytes() for p in (job / "artifacts").iterdir()} == before


def test_review_status_code_rule_precedes_the_kind_rule(tmp_path):
    row = ident(0, **INFRA, _ng_failure_http_status=503)
    job = write_job_dir(tmp_path / "job", n=1, results=[], failures=[row])
    policy = RecoveryOutcomePolicyConfig(
        by_failure_kind={"provider_unavailable": "retry"}, by_status_code={503: "fail"}
    )
    assert _decide(job, policy).identities["terminal"] == [(0, 0)]


@pytest.mark.parametrize(
    "fields",
    [
        {"default_action": "count_as_zero"},
        {"by_failure_kind": {"judge_failed": "count_as_zero"}},
        {"by_status_code": {503: "count_as_zero"}},
        {"by_failure_kind": {"not_a_registered_kind": "fail"}},
        {"by_status_code": {200: "fail"}},
    ],
)
def test_review_unsupported_recovery_policy_is_refused(fields):
    with pytest.raises(ValueError):
        RecoveryOutcomePolicyConfig(**fields)


def test_review_policy_never_reopens_settled_or_missing_outcomes(tmp_path):
    job = write_job_dir(tmp_path / "job", n=3, results=[ident(0, reward=0.0)], failures=[final_model_zero(1, 0)])
    fail_everything = RecoveryOutcomePolicyConfig(
        by_failure_kind={"model_transport_error": "fail"}, default_action="fail"
    )
    for policy in (None, fail_everything, RecoveryOutcomePolicyConfig(default_action="verify")):
        d = _decide(job, policy)
        assert d.identities["scored"] == [(0, 0)] and d.identities["model_final"] == [(1, 0)]
        assert d.identities["missing"] == [(2, 0)]
        [missing] = [e for e in d.evidence if e.identity == (2, 0)]
        assert missing.events == [] and missing.failure_kind is None and missing.failure_id is None


def _judge_failure(task: int, **extra) -> dict:
    return ident(task, _ng_failure_class="judge_failed", failure_kind="judge_failed", **extra)


@pytest.mark.parametrize("where", ["results", "failures"])
def test_review_same_saved_answer_judge_failure_verifies_or_fails_by_configuration(tmp_path, where):
    row = _judge_failure(1, response=SAVED_ANSWER, **({"reward": 0.0} if where == "results" else {}))
    results = [ident(0, reward=1.0)] + ([row] if where == "results" else [])
    job = write_job_dir(tmp_path / "job", n=2, results=results, failures=[row] if where == "failures" else [])
    before = {p.name: p.read_bytes() for p in (job / "artifacts").iterdir()}
    verify = _decide(job)
    fail = _decide(job, RecoveryOutcomePolicyConfig(by_failure_kind={"judge_failed": "fail"}))
    assert verify.identities["judge_recoverable"] == [(1, 0)] and verify.action == RecoveryAction.REVERIFY
    assert fail.identities["terminal"] == [(1, 0)] and fail.action == RecoveryAction.STOP
    assert fail.identities.get("judge_recoverable") is None
    assert {p.name: p.read_bytes() for p in (job / "artifacts").iterdir()} == before


def test_multistage_judge_failure_resumes_instead_of_reverifying(tmp_path):
    """`reverify --judge-failed-only` refuses multi-stage rows, so recovery must resume the collection for them."""
    results = [ident(0, reward=1.0)]
    failure = _judge_failure(1, response=SAVED_ANSWER, stage_index=1)
    job = write_job_dir(tmp_path / "job", n=2, results=results, failures=[failure])
    d = _decide(job)
    assert d.action == RecoveryAction.RESUBMIT, d.reason
    assert d.identities["retryable"] == [(1, 0)]
    assert d.identities.get("judge_recoverable") is None
    # A single-stage judge failure keeps the judge-only path.
    plain = write_job_dir(
        tmp_path / "plain", n=2, results=results, failures=[_judge_failure(1, response=SAVED_ANSWER)]
    )
    assert _decide(plain).action == RecoveryAction.REVERIFY


def test_review_verify_without_a_saved_answer_stops(tmp_path):
    job = write_job_dir(tmp_path / "job", n=1, results=[], failures=[_judge_failure(0)])
    d = _decide(job)
    assert d.action == RecoveryAction.STOP and d.identities["unknown"] == [(0, 0)]
    # A failure with no re-verification path cannot be verified either.
    job = write_job_dir(tmp_path / "job2", n=1, results=[], failures=[ident(0, **INFRA)])
    d = _decide(job, RecoveryOutcomePolicyConfig(by_failure_kind={"provider_unavailable": "verify"}))
    assert d.action == RecoveryAction.STOP and d.identities["unknown"] == [(0, 0)]


def test_review_policy_selects_on_the_failure_id_event_then_the_row_fields(tmp_path):
    selected = {
        "boundary": "model",
        "operation": "model_request",
        "failure_kind": "model_request_error",
        "http_status": 429,
        "operation_id": "op",
        "attempt_id": "op-1",
        "cause_id": None,
    }
    other = {**selected, "failure_kind": "provider_unavailable", "http_status": 503}
    row = ident(
        0,
        **{
            **INFRA,
            "_ng_failure_id": "sel",
            "_ng_failure_events": {"sel": selected, "x": other},
            "_ng_failure_http_status": 500,
        },
    )
    job = write_job_dir(tmp_path / "job", n=1, results=[], failures=[row])
    # The selected event's 429 wins over the row's 500; the unrelated event's 503 is never searched.
    assert _decide(job, RecoveryOutcomePolicyConfig(by_status_code={429: "fail"})).identities["terminal"] == [(0, 0)]
    assert _decide(job, RecoveryOutcomePolicyConfig(by_status_code={503: "fail"})).identities["retryable"] == [(0, 0)]
    assert _decide(job, RecoveryOutcomePolicyConfig(by_status_code={500: "fail"})).identities["retryable"] == [(0, 0)]
    # Without a selected event, the producer's own row status applies.
    plain = write_job_dir(
        tmp_path / "plain", n=1, results=[], failures=[ident(0, **INFRA, _ng_failure_http_status=500)]
    )
    assert _decide(plain, RecoveryOutcomePolicyConfig(by_status_code={500: "fail"})).identities["terminal"] == [(0, 0)]


def test_review_collector_keeps_saved_answer_presence_not_bytes(tmp_path):
    job = write_job_dir(
        tmp_path / "job", n=2, results=[], failures=[_judge_failure(0, response=SAVED_ANSWER), _judge_failure(1)]
    )
    got = collect(job)
    from nemo_gym.orchestration.recovery import _saved_answer

    assert [_saved_answer(row) for row in got.failures] == [True, False]
    assert got.failures[0]["response"]["id"] == SAVED_ANSWER["id"]
    assert "content" not in got.failures[0]["response"]["output"][0]
    assert "response" not in got.failures[1]
    assert '"text"' not in got.model_dump_json()


def test_review_retry_policy_cannot_exceed_spent_attempts(tmp_path):
    job = write_job_dir(tmp_path / "job", n=1, results=[], failures=[ident(0, **INFRA)] * 3)
    d = _decide(job, RecoveryOutcomePolicyConfig(default_action="retry"))
    assert d.identities["exhausted"] == [(0, 0)] and d.identities.get("retryable") is None


def test_review_recover_passes_the_outcome_policy_to_the_decision(tmp_path, fake_executor):
    fake_executor.job_dir = write_job_dir(
        tmp_path / "job", n=2, results=[ident(0, reward=1.0)], failures=[ident(1, **INFRA)]
    )
    out = submit.recover("gym-job-policy", "bench", outcome_policy=STOP_PROVIDER)
    assert out.decision.identities["terminal"] == [(1, 0)] and out.decision.action == RecoveryAction.STOP
    assert out.dispatched is None


@pytest.mark.parametrize("other", ["missing", "retryable", "judge"])
def test_policy_fail_is_not_replayed_by_mixed_dispatch(tmp_path, other):
    rows = [ident(0, **INFRA)]
    if other == "retryable":
        rows.append(ident(1, **INFRA, _ng_failure_http_status=429))
    elif other == "judge":
        rows.append(_judge_failure(1, response=SAVED_ANSWER))
    job = write_job_dir(tmp_path / "job", n=2, results=[], failures=rows)
    policy = RecoveryOutcomePolicyConfig(
        by_failure_kind={"provider_unavailable": "fail"}, by_status_code={429: "retry"}
    )
    d = _decide(job, policy)
    assert d.action == RecoveryAction.STOP


@pytest.mark.parametrize("where", ["results", "failures"])
def test_terminal_judge_failure_never_reopens_by_policy(tmp_path, where):
    row = _judge_failure(0, response=SAVED_ANSWER, _ng_failure_terminal=True)
    job = write_job_dir(
        tmp_path / "job",
        n=1,
        results=[row] if where == "results" else [],
        failures=[row] if where == "failures" else [],
    )
    d = _decide(job)
    assert d.action == RecoveryAction.STOP and d.identities["terminal"] == [(0, 0)]


@pytest.mark.parametrize("events", [{}, {"unrelated": {**POLICY_REQUEST, "cause_id": None}}])
def test_explicit_missing_final_failure_id_is_not_validated_from_unrelated_observations(tmp_path, events):
    job = write_job_dir(tmp_path / "job", n=1, results=[], failures=[_final_zero(events)])
    assert _decide(job).action == RecoveryAction.STOP


@pytest.mark.parametrize("provenance", ["read", ["read"], 1, {"resolved_config": "read", "servers": []}])
def test_mcqa_trial_operations_with_unsupported_provenance_are_unproven_not_raised(provenance) -> None:
    from nemo_gym.orchestration import recovery

    row = {recovery.TASK_INDEX: 0, recovery.ROLLOUT_INDEX: 0}
    decision = recovery.RecoveryDecision(
        action=recovery.RecoveryAction.NOOP,
        reason="scored",
        expected=1,
        counts={recovery.IdentityState.SCORED.value: 1},
        identities={recovery.IdentityState.SCORED.value: [(0, 0)]},
        evidence=[],
    )
    collected = recovery.CollectedEvidence(
        materialized=[row], results=[row], failures=[], quality={"source_matches": True, "provenance": provenance}
    )
    assert recovery._observed_trial_operations(decision, collected) is False


@pytest.mark.parametrize(
    "case",
    [
        "verifier_saved",
        "verifier_missing",
        "verifier_malformed",
        "infra_before_answer",
        "infra_unselected_verifier",
        "infra_unselected_model",
        "model_operation_budget_unknown",
    ],
)
def test_collector_projection_preserves_verification_stop_without_regenerating(tmp_path, case):
    from nemo_gym.core.failure_records import FailureBoundary
    from nemo_gym.core.failure_transport import FailureReportRow
    from nemo_gym.core.typed_exceptions import NgFailureEvent
    from nemo_gym.rollouts.rollout_collection import _agent_request_failure_row, _collection_failure_report

    verifier = NgFailureEvent(
        boundary=FailureBoundary.RESOURCES_SERVER,
        failure_kind="verifier_error",
        failure_reason="verification failed",
        operation="verify",
        operation_id="verify-1",
    )
    if case.startswith("verifier"):
        failure = NgFailureEvent(
            boundary=FailureBoundary.AGENT,
            failure_kind="agent_route_failed",
            failure_reason="agent verification failed",
            operation="agent.run",
        )
        failure.__cause__ = verifier
        status = 500
    elif case == "model_operation_budget_unknown":
        failure = NgFailureEvent(
            boundary=FailureBoundary.MODEL,
            failure_kind="model_request_error",
            failure_reason="model request failed",
            operation="model_request",
            operation_id="model-operation-1",
            attempt_id="model-attempt-2",
            http_status=503,
        )
        status = 500
    else:
        failure, status = TimeoutError("dispatch timed out"), None
    row = _agent_request_failure_row(failure, status)
    row = FailureReportRow.merged_into(row, _collection_failure_report(failure, row["_ng_failure_class"], {}))
    row.update(ident(0, _ng_attempt_index=0, _ng_failure_max_attempts=3))
    if case == "verifier_saved":
        row["ng_trajectory"] = {
            "invocations": [
                {
                    "invocation_id": "root",
                    "status": "completed",
                    "conversation": [{"role": "assistant", "content": "retained-secret-answer"}],
                }
            ],
            "turns": [
                {
                    "invocation_id": "root",
                    "answer": [
                        {
                            "role": "assistant",
                            "status": "completed",
                            "content": [{"type": "output_text", "text": "retained-secret-answer"}],
                        }
                    ],
                }
            ],
        }
    elif case == "verifier_malformed":
        row["ng_trajectory"] = "unreadable"
    elif case == "infra_unselected_verifier":
        # Recovered observations elsewhere in the report cannot replace the selected dispatch cause.
        row["_ng_failure_events"]["observed-verifier"] = {
            "event_id": "observed-verifier",
            "boundary": "resources_server",
            "operation": "verify",
            "failure_kind": "verifier_error",
        }
    elif case == "infra_unselected_model":
        row["_ng_failure_events"]["observed-model"] = {
            "event_id": "observed-model",
            "boundary": "model",
            "operation": "model_request",
            "failure_kind": "model_request_error",
            "operation_id": "recovered-operation",
        }
    job = write_job_dir(tmp_path / "job", n=1, results=[], failures=[row], retries=0, walltime=0)
    before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (job / "artifacts").iterdir()}
    collected = collect(job)
    projected = collected.failures[0]
    assert projected["_ng_failure_id"] == row["_ng_failure_id"]
    assert projected["_ng_failure_events"] == row["_ng_failure_events"]
    assert "retained-secret-answer" not in collected.model_dump_json()
    assert "ng_trajectory" not in projected and "response" not in projected and "reward" not in projected
    decision = decide_from_collected(collected, max_infra_retries=3, max_walltime_s=3600, live_job=False)
    assert decision.action == (
        RecoveryAction.STOP
        if case.startswith("verifier") or case == "model_operation_budget_unknown"
        else RecoveryAction.RESUBMIT
    )
    assert decision.evidence[0].attempts == 1 and decision.evidence[0].max_attempts == 3
    assert before == {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (job / "artifacts").iterdir()}


@pytest.mark.parametrize("saved", [True, False])
def test_projected_judge_answer_keeps_its_supported_reverification_path(tmp_path, saved):
    row = ident(0, _ng_failure_class="judge_failed", failure_kind="judge_failed", _ng_failure_max_attempts=3)
    if saved:
        row["response"] = SAVED_ANSWER
    job = write_job_dir(tmp_path / "job", n=1, results=[], failures=[row], retries=0, walltime=0)
    collected = collect(job)
    decision = decide_from_collected(collected, max_infra_retries=3, max_walltime_s=3600, live_job=False)
    assert decision.action == (RecoveryAction.REVERIFY if saved else RecoveryAction.STOP)
    from nemo_gym.orchestration.recovery import _saved_answer

    assert _saved_answer(collected.failures[0]) is saved
    assert '"text"' not in collected.model_dump_json()


def test_projected_explicit_final_model_zero_stays_settled(tmp_path):
    row = ident(
        0,
        failure_kind="model_context_limit",
        _ng_failure_counts_as_zero=True,
        _ng_failure_terminal=True,
        _ng_failure_max_attempts=3,
    )
    job = write_job_dir(tmp_path / "job", n=1, results=[], failures=[row], retries=0, walltime=0)
    collected = collect(job)
    decision = decide_from_collected(collected, max_infra_retries=3, max_walltime_s=3600, live_job=False)
    assert decision.action == RecoveryAction.NOOP
    assert decision.counts["model_final"] == 1


@pytest.mark.parametrize(
    "case",
    [
        "missing_cause",
        "self_cycle",
        "two_event_cycle",
        "nonstring_cause",
        "plain_dispatch",
        "legacy_empty",
        "legacy_observations",
        "unselected_broken",
        "dangling_id",
        "list_id",
        "dict_id",
    ],
)
@pytest.mark.parametrize("mode", ["default", "terminal", "explicit_fail"])
def test_projected_broken_selected_chain_cannot_reopen_policy_work(tmp_path, case, mode):
    from nemo_gym.core.failure_transport import FailureReportRow
    from nemo_gym.rollouts.rollout_collection import _agent_request_failure_row, _collection_failure_report

    failure = TimeoutError("dispatch did not reach the agent")
    row = _agent_request_failure_row(failure, None)
    row = FailureReportRow.merged_into(row, _collection_failure_report(failure, row["_ng_failure_class"], {}))
    row.update(ident(0, _ng_failure_max_attempts=3))
    selected_id = row["_ng_failure_id"]
    selected = row["_ng_failure_events"][selected_id]
    if case == "missing_cause":
        selected["cause_id"] = "missing"
    elif case == "self_cycle":
        selected["cause_id"] = selected_id
    elif case == "two_event_cycle":
        selected["cause_id"] = "other"
        row["_ng_failure_events"]["other"] = {**selected, "event_id": "other", "cause_id": selected_id}
    elif case == "nonstring_cause":
        selected["cause_id"] = ["invalid"]
    elif case == "legacy_empty":
        del row["_ng_failure_events"], row["_ng_failure_id"]
        row["failure_kind"] = "provider_unavailable"
    elif case == "legacy_observations":
        del row["_ng_failure_id"]
        row["failure_kind"] = "provider_unavailable"
    elif case == "dangling_id":
        row["_ng_failure_events"] = {}
        row["failure_kind"] = "provider_unavailable"
    elif case in {"list_id", "dict_id"}:
        row["_ng_failure_id"] = ["invalid"] if case == "list_id" else {"id": "invalid"}
        row["failure_kind"] = "provider_unavailable"
    elif case == "unselected_broken":
        row["_ng_failure_events"]["other"] = {
            "event_id": "other",
            "boundary": "resources_server",
            "operation": "verify",
            "cause_id": "missing",
        }
    if mode == "terminal":
        row["_ng_failure_terminal"] = True
    # The dispatch kind's own retry entry outranks a failing default, so failing it takes an entry as well.
    policy = (
        RecoveryOutcomePolicyConfig(default_action="fail", by_failure_kind={"collection_dispatch_failed": "fail"})
        if mode == "explicit_fail"
        else RecoveryOutcomePolicyConfig()
    )
    job = write_job_dir(tmp_path / "job", n=1, results=[], failures=[row], retries=0, walltime=0)
    before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (job / "artifacts").iterdir()}
    collected = collect(job)
    assert collected.failures[0].get("_ng_failure_id") == row.get("_ng_failure_id")
    assert collected.failures[0].get("_ng_failure_events") == row.get("_ng_failure_events")
    decision = decide_from_collected(
        collected, max_infra_retries=3, max_walltime_s=3600, live_job=False, outcome_policy=policy
    )
    if mode != "default":
        assert decision.identities["terminal"] == [(0, 0)] and decision.action == RecoveryAction.STOP
    elif case in {"plain_dispatch", "legacy_empty", "legacy_observations", "unselected_broken"}:
        assert decision.identities["retryable"] == [(0, 0)] and decision.action == RecoveryAction.RESUBMIT
    else:
        assert decision.identities["unknown"] == [(0, 0)] and decision.action == RecoveryAction.STOP
    assert decision.evidence[0].attempts == 1 and decision.evidence[0].max_attempts == 3
    assert before == {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (job / "artifacts").iterdir()}


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


def test_output_only_response_is_not_the_typed_producer_contract(tmp_path):
    from pydantic import ValidationError

    output_only = {"output": [{"type": "message", "content": [{"type": "output_text", "text": "synthetic"}]}]}
    with pytest.raises(ValidationError):
        NeMoGymResponse.model_validate(output_only)
    job = write_job_dir(tmp_path / "job", n=1, results=[], failures=[_judge_failure(0, response=output_only)])
    before = {path.name: path.read_bytes() for path in (job / "artifacts").iterdir()}
    decision = _decide(job)
    assert decision.action == RecoveryAction.STOP
    assert decision.identities["unknown"] == [(0, 0)]
    assert {path.name: path.read_bytes() for path in (job / "artifacts").iterdir()} == before


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
    import nemo_gym.orchestration.executors.slurm as slurm

    monkeypatch.setattr(slurm, "_reader_token", None)
    monkeypatch.setenv("GYM_READER_TOKEN", "caller-token")
    fake_executor.job_dir = write_job_dir(tmp_path / "job", n=1, results=[ident(0, reward=1.0)], failures=[])
    code, out = run_cli(["eval", "resubmit", "gym-job-t", "bench", "--recover", "--inspect"], capsys)
    assert out["reader_token"] == "caller-token"


def _graded(task: int) -> dict:
    """A scored row whose verifier recorded one programmatic grading operation bound to its saved answer and score."""
    from nemo_gym.trajectory.trajectory_record import grading_result_binding

    row = ident(
        task,
        _ng_environment_server="environment",
        _ng_result_type="single_agent_turn",
        reward=1.0,
        response={"id": f"resp-{task}", "status": "completed", "output": []},
    )
    operation = {"operation_id": f"grade-{task}", "kind": "programmatic", "outcome": "completed"}
    operation["result_binding_sha256"] = grading_result_binding(row)
    verifier = {"invocation_id": f"verify-{task}", "status": "completed", "grading_operations": [operation]}
    return row | {"ng_trajectory": {"verifier_invocations": [verifier]}}


def _quality_naming(server: str, expected: int) -> dict:
    agent = {"kind": "responses_api_agents", "implementation": "demo_agent", "resources_server": server}
    environment = {
        "kind": "environment_servers",
        "implementation": "single_agent_turn",
        "agent_server": "agent",
        "resources_server": server,
    }
    return unit_quality(
        expected=expected,
        provenance={
            "resolved_config": "read",
            "route_failures_to_sidecar": None,
            "servers": {"environment": environment, "agent": agent, server: {"kind": "resources_servers"}},
        },
    )


def test_a_run_whose_verifier_records_no_grading_is_unobserved_with_a_count_and_a_reason(tmp_path) -> None:
    """Full score and coverage, but nothing observed the grading: unverified, never silent and never healthy."""
    job = gate_job(tmp_path, expected=6, final_zeros=0)
    d, task = gate_of(job, quality=_quality_naming("demo_verifier", 6))

    assert d.action == RecoveryAction.NOOP and d.complete
    assert task["trial_counts"] == {
        "observed_healthy": 0,
        "unhealthy": 0,
        "unobserved": 6,
        "verifier_recorded": 0,
        "unobserved_reason": "no completed grading operation or grading record was recorded by the verifier (demo_verifier)",
    }
    [health] = [w for w in task["warnings"] if w.startswith("response/trial health")]
    health += " " + ((task["evaluation_health"] or {}).get("detail") or "")
    assert health.startswith("response/trial health unknown")
    assert (
        "trial: UNOBSERVED 6 of 6 expected trials: no completed grading operation or grading record was recorded "
        "by the verifier (demo_verifier), so the run is unverified, not healthy; observed healthy 0, unhealthy 0"
        in health
    )
    assert task["status"] == "UNVERIFIED"


def test_a_mixed_run_reports_observed_healthy_unhealthy_and_unobserved_trials(tmp_path) -> None:
    results = [_graded(t) for t in range(49)] + [ident(t, reward=1.0) for t in range(49, 99)]
    job = write_job_dir(tmp_path / "job", n=100, results=results, failures=[final_model_zero(99, 0)])
    (job / "artifacts" / "rollouts_aggregate_metrics.json").write_text(json.dumps({"agent": {"mean/reward": 0.5}}))
    write_postrun_receipt(job)

    _, task = gate_of(job, quality=_quality_naming("demo_verifier", 100))

    assert {key: task["trial_counts"][key] for key in ("observed_healthy", "unhealthy", "unobserved")} == {
        "observed_healthy": 49,
        "unhealthy": 1,
        "unobserved": 50,
    }
    [health] = [w for w in task["warnings"] if w.startswith("response/trial health")]
    health += " " + ((task["evaluation_health"] or {}).get("detail") or "")
    assert "trial: UNOBSERVED 50 of 100 expected trials" in health
    assert "observed healthy 49, unhealthy 1" in health

    # A grading binding alone cannot establish ownership when the environment provenance is missing.
    quality = _quality_naming("demo_verifier", 100)
    del quality["provenance"]["servers"]["environment"]
    _, unowned = gate_of(job, quality=quality)
    assert {key: unowned["trial_counts"][key] for key in ("observed_healthy", "unhealthy", "unobserved")} == {
        "observed_healthy": 0,
        "unhealthy": 1,
        "unobserved": 99,
    }
    [health] = [w for w in unowned["warnings"] if w.startswith("response/trial health")]
    # The warning names the cause; the full trial reason stays in the health detail.
    assert "99 of 100 rows have no recorded grading" in health
    detail = unowned["evaluation_health"]["detail"]
    assert "trial: UNOBSERVED 99 of 100 expected trials" in detail
    assert "observed healthy 0, unhealthy 1" in detail


_CURRENT_RECEIPT = {
    "run_id": "postrun-run",
    "current": True,
    "uploaded_rollouts": True,
    "rollout_projection_version": 1,
}


def test_both_runs_with_a_current_receipt_pass_with_a_note() -> None:
    from nemo_gym.orchestration.recovery import mlflow_errors, mlflow_notes

    local = {"live_run_id": "live-run", "receipt": _CURRENT_RECEIPT}
    assert mlflow_errors(local, "FINISHED", observed_run_id="live-run") == []
    assert mlflow_notes(local) == [
        "MLflow live run live-run and post-run export run postrun-run both exist: the post-run export receipt is "
        "current for the saved artifacts, so it is the run of record"
    ]


def test_both_runs_with_a_stale_receipt_fail_naming_the_stale_one() -> None:
    from nemo_gym.orchestration.recovery import mlflow_errors

    stale = {"live_run_id": "live-run", "receipt": {**_CURRENT_RECEIPT, "current": False}}
    errors = mlflow_errors(stale, "FINISHED", observed_run_id="live-run")
    assert (
        "MLflow live run live-run and a post-run export both exist, and the post-run export run postrun-run is "
        "stale: its receipt is not current for the saved artifacts, so neither is the run of record"
    ) in errors
    assert not any("names neither" in error for error in errors)


def _mlflow_gate(tmp_path: Path, *, tracking_required: bool | None) -> dict:
    """The gate of a complete run that exported nothing to MLflow, with the snapshot's tracking requirement."""
    from nemo_gym.orchestration.recovery import gate_summary_from_collected

    job = gate_job(tmp_path, expected=4, final_zeros=0)
    collected = collect_with_score(job, json.dumps(["agent", "mean/reward"]))
    collected.mlflow = {}
    d = decide_from_collected(collected, max_infra_retries=3, max_walltime_s=None, live_job=False)
    mlflow = {"configured": False, "tracking_required": tracking_required}
    collected.quality = unit_quality(
        expected=d.expected, provenance={"resolved_config": "read", "servers": {}, "mlflow": mlflow}
    )
    return gate_summary_from_collected(d, collected, max_unhealthy_fraction=0.02, mlflow_status=None)


def test_mlflow_not_required_by_design_passes_the_mlflow_part_with_a_note(tmp_path) -> None:
    task = _mlflow_gate(tmp_path, tracking_required=False)
    assert task["mlflow_tracking_required"] is False
    assert not any("MLflow" in error for error in task["errors"])
    assert task["mlflow_delivery"] == {"state": "not_required", "reasons": []}
    assert "no MLflow export by design (mlflow_tracking_required is false in the saved config)" in task["warnings"]


@pytest.mark.parametrize("tracking_required", [True, None])
def test_mlflow_required_or_unstated_is_a_pending_delivery_without_evidence(tmp_path, tracking_required) -> None:
    task = _mlflow_gate(tmp_path, tracking_required=tracking_required)
    assert task["mlflow_tracking_required"] is True
    assert task["mlflow_delivery"] == {
        "state": "pending",
        "reasons": [
            "no MLflow run evidence: no live run id and no post-run export receipt, and the saved config snapshot "
            "does not name all of the tracking URI, experiment and run name; exporter activation and whether "
            "tracking was required or disabled are not recorded"
        ],
    }
    assert not [e for e in task["errors"] if "MLflow" in e] and task["status"] != "FAILED"
    assert any(w.startswith("MLflow delivery pending: ") for w in task["warnings"])


@pytest.mark.parametrize(
    ("config_line", "state"),
    [
        ("mlflow_tracking_required: false", "not_required"),
        ("mlflow_tracking_required: 'False'  # by design", "not_required"),
        ("mlflow_tracking_required: true", "pending"),
        ("other_key: 1", "pending"),
    ],
)
def test_an_old_quality_summary_without_the_provenance_key_falls_back_to_the_saved_config(
    tmp_path, config_line, state
) -> None:
    """A run born on a pin that did not report `tracking_required` in provenance still honours its saved config."""
    from nemo_gym.orchestration.recovery import gate_summary_from_collected

    job = gate_job(tmp_path, expected=4, final_zeros=0)
    (job / "artifacts" / "rollouts_mlflow_export_done").unlink()
    (job / "artifacts" / "rollouts_resolved_config.yaml").write_text(
        f"mlflow_tracking_uri: http://x{chr(10)}{config_line}{chr(10)}"
    )
    collected = collect_with_score(job, json.dumps(["agent", "mean/reward"]))
    d = decide_from_collected(collected, max_infra_retries=3, max_walltime_s=None, live_job=False)
    collected.quality = unit_quality(
        expected=d.expected, provenance={"resolved_config": "read", "servers": {}, "mlflow": {"configured": True}}
    )
    task = gate_summary_from_collected(d, collected, max_unhealthy_fraction=0.02, mlflow_status=None)
    assert task["mlflow_delivery"]["state"] == state
    assert task["mlflow_tracking_required"] is (state == "pending")
    assert not [e for e in task["errors"] if "MLflow" in e]


def _row_with_a_model_call(task: int, *, operations_field: bool) -> dict:
    """A scored row with one captured model call; a current pin writes model_operations, an earlier one does not."""
    metadata = {"response_id": f"response-{task}", "status_code": 200, "response_status": "completed"}
    if operations_field:
        metadata["model_operations"] = None
    trajectory = {
        "task_id": str(task),
        "rollout_id": f"{task}-0",
        "invocations": [{"invocation_id": "root", "status": "completed", "model_calls": [{"model_call_id": "c"}]}],
        "model_calls": [{"model_call_id": "c", "response_metadata": metadata}],
    }
    return ident(task, reward=1.0, ng_trajectory=trajectory)


def _trial_gate_of(
    tmp_path: Path, *, expected: int, final_zeros: int, operations_field: bool, recovery: dict | None = None
) -> tuple:
    from nemo_gym.orchestration.recovery import trial_health

    scored = expected - final_zeros
    results = [_row_with_a_model_call(t, operations_field=operations_field) for t in range(scored)]
    failures = [final_model_zero(t, 0) for t in range(scored, expected)]
    job = write_job_dir(tmp_path / "job", n=expected, results=results, failures=failures)
    (job / "artifacts" / "rollouts_aggregate_metrics.json").write_text(json.dumps({"agent": {"mean/reward": 1.0}}))
    collected = collect_with_score(job, json.dumps(["agent", "mean/reward"]))
    d = decide_from_collected(collected, max_infra_retries=3, max_walltime_s=None, live_job=False)
    collected.quality = unit_quality(expected=d.expected)
    if recovery is not None:
        collected.quality["observed_upstream_recovery"] = recovery
    return d, trial_health(d, collected)


def test_a_complete_earlier_pin_run_is_certified_on_coverage_with_the_note(tmp_path) -> None:
    d, (passed, reason) = _trial_gate_of(tmp_path, expected=50, final_zeros=0, operations_field=False)
    assert d.complete and passed is True
    assert reason == (
        "trial: evaluation-side faults: 0 of 50 (0.00%) is below 2%; model faults: 0 of 50 (0.00%); exact failure "
        "count unavailable (earlier pin): the run's rows predate operation observations, so the complete run is "
        "certified on coverage"
    )


_RECOVERED = {"recovered_attempt_errors": 3, "recovered_operations": 2}


def test_an_earlier_pin_run_with_recovered_attempts_is_certified_on_coverage(tmp_path) -> None:
    # The pin recorded no operations, so its recovered attempts cannot be joined to one either.
    d, (passed, reason) = _trial_gate_of(
        tmp_path, expected=50, final_zeros=0, operations_field=False, recovery=_RECOVERED
    )
    assert d.complete and passed is True
    assert "(earlier pin)" in reason
    assert reason.endswith("; 3 recovered attempt error(s) cannot be joined to an operation")


def test_a_current_pin_run_with_unjoined_recovered_attempts_stays_unknown(tmp_path) -> None:
    _, (passed, reason) = _trial_gate_of(
        tmp_path, expected=50, final_zeros=0, operations_field=True, recovery=_RECOVERED
    )
    assert passed is None
    assert reason.startswith("trial: UNVERIFIED, recovered attempts not joined: the run counted 3 recovered")


def test_a_current_pin_run_with_uncovered_operations_stays_unknown(tmp_path) -> None:
    _, (passed, reason) = _trial_gate_of(tmp_path, expected=50, final_zeros=0, operations_field=True)
    # A current pin records operations, so its ungraded trials are unobserved, never certified on coverage.
    assert passed is None and "trial: UNOBSERVED 50 of 50 expected trials" in reason
    assert "earlier pin" not in reason


def test_an_earlier_pin_run_at_the_failure_limit_still_fails(tmp_path) -> None:
    _, (passed, reason) = _trial_gate_of(tmp_path, expected=100, final_zeros=2, operations_field=False)
    assert passed is False and "is not below 2%" in reason


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
    from nemo_gym.orchestration.recovery import JUDGE_JOB_JOURNAL_NAME

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


class _FakeRun:
    def __init__(self, status: str) -> None:
        self.info = SimpleNamespace(status=status)


class _FakeClient:
    """An MlflowClient stand-in: `get_run` answers for the runs it knows or raises what it was told to."""

    def __init__(self, runs: dict | None = None, error: Exception | None = None) -> None:
        self.runs, self.error, self.asked = runs or {}, error, []

    def get_run(self, run_id: str) -> _FakeRun:
        self.asked.append(run_id)
        if self.error is not None:
            raise self.error
        return _FakeRun(self.runs[run_id])


def _delivery_with_a_queried_status(tmp_path, *, uri, client, receipt=True, mlflow_status=None) -> dict:
    from nemo_gym.orchestration.recovery import gate_summary_from_collected

    job = gate_job(tmp_path, expected=4, final_zeros=0)
    if not receipt:
        (job / "artifacts" / "rollouts_mlflow_export_done").unlink()
    config = "mlflow_experiment_name: e" + chr(10) + (f"mlflow_tracking_uri: {uri}" + chr(10) if uri else "")
    (job / "artifacts" / "rollouts_resolved_config.yaml").write_text(config)
    write_postrun_receipt(job)
    collected = collect_with_score(job, json.dumps(["agent", "mean/reward"]))
    d = decide_from_collected(collected, max_infra_retries=3, max_walltime_s=None, live_job=False)
    collected.quality = unit_quality(expected=d.expected)
    return gate_summary_from_collected(
        d,
        collected,
        max_unhealthy_fraction=0.02,
        mlflow_status=mlflow_status,
        mlflow_run_id="postrun-run" if mlflow_status else None,
        status_client_factory=lambda _uri: client,
    )


def test_the_receipt_s_run_status_is_read_from_the_tracking_server_when_the_caller_gave_none(tmp_path) -> None:
    client = _FakeClient(runs={"postrun-run": "FINISHED"})
    task = _delivery_with_a_queried_status(tmp_path, uri="http://tracking.example", client=client)
    assert client.asked == ["postrun-run"]
    assert task["mlflow_delivery"] == {"state": "verified", "reasons": []}
    assert not any("MLflow delivery" in w for w in task["warnings"])


def test_a_queried_status_that_is_not_finished_is_unverified(tmp_path) -> None:
    task = _delivery_with_a_queried_status(
        tmp_path, uri="http://tracking.example", client=_FakeClient(runs={"postrun-run": "RUNNING"})
    )
    assert task["mlflow_delivery"] == {"state": "unverified", "reasons": ["MLflow run status RUNNING"]}


def test_a_failed_status_query_is_unverified_with_its_reason_never_a_silent_not_observed(tmp_path) -> None:
    task = _delivery_with_a_queried_status(
        tmp_path, uri="http://tracking.example", client=_FakeClient(error=ConnectionError("server unreachable"))
    )
    assert task["mlflow_delivery"] == {
        "state": "unverified",
        "reasons": ["status query failed: ConnectionError: server unreachable"],
    }
    assert not any("not observed" in w for w in task["warnings"])
    assert any(w.startswith("MLflow delivery unverified: status query failed") for w in task["warnings"])


def test_without_a_tracking_uri_the_status_is_not_queried_and_delivery_is_pending(tmp_path) -> None:
    client = _FakeClient(runs={"postrun-run": "FINISHED"})
    task = _delivery_with_a_queried_status(tmp_path, uri=None, client=client)
    assert client.asked == []
    assert task["mlflow_delivery"] == {"state": "pending", "reasons": ["status not queried"]}


def test_a_status_the_caller_passes_is_never_queried(tmp_path) -> None:
    client = _FakeClient(error=ConnectionError("must not be asked"))
    task = _delivery_with_a_queried_status(
        tmp_path, uri="http://tracking.example", client=client, mlflow_status="FINISHED"
    )
    assert client.asked == [] and task["mlflow_delivery"]["state"] == "verified"


def test_a_failed_status_query_never_shows_a_password_from_the_error_url(tmp_path) -> None:
    error = ConnectionError("cannot reach http://user:SECRETPW@tracking.example:5000/api")
    task = _delivery_with_a_queried_status(tmp_path, uri="http://tracking.example", client=_FakeClient(error=error))
    reasons = task["mlflow_delivery"]["reasons"]
    assert task["mlflow_delivery"]["state"] == "unverified" and reasons
    assert "SECRETPW" not in json.dumps(task) and "tracking.example" in reasons[0]


# ---------------------------------------------------------------------------
# Verification skipped by configuration (judge_skipped): complete, unverified by construction
# ---------------------------------------------------------------------------

_SKIP_REASON = "CritPt smoke mode (fire_after): the batch was not sent to the grading API"


def _skipped_job(tmp_path, kind: str = "judge_skipped", n: int = 5):
    rows = [
        ident(t, mask_sample=True, reward=0.0, failure_kind=kind, failure_reason=_SKIP_REASON, response=SAVED_ANSWER)
        for t in range(n)
    ]
    return write_job_dir(tmp_path / "job", n=n, results=rows, failures=[])


def test_an_all_skipped_run_is_a_complete_noop_and_the_gate_reads_unverified(tmp_path) -> None:
    from nemo_gym.orchestration.recovery import gate_summary_from_collected

    collected = collect(_skipped_job(tmp_path))
    decision = decide_from_collected(collected, max_infra_retries=3, max_walltime_s=None, live_job=False)
    assert decision.action == RecoveryAction.NOOP and decision.complete and decision.unverified
    assert "5 result(s) not graded by configuration (judge_skipped)" in decision.reason
    assert _SKIP_REASON in decision.reason

    task = gate_summary_from_collected(decision, collected, max_unhealthy_fraction=0.02)
    assert task["status"] == "UNVERIFIED" and task["score"] is None and task["errors"] == []
    assert task["scored_samples"] == 0 and decision.reason in task["warnings"]


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


def test_a_masked_result_of_any_other_kind_still_stops_the_run(tmp_path) -> None:
    from nemo_gym.orchestration.recovery import gate_summary_from_collected

    collected = collect(_skipped_job(tmp_path, kind="provider_unavailable", n=2))
    decision = decide_from_collected(collected, max_infra_retries=3, max_walltime_s=None, live_job=False)
    assert not decision.unverified and not decision.complete
    task = gate_summary_from_collected(decision, collected, max_unhealthy_fraction=0.02)
    assert task["status"] != "UNVERIFIED"
