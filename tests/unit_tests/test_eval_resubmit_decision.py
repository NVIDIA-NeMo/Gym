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
evidence. The collector is the real ``COLLECT_SCRIPT``, run as the executor runs it on a login node.
"""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from nemo_gym.orchestration import submit
from nemo_gym.orchestration.executors.base import BaseExecutor
from nemo_gym.orchestration.recovery import (
    COLLECT_SCRIPT,
    CollectedEvidence,
    RecoveryAction,
    RecoveryOutcome,
    decide_from_collected,
    gate_task_summary,
)


class _StartsWith:
    def __init__(self, prefix: str) -> None:
        self.prefix = prefix

    def __eq__(self, other: object) -> bool:
        return isinstance(other, str) and other.startswith(self.prefix)

    def __repr__(self) -> str:
        return f"startswith({self.prefix!r})"


# A report without run.stats.responses leaves the response gate unknown, and the trial count is a lower bound.
HEALTH_UNKNOWN = _StartsWith("response/trial health unknown: ['response: the health report has no verified response")


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
    assert got.results == [{"_ng_task_index": 0, "_ng_rollout_index": 0, "reward": 1.0}]
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


def test_missing_inventory_stops(tmp_path) -> None:
    job = tmp_path / "job"
    (job / "artifacts").mkdir(parents=True)
    d = decide_from_collected(collect(job), max_infra_retries=3, max_walltime_s=None, live_job=False)
    assert d.action == RecoveryAction.STOP and "inventory" in d.reason


class FakeExecutor:
    """Records every dispatch; ``supports_reverify`` stands for an executor with a saved-answer re-judgment path."""

    name = "fake"
    calls: list[str] = []
    job_dir: Path
    live = False
    supports_reverify = True

    def recovery_evidence(self, record, benchmark, *, score_key=None):
        FakeExecutor.calls.append("decide")
        out = subprocess.run(
            [
                sys.executable,
                "-c",
                COLLECT_SCRIPT,
                str(FakeExecutor.job_dir),
                *([json.dumps(score_key)] if score_key else []),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        collected = CollectedEvidence.model_validate_json(out.stdout)
        decision = decide_from_collected(
            collected, max_infra_retries=3, max_walltime_s=None, live_job=FakeExecutor.live
        )
        return decision, collected

    def reverify_judge_failures(self, record, benchmark, decision):
        if not FakeExecutor.supports_reverify:
            return BaseExecutor.reverify_judge_failures(self, record, benchmark, decision)
        FakeExecutor.calls.append("reverify")
        return RecoveryOutcome(decision=decision, ok=True, dispatched="reverify")

    def resubmit(self, record, benchmark, *, keep_budget=False):
        FakeExecutor.calls.append("resubmit" if keep_budget else "resubmit-resetting-budget")
        return type("Job", (), {"job_id": "4242"})()


@pytest.fixture
def fake_executor(monkeypatch):
    FakeExecutor.calls = []
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
        results=[ident(0, reward=1.0), ident(1, _ng_failure_class="judge_failed", reward=0.0)],
        failures=[],
    )
    out = submit.recover("gym-job-b", "bench")
    assert out.ok and out.dispatched == "reverify"
    assert fake_executor.calls == ["decide", "reverify"]


def test_recover_reports_an_executor_without_saved_answer_rejudgment_as_unsupported(tmp_path, fake_executor) -> None:
    fake_executor.supports_reverify = False
    fake_executor.job_dir = write_job_dir(
        tmp_path / "job", n=1, results=[ident(0, _ng_failure_class="judge_failed", reward=0.0)], failures=[]
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


@pytest.mark.parametrize("kind", ["provider_unavailable", "model_transport_error", None])
def test_a_terminal_counts_as_zero_row_is_settled_as_rollout_collection_settles_it(
    tmp_path, fake_executor, kind
) -> None:
    # The producer's flag and finality decide, as in _final_zero_failure_rows; no label list overrides them.
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


def test_recover_with_judge_failures_and_missing_work_queues_the_resume_and_leaves_rejudgment_to_it(
    tmp_path, fake_executor
) -> None:
    fake_executor.job_dir = write_job_dir(
        tmp_path / "job",
        n=3,
        results=[ident(0, reward=1.0), ident(1, _ng_failure_class="judge_failed")],
        failures=[],
        retries=0,
    )
    out = submit.recover("gym-job-d2", "bench")
    assert out.dispatched == "resubmit" and fake_executor.calls == ["decide", "resubmit"]
    assert out.decision.identities["judge_recoverable"] == [(1, 0)] and out.decision.identities["missing"] == [(2, 0)]


def test_recover_does_not_queue_beside_a_live_job(tmp_path, fake_executor) -> None:
    fake_executor.job_dir = write_job_dir(tmp_path / "job", n=2, results=[ident(0, reward=1.0)], failures=[])
    fake_executor.live = True
    out = submit.recover("gym-job-e", "bench")
    assert not out.ok and "still pending or running" in out.decision.reason and fake_executor.calls == ["decide"]


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
        results=[ident(0, reward=1.0), ident(1, _ng_failure_class="judge_failed")],
        failures=[ident(2, **INFRA)],
        retries=0,
    )
    d = decide_from_collected(collect(job), max_infra_retries=3, max_walltime_s=None, live_job=False)
    t = gate_task_summary(d, score=0.25, fe_valid=True, fe_reasons=[], health_passed=True, health_reasons=[])
    assert (t["expected_samples"], t["scored_samples"], t["status"]) == (4, 1, "RESUMING")
    assert any("judge failures" in e for e in t["errors"]) and any("no row" in e for e in t["errors"])


@pytest.mark.parametrize(("fe", "health"), [(None, True), (True, None), (False, True), (True, False)])
def test_gate_summary_unknown_or_failed_validity_or_health_is_a_blocker(tmp_path, fe, health) -> None:
    job = write_job_dir(tmp_path / "job", n=1, results=[ident(0, reward=1.0)], failures=[])
    d = decide_from_collected(collect(job), max_infra_retries=3, max_walltime_s=None, live_job=False)
    t = gate_task_summary(d, score=1.0, fe_valid=fe, fe_reasons=[], health_passed=health, health_reasons=[])
    assert t["status"] == "FAILED" and t["errors"]


def test_gate_summary_complete_run_succeeds_and_keeps_model_zeros_as_warnings(tmp_path) -> None:
    final = ident(1, failure_kind="model_context_limit", _ng_failure_terminal=True, _ng_failure_counts_as_zero=True)
    job = write_job_dir(tmp_path / "job", n=2, results=[ident(0, reward=1.0)], failures=[final])
    d = decide_from_collected(collect(job), max_infra_retries=3, max_walltime_s=None, live_job=False)
    t = gate_task_summary(d, score=0.5, fe_valid=True, fe_reasons=[], health_passed=True, health_reasons=[])
    assert (t["status"], t["expected_samples"], t["scored_samples"], t["errors"]) == ("SUCCESS", 2, 2, [])
    assert t["warnings"] and "final evaluated-model" in t["warnings"][0]


def write_quality(job: Path, *, expected, healthy, verdicts, partial=False, reasons=()) -> None:
    run = {
        "artifacts": {
            "expected": expected,
            "missing": 0,
            "healthy": healthy,
            "unhealthy_reasons": list(reasons),
            "partial": partial,
        },
        "verdicts": verdicts,
    }
    (job / "artifacts" / "quality_summary.json").write_text(json.dumps({"run": run}))


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
    receipt = {"version": 1, "run_id": run_id, "artifacts": artifact_digests(job)}
    (job / "artifacts" / "rollouts_mlflow_export_done").write_text(json.dumps(receipt))


GOOD_VERDICTS = {"healthy": 4, "unhealthy": 0, "unobserved": 0}


@pytest.mark.parametrize(
    ("quality", "criterion", "fe", "health"),
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
def test_quality_reader_keeps_validity_and_health_apart_and_fails_closed(
    tmp_path, quality, criterion, fe, health
) -> None:
    from nemo_gym.orchestration.recovery import quality_inputs

    job = complete_job(tmp_path)
    if quality is not None:
        write_quality(job, **quality)
    collected = collect_with_score(job, json.dumps(["agent", "mean/reward"]))
    d = decide_from_collected(collected, max_infra_retries=3, max_walltime_s=None, live_job=False)
    got_fe, _, got_health, reasons = quality_inputs(d, collected, max_unhealthy_fraction=criterion)
    assert (got_fe, got_health) == (fe, health) and reasons


def test_gate_summary_from_collected_reads_the_named_score_and_passes_only_when_everything_holds(tmp_path) -> None:
    from nemo_gym.orchestration.recovery import gate_summary_from_collected

    job = complete_job(tmp_path)
    write_quality(job, expected=4, healthy=True, verdicts=GOOD_VERDICTS)
    collected = collect_with_score(job, json.dumps(["agent", "mean/reward"]))
    d = decide_from_collected(collected, max_infra_retries=3, max_walltime_s=None, live_job=False)
    task = gate_summary_from_collected(d, collected, max_unhealthy_fraction=0.02, mlflow_status="FINISHED")
    assert (task["status"], task["score"], task["expected_samples"], task["scored_samples"], task["errors"]) == (
        "FAILED",
        0.75,
        4,
        4,
        [HEALTH_UNKNOWN],
    )
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
            "root": {
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
    # Unknown health still stops acceptance, without creating missing work.
    assert out.gate_task["status"] == "FAILED" and any("unknown" in e for e in out.gate_task["errors"])
    assert "missing" not in out.decision.counts


@pytest.mark.parametrize(
    "row",
    [
        dict(_ng_failure_class="judge_failed"),
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
    (job / "artifacts" / "quality_summary.json").write_text(
        json.dumps(
            {
                "run": {
                    "artifacts": {
                        "expected": 2,
                        "missing": 0,
                        "healthy": True,
                        "unhealthy_reasons": [],
                        "partial": False,
                    },
                    "verdicts": {"healthy": 2, "unhealthy": 0, "unobserved": 0},
                }
            }
        )
    )
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
            "--mlflow-status",
            "FINISHED",
        ],
        capsys,
    )
    assert code == 0 and out["dispatched"] is None and fake_executor.calls == ["decide"]
    assert out["gate_task"]["status"] == "FAILED" and out["gate_task"]["score"] == 0.5
    assert out["gate_task"]["errors"] == [HEALTH_UNKNOWN]
    assert out["gate_task"]["warnings"][0].startswith("per-rollout health within the configured criterion")


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
    assert out.gate_task["status"] == "RUNNING" and fake_executor.calls == ["decide"] and not out.ok


def test_a_live_mlflow_run_is_unverified_delivery_not_an_unfinished_export(tmp_path, fake_executor) -> None:
    fake_executor.job_dir = mlflow_job(tmp_path, mode="live")
    out = submit.recover(
        "gym-job-o", "bench", score_key=["agent", "mean/reward"], max_unhealthy_fraction=0.02, mlflow_status="FINISHED"
    )
    assert out.decision.action == RecoveryAction.NOOP and out.dispatched is None
    mlflow = [e for e in out.gate_task["errors"] if "MLflow" in e]
    assert out.gate_task["status"] == "FAILED" and mlflow == [
        "MLflow live run live-run: configured payload delivery is not verified (live uploads log failures and "
        "continue), so live acceptance is unsupported"
    ]


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
    assert any("chain stopped" in e for e in out.gate_task["errors"])


def mlflow_job(tmp_path: Path, *, mode: str) -> Path:
    """A finished run whose MLflow files are the ones each producer leaves (or a damaged version of them)."""
    job = write_job_dir(tmp_path / "job", n=1, results=[ident(0, reward=1.0)], failures=[])
    (job / "artifacts" / "quality_summary.json").write_text(
        json.dumps(
            {
                "run": {
                    "artifacts": {
                        "expected": 1,
                        "missing": 0,
                        "healthy": True,
                        "unhealthy_reasons": [],
                        "partial": False,
                    },
                    "verdicts": {"healthy": 1, "unhealthy": 0, "unobserved": 0},
                }
            }
        )
    )
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
        ("postrun", None, ["MLflow terminal status not observed"]),
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
        (
            "both",
            "FINISHED",
            ["MLflow live run live-run and a post-run export both exist: the observed status names neither"],
        ),
        ("none", None, ["no MLflow run evidence: no live run id and no post-run export receipt"]),
    ],
)
def test_mlflow_delivery_follows_the_producer_of_its_evidence(tmp_path, fake_executor, mode, status, mlflow) -> None:
    fake_executor.job_dir = mlflow_job(tmp_path, mode=mode)
    out = submit.recover(
        "gym-job-r", "bench", score_key=["agent", "mean/reward"], max_unhealthy_fraction=0.02, mlflow_status=status
    )
    assert out.dispatched is None and fake_executor.calls == ["decide"]
    # No case accepts: response/trial health has no producer. Only a current post-run receipt with FINISHED has no
    # MLflow error.
    assert out.gate_task["status"] == "FAILED" and out.gate_task["errors"] == [HEALTH_UNKNOWN] + mlflow


def test_mlflow_status_for_a_run_without_an_mlflow_id_is_contradictory(tmp_path, fake_executor) -> None:
    fake_executor.job_dir = mlflow_job(tmp_path, mode="none")
    out = submit.recover(
        "gym-job-s", "bench", score_key=["agent", "mean/reward"], max_unhealthy_fraction=0.02, mlflow_status="FINISHED"
    )
    assert out.gate_task["status"] == "FAILED" and any("no MLflow run id" in e for e in out.gate_task["errors"])


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
    assert code == 0 and out["mlflow"]["observed_status"] == "FAILED" and out["gate_task"]["status"] == "FAILED"


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
        ({}, ["11"], True, RecoveryAction.RESUBMIT),
        ({".gym_accumulated_walltime": "3600"}, ["11", "12"], True, RecoveryAction.RESUBMIT),
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


@pytest.mark.parametrize(
    "row",
    [
        dict(),
        dict(reward="1"),
        dict(reward=1.0, mask_sample=True),
        dict(reward=0.0, evaluation_completed=False),
    ],
)
def test_a_result_without_a_usable_grade_is_unknown_not_scored(tmp_path, row) -> None:
    job = write_job_dir(tmp_path / "job", n=2, results=[ident(0, reward=1.0), ident(1, **row)], failures=[], retries=0)
    d = decide_from_collected(collect(job), max_infra_retries=3, max_walltime_s=None, live_job=False)
    assert d.identities.get("unknown") == [(1, 0)] and d.action == RecoveryAction.STOP


def test_a_legitimate_model_zero_grade_stays_scored(tmp_path) -> None:
    job = write_job_dir(tmp_path / "job", n=2, results=[ident(0, reward=1.0), ident(1, reward=0.0)], failures=[])
    d = decide_from_collected(collect(job), max_infra_retries=3, max_walltime_s=None, live_job=False)
    assert d.counts == {"scored": 2} and d.action == RecoveryAction.NOOP


def test_an_unavailable_score_is_an_explicit_blocker(tmp_path) -> None:
    from nemo_gym.orchestration.recovery import gate_summary_from_collected

    job = complete_job(tmp_path)
    write_quality(job, expected=4, healthy=True, verdicts=GOOD_VERDICTS)
    collected = collect(job)
    d = decide_from_collected(collected, max_infra_retries=3, max_walltime_s=None, live_job=False)
    task = gate_summary_from_collected(d, collected, max_unhealthy_fraction=0.02)
    assert (
        task["score"] is None and task["status"] == "FAILED" and any("score unavailable" in e for e in task["errors"])
    )


def test_verdict_counts_that_do_not_cover_the_inventory_make_health_unknown(tmp_path) -> None:
    from nemo_gym.orchestration.recovery import quality_inputs

    job = complete_job(tmp_path)
    write_quality(job, expected=4, healthy=True, verdicts={"healthy": 9, "unhealthy": 0, "unobserved": 0})
    collected = collect(job)
    d = decide_from_collected(collected, max_infra_retries=3, max_walltime_s=None, live_job=False)
    _, _, health, reasons = quality_inputs(d, collected, max_unhealthy_fraction=0.02)
    assert health is None and "do not cover" in reasons[0]


@pytest.mark.parametrize("score", [float("nan"), float("inf"), float("-inf")])
def test_a_nonfinite_aggregate_score_cannot_be_accepted(tmp_path, score) -> None:
    from nemo_gym.orchestration.recovery import gate_summary_from_collected

    job = complete_job(tmp_path)
    write_quality(job, expected=4, healthy=True, verdicts=GOOD_VERDICTS)
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
    return {"_ng_task_index": task, "_ng_rollout_index": 0, "_ng_omitted_rollout": True}


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
    write_quality(job, expected=100, healthy=True, verdicts=verdicts)
    collected = collect(job)
    d = decide_from_collected(collected, max_infra_retries=3, max_walltime_s=None, live_job=False)
    assert quality_inputs(d, collected, max_unhealthy_fraction=criterion)[2] is health


def test_absent_mlflow_evidence_never_passes(tmp_path) -> None:
    from nemo_gym.orchestration.recovery import gate_summary_from_collected

    job = complete_job(tmp_path)
    write_quality(job, expected=4, healthy=True, verdicts=GOOD_VERDICTS)
    (job / "artifacts" / "rollouts_mlflow_export_done").unlink()
    collected = collect_with_score(job, json.dumps(["agent", "mean/reward"]))
    d = decide_from_collected(collected, max_infra_retries=3, max_walltime_s=None, live_job=False)
    task = gate_summary_from_collected(d, collected, max_unhealthy_fraction=0.02)
    assert (
        task["status"] == "FAILED"
        and "no MLflow run evidence: no live run id and no post-run export receipt" in (task["errors"])
    )


def test_an_interrupted_row_with_failure_events_restarts_like_rollout_collection(tmp_path, fake_executor) -> None:
    # Rollout collection (_latest_failure_rows, _load_from_cache) treats any latest interrupted row as missing and
    # runs the input again, whatever events it carries; recovery selects the same input.
    row = interrupted(1) | {
        "_ng_failure_class": "agent_run_error",
        "_ng_failure_id": "f",
        "_ng_failure_events": {"e": {"boundary": "agent", "operation": "run"}},
    }
    fake_executor.job_dir = write_job_dir(
        tmp_path / "job", n=2, results=[ident(0, reward=1.0)], failures=[row], retries=0
    )
    out = submit.recover("gym-job-z", "bench")
    assert out.decision.identities["missing"] == [(1, 0)] and "unknown" not in out.decision.identities
    assert fake_executor.calls == ["decide", "resubmit"]


def test_named_health_facts_below_the_criterion_reach_the_gate_warnings(tmp_path) -> None:
    from nemo_gym.orchestration.recovery import gate_summary_from_collected

    # Shape of a real run (chartqapro, inc52): nonzero issues below the limit and an unobserved run-level check.
    job = complete_job(tmp_path)
    write_quality(job, expected=4, healthy=True, verdicts={"healthy": 3, "unhealthy": 1, "unobserved": 0})
    summary_path = job / "artifacts" / "quality_summary.json"
    summary = json.loads(summary_path.read_text())
    summary["run"]["issues"] = {
        "model_call_runaway_generation": 1,
        "task_consistently_unhealthy": 1,
        "agent_turn_hollow": 0,
    }
    summary["run"]["ignored_checks"] = []
    summary["run"]["artifacts"]["coverage"] = {
        "model_call_runaway_generation": {"evaluated": 4, "unobserved": 0, "ignored": 0},
        "run_excessive_failures": {"evaluated": 0, "unobserved": 1, "ignored": 0},
    }
    summary_path.write_text(json.dumps(summary))
    collected = collect_with_score(job, json.dumps(["agent", "mean/reward"]))
    d = decide_from_collected(collected, max_infra_retries=3, max_walltime_s=None, live_job=False)
    task = gate_summary_from_collected(d, collected, max_unhealthy_fraction=0.5, mlflow_status="FINISHED")
    assert task["status"] == "FAILED"
    assert task["warnings"][-3].startswith("per-rollout health within the configured criterion")
    assert task["warnings"][-2:] == [
        "health-check issues (nonzero): model_call_runaway_generation 1, task_consistently_unhealthy 1",
        "health checks not fully evaluated: run_excessive_failures (evaluated 0, unobserved 1, ignored 0)",
    ]


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
    base = pytest.importorskip("nemo_gym.exporters.base")
    if not hasattr(base, "_artifact_version"):
        pytest.skip("this producer writes a run-id-only post-run marker")
    from nemo_gym.run_artifacts import RunArtifacts

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
        mlflow_errors({"live_run_id": None, "receipt": {"run_id": "postrun-run", "current": True}}, "FINISHED") == []
    )


def gate_job(
    tmp_path: Path, *, expected: int, final_zeros: int, missing: int = 0, responses: dict | None = None
) -> Path:
    """Reader fixture (hand-written report): scored rows, settled zeros and a quality summary."""
    scored = expected - final_zeros - missing
    results = [ident(t, reward=float(t % 2)) for t in range(scored)]
    failures = [final_model_zero(t, 0) for t in range(scored, scored + final_zeros)]
    job = write_job_dir(tmp_path / "job", n=expected, results=results, failures=failures)
    (job / "artifacts" / "rollouts_aggregate_metrics.json").write_text(json.dumps({"agent": {"mean/reward": 0.5}}))
    write_quality(
        job, expected=expected, healthy=True, verdicts={"healthy": expected, "unhealthy": 0, "unobserved": 0}
    )
    if responses is not None:
        path = job / "artifacts" / "quality_summary.json"
        summary = json.loads(path.read_text())
        summary["run"]["stats"] = {"responses": responses}
        path.write_text(json.dumps(summary))
    write_postrun_receipt(job)
    return job


def gate_of(job: Path) -> tuple:
    from nemo_gym.orchestration.recovery import gate_summary_from_collected

    collected = collect_with_score(job, json.dumps(["agent", "mean/reward"]))
    d = decide_from_collected(collected, max_infra_retries=3, max_walltime_s=None, live_job=False)
    return d, gate_summary_from_collected(d, collected, max_unhealthy_fraction=0.02, mlflow_status="FINISHED")


@pytest.mark.parametrize(("final_zeros", "failed"), [(0, False), (1, False), (2, True), (3, True)])
def test_known_trial_failures_fail_at_two_percent_and_are_only_a_lower_bound_below_it(
    tmp_path, final_zeros, failed
) -> None:
    job = gate_job(tmp_path, expected=100, final_zeros=final_zeros)
    rollouts = (job / "artifacts" / "rollouts.jsonl").read_bytes()
    d, task = gate_of(job)
    # Settled answers stand: the action is NOOP whatever the health, and nothing rewrites the main output.
    assert d.action == RecoveryAction.NOOP and (job / "artifacts" / "rollouts.jsonl").read_bytes() == rollouts
    assert task["status"] == "FAILED" and task["scored_samples"] == 100 and task["score"] == 0.5
    [health] = [e for e in task["errors"] if e.startswith("response/trial health")]
    rate = f"{final_zeros} settled zero-scored failure(s) of 100 expected trials ({final_zeros / 100:.2%})"
    if failed:
        assert health.startswith("response/trial health failed") and f"trial: {rate} is not below 2%" in health
    else:
        assert health.startswith("response/trial health unknown") and "known failures are a lower bound" in health
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
    d, task = gate_of(gate_job(tmp_path, expected=100, final_zeros=0, responses=responses))
    assert d.action == RecoveryAction.NOOP and task["status"] == "FAILED"
    [health] = [e for e in task["errors"] if e.startswith("response/trial health")]
    within = [w for w in task["warnings"] if w.startswith("health gate within the bounded failure policy: response:")]
    if state == "within":
        assert within and "4 faulted of 250 selected raw responses (1.60%)" in within[0]
        assert health.startswith("response/trial health unknown") and "response:" not in health
    elif state == "failed":
        assert health.startswith("response/trial health failed") and "5 faulted of 250" in health and not within
    else:
        assert health.startswith("response/trial health unknown") and "response:" in health and not within


def test_the_trial_gate_is_not_evaluated_on_an_incomplete_run(tmp_path) -> None:
    d, task = gate_of(gate_job(tmp_path, expected=100, final_zeros=0, missing=1))
    assert d.action == RecoveryAction.RESUBMIT and not d.complete
    [health] = [e for e in task["errors"] if e.startswith("response/trial health")]
    assert "trial: not evaluated, the run is not complete" in health
