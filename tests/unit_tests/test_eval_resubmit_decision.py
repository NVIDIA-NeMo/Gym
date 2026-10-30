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
    assert len(got.failures) == 1 and got.failures[0]["_ng_attempt_index"] == 1
    assert got.failures[0]["_ng_failure_events"]["e"]["attempt_id"] == "a1"
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

    def resubmit(self, record, benchmark):
        FakeExecutor.calls.append("resubmit")
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
        dict(failure_kind="judge_failed", _ng_failure_terminal=True),
        dict(failure_kind="provider_unavailable", _ng_failure_terminal=True),
        dict(failure_kind="model_context_limit", _ng_failure_terminal=False),
        dict(
            failure_kind="model_context_limit",
            _ng_failure_terminal=True,
            _ng_failure_events={"e": {"boundary": "resources_server", "operation": "judge"}},
        ),
    ],
)
def test_counts_as_zero_without_terminal_model_ownership_is_not_final(tmp_path, fake_executor, contradiction) -> None:
    row = ident(1, _ng_failure_counts_as_zero=True, **contradiction)
    fake_executor.job_dir = write_job_dir(tmp_path / "job", n=2, results=[ident(0, reward=1.0)], failures=[row])
    out = submit.recover("gym-job-c2", "bench")
    assert out.decision.identities.get("model_final") is None
    assert not out.ok and out.decision.action == RecoveryAction.STOP and fake_executor.calls == ["decide"]


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
    assert fake_executor.calls == ["resubmit"]


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
    return job


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
    task = gate_summary_from_collected(d, collected, max_unhealthy_fraction=0.02)
    assert (task["status"], task["score"], task["expected_samples"], task["scored_samples"], task["errors"]) == (
        "SUCCESS",
        0.75,
        4,
        4,
        [],
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
        ],
        capsys,
    )
    assert code == 0 and out["dispatched"] is None and fake_executor.calls == ["decide"]
    assert out["gate_task"] == {
        "status": "SUCCESS",
        "expected_samples": 2,
        "scored_samples": 2,
        "score": 0.5,
        "errors": [],
        "warnings": [],
    }


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


def test_mlflow_run_without_finished_export_blocks_acceptance(tmp_path, fake_executor) -> None:
    job = write_job_dir(tmp_path / "job", n=1, results=[ident(0, reward=1.0)], failures=[])
    (job / "artifacts" / "rollouts_mlflow_run_id").write_text("run")
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
    fake_executor.job_dir = job
    out = submit.recover("gym-job-o", "bench", score_key=["agent", "mean/reward"], max_unhealthy_fraction=0.02)
    assert out.decision.action == RecoveryAction.NOOP and out.dispatched is None
    assert out.gate_task["status"] == "FAILED" and any("MLflow" in e for e in out.gate_task["errors"])
    (job / "artifacts" / "rollouts_mlflow_export_done").write_text("")
    unobserved = submit.recover("gym-job-o2", "bench", score_key=["agent", "mean/reward"], max_unhealthy_fraction=0.02)
    assert unobserved.gate_task["status"] == "FAILED"
    done = submit.recover(
        "gym-job-o3",
        "bench",
        score_key=["agent", "mean/reward"],
        max_unhealthy_fraction=0.02,
        mlflow_status="FINISHED",
    )
    assert done.gate_task["status"] == "SUCCESS" and done.gate_task["errors"] == []


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


def mlflow_job(tmp_path: Path, *, marker: bool) -> Path:
    job = write_job_dir(tmp_path / "job", n=1, results=[ident(0, reward=1.0)], failures=[])
    (job / "artifacts" / "rollouts_mlflow_run_id").write_text("run")
    if marker:
        (job / "artifacts" / "rollouts_mlflow_export_done").write_text("")
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
    return job


@pytest.mark.parametrize(
    ("marker", "status", "accepted"),
    [
        (True, "FINISHED", True),
        (True, "FAILED", False),
        (True, None, False),
        (True, "RUNNING", False),
        (False, "FINISHED", False),
    ],
)
def test_mlflow_acceptance_needs_the_export_marker_and_an_observed_finished_status(
    tmp_path, fake_executor, marker, status, accepted
) -> None:
    fake_executor.job_dir = mlflow_job(tmp_path, marker=marker)
    out = submit.recover(
        "gym-job-r", "bench", score_key=["agent", "mean/reward"], max_unhealthy_fraction=0.02, mlflow_status=status
    )
    assert out.dispatched is None and fake_executor.calls == ["decide"]
    assert (out.gate_task["status"] == "SUCCESS") is accepted
    assert out.mlflow == {"local": {"run_id_present": True, "export_done": marker}, "observed_status": status}


def test_mlflow_status_for_a_run_without_an_mlflow_id_is_contradictory(tmp_path, fake_executor) -> None:
    job = mlflow_job(tmp_path, marker=True)
    (job / "artifacts" / "rollouts_mlflow_run_id").unlink()
    fake_executor.job_dir = job
    out = submit.recover(
        "gym-job-s", "bench", score_key=["agent", "mean/reward"], max_unhealthy_fraction=0.02, mlflow_status="FINISHED"
    )
    assert out.gate_task["status"] == "FAILED" and any("no MLflow run id" in e for e in out.gate_task["errors"])


def test_cli_passes_the_observed_mlflow_status(tmp_path, fake_executor, capsys) -> None:
    fake_executor.job_dir = mlflow_job(tmp_path, marker=True)
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
