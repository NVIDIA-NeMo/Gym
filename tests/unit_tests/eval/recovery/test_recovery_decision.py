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
"""What recovery decides from a saved run's evidence: the retry, stop and gate verdicts for each kind of failure.

Each case writes a job dir, reads it with the real stdlib collector and decides from the collected evidence; no executor is involved.
"""

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from nemo_gym.core.responses_api_types import NeMoGymResponse
from nemo_gym.eval.recovery.decision import (
    RecoveryAction,
)
from nemo_gym.eval.recovery.evidence import (
    COLLECT_SCRIPT,
    CollectedEvidence,
)
from nemo_gym.eval.recovery.gate import (
    GateCriteria,
    gate_task_summary,
)
from nemo_gym.eval.recovery.planner import decide_from_collected
from nemo_gym.eval.recovery.policy import RecoveryOutcomePolicyConfig
from nemo_gym.rollouts.rollout_collection import RolloutCollectionConfig, RolloutCollectionHelper
from tests.helpers.recovery_evidence import (
    _SKIP_REASON,
    HEALTH_UNKNOWN,
    INFRA,
    SAVED_ANSWER,
    STOP_PROVIDER,
    _judge_failure,
    _skipped_job,
    artifact_digests,
    collect,
    collect_with_score,
    complete_job,
    final_model_zero,
    ident,
    produce_quality,
    write_job_dir,
    write_postrun_receipt,
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
    from nemo_gym.eval.recovery.gate import quality_inputs

    job = complete_job(tmp_path)
    collected = collect_with_score(job, json.dumps(["agent", "mean/reward"]))
    collected.quality = unit_quality(**quality) if quality is not None else None
    d = decide_from_collected(collected, max_infra_retries=3, max_walltime_s=None, live_job=False)
    got_summary, _, got_health, reasons = quality_inputs(d, collected, max_unhealthy_fraction=criterion)
    assert (got_summary, got_health) == (summary_valid, health) and reasons


def test_gate_summary_from_collected_reads_the_named_score_and_passes_only_when_everything_holds(tmp_path) -> None:
    from nemo_gym.eval.recovery.gate import gate_summary_from_collected

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
    from nemo_gym.eval.recovery.gate import gate_summary_from_collected

    job = complete_job(tmp_path)
    produce_quality(job)
    collected = collect(job)
    d = decide_from_collected(collected, max_infra_retries=3, max_walltime_s=None, live_job=False)
    task = gate_summary_from_collected(d, collected, max_unhealthy_fraction=0.02)
    assert (
        task["score"] is None and task["status"] == "FAILED" and any("score unavailable" in e for e in task["errors"])
    )


def test_verdict_counts_that_do_not_cover_the_inventory_make_health_unknown(tmp_path) -> None:
    from nemo_gym.eval.recovery.gate import quality_inputs

    job = complete_job(tmp_path)
    collected = collect(job)
    collected.quality = unit_quality(expected=4, verdicts={"healthy": 9, "unhealthy": 0, "unobserved": 0})
    d = decide_from_collected(collected, max_infra_retries=3, max_walltime_s=None, live_job=False)
    _, _, health, reasons = quality_inputs(d, collected, max_unhealthy_fraction=0.02)
    assert health is None and "do not cover" in reasons[0]


@pytest.mark.parametrize("score", [float("nan"), float("inf"), float("-inf")])
def test_a_nonfinite_aggregate_score_cannot_be_accepted(tmp_path, score) -> None:
    from nemo_gym.eval.recovery.gate import gate_summary_from_collected

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
    from nemo_gym.eval.recovery.gate import quality_inputs

    job = write_job_dir(tmp_path / "job", n=100, results=[ident(t, reward=1.0) for t in range(100)], failures=[])
    collected = collect(job)
    collected.quality = unit_quality(expected=100, verdicts=verdicts)
    d = decide_from_collected(collected, max_infra_retries=3, max_walltime_s=None, live_job=False)
    assert quality_inputs(d, collected, max_unhealthy_fraction=criterion)[2] is health


def test_absent_mlflow_evidence_is_a_pending_delivery_fact_not_a_failure(tmp_path) -> None:
    from nemo_gym.eval.recovery.gate import gate_summary_from_collected

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


def test_named_health_facts_below_the_criterion_reach_the_gate_warnings(tmp_path) -> None:
    from nemo_gym.eval.recovery.gate import gate_summary_from_collected

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
    from nemo_gym.eval.recovery.gate import health_check_facts

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
    from nemo_gym.eval.recovery.gate import mlflow_errors

    assert mlflow_errors(local, "FINISHED")


def test_a_current_post_run_receipt_with_finished_is_the_only_mlflow_pass() -> None:
    from nemo_gym.eval.recovery.gate import mlflow_errors

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
    from nemo_gym.eval.recovery.gate import gate_summary_from_collected

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
    from nemo_gym.eval.recovery.trials import trial_boundary

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
    from nemo_gym.eval.recovery.gate import gate_summary_from_collected

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
    from nemo_gym.eval.recovery.gate import mlflow_errors

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
    from nemo_gym.eval.recovery.gate import mlflow_errors

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


@pytest.mark.parametrize(
    ("scope", "message"),
    [
        (False, "records that the rollouts were not uploaded"),
        (None, "does not record whether the rollouts were uploaded"),
        ("yes", "does not record whether the rollouts were uploaded"),
    ],
)
def test_a_receipt_without_an_uploaded_rollouts_record_is_not_delivery(scope, message) -> None:
    from nemo_gym.eval.recovery.gate import mlflow_errors

    receipt = {"run_id": "postrun-run", "current": True, "uploaded_rollouts": scope}
    errors = mlflow_errors({"live_run_id": None, "receipt": receipt}, "FINISHED", observed_run_id="postrun-run")
    assert len(errors) == 1 and message in errors[0]


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


def test_review_boolean_expected_count_is_not_one(tmp_path):
    from nemo_gym.eval.recovery.gate import quality_inputs

    job = write_job_dir(tmp_path / "job", n=1, results=[ident(0, reward=1.0)], failures=[])
    got = collect(job)
    got.quality = unit_quality(expected=True, verdicts={"healthy": 1, "unhealthy": 0, "unobserved": 0})
    decision = decide_from_collected(got, max_infra_retries=3, max_walltime_s=None, live_job=False)
    assert quality_inputs(decision, got, max_unhealthy_fraction=0.02)[0] is not True


@pytest.mark.parametrize("sources", [["other.jsonl"], ["rollouts.jsonl", "other.jsonl"], None])
def test_review_quality_from_another_rollout_is_not_accepted(tmp_path, sources):
    from nemo_gym.eval.recovery.gate import quality_inputs

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
    from nemo_gym.eval.recovery.continuation import accounted_chain_walltime

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
    from nemo_gym.eval.recovery.continuation import accounted_chain_walltime

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
    from nemo_gym.eval.recovery.gate import gate_summary_from_collected

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


def test_review_retryable_failures_are_not_labelled_infrastructure(tmp_path):
    job = write_job_dir(tmp_path / "job", n=2, results=[ident(0, reward=1.0)], failures=[ident(1, **INFRA)])
    d = decide_from_collected(collect(job), max_infra_retries=3, max_walltime_s=None, live_job=False)
    task = gate_task_summary(
        d, score=None, summary_valid=None, summary_reasons=[], health_passed=None, health_reasons=[]
    )
    assert any("retryable failures awaiting another attempt" in e for e in task["errors"])
    assert not any("infra" in e for e in task["errors"])


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
    from nemo_gym.eval.recovery.policy import _saved_answer

    assert [_saved_answer(row) for row in got.failures] == [True, False]
    assert got.failures[0]["response"]["id"] == SAVED_ANSWER["id"]
    assert "content" not in got.failures[0]["response"]["output"][0]
    assert "response" not in got.failures[1]
    assert '"text"' not in got.model_dump_json()


def test_review_retry_policy_cannot_exceed_spent_attempts(tmp_path):
    job = write_job_dir(tmp_path / "job", n=1, results=[], failures=[ident(0, **INFRA)] * 3)
    d = _decide(job, RecoveryOutcomePolicyConfig(default_action="retry"))
    assert d.identities["exhausted"] == [(0, 0)] and d.identities.get("retryable") is None


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
    from nemo_gym.eval.recovery.decision import IdentityState, RecoveryAction, RecoveryDecision
    from nemo_gym.eval.recovery.evidence import ROLLOUT_INDEX, TASK_INDEX, CollectedEvidence
    from nemo_gym.eval.recovery.trials import _observed_trial_operations

    row = {TASK_INDEX: 0, ROLLOUT_INDEX: 0}
    decision = RecoveryDecision(
        action=RecoveryAction.NOOP,
        reason="scored",
        expected=1,
        counts={IdentityState.SCORED.value: 1},
        identities={IdentityState.SCORED.value: [(0, 0)]},
        evidence=[],
    )
    collected = CollectedEvidence(
        materialized=[row], results=[row], failures=[], quality={"source_matches": True, "provenance": provenance}
    )
    assert _observed_trial_operations(decision, collected) is False


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
    from nemo_gym.eval.recovery.policy import _saved_answer

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
    from nemo_gym.eval.recovery.gate import (
        mlflow_errors,
        mlflow_notes,
    )

    local = {"live_run_id": "live-run", "receipt": _CURRENT_RECEIPT}
    assert mlflow_errors(local, "FINISHED", observed_run_id="live-run") == []
    assert mlflow_notes(local) == [
        "MLflow live run live-run and post-run export run postrun-run both exist: the post-run export receipt is "
        "current for the saved artifacts, so it is the run of record"
    ]


def test_both_runs_with_a_stale_receipt_fail_naming_the_stale_one() -> None:
    from nemo_gym.eval.recovery.gate import mlflow_errors

    stale = {"live_run_id": "live-run", "receipt": {**_CURRENT_RECEIPT, "current": False}}
    errors = mlflow_errors(stale, "FINISHED", observed_run_id="live-run")
    assert (
        "MLflow live run live-run and a post-run export both exist, and the post-run export run postrun-run is "
        "stale: its receipt is not current for the saved artifacts, so neither is the run of record"
    ) in errors
    assert not any("names neither" in error for error in errors)


def _mlflow_gate(tmp_path: Path, *, tracking_required: bool | None) -> dict:
    """The gate of a complete run that exported nothing to MLflow, with the snapshot's tracking requirement."""
    from nemo_gym.eval.recovery.gate import gate_summary_from_collected

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
    from nemo_gym.eval.recovery.gate import gate_summary_from_collected

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
    from nemo_gym.eval.recovery.gate import trial_health

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
    from nemo_gym.eval.recovery.gate import gate_summary_from_collected

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


def test_an_all_skipped_run_is_a_complete_noop_and_the_gate_reads_unverified(tmp_path) -> None:
    from nemo_gym.eval.recovery.gate import gate_summary_from_collected

    collected = collect(_skipped_job(tmp_path))
    decision = decide_from_collected(collected, max_infra_retries=3, max_walltime_s=None, live_job=False)
    assert decision.action == RecoveryAction.NOOP and decision.complete and decision.unverified
    assert "5 result(s) not graded by configuration (judge_skipped)" in decision.reason
    assert _SKIP_REASON in decision.reason

    task = gate_summary_from_collected(decision, collected, max_unhealthy_fraction=0.02)
    assert task["status"] == "UNVERIFIED" and task["score"] is None and task["errors"] == []
    assert task["scored_samples"] == 0 and decision.reason in task["warnings"]


def test_a_masked_result_of_any_other_kind_still_stops_the_run(tmp_path) -> None:
    from nemo_gym.eval.recovery.gate import gate_summary_from_collected

    collected = collect(_skipped_job(tmp_path, kind="provider_unavailable", n=2))
    decision = decide_from_collected(collected, max_infra_retries=3, max_walltime_s=None, live_job=False)
    assert not decision.unverified and not decision.complete
    task = gate_summary_from_collected(decision, collected, max_unhealthy_fraction=0.02)
    assert task["status"] != "UNVERIFIED"


@pytest.mark.parametrize(
    "marker, ended_itself",
    [
        (None, False),
        # What a job writes when it ends its own chain with nothing eligible for another attempt.
        ("no_eligible_retry", True),
        ("Continuation 10 started; follow-up not yet queued", False),
        ("GYM_ACCOUNTING_UNKNOWN: rejudge acknowledgement unknown", False),
    ],
)
def test_the_collector_names_a_chain_that_ended_itself(tmp_path, marker, ended_itself) -> None:
    from nemo_gym.eval.recovery.markers import NO_ELIGIBLE_RETRY_REASON

    job = write_job_dir(tmp_path, n=1, results=[], failures=[], chain=["10"])
    if marker is not None:
        text = NO_ELIGIBLE_RETRY_REASON if marker == "no_eligible_retry" else marker
        (job / ".gym_chain_incomplete").write_text(text + "\n")
    collected = collect(job)
    assert collected.chain_incomplete is (marker is not None)
    assert collected.chain_ended_itself is ended_itself
