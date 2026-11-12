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
"""`gym eval resubmit --recover --rerun-masked-without-artifact --rerun-reason TEXT`: an explicit, journaled rerun of
a masked result that kept a saved answer but has neither a gradeable artifact nor a recorded grade.

Without the flag such a row is retained (masked, never run again, not measured); with it, the rows the planner finds
qualifying at that moment are journaled, and the Store and the planner then treat each like an infrastructure-masked
result: one failed attempt, retried within its attempt cap, superseded by its next attempt.
"""

import hashlib
import json

import pytest

from nemo_gym.orchestration.recovery import (
    RERUN_MASKED_JOURNAL_NAME,
    decide_from_collected,
    masked_without_artifact_kind,
    select_managed_evidence,
)
from nemo_gym.rollouts.rollout_store import RolloutStore
from tests.unit_tests.orchestration.test_eval_resubmit import _handle, _resumable_run
from tests.unit_tests.orchestration.test_eval_resubmit import _local_index as _local_index
from tests.unit_tests.orchestration.test_eval_resubmit import slurm_bin as slurm_bin
from tests.unit_tests.orchestration.test_managed_recovery_selection import collect
from tests.unit_tests.orchestration.test_masked_infrastructure_recovery import _ANSWER, _masked, _run


FLAG = "--rerun-masked-without-artifact --rerun-reason"


def _kept(kind="verifier_error", **fields):
    """A masked result that kept its saved answer: no artifact to grade and no recorded grade."""
    return _masked(kind) | _ANSWER | fields


def _journal(job, *entries):
    path = job / "artifacts" / RERUN_MASKED_JOURNAL_NAME
    path.write_text("".join(json.dumps(entry) + "\n" for entry in entries))
    return path


def _approval(identity=(1, 0), attempt=0, attempts=0, reason="the patch artifact is missing"):
    return {
        "at": "2026-01-01T00:00:00Z",
        "reason": reason,
        "identities": [{"identity": list(identity), "attempt": attempt, "attempts": attempts}],
    }


def _decide(job, **options):
    selected = select_managed_evidence(collect(job), job)
    assert selected.managed_reader_error is None
    return decide_from_collected(selected, max_infra_retries=3, max_walltime_s=None, live_job=False, **options)


def _evidence(decision, identity):
    return next(item for item in decision.evidence if item.identity == identity)


# The predicate.


def test_a_masked_row_that_kept_its_answer_with_no_artifact_and_no_grade_qualifies():
    assert masked_without_artifact_kind(_kept("verifier_error")) == "verifier_error"
    assert masked_without_artifact_kind(_kept("session_lost")) == "session_lost"


@pytest.mark.parametrize(
    "row",
    [
        pytest.param(_masked("session_lost"), id="no-saved-answer-is-an-ordinary-infrastructure-retry"),
        pytest.param(_kept(evaluation_completed=True), id="graded"),
        pytest.param(_kept("judge_failed"), id="judge-regrade-kind"),
        pytest.param(_kept("transport_error"), id="verify-transport-kind"),
        pytest.param(_kept("agent_timeout"), id="model-kind"),
        pytest.param(_kept("model_context_limit"), id="model-kind-2"),
        pytest.param(_kept("env:own_kind"), id="namespaced-kind"),
        pytest.param(_kept("not_registered"), id="unregistered-kind"),
        pytest.param({k: v for k, v in _kept().items() if k != "failure_kind"}, id="no-kind"),
        pytest.param(_kept(mask_sample=False), id="not-masked"),
        pytest.param(_kept(_ng_failure_class="x"), id="failure-row"),
        pytest.param(_kept(_ng_omitted_rollout=True), id="omitted-stand-in"),
        pytest.param(_kept(_ng_rejudge_operation_id="op"), id="rejudged"),
    ],
)
def test_every_other_exclusion_still_applies(row):
    assert masked_without_artifact_kind(row) is None


def test_a_row_that_re_verification_can_grade_is_excluded_by_that_rule_alone():
    from nemo_gym.orchestration.recovery import reverification_cause
    from nemo_gym.rollouts.rollout_recovery import masked_kept_answer_kind

    patch = "diff --git a/x b/x\n"
    row = _kept(
        model_patch=patch,
        model_patch_sha256=hashlib.sha256(patch.encode()).hexdigest(),
        patch_collection_time_s=1.0,
        verification_time_s=2.0,
    )
    # Its saved patch is graded again by re-verification, so it is not a row without an artifact.
    assert reverification_cause(row) is not None
    assert masked_kept_answer_kind(row) == "verifier_error"
    assert masked_without_artifact_kind(row) is None


# The planner and its inspect report.


def test_without_the_flag_the_row_stays_masked_and_inspect_names_it_in_its_own_bucket(tmp_path):
    output, _ = _run(tmp_path, [[_kept()]])
    before = output.read_bytes()
    decision = _decide(tmp_path)
    assert (decision.action.value, decision.counts) == ("stop", {"scored": 1, "masked": 1})
    assert decision.masked_without_artifact == [(1, 0)]
    assert decision.rerun_masked == []
    item = _evidence(decision, (1, 0))
    assert item.state.value == "masked" and FLAG in (item.reason or "")
    assert "1 masked result(s) without a gradeable artifact" in decision.reason and FLAG in decision.reason
    assert output.read_bytes() == before
    assert not (tmp_path / "artifacts" / RERUN_MASKED_JOURNAL_NAME).exists()
    store = RolloutStore.read(output, import_legacy=False)
    assert store.pending() == [] and store.attempt_count({"_ng_task_index": 1, "_ng_rollout_index": 0}) == 0


def test_a_masked_row_without_artifact_keeps_an_operator_continuation_of_unknown_accounting(tmp_path):
    # The masked-without-artifact note extends the decision's display reason. The unknown accounting is still the
    # cause of the stop, so an explicit operator budget continues the run: the masked row stays retained (masked,
    # not rerun) and the missing one resumes.
    from nemo_gym.orchestration.recovery import OperatorBudget

    output, _ = _run(tmp_path, [[_kept()], []])
    (tmp_path / ".gym_infra_retries").write_text("unreadable" + chr(10))
    before = output.read_bytes()
    budget = OperatorBudget(
        user="example-operator", max_walltime="20:00:00", reason="continue after a lost submission reply", retries=2
    )

    decision = _decide(tmp_path, operator_budget=budget)

    assert decision.action.value == "resubmit", decision.reason
    assert decision.counts == {"scored": 1, "masked": 1, "missing": 1}
    assert decision.masked_without_artifact == [(1, 0)] and decision.rerun_masked == []
    assert decision.operator_continuation is not None
    assert "1 masked result(s) without a gradeable artifact" in decision.reason
    assert _evidence(decision, (1, 0)).state.value == "masked"
    assert output.read_bytes() == before


def test_a_masked_row_with_a_model_kind_is_not_in_the_bucket(tmp_path):
    _run(tmp_path, [[_kept("agent_timeout")]])
    decision = _decide(tmp_path)
    assert decision.counts == {"scored": 1, "masked": 1} and decision.masked_without_artifact == []
    assert _evidence(decision, (1, 0)).reason is None


def test_asking_for_the_rerun_plans_the_row_as_a_retry_and_lists_it_for_the_journal(tmp_path):
    _run(tmp_path, [[_kept()]])
    decision = _decide(tmp_path, rerun_masked_without_artifact=True)
    assert (decision.action.value, decision.counts) == ("resubmit", {"scored": 1, "retryable": 1})
    item = _evidence(decision, (1, 0))
    assert (item.failure_kind, item.attempts, item.max_attempts) == ("verifier_error", 1, 3)
    assert [(a.identity, a.attempts, a.attempt) for a in decision.rerun_masked] == [((1, 0), 0, 0)]
    assert decision.masked_without_artifact == []


def test_asking_for_the_rerun_never_exceeds_the_attempt_cap(tmp_path):
    # Two earlier infrastructure attempts spent two of the three; the masked row would be the third.
    _run(tmp_path, [[_masked("session_lost"), _masked("session_lost"), _kept()]])
    decision = _decide(tmp_path, rerun_masked_without_artifact=True)
    assert (decision.action.value, decision.counts) == ("stop", {"scored": 1, "exhausted": 1})
    item = _evidence(decision, (1, 0))
    assert (item.attempts, item.max_attempts) == (3, 3)
    # Nothing would run, so nothing is approved.
    assert decision.rerun_masked == []


def test_a_journaled_approval_makes_the_planner_and_the_store_retry_the_row(tmp_path):
    output, _ = _run(tmp_path, [[_kept()]])
    _journal(tmp_path, _approval())
    store = RolloutStore.read(output, import_legacy=False)
    [pending] = store.pending()
    assert (pending["_ng_task_index"], pending["_ng_attempt_index"]) == (1, 1)
    assert store.attempt_count({"_ng_task_index": 1, "_ng_rollout_index": 0}) == 1
    decision = _decide(tmp_path)
    assert (decision.action.value, decision.counts) == ("resubmit", {"scored": 1, "retryable": 1})
    item = _evidence(decision, (1, 0))
    assert (item.attempts, item.max_attempts) == (1, 3)
    assert decision.masked_without_artifact == [] and decision.rerun_masked == []


def test_a_journaled_approval_at_the_attempt_cap_is_exhausted_and_not_run(tmp_path):
    output, _ = _run(tmp_path, [[_masked("session_lost"), _masked("session_lost"), _kept()]])
    _journal(tmp_path, _approval(attempt=2, attempts=2))
    store = RolloutStore.read(output, import_legacy=False)
    assert store.pending() == []
    assert store.attempt_count({"_ng_task_index": 1, "_ng_rollout_index": 0}) == 3
    decision = _decide(tmp_path)
    assert (decision.action.value, decision.counts) == ("stop", {"scored": 1, "exhausted": 1})
    assert (_evidence(decision, (1, 0)).attempts, _evidence(decision, (1, 0)).max_attempts) == (3, 3)


def test_the_new_attempt_supersedes_the_approved_row_and_decides_for_itself(tmp_path):
    output, prepare = _run(tmp_path, [[_kept()]])
    _journal(tmp_path, _approval())
    with RolloutStore.start_or_resume(output, prepare, resume=True) as store:
        [row] = store.pending()
        store.record_dispatch(row)
        # The second attempt is masked with a kept answer again: the earlier approval does not reopen it.
        store.record_outcome(row | _kept())
    store = RolloutStore.read(output, import_legacy=False)
    assert store.pending() == []
    assert store.attempt_count({"_ng_task_index": 1, "_ng_rollout_index": 0}) == 1
    decision = _decide(tmp_path)
    assert (decision.action.value, decision.counts) == ("stop", {"scored": 1, "masked": 1})
    # It is a new row the operator has not approved: it is named again, never reopened silently.
    assert decision.masked_without_artifact == [(1, 0)]


def test_an_approval_covers_only_the_attempt_it_names_and_only_a_qualifying_row(tmp_path):
    output, _ = _run(tmp_path, [[_kept(evaluation_completed=True)], [_kept("agent_timeout")], [_kept()]])
    _journal(
        tmp_path,
        {
            "at": "x",
            "reason": "r",
            "identities": [
                {"identity": [1, 0], "attempt": 0, "attempts": 0},
                {"identity": [2, 0], "attempt": 0, "attempts": 0},
                {"identity": [3, 0], "attempt": 5, "attempts": 0},
            ],
        },
    )
    assert RolloutStore.read(output, import_legacy=False).pending() == []


def test_an_unreadable_journal_line_reopens_nothing(tmp_path):
    output, _ = _run(tmp_path, [[_kept()]])
    (tmp_path / "artifacts" / RERUN_MASKED_JOURNAL_NAME).write_text('{"identities": [\nnot json\n[1, 2]\n')
    assert RolloutStore.read(output, import_legacy=False).pending() == []


def test_a_store_reader_that_predates_the_rerun_is_refused(tmp_path):
    _run(tmp_path, [[_kept()]])
    selected = select_managed_evidence(collect(tmp_path), tmp_path)
    assert selected.rerun_masked_applied is True
    old = selected.model_copy(update={"rerun_masked_applied": None})
    decision = decide_from_collected(
        old, max_infra_retries=3, max_walltime_s=None, live_job=False, rerun_masked_without_artifact=True
    )
    assert decision.action.value == "stop"
    assert decision.reason.startswith("--rerun-masked-without-artifact is refused: the run's pinned Gym predates")
    assert decision.rerun_masked == []


# The recovery command.


def _staged(tmp_path, outcomes):
    job, _, _ = _resumable_run(tmp_path)
    for name in ("rollouts.jsonl", "rollouts_materialized_inputs.jsonl"):
        (job / "artifacts" / name).unlink()
    _run(job, outcomes)
    (job / ".gym_accumulated_walltime").unlink()
    (job / ".gym_job_chain").write_text("100\n101\n")
    return job


def _accounting(slurm_bin):
    (slurm_bin / "accounting").write_text("100|TIMEOUT|7200|0\n101|FAILED|60|0\n")


def test_recover_with_the_flag_journals_one_line_and_queues_the_job(tmp_path, slurm_bin, monkeypatch):
    from nemo_gym.orchestration.executors import slurm as slurm_module
    from nemo_gym.orchestration.submit import inspect_recovery, recover

    monkeypatch.setattr(slurm_module.getpass, "getuser", lambda: "operator-a")
    job = _staged(tmp_path, [[_kept()]])
    _accounting(slurm_bin)
    assert inspect_recovery(*_handle(job)).decision.action.value == "stop"

    out = recover(*_handle(job), rerun_masked_without_artifact=True, rerun_reason="  the patch artifact is missing  ")

    assert out.dispatched == "resubmit", out.decision.reason
    lines = (job / "artifacts" / RERUN_MASKED_JOURNAL_NAME).read_text().splitlines()
    [entry] = [json.loads(line) for line in lines]
    assert entry["reason"] == "the patch artifact is missing" and entry["at"] and entry["user"] == "operator-a"
    assert entry["identities"] == [{"identity": [1, 0], "attempt": 0, "attempts": 0}]
    # A later recovery reads the approval: the row is a retry with its attempt counted.
    later = inspect_recovery(*_handle(job))
    assert later.decision.action.value == "resubmit", later.decision.reason
    assert later.decision.counts == {"scored": 1, "retryable": 1}
    assert later.decision.masked_without_artifact == []


def test_inspect_reports_the_bucket_and_writes_nothing(tmp_path, slurm_bin):
    from nemo_gym.orchestration.submit import inspect_recovery

    job = _staged(tmp_path, [[_kept()]])
    _accounting(slurm_bin)
    before = {p: p.read_bytes() for p in job.rglob("*") if p.is_file()}

    outcome = inspect_recovery(*_handle(job))

    assert outcome.dispatched is None and outcome.decision.action.value == "stop"
    assert outcome.decision.masked_without_artifact == [(1, 0)]
    assert FLAG in outcome.decision.reason
    assert {p: p.read_bytes() for p in job.rglob("*") if p.is_file()} == before


def test_inspect_with_the_flag_still_writes_nothing_and_keeps_the_row_masked(tmp_path, slurm_bin):
    from nemo_gym.orchestration.submit import inspect_recovery

    job = _staged(tmp_path, [[_kept()]])
    _accounting(slurm_bin)

    outcome = inspect_recovery(*_handle(job), rerun_masked_without_artifact=True, rerun_reason="why")

    assert outcome.decision.counts == {"scored": 1, "masked": 1}
    assert outcome.decision.masked_without_artifact == [(1, 0)]
    assert not (job / "artifacts" / RERUN_MASKED_JOURNAL_NAME).exists()


def test_recover_without_the_flag_keeps_the_row_and_writes_nothing(tmp_path, slurm_bin):
    from nemo_gym.orchestration.submit import recover

    job = _staged(tmp_path, [[_kept()]])
    _accounting(slurm_bin)

    out = recover(*_handle(job))

    assert out.dispatched is None and out.decision.counts == {"scored": 1, "masked": 1}
    assert not (job / "artifacts" / RERUN_MASKED_JOURNAL_NAME).exists() and not (slurm_bin / "env").exists()


@pytest.mark.parametrize("reason", [None, "", "   "])
def test_recover_refuses_the_flag_without_a_reason_and_writes_nothing(tmp_path, slurm_bin, reason):
    from nemo_gym.orchestration.submit import recover

    job = _staged(tmp_path, [[_kept()]])
    _accounting(slurm_bin)
    with pytest.raises(ValueError, match="needs a nonempty --rerun-reason"):
        recover(*_handle(job), rerun_masked_without_artifact=True, rerun_reason=reason)
    assert not (job / "artifacts" / RERUN_MASKED_JOURNAL_NAME).exists() and not (slurm_bin / "env").exists()


def test_a_job_the_scheduler_refuses_leaves_no_approval(tmp_path, slurm_bin):
    from nemo_gym.orchestration.submit import recover

    job = _staged(tmp_path, [[_kept()]])
    _accounting(slurm_bin)
    journal = job / "artifacts" / RERUN_MASKED_JOURNAL_NAME
    earlier = json.dumps(_approval(identity=(9, 9))) + "\n"
    journal.write_text(earlier)
    (slurm_bin / "sbatch").write_text("#!/bin/bash\necho 'sbatch: error: Batch job submission failed'; exit 1\n")

    out = recover(*_handle(job), rerun_masked_without_artifact=True, rerun_reason="why")

    assert not out.ok and out.dispatched is None
    assert out.scheduler_refusal["job_dir_restored"] is True
    assert "rerun-masked approval" in out.scheduler_refusal["restored"]
    assert journal.read_text() == earlier


def test_the_flag_with_nothing_qualifying_writes_no_approval(tmp_path, slurm_bin):
    from nemo_gym.orchestration.submit import recover

    job = _staged(tmp_path, [[_masked("session_lost")]])
    _accounting(slurm_bin)

    out = recover(*_handle(job), rerun_masked_without_artifact=True, rerun_reason="why")

    assert out.dispatched == "resubmit"  # the ordinary infrastructure retry
    assert not (job / "artifacts" / RERUN_MASKED_JOURNAL_NAME).exists()


# The command line.


def _argv(*extra):
    return ["eval", "resubmit", "job", "bench", *extra]


@pytest.mark.parametrize(
    "extra,message",
    [
        (["--rerun-masked-without-artifact", "--rerun-reason", "why"], "need --recover"),
        (["--recover", "--rerun-masked-without-artifact"], "needs a nonempty --rerun-reason"),
        (["--recover", "--rerun-masked-without-artifact", "--rerun-reason", ""], "needs a nonempty --rerun-reason"),
        (["--recover", "--rerun-masked-without-artifact", "--rerun-reason", "  "], "needs a nonempty --rerun-reason"),
        (["--recover", "--rerun-reason", "why"], "--rerun-reason goes with --rerun-masked-without-artifact"),
    ],
)
def test_the_command_refuses_a_bad_combination_before_doing_anything(monkeypatch, capsys, extra, message):
    from nemo_gym.cli.main import _eval_resubmit, build_parser
    from nemo_gym.orchestration import submit

    def forbidden(*args, **kwargs):
        raise AssertionError("recovery must not start")

    monkeypatch.setattr(submit, "recover", forbidden)
    monkeypatch.setattr(submit, "inspect_recovery", forbidden)
    with pytest.raises(SystemExit):
        _eval_resubmit(build_parser().parse_args(_argv(*extra)), [])
    assert message in capsys.readouterr().err


@pytest.mark.parametrize("inspect_only", [False, True])
def test_the_command_passes_the_stripped_reason_to_recovery_and_inspect(monkeypatch, capsys, inspect_only):
    from nemo_gym.cli.main import _eval_resubmit, build_parser
    from nemo_gym.orchestration import submit
    from nemo_gym.orchestration.recovery import RecoveryAction, RecoveryDecision, RecoveryOutcome

    seen = {}

    def fake(name):
        def call(*args, **kwargs):
            seen[name] = kwargs
            decision = RecoveryDecision(
                action=RecoveryAction.NOOP, reason="r", expected=0, counts={}, identities={}, evidence=[]
            )
            return RecoveryOutcome(decision=decision, ok=True, dispatched=None)

        return call

    monkeypatch.setattr(submit, "recover", fake("recover"))
    monkeypatch.setattr(submit, "inspect_recovery", fake("inspect"))
    extra = ["--recover", "--rerun-masked-without-artifact", "--rerun-reason", " why "] + (
        ["--inspect"] if inspect_only else []
    )
    try:
        _eval_resubmit(build_parser().parse_args(_argv(*extra)), [])
    except SystemExit:
        pass
    called = seen["inspect" if inspect_only else "recover"]
    assert called["rerun_masked_without_artifact"] is True and called["rerun_reason"] == "why"
    capsys.readouterr()


def test_an_approval_belongs_to_the_rollouts_file_it_was_journaled_beside(tmp_path):
    # Two rollouts files in one directory: an approval journaled for one does not reopen the other's rows.
    from nemo_gym.rollouts.rollout_recovery import load_rerun_masked_approvals
    from nemo_gym.utils.path_utils import rerun_masked_journal_path_for

    first, second = tmp_path / "rollouts.jsonl", tmp_path / "other_rollouts.jsonl"
    rerun_masked_journal_path_for(first).write_text(
        json.dumps({"reason": "r", "identities": [{"identity": [3, 1], "attempt": 0, "attempts": 1}]}) + "\n"
    )

    assert rerun_masked_journal_path_for(first).name == RERUN_MASKED_JOURNAL_NAME
    assert load_rerun_masked_approvals(first) == frozenset({(3, 1, 0)})
    assert load_rerun_masked_approvals(second) == frozenset()
