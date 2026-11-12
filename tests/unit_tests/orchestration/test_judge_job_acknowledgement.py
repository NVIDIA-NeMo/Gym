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
"""`gym eval resubmit --recover --acknowledge-failed-judge-jobs --acknowledge-reason TEXT`: after the last two judge-only
jobs ended without settling a saved answer, an explicit, journaled acknowledgement lets exactly one more queue.

The acknowledgement names the failed jobs it covers; a later pair of failures needs one of its own.
"""

import json

import pytest

from nemo_gym.orchestration.recovery import (
    ACKNOWLEDGE_JUDGE_JOBS_FLAGS,
    JUDGE_JOB_ACK_JOURNAL_NAME,
    RecoveryAction,
    failed_judge_jobs_blocking,
    failed_judge_jobs_stop,
)
from nemo_gym.orchestration.submit import inspect_recovery, recover
from tests.unit_tests.orchestration.test_eval_resubmit import _handle
from tests.unit_tests.orchestration.test_eval_resubmit import (
    _local_index as _local_index,  # noqa: F401 - autouse fixture
)
from tests.unit_tests.orchestration.test_slurm_judge_only_job import _journal, _model_served_run
from tests.unit_tests.orchestration.test_slurm_judge_only_job import (
    judge_slurm_bin as judge_slurm_bin,  # noqa: F401 - fixture
)


FLAGS = "--acknowledge-failed-judge-jobs --acknowledge-reason"


def _acknowledgements(job_dir):
    path = job_dir / JUDGE_JOB_ACK_JOURNAL_NAME
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def _sbatch_count(bins):
    return int((bins / "sbatch_count").read_text())


def _fail_jobs(bins, *job_ids):
    (bins / "judge_accounting").write_text("".join(f"{job_id}|FAILED|30|1:0\n" for job_id in job_ids))


def _blocked_run(tmp_path, bins):
    """A run whose last two judge-only jobs (4242, 4243) ended without a receipt: recovery queues no third."""
    job_dir = _model_served_run(tmp_path)
    assert recover(*_handle(job_dir), judge_partition="cpu").job_id == "4242"
    _fail_jobs(bins, "4242")
    assert recover(*_handle(job_dir), judge_partition="cpu").job_id == "4243"
    _fail_jobs(bins, "4242", "4243")
    return job_dir


def test_the_guard_stops_without_an_acknowledgement_and_names_the_way_on(tmp_path, judge_slurm_bin):
    job_dir = _blocked_run(tmp_path, judge_slurm_bin)
    out = recover(*_handle(job_dir), judge_partition="cpu")
    assert not out.ok and out.dispatched is None and out.decision.action == RecoveryAction.STOP
    assert "4242, 4243" in out.unsupported and "before another is queued" in out.unsupported
    assert f"{FLAGS} " in out.unsupported and "allows one more" in out.unsupported
    assert _sbatch_count(judge_slurm_bin) == 2 and _acknowledgements(job_dir) == []


def test_an_acknowledgement_lets_exactly_one_more_job_queue_and_is_journaled(tmp_path, judge_slurm_bin, monkeypatch):
    from nemo_gym.orchestration.executors import slurm as slurm_module

    monkeypatch.setattr(slurm_module.getpass, "getuser", lambda: "operator-a")
    job_dir = _blocked_run(tmp_path, judge_slurm_bin)

    out = recover(
        *_handle(job_dir),
        judge_partition="cpu",
        acknowledge_failed_judge_jobs=True,
        acknowledge_reason="  read the receipts and logs  ",
    )

    assert out.ok and out.dispatched == "reverify" and out.job_id == "4244"
    assert _sbatch_count(judge_slurm_bin) == 3
    [entry] = _acknowledgements(job_dir)
    assert entry["job_ids"] == ["4242", "4243"] and entry["reason"] == "read the receipts and logs"
    assert entry["user"] == "operator-a" and entry["at"]
    # The judge-job journal keeps one entry per job: the acknowledgement is a line of its own journal.
    assert [job["job_id"] for job in _journal(job_dir)] == ["4242", "4243", "4244"]


def test_a_third_failure_stops_again_and_the_earlier_acknowledgement_does_not_cover_it(tmp_path, judge_slurm_bin):
    job_dir = _blocked_run(tmp_path, judge_slurm_bin)
    recover(
        *_handle(job_dir), judge_partition="cpu", acknowledge_failed_judge_jobs=True, acknowledge_reason="read them"
    )
    _fail_jobs(judge_slurm_bin, "4242", "4243", "4244")

    out = recover(*_handle(job_dir), judge_partition="cpu")

    assert not out.ok and out.dispatched is None
    assert "4243, 4244" in out.unsupported and FLAGS in out.unsupported
    assert _sbatch_count(judge_slurm_bin) == 3 and len(_acknowledgements(job_dir)) == 1
    # A new acknowledgement, naming the new pair, lets one more through.
    again = recover(
        *_handle(job_dir), judge_partition="cpu", acknowledge_failed_judge_jobs=True, acknowledge_reason="read again"
    )
    assert again.dispatched == "reverify" and again.job_id == "4245"
    assert [entry["job_ids"] for entry in _acknowledgements(job_dir)] == [["4242", "4243"], ["4243", "4244"]]


def test_an_acknowledgement_naming_other_jobs_does_not_unblock(tmp_path, judge_slurm_bin):
    job_dir = _blocked_run(tmp_path, judge_slurm_bin)
    (job_dir / JUDGE_JOB_ACK_JOURNAL_NAME).write_text(
        json.dumps({"at": "x", "user": "u", "reason": "r", "job_ids": ["9001", "9002"]})
        + "\n"
        + json.dumps({"at": "x", "user": "u", "reason": "r", "job_ids": ["4242"]})
        + "\n"
    )
    out = recover(*_handle(job_dir), judge_partition="cpu")
    assert not out.ok and out.dispatched is None and "4242, 4243" in out.unsupported
    assert _sbatch_count(judge_slurm_bin) == 2


@pytest.mark.parametrize("reason", [None, "", "   "])
def test_a_missing_or_empty_reason_is_refused_and_writes_nothing(tmp_path, judge_slurm_bin, reason):
    job_dir = _blocked_run(tmp_path, judge_slurm_bin)
    with pytest.raises(ValueError, match="needs a nonempty --acknowledge-reason"):
        recover(
            *_handle(job_dir),
            judge_partition="cpu",
            acknowledge_failed_judge_jobs=True,
            acknowledge_reason=reason,
        )
    assert _acknowledgements(job_dir) == [] and _sbatch_count(judge_slurm_bin) == 2


def test_a_reason_without_the_flag_is_refused(tmp_path, judge_slurm_bin):
    job_dir = _blocked_run(tmp_path, judge_slurm_bin)
    with pytest.raises(ValueError, match="goes with --acknowledge-failed-judge-jobs"):
        recover(*_handle(job_dir), judge_partition="cpu", acknowledge_reason="why")
    assert _acknowledgements(job_dir) == []


def test_an_acknowledgement_is_not_written_when_no_job_is_queued(tmp_path, judge_slurm_bin):
    job_dir = _blocked_run(tmp_path, judge_slurm_bin)
    (judge_slurm_bin / "sbatch").write_text(
        "#!/bin/bash\necho 'sbatch: error: Batch job submission failed' >&2; exit 1\n"
    )
    out = recover(
        *_handle(job_dir), judge_partition="cpu", acknowledge_failed_judge_jobs=True, acknowledge_reason="read them"
    )
    assert not out.ok and out.dispatched is None
    assert _acknowledgements(job_dir) == [] and [job["job_id"] for job in _journal(job_dir)] == ["4242", "4243"]


def test_the_flag_without_a_block_writes_no_acknowledgement(tmp_path, judge_slurm_bin):
    job_dir = _model_served_run(tmp_path)
    out = recover(
        *_handle(job_dir), judge_partition="cpu", acknowledge_failed_judge_jobs=True, acknowledge_reason="why"
    )
    assert out.dispatched == "reverify" and _acknowledgements(job_dir) == []


@pytest.mark.parametrize("flagged", [False, True])
def test_inspect_names_the_blocked_state_and_the_flag_and_writes_nothing(tmp_path, judge_slurm_bin, flagged):
    job_dir = _blocked_run(tmp_path, judge_slurm_bin)
    before = {p: p.read_bytes() for p in job_dir.rglob("*") if p.is_file()}
    options = {"acknowledge_failed_judge_jobs": True, "acknowledge_reason": "why"} if flagged else {}

    outcome = inspect_recovery(*_handle(job_dir), **options)

    assert outcome.dispatched is None
    assert "4242, 4243" in outcome.decision.reason and FLAGS in outcome.decision.reason
    assert {p: p.read_bytes() for p in job_dir.rglob("*") if p.is_file()} == before
    assert _sbatch_count(judge_slurm_bin) == 2


def test_inspect_does_not_name_a_block_once_an_acknowledgement_covers_it(tmp_path, judge_slurm_bin):
    job_dir = _blocked_run(tmp_path, judge_slurm_bin)
    recover(
        *_handle(job_dir), judge_partition="cpu", acknowledge_failed_judge_jobs=True, acknowledge_reason="read them"
    )
    assert FLAGS not in inspect_recovery(*_handle(job_dir)).decision.reason


def _argv(*extra):
    return ["eval", "resubmit", "job", "bench", *extra]


@pytest.mark.parametrize(
    "extra,message",
    [
        (["--acknowledge-failed-judge-jobs", "--acknowledge-reason", "why"], "need --recover"),
        (["--recover", "--acknowledge-failed-judge-jobs"], "needs a nonempty --acknowledge-reason"),
        (["--recover", "--acknowledge-failed-judge-jobs", "--acknowledge-reason", " "], "needs a nonempty"),
        (
            ["--recover", "--acknowledge-reason", "why"],
            "--acknowledge-reason goes with --acknowledge-failed-judge-jobs",
        ),
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
    from nemo_gym.orchestration.recovery import RecoveryDecision, RecoveryOutcome

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
    extra = ["--recover", "--acknowledge-failed-judge-jobs", "--acknowledge-reason", " why "] + (
        ["--inspect"] if inspect_only else []
    )
    try:
        _eval_resubmit(build_parser().parse_args(_argv(*extra)), [])
    except SystemExit:
        pass
    called = seen["inspect" if inspect_only else "recover"]
    assert called["acknowledge_failed_judge_jobs"] is True and called["acknowledge_reason"] == "why"
    capsys.readouterr()


def test_a_blocking_job_without_an_id_cannot_be_acknowledged_and_the_stop_says_so():
    # An acknowledgement names the jobs it covers, so a pair with an id-less job stays blocked and the stop offers no flag.
    jobs = [{"state": "failed"}, {"state": "failed", "job_id": "12"}]

    blocked = failed_judge_jobs_blocking(jobs, [{"job_ids": ["12"]}])

    assert blocked == [None, "12"]
    stop = failed_judge_jobs_stop(blocked, "reverification")
    assert "(id unknown), 12" in stop and "cannot be acknowledged" in stop
    assert ACKNOWLEDGE_JUDGE_JOBS_FLAGS not in stop


def test_the_verdict_names_the_block_and_the_acknowledgement_before_and_after_a_refused_recovery(
    tmp_path, judge_slurm_bin
):
    job_dir = _blocked_run(tmp_path, judge_slurm_bin)
    reason = (
        "the last 2 judge-only jobs (4242, 4243) ended without settling a saved answer, so recovery queues no "
        "other until they are read and acknowledged"
    )

    inspected = inspect_recovery(*_handle(job_dir))
    refused = recover(*_handle(job_dir), judge_partition="cpu")

    assert inspected.decision.action == RecoveryAction.REVERIFY and refused.decision.action == RecoveryAction.STOP
    # The acknowledgement command repeats the options of the call it answers (the refused one named a partition).
    for out, where, asked in (
        (inspected, "the run's reverification directory", ""),
        (refused, str(job_dir / "reverification"), " --judge-partition cpu"),
    ):
        assert out.operator_verdict == {
            "state": "needs_action",
            "reason": reason,
            "action": f"read their receipts and logs under {where}, then "
            f"`gym eval resubmit gym-job-x bench --recover{asked} {FLAGS} TEXT` allows one more judge-only job",
        }
    assert _sbatch_count(judge_slurm_bin) == 2 and _acknowledgements(job_dir) == []


def test_the_verdict_of_a_block_without_a_job_id_offers_no_acknowledgement():
    from nemo_gym.orchestration.recovery import IdentityState, RecoveryDecision, operator_verdict

    stop = failed_judge_jobs_stop([None, "12"], "reverification")
    decision = RecoveryDecision(
        action=RecoveryAction.REVERIFY,
        reason=f"1 saved answer(s) can be re-judged; {stop}",
        expected=2,
        counts={IdentityState.SCORED: 1, IdentityState.JUDGE_RECOVERABLE: 1},
        identities={},
        evidence=[],
    )
    verdict = operator_verdict({"status": "PLANNED", "expected_samples": 2, "errors": []}, decision)
    assert verdict["state"] == "needs_action" and "(id unknown), 12" in verdict["reason"]
    assert ACKNOWLEDGE_JUDGE_JOBS_FLAGS not in verdict["action"] and "recover` now" not in verdict["action"]
    assert verdict["action"] == (
        "read their receipts and logs under reverification; a job without an id cannot be acknowledged, so no "
        "further judge-only job is allowed and no recovery command is known"
    )


@pytest.mark.parametrize("inspect_only", [True, False])
def test_the_command_prints_one_verdict_in_the_json_the_gate_and_stderr(
    tmp_path, judge_slurm_bin, capsys, inspect_only
):
    from nemo_gym.cli.main import _eval_resubmit, build_parser
    from nemo_gym.orchestration.recovery import operator_verdict_line

    job_dir = _blocked_run(tmp_path, judge_slurm_bin)
    argv = ["eval", "resubmit", *_handle(job_dir), "--recover"]
    argv += ["--inspect"] if inspect_only else ["--judge-partition", "cpu"]
    capsys.readouterr()
    with pytest.raises(SystemExit) as error:
        _eval_resubmit(build_parser().parse_args(argv), [])
    assert error.value.code == 1

    out, err = capsys.readouterr()
    payload = json.loads(out)
    verdict = payload["operator_verdict"]
    assert verdict is not None and verdict["state"] == "needs_action" and "4242, 4243" in verdict["reason"]
    assert payload["gate_task"]["operator_verdict"] == verdict
    assert operator_verdict_line(verdict) in err.splitlines()
    assert _sbatch_count(judge_slurm_bin) == 2
