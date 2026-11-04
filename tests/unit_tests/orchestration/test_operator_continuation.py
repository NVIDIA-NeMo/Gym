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
"""`gym eval resubmit --recover --max-walltime T --walltime-reason R [--extend-retries N]` for a run whose chain
accounting is unknown: an explicit, journaled OPERATOR-CONTINUATION that fills only the unknown counters."""

import hashlib
import json

import pytest

from nemo_gym.orchestration.executors.resume_script import ACCOUNTING_UNKNOWN_PREFIX, RETRY_EXTENSION_NAME
from nemo_gym.orchestration.executors.slurm import SlurmExecutor
from nemo_gym.orchestration.jobs import load_submission_record
from nemo_gym.orchestration.recovery import (
    OPERATOR_CONTINUATION_JOURNAL_NAME,
    OPERATOR_CONTINUATION_NOTE,
    UNKNOWN_BUDGET_HINT,
    UNKNOWN_BUDGET_REASON,
)
from tests.unit_tests.orchestration.test_eval_resubmit import _handle
from tests.unit_tests.orchestration.test_eval_resubmit import _local_index as _local_index
from tests.unit_tests.orchestration.test_eval_resubmit import slurm_bin as slurm_bin
from tests.unit_tests.orchestration.test_resubmit_repin import launcher_at as launcher_at
from tests.unit_tests.orchestration.test_resubmit_walltime import (
    _launcher_at_the_pinned_revision as _launcher_at_the_pinned_revision,
)
from tests.unit_tests.orchestration.test_resubmit_walltime import _timed_run


MARKER = f"{ACCOUNTING_UNKNOWN_PREFIX} Resubmit acknowledgement was not confirmed.\n"
BUDGET = {"max_walltime": "20:00:00", "walltime_reason": "continue after a lost submission reply"}
ACCOUNTING_FILES = (
    ".gym_infra_retries",
    ".gym_accumulated_walltime",
    RETRY_EXTENSION_NAME,
    ".gym_chain_incomplete",
    ".gym_job_chain",
    OPERATOR_CONTINUATION_JOURNAL_NAME,
    "job.sh",
)


@pytest.fixture(autouse=True)
def _operator(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LOGNAME", "example-operator")


def _unknown_run(tmp_path, *, unreadable=(".gym_infra_retries",)):
    """The walltime test run (2 of 3 retries and 7200 s spent, one row missing) with its accounting unknown."""
    job_dir = _timed_run(tmp_path)
    (job_dir / ".gym_chain_incomplete").write_text(MARKER)
    for name in unreadable:
        (job_dir / name).write_text("unreadable\n")
    return job_dir


def _saved_rows(job_dir) -> dict[str, str]:
    names = ("artifacts/rollouts.jsonl", "artifacts/rollouts_materialized_inputs.jsonl")
    return {name: hashlib.sha256((job_dir / name).read_bytes()).hexdigest() for name in names}


def _accounting(job_dir) -> dict[str, str | None]:
    return {name: (job_dir / name).read_text() if (job_dir / name).exists() else None for name in ACCOUNTING_FILES}


def test_an_explicit_budget_continues_the_run_and_records_it_in_the_decision_and_the_journal(tmp_path, slurm_bin):
    from nemo_gym.orchestration.submit import recover

    job_dir = _unknown_run(tmp_path)
    saved = _saved_rows(job_dir)

    out = recover(*_handle(job_dir), extend_retries=2, **BUDGET)

    assert (out.dispatched, out.job_id) == ("resubmit", "4242"), out.decision.reason
    record = out.decision.operator_continuation
    assert record == {
        "kind": "OPERATOR-CONTINUATION",
        "user": "example-operator",
        "reason": "continue after a lost submission reply",
        "budget": {"infra_retries_remaining": 2, "infra_retries_requested": 2, "max_walltime": "20:00:00"},
        "unknown": ["infra_retries_used"],
        "accounting_unknown_reason": "Resubmit acknowledgement was not confirmed.",
        "writes": {".gym_infra_retries": {"previous": "unknown", "value": 1}},
        "epoch": {"chain_length": 2, "walltime_s": 7200.0},
    }
    assert "OPERATOR-CONTINUATION by example-operator (continue after a lost submission reply)" in out.decision.reason
    [entry] = [json.loads(line) for line in (job_dir / OPERATOR_CONTINUATION_JOURNAL_NAME).read_text().splitlines()]
    assert entry == record | {"job_id": "4242", "at": entry["at"]} and entry["at"]
    # N=2 retries remain from now: one of the three configured ones counts as spent. The note replaces the marker.
    assert (job_dir / ".gym_infra_retries").read_text() == "1\n"
    assert (job_dir / ".gym_chain_incomplete").read_text() == OPERATOR_CONTINUATION_NOTE + "\n"
    assert not (job_dir / RETRY_EXTENSION_NAME).exists() and not list(job_dir.glob("*.operator-previous"))
    # The saved rows stay as they are: the scored row is kept, only the missing one runs again.
    assert _saved_rows(job_dir) == saved
    assert out.decision.counts == {"scored": 1, "missing": 1}


@pytest.mark.parametrize(
    "unreadable,filled,known,kept",
    [
        (".gym_infra_retries", "2\n", ".gym_accumulated_walltime", "7200\n"),
        (".gym_accumulated_walltime", "0\n", ".gym_infra_retries", "2\n"),
    ],
)
def test_a_continuation_fills_only_the_unknown_counter_and_keeps_the_known_one(
    tmp_path, slurm_bin, unreadable, filled, known, kept
):
    from nemo_gym.orchestration.submit import recover

    job_dir = _unknown_run(tmp_path, unreadable=(unreadable,))

    out = recover(*_handle(job_dir), **BUDGET)

    assert out.dispatched == "resubmit", out.decision.reason
    assert (job_dir / unreadable).read_text() == filled
    assert (job_dir / known).read_text() == kept
    assert list(out.decision.operator_continuation["writes"]) == [unreadable]


def test_without_a_budget_unknown_accounting_stops_as_before_and_names_the_options(tmp_path, slurm_bin):
    from nemo_gym.orchestration.submit import inspect_recovery, recover

    job_dir = _unknown_run(tmp_path)
    before = _accounting(job_dir)

    for out in (inspect_recovery(*_handle(job_dir)), recover(*_handle(job_dir))):
        assert (out.decision.action.value, out.dispatched) == ("stop", None)
        assert out.decision.reason == f"{UNKNOWN_BUDGET_REASON}\n{UNKNOWN_BUDGET_HINT}"
        assert out.decision.operator_continuation is None
    # More retries without a walltime and its reason are not an operator budget.
    extended = recover(*_handle(job_dir), extend_retries=1)
    assert (extended.decision.action.value, extended.dispatched) == ("stop", None)
    assert extended.decision.reason.startswith(UNKNOWN_BUDGET_REASON)
    assert extended.decision.reason.endswith(UNKNOWN_BUDGET_HINT)
    assert _accounting(job_dir) == before and not (slurm_bin / "env").exists()


@pytest.mark.parametrize("live", ["queued in the job dir", "requeued chain job"])
def test_a_live_or_pending_job_of_the_run_still_stops_the_continuation(tmp_path, slurm_bin, live):
    from nemo_gym.orchestration.submit import recover

    job_dir = _unknown_run(tmp_path)
    if live == "queued in the job dir":
        (slurm_bin / "squeue_workdirs").write_text(f"{job_dir}\n")  # a follow-up still waits there
    else:
        (slurm_bin / "accounting").write_text("100|FAILED|7200|0\n101|PENDING|0|1\n")
    before = _accounting(job_dir)

    out = recover(*_handle(job_dir), **BUDGET)

    assert (out.decision.action.value, out.dispatched, out.ok) == ("stop", None, False)
    job = "continuation 101 (PENDING)" if live == "requeued chain job" else "a job of this run"
    assert out.decision.reason.startswith(
        f"the chain accounting is unknown and {job} is still pending or running, "
        "so an operator continuation is refused;"
    )
    assert _accounting(job_dir) == before and not (slurm_bin / "env").exists()


def test_an_unreachable_scheduler_stops_the_continuation_and_says_so(tmp_path, slurm_bin):
    from nemo_gym.orchestration.submit import recover

    job_dir = _unknown_run(tmp_path)
    (slurm_bin / "squeue").write_text(
        "#!/bin/bash\necho 'squeue: error: Unable to contact the controller' >&2\nexit 1\n"
    )
    before = _accounting(job_dir)

    out = recover(*_handle(job_dir), **BUDGET)

    assert (out.decision.action.value, out.dispatched) == ("stop", None)
    assert out.decision.reason.startswith(
        f"{UNKNOWN_BUDGET_REASON}; the scheduler query failed, so an operator continuation cannot rule out a live "
        "or pending job;"
    )
    assert _accounting(job_dir) == before and not (slurm_bin / "env").exists()
    # Without an operator budget, the failed query is an error as before.
    with pytest.raises(RuntimeError, match="Command failed"):
        recover(*_handle(job_dir))


def test_a_continuation_whose_job_is_not_queued_changes_nothing(tmp_path, slurm_bin):
    from nemo_gym.orchestration.submit import recover

    job_dir = _unknown_run(tmp_path, unreadable=(".gym_infra_retries", ".gym_accumulated_walltime"))
    before = _accounting(job_dir)
    (slurm_bin / "sbatch").write_text("#!/bin/bash\necho 'sbatch: error: Batch job submission failed'; exit 1\n")

    out = recover(*_handle(job_dir), **BUDGET)
    assert not out.ok and out.dispatched is None
    assert out.scheduler_refusal["scheduler_text"] == "sbatch: error: Batch job submission failed"
    assert out.scheduler_refusal["job_dir_restored"] is True

    assert _accounting(job_dir) == before and not list(job_dir.glob("*.operator-previous"))


@pytest.mark.parametrize("ran", [False, True])
def test_the_next_recovery_after_a_continuation_reads_a_known_budget(tmp_path, slurm_bin, ran):
    from nemo_gym.orchestration.submit import inspect_recovery, recover

    job_dir = _unknown_run(tmp_path, unreadable=(".gym_infra_retries", ".gym_accumulated_walltime"))
    assert recover(*_handle(job_dir), **BUDGET).dispatched == "resubmit"
    if ran:
        # The continuation started, recorded itself in the chain and ended after 30 s without a follow-up.
        with (job_dir / ".gym_job_chain").open("a") as chain:
            chain.write("4242\n")
        with (slurm_bin / "accounting").open("a") as accounting:
            accounting.write("4242|FAILED|30|0\n")

    later = inspect_recovery(*_handle(job_dir))

    assert later.decision.action.value == "resubmit", later.decision.reason
    assert UNKNOWN_BUDGET_REASON not in later.decision.reason and later.decision.operator_continuation is None
    record = load_submission_record(_handle(job_dir)[0])
    _, collected = SlurmExecutor().recovery_evidence(record, "gsm8k")
    # The chain is accounted from the continuation's epoch: its zero baseline plus what ran since.
    assert collected.accounted_walltime_s == (30.0 if ran else 0.0)
    assert (collected.infra_retries_state, collected.walltime_state) == ("ok", "ok")


def test_a_client_lost_after_the_acknowledgement_still_leaves_the_epoch(tmp_path, slurm_bin, monkeypatch):
    from nemo_gym.orchestration.executors.connection import LocalConnection
    from nemo_gym.orchestration.submit import inspect_recovery, recover

    job_dir = _unknown_run(tmp_path, unreadable=(".gym_infra_retries", ".gym_accumulated_walltime"))
    run = LocalConnection.run
    acknowledged = []

    def lost_after_the_acknowledgement(self, commands):
        # The queue acknowledged the job; the client or its SSH session fails before any later remote step.
        if acknowledged:
            raise RuntimeError("client lost after the queue acknowledged the job")
        output = run(self, commands)
        acknowledged.extend(command for command in commands if command == "_gym_submit_started=1")
        return output

    with monkeypatch.context() as patched:
        patched.setattr(LocalConnection, "run", lost_after_the_acknowledgement)
        with pytest.raises(RuntimeError, match="client lost"):
            recover(*_handle(job_dir), **BUDGET)
    assert acknowledged

    [entry] = [json.loads(line) for line in (job_dir / OPERATOR_CONTINUATION_JOURNAL_NAME).read_text().splitlines()]
    assert (entry["job_id"], entry["epoch"]) == ("4242", {"chain_length": 2, "walltime_s": 0.0})
    assert not list(job_dir.glob("*.operator-previous")) and not list(job_dir.glob("*.tmp"))
    record = load_submission_record(_handle(job_dir)[0])
    _, collected = SlurmExecutor().recovery_evidence(record, "gsm8k")
    # The fresh unknown-budget baseline is reconstructed from the epoch written under the lock.
    assert collected.operator_epoch is not None and collected.accounted_walltime_s == 0.0
    assert (collected.infra_retries_state, collected.walltime_state) == ("ok", "ok")
    later = inspect_recovery(*_handle(job_dir))
    assert UNKNOWN_BUDGET_REASON not in later.decision.reason


def test_an_explicit_extension_raises_a_readable_spent_count_under_an_operator_continuation(tmp_path, slurm_bin):
    """Spent retries readable and used up, walltime unknown: --extend-retries adds a retry instead of being lost."""
    from nemo_gym.orchestration.submit import recover

    job_dir = _unknown_run(tmp_path, unreadable=(".gym_accumulated_walltime",))
    (job_dir / ".gym_infra_retries").write_text("3\n")

    stopped = recover(*_handle(job_dir), **BUDGET)
    assert stopped.dispatched is None and "0 infra retries remain" in stopped.decision.reason, stopped.decision.reason

    out = recover(*_handle(job_dir), extend_retries=1, **BUDGET)

    assert out.dispatched == "resubmit", out.decision.reason
    record = out.decision.operator_continuation
    assert record["budget"]["infra_retries_remaining"] == 1
    assert record["budget"]["infra_retry_extension_added"] == 1
    assert record["writes"] == {
        ".gym_accumulated_walltime": {"previous": "unknown", "value": 0},
        RETRY_EXTENSION_NAME: {"previous": 0, "value": 1},
    }
    assert "(1 more recorded as the run's retry extension)" in out.decision.reason
    # The spent count is kept; the extension is the journaled raise.
    assert (job_dir / ".gym_infra_retries").read_text() == "3\n"
    assert (job_dir / RETRY_EXTENSION_NAME).read_text() == "1\n"
