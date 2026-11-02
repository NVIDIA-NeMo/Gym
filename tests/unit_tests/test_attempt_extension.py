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
"""`gym eval resubmit --recover --extend-attempts N`: an explicit, journaled extension of every rollout's attempt cap,
never a reset, and never a second chance for a rollout that ended with a model failure."""

import json

import pytest

from nemo_gym.orchestration.executors.slurm import RETRY_EXTENSION_JOURNAL_NAME
from nemo_gym.orchestration.recovery import decide_from_collected, select_managed_evidence
from nemo_gym.rollout_recovery import RunManifest
from nemo_gym.rollout_store import RolloutStore
from tests.unit_tests.test_eval_resubmit import _handle, _provider_failures, _resumable_run
from tests.unit_tests.test_eval_resubmit import _local_index as _local_index
from tests.unit_tests.test_eval_resubmit import slurm_bin as slurm_bin
from tests.unit_tests.test_managed_recovery_selection import collect, prepare_managed


EXTENSION = "rollouts_attempt_extension"


def _decide(job, **options):
    selected = select_managed_evidence(collect(job), job)
    assert selected.managed_reader_error is None
    return decide_from_collected(selected, max_infra_retries=3, max_walltime_s=None, live_job=False, **options)


def _reopen(job, n=2):
    """The writer a resumed segment opens on ``prepare_managed``'s run."""
    rows = [{"_ng_task_index": i, "_ng_rollout_index": 0} for i in range(n)]

    def prepare():
        inputs = json.loads(json.dumps(rows))
        manifest = RunManifest.create(job / "source.jsonl", inputs, {}, {})
        return inputs, manifest.model_copy(update={"max_rollout_attempts": 3})

    return RolloutStore.start_or_resume(job / "artifacts" / "rollouts.jsonl", prepare, resume=True), rows


def test_an_exhausted_run_stops_and_names_the_attempt_extension(tmp_path):
    prepare_managed(tmp_path, "exhausted")

    result = _decide(tmp_path)

    assert (result.action.value, result.counts) == ("stop", {"exhausted": 1})
    assert "gym eval resubmit --recover --extend-attempts N" in result.reason
    assert "--extend-retries raises infra retries only" in result.reason


def test_a_requested_extension_resumes_each_exhausted_rollout(tmp_path):
    prepare_managed(tmp_path, "exhausted")

    result = _decide(tmp_path, extend_attempts=1)

    assert (result.action.value, result.counts) == ("resubmit", {"retryable": 1})
    assert result.evidence[0].max_attempts == 4
    assert result.reason.endswith("(attempt cap 4: 3 configured + 0 recorded extension + 1 requested now)")


@pytest.mark.parametrize("case,state", [("final_zero", "model_final"), ("healthy", "scored")])
def test_an_extension_never_reopens_a_model_final_or_scored_rollout(tmp_path, case, state):
    prepare_managed(tmp_path, case)

    result = _decide(tmp_path, extend_attempts=2)

    assert (result.action.value, result.counts) == ("noop", {state: 1})


def test_the_store_runs_exactly_the_extended_attempts(tmp_path):
    prepare_managed(tmp_path, "exhausted", n=2)
    output = tmp_path / "artifacts" / "rollouts.jsonl"
    assert RolloutStore.read(output, import_legacy=False).pending() == []

    (output.parent / EXTENSION).write_text("1\n")
    store = RolloutStore.read(output, import_legacy=False)
    [row] = store.pending()
    assert (row["_ng_task_index"], row["_ng_attempt_index"]) == (0, 3)
    coverage = store.coverage()
    assert (coverage["max_rollout_attempts"], coverage["attempt_extension"]) == (4, 1)

    # The extra attempt fails on infrastructure again: the rollout has spent its extended cap.
    writer, rows = _reopen(tmp_path)
    with writer as store:
        attempt = dict(rows[0])
        store.allocate_attempt(attempt)
        store.record_outcome(attempt | json.loads(_provider_failures(0, 4, 4).splitlines()[-1]))
    assert RolloutStore.read(output, import_legacy=False).pending() == []


def test_the_two_extensions_are_independent(tmp_path):
    prepare_managed(tmp_path, "exhausted")

    # More infra retries do not reopen a rollout that spent its attempts.
    assert _decide(tmp_path, extend_retries=2).action.value == "stop"
    # More attempts do not raise the run's infra-retry cap.
    result = _decide(tmp_path, extend_attempts=1)
    assert result.action.value == "resubmit" and "infra-retry cap" not in result.reason


def test_an_extension_is_refused_when_the_runs_store_reader_predates_it(tmp_path):
    prepare_managed(tmp_path, "exhausted")
    selected = select_managed_evidence(collect(tmp_path), tmp_path)
    # What a Store reader at an earlier pinned Gym returns: no applied extension, no replay facts.
    old = selected.model_copy(
        update={
            "attempt_extension_applied": None,
            "managed_selection": [fact.model_copy(update={"replayable": None}) for fact in selected.managed_selection],
        }
    )

    result = decide_from_collected(old, max_infra_retries=3, max_walltime_s=None, live_job=False, extend_attempts=1)

    assert result.action.value == "stop"
    assert result.reason.startswith("--extend-attempts 1 is refused: the run's pinned Gym predates attempt extensions")


def test_recover_records_the_extension_journals_it_and_a_later_recovery_counts_it(tmp_path, slurm_bin):
    from nemo_gym.orchestration.submit import inspect_recovery, recover

    job, _, _ = _resumable_run(tmp_path)
    prepare_managed(job, "exhausted", n=2)
    assert inspect_recovery(*_handle(job)).decision.action.value == "stop"

    out = recover(*_handle(job), extend_attempts=1)

    assert out.dispatched == "resubmit", out.decision.reason
    assert "(attempt cap 4: 3 configured + 0 recorded extension + 1 requested now)" in out.decision.reason
    assert (job / "artifacts" / EXTENSION).read_text() == "1\n"
    [entry] = [json.loads(line) for line in (job / RETRY_EXTENSION_JOURNAL_NAME).read_text().splitlines()]
    assert (entry["kind"], entry["by"]) == ("attempts", 1) and entry["at"]
    # The infra-retry count is untouched: the two budgets are separate.
    assert (job / ".gym_infra_retries").read_text() == "2\n"

    assert out.gate_task["retry_budget"] == {
        "infra": {"configured": 3, "extension": 0},
        "attempts": {"configured": 3, "extension": 0},
    }

    # A later recovery counts the recorded extension without asking again, and the gate reports it.
    later = inspect_recovery(*_handle(job))
    assert later.decision.action.value == "resubmit", later.decision.reason
    assert later.gate_task["retry_budget"]["attempts"] == {"configured": 3, "extension": 1}


@pytest.mark.parametrize("extension,continues", [(None, False), ("1", True)])
def test_the_in_job_admission_reads_the_recorded_extension(tmp_path, monkeypatch, capsys, extension, continues):
    from nemo_gym.orchestration.recovery import continuation_admission

    prepare_managed(tmp_path, "exhausted")
    # Segment 200 continues segment 100, which timed out after 60 s.
    monkeypatch.setenv("SLURM_JOB_ID", "200")
    (tmp_path / ".gym_job_chain").write_text("100\n200\n")
    (tmp_path / ".gym_accumulated_walltime").write_text("60\n")
    if extension is not None:
        (tmp_path / "artifacts" / EXTENSION).write_text(extension + "\n")

    admission = continuation_admission(str(tmp_path), 3, None, "0:15", "100", "TIMEOUT")

    assert (admission in ("segment", "retry")) is continues, capsys.readouterr().out
    if not continues:
        assert "used every attempt" in capsys.readouterr().out


def test_an_extension_needs_a_whole_number_in_range():
    from nemo_gym.orchestration.executors.slurm import SlurmExecutor

    with pytest.raises(RuntimeError, match="needs --recover"):
        SlurmExecutor._attempt_extension_commands(1, keep_budget=False)
    for value in (0, 101, 1.5):
        with pytest.raises(ValueError, match="from 1 to 100"):
            SlurmExecutor._attempt_extension_commands(value, keep_budget=True)
