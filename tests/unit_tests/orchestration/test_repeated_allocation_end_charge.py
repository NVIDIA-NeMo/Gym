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
"""A rollout in flight when its allocation ended re-runs for free once; every later such end spends an attempt.

Real Store; each segment is one writer that reserves rows and ends with some of them still in flight.
"""

import json

import pytest

from nemo_gym.core.failure_kinds import (
    MODEL_FAULT_KINDS,
    ROLLOUT_ALLOCATION_ENDED,
    SERVER_FAULT_KINDS,
    is_registered,
)
from nemo_gym.orchestration.recovery import failure_facts, fault_class
from nemo_gym.rollouts.rollout_records import read_records
from nemo_gym.rollouts.rollout_recovery import RunManifest
from nemo_gym.rollouts.rollout_store import RolloutStore
from nemo_gym.utils.path_utils import failures_path_for
from tests.unit_tests.orchestration.test_allocation_cut_recovery import _decide


ROWS = [{"_ng_task_index": i, "_ng_rollout_index": 0} for i in range(3)]


@pytest.fixture
def run(tmp_path):
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    source = tmp_path / "source.jsonl"
    source.write_text("".join(json.dumps(row) + "\n" for row in ROWS))

    def prepare():
        inputs = json.loads(json.dumps(ROWS))
        return inputs, RunManifest.create(source, inputs, {}, {}).model_copy(update={"max_rollout_attempts": 2})

    return artifacts / "rollouts.jsonl", prepare


def _segment(run, *, cut=(), finish=(), fail=(), resume=True, resubmit=False):
    """One allocation: resume, reserve the pending rows named, save outcomes for some, end with ``cut`` in flight."""
    output, prepare = run
    store = RolloutStore.start_or_resume(output, prepare, resume=resume, resubmit=resubmit)
    with store:
        pending = {row["_ng_task_index"]: row for row in store.pending()}
        for index in (*cut, *finish, *fail):
            store.record_dispatch(pending[index])
        for index in finish:
            store.record_outcome(pending[index] | {"reward": 1.0})
        for index in fail:
            store.record_outcome(pending[index] | {"_ng_failure_class": "agent_run_error"})
    return store


def _charges(output):
    return [
        (row["_ng_task_index"], row["_ng_attempt_index"])
        for row in read_records(failures_path_for(output))
        if row.get("failure_kind") == ROLLOUT_ALLOCATION_ENDED
    ]


def _resume(run):
    output, prepare = run
    return RolloutStore.start_or_resume(output, prepare, resume=True)


def test_the_first_allocation_end_is_free(run):
    _segment(run, cut=(0,), resume=False)
    store = _resume(run)
    assert [row["_ng_task_index"] for row in store.pending()] == [0, 1, 2]
    assert store.attempt_count(ROWS[0]) == 0
    assert _charges(run[0]) == []


def test_the_second_end_of_the_same_identity_charges_one_attempt(run):
    _segment(run, cut=(0,), resume=False)
    _segment(run, cut=(0,))
    store = _resume(run)
    assert store.attempt_count(ROWS[0]) == 1
    # The first end stays free: only the later reservation is charged, as a failure of that attempt.
    assert _charges(run[0]) == [(0, 1)]
    (row,) = [row for row in store.pending() if row["_ng_task_index"] == 0]
    assert row["_ng_attempt_index"] == 2
    assert store.coverage()["failed_by_class"] == {ROLLOUT_ALLOCATION_ENDED: 1}
    # Charging is idempotent: a second resume finds the charge saved and adds none.
    _resume(run)
    assert _charges(run[0]) == [(0, 1)]


def test_a_read_only_reader_never_charges(run):
    _segment(run, cut=(0,), resume=False)
    _segment(run, cut=(0,))
    before = failures_path_for(run[0]).read_bytes()
    RolloutStore.read(run[0])
    assert failures_path_for(run[0]).read_bytes() == before


def test_the_cap_makes_the_identity_final_and_it_is_not_dispatched_again(run):
    _segment(run, cut=(0,), finish=(1, 2), resume=False)
    _segment(run, cut=(0,))
    _segment(run, cut=(0,))
    store = _resume(run)
    assert _charges(run[0]) == [(0, 1), (0, 2)]
    assert store.pending() == []
    coverage = store.coverage()
    assert coverage["failed_final"] == 1
    assert coverage["failure_budget_exhausted"] == 1
    assert coverage["failed_by_class"] == {ROLLOUT_ALLOCATION_ENDED: 1}
    assert coverage["successful"] == 2


def test_the_kind_is_registered_and_evaluation_side():
    assert ROLLOUT_ALLOCATION_ENDED == "rollout_allocation_ended"
    assert is_registered(ROLLOUT_ALLOCATION_ENDED)
    assert ROLLOUT_ALLOCATION_ENDED not in MODEL_FAULT_KINDS | SERVER_FAULT_KINDS


def test_a_finished_row_is_never_charged(run):
    _segment(run, cut=(0, 1), resume=False)
    _segment(run, cut=(1,), finish=(0,))
    _resume(run)
    # Row 0 ended in flight once and then finished; row 1 ended twice and is charged once.
    assert _charges(run[0]) == [(1, 1)]
    _segment(run, finish=(1,))
    store = _resume(run)
    assert _charges(run[0]) == [(1, 1)]
    assert store.coverage()["successful"] == 2


def test_failures_and_ends_share_the_cap(run):
    _segment(run, cut=(0,), resume=False)
    _segment(run, fail=(0,))
    _segment(run, cut=(0,))
    store = _resume(run)
    assert store.attempt_count(ROWS[0]) == 2
    assert _charges(run[0]) == [(0, 2)]
    assert [row["_ng_task_index"] for row in store.pending()] == [1, 2]


def test_a_resubmit_gives_the_charged_identity_a_fresh_budget(run):
    _segment(run, cut=(0,), finish=(1, 2), resume=False)
    _segment(run, cut=(0,))
    _segment(run, cut=(0,))
    assert _resume(run).pending() == []
    output, prepare = run
    store = RolloutStore.start_or_resume(output, prepare, resume=True, resubmit=True)
    (row,) = store.pending()
    assert (row["_ng_task_index"], store.attempt_count(ROWS[0])) == (0, 0)


def test_the_guard_reports_the_final_identity_by_its_cause(run):
    output = run[0]
    _segment(run, cut=(0,), finish=(1, 2), resume=False)
    _segment(run, cut=(0,))
    _segment(run, cut=(0,))
    job = output.parent.parent
    # Before the next writer charges the last end, the guard still sees a reservation it can resume.
    _, decision = _decide(job, allocation_cut=True)
    assert decision.action.value == "resubmit"
    # The next segment charges it and runs nothing; the chain then stops with the identity final, named by its kind.
    _segment(run)
    collected, decision = _decide(job, allocation_cut=True)
    assert decision.action.value == "stop"
    assert decision.counts == {"exhausted": 1, "scored": 2}
    (final,) = [item for item in decision.evidence if item.state.value == "exhausted"]
    assert (final.failure_kind, final.attempts, final.max_attempts) == (ROLLOUT_ALLOCATION_ENDED, 2, 2)
    # Reported by its cause, and an evaluation-side fault: not the model's.
    facts = failure_facts(decision, collected)
    assert facts["trial_failures"]["by_cause"]["allocation"]["count"] == 1
    assert fault_class(ROLLOUT_ALLOCATION_ENDED) is None
