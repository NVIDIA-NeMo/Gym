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
"""A follow-up that the chain's last allocation queued and that ended before it recorded itself in the chain."""

import pytest

from nemo_gym.orchestration.executors.resume_script import RETRY_FILE_NAME
from nemo_gym.orchestration.recovery import stopped_followup
from tests.unit_tests.orchestration.test_allocation_cut_recovery import _cut_run
from tests.unit_tests.orchestration.test_eval_resubmit import _handle, _resumable_run
from tests.unit_tests.orchestration.test_eval_resubmit import _local_index as _local_index
from tests.unit_tests.orchestration.test_eval_resubmit import slurm_bin as slurm_bin


_KILLED = (
    "the chain stopped: its follow-up 101 ended FAILED with signal 53 after 14 s (cause not recorded), before it "
    "recorded itself in the chain, so no job continued the run"
)


def _inspect_stopped_chain(tmp_path, slurm_bin, followup_accounting, *, retries_used=2):
    """Allocation 100 hit its time limit with work in flight; the follow-up 101 it claimed is not in the chain."""
    from nemo_gym.orchestration.submit import inspect_recovery

    job, _, _ = _resumable_run(tmp_path)
    for name in ("rollouts.jsonl", "rollouts_materialized_inputs.jsonl"):
        (job / "artifacts" / name).unlink()
    _cut_run(job, scored=2, in_flight=2, not_started=2, chain=False)
    # The first allocation of the chain: no predecessor walltime is stored yet.
    (job / ".gym_job_chain").write_text("100\n")
    (job / ".gym_accumulated_walltime").unlink()
    (job / RETRY_FILE_NAME).write_text(f"{retries_used}\n")
    (job / ".gym_followup_queued.100").write_text("101\n")
    (slurm_bin / "accounting").write_text("100|TIMEOUT|7200|0\n")
    (slurm_bin / "followup_accounting").write_text(followup_accounting)
    (slurm_bin / "sacct").write_text(
        '#!/bin/bash\nif [[ "$*" == *ExitCode* ]]; then cat "$(dirname "$0")/followup_accounting"; '
        'else cat "$(dirname "$0")/accounting"; fi\n'
    )
    return inspect_recovery(*_handle(job)).decision


def test_a_follow_up_killed_before_it_recorded_itself_is_named_and_the_run_is_resubmitted(tmp_path, slurm_bin):
    decision = _inspect_stopped_chain(tmp_path, slurm_bin, "101|FAILED|14|0:53\n")
    assert decision.action.value == "resubmit", decision.reason
    assert decision.counts == {"scored": 2, "missing": 4}
    assert decision.reason.startswith("4 identity(ies) need the policy again")
    assert decision.reason.endswith(f"; {_KILLED}")


def test_the_resubmit_of_a_killed_follow_up_counts_against_the_infra_retry_cap(tmp_path, slurm_bin):
    decision = _inspect_stopped_chain(tmp_path, slurm_bin, "101|FAILED|14|0:53\n", retries_used=3)
    assert decision.action.value == "stop"
    assert decision.reason == f"the stored infra-retry budget is used (3 of 3); {_KILLED}"


def test_a_follow_up_its_dependency_cancelled_before_it_started_is_not_named(tmp_path, slurm_bin):
    decision = _inspect_stopped_chain(tmp_path, slurm_bin, "101|CANCELLED|0|0:0\n")
    assert decision.action.value == "resubmit", decision.reason
    assert "follow-up" not in decision.reason


@pytest.mark.parametrize(
    "accounting,expected",
    [
        (
            "101|NODE_FAIL|30|0:0\n",
            "the chain stopped on infrastructure: its follow-up 101 ended NODE_FAIL after 30 s",
        ),
        ("101|FAILED|14|1:0\n", "the chain stopped: its follow-up 101 ended FAILED with exit code 1 after 14 s"),
        # A signal's sender and reason are not in the record, whatever the state.
        (
            "101|CANCELLED by 7|9|0:15\n",
            "the chain stopped: its follow-up 101 ended CANCELLED with signal 15 after 9 s (cause not recorded),",
        ),
        (
            "101|NODE_FAIL|30|0:9\n",
            "the chain stopped on infrastructure: its follow-up 101 ended NODE_FAIL with signal 9 after 30 s,",
        ),
        ("101|PENDING|0|0:0\n", None),
        ("101|RUNNING|5|0:0\n", None),
        ("102|FAILED|14|0:53\n", None),
        ("101|FAILED|14|0:53\n101|FAILED|3|0:53\n", None),
        ("", None),
    ],
    ids=[
        "node-fail",
        "exit-code",
        "cancelled-by-signal",
        "node-fail-by-signal",
        "pending",
        "running",
        "other-job",
        "two-rows",
        "no-record",
    ],
)
def test_the_follow_up_record_names_only_a_started_and_ended_job(accounting, expected):
    note = stopped_followup("101", accounting)
    if expected is None:
        assert note is None
    else:
        assert note is not None and note.startswith(expected), note
