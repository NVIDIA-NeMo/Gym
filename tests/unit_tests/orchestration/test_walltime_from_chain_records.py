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
"""A capped chain whose guard stopped before it stored the used walltime: the chain's own records give it."""

import pytest

from nemo_gym.orchestration.recovery import accounted_chain_walltime
from tests.unit_tests.orchestration.test_allocation_cut_recovery import _cut_run
from tests.unit_tests.orchestration.test_eval_resubmit import _handle, _resumable_run
from tests.unit_tests.orchestration.test_eval_resubmit import _local_index as _local_index
from tests.unit_tests.orchestration.test_eval_resubmit import slurm_bin as slurm_bin


def _inspect_unstored_chain(tmp_path, slurm_bin, accounting, *, in_flight=2):
    """Segment 100 ended with work in flight; guard 101 stopped before it stored 100's walltime (16 h cap)."""
    from nemo_gym.orchestration.submit import inspect_recovery

    job, _, _ = _resumable_run(tmp_path)
    for name in ("rollouts.jsonl", "rollouts_materialized_inputs.jsonl"):
        (job / "artifacts" / name).unlink()
    _cut_run(job, scored=2, in_flight=in_flight, not_started=2, chain=False)
    (job / ".gym_job_chain").write_text("100\n101\n")
    (job / ".gym_accumulated_walltime").unlink()
    if accounting is None:
        (slurm_bin / "sacct").write_text("#!/bin/bash\necho 'sacct: error: slurmdbd unavailable' >&2\nexit 1\n")
    else:
        (slurm_bin / "accounting").write_text(accounting)
    return inspect_recovery(*_handle(job)).decision


@pytest.mark.parametrize(
    "accounting,total",
    [("100|CANCELLED|7200|0\n101|FAILED|60|0\n", 7260), ("100|NODE_FAIL|7200|0\n101|NODE_FAIL|30|0\n", 7230)],
    ids=["cancelled-segment", "node-fail"],
)
def test_the_chain_records_give_the_used_walltime_when_no_allocation_stored_it(tmp_path, slurm_bin, accounting, total):
    decision = _inspect_unstored_chain(tmp_path, slurm_bin, accounting)
    assert decision.action.value == "resubmit", decision.reason
    assert decision.counts == {"scored": 2, "missing": 4}
    assert decision.reason.endswith(
        f"; the used walltime {total} s is the sum of the chain's 2 scheduler records, as no allocation stored it"
    )


def test_unavailable_chain_records_still_stop(tmp_path, slurm_bin):
    # Nothing in flight, so the walltime check is the one that decides.
    decision = _inspect_unstored_chain(tmp_path, slurm_bin, None, in_flight=0)
    reason, _, hint = decision.reason.partition("\n")
    assert (decision.action.value, reason) == (
        "stop",
        "the run caps its walltime but the used walltime is unavailable",
    )
    assert "--recover --max-walltime HH:MM:SS --walltime-reason TEXT" in hint


def test_a_derived_walltime_at_the_cap_stops_and_says_where_it_came_from(tmp_path, slurm_bin):
    decision = _inspect_unstored_chain(tmp_path, slurm_bin, "100|CANCELLED|57600|0\n101|FAILED|60|0\n")
    assert decision.action.value == "stop"
    assert decision.reason == (
        "the stored walltime budget is used (57660 of 57600 s); the used walltime 57660 s is the sum of the chain's 2 "
        "scheduler records, as no allocation stored it"
    )


def test_no_stored_walltime_sums_the_records_and_a_stored_one_must_still_match():
    accounting = "101|CANCELLED|10\n102|FAILED|20\n"
    assert accounted_chain_walltime(["101", "102"], accounting, stored_walltime_s=None) == 30
    assert accounted_chain_walltime(["101", "102"], accounting, stored_walltime_s=0) is None
    assert accounted_chain_walltime(["101", "102"], "101|CANCELLED|10\n", stored_walltime_s=None) is None


def test_a_segment_cancelled_by_the_operator_is_accounted_from_its_record():
    # The follow-up's prologue stops at the cancel before storing it, so only the first segment is stored.
    chain = ["101", "102", "103"]
    accounting = "101|TIMEOUT|1146\n102|CANCELLED by 1000|759\n103|FAILED|14\n"
    assert accounted_chain_walltime(chain, accounting, stored_walltime_s=1146) == 1919


def test_the_cancel_allowance_needs_the_cancelled_segment_second_to_last():
    accounting = "101|TIMEOUT|1146\n102|FAILED|759\n103|FAILED|14\n"
    assert accounted_chain_walltime(["101", "102", "103"], accounting, stored_walltime_s=1146) is None
