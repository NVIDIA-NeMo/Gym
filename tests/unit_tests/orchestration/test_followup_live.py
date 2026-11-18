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
"""A queued follow-up or its sentinel is a job of the run before it appends itself to the chain."""

from pathlib import Path

import pytest

from tests.unit_tests.orchestration.test_eval_resubmit import _handle
from tests.unit_tests.orchestration.test_eval_resubmit import _local_index as _local_index
from tests.unit_tests.orchestration.test_eval_resubmit import slurm_bin as slurm_bin
from tests.unit_tests.orchestration.test_fresh_restart import REASON, _cut_before_inventory


# Like the real sacct, the fake answers only for the ids it is asked about.
_SACCT = """#!/bin/bash
while [[ $# -gt 0 ]]; do [[ $1 == -j ]] && ids=$2; shift; done
echo "$ids" >> "$(dirname "$0")/sacct_ids"
IFS=, read -ra want <<< "$ids"
while IFS='|' read -r id rest; do
    for w in "${want[@]}"; do [[ $id == "$w" ]] && echo "$id|$rest"; done
done < "$(dirname "$0")/accounting"
exit 0
"""


def _chain(tmp_path: Path, slurm_bin: Path, accounting: str, files: dict[str, str]) -> Path:
    """Allocations 100 and 101 ended before the task inventory was written; ``files`` are the run's claims."""
    job_dir = _cut_before_inventory(tmp_path, slurm_bin)
    (slurm_bin / "sacct").write_text(_SACCT)
    (slurm_bin / "accounting").write_text("100|TIMEOUT|14400|0\n101|FAILED|30|0\n" + accounting)
    for name, value in files.items():
        (job_dir / name).write_text(value + "\n")
    return job_dir


@pytest.mark.parametrize(
    "files,live",
    [
        ({".gym_followup_queued.101": "102"}, "102 (PENDING)"),
        ({".gym_followup_queued.100": "101", ".gym_followup_sentinel.101": "103"}, "103 (PENDING)"),
    ],
    ids=["queued-follow-up", "pending-sentinel"],
)
def test_a_pending_follow_up_or_sentinel_is_live_and_nothing_is_decided(tmp_path, slurm_bin, files, live):
    from nemo_gym.orchestration.submit import inspect_recovery, recover

    job_dir = _chain(tmp_path, slurm_bin, "102|PENDING|0|0\n103|PENDING|0|0\n", files)
    for out in (inspect_recovery(*_handle(job_dir)), recover(*_handle(job_dir))):
        assert (out.decision.action.value, out.decision.scheduler_live) == ("noop", True)
        assert out.decision.reason == f"continuation {live} live; nothing to decide"
        assert (out.dispatched, out.gate_task["status"]) == (None, "RUNNING")
    # The claimed ids were asked about, not only the chain's.
    assert live.split()[0] in (slurm_bin / "sacct_ids").read_text().split("\n")[0].split(",")
    assert not (slurm_bin / "env").exists()


def test_ended_follow_ups_and_sentinels_leave_the_decision_unchanged(tmp_path, slurm_bin):
    from nemo_gym.orchestration.submit import inspect_recovery

    files = {".gym_followup_queued.100": "101", ".gym_followup_sentinel.100": "103"}
    job_dir = _chain(tmp_path, slurm_bin, "103|COMPLETED|5|0\n", files)
    out = inspect_recovery(*_handle(job_dir))
    assert (out.decision.action.value, out.decision.scheduler_live) == ("fresh-restart", False)
    assert out.decision.reason.startswith(REASON)
