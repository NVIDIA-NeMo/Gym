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
"""The `gym eval resubmit` command flow that the recovery decisions of tests/unit_tests/eval/recovery/test_masked_infrastructure_recovery.py feed."""

from tests.unit_tests.eval.recovery.test_masked_infrastructure_recovery import _masked, _run
from tests.unit_tests.orchestration.test_eval_resubmit import slurm_bin as slurm_bin


def test_inspect_of_a_saved_run_with_one_masked_session_lost_result_resubmits(tmp_path, slurm_bin):
    from nemo_gym.orchestration.submit import inspect_recovery
    from tests.unit_tests.orchestration.test_eval_resubmit import _handle, _resumable_run

    job, _, _ = _resumable_run(tmp_path)
    for name in ("rollouts.jsonl", "rollouts_materialized_inputs.jsonl"):
        (job / "artifacts" / name).unlink()
    _run(job, [[_masked("session_lost")]])
    # The chain's walltime is read from the scheduler's accounting, as on the operator's login host.
    (job / ".gym_accumulated_walltime").unlink()
    (job / ".gym_job_chain").write_text("100\n101\n")
    (slurm_bin / "accounting").write_text("100|TIMEOUT|7200|0\n101|FAILED|60|0\n")
    outcome = inspect_recovery(*_handle(job))
    assert outcome.dispatched is None
    assert outcome.decision.action.value == "resubmit", outcome.decision.reason
    assert outcome.decision.counts == {"scored": 1, "retryable": 1}
