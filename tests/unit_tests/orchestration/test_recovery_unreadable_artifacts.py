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
"""The `gym eval resubmit` command flow that the recovery decisions of tests/unit_tests/eval/recovery/test_recovery_unreadable_artifacts.py feed."""

import os

import pytest

from nemo_gym.eval.recovery.decision import RecoveryAction
from nemo_gym.orchestration import submit
from tests.unit_tests.eval.recovery.test_recovery_unreadable_artifacts import _run
from tests.unit_tests.orchestration.test_eval_resubmit_decision import FakeExecutor
from tests.unit_tests.orchestration.test_eval_resubmit_decision import fake_executor as fake_executor


pytestmark = pytest.mark.skipif(
    not hasattr(os, "geteuid") or os.geteuid() == 0, reason="root reads a file whatever its mode"
)


@pytest.mark.parametrize("name", ["rollouts_aggregate_metrics.json", "quality_summary.json"])
def test_the_verdict_of_an_unreadable_metric_or_summary_is_to_make_it_readable_not_a_fresh_submit(
    tmp_path, fake_executor, name
):
    collected = _run(tmp_path / "job", unreadable=name)  # a complete run; the file is readable again after
    assert collected.unreadable_files == [f"artifacts/{name}"]
    FakeExecutor.job_dir = tmp_path / "job"
    target = tmp_path / "job" / "artifacts" / name
    target.chmod(0)
    try:
        out = submit.inspect_recovery("gym-job-unreadable", "bench")
    finally:
        target.chmod(0o600)

    assert out.decision.action == RecoveryAction.NOOP and out.gate_task["status"] == "FAILED"
    assert out.operator_verdict == {
        "state": "unknown",
        "reason": f"unreadable artifacts, read as unknown: ['artifacts/{name}']",
        "action": "make those files readable again and inspect again",
    }
