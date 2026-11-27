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
"""The `gym eval resubmit` command flow that the recovery decisions of tests/unit_tests/eval/recovery/test_shared_fault_classes.py feed."""

import pytest

from tests.helpers.recovery_evidence import complete_job
from tests.unit_tests.orchestration.test_eval_resubmit_decision import fake_executor as fake_executor
from tests.unit_tests.orchestration.test_eval_resubmit_decision import run_cli


@pytest.mark.parametrize("value,expected", [("false", False), ("true", True)])
def test_the_recover_command_sets_it_on_the_gate(tmp_path, fake_executor, capsys, value, expected):
    fake_executor.job_dir = complete_job(tmp_path)
    _, out = run_cli(
        ["eval", "resubmit", "gym-job-k", "bench", "--recover", "--policy-timeout-counts-as-model", value], capsys
    )
    assert out["gate_task"]["criteria"]["evaluation"]["policy_timeout_counts_as_model"] is expected
