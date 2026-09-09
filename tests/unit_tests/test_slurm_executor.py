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

from pathlib import Path

import pytest

from nemo_gym.orchestration.executors.slurm import _parse_sbatch_results, _sbatch_command


def test_sbatch_command_captures_the_status_of_sbatch_not_the_pipeline():
    command = _sbatch_command("gsm8k", Path("/jobs/run/gsm8k/job.sh"))
    # rc must be captured immediately after sbatch; taking $? after a pipe
    # would report tr's status and call every failure a success.
    assert "rc=$?" in command
    assert command.index("rc=$?") < command.index("tr ")
    assert "sbatch --parsable /jobs/run/gsm8k/job.sh" in command
    assert "__GYM_JOB:gsm8k:$rc:$out" in command


@pytest.mark.parametrize("name", ["has space", "has:colon", "has$dollar", "", "a;b"])
def test_sbatch_command_rejects_a_benchmark_name_that_would_break_the_marker(name):
    with pytest.raises(ValueError, match="benchmark name"):
        _sbatch_command(name, Path("/jobs/run/job.sh"))


@pytest.mark.parametrize("name", ["gsm8k", "gpqa-no-tools", "tau2.airline", "aime_24"])
def test_sbatch_command_accepts_ordinary_benchmark_names(name):
    assert f"__GYM_JOB:{name}:$rc:$out" in _sbatch_command(name, Path("/jobs/run/job.sh"))


def test_parse_sbatch_results_reads_a_successful_submission():
    assert _parse_sbatch_results("__GYM_JOB:gsm8k:0:12345 ") == {"gsm8k": ("12345", None)}


def test_parse_sbatch_results_drops_the_federation_cluster_suffix():
    assert _parse_sbatch_results("__GYM_JOB:gsm8k:0:12345;hsg ") == {"gsm8k": ("12345", None)}


def test_parse_sbatch_results_ignores_unrelated_output():
    output = "Loading modules\n__GYM_JOB:gsm8k:0:12345 \nsome trailing chatter"
    assert _parse_sbatch_results(output) == {"gsm8k": ("12345", None)}


def test_parse_sbatch_results_records_a_failure_with_its_message():
    output = "__GYM_JOB:gsm8k:1:sbatch: error: Invalid account 'nope' "
    assert _parse_sbatch_results(output) == {"gsm8k": (None, "sbatch: error: Invalid account 'nope'")}


def test_parse_sbatch_results_falls_back_to_the_exit_code_when_sbatch_said_nothing():
    assert _parse_sbatch_results("__GYM_JOB:gsm8k:1: ") == {"gsm8k": (None, "sbatch exited 1")}


def test_a_failure_mid_list_does_not_shift_the_benchmarks_after_it():
    # The regression test for the positional-zip bug: bench_b fails, and bench_c
    # must still get its own job id rather than inheriting the next one along.
    output = "\n".join(
        [
            "__GYM_JOB:bench_a:0:111 ",
            "__GYM_JOB:bench_b:1:sbatch: error: Invalid account ",
            "__GYM_JOB:bench_c:0:333 ",
        ]
    )

    results = _parse_sbatch_results(output)

    assert results["bench_a"] == ("111", None)
    assert results["bench_b"][0] is None
    assert results["bench_c"] == ("333", None)
