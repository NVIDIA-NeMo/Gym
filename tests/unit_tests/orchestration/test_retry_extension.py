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
"""`gym eval resubmit --recover --extend-retries N`: an explicit, journaled infra-retry extension, never a reset."""

import json

import pytest

from nemo_gym.orchestration.api import ResumeConfig
from nemo_gym.orchestration.executors.resume_script import RETRY_EXTENSION_MARKER, RETRY_EXTENSION_NAME
from nemo_gym.orchestration.executors.slurm import RETRY_EXTENSION_JOURNAL_NAME
from tests.unit_tests.orchestration.test_eval_resubmit import _handle, _resumable_run, _sbatch_env
from tests.unit_tests.orchestration.test_eval_resubmit import _local_index as _local_index
from tests.unit_tests.orchestration.test_eval_resubmit import slurm_bin as slurm_bin
from tests.unit_tests.orchestration.test_resume_script import _run_prologue


@pytest.mark.parametrize("extension,continues", [(None, False), ("1", True)])
def test_the_prologue_adds_the_recorded_extension_to_the_configured_cap(tmp_path, extension, continues):
    (tmp_path / ".gym_infra_retries").write_text("2\n")
    if extension is not None:
        (tmp_path / RETRY_EXTENSION_NAME).write_text(extension + "\n")
    result = _run_prologue(tmp_path, ResumeConfig(max_retries=2), None, keep_budget=True)
    assert ("REACHED_WORK" in result.stdout) is continues, result.stdout + result.stderr
    if continues:
        assert "Infra retry 3/3..." in result.stdout
        assert (tmp_path / ".gym_infra_retries").read_text().strip() == "3"
    else:
        assert "Infra retry limit (2) reached" in result.stdout


def _spent_run(tmp_path):
    job_dir, script, resume = _resumable_run(tmp_path)
    (job_dir / ".gym_infra_retries").write_text("3\n")  # max_retries 3: the configured budget is spent
    return job_dir, script


def test_recover_stops_on_a_spent_budget_and_an_explicit_extension_continues_it(tmp_path, slurm_bin):
    from nemo_gym.orchestration.submit import inspect_recovery, recover

    job_dir, script = _spent_run(tmp_path)
    assert RETRY_EXTENSION_MARKER in script
    spent = inspect_recovery(*_handle(job_dir))
    assert spent.decision.action.value == "stop"
    assert spent.decision.reason == "the stored infra-retry budget is used (3 of 3)"

    out = recover(*_handle(job_dir), extend_retries=1)
    assert out.dispatched == "resubmit", out.decision.reason
    assert out.decision.reason.endswith("(infra-retry cap 4: 3 configured + 0 recorded extension + 1 requested now)")
    assert _sbatch_env(slurm_bin)["GYM_RESUBMIT_KEEP_BUDGET"] == "1"
    assert (job_dir / RETRY_EXTENSION_NAME).read_text() == "1\n"
    assert (job_dir / ".gym_infra_retries").read_text() == "3\n"  # spent count kept, never reset
    [entry] = [json.loads(line) for line in (job_dir / RETRY_EXTENSION_JOURNAL_NAME).read_text().splitlines()]
    assert entry["by"] == 1

    # A later recovery counts the recorded extension without asking again.
    later = inspect_recovery(*_handle(job_dir))
    assert later.decision.action.value == "resubmit"
    assert later.decision.reason.endswith("(infra-retry cap 4: 3 configured + 1 recorded extension)")


def test_an_extension_is_refused_for_a_job_script_that_cannot_read_it(tmp_path, slurm_bin):
    from nemo_gym.orchestration.submit import recover

    job_dir, script = _spent_run(tmp_path)
    (job_dir / "job.sh").write_text(script.replace(RETRY_EXTENSION_MARKER + "\n", ""))
    with pytest.raises(RuntimeError, match="predates recorded retry extensions"):
        recover(*_handle(job_dir), extend_retries=1)
    assert not (job_dir / RETRY_EXTENSION_NAME).exists()
    assert not (slurm_bin / "env").exists()


def test_an_extension_whose_job_is_not_queued_is_not_recorded(tmp_path, slurm_bin):
    from nemo_gym.orchestration.submit import recover

    job_dir, _ = _spent_run(tmp_path)
    (slurm_bin / "sbatch").write_text("#!/bin/bash\necho 'sbatch: error: Batch job submission failed'; exit 1\n")
    out = recover(*_handle(job_dir), extend_retries=2)
    assert not out.ok and out.dispatched is None
    assert out.scheduler_refusal["scheduler_text"] == "sbatch: error: Batch job submission failed"
    assert out.scheduler_refusal["job_dir_restored"] is True
    assert not (job_dir / RETRY_EXTENSION_NAME).exists()
    assert not (job_dir / RETRY_EXTENSION_JOURNAL_NAME).exists()
