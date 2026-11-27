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
"""The resume script's guard after a cut allocation: what it continues, resumes or ends, from the saved rows.

Real Store, real stdlib collector and the real auto-resume prologue; only the scheduler is simulated.
"""

import subprocess
from pathlib import Path

import pytest

from nemo_gym.cli.utils import exit_cleanly_on_config_error
from nemo_gym.config.config_types import ConfigError
from nemo_gym.orchestration.api import ResumeConfig, SubmitConfig
from nemo_gym.orchestration.executors.script_templates import render_driver_entrypoint
from nemo_gym.orchestration.executors.slurm.script import build_sbatch_script
from nemo_gym.rollouts.rollout_recovery import validate_resume
from nemo_gym.server_runtime.supervisor.server_supervisor import ServerSpinupTimeoutError
from tests.helpers.recovery_cut_runs import _cut_run, _digests
from tests.unit_tests.orchestration.test_eval_resubmit import _local_index as _local_index
from tests.unit_tests.orchestration.test_eval_resubmit import slurm_bin as slurm_bin
from tests.unit_tests.orchestration.test_resume_script import _run_prologue


@pytest.mark.parametrize("state,debit", [("TIMEOUT", False), ("PREEMPTED", True)])
def test_the_guard_continues_a_cut_allocation_from_its_saved_rows(tmp_path, state, debit):
    _cut_run(tmp_path)
    before = _digests(tmp_path)
    result = _run_prologue(tmp_path, ResumeConfig(max_retries=2), state, exit_code="0:15", prev_job_id="100")
    assert "REACHED_WORK" in result.stdout, result.stdout + result.stderr
    assert "Saved recovery evidence: 4 identity(ies) need the policy again" in result.stdout
    assert (tmp_path / ".gym_infra_retries").exists() is debit
    assert not (tmp_path / ".gym_chain_incomplete").exists()
    assert _digests(tmp_path) == before


def test_the_guard_stop_names_the_three_populations_when_the_chain_does_not_name_the_store_writer(tmp_path):
    _cut_run(tmp_path, allocation="99")
    (tmp_path / ".gym_job_chain").write_text("98\n100\n")
    result = _run_prologue(tmp_path, ResumeConfig(max_retries=2), "TIMEOUT", exit_code="0:15", prev_job_id="100")
    assert "REACHED_WORK" not in result.stdout
    marker = (tmp_path / ".gym_chain_incomplete").read_text()
    assert marker.startswith(
        "Saved recovery evidence has no eligible retries; 2 reservation(s) have no saved outcome and the allocation "
        "that made them is not known to have ended, so whether they ran is unknown"
    )
    assert "nobody owns them" not in marker
    assert "retryable 2 (resumed), unknown 2 (2 reservation_without_outcome)" in marker
    # The guard's own provisional line is not quoted into its stop: its sentinel would read it as this allocation
    # dying before its guard and run the same stop again.
    assert "Continuation 200 started" not in marker and "the chain recorded" not in marker
    assert marker.rstrip().endswith("(previous job 100 ended TIMEOUT)")


def test_the_guard_resumes_reservations_of_an_earlier_allocation_after_a_segment_that_saved_nothing(tmp_path):
    # Allocation 99 made the reservations and timed out; 100 timed out before it opened the Store.
    _cut_run(tmp_path, allocation="99")
    (tmp_path / ".gym_job_chain").write_text("99\n100\n")
    before = _digests(tmp_path)
    result = _run_prologue(tmp_path, ResumeConfig(max_retries=2), "TIMEOUT", exit_code="0:15", prev_job_id="100")
    assert "REACHED_WORK" in result.stdout, result.stdout + result.stderr
    assert "Saved recovery evidence: 4 identity(ies) need the policy again" in result.stdout
    assert _digests(tmp_path) == before


def test_the_guard_resumes_reservations_of_a_driver_that_failed(tmp_path):
    _cut_run(tmp_path)
    result = _run_prologue(tmp_path, ResumeConfig(max_retries=2), "FAILED", exit_code="1:0", prev_job_id="100")
    assert "REACHED_WORK" in result.stdout, result.stdout + result.stderr
    assert "Saved recovery evidence: 4 identity(ies) need the policy again" in result.stdout
    assert (tmp_path / ".gym_infra_retries").read_text().strip() == "1"


def test_the_guard_never_continues_after_an_operator_cancel(tmp_path):
    _cut_run(tmp_path)
    result = _run_prologue(tmp_path, ResumeConfig(max_retries=2), "CANCELLED", exit_code="0:15", prev_job_id="100")
    assert "REACHED_WORK" not in result.stdout
    assert (tmp_path / ".gym_chain_incomplete").read_text() == (
        "Previous job 100 was cancelled. Chain stopped with rollouts missing; continue with gym eval resubmit "
        "--recover, which carries the retry and walltime budgets over.\n"
    )


def test_the_guard_ends_the_chain_after_a_timeout_before_the_task_inventory(tmp_path):
    job = tmp_path / "gym-job-x" / "gsm8k"
    (job / "artifacts").mkdir(parents=True)
    result = _run_prologue(job, ResumeConfig(max_retries=2), "TIMEOUT", exit_code="0:15", elapsed="14400")
    stop = (
        "Previous job 100 ended TIMEOUT: the segment ended before the task inventory was written; nothing can be "
        "resumed; fresh restart: gym eval resubmit gym-job-x gsm8k --recover --max-walltime HH:MM:SS "
        "--walltime-reason TEXT"
    )
    assert (result.returncode, "REACHED_WORK" in result.stdout) == (1, False)
    assert stop in result.stdout
    assert (job / ".gym_chain_incomplete").read_text() == stop + "\n"
    # No successor is queued, and the segment's elapsed time is kept with the spent history.
    assert not (job / "sbatch_calls").exists()
    assert (job / ".gym_accumulated_walltime").read_text() == "14400\n"


def test_a_preemption_before_the_task_inventory_still_spends_an_infra_retry(tmp_path):
    job = tmp_path / "gym-job-x" / "gsm8k"
    (job / "artifacts").mkdir(parents=True)
    result = _run_prologue(job, ResumeConfig(max_retries=2), "PREEMPTED", exit_code="0:15")
    assert "REACHED_WORK" in result.stdout, result.stdout + result.stderr
    assert (job / ".gym_infra_retries").read_text() == "1\n"
    assert not (job / ".gym_chain_incomplete").exists()


# A saved-state reader that would continue the chain: only the driver's failure kind can stop it before.
_ADMISSION_THAT_CONTINUES = "touch admission_ran; echo __GYM_CONTINUATION:retry"


_INSTALL_LOG = (
    "Cloning into '/tmp/gym-install-x/gym'...\n"
    "fatal: unable to access 'https://example.invalid/gym.git/': Could not resolve host: example.invalid\n"
    "error: No pyproject.toml found in the current directory\n"
)


_DRIVER_FAILURE_STOPS = {
    "provider_credentials_rejected": (
        "Previous job 100 ended FAILED: provider_credentials_rejected: the provider refused the "
        "credentials; the chain does not retry it; fix the credentials, then continue with gym eval resubmit "
        "--recover"
    ),
    "provider_quota_exhausted": (
        "Previous job 100 ended FAILED: provider_quota_exhausted: the grading provider's quota is exhausted; the "
        "chain does not retry it; after the reset continue with gym eval resubmit --recover (re-judges the saved "
        "answers, no model calls)"
    ),
    "environment_protocol_violation": (
        "Previous job 100 ended FAILED: environment_protocol_violation: an environment server answered with a reply "
        "the collector cannot use; the chain does not retry it; fix the server or the benchmark pairing, then gym "
        "eval submit again"
    ),
    "server_spinup_timeout": (
        "Previous job 100 ended FAILED: server_spinup_timeout: the Gym servers did not become ready within "
        "server_spinup_timeout_seconds; the chain does not retry it; raise the limit with gym eval resubmit --recover "
        "--server-spinup-timeout-seconds N --spinup-timeout-reason TEXT"
    ),
    "driver_install_failed": (
        "Previous job 100 ended FAILED: driver_install_failed: fatal: unable to access "
        "'https://example.invalid/gym.git/': Could not resolve host: example.invalid; the chain does not retry it; "
        "see logs/driver-100.log"
    ),
    "environment_prepare_failed": (
        "Previous job 100 ended FAILED: environment_prepare_failed: EVAL STOPPED: data preparation failed; the chain "
        "does not retry it; fix the benchmark's preparation, then continue with gym eval resubmit --recover"
    ),
    "collection_resume_refused": (
        "Previous job 100 ended FAILED: collection_resume_refused: the saved rollouts do not verify against this "
        "run's inputs and configuration; the chain does not retry it; start a fresh run or set "
        "allow_unsafe_resume=true"
    ),
}


@pytest.mark.parametrize("kind", sorted(_DRIVER_FAILURE_STOPS))
def test_a_driver_failure_the_next_segment_would_repeat_ends_the_chain_at_once(tmp_path, kind):
    job = tmp_path / "gym-job-x" / "gsm8k"
    (job / "logs").mkdir(parents=True)
    (job / "logs" / "driver-100.log").write_text(_INSTALL_LOG)
    (job / ".gym_driver_failure.100").write_text(kind + "\n")
    result = _run_prologue(
        job, ResumeConfig(max_retries=2), "FAILED", elapsed="300", admission_command=_ADMISSION_THAT_CONTINUES
    )
    stop = _DRIVER_FAILURE_STOPS[kind]
    assert (result.returncode, "REACHED_WORK" in result.stdout) == (1, False), result.stdout + result.stderr
    assert stop in result.stdout
    assert (job / ".gym_chain_incomplete").read_text() == stop + "\n"
    # Saved state is not read, no retry is spent and no successor is queued; the elapsed time is kept.
    assert not (job / "admission_ran").exists()
    assert not (job / ".gym_infra_retries").exists()
    assert not (job / "sbatch_calls").exists()
    assert (job / ".gym_accumulated_walltime").read_text() == "300\n"


def test_the_guard_reads_only_the_kind_its_previous_allocation_left(tmp_path):
    """A kind an earlier allocation (99) left does not stop the chain after allocation 100."""
    job = tmp_path / "gym-job-x" / "gsm8k"
    job.mkdir(parents=True)
    (job / ".gym_driver_failure.99").write_text("collection_resume_refused\n")
    result = _run_prologue(job, ResumeConfig(max_retries=2), "FAILED", admission_command=_ADMISSION_THAT_CONTINUES)
    assert "REACHED_WORK" in result.stdout, result.stdout + result.stderr
    assert (job / "admission_ran").exists()


def test_a_failed_install_leaves_its_kind_and_the_guard_names_its_error_line(tmp_path):
    job = tmp_path / "gym-job-x" / "gsm8k"
    (job / "logs").mkdir(parents=True)
    fake_bin = tmp_path / "fake-bin"
    fake_bin.mkdir()
    (fake_bin / "curl").write_text(
        "#!/bin/bash\necho 'curl: (6) Could not resolve host: example.invalid' >&2\nexit 6\n"
    )
    (fake_bin / "curl").chmod(0o755)
    entrypoint = render_driver_entrypoint("https://example.invalid/gym.git", "main", None, record_install_failure=True)
    (tmp_path / "driver.sh").write_text(f"GYM_CMD=(true)\n{entrypoint}\n")
    driver = subprocess.run(
        ["bash", str(tmp_path / "driver.sh")],
        env={
            "PATH": f"{fake_bin}:/usr/bin:/bin",
            "HOME": str(tmp_path),
            "GYM_DRIVER_FAILURE_FILE": str(job / ".gym_driver_failure.100"),
        },
        capture_output=True,
        text=True,
    )
    assert driver.returncode == 6, driver.stdout + driver.stderr
    assert (job / ".gym_driver_failure.100").read_text() == "driver_install_failed\n"
    # srun --output keeps both streams in the driver log. No line says error or fatal, so the last one is named.
    (job / "logs" / "driver-100.log").write_text(driver.stdout + driver.stderr)
    result = _run_prologue(
        job, ResumeConfig(max_retries=2), "FAILED", exit_code="6:0", admission_command=_ADMISSION_THAT_CONTINUES
    )
    stop = (
        "Previous job 100 ended FAILED: driver_install_failed: curl: (6) Could not resolve host: example.invalid; "
        "the chain does not retry it; see logs/driver-100.log"
    )
    assert (job / ".gym_chain_incomplete").read_text() == stop + "\n", result.stdout + result.stderr
    assert not (job / "admission_ran").exists()


def test_a_resumable_driver_step_names_its_failure_file_and_traps_only_its_own_install():
    config = SubmitConfig.model_validate(
        {
            "services": {"vllm_model": {"type": "vllm", "container": "vllm:latest", "model": "org/model"}},
            "compute": {"cluster": {"type": "slurm", "account": "my-account", "hostname": "login.example.invalid"}},
            "driver": {
                "gym_install": {"repo": "https://example.invalid/gym.git", "ref": "main"},
                "benchmarks": {"gsm8k": {"resumable": {"max_retries": 2}}},
            },
            "job": {"output_path": "/remote/jobs"},
        }
    )
    compute = next(iter(config.compute.values()))
    job_dir = Path("/remote/jobs/gym-job-x/gsm8k")
    script = build_sbatch_script(config, "gsm8k", config.driver.benchmarks["gsm8k"], compute, job_dir)
    driver_line = next(line for line in script.splitlines() if "--output=logs/driver-$SLURM_JOB_ID.log" in line)
    assert (
        'GYM_DRIVER_FAILURE_FILE=/remote/jobs/gym-job-x/gsm8k/.gym_driver_failure."$SLURM_JOB_ID" srun ' in driver_line
    )
    # The saved-state reader installs the same checkout; its own failure already stops the chain.
    assert script.count("echo driver_install_failed") == 1


@pytest.mark.parametrize("raised", ["spinup", "refusal", "other config error"])
def test_the_driver_records_the_kind_of_a_failure_the_next_segment_would_repeat(tmp_path, monkeypatch, raised):
    record = tmp_path / ".gym_driver_failure.100"
    monkeypatch.setenv("GYM_DRIVER_FAILURE_FILE", str(record))

    @exit_cleanly_on_config_error
    def driver():
        if raised == "spinup":
            raise ServerSpinupTimeoutError("Timed out after 600s waiting for Gym servers to become ready")
        if raised == "refusal":
            # Saved rollouts without a run manifest: their identity cannot be verified.
            validate_resume(tmp_path / "rollouts_manifest.json", None, tmp_path / "rollouts_materialized_inputs.jsonl")
        raise ConfigError("config_paths is missing")

    with pytest.raises(ServerSpinupTimeoutError if raised == "spinup" else SystemExit):
        driver()
    expected = {"spinup": "server_spinup_timeout\n", "refusal": "collection_resume_refused\n"}.get(raised)
    assert (record.read_text() if record.exists() else None) == expected
