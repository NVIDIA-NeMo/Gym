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
"""A scheduler cut (walltime or preemption) leaves reservations without rows; recovery resumes them.

Real Store, real stdlib collector and the real auto-resume prologue; only the scheduler is simulated.
"""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from nemo_gym.cli.utils import exit_cleanly_on_config_error
from nemo_gym.config.config_types import ConfigError
from nemo_gym.orchestration.api import ResumeConfig, SubmitConfig
from nemo_gym.orchestration.executors.script_templates import render_driver_entrypoint
from nemo_gym.orchestration.executors.slurm_script import build_sbatch_script
from nemo_gym.orchestration.recovery import (
    COLLECT_SCRIPT,
    CollectedEvidence,
    allocation_cut_from_accounting,
    chain_accounting,
    decide_from_collected,
    select_managed_evidence,
)
from nemo_gym.rollouts.rollout_recovery import RunManifest, validate_resume
from nemo_gym.rollouts.rollout_store import RolloutStore
from nemo_gym.server_runtime.supervisor.server_supervisor import ServerSpinupTimeoutError
from tests.unit_tests.orchestration.test_eval_resubmit import _handle, _resumable_run, _sbatch_env
from tests.unit_tests.orchestration.test_eval_resubmit import _local_index as _local_index
from tests.unit_tests.orchestration.test_eval_resubmit import slurm_bin as slurm_bin
from tests.unit_tests.orchestration.test_resume_script import _run_prologue


def _cut_run(job, *, scored=2, in_flight=2, not_started=2, masked=0, unowned=0, allocation="100", chain=True):
    """A run cut while ``in_flight`` reservations had no row; ``masked`` results and ``unowned`` failures."""
    artifacts = job / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    total = scored + in_flight + not_started + masked + unowned
    rows = [{"_ng_task_index": i, "_ng_rollout_index": 0} for i in range(total)]
    source = job / "source.jsonl"
    source.write_text("".join(json.dumps(row) + "\n" for row in rows))

    def prepare():
        inputs = json.loads(json.dumps(rows))
        return inputs, RunManifest.create(source, inputs, {}, {}).model_copy(update={"max_rollout_attempts": 3})

    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("SLURM_JOB_ID", allocation)
        with RolloutStore.start_or_resume(artifacts / "rollouts.jsonl", prepare, resume=False) as store:
            index = 0
            for _ in range(scored):
                row = dict(rows[index])
                store.record_dispatch(row)
                store.record_outcome(row | {"reward": 1.0})
                index += 1
            for _ in range(masked):
                row = dict(rows[index])
                store.record_dispatch(row)
                store.record_outcome(row | {"reward": 0.0, "mask_sample": True, "evaluation_completed": False})
                index += 1
            for _ in range(unowned):
                row = dict(rows[index])
                store.record_dispatch(row)
                store.record_outcome(row | {"_ng_failure_class": "unowned_class", "failure_kind": "not_registered"})
                index += 1
            for _ in range(in_flight):
                # Reserved and dispatched; the allocation ended before any outcome was saved.
                store.record_dispatch(dict(rows[index]))
                index += 1
    if chain:
        (job / ".gym_job_chain").write_text(f"{allocation}\n")
        (job / ".gym_accumulated_walltime").write_text("14400\n")
    return artifacts / "rollouts.jsonl"


def _decide(job, *, allocation_cut, live_job=False):
    raw = subprocess.run(
        [sys.executable, "-I", "-S", "-c", COLLECT_SCRIPT, str(job)], capture_output=True, text=True, check=True
    )
    collected = select_managed_evidence(CollectedEvidence.model_validate_json(raw.stdout), job)
    assert collected.managed_reader_error is None
    return collected, decide_from_collected(
        collected, max_infra_retries=2, max_walltime_s=None, live_job=live_job, allocation_cut=allocation_cut
    )


def _digests(job):
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (job / "artifacts").glob("*") if p.is_file()}


def test_the_in_job_admission_verifies_and_reads_the_store_once(tmp_path, monkeypatch, capsys):
    from nemo_gym.orchestration import recovery

    job = tmp_path / "job"
    _cut_run(job, scored=2, in_flight=2, not_started=2)
    calls = []
    select = recovery.select_managed_evidence
    monkeypatch.setattr(
        recovery,
        "select_managed_evidence",
        lambda collected, job_dir: calls.append(job_dir) or select(collected, job_dir),
    )
    assert recovery.continuation_admission(str(job), 2, None, "0:15", "100", "TIMEOUT") == "segment"
    # Each selection hashes every source file and reads the whole Store: one per admission is enough.
    assert calls == [job]
    assert "4 identity(ies) need the policy again" in capsys.readouterr().out


def test_in_flight_reservations_at_a_scheduler_cut_resume_with_never_started_rows(tmp_path):
    _cut_run(tmp_path)
    before = _digests(tmp_path)
    collected, decision = _decide(tmp_path, allocation_cut=True)
    assert collected.coverage_allocation_id == "100"
    assert decision.action.value == "resubmit"
    assert decision.counts == {"scored": 2, "missing": 4}
    assert decision.reason.startswith("4 identity(ies) need the policy again")
    assert _digests(tmp_path) == before


def test_without_a_scheduler_cut_reservations_stay_unknown_and_the_stop_names_three_populations(tmp_path):
    _cut_run(tmp_path)
    _, decision = _decide(tmp_path, allocation_cut=False)
    assert decision.action.value == "stop"
    assert decision.counts == {"scored": 2, "unknown": 2, "missing": 2}
    assert "a resume would run 2 of them again" in decision.reason
    assert (
        "retryable 2 (resumed), unknown 2 (2 reservation_without_outcome) (reported, not retried), "
        "retained 2 (saved rows kept)"
    ) in decision.reason


def test_a_scheduler_cut_is_not_applied_while_a_job_of_the_run_is_live(tmp_path):
    _cut_run(tmp_path)
    _, decision = _decide(tmp_path, allocation_cut=True, live_job=True)
    assert (decision.action.value, decision.scheduler_live) == ("noop", True)
    assert decision.counts["unknown"] == 2


def test_a_live_job_is_reported_before_rows_nobody_owns(tmp_path):
    from nemo_gym.orchestration.recovery import gate_task_summary

    _cut_run(tmp_path, masked=1)
    _, decision = _decide(tmp_path, allocation_cut=False, live_job=True)
    assert (decision.action.value, decision.scheduler_live) == ("noop", True)
    assert (
        decision.reason == "a job of this run is still pending or running; nothing to decide, it is not queued twice"
    )
    gate = gate_task_summary(
        decision, score=None, summary_valid=None, summary_reasons=[], health_passed=None, health_reasons=[]
    )
    assert gate["status"] == "RUNNING"


def test_a_pending_job_that_has_not_written_its_inventory_is_reported_as_live():
    collected = CollectedEvidence(materialized=None, results=None, failures=[])
    pending = decide_from_collected(collected, max_infra_retries=2, max_walltime_s=None, live_job=True)
    assert (pending.action.value, pending.scheduler_live) == ("noop", True)
    assert pending.reason == "a job of this run is still pending or running; nothing to decide, it is not queued twice"
    named = decide_from_collected(
        collected, max_infra_retries=2, max_walltime_s=None, live_job=True, live_job_name="101 (RUNNING)"
    )
    assert (named.action.value, named.reason) == ("noop", "continuation 101 (RUNNING) live; nothing to decide")


def test_a_run_that_ended_before_its_inventory_is_a_fresh_restart():
    collected = CollectedEvidence(materialized=None, results=None, failures=[])
    ended = decide_from_collected(collected, max_infra_retries=2, max_walltime_s=None, live_job=False)
    assert (ended.action.value, ended.scheduler_live, ended.expected) == ("fresh-restart", False, 0)
    assert ended.reason == (
        "nothing to recover: the run ended before the task inventory was written; "
        "resubmit it as a fresh run with a longer walltime"
    )
    # Rows saved without an inventory did not end before it, and a fresh run does not start over them.
    saved = collected.model_copy(update={"results": [{"_ng_task_index": 0, "_ng_rollout_index": 0}]})
    stopped = decide_from_collected(saved, max_infra_retries=2, max_walltime_s=None, live_job=False)
    assert stopped.action.value == "stop"
    assert stopped.reason == "the run has no materialized inventory to recover against"


def test_masked_results_are_retained_and_do_not_block_never_started_rows(tmp_path):
    _cut_run(tmp_path, scored=1, in_flight=0, not_started=3, masked=2)
    _, decision = _decide(tmp_path, allocation_cut=False)
    assert decision.action.value == "resubmit"
    assert decision.counts == {"scored": 1, "masked": 2, "missing": 3}
    assert decision.reason == (
        "3 identity(ies) need the policy again: a Gym resume runs only missing and retryable ones; "
        "retryable 3 (resumed), unknown 0 (reported, not retried), retained 1 (saved rows kept), "
        "masked 2 (completed, not measured, kept)"
    )


def test_unknown_failures_a_resume_would_rerun_still_stop(tmp_path):
    _cut_run(tmp_path, scored=1, in_flight=0, not_started=1, unowned=1)
    _, decision = _decide(tmp_path, allocation_cut=True)
    assert decision.action.value == "stop"
    assert "nobody owns them and a resume would run 1 of them again" in decision.reason
    assert "unknown 1 (1 unregistered kind not_registered)" in decision.reason


def test_only_masked_results_left_stop_and_say_they_are_kept(tmp_path):
    from nemo_gym.orchestration.recovery import gate_task_summary

    _cut_run(tmp_path, scored=1, in_flight=0, not_started=0, masked=2)
    _, decision = _decide(tmp_path, allocation_cut=True)
    assert decision.action.value == "stop"
    assert decision.reason == (
        "2 completed result(s) are masked and not measured: they are kept and a resume would run nothing; the run "
        "is not complete; retryable 0 (resumed), unknown 0 (reported, not retried), retained 1 (saved rows kept), "
        "masked 2 (completed, not measured, kept)"
    )
    assert not decision.complete
    gate = gate_task_summary(
        decision, score=1.0, summary_valid=True, summary_reasons=[], health_passed=True, health_reasons=[]
    )
    assert gate["status"] == "FAILED"
    assert gate["errors"][0] == "2 masked results, completed but not measured: [(1, 0), (2, 0)]"
    assert all("nobody owns" not in error for error in gate["errors"])


@pytest.mark.parametrize(
    "accounting,allocation,cut",
    [
        ("100|TIMEOUT|14400|0\n101|FAILED|90|0", "100", True),
        ("100|PREEMPTED|300|0", "100", True),
        ("100|CANCELLED by 7|300|0", "100", True),
        ("100|NODE_FAIL|300|0", "100", True),
        # A requeued id: only its first incarnation ran work; the requeued one stopped in the prologue.
        ("100|PREEMPTED|15000|0\n100|FAILED|14|1\n101|FAILED|16|0", "100", True),
        # Any ending counts: a failed driver, or one that completed through its no-progress exit.
        ("100|FAILED|300|0", "100", True),
        ("100|COMPLETED|16|0", "100", True),
        ("100|OUT_OF_MEMORY|300|0", "100", True),
        # Still alive, or its state is unknown: not an ending.
        ("100|RUNNING|300|0", "100", False),
        ("100|PENDING|0|0", "100", False),
        ("101|FAILED|60|0", "100", False),
        ("100|TIMEOUT|14400|0", "101", False),
        ("100|TIMEOUT|14400|0", None, False),
    ],
)
def test_the_cut_is_read_from_the_allocation_that_last_wrote_the_store(accounting, allocation, cut):
    assert allocation_cut_from_accounting(chain_accounting(accounting), allocation) is cut


@pytest.mark.parametrize("accounting", ["malformed", "100|TIMEOUT|14400", "100|TIMEOUT|x|0"])
def test_malformed_chain_accounting_is_unknown(accounting):
    assert chain_accounting(accounting) is None


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


@pytest.mark.parametrize("state", ["CANCELLED by 1000", "NODE_FAIL"])
def test_recover_resumes_reservations_after_an_operator_cancel_or_a_node_failure(tmp_path, slurm_bin, state):
    from nemo_gym.orchestration.submit import recover

    job, _, _ = _resumable_run(tmp_path)
    for name in ("rollouts.jsonl", "rollouts_materialized_inputs.jsonl"):
        (job / "artifacts" / name).unlink()
    _cut_run(job, scored=2, in_flight=2, not_started=1, chain=False)
    (slurm_bin / "accounting").write_text(f"100|{state}|7200|0\n101|FAILED|60|0\n")
    before = _digests(job)
    result = recover(*_handle(job))
    assert result.decision.action.value == "resubmit", result.decision.reason
    assert result.decision.counts == {"scored": 2, "missing": 3}
    assert result.dispatched == "resubmit"
    assert _sbatch_env(slurm_bin)["GYM_RESUBMIT_KEEP_BUDGET"] == "1"
    assert _digests(job) == before


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
    # After the inventory a timeout still continues: test_the_guard_continues_a_cut_allocation_from_its_saved_rows.


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


_REQUEUE_MARKER = (
    "GYM_ACCOUNTING_UNKNOWN: Requeued allocation elapsed time is unknown; "
    "recover only after complete terminal accounting.\n"
)


def _requeued_run(tmp_path, slurm_bin, accounting, *, marker=_REQUEUE_MARKER, stored_walltime=None):
    """Allocation 100 ran the Store, was preempted and requeued; the requeue and its follow-up 101 stopped early."""
    job, script, resume = _resumable_run(tmp_path)
    for name in ("rollouts.jsonl", "rollouts_materialized_inputs.jsonl"):
        (job / "artifacts" / name).unlink()
    _cut_run(job, scored=3, in_flight=2, not_started=0, chain=False)
    (job / ".gym_infra_retries").unlink()
    (job / ".gym_accumulated_walltime").unlink()
    if stored_walltime is not None:
        (job / ".gym_accumulated_walltime").write_text(f"{stored_walltime}\n")
    (job / ".gym_chain_incomplete").write_text(marker)
    (job / ".gym_followup_queued.100").write_text("101\n")
    (slurm_bin / "accounting").write_text(accounting)
    return job, script, resume


_TERMINAL_REQUEUE = "100|PREEMPTED|15000|0\n100|FAILED|14|1\n101|FAILED|16|0\n"


def test_a_terminal_requeued_chain_is_reconciled_and_resumes_its_in_flight_rows(tmp_path, slurm_bin):
    from nemo_gym.orchestration.executors.resume_script import REQUEUE_RECONCILED_NOTE
    from nemo_gym.orchestration.submit import inspect_recovery, recover
    from tests.unit_tests.orchestration.test_eval_resubmit import _start_queued_job

    job, script, resume = _requeued_run(tmp_path, slurm_bin, _TERMINAL_REQUEUE)
    before = _digests(job)
    inspected = inspect_recovery(*_handle(job))
    assert inspected.decision.action.value == "resubmit", inspected.decision.reason
    assert inspected.decision.counts == {"scored": 3, "missing": 2}
    assert inspected.dispatched is None and (job / ".gym_chain_incomplete").read_text() == _REQUEUE_MARKER

    result = recover(*_handle(job))
    assert result.dispatched == "resubmit"
    assert _sbatch_env(slurm_bin)["GYM_RESUBMIT_KEEP_BUDGET"] == "1"
    # Every incarnation is charged once: 15000 + 14 + 16 seconds.
    assert (job / ".gym_accumulated_walltime").read_text() == "15030\n"
    assert (job / ".gym_chain_incomplete").read_text() == REQUEUE_RECONCILED_NOTE + "\n"
    assert _digests(job) == before
    # The queued job passes the prologue's accounting checks, spends one infra retry and resumes saved rows.
    args = _start_queued_job(script, resume, job, slurm_bin)
    assert args[-1] == "++resume_from_cache=True"
    assert (job / ".gym_infra_retries").read_text() == "1\n"


def test_a_requeued_then_failed_incarnation_is_accounted_and_recoverable(tmp_path, slurm_bin):
    from nemo_gym.orchestration.executors.resume_script import REQUEUE_RECONCILED_NOTE
    from nemo_gym.orchestration.submit import recover

    # A node prolog failure: the first incarnation is REQUEUED with its Start and End recorded, the second FAILED.
    job, _, _ = _requeued_run(tmp_path, slurm_bin, "100|REQUEUED|36|0\n100|FAILED|15|1\n101|FAILED|16|0\n")
    result = recover(*_handle(job))
    assert result.dispatched == "resubmit", result.decision.reason
    assert (job / ".gym_accumulated_walltime").read_text() == "67\n"
    assert (job / ".gym_chain_incomplete").read_text() == REQUEUE_RECONCILED_NOTE + "\n"


def test_a_requeued_incarnation_without_an_end_still_blocks_recovery(tmp_path, slurm_bin):
    from nemo_gym.orchestration.submit import recover

    # The last incarnation is still REQUEUED (no End yet): its elapsed time is unknown.
    job, _, _ = _requeued_run(tmp_path, slurm_bin, "100|REQUEUED|36|0\n101|FAILED|16|0\n")
    result = recover(*_handle(job))
    # A requeued incarnation that has not ended is a live continuation: nothing is decided or queued.
    assert (result.decision.action.value, result.dispatched) == ("noop", None)
    assert (job / ".gym_chain_incomplete").read_text() == _REQUEUE_MARKER


@pytest.mark.parametrize(
    "accounting,marker,stored",
    [
        # The first incarnation's record is absent: the requeued chain is not completely accounted.
        ("100|FAILED|14|1\n101|FAILED|16|0\n", _REQUEUE_MARKER, None),
        # The requeued incarnation is still running.
        ("100|PREEMPTED|15000|0\n100|RUNNING|14|1\n101|FAILED|16|0\n", _REQUEUE_MARKER, None),
        # The stored counter does not match the jobs before the requeue: the budget epoch is unknown.
        (_TERMINAL_REQUEUE, _REQUEUE_MARKER, 7200),
        # Another unknown reason is never reconciled from scheduler records.
        (
            _TERMINAL_REQUEUE,
            "GYM_ACCOUNTING_UNKNOWN: Follow-up submission may have succeeded without an observed reply.\n",
            None,
        ),
    ],
)
def test_an_incomplete_or_other_unknown_accounting_is_not_reconciled(tmp_path, slurm_bin, accounting, marker, stored):
    from nemo_gym.orchestration.submit import recover

    job, _, _ = _requeued_run(tmp_path, slurm_bin, accounting, marker=marker, stored_walltime=stored)
    result = recover(*_handle(job))
    # A requeued incarnation that still runs is a live continuation: nothing to decide, nothing queued.
    live = "RUNNING" in accounting
    assert result.decision.action.value == ("noop" if live else "stop") and result.dispatched is None
    assert not (slurm_bin / "env").exists()
    assert (job / ".gym_chain_incomplete").read_text() == marker


_TWICE_REQUEUED = "100|PREEMPTED|15000|0\n100|FAILED|14|1\n101|FAILED|16|0\n102|PREEMPTED|16455|0\n102|FAILED|13|1\n103|FAILED|12|0\n"


def _twice_requeued_run(tmp_path, slurm_bin, *, stored_walltime):
    """100's requeue was reconciled and the chain went on; 102 ran the Store, was preempted and requeued."""
    job, script, resume = _resumable_run(tmp_path)
    for name in ("rollouts.jsonl", "rollouts_materialized_inputs.jsonl"):
        (job / "artifacts" / name).unlink()
    _cut_run(job, scored=3, in_flight=2, not_started=0, allocation="102", chain=False)
    (job / ".gym_job_chain").write_text("100\n101\n102\n103\n")
    (job / ".gym_infra_retries").write_text("1\n")
    (job / ".gym_accumulated_walltime").write_text(f"{stored_walltime}\n")
    (job / ".gym_chain_incomplete").write_text(_REQUEUE_MARKER)
    (job / ".gym_followup_queued.102").write_text("103\n")
    (slurm_bin / "accounting").write_text(_TWICE_REQUEUED)
    return job


def test_a_second_requeue_is_reconciled_from_every_allocation_and_named(tmp_path, slurm_bin):
    from nemo_gym.orchestration.executors.resume_script import REQUEUE_RECONCILED_NOTE
    from nemo_gym.orchestration.submit import inspect_recovery, recover

    # Stored before 102 ran: both incarnations of 100 and the 16 s of 101.
    job = _twice_requeued_run(tmp_path, slurm_bin, stored_walltime=15030)
    decision = inspect_recovery(*_handle(job)).decision
    assert decision.action.value == "resubmit", decision.reason
    assert decision.counts == {"scored": 3, "missing": 2}
    assert decision.reason.endswith("; job 102 was preempted and requeued; its 2 allocations account for 16468 s")

    result = recover(*_handle(job))
    assert result.dispatched == "resubmit"
    # Every allocation of the chain is charged once: 15030 + 16455 + 13 + 12 seconds.
    assert (job / ".gym_accumulated_walltime").read_text() == "31510\n"
    assert (job / ".gym_chain_incomplete").read_text() == REQUEUE_RECONCILED_NOTE + "\n"


def test_a_second_requeue_whose_stored_walltime_does_not_match_is_not_reconciled(tmp_path, slurm_bin):
    from nemo_gym.orchestration.submit import recover

    # 15000 leaves out the 14 s and 16 s before 102: which allocations it covers is unknown.
    job = _twice_requeued_run(tmp_path, slurm_bin, stored_walltime=15000)
    result = recover(*_handle(job))
    assert (result.decision.action.value, result.dispatched) == ("stop", None)
    assert "requeued" not in result.decision.reason
    assert (job / ".gym_chain_incomplete").read_text() == _REQUEUE_MARKER


def _inspected_cut(tmp_path, slurm_bin, accounting):
    """A cut allocation 100 of a chain 100, 101, inspected the way the operator's login host does it."""
    from nemo_gym.orchestration.submit import inspect_recovery

    job, _, _ = _resumable_run(tmp_path)
    for name in ("rollouts.jsonl", "rollouts_materialized_inputs.jsonl"):
        (job / "artifacts" / name).unlink()
    _cut_run(job, scored=2, in_flight=2, not_started=2, chain=False)
    (job / ".gym_job_chain").write_text("100\n101\n")
    (slurm_bin / "accounting").write_text(accounting)
    return job, inspect_recovery(*_handle(job))


@pytest.mark.parametrize("state", ["PENDING", "RUNNING"])
def test_the_inspect_reports_a_live_continuation_of_the_chain_and_decides_nothing(tmp_path, slurm_bin, state):
    job, inspected = _inspected_cut(tmp_path, slurm_bin, f"100|TIMEOUT|7200|0\n101|{state}|0|0\n")
    decision = inspected.decision
    assert (decision.action.value, decision.scheduler_live) == ("noop", True), decision.reason
    assert decision.reason == f"continuation 101 ({state}) live; nothing to decide"
    assert inspected.gate_task["status"] == "RUNNING" and inspected.dispatched is None
    assert not (slurm_bin / "env").exists()


def test_the_inspect_and_the_guard_classify_a_cut_allocation_the_same_way(tmp_path, slurm_bin):
    job, inspected = _inspected_cut(tmp_path, slurm_bin, "100|TIMEOUT|7200|0\n101|FAILED|60|0\n")
    assert inspected.decision.action.value == "resubmit", inspected.decision.reason
    assert inspected.decision.counts == {"scored": 2, "missing": 4}
    assert inspected.decision.reason.startswith("4 identity(ies) need the policy again")
    # The in-allocation guard of the next job, on the same saved files, sees the same four.
    result = _run_prologue(job, ResumeConfig(max_retries=2), "TIMEOUT", exit_code="0:15", prev_job_id="100")
    assert "Saved recovery evidence: 4 identity(ies) need the policy again" in result.stdout, result.stdout


@pytest.mark.parametrize("state", ["FAILED", "COMPLETED"])
def test_the_inspect_resumes_reservations_of_an_allocation_that_ended_by_any_path(tmp_path, slurm_bin, state):
    _, inspected = _inspected_cut(tmp_path, slurm_bin, f"100|{state}|7200|0\n101|FAILED|60|0\n")
    assert inspected.decision.action.value == "resubmit", inspected.decision.reason
    assert inspected.decision.counts == {"scored": 2, "missing": 4}


def test_reservations_of_an_allocation_still_running_are_not_decided(tmp_path, slurm_bin):
    _, inspected = _inspected_cut(tmp_path, slurm_bin, "100|RUNNING|7200|0\n")
    assert (inspected.decision.action.value, inspected.decision.scheduler_live) == ("noop", True)
    assert inspected.decision.counts["unknown"] == 2


def test_reservations_without_a_known_ending_never_read_as_failures_nobody_owns(tmp_path, slurm_bin):
    # The accounting has no record of the allocation that made them.
    _, inspected = _inspected_cut(tmp_path, slurm_bin, "101|FAILED|60|0\n")
    reason = inspected.decision.reason
    assert inspected.decision.action.value == "stop"
    assert reason.startswith(
        "2 reservation(s) have no saved outcome and the allocation that made them is not known to have ended"
    )
    assert "nobody owns them" not in reason


def test_an_installer_error_with_a_credential_never_leaves_the_run_host(tmp_path):
    secret = "synthetic-install-secret"
    job = tmp_path / "gym-job-x" / "gsm8k"
    job.mkdir(parents=True)
    (job / ".gym_chain_incomplete").write_text(
        "Previous job 100 ended FAILED: driver_install_failed: fatal: unable to access "
        f"'https://user:{secret}@example.invalid/gym.git/': Could not resolve host; the chain does not retry it; "
        "see logs/driver-100.log\n"
    )
    raw = subprocess.run(
        [sys.executable, "-I", "-S", "-c", COLLECT_SCRIPT, str(job)], capture_output=True, text=True, check=True
    )
    assert secret not in raw.stdout
    collected = CollectedEvidence.model_validate_json(raw.stdout)
    # The typed failure and the driver-log reference stay; the raw log on the run host holds the detail.
    assert "driver_install_failed" in collected.chain_continuation_note
    assert "see logs/driver-100.log" in collected.chain_continuation_note
    decision = decide_from_collected(collected, max_infra_retries=2, max_walltime_s=None, live_job=False)
    assert secret not in decision.reason
    assert "driver_install_failed" in decision.reason
