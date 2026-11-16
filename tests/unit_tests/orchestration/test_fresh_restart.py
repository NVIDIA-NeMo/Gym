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
"""A run that ended before Gym wrote its task inventory: recovery plans a fresh restart with a longer walltime."""

import hashlib
import json
from pathlib import Path

import pytest

from nemo_gym.orchestration.api import ResumeConfig
from nemo_gym.orchestration.executors.slurm import WALLTIME_JOURNAL_NAME
from tests.unit_tests.orchestration.test_eval_resubmit import _handle, _sbatch_env, _start_queued_job
from tests.unit_tests.orchestration.test_eval_resubmit import _local_index as _local_index
from tests.unit_tests.orchestration.test_eval_resubmit import slurm_bin as slurm_bin
from tests.unit_tests.orchestration.test_eval_resubmit_decision import run_cli
from tests.unit_tests.orchestration.test_resubmit_repin import launcher_at as launcher_at
from tests.unit_tests.orchestration.test_resubmit_walltime import (
    _launcher_at_the_pinned_revision as _launcher_at_the_pinned_revision,
)
from tests.unit_tests.orchestration.test_resubmit_walltime import _timed_run


REASON = (
    "nothing to recover: the run ended before the task inventory was written; "
    "resubmit it as a fresh run with a longer walltime"
)
COMMAND = "gym eval resubmit gym-job-x gsm8k --recover --max-walltime HH:MM:SS --walltime-reason TEXT"


def _cut_before_inventory(tmp_path: Path, slurm_bin: Path) -> Path:
    """Allocation 100 used its 4 h walltime before Gym wrote the task inventory; follow-up 101 ended the chain."""
    job_dir = _timed_run(tmp_path)
    for name in ("rollouts.jsonl", "rollouts_materialized_inputs.jsonl"):
        (job_dir / "artifacts" / name).unlink()
    # What the cut run saved: its resolved config and its driver log.
    (job_dir / "artifacts" / "rollouts_resolved_config.yaml").write_text("config_paths: []\n")
    (job_dir / "logs").mkdir()
    (job_dir / "logs" / "driver-100.log").write_text("waiting for the model server\n")
    (job_dir / ".gym_accumulated_walltime").write_text("14400\n")
    (slurm_bin / "accounting").write_text("100|TIMEOUT|14400|0\n101|FAILED|30|0\n")
    return job_dir


def _saved(job_dir: Path) -> dict[str, str]:
    """The cut run's saved files: its artifacts and its logs."""
    return {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for folder in ("artifacts", "logs")
        for path in (job_dir / folder).rglob("*")
        if path.is_file()
    }


def test_without_max_walltime_the_plan_names_the_fresh_restart_command_and_runs_nothing(tmp_path, slurm_bin, capsys):
    from nemo_gym.orchestration.submit import inspect_recovery, recover

    job_dir = _cut_before_inventory(tmp_path, slurm_bin)
    before, script = _saved(job_dir), (job_dir / "job.sh").read_text()

    out = recover(*_handle(job_dir))
    assert (out.decision.action.value, out.ok, out.dispatched) == ("fresh-restart", False, None)
    assert out.decision.reason == f"{REASON}; --max-walltime is required: {COMMAND}"
    assert out.gate_task["status"] == "PLANNED"
    inspected = inspect_recovery(*_handle(job_dir))
    assert (inspected.decision.reason, inspected.dispatched) == (out.decision.reason, None)
    warning = "PLANNED (NOT DISPATCHED): --inspect reports the fresh-restart decision and runs nothing"
    assert warning in inspected.gate_task["warnings"]
    code, printed = run_cli(["eval", "resubmit", "gym-job-x", "gsm8k", "--recover"], capsys)
    assert (code, printed["decision"]["action"], printed["dispatched"]) == (1, "fresh-restart", None)
    assert printed["decision"]["reason"].endswith(COMMAND)

    # Nothing was queued, rendered or recorded, and the cut run's files are as it left them.
    assert not (slurm_bin / "env").exists() and (job_dir / "job.sh").read_text() == script
    assert not (job_dir.parent / WALLTIME_JOURNAL_NAME).exists() and not (job_dir / ".gym_chain_incomplete").exists()
    assert _saved(job_dir) == before


def test_with_a_longer_walltime_the_fresh_restart_is_queued_and_recorded(tmp_path, slurm_bin):
    from nemo_gym.orchestration.submit import recover

    job_dir = _cut_before_inventory(tmp_path, slurm_bin)
    run_dir = job_dir.parent
    before = _saved(job_dir)

    out = recover(*_handle(job_dir), max_walltime="08:00:00", walltime_reason="startup takes longer than 4 h")
    assert (out.decision.action.value, out.decision.reason) == ("fresh-restart", REASON)
    assert (out.ok, out.dispatched, out.job_id, out.gate_task["status"]) == (True, "fresh-restart", "4242", "RESUMING")
    # A fresh run: queued as a plain resubmit, whose job resets the retry and walltime budgets.
    env = _sbatch_env(slurm_bin)
    assert env["GYM_RESUBMIT"] == "1" and "GYM_RESUBMIT_KEEP_BUDGET" not in env
    script = (job_dir / "job.sh").read_text()
    assert "#SBATCH --time=08:00:00" in script
    # The receipt and the chain-file entry: FRESH-RESTART, why, the old walltime and the new one.
    [entry] = [json.loads(line) for line in (run_dir / WALLTIME_JOURNAL_NAME).read_text().splitlines()]
    assert {key: entry[key] for key in ("plan", "why", "from", "to", "reason")} == {
        "plan": "FRESH-RESTART",
        "why": "the run ended before the task inventory was written",
        "from": "04:00:00",
        "to": "08:00:00",
        "reason": "startup takes longer than 4 h",
    }
    assert (job_dir / ".gym_chain_incomplete").read_text() == (
        "FRESH-RESTART: the run ended before the task inventory was written; walltime 04:00:00 -> 08:00:00\n"
    )
    [job] = json.loads((run_dir / "gym-job.json").read_text())["benchmarks"]
    assert (job["job_id"], job["walltime"]) == ("4242", "08:00:00")
    assert _saved(job_dir) == before

    # The queued job starts the run from the beginning with fresh budgets, and the chain it continues drops the note.
    args = _start_queued_job(script, ResumeConfig(max_retries=3, max_walltime="16:00:00"), job_dir, slurm_bin)
    assert args[-2:] == ["++resubmit=True", "++resume_from_cache=True"]
    assert not (job_dir / ".gym_infra_retries").exists() and not (job_dir / ".gym_accumulated_walltime").exists()
    assert not (job_dir / ".gym_chain_incomplete").exists()


def test_a_fresh_restart_rechecks_the_collected_chain_under_the_lock(tmp_path, slurm_bin, monkeypatch):
    from nemo_gym.orchestration.executors.slurm import SlurmExecutor
    from nemo_gym.orchestration.submit import recover

    job_dir = _cut_before_inventory(tmp_path, slurm_bin)
    collect = SlurmExecutor.recovery_evidence

    def collect_then_start_a_job(self, *args, **kwargs):
        # An allocation of the run starts and records itself after the evidence was collected.
        result = collect(self, *args, **kwargs)
        (job_dir / ".gym_job_chain").write_text("100\n101\n102\n")
        return result

    monkeypatch.setattr(SlurmExecutor, "recovery_evidence", collect_then_start_a_job)
    script = (job_dir / "job.sh").read_text()
    with pytest.raises(RuntimeError, match="changed since the recovery evidence was collected"):
        recover(*_handle(job_dir), max_walltime="08:00:00", walltime_reason="startup takes longer than 4 h")
    assert (job_dir / "job.sh").read_text() == script and not (slurm_bin / "env").exists()
    assert not (job_dir.parent / WALLTIME_JOURNAL_NAME).exists() and not (job_dir / ".gym_chain_incomplete").exists()


@pytest.mark.parametrize("live", ["queued", "chain"])
def test_while_a_job_of_the_run_is_live_nothing_is_planned_or_queued(tmp_path, slurm_bin, live):
    from nemo_gym.orchestration.submit import recover

    job_dir = _cut_before_inventory(tmp_path, slurm_bin)
    if live == "queued":
        (slurm_bin / "squeue_workdirs").write_text(f"{job_dir}\n")
        reason = "a job of this run is still pending or running; nothing to decide, it is not queued twice"
    else:
        (slurm_bin / "accounting").write_text("100|TIMEOUT|14400|0\n101|RUNNING|30|0\n")
        reason = "continuation 101 (RUNNING) live; nothing to decide"

    out = recover(*_handle(job_dir), max_walltime="08:00:00", walltime_reason="startup takes longer than 4 h")
    assert (out.decision.action.value, out.decision.scheduler_live, out.decision.reason) == ("noop", True, reason)
    assert (out.dispatched, out.gate_task["status"]) == (None, "RUNNING")
    assert not (slurm_bin / "env").exists() and not (job_dir.parent / WALLTIME_JOURNAL_NAME).exists()


@pytest.mark.parametrize(
    "flags,error",
    [
        ({"max_walltime": "04:00:00"}, "--max-walltime 04:00:00 is not longer than the recorded walltime 04:00:00"),
        ({"max_walltime": "08:00:00", "extend_retries": 2}, "a fresh restart takes only --max-walltime and"),
    ],
)
def test_a_fresh_restart_is_refused_before_anything_changes(tmp_path, slurm_bin, flags, error):
    from nemo_gym.orchestration.submit import recover

    job_dir = _cut_before_inventory(tmp_path, slurm_bin)
    files = {path: path.read_bytes() for path in (job_dir / "job.sh", job_dir.parent / "resolved-config.yaml")}
    with pytest.raises(ValueError, match=error):
        recover(*_handle(job_dir), walltime_reason="startup takes longer than 4 h", **flags)
    assert {path: path.read_bytes() for path in files} == files
    assert not (slurm_bin / "env").exists() and not (job_dir.parent / WALLTIME_JOURNAL_NAME).exists()
    assert not (job_dir / ".gym_chain_incomplete").exists()


def test_a_fresh_restart_from_a_launcher_of_another_revision_is_refused(tmp_path, slurm_bin, monkeypatch):
    import nemo_gym.orchestration.executors.slurm as slurm_executor
    from nemo_gym.orchestration.executors.slurm import IncompatibleRerenderError
    from nemo_gym.orchestration.submit import recover
    from tests.unit_tests.orchestration.test_resubmit_repin import NEW, OLD

    monkeypatch.setattr(slurm_executor, "installed_gym_commit", lambda: NEW)
    job_dir = _cut_before_inventory(tmp_path, slurm_bin)
    files = {path: path.read_bytes() for path in (job_dir / "job.sh", job_dir.parent / "resolved-config.yaml")}
    with pytest.raises(IncompatibleRerenderError, match=f"pinned to Gym {OLD}"):
        recover(*_handle(job_dir), max_walltime="08:00:00", walltime_reason="startup takes longer than 4 h")
    assert {path: path.read_bytes() for path in files} == files
    assert not (slurm_bin / "env").exists() and not (job_dir.parent / WALLTIME_JOURNAL_NAME).exists()
    assert not (job_dir / ".gym_chain_incomplete").exists()


def test_the_verdict_of_a_fresh_restart_is_its_own_resubmit_command_not_a_new_submission(tmp_path, slurm_bin):
    from nemo_gym.orchestration.submit import inspect_recovery, recover

    job_dir = _cut_before_inventory(tmp_path, slurm_bin)

    for out in (recover(*_handle(job_dir)), inspect_recovery(*_handle(job_dir))):
        verdict = out.operator_verdict
        assert verdict["state"] == "needs_action" and "`gym eval submit`" not in verdict["action"]
        assert verdict["action"].endswith(f"`{COMMAND}`")
