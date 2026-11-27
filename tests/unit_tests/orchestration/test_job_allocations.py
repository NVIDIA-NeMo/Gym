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
import json
import os
from pathlib import Path

import pytest

from nemo_gym.orchestration.executors.slurm.job_allocations import (
    SLURM_ENVIRONMENT_SOURCE,
    own_allocation_row,
    record_own_allocation,
)
from nemo_gym.orchestration.executors.slurm.resume_script import (
    JOB_ALLOCATIONS_LOCK_NAME,
    JOB_ALLOCATIONS_NAME,
    JOB_CHAIN_NAME,
)
from nemo_gym.orchestration.export_step import BENCH_DIR_ENV, record_allocation_before_export


# What a driver step of a one-node job with `--gpus-per-node=4` sees (Slurm sets these for the step).
DRIVER_STEP_ENV = {
    "SLURM_JOB_ID": "2352017",
    "SLURM_JOB_START_TIME": "1795783189",
    "SLURM_JOB_NUM_NODES": "1",
    "SLURM_JOB_NODELIST": "nvl72d205-T16",
    "SLURM_GPUS_PER_NODE": "4",
    "SLURM_GPUS_ON_NODE": "4",
}
NOW = 1795791145.7


def _rows(job_dir: Path) -> list[dict]:
    path = job_dir / JOB_ALLOCATIONS_NAME
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


@pytest.fixture
def job_dir(tmp_path: Path) -> Path:
    (tmp_path / JOB_CHAIN_NAME).write_text("2352017\n")
    return tmp_path


def test_the_running_allocation_is_read_from_the_driver_steps_slurm_environment() -> None:
    assert own_allocation_row(DRIVER_STEP_ENV, NOW) == {
        "alloc_tres": "",
        "elapsed_s": 7956,
        "end_epoch_s": None,
        "ended": False,
        "gpus": 4,
        "job_id": "2352017",
        "node_list": "nvl72d205-T16",
        "nodes": 1,
        "recorded_at": "2026-11-27T14:52:25Z",
        "restarts": 0,
        "source": SLURM_ENVIRONMENT_SOURCE,
        "start_epoch_s": 1795783189,
        "state": "RUNNING",
    }


@pytest.mark.parametrize(
    "gpus_per_node,nodes,gpus",
    [("gb300:4", "2", 8), (None, "2", None), ("mixed", "1", None)],
    ids=["typed-request", "no-uniform-request", "unparseable"],
)
def test_gpus_come_only_from_a_uniform_per_node_request(gpus_per_node: str | None, nodes: str, gpus: int | None):
    env = {**DRIVER_STEP_ENV, "SLURM_JOB_NUM_NODES": nodes}
    env.pop("SLURM_GPUS_PER_NODE")
    if gpus_per_node is not None:
        env["SLURM_GPUS_PER_NODE"] = gpus_per_node

    # The step's own count (SLURM_GPUS_ON_NODE) is one node's, so it is never multiplied up.
    assert own_allocation_row(env, NOW)["gpus"] == gpus


@pytest.mark.parametrize("missing", ["SLURM_JOB_ID", "SLURM_JOB_START_TIME", "SLURM_JOB_NUM_NODES"])
def test_an_incomplete_environment_gives_no_row(missing: str) -> None:
    assert own_allocation_row({k: v for k, v in DRIVER_STEP_ENV.items() if k != missing}, NOW) is None


def test_a_requeued_incarnation_keeps_its_restart_count() -> None:
    assert own_allocation_row({**DRIVER_STEP_ENV, "SLURM_RESTART_COUNT": "2"}, NOW)["restarts"] == 2


def test_the_row_is_appended_once_in_the_job_scripts_format(job_dir: Path) -> None:
    (job_dir / JOB_ALLOCATIONS_NAME).write_text('{"ended":true,"job_id":"2352000","restarts":0}\n')

    assert record_own_allocation(job_dir, environ=DRIVER_STEP_ENV, now=lambda: NOW)
    # A second export of the same job adds nothing.
    assert not record_own_allocation(job_dir, environ=DRIVER_STEP_ENV, now=lambda: NOW + 60)

    lines = (job_dir / JOB_ALLOCATIONS_NAME).read_text().splitlines()
    assert len(lines) == 2
    # Compact, sorted keys: the job script's reader matches `"job_id":"<id>"`, `"restarts":<n>` and `"ended":true`.
    assert lines[1].startswith('{"alloc_tres":"","elapsed_s":7956,"end_epoch_s":null,"ended":false,"gpus":4,')
    assert not (job_dir / JOB_ALLOCATIONS_LOCK_NAME).exists()


def test_a_job_that_is_not_resumable_records_nothing(tmp_path: Path) -> None:
    assert not record_own_allocation(tmp_path, environ=DRIVER_STEP_ENV, now=lambda: NOW)
    assert not (tmp_path / JOB_ALLOCATIONS_NAME).exists()


def test_a_held_lock_makes_it_give_up_without_a_row(job_dir: Path) -> None:
    lock = job_dir / JOB_ALLOCATIONS_LOCK_NAME
    lock.write_text("")
    os.utime(lock, (NOW, NOW))
    waits: list[float] = []

    assert not record_own_allocation(job_dir, environ=DRIVER_STEP_ENV, now=lambda: NOW + 1, sleep=waits.append)
    assert _rows(job_dir) == [] and len(waits) == 50
    # The lock is the other writer's: it stays.
    assert lock.exists()


def test_a_stale_lock_is_taken_over(job_dir: Path) -> None:
    lock = job_dir / JOB_ALLOCATIONS_LOCK_NAME
    lock.write_text("")
    os.utime(lock, (NOW - 120, NOW - 120))

    assert record_own_allocation(job_dir, environ=DRIVER_STEP_ENV, now=lambda: NOW)
    assert [row["job_id"] for row in _rows(job_dir)] == ["2352017"]
    assert not lock.exists()


def test_the_export_step_records_the_allocation_of_the_job_it_runs_in(job_dir: Path, monkeypatch) -> None:
    monkeypatch.setattr("nemo_gym.orchestration.executors.slurm.job_allocations.time.time", lambda: NOW)

    record_allocation_before_export({**DRIVER_STEP_ENV, BENCH_DIR_ENV: str(job_dir)})

    assert [(row["job_id"], row["ended"], row["gpus"]) for row in _rows(job_dir)] == [("2352017", False, 4)]


def test_the_export_step_outside_a_submitted_job_records_nothing(job_dir: Path) -> None:
    record_allocation_before_export(DRIVER_STEP_ENV)

    assert _rows(job_dir) == []


def test_a_job_directory_that_cannot_be_written_does_not_stop_the_export(job_dir: Path, monkeypatch, capsys):
    def refuse(*_args, **_kwargs):
        raise PermissionError("read-only file system")

    monkeypatch.setattr("nemo_gym.orchestration.executors.slurm.job_allocations.os.open", refuse)

    record_allocation_before_export({**DRIVER_STEP_ENV, BENCH_DIR_ENV: str(job_dir)})

    assert "the job's allocation was not recorded before the export: read-only file system" in capsys.readouterr().err
