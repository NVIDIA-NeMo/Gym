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
"""A `--recover --max-walltime` raise above the partition's MaxTime: segments under the raised cap, or a typed refusal."""

import json
from pathlib import Path

import pytest
import yaml

from nemo_gym.orchestration.executors import slurm as sm
from nemo_gym.orchestration.executors.connection import LocalConnection
from nemo_gym.orchestration.executors.slurm import WALLTIME_JOURNAL_NAME, SlurmExecutor
from nemo_gym.orchestration.executors.slurm_script import build_sbatch_script
from nemo_gym.orchestration.recovery import RecoveryAction
from tests.unit_tests.orchestration.test_eval_resubmit import _handle
from tests.unit_tests.orchestration.test_eval_resubmit import _local_index as _local_index
from tests.unit_tests.orchestration.test_eval_resubmit import slurm_bin as slurm_bin
from tests.unit_tests.orchestration.test_resubmit_repin import launcher_at as launcher_at
from tests.unit_tests.orchestration.test_resubmit_walltime import (
    _launcher_at_the_pinned_revision as _launcher_at_the_pinned_revision,
)
from tests.unit_tests.orchestration.test_resubmit_walltime import _timed_run


REFUSAL = "sbatch: error: Batch job submission failed: Requested time limit is invalid (missing or exceeds some limit)"


def _run(tmp_path: Path, cap: str | None = "08:00:00") -> Path:
    """The 4 h-segment run of test_resubmit_walltime, with its chain cap set to ``cap``."""
    job_dir = _timed_run(tmp_path)
    resolved = job_dir.parent / "resolved-config.yaml"
    saved = yaml.safe_load(resolved.read_text())
    saved["driver"]["benchmarks"]["gsm8k"]["resumable"]["max_walltime"] = cap
    resolved.write_text(yaml.safe_dump(saved, sort_keys=False))
    config = SlurmExecutor._saved_config(LocalConnection(), job_dir.parent)
    bench = config.driver.benchmarks["gsm8k"]
    (job_dir / "job.sh").write_text(
        build_sbatch_script(config, "gsm8k", bench, next(iter(config.compute.values())), job_dir)
    )
    return job_dir


def _sinfo(slurm_bin: Path, listing: str) -> None:
    """sinfo answers the default-partition listing (`%P %l`) with ``listing`` and records its arguments."""
    (slurm_bin / "sinfo").write_text(f'#!/bin/bash\necho "$*" >> "$(dirname "$0")/sinfo_args"\nprintf "{listing}"\n')
    (slurm_bin / "sinfo").chmod(0o755)


def test_a_raise_above_the_partition_max_time_runs_max_time_segments_under_the_raised_cap(tmp_path, slurm_bin):
    from nemo_gym.orchestration.submit import recover

    job_dir = _run(tmp_path)
    run_dir = job_dir.parent
    _sinfo(slurm_bin, "batch* 4:00:00\\nlong 2-00:00:00\\n")
    out = recover(*_handle(job_dir), max_walltime="12:00:00", walltime_reason="slow but healthy policy")
    assert out.ok and out.dispatched == "resubmit", out.decision.reason
    assert out.walltime_plan == {
        "partition": "batch",
        "max_time": "4:00:00",
        "requested": "12:00:00",
        "segment": "04:00:00",
        "max_walltime_from": "08:00:00",
        "max_walltime_to": "12:00:00",
        "refused": None,
        "note": "--max-walltime 12:00:00 exceeds partition batch MaxTime 4:00:00: each segment requests 04:00:00 "
        "and the chain continues under the 12:00:00 cap",
    }
    assert "#SBATCH --time=04:00:00" in (job_dir / "job.sh").read_text()
    saved = yaml.safe_load((run_dir / "resolved-config.yaml").read_text())
    assert saved["compute"]["cluster"]["walltime"] == "04:00:00"
    assert saved["driver"]["benchmarks"]["gsm8k"]["resumable"]["max_walltime"] == "12:00:00"
    [entry] = [json.loads(line) for line in (run_dir / WALLTIME_JOURNAL_NAME).read_text().splitlines()]
    assert (entry["to"], entry["segment"], entry["max_walltime_to"]) == ("12:00:00", "04:00:00", "12:00:00")
    [job] = json.loads((run_dir / "gym-job.json").read_text())["benchmarks"]
    assert job["walltime"] == "04:00:00"
    assert (slurm_bin / "sinfo_args").read_text().splitlines() == ["-h -o %P %l"]


@pytest.mark.parametrize("max_time,segment", [("4:00:00", "04:00:00"), ("1-00:00:00", "12:00:00")])
def test_a_raise_on_a_run_without_a_chain_cap_sets_the_cap_to_the_raise(tmp_path, slurm_bin, max_time, segment):
    from nemo_gym.orchestration.submit import recover

    job_dir = _run(tmp_path, cap=None)
    run_dir = job_dir.parent
    _sinfo(slurm_bin, f"batch* {max_time}\\n")
    out = recover(*_handle(job_dir), max_walltime="12:00:00", walltime_reason="slow but healthy policy")
    assert out.ok and out.dispatched == "resubmit", out.decision.reason
    # A raise always bounds the total: the uncapped chain gets the raise as its cap, and segments fit the partition.
    assert out.decision.reason.endswith("; walltime cap set to 12:00:00 for this plan (the run had none)")
    plan = out.walltime_plan
    assert (plan["segment"], plan["max_walltime_from"], plan["max_walltime_to"]) == (segment, None, "12:00:00")
    assert f"#SBATCH --time={segment}" in (job_dir / "job.sh").read_text()
    saved = yaml.safe_load((run_dir / "resolved-config.yaml").read_text())
    assert saved["driver"]["benchmarks"]["gsm8k"]["resumable"]["max_walltime"] == "12:00:00"
    assert saved["compute"]["cluster"]["walltime"] == segment
    [entry] = [json.loads(line) for line in (run_dir / WALLTIME_JOURNAL_NAME).read_text().splitlines()]
    assert (entry["max_walltime_from"], entry["max_walltime_to"], entry["segment"]) == (None, "12:00:00", segment)


def test_a_raise_the_partition_takes_renders_as_before(tmp_path, slurm_bin):
    from nemo_gym.orchestration.submit import recover

    job_dir = _run(tmp_path)
    _sinfo(slurm_bin, "batch* 1-00:00:00\\n")
    out = recover(*_handle(job_dir), max_walltime="12:00:00", walltime_reason="slow but healthy policy")
    assert out.ok and out.dispatched == "resubmit", out.decision.reason
    assert (out.walltime_plan["segment"], out.walltime_plan["refused"], out.walltime_plan["note"]) == (
        "12:00:00",
        None,
        None,
    )
    assert "#SBATCH --time=12:00:00" in (job_dir / "job.sh").read_text()


def test_an_unknown_max_time_keeps_the_whole_raise_and_says_so(tmp_path, slurm_bin):
    from nemo_gym.orchestration.submit import recover

    job_dir = _run(tmp_path)  # The fixture's sinfo fails: the query fails.
    out = recover(*_handle(job_dir), max_walltime="12:00:00", walltime_reason="slow but healthy policy")
    assert out.ok and out.dispatched == "resubmit", out.decision.reason
    assert out.walltime_plan["segment"] == "12:00:00" and out.walltime_plan["max_time"] is None
    assert out.walltime_plan["note"].startswith("partition (default) MaxTime unknown (")
    assert out.walltime_plan["note"].endswith("each segment requests the whole --max-walltime 12:00:00 as before")
    assert "#SBATCH --time=12:00:00" in (job_dir / "job.sh").read_text()


def test_a_partition_too_short_for_one_segment_is_refused_before_the_evidence_read(tmp_path, slurm_bin, monkeypatch):
    from nemo_gym.orchestration.submit import recover

    job_dir = _run(tmp_path)
    files = {path: path.read_bytes() for path in [job_dir / "job.sh", job_dir.parent / "resolved-config.yaml"]}
    _sinfo(slurm_bin, "batch* 2:00:00\\n")

    def no_evidence_read(*args, **kwargs):
        raise AssertionError("the evidence was read")

    monkeypatch.setattr(SlurmExecutor, "recovery_evidence", no_evidence_read)
    out = recover(*_handle(job_dir), max_walltime="12:00:00", walltime_reason="slow but healthy policy")
    assert not out.ok and out.dispatched is None and out.decision.action == RecoveryAction.STOP
    assert out.walltime_plan["refused"] == (
        "partition batch MaxTime 2:00:00 is below the run's segment walltime 04:00:00, so not even one segment of "
        "the --max-walltime 12:00:00 raise fits"
    )
    assert (out.walltime_plan["partition"], out.walltime_plan["max_time"], out.walltime_plan["requested"]) == (
        "batch",
        "2:00:00",
        "12:00:00",
    )
    assert out.decision.reason == f"walltime raise refused: {out.walltime_plan['refused']}"
    assert out.operator_verdict["state"] == "needs_action" and out.gate_task is None
    assert out.walltime_plan["refused"] in out.operator_verdict["reason"]
    assert "reader" not in out.operator_verdict["action"]
    assert out.walltime_plan["segment"] is None
    assert {path: path.read_bytes() for path in files} == files and not (slurm_bin / "env").exists()


def test_a_run_without_segments_cannot_take_a_raise_above_max_time():
    plan = sm._plan_walltime(
        sm.SubmitConfig.model_validate(
            {
                "services": {},
                "compute": {"c": {"type": "slurm", "account": "a", "walltime": "04:00:00"}},
                "driver": {"benchmarks": {"b": {}}},
                "job": {"output_path": "/tmp/x"},
                "otel": {"enabled": False},
            }
        ),
        "c",
        "b",
        "12:00:00",
        {"partition": "batch", "max_time": "4:00:00", "max_time_s": 14400, "error": None},
    )
    assert plan["refused"].startswith("--max-walltime 12:00:00 exceeds partition batch MaxTime 4:00:00, and the run")


@pytest.mark.parametrize("ambiguous", [False, True])
def test_a_scheduler_refusal_returns_an_outcome_with_its_text(tmp_path, slurm_bin, ambiguous):
    from nemo_gym.orchestration.submit import recover

    job_dir = _run(tmp_path)
    run_dir = job_dir.parent
    files = {path: path.read_bytes() for path in [job_dir / "job.sh", run_dir / "resolved-config.yaml"]}
    text = (
        "sbatch: error: Batch job submission failed: Socket timed out on send/recv operation" if ambiguous else REFUSAL
    )
    (slurm_bin / "sbatch").write_text(f"#!/bin/bash\necho '{text}' >&2; exit 1\n")
    out = recover(*_handle(job_dir), max_walltime="12:00:00", walltime_reason="slow but healthy policy")
    assert not out.ok and out.dispatched is None
    refusal = out.scheduler_refusal
    assert refusal["scheduler_text"] == text and refusal["benchmark"] == "gsm8k"
    assert refusal["restored"] == ["job.sh", "resolved-config.yaml", "resubmit lock released"]
    assert {path: path.read_bytes() for path in files} == files
    assert not (run_dir / WALLTIME_JOURNAL_NAME).exists() and not (job_dir / sm._RESUBMIT_LOCK).exists()
    assert f"the scheduler refused the job: {text}" in out.decision.reason
    json.loads(out.model_dump_json())
    verdict = out.operator_verdict
    assert out.gate_task["operator_verdict"] == verdict and verdict["state"] == "needs_action"
    assert "recover` now" not in verdict["action"]
    if ambiguous:
        assert "whether a job was queued is unknown" in verdict["reason"] and "reconcile" in verdict["action"]
    else:
        assert text in verdict["reason"] and "correct" in verdict["action"]
    if ambiguous:
        # A lost reply may still have queued a job: the chain records that its accounting is unknown, as before.
        assert refusal["job_dir_restored"] is False and refusal["chain_marker"].startswith(
            sm.ACCOUNTING_UNKNOWN_PREFIX
        )
    else:
        assert refusal["job_dir_restored"] is True and refusal["chain_marker"] is None
        assert "nothing was queued and the job dir was restored" in out.decision.reason
        assert not (job_dir / ".gym_chain_incomplete").exists()


@pytest.mark.parametrize(
    "text,seconds",
    [
        ("UNLIMITED", None),
        ("infinite", None),
        ("4:00:00", 14400),
        ("1-00:00:00", 86400),
        ("2-12", 216000),
        ("1-02:30", 95400),
        ("30", 1800),
    ],
)
def test_partition_max_time_parses_slurm_durations(text, seconds):
    assert sm._slurm_time_limit_seconds(text) == seconds


def test_partition_max_time_rejects_an_unknown_limit_and_reads_the_pool_partition():
    with pytest.raises(ValueError):
        sm._slurm_time_limit_seconds("n/a")

    class Conn:
        def __init__(self, answer):
            self.answer, self.commands = answer, []

        def run(self, commands):
            self.commands += commands
            return self.answer

    compute = sm.SlurmComputeConfig.model_validate(
        {"type": "slurm", "account": "a", "node_pools": {"gpu": {"partition": "batch"}}}
    )
    conn = Conn("batch* 4:00:00\n")
    assert sm._partition_time_limit(conn, compute) == {
        "partition": "batch",
        "max_time": "4:00:00",
        "max_time_s": 14400,
        "error": None,
    }
    assert conn.commands == ["sinfo -h -p batch -o '%P %l'"]
    assert sm._partition_time_limit(Conn("batch UNLIMITED\n"), compute)["max_time_s"] is None
    assert sm._partition_time_limit(Conn(""), compute)["error"] == "sinfo lists no partition batch"


def test_node_pools_on_several_partitions_take_the_lowest_max_time():
    class Conn:
        def __init__(self, answer):
            self.answer, self.commands = answer, []

        def run(self, commands):
            self.commands += commands
            return self.answer

    compute = sm.SlurmComputeConfig.model_validate(
        {
            "type": "slurm",
            "account": "a",
            "extra_args": {"partition": "ignored"},
            "node_pools": {"gpu": {"partition": "long"}, "cpu": {"partition": "batch"}, "io": {"partition": "open"}},
        }
    )
    conn = Conn("long 2-00:00:00\nbatch 4:00:00\nopen UNLIMITED\n")
    assert sm._partition_time_limit(conn, compute) == {
        "partition": "batch",
        "max_time": "4:00:00",
        "max_time_s": 14400,
        "error": None,
    }
    assert conn.commands == ["sinfo -h -p batch,long,open -o '%P %l'"]
    # Every pool's partition must be listed; one sinfo does not know leaves the limit unknown.
    assert sm._partition_time_limit(Conn("long 2-00:00:00\nbatch 4:00:00\n"), compute)["error"] == (
        "sinfo lists no partition open"
    )
