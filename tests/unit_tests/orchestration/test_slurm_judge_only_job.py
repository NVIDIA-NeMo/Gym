# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""`gym eval resubmit --recover` on a run that served its policy on GPUs and only needs its saved answers graded again
queues a judge-only job on a CPU partition instead of refusing; the job settles the rows through the run's own settle
path and ends with a receipt that `--inspect` reports. The scheduler is faked at the process level; everything else is
the real executor, collector and settle path."""

import json
import os
import re
import shlex
import sys
from pathlib import Path
from unittest import mock

import pytest
from hydra.core.override_parser.overrides_parser import OverridesParser

from nemo_gym.cli import eval as eval_cli
from nemo_gym.cli import main as cli_main
from nemo_gym.eval.rollout_reverification import RolloutReverificationConfig
from nemo_gym.orchestration.api import SubmitConfig
from nemo_gym.orchestration.executors import slurm as sm
from nemo_gym.orchestration.executors.connection import LocalConnection
from nemo_gym.orchestration.executors.slurm import READER_SRUN_ENV, SlurmExecutor
from nemo_gym.orchestration.executors.slurm_script import build_sbatch_script
from nemo_gym.orchestration.jobs import MANIFEST_NAME, BenchmarkJob, SubmissionRecord
from nemo_gym.orchestration.recovery import JUDGE_JOB_JOURNAL_NAME, RecoveryAction
from nemo_gym.orchestration.submit import inspect_recovery, recover
from nemo_gym.rollouts.rollout_collection import RolloutAggregationHelper
from nemo_gym.rollouts.rollout_recovery import RunManifest
from nemo_gym.rollouts.rollout_store import RolloutStore
from tests.unit_tests.eval.test_reverify_saved_results import verifier as verifier  # noqa: F401 - fixture
from tests.unit_tests.eval.test_reverify_saved_results import verify_transport_masked
from tests.unit_tests.orchestration.test_eval_recovery_gate import scored
from tests.unit_tests.orchestration.test_eval_resubmit import _handle
from tests.unit_tests.orchestration.test_eval_resubmit import (
    _local_index as _local_index,  # noqa: F401 - autouse fixture
)


def _model_served_run(
    tmp_path: Path, *, cpu_pool: bool = False, judge_pool: str | None = None, run: dict | None = None
) -> Path:
    """A submitted run with a vLLM policy on a GPU pool, whose managed Store holds one scored row and one result
    masked by a verify call lost in transport, with its chain accounting complete. With ``judge_pool`` it also serves
    its judge as a vLLM service ``wildguard`` on that pool: "judge", a GPU pool of its own, or "gpu", the policy's."""
    pools = {"gpu": {"partition": "gpu", "nodes": 1, "gpus_per_node": 8}}
    if cpu_pool:
        pools["scorer"] = {"partition": "cpu_long", "nodes": 1, "gpus_per_node": 0}
    services = {"policy": {"type": "vllm", "container": "vllm:latest", "model": "org/model", "node_pool": "gpu"}}
    if judge_pool is not None:
        if judge_pool != "gpu":
            pools[judge_pool] = {"partition": "gpu", "nodes": 1, "gpus_per_node": 8}
        services["wildguard"] = {
            "type": "vllm",
            "container": "vllm:latest",
            "model": "org/wildguard",
            "port": 8001,
            "node_pool": judge_pool,
        }
    config = SubmitConfig.model_validate(
        {
            "services": services,
            "compute": {
                "cluster": {
                    "type": "slurm",
                    "account": "a",
                    "hostname": None,
                    "walltime": "04:00:00",
                    "node_pools": pools,
                    **({"extra_args": {"qos": "gpu_normal"}} if judge_pool is not None else {}),
                }
            },
            "driver": {
                "container": "python:3.12",
                "policy_model": "policy",
                "gym_install": {"ref": "a" * 40},
                "benchmarks": {
                    "bench": {
                        "run": {"config_paths": ["fixture.yaml"], **(run or {})},
                        "resumable": {"max_retries": 3, "max_walltime": "16:00:00"},
                    }
                },
            },
            "job": {"output_path": str(tmp_path)},
            "otel": {"enabled": False},
        }
    )
    run_dir = tmp_path / "gym-job-x"
    job_dir = run_dir / "bench"
    artifacts = job_dir / "artifacts"
    artifacts.mkdir(parents=True)
    record = SubmissionRecord(
        gym_job_id=run_dir.name,
        gym_version="0",
        submitted_at="2026-01-01T00:00:00Z",
        run_dir=str(run_dir),
        cluster="cluster",
        executor="slurm",
        submitted_by="me",
        benchmarks=[BenchmarkJob(benchmark="bench", job_dir=str(job_dir), job_id="100")],
    )
    SlurmExecutor().persist(record, config, LocalConnection().write_text)
    (job_dir / "job.sh").write_text("#!/bin/bash\nGYM_CMD=(\n    gym eval run\n)\n")
    rows = [{"_ng_task_index": i, "_ng_rollout_index": 0, "agent_ref": {"name": "agent"}} for i in range(2)]
    source = job_dir / "source.jsonl"
    source.write_text("".join(json.dumps(row) + "\n" for row in rows))

    def prepare():
        inputs = json.loads(json.dumps(rows))
        manifest = RunManifest.create(source, inputs, {}, {"agent": {"responses_api_agents": {}}})
        return inputs, manifest.model_copy(update={"max_rollout_attempts": 3})

    output = artifacts / "rollouts.jsonl"
    with RolloutStore.start_or_resume(output, prepare, resume=False) as store:
        for row, outcome in zip(store.pending(), [verify_transport_masked(0), scored(1)]):
            store.record_dispatch(row)
            store.record_outcome(row | {k: v for k, v in outcome.items() if not k.startswith("_ng_task")})
    (artifacts / "rollouts_resolved_config.yaml").write_text("fixture: true\n")
    (job_dir / ".gym_job_chain").write_text("100\n")
    (job_dir / ".gym_infra_retries").write_text("0\n")
    (job_dir / ".gym_accumulated_walltime").write_text("7200\n")
    return job_dir


@pytest.fixture
def judge_slurm_bin(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Process-level scheduler fakes: squeue answers a job query from `squeue_jobs` and a workdir query from
    `squeue_workdirs`; sbatch records its arguments and numbers each job from 4242; sacct answers the judge-job
    query (`JobIDRaw,...,ExitCode`) from `judge_accounting` and the chain query from `accounting`; sinfo lists the
    partitions' MaxTime."""
    bins = tmp_path / "bin"
    bins.mkdir()
    (bins / "squeue").write_text(
        '#!/bin/bash\nhere="$(dirname "$0")"\n'
        'if [[ " $* " == *" -j "* ]]; then cat "$here/squeue_jobs" 2>/dev/null; else cat "$here/squeue_workdirs" 2>/dev/null; fi\n'
        "exit 0\n"
    )
    (bins / "sbatch").write_text(
        '#!/bin/bash\nhere="$(dirname "$0")"\n'
        'n=$(cat "$here/sbatch_count" 2>/dev/null || echo 0); n=$((n + 1)); echo "$n" > "$here/sbatch_count"\n'
        'echo "$PWD" > "$here/cwd"\nenv > "$here/env"\numask > "$here/umask"\nprintf "%s\\n" "$@" >> "$here/sbatch_args"\n'
        "echo $((4241 + n))\n"
    )
    (bins / "sacct").write_text(
        '#!/bin/bash\nhere="$(dirname "$0")"\n'
        'if [[ "$*" == *"JobIDRaw,State,ElapsedRaw,ExitCode"* ]]; then cat "$here/judge_accounting" 2>/dev/null; '
        'else cat "$here/accounting"; fi\n'
    )
    (bins / "accounting").write_text("100|COMPLETED|7200|0\n")
    (bins / "sinfo").write_text('#!/bin/bash\nprintf "cpu 8:00:00\\ncpu_long 2-00:00:00\\ngpu 4:00:00\\n"\n')
    for stub in ("squeue", "sbatch", "sacct", "sinfo"):
        (bins / stub).chmod(0o755)
    monkeypatch.setenv("PATH", f"{bins}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.delenv(READER_SRUN_ENV, raising=False)
    monkeypatch.setattr(sm, "installed_gym_commit", lambda: "a" * 40)
    return bins


def _journal(job_dir: Path) -> list[dict]:
    path = job_dir / JUDGE_JOB_JOURNAL_NAME
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def _accounting(job_dir: Path) -> tuple[str, str, str]:
    return tuple(
        (job_dir / name).read_text() for name in (".gym_job_chain", ".gym_infra_retries", ".gym_accumulated_walltime")
    )


def _queued_script(job_dir: Path) -> str:
    (script,) = job_dir.glob("reverify-*.sh")
    return script.read_text()


def _rendered_gym_commands(script: str) -> list[list[str]]:
    """Every `gym ...` command a rendered script runs, as its words: one-line commands and the `GYM_CMD=( ... )`
    array of a collection job."""
    commands = [
        shlex.split(line.strip().replace("'\"'\"'", "'"))
        for line in script.splitlines()
        if line.strip().startswith("gym ")
    ]
    for body in re.findall(r"GYM_CMD=\(\n(.*?)\n\)", script, re.S):
        commands.append(shlex.split(" ".join(line.strip() for line in body.splitlines() if line.strip())))
    return commands


def _unrecognized_flags(words: list[str]) -> list[str]:
    """The flags of a rendered `gym ...` command the real CLI does not define: what `main()` rejects with
    "unrecognized arguments" (exit 2) before any work."""
    _, rest = cli_main.build_parser().parse_known_args(words[1:])
    return [token for token in rest if token.startswith("-")]


def _queued_reverify_config(script: str) -> RolloutReverificationConfig:
    """The configuration the queued script's `gym eval reverify` line hands the job, parsed by the real `gym` CLI.

    A flag the launcher renders but the CLI does not define fails here as it fails in the job (exit 2).
    """
    (line,) = [line.strip() for line in script.splitlines() if line.strip().startswith("gym eval reverify")]
    words = shlex.split(line.replace("'\"'\"'", "'"))
    dispatched = []
    with (
        mock.patch.object(sys, "argv", words),
        mock.patch.object(cli_main, "dispatch", lambda target, overrides: dispatched.append((target, overrides))),
    ):
        cli_main.main()
    ((target, overrides),) = dispatched
    assert target == "nemo_gym.cli.eval:reverify_rollouts"
    values: dict = {}
    for override in OverridesParser.create().parse_overrides(overrides):
        values[override.key_or_group] = override.value()
    values.pop("config_paths", None)
    values.pop("model_type", None)
    return RolloutReverificationConfig.model_validate(values | {"disable_aggregation": True})


def test_recover_queues_a_judge_only_job_on_the_cpu_partition_for_a_model_served_run(
    tmp_path, judge_slurm_bin
) -> None:
    job_dir = _model_served_run(tmp_path)
    before = _accounting(job_dir)

    out = recover(*_handle(job_dir), judge_partition="cpu")

    assert out.ok and out.dispatched == "reverify" and out.job_id == "4242", out.decision.reason
    assert out.decision.action == RecoveryAction.REVERIFY and out.decision.counts == {
        "judge_recoverable": 1,
        "scored": 1,
    }
    job = out.judge_job
    assert job["job_id"] == "4242" and job["partition"] == "cpu" and job["partition_source"] == "--judge-partition"
    assert job["walltime"] == "04:00:00" and job["rows"] == 1 and job["gym_commit"] == "a" * 40
    assert job["receipt"].endswith("receipt.json") and "/reverification/" in job["receipt"]
    script = _queued_script(job_dir)
    assert "#SBATCH --partition=cpu" in script and "#SBATCH --nodes=1" in script
    assert "#SBATCH --gpus-per-node=0" in script and "#SBATCH --time=04:00:00" in script
    assert "#SBATCH --no-requeue" in script and "vllm serve" not in script and "gym eval run" not in script
    assert "afternotok" not in script
    assert "--judge-failed-only" in script and "--settle-into-source" in script and "--settle-receipt" in script
    # A pass that died left admissions without an outcome: the job withdraws them instead of refusing.
    queued = _queued_reverify_config(script)
    assert queued.withdraw_unresolved_admissions is True and queued.skip_spent_judge_failures is True
    assert "--model-type openai_model" in script
    # Observability is on, so the job names where its model calls are captured: its own folder.
    (line,) = [line for line in script.splitlines() if line.strip().startswith("gym eval reverify")]
    assert "+observability_enabled=True" in line
    assert f"+model_call_capture_dir={Path(job['receipt']).parent / 'model-calls'}" in line
    # The judge-only job is not a segment of the policy chain: no chain entry, no walltime charged, no retry spent.
    assert _accounting(job_dir) == before
    (entry,) = _journal(job_dir)
    assert entry["job_id"] == "4242" and entry["partition"] == "cpu" and entry["receipt"] == job["receipt"]
    assert out.gate_task["status"] == "RESUMING" and out.gate_task["judge_jobs"][-1]["job_id"] == "4242"
    record = SubmissionRecord.load(json.loads((job_dir.parent / MANIFEST_NAME).read_text()))
    assert record.benchmark_job("bench").job_id == "4242"
    assert "--no-requeue" in (judge_slurm_bin / "sbatch_args").read_text()
    assert not (job_dir / sm._RESUBMIT_LOCK).exists()


def test_a_saved_cpu_pool_queues_the_judge_only_job_without_an_option(tmp_path, judge_slurm_bin) -> None:
    job_dir = _model_served_run(tmp_path, cpu_pool=True)
    out = recover(*_handle(job_dir))
    assert out.ok and out.dispatched == "reverify", out.decision.reason
    assert (
        out.judge_job["partition"] == "cpu_long"
        and out.judge_job["partition_source"] == "the saved node pool 'scorer'"
    )
    assert "#SBATCH --partition=cpu_long" in _queued_script(job_dir)


def test_without_a_cpu_partition_the_refusal_names_the_option_and_queues_nothing(tmp_path, judge_slurm_bin) -> None:
    job_dir = _model_served_run(tmp_path)
    out = recover(*_handle(job_dir))
    assert not out.ok and out.dispatched is None and out.decision.action == RecoveryAction.STOP
    assert "CPU-only judge execution profile" in out.unsupported and "--judge-partition" in out.unsupported
    assert not (judge_slurm_bin / "sbatch_args").exists() and _journal(job_dir) == []
    assert out.gate_task["status"] == "FAILED"


def test_recover_queues_a_served_judge_job_on_the_runs_judge_pool(tmp_path, judge_slurm_bin) -> None:
    # The launcher's CPU placement comes with every recovery; a run that serves its judge does not use it.
    job_dir = _model_served_run(tmp_path, judge_pool="judge")
    before = _accounting(job_dir)

    out = recover(*_handle(job_dir), judge_partition="cpu", judge_qos="cpu_qos")

    assert out.ok and out.dispatched == "reverify" and out.job_id == "4242", out.decision.reason
    job = out.judge_job
    assert job["kind"] == "served-judge" and job["services"] == ["wildguard"] and job["pools"] == ["judge"]
    assert job["partition"] == "gpu" and job["qos"] == "gpu_normal" and job["rows"] == 1
    assert job["unused"] == "--judge-partition/--judge-qos apply to the CPU judge profile only; not used"
    script = _queued_script(job_dir)
    assert "#SBATCH --partition=gpu" in script and "#SBATCH --nodes=1" in script
    assert "#SBATCH --gpus-per-node=8" in script and "#SBATCH --qos=gpu_normal" in script
    assert "--partition=cpu" not in script and "cpu_qos" not in script and "#SBATCH --no-requeue" in script
    # The judge pool's hosts are exported and the judge service is up before the judge-only verifier command.
    (reverify,) = [line for line in script.splitlines() if line.strip().startswith("gym eval reverify")]
    assert script.index("export GYM_POOL_JUDGE_NODES=") < script.index("org/wildguard") < script.index(reverify)
    assert "vllm serve org/model" not in script and "GYM_POOL_GPU_NODES" not in script and "gym eval run" not in script
    assert _accounting(job_dir) == before
    (entry,) = _journal(job_dir)
    assert entry["kind"] == "served-judge" and entry["services"] == ["wildguard"]

    (judge_slurm_bin / "squeue_workdirs").write_text(f"{job_dir}\n")
    (judge_slurm_bin / "judge_accounting").write_text("4242|PENDING|0|0:0\n")
    inspected = inspect_recovery(*_handle(job_dir))
    (listed,) = inspected.gate_task["judge_jobs"]
    assert listed["kind"] == "served-judge" and listed["services"] == ["wildguard"]
    assert any("judge-only job 4242 on gpu serving wildguard" in w for w in inspected.gate_task["warnings"])


def test_a_judge_service_on_the_policys_pool_is_refused_and_nothing_is_queued(tmp_path, judge_slurm_bin) -> None:
    job_dir = _model_served_run(tmp_path, judge_pool="gpu")
    out = recover(*_handle(job_dir), judge_partition="cpu")
    assert not out.ok and out.dispatched is None and out.decision.action == RecoveryAction.STOP
    assert "wildguard" in out.unsupported and "a separate node pool" in out.unsupported
    assert not (judge_slurm_bin / "sbatch_args").exists() and _journal(job_dir) == []


def test_a_live_judge_only_job_is_reported_and_never_queued_beside(tmp_path, judge_slurm_bin) -> None:
    job_dir = _model_served_run(tmp_path)
    assert recover(*_handle(job_dir), judge_partition="cpu").dispatched == "reverify"
    (judge_slurm_bin / "squeue_workdirs").write_text(f"{job_dir}\n")
    (judge_slurm_bin / "judge_accounting").write_text("4242|RUNNING|30|0:0\n")

    inspected = inspect_recovery(*_handle(job_dir))
    assert inspected.decision.action == RecoveryAction.NOOP and inspected.decision.scheduler_live
    assert "judge-only job 4242 (RUNNING)" in inspected.decision.reason
    assert inspected.gate_task["status"] == "RUNNING"
    (entry,) = inspected.gate_task["judge_jobs"]
    assert entry["job_id"] == "4242" and entry["state"] == "running"

    again = recover(*_handle(job_dir), judge_partition="cpu")
    assert again.dispatched is None and (judge_slurm_bin / "sbatch_count").read_text().strip() == "1"


def test_a_judge_only_job_that_ended_without_a_receipt_is_reported_and_a_second_failure_stops(
    tmp_path, judge_slurm_bin
) -> None:
    job_dir = _model_served_run(tmp_path)
    assert recover(*_handle(job_dir), judge_partition="cpu").job_id == "4242"
    (judge_slurm_bin / "judge_accounting").write_text("4242|FAILED|30|1:0\n")

    inspected = inspect_recovery(*_handle(job_dir))
    assert inspected.decision.action == RecoveryAction.REVERIFY
    (entry,) = inspected.gate_task["judge_jobs"]
    assert entry["job_id"] == "4242" and entry["state"] == "failed" and "FAILED" in entry["scheduler_state"]
    assert any(
        "judge-only job 4242" in warning and "no receipt" in warning for warning in inspected.gate_task["warnings"]
    )

    second = recover(*_handle(job_dir), judge_partition="cpu")
    assert second.dispatched == "reverify" and second.job_id == "4243"
    (judge_slurm_bin / "judge_accounting").write_text("4242|FAILED|30|1:0\n4243|FAILED|30|1:0\n")

    third = recover(*_handle(job_dir), judge_partition="cpu")
    assert not third.ok and third.dispatched is None and third.decision.action == RecoveryAction.STOP
    assert (
        "4242" in third.unsupported and "4243" in third.unsupported and "ended without settling" in third.unsupported
    )
    assert (judge_slurm_bin / "sbatch_count").read_text().strip() == "2"
    assert [entry["job_id"] for entry in _journal(job_dir)] == ["4242", "4243"]


def test_the_queued_job_settles_the_run_through_its_own_settle_path_and_inspect_reports_the_receipt(
    tmp_path,
    judge_slurm_bin,
    monkeypatch,
    verifier,  # noqa: F811
) -> None:
    job_dir = _model_served_run(tmp_path)
    out = recover(*_handle(job_dir), judge_partition="cpu")
    assert out.dispatched == "reverify"
    aggregations = []

    async def run_from_config(self, config):
        aggregations.append(config)

    monkeypatch.setattr(RolloutAggregationHelper, "run_from_config", run_from_config)
    monkeypatch.setenv("SLURM_JOB_ID", "4242")

    # The job: the queued script's own reverify command, run against the run's artifacts.
    config = _queued_reverify_config(_queued_script(job_dir))
    assert config.settle_receipt_fpath == out.judge_job["receipt"]
    eval_cli._reverify_with_receipt({"config_paths": ["fixture.yaml"], "health_check_workers": 1}, config)

    assert verifier["posted"] == [("rs", "/verify")]  # one verify call of the saved answer; the policy is never called
    (judge_slurm_bin / "judge_accounting").write_text("4242|COMPLETED|30|0:0\n")
    inspected = inspect_recovery(*_handle(job_dir))
    assert inspected.decision.action == RecoveryAction.NOOP and inspected.decision.complete, inspected.decision.reason
    (entry,) = inspected.gate_task["judge_jobs"]
    assert entry["job_id"] == "4242" and entry["state"] == "completed" and entry["settled"] == 1
    assert entry["selected"] == 1 and entry["withdrawn"] == 0 and entry["unsupported"] == 0
    assert any(
        "judge-only job 4242" in warning and "settled 1" in warning for warning in inspected.gate_task["warnings"]
    )
    coverage = json.loads((job_dir / "artifacts" / "rollouts_coverage.json").read_text())
    assert coverage["masked"] == 0 and coverage["allocation_id"] == "4242"
    assert len(aggregations) == 1 and aggregations[0].merge_shards is False
    assert (job_dir / "artifacts" / "quality_summary.json").exists()
    # The chain accounting is still the policy chain's.
    assert (job_dir / ".gym_job_chain").read_text() == "100\n"


def _modes(root: Path) -> dict[str, int]:
    """The permission bits of everything under (and including) ``root``, keyed by its path relative to ``root``."""
    return {str(path.relative_to(root)): path.stat().st_mode & 0o777 for path in [root, *root.rglob("*")]}


def test_the_judge_only_jobs_outputs_and_journals_are_readable_like_the_runs_other_artifacts(
    tmp_path,
    judge_slurm_bin,
    monkeypatch,
    verifier,  # noqa: F811
) -> None:
    """Only the staged driver script is owner-only. The job inherits the dispatch's umask from sbatch, so a dispatch that
    left ``umask 077`` set made the reverification folder 0700 and its files 0600: a second user reading the run to
    certify it stopped with "Permission denied"."""
    previous = os.umask(0o022)  # the usual login umask, which the run's other artifacts are created under
    try:
        job_dir = _model_served_run(tmp_path)
        out = recover(*_handle(job_dir), judge_partition="cpu")
        assert out.dispatched == "reverify", out.decision.reason

        # What a reader of the run needs: the journal inspect reads is readable like the rest of the run directory.
        assert (job_dir / JUDGE_JOB_JOURNAL_NAME).stat().st_mode & 0o777 == 0o644
        # The driver script is the one owner-only file the dispatch stages.
        (script,) = job_dir.glob("reverify-*.sh")
        assert script.stat().st_mode & 0o777 == 0o600
        # The job inherits the dispatch's umask from sbatch: the caller's, not the private one of the script.
        inherited = int((judge_slurm_bin / "umask").read_text(), 8)
        assert inherited == 0o022

        # The job, under the umask it inherited, creates the reverification folder and its receipt.
        async def run_from_config(self, config):
            pass

        monkeypatch.setattr(RolloutAggregationHelper, "run_from_config", run_from_config)
        monkeypatch.setenv("SLURM_JOB_ID", "4242")
        os.umask(inherited)
        config = _queued_reverify_config(_queued_script(job_dir))
        eval_cli._reverify_with_receipt({"config_paths": ["fixture.yaml"], "health_check_workers": 1}, config)
    finally:
        os.umask(previous)

    receipt = Path(out.judge_job["receipt"])
    assert receipt.exists()
    modes = _modes(job_dir / "reverification")
    assert len(modes) > 2, modes
    for name, mode in modes.items():
        assert mode & 0o044 == 0o044 and mode & 0o200, (name, oct(mode))
    assert modes["."] == 0o755 and modes[str(receipt.relative_to(job_dir / "reverification"))] == 0o644


def test_the_judge_only_job_does_not_inherit_the_collection_limit(tmp_path, judge_slurm_bin) -> None:
    # A collection `limit` counts tasks; in the job's reverify it would cap the rows it re-judges.
    job_dir = _model_served_run(tmp_path, run={"limit": 1})
    out = recover(*_handle(job_dir), judge_partition="cpu")

    assert out.dispatched == "reverify", out.decision.reason
    assert _queued_reverify_config(_queued_script(job_dir)).limit is None


@pytest.mark.parametrize("render", ["judge-only job", "judge-only job on a saved CPU pool", "the run's own job"])
def test_every_gym_command_a_rendered_script_runs_parses_with_the_real_cli(
    tmp_path, judge_slurm_bin, render: str
) -> None:
    """A flag the launcher renders but the CLI does not define stops the Slurm job at argument parsing (exit 2):
    every `gym ...` command of every script the launcher renders here must parse with the real CLI."""
    submitted = []
    persist = SlurmExecutor.persist
    with mock.patch.object(
        SlurmExecutor,
        "persist",
        lambda self, record, config, write: submitted.append(config) or persist(self, record, config, write),
    ):
        job_dir = _model_served_run(tmp_path, cpu_pool=render.endswith("CPU pool"))
    if render == "the run's own job":
        (config,) = submitted
        bench = config.driver.benchmarks["bench"]
        script = build_sbatch_script(config, "bench", bench, config.compute["cluster"], job_dir)
    else:
        out = recover(*_handle(job_dir), **({} if render.endswith("CPU pool") else {"judge_partition": "cpu"}))
        assert out.dispatched == "reverify", out.decision.reason
        script = _queued_script(job_dir)
    commands = _rendered_gym_commands(script)
    assert commands
    for words in commands:
        assert _unrecognized_flags(words) == [], words


def test_an_old_pin_model_served_run_is_rejudged_headless_on_the_installed_revision(
    tmp_path, judge_slurm_bin, monkeypatch
) -> None:
    """No flags: the plan is made on the installed Gym and its judge-only job runs there, journaled as a repin."""
    monkeypatch.setenv("NEMO_GYM_SUBMIT_NO_REF_CHECK", "true")
    monkeypatch.setattr(sm, "installed_gym_commit", lambda: "b" * 40)
    job_dir = _model_served_run(tmp_path)

    out = recover(*_handle(job_dir), judge_partition="cpu")

    assert out.ok and out.dispatched == "reverify", out.decision.reason
    assert out.pin_plan["repin"] is True and out.pin_plan["pinned"] == "a" * 40
    assert out.pin_plan["reason"] == (
        f"automatic repin: planned on {'b' * 40} for a run pinned at {'a' * 40}: "
        "its judge-only job runs the installed recovery source"
    )
    assert out.judge_job["gym_commit"] == "b" * 40
    assert "b" * 40 in _queued_script(job_dir)


def test_the_judge_only_job_skips_the_policy_endpoint_readiness_wait(tmp_path, judge_slurm_bin) -> None:
    # It only judges saved answers and never calls the policy model, whose endpoint left with the run's allocation:
    # waiting for its readiness would time the job out before any judging.
    job_dir = _model_served_run(tmp_path)

    out = recover(*_handle(job_dir), judge_partition="cpu")

    assert out.ok and out.dispatched == "reverify", out.decision.reason
    (line,) = [line for line in _queued_script(job_dir).splitlines() if line.strip().startswith("gym eval reverify")]
    assert line.count("model_endpoint_readiness_timeout_seconds") == 1
    assert line.rstrip().endswith("++model_endpoint_readiness_timeout_seconds=0") or (
        "++model_endpoint_readiness_timeout_seconds=0 " in line
    )


def test_without_a_cpu_partition_the_verdict_names_the_judge_partition_not_a_fresh_submit(
    tmp_path, judge_slurm_bin
) -> None:
    job_dir = _model_served_run(tmp_path)

    refused = recover(*_handle(job_dir))

    assert refused.dispatched is None and refused.decision.action == RecoveryAction.STOP
    verdict = refused.operator_verdict
    assert verdict["state"] == "needs_action" and "`gym eval submit`" not in verdict["action"]
    assert "`gym eval resubmit --recover --judge-partition PARTITION`" in verdict["action"]
    # The option the verdict names queues the judge-only job.
    assert recover(*_handle(job_dir), judge_partition="cpu").job_id == "4242"


def test_a_judge_only_job_the_scheduler_refuses_names_the_judge_qos_not_a_fresh_submit(
    tmp_path, judge_slurm_bin
) -> None:
    job_dir = _model_served_run(tmp_path)
    (judge_slurm_bin / "sbatch").write_text(
        "#!/bin/bash\necho 'sbatch: error: Job violates accounting/QOS policy (QOSMinGRES)' >&2\nexit 1\n"
    )

    out = recover(*_handle(job_dir), judge_partition="cpu")

    assert out.dispatched is None and out.decision.action == RecoveryAction.STOP
    assert "QOSMinGRES" in out.unsupported and _journal(job_dir) == []
    verdict = out.operator_verdict
    assert verdict["state"] == "needs_action" and "`gym eval submit`" not in verdict["action"]
    assert "--judge-qos QOS" in verdict["action"] and "--judge-partition PARTITION" in verdict["action"]
