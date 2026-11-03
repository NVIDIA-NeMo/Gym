# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The saved answers of a run that served its policy on GPUs are re-judged by a judge-only job on a CPU-only execution
profile derived from the run's saved submission config: the run's own judge and verifier servers, no policy service,
one node of a CPU partition, a bounded walltime."""

from pathlib import Path

import pytest

from nemo_gym.orchestration.api import SubmitConfig
from nemo_gym.orchestration.executors.judge_profile import (
    DEFAULT_JUDGE_WALLTIME,
    JUDGE_POOL_NAME,
    JudgeProfileUnsupported,
    choose_judge_partition,
    is_cpu_only_submission,
    judge_execution_profile,
    judge_job_config,
    reader_step_partition,
)
from nemo_gym.orchestration.executors.slurm_script import build_sbatch_script


def model_served(
    tmp_path: Path,
    *,
    pools: dict | None = None,
    services: dict | None = None,
    extra_args: dict | None = None,
    walltime: str | None = "04:00:00",
) -> SubmitConfig:
    """A submission that serves its policy with vLLM on a GPU pool and judges with servers of the benchmark config."""
    compute = {
        "type": "slurm",
        "account": "a",
        "hostname": None,
        "node_pools": pools if pools is not None else {"gpu": {"partition": "gpu", "nodes": 1, "gpus_per_node": 8}},
        "extra_args": extra_args or {},
    }
    if walltime is not None:
        compute["walltime"] = walltime
    return SubmitConfig.model_validate(
        {
            "services": services
            if services is not None
            else {"policy": {"type": "vllm", "container": "vllm:latest", "model": "org/model", "node_pool": "gpu"}},
            "compute": {"cluster": compute},
            "driver": {
                "container": "python:3.12",
                "policy_model": "policy",
                "gym_install": {"ref": "a" * 40},
                "env": {"JUDGE_API_KEY": "lit:secret"},
                "benchmarks": {
                    "bench": {
                        "run": {"config_paths": ["fixture.yaml"]},
                        "resumable": {"max_retries": 3, "max_walltime": "16:00:00"},
                    }
                },
            },
            "job": {"output_path": str(tmp_path)},
            "otel": {"enabled": False},
        }
    )


def cpu_only(tmp_path: Path) -> SubmitConfig:
    return SubmitConfig.model_validate(
        {
            "services": {},
            "compute": {
                "cluster": {
                    "type": "slurm",
                    "account": "a",
                    "walltime": "00:05:00",
                    "node_pools": {"cpu": {"partition": "cpu", "gpus_per_node": 0}},
                }
            },
            "driver": {"gym_install": {"ref": "a" * 40}, "benchmarks": {"bench": {"run": {}}}},
            "job": {"output_path": str(tmp_path)},
            "otel": {"enabled": False},
        }
    )


# --- which submissions already have a CPU-only judge profile -----------------------------------------------------


def test_only_a_submission_without_services_or_gpus_is_cpu_only(tmp_path) -> None:
    assert is_cpu_only_submission(cpu_only(tmp_path))
    assert not is_cpu_only_submission(model_served(tmp_path))


# --- the CPU partition -------------------------------------------------------------------------------------------


def test_a_saved_cpu_node_pool_names_the_judge_partition(tmp_path) -> None:
    config = model_served(
        tmp_path,
        pools={
            "gpu": {"partition": "gpu", "nodes": 1, "gpus_per_node": 8},
            "scorer": {"partition": "cpu_long", "nodes": 1, "gpus_per_node": 0, "extra_args": {"qos": "cpu"}},
        },
    )
    partition, source = choose_judge_partition(config, "cluster", reader_srun="-p cpu_short")
    assert partition == "cpu_long" and source == "the saved node pool 'scorer'"


@pytest.mark.parametrize("args", ["-p cpu_short -c 4", "--partition cpu_short", "--partition=cpu_short -c 4"])
def test_the_reader_placement_names_the_partition_when_the_saved_config_has_no_cpu_pool(tmp_path, args) -> None:
    assert reader_step_partition(args) == "cpu_short"
    partition, source = choose_judge_partition(model_served(tmp_path), "cluster", reader_srun=args)
    assert partition == "cpu_short" and "NEMO_GYM_REMOTE_READER_SRUN" in source


def test_an_explicit_judge_partition_wins(tmp_path) -> None:
    partition, source = choose_judge_partition(
        model_served(tmp_path), "cluster", partition="ops", reader_srun="-p cpu_short"
    )
    assert partition == "ops" and source == "--judge-partition"


def test_no_cpu_partition_is_a_typed_refusal_naming_every_source(tmp_path) -> None:
    with pytest.raises(JudgeProfileUnsupported) as refused:
        choose_judge_partition(model_served(tmp_path), "cluster", reader_srun="-c 4")
    text = str(refused.value)
    assert "no CPU partition" in text
    assert "gpus_per_node" in text and "NEMO_GYM_REMOTE_READER_SRUN" in text and "--judge-partition" in text


def test_a_submission_that_serves_more_than_its_policy_is_refused(tmp_path) -> None:
    config = model_served(
        tmp_path,
        services={
            "policy": {"type": "vllm", "container": "vllm:latest", "model": "org/model", "node_pool": "gpu"},
            "judge_vllm": {"type": "vllm", "container": "vllm:latest", "model": "org/judge", "port": 8001},
        },
    )
    with pytest.raises(JudgeProfileUnsupported, match="judge_vllm"):
        choose_judge_partition(config, "cluster", partition="cpu")


# --- the bounded walltime ----------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "walltime,limit,expected,bounds",
    [
        ("04:00:00", {"max_time": "8:00:00", "max_time_s": 28800, "error": None}, "04:00:00", ("segment",)),
        ("04:00:00", {"max_time": "2:00:00", "max_time_s": 7200, "error": None}, "02:00:00", ("segment", "MaxTime")),
        ("04:00:00", {"max_time": None, "max_time_s": None, "error": "sinfo failed"}, "04:00:00", ("segment",)),
        (None, {"max_time": "1:00:00", "max_time_s": 3600, "error": None}, "01:00:00", ("MaxTime",)),
        (None, {"max_time": None, "max_time_s": None, "error": None}, DEFAULT_JUDGE_WALLTIME, ("default",)),
    ],
)
def test_the_walltime_is_the_segment_walltime_bounded_by_the_partition_max_time(
    tmp_path, walltime, limit, expected, bounds
) -> None:
    profile = judge_execution_profile(
        model_served(tmp_path, walltime=walltime), "cluster", "cpu", "--judge-partition", limit
    )
    assert profile.walltime == expected
    assert profile.partition == "cpu"
    for bound in bounds:
        assert any(bound in text for text in profile.walltime_bounds), (bound, profile.walltime_bounds)
    pool = profile.compute.node_pools[JUDGE_POOL_NAME]
    assert (pool.partition, pool.nodes, pool.ntasks_per_node, pool.gpus_per_node) == ("cpu", 1, 1, 0)
    assert profile.compute.walltime == expected
    report = profile.report()
    assert report["partition"] == "cpu" and report["walltime"] == expected and report["partition_source"]


# --- the judge job's submission config and script ----------------------------------------------------------------


def test_the_judge_job_runs_the_reverify_command_on_one_cpu_node_without_any_service(tmp_path) -> None:
    config = model_served(tmp_path, extra_args={"qos": "normal", "gres": "gpu:8", "comment": "campaign"})
    profile = judge_execution_profile(
        config, "cluster", "cpu", "--judge-partition", {"max_time": "8:00:00", "max_time_s": 28800, "error": None}
    )
    command = "gym eval reverify --judge-failed-only --settle-into-source --settle-receipt /r/receipt.json"
    job = judge_job_config(config, "cluster", "bench", profile, command)
    assert job.services == {} and job.driver.policy_model is None
    assert list(job.driver.benchmarks) == ["bench"]
    bench = job.driver.benchmarks["bench"]
    assert bench.command == command and bench.run == {} and bench.prepare == {} and bench.resume_config is None
    # The run's judge and verifier settings travel with the driver: its env, mounts, container and pinned install.
    assert job.driver.env == config.driver.env and job.driver.gym_install == config.driver.gym_install
    assert job.driver.container == config.driver.container
    compute = job.compute["cluster"]
    # GPU resources, node-shape directives and the QOS of the original allocation do not follow; the rest does.
    assert compute.extra_args == {"comment": "campaign"}
    script = build_sbatch_script(job, "bench", bench, compute, Path("/jobs/bench"))
    assert "#SBATCH --partition=cpu" in script and "#SBATCH --nodes=1" in script
    assert "#SBATCH --gpus-per-node=0" in script and "#SBATCH --time=04:00:00" in script
    assert "--qos" not in script and "--gres" not in script
    assert "vllm serve" not in script and "gym eval run" not in script and "afternotok" not in script
    assert command in script


def test_a_repinned_judge_job_runs_the_launcher_revision(tmp_path) -> None:
    config = model_served(tmp_path)
    profile = judge_execution_profile(config, "cluster", "cpu", "--judge-partition", {"max_time_s": None})
    job = judge_job_config(config, "cluster", "bench", profile, "gym eval reverify", gym_ref="b" * 40)
    assert job.driver.gym_install.ref == "b" * 40 and job.driver.gym_install.repo == config.driver.gym_install.repo


# --- the judge job's QOS -----------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "reader_srun,expected",
    [
        ("-p cpu -q cpu_normal -c 4", "cpu_normal"),
        ("--qos=cpu_normal", "cpu_normal"),
        ("-p other_cpu --qos cpu_normal", None),
        (None, None),
    ],
)
def test_the_runs_gpu_qos_does_not_follow_and_the_reader_placement_names_one(tmp_path, reader_srun, expected) -> None:
    # A GPU QOS that requires GPUs refuses a job on a CPU partition (QOSMinGRES), so the run's QOS stays behind.
    config = model_served(tmp_path, extra_args={"qos": "normal", "comment": "campaign"})
    profile = judge_execution_profile(
        config, "cluster", "cpu", "--judge-partition", {"max_time_s": None}, reader_srun=reader_srun
    )
    job = judge_job_config(config, "cluster", "bench", profile, "gym eval reverify")
    compute = job.compute["cluster"]
    script = build_sbatch_script(job, "bench", job.driver.benchmarks["bench"], compute, Path("/jobs/bench"))
    assert "--qos=normal" not in script
    assert profile.qos == expected and profile.report()["qos"] == expected
    if expected:
        assert f"#SBATCH --qos={expected}" in script and "NEMO_GYM_REMOTE_READER_SRUN" in profile.qos_source
    else:
        assert "--qos" not in script and profile.qos_source == "the partition's default"


def test_a_saved_cpu_pool_keeps_its_own_qos(tmp_path) -> None:
    config = model_served(
        tmp_path,
        extra_args={"qos": "normal"},
        pools={
            "gpu": {"partition": "gpu", "nodes": 1, "gpus_per_node": 8},
            "scorer": {"partition": "cpu_long", "nodes": 1, "gpus_per_node": 0, "extra_args": {"qos": "cpu"}},
        },
    )
    profile = judge_execution_profile(
        config, "cluster", "cpu_long", "the saved node pool 'scorer'", {"max_time_s": None}, reader_srun="-q other"
    )
    assert profile.qos == "cpu" and profile.qos_source == "the saved CPU node pool"
    assert profile.compute.node_pools[JUDGE_POOL_NAME].extra_args == {"qos": "cpu"}
    assert "qos" not in profile.compute.extra_args


def test_an_explicit_judge_qos_wins_over_the_saved_pool_and_the_reader_placement(tmp_path) -> None:
    config = model_served(
        tmp_path,
        extra_args={"qos": "normal"},
        pools={
            "gpu": {"partition": "gpu", "nodes": 1, "gpus_per_node": 8},
            "scorer": {"partition": "cpu_long", "nodes": 1, "gpus_per_node": 0, "extra_args": {"qos": "cpu"}},
        },
    )
    profile = judge_execution_profile(
        config, "cluster", "cpu_long", "--judge-partition", {"max_time_s": None}, qos="cpu_ops", reader_srun="-q other"
    )
    assert profile.qos == "cpu_ops" and profile.qos_source == "--judge-qos"
    job = judge_job_config(config, "cluster", "bench", profile, "gym eval reverify")
    compute = job.compute["cluster"]
    script = build_sbatch_script(job, "bench", job.driver.benchmarks["bench"], compute, Path("/jobs/bench"))
    assert script.count("--qos=") == 1 and "#SBATCH --qos=cpu_ops" in script
