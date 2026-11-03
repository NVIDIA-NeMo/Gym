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
"""A CPU-only execution profile for grading a run's saved answers again.

A run that served its policy on GPUs has no allocation a judge-only job could reuse: its node pools are GPU pools and
its services are the policy's. The profile derived here keeps what grading needs from the saved submission config,
the driver with its env, mounts, container and pinned install, so the run's own judge and verifier servers start as
they did, and replaces what grading does not need: no service, no policy model, one node of a CPU partition and a
bounded walltime. The partition comes from the saved config when it declares a CPU node pool (``gpus_per_node`` 0),
else from the launcher's reader placement (``NEMO_GYM_REMOTE_READER_SRUN``, the CPU partition the heavy readers already
run on), else from ``--judge-partition``; without one the profile is refused, naming the three. The walltime is the
run's segment walltime bounded by the partition's MaxTime, or a default when neither is known. The run's QOS belongs
to its GPU allocation and does not follow (a GPU QOS can refuse a job without GPUs): the judge job takes ``--judge-qos``,
else the saved CPU pool's own QOS, else the QOS the reader placement names (``-q``/``--qos``) for that partition, else
the partition's default.
"""

import shlex
from dataclasses import dataclass
from typing import Any

from nemo_gym.orchestration.api import NodePool, SlurmComputeConfig, SubmitConfig
from nemo_gym.orchestration.executors.resume_script import _walltime_to_seconds


# The derived compute's one node pool: one CPU node of the judge partition.
JUDGE_POOL_NAME = "judge"
# The judge job's walltime when the run records no segment walltime and the partition's MaxTime is unknown.
DEFAULT_JUDGE_WALLTIME = "04:00:00"
JUDGE_WALLTIME_REASON_DEFAULT = (
    f"the default judge walltime {DEFAULT_JUDGE_WALLTIME}: the run records no segment walltime and the partition's "
    "MaxTime is unknown"
)
# Job-level directives of the original allocation's shape, GPU resources and QOS, which one CPU judge node does not
# take: a GPU QOS can require GPUs of every job (QOSMinGRES).
_ALLOCATION_SHAPE_DIRECTIVES = frozenset(
    {"partition", "nodes", "ntasks-per-node", "gpus", "gpus-per-node", "gpus-per-task", "gres", "qos"}
)
_NO_PROFILE = "this submission has no existing CPU-only judge execution profile and none can be derived"


class JudgeProfileUnsupported(ValueError):
    """No CPU-only judge execution profile can be derived from the submission: the message says why and what would
    allow one."""


@dataclass(frozen=True)
class JudgeExecutionProfile:
    """Where and for how long a judge-only job of a run grades its saved answers again."""

    partition: str
    # Which of the three sources named the partition (see ``choose_judge_partition``).
    partition_source: str
    walltime: str
    # What bounded the walltime, each as a sentence fragment for the operator's log.
    walltime_bounds: tuple[str, ...]
    # The run's compute with one CPU node of the partition and the bounded walltime in place of its allocation.
    compute: SlurmComputeConfig
    # The QOS the job asks for and where it came from; None takes the partition's default.
    qos: str | None = None
    qos_source: str = "the partition's default"

    def report(self) -> dict[str, Any]:
        return {
            "partition": self.partition,
            "partition_source": self.partition_source,
            "qos": self.qos,
            "qos_source": self.qos_source,
            "walltime": self.walltime,
            "walltime_bounds": list(self.walltime_bounds),
        }


def is_cpu_only_submission(config: SubmitConfig) -> bool:
    """Whether the submission already runs on CPU nodes only: no service, no policy model, and every node pool with
    ``gpus_per_node`` 0. Such a run's own compute is its judge execution profile."""
    compute = next(iter(config.compute.values()))
    return (
        isinstance(compute, SlurmComputeConfig)
        and not config.services
        and not config.driver.policy_model
        and bool(compute.node_pools)
        and all(pool.gpus_per_node == 0 for pool in compute.node_pools.values())
    )


def _reader_step_option(args: str | None, short: str, long: str) -> str | None:
    if not args or not args.strip():
        return None
    try:
        words = shlex.split(args)
    except ValueError:
        return None
    for index, word in enumerate(words):
        if word in (short, long):
            return words[index + 1] if index + 1 < len(words) and words[index + 1] else None
        if word.startswith(f"{long}="):
            return word.partition("=")[2] or None
        if word.startswith(short) and not word.startswith("--") and len(word) > 2:
            return word[2:]
    return None


def reader_step_partition(args: str | None) -> str | None:
    """The partition the reader step's srun arguments (``NEMO_GYM_REMOTE_READER_SRUN``) name, if any."""
    return _reader_step_option(args, "-p", "--partition")


def reader_step_qos(args: str | None) -> str | None:
    """The QOS the reader step's srun arguments (``NEMO_GYM_REMOTE_READER_SRUN``) name, if any."""
    return _reader_step_option(args, "-q", "--qos")


def choose_judge_partition(
    config: SubmitConfig, cluster: str, *, partition: str | None = None, reader_srun: str | None = None
) -> tuple[str, str]:
    """The CPU partition a judge-only job of this submission queues on, and where the name came from.

    ``partition`` (``--judge-partition``) wins; else the first saved node pool with ``gpus_per_node`` 0 names it; else
    the reader step's arguments (``reader_srun``, the value of ``NEMO_GYM_REMOTE_READER_SRUN``). A submission that
    serves anything beside its policy model has no CPU-only profile: a judge job cannot start those services.
    """
    compute = config.compute[cluster]
    if not isinstance(compute, SlurmComputeConfig):
        raise JudgeProfileUnsupported(f"{_NO_PROFILE}: the saved compute is not Slurm")
    others = sorted(name for name in config.services if name != config.driver.policy_model)
    if others:
        raise JudgeProfileUnsupported(
            f"{_NO_PROFILE}: service(s) {', '.join(others)} run on its allocation beside the policy, and a CPU-only "
            "judge job cannot serve them; policy services are not restarted"
        )
    if partition:
        return partition, "--judge-partition"
    for name, pool in compute.node_pools.items():
        if pool.gpus_per_node == 0:
            return pool.partition, f"the saved node pool {name!r}"
    reader = reader_step_partition(reader_srun)
    if reader:
        return reader, "the launcher's reader placement (NEMO_GYM_REMOTE_READER_SRUN)"
    raise JudgeProfileUnsupported(
        f"{_NO_PROFILE}: no CPU partition is known (the saved compute declares no node pool with gpus_per_node 0, "
        "the launcher's NEMO_GYM_REMOTE_READER_SRUN names no partition, and no --judge-partition was given); policy "
        "services are not restarted"
    )


def _slurm_duration(seconds: int) -> str:
    return f"{seconds // 3600:02}:{seconds % 3600 // 60:02}:{seconds % 60:02}"


def bound_judge_walltime(segment: str | None, limit: dict[str, Any] | None) -> tuple[str, tuple[str, ...]]:
    """The judge job's walltime: the run's segment walltime ``segment``, bounded by the partition's MaxTime in
    ``limit`` (as ``_partition_time_limit`` reports it), or the default when neither is known; with what bounded it."""
    candidates: list[int] = []
    bounds: list[str] = []
    if segment:
        try:
            candidates.append(_walltime_to_seconds(segment))
            bounds.append(f"the run's segment walltime {segment}")
        except ValueError:
            pass
    max_s = (limit or {}).get("max_time_s")
    if type(max_s) is int and max_s > 0:
        candidates.append(max_s)
        bounds.append(f"partition MaxTime {(limit or {}).get('max_time')}")
    if not candidates:
        return DEFAULT_JUDGE_WALLTIME, (JUDGE_WALLTIME_REASON_DEFAULT,)
    return _slurm_duration(min(candidates)), tuple(bounds)


def judge_execution_profile(
    config: SubmitConfig,
    cluster: str,
    partition: str,
    partition_source: str,
    limit: dict[str, Any] | None,
    *,
    qos: str | None = None,
    reader_srun: str | None = None,
) -> JudgeExecutionProfile:
    """The profile of a judge-only job of this submission on ``partition`` (see ``choose_judge_partition``), with the
    partition's MaxTime ``limit`` bounding the walltime (see ``bound_judge_walltime``).

    The derived compute keeps the account, forwarded env and job-level directives of the run, less the directives
    that shape the original allocation, request its GPUs or name its QOS; a saved CPU pool of the partition lends its
    own directives to the judge node. The job's QOS is ``qos`` (``--judge-qos``), else the saved CPU pool's, else the
    reader placement's (``reader_srun``, the value of ``NEMO_GYM_REMOTE_READER_SRUN``) when that placement names this
    partition or none, else the partition's default.
    """
    compute = config.compute[cluster]
    walltime, bounds = bound_judge_walltime(compute.walltime, limit)
    saved_pool = next(
        (pool for pool in compute.node_pools.values() if pool.partition == partition and pool.gpus_per_node == 0),
        None,
    )
    pool = NodePool(
        partition=partition,
        nodes=1,
        ntasks_per_node=1,
        gpus_per_node=0,
        extra_args=dict(saved_pool.extra_args) if saved_pool is not None else {},
    )
    extra_args = {k: v for k, v in compute.extra_args.items() if k not in _ALLOCATION_SHAPE_DIRECTIVES}
    qos_source = "the partition's default"
    if qos:
        qos_source = "--judge-qos"
        extra_args["qos"] = qos
        pool.extra_args.pop("qos", None)
    elif saved_pool is not None and saved_pool.extra_args.get("qos"):
        qos, qos_source = str(saved_pool.extra_args["qos"]), "the saved CPU node pool"
    elif reader_step_qos(reader_srun) and reader_step_partition(reader_srun) in (None, partition):
        qos, qos_source = reader_step_qos(reader_srun), "the launcher's reader placement (NEMO_GYM_REMOTE_READER_SRUN)"
        extra_args["qos"] = qos
    judge_compute = compute.model_copy(
        update={"walltime": walltime, "node_pools": {JUDGE_POOL_NAME: pool}, "extra_args": extra_args}
    )
    return JudgeExecutionProfile(partition, partition_source, walltime, bounds, judge_compute, qos, qos_source)


def judge_job_config(
    config: SubmitConfig,
    cluster: str,
    benchmark: str,
    profile: JudgeExecutionProfile,
    command: str,
    *,
    gym_ref: str | None = None,
) -> SubmitConfig:
    """The submission config of a judge-only job of ``benchmark``: the run's driver (its env, mounts, container and
    pinned install, at ``gym_ref`` when given) runs ``command`` in place of ``gym eval run``, with no service, no
    policy model, no preparation, no resume chain, on the profile's compute."""
    original = config.driver.benchmarks[benchmark]
    bench = original.model_copy(update={"prepare": {}, "run": {}, "command": command, "resumable": False})
    install = config.driver.gym_install
    if gym_ref is not None and install is not None:
        install = install.model_copy(update={"ref": gym_ref})
    driver = config.driver.model_copy(
        update={"benchmarks": {benchmark: bench}, "policy_model": None, "gym_install": install}
    )
    return config.model_copy(update={"services": {}, "driver": driver, "compute": {cluster: profile.compute}})
