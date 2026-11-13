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
"""The execution profile of a judge-only job that grades a run's saved answers again.

A run that served its policy on GPUs has no allocation a judge-only job could reuse: its node pools are GPU pools and
its services are the policy's. The profile derived here keeps what grading needs from the saved submission config,
the driver with its env, mounts, container and pinned install, so the run's own judge and verifier servers start as
they did, and replaces what grading does not need. It is one of two kinds.

A run that serves nothing beside its policy gets the CPU profile: no service, no policy model, one node of a CPU
partition and a bounded walltime. The partition comes from the saved config when it declares a CPU node pool
(``gpus_per_node`` 0), else from the launcher's reader placement (``NEMO_GYM_REMOTE_READER_SRUN``, the CPU partition the
heavy readers already run on), else from ``--judge-partition``; without one the profile is refused, naming the three.
The run's QOS belongs to its GPU allocation and does not follow (a GPU QOS can refuse a job without GPUs): the judge
job takes ``--judge-qos``, else the saved CPU pool's own QOS, else the QOS the reader placement names (``-q``/``--qos``)
for that partition, else the partition's default.

A run that serves its judge as a service of its own, on a node pool of its own beside the policy's (a vLLM judge its
judge server reaches through ``GYM_POOL_*_NODES``), gets the served-judge profile: those services on their saved
node pools, with their saved shape and partition, and no policy service, policy model or policy pool. It is a GPU job
like the run, so it keeps the run's job-level QOS; ``--judge-partition`` and ``--judge-qos`` name a CPU placement and
are not used (a launcher that passes them on every recovery would put the GPU pool on a CPU partition or QOS). A
service that runs on the policy's own nodes has no served-judge profile: the judge job does not start the policy's
allocation.

Either way the walltime is the run's segment walltime bounded by the partition's MaxTime, or a default when neither is
known.
"""

import re
import shlex
from dataclasses import dataclass
from typing import Any

from nemo_gym.orchestration.api import (
    NodePool,
    RayServiceConfig,
    ServiceConfig,
    SlurmComputeConfig,
    SubmitConfig,
    VllmPDServiceConfig,
)
from nemo_gym.orchestration.executors.resume_script import _walltime_to_seconds


# The profile kinds: one CPU node without any service, or the run's own judge services on their saved node pools.
CPU_PROFILE = "cpu"
SERVED_JUDGE_PROFILE = "served-judge"
# The CPU profile's one node pool: one CPU node of the judge partition.
JUDGE_POOL_NAME = "judge"
# What a served-judge profile says of a --judge-partition or --judge-qos it was given.
CPU_PLACEMENT_UNUSED = "--judge-partition/--judge-qos apply to the CPU judge profile only; not used"
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
# Job-level directives of the original allocation's size, which the kept node pools state for themselves; its QOS and
# per-node GRES stay with the served-judge job, a GPU job on nodes of the same allocation's shape.
_SERVED_SHAPE_DIRECTIVES = _ALLOCATION_SHAPE_DIRECTIVES - {"gres", "qos"}
# What choose_judge_partition's refusal says when no CPU partition is named anywhere; the dispatch classifies on it.
NO_CPU_PARTITION = "no CPU partition is known"
_NO_PROFILE = "this submission has no existing CPU-only judge execution profile and none can be derived"
_NO_SERVED_PROFILE = "this submission has no served-judge execution profile"


class JudgeProfileUnsupported(ValueError):
    """No judge execution profile can be derived from the submission: the message says why and what would allow
    one."""


# The QOS source of a job that names no QOS. A named constant, so the reference docs show its name rather than a
# quoted default the page generator cannot embed.
PARTITION_DEFAULT_QOS = "the partition's default"


@dataclass(frozen=True)
class JudgeExecutionProfile:
    """Where and for how long a judge-only job of a run grades its saved answers again, and what it starts."""

    partition: str
    # Which source named the partition: one of ``choose_judge_partition``'s three, or the kept node pools.
    partition_source: str
    walltime: str
    # What bounded the walltime, each as a sentence fragment for the operator's log.
    walltime_bounds: tuple[str, ...]
    # The run's compute with the profile's node pools and the bounded walltime in place of its allocation.
    compute: SlurmComputeConfig
    # The QOS the job asks for and where it came from; None takes the partition's default.
    qos: str | None = None
    qos_source: str = PARTITION_DEFAULT_QOS
    # CPU_PROFILE or SERVED_JUDGE_PROFILE, the node pools the job runs on and the saved services it starts.
    kind: str = CPU_PROFILE
    pools: tuple[str, ...] = (JUDGE_POOL_NAME,)
    services: tuple[str, ...] = ()
    # CPU_PLACEMENT_UNUSED when a served-judge profile was given --judge-partition or --judge-qos, else None.
    unused: str | None = None

    def report(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "partition": self.partition,
            "partition_source": self.partition_source,
            "pools": list(self.pools),
            "services": list(self.services),
            "qos": self.qos,
            "qos_source": self.qos_source,
            "walltime": self.walltime,
            "walltime_bounds": list(self.walltime_bounds),
            **({"unused": self.unused} if self.unused else {}),
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
    the reader step's arguments (``reader_srun``, the value of ``NEMO_GYM_REMOTE_READER_SRUN``). Only a submission
    that serves nothing beside its policy model takes the CPU profile; one that does takes the served-judge profile
    (``served_judge_placement``).
    """
    compute = config.compute[cluster]
    if not isinstance(compute, SlurmComputeConfig):
        raise JudgeProfileUnsupported(f"{_NO_PROFILE}: the saved compute is not Slurm")
    if partition:
        return partition, "--judge-partition"
    for name, pool in compute.node_pools.items():
        if pool.gpus_per_node == 0:
            return pool.partition, f"the saved node pool {name!r}"
    reader = reader_step_partition(reader_srun)
    if reader:
        return reader, "the launcher's reader placement (NEMO_GYM_REMOTE_READER_SRUN)"
    raise JudgeProfileUnsupported(
        f"{_NO_PROFILE}: {NO_CPU_PARTITION} (the saved compute declares no node pool with gpus_per_node 0, "
        "the launcher's NEMO_GYM_REMOTE_READER_SRUN names no partition, and no --judge-partition was given); policy "
        "services are not restarted"
    )


# The scheduler's refusals of a job's placement: its partition, QOS, account or GPU request. Anything else the
# scheduler answers (a controller it cannot reach, a timeout) is not a placement problem.
_PLACEMENT_REFUSAL = re.compile(r"partition|\bqos|gres|account|node configuration", re.IGNORECASE)


# What a scheduler that did not answer says (an unreachable controller, a timeout, a failed query); waiting helps.
_SCHEDULER_UNAVAILABLE = re.compile(
    r"unable to contact|connection (?:refused|reset|timed out)|timed out|socket|slurm_load|"
    r"temporarily unavailable|scheduler unknown|unable to send|network is unreachable|no route to host",
    re.IGNORECASE,
)


def placement_refusal(diagnostic: str, served: bool) -> str:
    """The ``RecoveryDecision.judge_refusal`` of a judge-only job the scheduler did not take: its partition or QOS
    (``cpu_placement``, or ``served_placement`` for a run that serves its judge on its own saved node pool), else
    ``other`` when the scheduler did not answer, else ``refused`` (a definite refusal of another kind)."""
    if not _PLACEMENT_REFUSAL.search(diagnostic):
        return "other" if _SCHEDULER_UNAVAILABLE.search(diagnostic) else "refused"
    return "served_placement" if served else "cpu_placement"


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
    if served_judge_placement(config, cluster) is not None:
        raise JudgeProfileUnsupported(
            f"{_NO_PROFILE}: it serves {', '.join(n for n in config.services if n != config.driver.policy_model)} "
            "beside its policy, which only the served-judge profile starts"
        )
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
    qos_source = PARTITION_DEFAULT_QOS
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


def _service_pools(name: str, service: ServiceConfig) -> set[str | None]:
    """The node pools a saved service runs on; None for a part that takes the whole allocation (no ``node_pool``),
    the driver's node (a Ray cluster without ``node_pools``) or the first node (a prefill/decode router)."""
    if isinstance(service, RayServiceConfig):
        return set(service.node_pools) or {None}
    if isinstance(service, VllmPDServiceConfig):
        return {tier.node_pool for tier in service.tiers(name).values()} | {service.node_pool}
    return {service.node_pool}


def served_judge_placement(config: SubmitConfig, cluster: str) -> tuple[tuple[str, ...], dict[str, NodePool]] | None:
    """The saved services beside the policy a judge-only job of this submission starts, and the saved node pools
    they run on, in the saved order; None when it serves nothing beside its policy (the CPU profile).

    A service on a node of the policy's, its pool or the whole allocation, is refused, naming it: the judge job does
    not start the policy's allocation, and a separate node pool for that service would allow it.
    """
    compute = config.compute[cluster]
    if not isinstance(compute, SlurmComputeConfig):
        raise JudgeProfileUnsupported(f"{_NO_SERVED_PROFILE}: the saved compute is not Slurm")
    policy = config.driver.policy_model
    services = tuple(name for name in config.services if name != policy)
    if not services:
        return None
    policy_pools = _service_pools(policy, config.services[policy]) if policy in config.services else set()
    pools = {name: _service_pools(name, config.services[name]) for name in services}
    shared = sorted(name for name in services if None in pools[name] or pools[name] & policy_pools)
    if shared:
        raise JudgeProfileUnsupported(
            f"{_NO_SERVED_PROFILE}: service(s) {', '.join(shared)} run on the policy's nodes (its node pool, or no node_pool "
            "of their own, which is the whole allocation), and a judge-only job does not start the policy's "
            "allocation; a separate node pool for each of them, beside the policy's, would allow it; policy services "
            "are not restarted"
        )
    kept = set().union(*pools.values())
    return services, {name: pool for name, pool in compute.node_pools.items() if name in kept}


def served_judge_profile(
    config: SubmitConfig,
    cluster: str,
    limit: dict[str, Any] | None,
    *,
    partition: str | None = None,
    qos: str | None = None,
) -> JudgeExecutionProfile:
    """The served-judge profile of a submission that serves its judge beside its policy (see
    ``served_judge_placement``), with the kept pools' partition MaxTime ``limit`` bounding the walltime.

    The derived compute is the run's, with only the kept node pools in their saved shape (nodes, tasks, GPUs,
    partition and directives) and the bounded walltime; its account, forwarded env and job-level directives stay, less
    those sizing the original allocation, which the kept pools state for themselves. It is a GPU job like the run: the
    QOS stays the run's job-level one, else the kept pools', else the partition's default. ``partition``
    (``--judge-partition``) and ``qos`` (``--judge-qos``) name a CPU placement and are recorded as not used.
    """
    placement = served_judge_placement(config, cluster)
    if placement is None:
        raise JudgeProfileUnsupported(f"{_NO_SERVED_PROFILE}: it serves nothing beside its policy")
    services, pools = placement
    compute = config.compute[cluster]
    walltime, bounds = bound_judge_walltime(compute.walltime, limit)
    extra_args = {k: v for k, v in compute.extra_args.items() if k not in _SERVED_SHAPE_DIRECTIVES}
    pool_qos = next((str(pool.extra_args["qos"]) for pool in pools.values() if pool.extra_args.get("qos")), None)
    if extra_args.get("qos"):
        job_qos, qos_source = str(extra_args["qos"]), "the run's job-level QOS"
    elif pool_qos:
        job_qos, qos_source = pool_qos, "the saved node pool's QOS"
    else:
        job_qos, qos_source = None, PARTITION_DEFAULT_QOS
    judge_compute = compute.model_copy(update={"walltime": walltime, "node_pools": pools, "extra_args": extra_args})
    return JudgeExecutionProfile(
        partition=",".join(sorted({pool.partition for pool in pools.values()})),
        partition_source=f"the saved node pool(s) {', '.join(repr(name) for name in pools)}",
        walltime=walltime,
        walltime_bounds=bounds,
        compute=judge_compute,
        qos=job_qos,
        qos_source=qos_source,
        kind=SERVED_JUDGE_PROFILE,
        pools=tuple(pools),
        services=services,
        unused=CPU_PLACEMENT_UNUSED if partition or qos else None,
    )


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
    pinned install, at ``gym_ref`` when given) runs ``command`` in place of ``gym eval run``, with no policy model, no
    preparation, no resume chain, on the profile's compute. It starts the profile's services, the saved judge
    services of a served-judge profile, and none of a CPU profile."""
    original = config.driver.benchmarks[benchmark]
    bench = original.model_copy(update={"prepare": {}, "run": {}, "command": command, "resumable": False})
    install = config.driver.gym_install
    if gym_ref is not None and install is not None:
        install = install.model_copy(update={"ref": gym_ref})
    driver = config.driver.model_copy(
        update={"benchmarks": {benchmark: bench}, "policy_model": None, "gym_install": install}
    )
    services = {name: config.services[name] for name in profile.services}
    return config.model_copy(update={"services": services, "driver": driver, "compute": {cluster: profile.compute}})
