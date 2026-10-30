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

"""The batch-script step that writes a Slurm job's service declaration when the job starts.

Everything known at submit time (services, layouts, the allocation nodes each one runs on) is
rendered into the script. The two facts only the running job knows are filled in there: each node's
hostname, and the GPUs `nvidia-smi` lists in a job step on that node (index and UUID). The step runs
before the services start; a node whose GPUs cannot be listed is declared with the reason and no GPU
rows. The shape is `nemo_gym.observability.service_declaration.ServiceDeclaration`.
"""

import json
import shlex
from pathlib import Path
from typing import Literal, Mapping

from nemo_gym.observability.service_declaration import SERVICE_DECLARATION_SCHEMA_VERSION, DeclaredService
from nemo_gym.orchestration.api import (
    BaseModelServiceConfig,
    VllmNodeServiceConfig,
    VllmPDServiceConfig,
    VllmPDTierConfig,
    VllmServiceConfig,
)


PoolRole = Literal["prefill", "decode", "router"]

# Seconds the GPU listing may take across all nodes before the job goes on without it.
GPU_LISTING_TIMEOUT_SECONDS = 120
GPUS_UNLISTED_REASON = "nvidia-smi listed no GPUs in a job step on this node"
GPU_LISTING_TIMED_OUT_REASON = f"the GPU listing step timed out after {GPU_LISTING_TIMEOUT_SECONDS} s"
GPU_LISTING_FAILED_REASON = "the GPU listing step failed with exit status"
NVIDIA_SMI_FAILED_REASON = "nvidia-smi failed on this node with exit status"


def pool_role(service: BaseModelServiceConfig) -> PoolRole | None:
    """The part a service plays in a prefill/decode pair; None outside one."""
    if isinstance(service, VllmPDServiceConfig):
        return "router"
    if isinstance(service, VllmPDTierConfig):
        return _TIER_ROLES[service.kv_transfer_config["kv_role"]]
    return None


def declared_service(
    name: str,
    service: BaseModelServiceConfig,
    *,
    component: str,
    node_indices: list[int],
    service_nodes: int,
    ray_serve: bool,
) -> DeclaredService:
    """One service's declaration from its config and the placement the batch script gives it.

    `node_indices` are the allocation nodes its step runs on, `service_nodes` how many nodes its
    command is sized for, and `ray_serve` whether the Ray Serve gateway runs its instances.
    """
    common = {
        "service": f"{component}/{name}",
        "gym_service": name,
        "pool_role": pool_role(service),
        "node_pool": service.node_pool,
        "node_indices": node_indices,
        "port": service.port,
    }
    if not isinstance(service, VllmServiceConfig):
        # The prefill/decode router fronts the tiers and holds no GPU.
        return DeclaredService(kind="vllm_pd_router", gpus_per_node=0, **common)
    server_per_node = isinstance(service, (VllmNodeServiceConfig, VllmPDTierConfig)) and service.server_per_node
    return DeclaredService(
        kind="vllm_pd_tier" if isinstance(service, VllmPDTierConfig) else "vllm",
        tensor_parallel_size=service.tensor_parallel_size,
        pipeline_parallel_size=service.pipeline_parallel_size,
        data_parallel_size=service.number_of_instances,
        ray_serve=ray_serve,
        server_per_node=server_per_node,
        gpus_per_node=_gpus_per_node(service, service_nodes, ray_serve=ray_serve, server_per_node=server_per_node),
        **common,
    )


def render_declaration_writer(
    services: list[DeclaredService],
    *,
    hosts: Mapping[int, str],
    pin_nodes: bool,
    run_id: str,
    benchmark: str,
    declaration_path: Path,
) -> str:
    """Batch-script lines that list each node's GPUs and write the declaration to `declaration_path`.

    `hosts` maps every node index the services use to the shell expansion of its hostname. One
    `srun` step covers all of them, so the listing costs one step whatever the node count; with
    `pin_nodes` it names them, which a single-node job does not need.
    """
    nodes = sorted({index for service in services for index in service.node_indices})
    on_node = {index: [s for s in services if index in s.node_indices] for index in nodes}
    nodelist = f' --nodelist="{",".join(hosts[index] for index in nodes)}"' if pin_nodes else ""
    listing = (
        # --wait=0: Slurm's WaitTime must not kill slower nodes' listings once the first node finishes.
        f"_gym_decl_gpus=$(timeout {GPU_LISTING_TIMEOUT_SECONDS} srun --overlap --wait=0{nodelist}"
        f" --nodes={len(nodes)} --ntasks={len(nodes)}"
        f" bash -c {shlex.quote(_LIST_GPUS)} 2>>logs/service_declaration-$SLURM_JOB_ID.log)"
    )
    node_calls = [
        f'_gym_decl_node {index} "{hosts[index]}" '
        f"{_shell_json([s.service for s in on_node[index]])} "
        f"{_shell_json([s.service for s in on_node[index] if s.gpus_per_node != 0])}"
        for index in nodes
    ]
    static = [s.model_dump(mode="json", exclude_none=True) for s in services]
    tmp_path = shlex.quote(f"{declaration_path}.tmp")
    write = (
        f"printf {shlex.quote(_DECLARATION_FORMAT)}"
        f' {_shell_json(run_id)} {_shell_json(benchmark)} "$SLURM_JOB_ID" {_shell_json(static)}'
        ' "$(IFS=,; echo "${_gym_decl_nodes[*]}")" "$(IFS=,; echo "${_gym_decl_rows[*]}")"'
        f" > {tmp_path}"
        # A copy per job, since the next job of a resume chain rewrites the run folder's file.
        f" && cp {tmp_path} logs/service_declaration-$SLURM_JOB_ID.json"
        f" && mv {tmp_path} {shlex.quote(str(declaration_path))}"
    )
    return "\n".join(
        [
            "# service declaration: which model service runs on which node and GPUs",
            listing,
            "_gym_decl_status=$?",
            _LISTING_REASON,
            _NODE_FUNCTION,
            *node_calls,
            f"mkdir -p {shlex.quote(str(declaration_path.parent))}",
            write,
        ]
    )


_TIER_ROLES: dict[str, PoolRole] = {"kv_producer": "prefill", "kv_consumer": "decode"}

# Runs once per node inside the listing step. SLURMD_NODENAME is the node's name as
# `scontrol show hostnames` lists it, which is how each line finds its node again.
# A failing nvidia-smi prints a marker line with its exit status instead of failing the task, so one node's failure
# never ends the other nodes' listings, whatever the cluster's KillOnBadExit.
_LIST_GPUS = (
    "out=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader); s=$?; "
    'if [ "$s" -ne 0 ]; then echo "$SLURMD_NODENAME, nvidia-smi-exit, $s"; '
    'else printf "%s\\n" "$out" | sed "s/^/$SLURMD_NODENAME, /"; fi'
)

# Only lines that look like `<host>, <index>, GPU-<uuid>` become rows, so stray output never turns
# into a GPU. ENVIRON carries the services JSON without awk's escape processing.
_NODE_FUNCTION = r"""_gym_decl_nodes=()
_gym_decl_rows=()
_gym_decl_node() {
    # <node index> <hostname> <services on the node> <its GPU-using services>
    local host="$2" rows smi_status reason
    rows=$(printf '%s\n' "$_gym_decl_gpus" | GYM_DECL_SERVICES="$4" awk -F' *, *' -v host="$host" -v node="$1" '
        $1 == host && $2 ~ /^[0-9]+$/ && $3 ~ /^GPU-[0-9A-Za-z-]+$/ {
            printf "%s{\"gpu_uuid\":\"%s\",\"node_index\":%d,\"gpu_index\":%d,\"services\":%s}", sep, $3, node, $2, ENVIRON["GYM_DECL_SERVICES"]
            sep = ","
        }')
    if [ -n "$rows" ]; then
        _gym_decl_rows+=("$rows")
        _gym_decl_nodes+=("{\"index\":$1,\"hostname\":\"$host\",\"services\":$3,\"gpus_observed\":true}")
    else
        smi_status=$(printf '%s\n' "$_gym_decl_gpus" | awk -F' *, *' -v host="$host" '$1 == host && $2 == "nvidia-smi-exit" && $3 ~ /^[0-9]+$/ {print $3; exit}')
        reason="$_gym_decl_reason"
        [ -n "$smi_status" ] && reason="NVIDIA_SMI_FAILED $smi_status"
        _gym_decl_nodes+=("{\"index\":$1,\"hostname\":\"$host\",\"services\":$3,\"gpus_observed\":false,\"gpus_unavailable_reason\":\"$reason\"}")
    fi
}""".replace("NVIDIA_SMI_FAILED", NVIDIA_SMI_FAILED_REASON)

# Why a node has no GPU rows, from the listing step's exit status; the step's stderr is in
# logs/service_declaration-<job>.log.
_LISTING_REASON = f"""case "$_gym_decl_status" in
    0) _gym_decl_reason="{GPUS_UNLISTED_REASON}" ;;
    124) _gym_decl_reason="{GPU_LISTING_TIMED_OUT_REASON}" ;;
    *) _gym_decl_reason="{GPU_LISTING_FAILED_REASON} $_gym_decl_status" ;;
esac"""

_DECLARATION_FORMAT = (
    f'{{"schema_version":{SERVICE_DECLARATION_SCHEMA_VERSION},"status":"available",'
    '"run_id":%s,"benchmark":%s,"slurm_job_id":"%s","services":%s,"nodes":[%s],"gpus":[%s]}\\n'
)


def _gpus_per_node(
    service: VllmServiceConfig, service_nodes: int, *, ray_serve: bool, server_per_node: bool
) -> int | None:
    """GPUs each node of the service runs a process on, when the launch fixes it."""
    per_replica = service.tensor_parallel_size * service.pipeline_parallel_size
    if server_per_node:
        return per_replica
    if service_nodes <= 1:
        return per_replica * service.number_of_instances
    if ray_serve or service.number_of_instances == 1:
        # Ray places the replicas, or one replica's ranks, across the nodes.
        return None
    # Multi-node data parallelism: each node runs an equal share of the replicas.
    return per_replica * (service.number_of_instances // service_nodes)


def _shell_json(value: object) -> str:
    return shlex.quote(json.dumps(value, separators=(",", ":")))
