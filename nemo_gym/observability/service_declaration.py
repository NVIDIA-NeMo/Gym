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

"""Which model service ran on which node and GPUs, so GPU and engine series can be tied to a service.

`gym eval submit` writes `<stem>_service_declaration.json` next to the rollouts when a Slurm job starts
(see `nemo_gym.orchestration.executors.service_declaration`). A run whose services Gym did not place
records an unavailable declaration with the reason instead, so a missing declaration is never read as
an empty one. Unknown values are absent from the file, never guessed.
"""

import logging
import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ValidationError

from nemo_gym.path_utils import service_declaration_path_for


logger = logging.getLogger(__name__)

SERVICE_DECLARATION_SCHEMA_VERSION = 1
NOT_PLACED_REASON = (
    "No service placement was recorded: the model services of this run were not placed by `gym eval submit` on Slurm."
)


class DeclaredService(BaseModel):
    """One deployed model service and where its layout puts it."""

    # `<component>/<service>`: the scrape job name, so the engine series' `service.name.override`.
    service: str
    # The engine series' `gym_service` label: the name under `services:` (a tier as `<name>-prefill`).
    gym_service: str
    kind: Literal["vllm", "vllm_pd_tier", "vllm_pd_router"]
    pool_role: Literal["prefill", "decode", "router"] | None = None
    node_pool: str | None = None
    # Indexes into the allocation's node list (`scontrol show hostnames` order); see `ServiceDeclaration.nodes`.
    node_indices: list[int]
    port: int
    tensor_parallel_size: int | None = None
    pipeline_parallel_size: int | None = None
    # `number_of_instances`: data-parallel ranks of one engine, or independent replicas behind the Ray Serve gateway.
    data_parallel_size: int | None = None
    ray_serve: bool | None = None
    server_per_node: bool | None = None
    # GPUs the layout uses on each of its nodes; absent when Ray decides the split.
    gpus_per_node: int | None = None


class DeclaredNode(BaseModel):
    """One node a declared service runs on, as the job saw it at start."""

    index: int
    hostname: str
    services: list[str]
    gpus_observed: bool
    gpus_unavailable_reason: str | None = None


class DeclaredGpu(BaseModel):
    """One GPU of a declared service's node: the row that joins a DCGM series to services.

    `services` lists every GPU-using service on the node. More than one means the services share the
    node and Gym does not know which of them uses this GPU.
    """

    # The DCGM `UUID` label.
    gpu_uuid: str
    node_index: int
    # The NVML index `nvidia-smi` reports, which is DCGM's `gpu` label.
    gpu_index: int
    services: list[str]


class ServiceDeclaration(BaseModel):
    """The run folder's service declaration: available with placements, or unavailable with a reason."""

    schema_version: Literal[1] = SERVICE_DECLARATION_SCHEMA_VERSION
    status: Literal["available", "unavailable"]
    reason: str | None = None
    # The collector's `run_id` resource attribute (the gym job id), `benchmark` and `slurm_job_id`.
    run_id: str | None = None
    benchmark: str | None = None
    slurm_job_id: str | None = None
    services: list[DeclaredService] = []
    nodes: list[DeclaredNode] = []
    gpus: list[DeclaredGpu] = []


def record_service_declaration(output_fpath: Path) -> ServiceDeclaration:
    """Keep the declaration this Slurm job wrote at start, or write an unavailable one in its place.

    A declaration is kept only when it is available and names the current `SLURM_JOB_ID`, so a run
    folder reused by a later run never keeps another job's placement. Never raises on I/O: a
    declaration that cannot be written is logged and the run goes on.
    """
    path = service_declaration_path_for(output_fpath)
    job_id = os.environ.get("SLURM_JOB_ID") or None
    existing = _read_declaration(path)
    if existing is not None and existing.status == "available":
        if job_id is not None and existing.slurm_job_id == job_id:
            return existing
        reason = (
            f"The service declaration in the run folder belongs to Slurm job {existing.slurm_job_id}, "
            f"not to this run (Slurm job {job_id or 'none'})."
        )
    else:
        reason = NOT_PLACED_REASON
    declaration = ServiceDeclaration(status="unavailable", reason=reason, slurm_job_id=job_id)
    try:
        path.write_text(declaration.model_dump_json(exclude_none=True, indent=2) + "\n")
    except OSError as error:
        logger.warning("Could not write the service declaration to %s: %s", path, error)
    return declaration


def _read_declaration(path: Path) -> ServiceDeclaration | None:
    try:
        return ServiceDeclaration.model_validate_json(path.read_bytes())
    except FileNotFoundError:
        return None
    except (OSError, ValidationError) as error:
        logger.warning("Ignoring an unreadable service declaration at %s: %s", path, error)
        return None
