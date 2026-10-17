# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-data schema for the oragentbench server.

Rows carry the sandbox coordinates top level, as the ``terminal_bench_2_1`` server does, plus
the difficulty stratum the per-stratum metrics group on. ``seed_session()`` starts the task
container from ``docker_image``; the step list (single step, or the ``[[steps]]`` of a Harbor
multi-step task) is read from ``task.toml`` under ``task_folder`` at seed time, and ``verify()``
uploads each step's ``tests/`` (and ``solution/`` in model-free validation modes) from there.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class TaskData(BaseModel):
    model_config = ConfigDict(extra="allow")

    task_name: str = Field(
        description=(
            "Upstream task id from task.toml, e.g. 'oragentbench/airport_gate_assignment'; keys sandbox "
            "metadata (instance_id) and is echoed in the verify response."
        ),
        json_schema_extra={"consumed_by": ["seed_session", "verify"]},
    )
    docker_image: str = Field(
        description="Locally built per-task image the task container is started from.",
        json_schema_extra={"consumed_by": ["seed_session", "verify"]},
    )
    task_folder: str = Field(
        description=(
            "Path to the Harbor task directory (repo-relative or absolute); holds task.toml, "
            "instruction.md, tests/, solution/ and, for multi-step tasks, steps/<name>/."
        ),
        json_schema_extra={"consumed_by": ["seed_session", "verify"]},
    )
    difficulty: Literal["easy", "medium", "hard"] = Field(
        description="Upstream difficulty band from difficulty.json; groups the per-stratum pass rates.",
        json_schema_extra={"consumed_by": ["verify", "compute_metrics"]},
    )
