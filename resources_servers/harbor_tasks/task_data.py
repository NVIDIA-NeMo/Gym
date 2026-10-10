# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-data schema for the harbor_tasks server.

A row names one task of a configured Harbor dataset. ``seed_session`` resolves the task directory from the
``harbor_datasets`` alias, starts a sandbox from the task's ``[environment].docker_image``, and hands it to the
agent. ``verify`` runs the task's Harbor verifier (``tests/test.sh``) in that sandbox. The instruction is not part of
``TaskData``: ``prepare.py`` copies ``instruction.md`` into ``responses_create_params.input`` so any agent can run the
task.
"""

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class TaskData(BaseModel):
    model_config = ConfigDict(extra="allow")

    harbor_dataset: str = Field(
        description="Alias of a dataset under the server's harbor_datasets config.",
        json_schema_extra={"consumed_by": ["seed_session", "verify"]},
    )
    task_name: str = Field(
        description="Harbor task name within the dataset: the task.toml [task].name or the task directory name.",
        json_schema_extra={"consumed_by": ["seed_session", "verify"]},
    )
    task_id: Optional[str] = Field(
        default=None,
        description="Stable task identity for rollout collection; the Harbor task name.",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
