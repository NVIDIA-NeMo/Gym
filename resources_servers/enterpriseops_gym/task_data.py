# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-data schema for the enterpriseops_gym server.

Rows are written by ``benchmarks/enterpriseops_gym/prepare.py`` from the upstream EnterpriseOps-Gym task set, all
fields top-level. ``/seed_session`` validates its ``task_data`` with ``TaskData``; ``/verify`` reads ``task_id`` and
``domain`` and scores the verifiers the seed received.
"""

import re
from typing import Dict, List, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator


# The header that names a task database on every gym call; the server sets it, never the task.
DATABASE_ID_HEADER = "x-database-id"
_CONTEXT_HEADER = re.compile(r"^x-[a-z0-9-]+$")

ComparisonType = Literal["equals", "greater_than", "less_than", "contains"]


class SQLValidationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(description="Read query run against the gym database through its /api/sql-runner.")
    expected_value: JsonValue
    comparison_type: ComparisonType = "equals"


class DatabaseStateVerifier(BaseModel):
    """One ``database_state`` verifier of a task. The public task set has no other verifier type."""

    model_config = ConfigDict(extra="forbid")

    verifier_type: Literal["database_state"]
    name: str
    description: str = ""
    gym_name: str = Field(description="The gym server whose database the query runs against.")
    validation_config: SQLValidationConfig


class GymServerTask(BaseModel):
    """One gym server of a task: the dump its database is seeded from and the context headers of every call."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(description="Gym server name, e.g. gym-email-mcp; selects the image in the server config.")
    seed_database_file: str = Field(description="Path of the task's SQL dump inside the upstream gym_dbs.zip.")
    headers: Dict[str, str] = Field(
        default_factory=dict,
        description="Lowercase x- headers carrying the task context, such as the acting user's token.",
    )

    @model_validator(mode="after")
    def _context_headers_only(self) -> "GymServerTask":
        for name in self.headers:
            if not _CONTEXT_HEADER.match(name) or name == DATABASE_ID_HEADER:
                raise ValueError(f"Gym server {self.name}: {name!r} is not a lowercase x- context header")
        return self


class TaskData(BaseModel):
    model_config = ConfigDict(extra="allow")

    task_id: str = Field(
        description="Upstream task ID; checked against the seed's TaskId and echoed by /verify.",
        json_schema_extra={"consumed_by": ["verify", "provenance"]},
    )
    domain: str = Field(
        description="Upstream domain (calendar, csm, drive, email, hr, hybrid, itsm, teams); groups the metrics.",
        json_schema_extra={"consumed_by": ["metrics"]},
    )
    gym_servers: List[GymServerTask] = Field(
        min_length=1,
        description="The gym servers to start, in the task's order; the first one is lent to the agent.",
        json_schema_extra={"consumed_by": ["verify"]},
    )
    tool_servers: Dict[str, str] = Field(
        min_length=1,
        description="The tools the agent may call, each mapped to the gym server that serves it.",
        json_schema_extra={"consumed_by": ["verify"]},
    )
    verifiers: List[DatabaseStateVerifier] = Field(
        min_length=1,
        description="SQL checks on the final database state; the task passes when all of them pass.",
        json_schema_extra={"consumed_by": ["verify"]},
    )

    @model_validator(mode="after")
    def _references_known_gyms(self) -> "TaskData":
        names = [gym.name for gym in self.gym_servers]
        if len(set(names)) != len(names):
            raise ValueError(f"Task {self.task_id} names a gym server twice: {names}")
        unknown = ({*self.tool_servers.values()} | {v.gym_name for v in self.verifiers}) - set(names)
        if unknown:
            raise ValueError(f"Task {self.task_id} refers to gym servers it does not start: {sorted(unknown)}")
        return self
