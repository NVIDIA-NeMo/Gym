# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-data schema for sealed agentic instruction-following rows.

A row is one task of its backend, written flat: the backend's own fields, the sealed ``agentic_if``
configuration, and ``task_id``, the content digest of everything else. The row's own ``agentic_if.task_backend``
selects the backend's schema; ``/seed_session`` validates the configuration and its receipts.
"""

from typing import Annotated, Union

from pydantic import BaseModel, ConfigDict, Discriminator, Field, JsonValue, Tag

from resources_servers.scale_swe.task_data import TaskData as ScaleSWETaskData
from resources_servers.terminal_bench_2_1.task_data import TaskData as TerminalBench21TaskData


class SealedTask(BaseModel):
    """What every agentic IF row adds to its backend's own task."""

    model_config = ConfigDict(extra="allow")

    task_id: str = Field(
        pattern=r"^[0-9a-f]{64}$",
        description="Content digest of the rest of the row; collation keeps it as the task identity.",
    )
    agentic_if: dict[str, JsonValue] = Field(
        description="Harness, task backend, tools, events, continuation and private checks, with their receipts.",
        json_schema_extra={"consumed_by": ["prompt", "verify"]},
    )


class SWETask(SealedTask, ScaleSWETaskData):
    """A sealed Scale-SWE task."""


class TerminalBench21Task(SealedTask, TerminalBench21TaskData):
    """A sealed Terminal-Bench 2.1 task."""

    workdir: str | None = Field(
        default=None, description="Directory the harness starts in; the image's working directory when omitted."
    )
    agent_user: str | int | None = Field(
        default=None,
        description=(
            "Account name or uid the harness runs as inside the sandbox; None keeps the image default. The verifier "
            "runs as the image default, so a non-root agent_user needs a root-default image with that account."
        ),
    )
    task_hash: str | None = Field(
        default=None,
        description="The task source's digest of the exact task content in task_folder.",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    image_digest: str | None = Field(
        default=None,
        description="Registry digest of docker_image.",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    canonical_image_digest: str | None = Field(
        default=None,
        description="Registry digest of the task source's canonical image for this task.",
        json_schema_extra={"consumed_by": ["provenance"]},
    )


def _task_backend(row: object) -> str | None:
    configuration = row.get("agentic_if") if isinstance(row, dict) else getattr(row, "agentic_if", None)
    return configuration.get("task_backend") if isinstance(configuration, dict) else None


TaskData = Annotated[
    Union[Annotated[SWETask, Tag("swe")], Annotated[TerminalBench21Task, Tag("tb21")]],
    Discriminator(_task_backend),
]
