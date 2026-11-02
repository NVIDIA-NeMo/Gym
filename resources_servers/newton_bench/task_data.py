# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-data schema for the newton_bench server (scientific law discovery).

Session-seeded flow: task fields never appear on the verify body
(``NewtonBenchVerifyRequest`` is an empty ``BaseVerifyRequest`` subclass). They reach the server
via ``/seed_session`` (``NewtonBenchSeedSessionRequest`` in app.py, where all five fields are
required without defaults) and are stored in session metadata that ``verify()`` reads back to
select the ground-truth law and score the extracted equation. Required-ness here mirrors that
seed-session wire model. ``id`` appears in committed rows but is never consumed by any endpoint.
"""

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class TaskData(BaseModel):
    model_config = ConfigDict(extra="allow")

    module_name: str = Field(
        description=(
            "Physics module implementing the ground-truth law (e.g. gravity); selects the simulation and "
            "the reference equation at verify time. Missing module_name in session metadata yields reward 0."
        ),
        json_schema_extra={"consumed_by": ["verify"]},
    )
    difficulty: str = Field(
        description="Task difficulty tier (easy/medium/hard); parameterizes the law variant being simulated.",
        json_schema_extra={"consumed_by": ["verify"]},
    )
    system: str = Field(
        description="Physical system identifier within the module; selects the concrete equation form.",
        json_schema_extra={"consumed_by": ["verify"]},
    )
    law_version: str = Field(
        description="Version of the ground-truth law used for symbolic/numeric comparison.",
        json_schema_extra={"consumed_by": ["verify"]},
    )
    noise_level: float = Field(
        description="Observation noise applied by the simulated experiment tools.",
        json_schema_extra={"consumed_by": ["verify"]},
    )
    id: Optional[int] = Field(
        default=None,
        description="Source dataset row index; present in committed rows but never read by server code.",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
