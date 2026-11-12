# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-data schema for the officeenv server.

A row is what ``officeenv emit`` (``officeenv.emit.gym_row``) writes: ``responses_create_params``
plus three task fields at the top level, with no verifier_metadata. ``officeenv``, the TaskSpec,
is the only field the wire requires (``OfficeEnvSeedSessionRequest`` in app.py): the seed builds
the episode's world from it, and its checks produce the reward at verify. OfficeEnv validates the
spec itself when seeding (a malformed spec fails the seed with 400 ``invalid_task``), so it stays an
untyped object here rather than a copy of OfficeEnv's TaskSpec model.
"""

from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field


class TaskData(BaseModel):
    model_config = ConfigDict(extra="allow")

    officeenv: dict[str, Any] = Field(
        description=(
            "The OfficeEnv TaskSpec: connected apps and their data, the acting user, the exposure mode, "
            "faults, data injections and the final-state checks that produce the reward."
        ),
        json_schema_extra={"consumed_by": ["verify"]},
    )
    officeenv_version: Optional[str] = Field(
        default=None,
        description=(
            "The OfficeEnv version that emitted the row. The seed refuses another major version "
            "(400 incompatible_row); rows emitted before 1.0.0 carry none and run as they are."
        ),
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    task_id: Optional[str] = Field(
        default=None,
        description="The spec's task_id, copied to the top level by the emitter; unread by the server.",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
