# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native task identity shared by dataset validation and the ClawEval /run API."""

from pydantic import BaseModel, ConfigDict, Field


class TaskData(BaseModel):
    model_config = ConfigDict(extra="allow")

    task_id: str = Field(
        pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]*$",
        description="Native task directory name.",
        json_schema_extra={"legacy_location": "verifier_metadata"},
    )
    task_sha256: str = Field(
        pattern=r"^[a-f0-9]{64}$",
        description="SHA-256 of the native task.yaml used during preparation.",
        json_schema_extra={"legacy_location": "verifier_metadata"},
    )
    split: str | None = Field(
        default=None,
        description="Native task split, retained for provenance.",
        json_schema_extra={"legacy_location": "verifier_metadata"},
    )
