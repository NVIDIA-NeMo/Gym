# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Model-visible rows reference trusted, separately prepared task packages."""

from pydantic import BaseModel, ConfigDict, Field


class TaskData(BaseModel):
    model_config = ConfigDict(extra="allow")

    task_id: str = Field(json_schema_extra={"consumed_by": ["verify"]})
    image: str = Field(json_schema_extra={"consumed_by": ["verify"]})
    task_fingerprint: str = Field(pattern=r"^[a-f0-9]{64}$", json_schema_extra={"consumed_by": ["verify"]})
    public_source: dict[str, str] | None = Field(
        default=None,
        description="Optional public-example provenance; not used by the verifier.",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
