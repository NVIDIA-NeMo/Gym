# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepared AgentVista task identities; answers and upstream assets stay external."""

from pydantic import BaseModel, ConfigDict, Field


class TaskData(BaseModel):
    model_config = ConfigDict(extra="allow")

    task_id: str = Field(
        json_schema_extra={"legacy_location": "verifier_metadata", "consumed_by": ["verify", "provenance"]}
    )
    source_sha256: str = Field(
        json_schema_extra={"legacy_location": "verifier_metadata", "consumed_by": ["verify", "provenance"]}
    )
    category_id: str = Field(json_schema_extra={"legacy_location": "verifier_metadata", "consumed_by": ["metrics"]})
    harness_version: str = Field(
        json_schema_extra={"legacy_location": "verifier_metadata", "consumed_by": ["verify", "provenance"]}
    )
