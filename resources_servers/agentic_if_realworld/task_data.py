# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Frozen task identifiers and private-release bindings; no criteria or answers are model input."""

from pydantic import BaseModel, ConfigDict, Field


class TaskData(BaseModel):
    model_config = ConfigDict(extra="allow")

    task_id: str = Field(
        description="Frozen task identifier selecting the input and private criteria.",
        json_schema_extra={"consumed_by": ["verify", "provenance"]},
    )
    item_id: str = Field(
        description="Verifier binding to the same frozen task identifier.",
        json_schema_extra={"consumed_by": ["verify", "provenance"], "legacy_location": "verifier_metadata"},
    )
    dataset_sha256: str = Field(
        pattern="^[a-f0-9]{64}$",
        description="SHA-256 of the frozen private release manifest.",
        json_schema_extra={"consumed_by": ["verify", "provenance"], "legacy_location": "verifier_metadata"},
    )
    input_sha256: str = Field(
        pattern="^[a-f0-9]{64}$",
        description="SHA-256 of the exact model-visible request.",
        json_schema_extra={"consumed_by": ["verify", "provenance"], "legacy_location": "verifier_metadata"},
    )
