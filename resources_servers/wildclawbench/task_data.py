# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepared native task metadata, read from verifier_metadata when the Resources session is seeded.

Keep this module dependency-light so collation can validate source-bound rows without native assets,
Docker/Enroot, or importing either server. Execution keys are injected by the Environment Server,
not supplied in the dataset.
"""

from pydantic import BaseModel, ConfigDict, Field


_LEGACY = {"legacy_location": "verifier_metadata"}


class TaskData(BaseModel):
    model_config = ConfigDict(extra="allow")

    task_id: str = Field(min_length=1, json_schema_extra=_LEGACY)
    task_path: str = Field(min_length=1, json_schema_extra=_LEGACY)
    source_revision: str = Field(pattern=r"^[a-f0-9]{40}$", json_schema_extra=_LEGACY)
    task_sha256: str = Field(pattern=r"^[a-f0-9]{64}$", json_schema_extra=_LEGACY)
    workspace_sha256: str = Field(pattern=r"^[a-f0-9]{64}$", json_schema_extra=_LEGACY)
    workspace: str = Field(min_length=1, json_schema_extra=_LEGACY)
    category: str = Field(min_length=1, json_schema_extra=_LEGACY)
    modality: str = Field(min_length=1, json_schema_extra=_LEGACY)
    timeout_seconds: int = Field(gt=0, json_schema_extra=_LEGACY)
