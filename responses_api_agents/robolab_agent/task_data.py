# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-data schema for the RoboLab agent dataset rows."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue


class TaskData(BaseModel):
    model_config = ConfigDict(extra="allow")

    case_id: str = Field(min_length=1)
    manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    official_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    case: dict[str, JsonValue]
    seed: int = Field(ge=0)
    native_run_plan: dict[str, JsonValue]
    score_source: Literal["native_summary"]
