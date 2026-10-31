# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-data schema for the GSM8K reference scorer."""

from pydantic import BaseModel, ConfigDict, Field


class TaskData(BaseModel):
    model_config = ConfigDict(extra="allow")

    expected_answer: str = Field(
        description="Worked reference solution ending in #### <number>.",
        json_schema_extra={"consumed_by": ["verify"]},
    )
    language_code: str | None = Field(
        default=None,
        description="Dataset language code used for metric breakdowns.",
        json_schema_extra={"consumed_by": ["metrics"]},
    )
