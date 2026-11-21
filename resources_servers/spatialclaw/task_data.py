# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task fields shared by preparation, validation, and native verification."""

from typing import Any

from pydantic import BaseModel, ConfigDict


class TaskData(BaseModel):
    model_config = ConfigDict(extra="allow")

    sample_id: str
    answer: Any = None
    question_type: str | None = None
    benchmark_name: str | None = None
    dataset_config: str | None = None
