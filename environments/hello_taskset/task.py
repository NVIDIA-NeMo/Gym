# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Task-data model for Hello Taskset."""

from pydantic import BaseModel, ConfigDict, Field


class TaskData(BaseModel):
    """Values that vary between tasks in the taskset."""

    model_config = ConfigDict(extra="forbid")

    expected_path: str = Field(
        pattern=r"^/workspace/.+",
        description="Absolute path to the file the verifier checks.",
    )
    expected_content: str = Field(
        description="UTF-8 text expected in the file.",
    )
