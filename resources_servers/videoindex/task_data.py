# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-data schema of Video-Index rows: two to ten choices, no VideoMME-v2 level or group."""

from typing import Self

from pydantic import model_validator

from resources_servers.video_mcqa.task_data import TaskData as SharedTaskData


class TaskData(SharedTaskData):
    """Video-Index v1.0 and v1.1 rows (single-turn or agentic), each with 2 to 10 choices."""

    @model_validator(mode="after")
    def ten_or_fewer_choices(self) -> Self:
        if not 2 <= self.option_count <= 10:
            raise ValueError("Video-Index requires between two and ten choices")
        return self
