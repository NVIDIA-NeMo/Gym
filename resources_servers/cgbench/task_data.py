# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-data schema of CG-Bench long_acc rows: two to eight choices, no VideoMME-v2 level or group."""

from typing import Self

from pydantic import model_validator

from resources_servers.video_mcqa.task_data import TaskData as SharedTaskData


class TaskData(SharedTaskData):
    """CG-Bench, CG-Bench-mini, and CG-Bench-hard rows (single-turn or agentic), each with 2 to 8 choices."""

    @model_validator(mode="after")
    def eight_or_fewer_choices(self) -> Self:
        if not 2 <= self.option_count <= 8:
            raise ValueError("CG-Bench requires between two and eight choices")
        return self
