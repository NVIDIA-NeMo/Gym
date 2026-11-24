# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-data schema for the videomme_v1 server: the shared video MCQA rows (video_mcqa), with four choices."""

from typing import Self

from pydantic import model_validator

from resources_servers.video_mcqa.task_data import TaskData as SharedTaskData


class TaskData(SharedTaskData):
    """VideoMME-v1 rows: four choices, and no VideoMME-v2 level or group."""

    @model_validator(mode="after")
    def four_choices(self) -> Self:
        if self.option_count != 4:
            raise ValueError("VideoMME-v1 questions have four choices")
        return self
