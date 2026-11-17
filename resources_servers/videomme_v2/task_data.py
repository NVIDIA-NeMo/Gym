# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-data schema for the videomme_v2 server: the shared video MCQA rows (video_mcqa)."""

from resources_servers.video_mcqa.task_data import VideoMCQATaskData


class TaskData(VideoMCQATaskData):
    """VideoMME-v2 rows: any option count, with the optional level and four-question group."""
