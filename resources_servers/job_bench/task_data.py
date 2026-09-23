# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pydantic import BaseModel, ConfigDict


class TaskData(BaseModel):
    model_config = ConfigDict(extra="allow")

    task_id: str
    task_dir: str
    rubrics_file: str
