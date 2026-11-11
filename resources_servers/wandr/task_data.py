# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from pydantic import BaseModel, ConfigDict


class TaskData(BaseModel):
    model_config = ConfigDict(extra="allow")

    task_name: str
    task_id: str
    required_output_files: list[str]
