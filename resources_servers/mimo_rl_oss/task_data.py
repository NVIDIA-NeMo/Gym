# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-data schema for mimo_rl_oss: each row carries mimoagent's instance dict for one MiMo-V2.6-RL-oss task."""

from typing import Any, Optional

from pydantic import BaseModel, ConfigDict


class TaskData(BaseModel):
    model_config = ConfigDict(extra="allow")

    # mimoagent instance: dataset_type, docker_image, cwd, instance_id, problem_statement, plus per-type grading fields.
    instance: dict[str, Any]
    subset: Optional[str] = None
