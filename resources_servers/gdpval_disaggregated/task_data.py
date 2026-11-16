# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dataset schema for the disaggregated GDPval resources server.

Gym rows carry the prompt and task selector in ``responses_create_params``; the
agent reads ``metadata.task_id`` and supplies it to seed and verification calls.
The resources server loads task content, references, and rubrics from its separate
configured task catalog. They are not task-owned fields of the Gym input row.
"""

from pydantic import BaseModel, ConfigDict


class TaskData(BaseModel):
    """Gym input rows have no task-owned fields outside the framework request."""

    model_config = ConfigDict(extra="allow")
