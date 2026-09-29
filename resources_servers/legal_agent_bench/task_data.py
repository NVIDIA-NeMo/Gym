# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-data schema for the legal_agent_bench server (pointer rows).

The row alone is NOT the task: it only points at a Harbor task. The actual instruction lives in
the Harbor task assets (instruction.md carrying a '<!-- lab_task_id:... -->' marker) prepared by
this server under ``config.harbor_tasks_cache_dir``/``config.harbor_tasks_dir``, and
``responses_create_params.input`` is an empty list. The server's own verify() is a stub (bare
``BaseVerifyRequest``, always reward=0.0) — Harbor executes the task-local verifier and the agent
bridge returns its reward. ``instance_id`` is required by the shared harbor agent's wire model
(``HarborRunRequest`` at responses_api_agents/harbor_agent/app.py, ``instance_id: str``), which
parses it as '<dataset_alias>::<task_name>' to select the Harbor task.
"""

from pydantic import BaseModel, ConfigDict, Field


class TaskData(BaseModel):
    model_config = ConfigDict(extra="allow")

    instance_id: str = Field(
        description=(
            "Harbor task pointer in the form 'legal_agent_bench::<task_name>'. Required by the harbor "
            "agent's HarborRunRequest (not by this server's stub verify wire); selects the out-of-row "
            "Harbor task whose assets supply both the instruction and the task-local verifier."
        ),
        json_schema_extra={"consumed_by": ["prompt", "verify"]},
    )
