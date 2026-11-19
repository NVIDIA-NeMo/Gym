# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-data schema for the self-contained harbor_agent (no resources server).

``HarborAgentRunRequest`` (app.py) wire-requires ``instance_id``; run() splits it on ``::`` to
pick the Harbor dataset alias and task, launches the Harbor trial, and names the output artifact
after it. Benchmarks that bridge through Harbor (legal_agent_bench, terminal_bench_2_1 input
sets) commit rows of exactly this shape.
"""

from pydantic import BaseModel, ConfigDict, Field


class TaskData(BaseModel):
    model_config = ConfigDict(extra="allow")

    instance_id: str = Field(
        description=(
            "Harbor task coordinate in the form '<dataset_alias>::<task_name>'; the alias must "
            "match a dataset configured on the agent, and the task name selects the trial."
        ),
        json_schema_extra={"consumed_by": ["run"]},
    )
