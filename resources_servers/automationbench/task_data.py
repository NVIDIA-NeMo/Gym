# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-data schema for the automationbench server.

Rows are written by ``benchmarks/automationbench/prepare.py`` from the upstream AutomationBench task set.
``/seed_session`` validates its ``task_data`` with ``TaskData`` and builds the task's world from ``info``;
``/verify`` scores that world and echoes ``task_id``. The rows also serve the deprecated ``verifiers_agent`` recipe,
whose fields this server ignores.
"""

from typing import Any, Dict, List

from pydantic import BaseModel, ConfigDict, Field


class TaskInfo(BaseModel):
    """The upstream task: its initial world, the assertions that grade the final world, and its name."""

    model_config = ConfigDict(extra="allow")

    task_name: str = Field(description="Upstream task name, `<domain>.<task>`; its domain groups the metrics.")
    initial_state: Dict[str, Any] = Field(
        description="The WorldState the episode starts from, keyed by service (gmail, salesforce, ...)."
    )
    assertions: List[Dict[str, Any]] = Field(
        min_length=1, description="Checks on the final world; the guardrail-gated share of them is the reward."
    )
    zapier_tools: List[str] = Field(
        default_factory=list,
        description="Upstream's per-task Zapier tools. The api toolset ignores them except to scope the services.",
    )


class TaskData(BaseModel):
    model_config = ConfigDict(extra="allow")

    task_id: str = Field(
        description="`<task_name>/<example_id>`; echoed by /verify.",
        json_schema_extra={"consumed_by": ["verify", "provenance"]},
    )
    info: TaskInfo = Field(
        description="The upstream task, as verifiers' AutomationBench environment receives it.",
        json_schema_extra={"consumed_by": ["verify"]},
    )

    # Read only by config_legacy_agent.yaml's verifiers_agent.
    task_idx: int | None = None
    vf_env_id: str | None = None
    question: str = ""
    answer: str = ""
    example_id: int | str = 0
