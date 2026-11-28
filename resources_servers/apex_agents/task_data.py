# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-data schema for the apex_agents server.

One APEX-Agents task from mercor/apex-agents, as benchmarks/apex_agents/prepare.py writes it: the task and
world identity at the top level, and the held-out grading fields inside ``verifier_metadata``, which the
agent never sends into the sandbox. Required-ness mirrors the wire: ``task_id`` and ``world_id`` are
required by ``ApexSeedSessionRequest`` (this server's app.py) and ``ApexAgentRunRequest``
(responses_api_agents/apex_agent/app.py), and ``rubric`` by ``ApexVerifierMetadata``.
"""

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


_LEGACY = "verifier_metadata"


class TaskData(BaseModel):
    model_config = ConfigDict(extra="allow")

    task_id: str = Field(
        pattern=r"^task[-_][A-Za-z0-9_-]+$",
        description="Upstream APEX-Agents task id; selects the task's input files and names its artifacts.",
        json_schema_extra={"consumed_by": ["verify", "provenance"]},
    )
    world_id: str = Field(
        pattern=r"^world_[0-9a-f]{32}$",
        description="Archipelago world the task runs in; selects the cached world ZIP restored into the sandbox.",
        json_schema_extra={"consumed_by": ["verify"]},
    )
    task_input_files: Optional[str] = Field(
        default=None,
        pattern=r"^snap_[0-9a-f]{32}$",
        description="Snapshot id of task-specific attachments overlaid on the world; null when the task has none.",
    )
    domain: Optional[str] = Field(
        default=None,
        description="Professional domain (investment banking, management consulting, corporate law).",
        json_schema_extra={"consumed_by": ["metrics", "provenance"]},
    )
    foundry_services: List[str] = Field(
        default_factory=list,
        description="Optional gateway services the world needs (edgar, fmp); empty for the public 452-task set.",
    )
    runtime_mode: Literal["world_zip", "prebuilt_world"] = Field(
        default="world_zip",
        description="world_zip restores the public world ZIP; prebuilt_world runs a private delivery's world image.",
    )
    task_slug: Optional[str] = Field(
        default=None,
        pattern=r"^[a-z0-9][a-z0-9-]*$",
        description="Task name inside a prebuilt world; prebuilt_world rows only.",
    )
    rubric: List[Dict[str, Any]] = Field(
        description="Binary rubric criteria, each with its grading target; the judge scores every criterion.",
        json_schema_extra={"legacy_location": _LEGACY, "consumed_by": ["verify"]},
    )
    scoring: Literal["all_pass", "weighted_average"] = Field(
        default="all_pass",
        description="all_pass gives reward 1 only when every criterion passes (APEX-Agents Pass@1).",
        json_schema_extra={"legacy_location": _LEGACY, "consumed_by": ["verify"]},
    )
    expected_output: Optional[str] = Field(
        default=None,
        description="Upstream expected output kind (message_in_console, make_new_sheet, ...).",
        json_schema_extra={"legacy_location": _LEGACY, "consumed_by": ["verify"]},
    )
    gold_response: Optional[str] = Field(
        default=None,
        description="Upstream reference answer; held out from the model.",
        json_schema_extra={"legacy_location": _LEGACY, "consumed_by": ["verify"]},
    )
    gold_response_type: Optional[str] = Field(
        default=None,
        json_schema_extra={"legacy_location": _LEGACY, "consumed_by": ["provenance"]},
    )
    task_name: Optional[str] = Field(
        default=None,
        json_schema_extra={"legacy_location": _LEGACY, "consumed_by": ["provenance"]},
    )
