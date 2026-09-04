# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-data schema for the indian_banking server.

A gymnasium-style environment (no /verify): scoring happens on the terminal step()
from the episode's session-state store. Every field below is a top-level row extra
consumed by reset()/step() or by core/reward.py.
"""

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class GoldAction(BaseModel):
    """One gold tool call; also the shape of initialization_actions."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(json_schema_extra={"consumed_by": ["verify"]})
    arguments: Dict[str, Any] = Field(default_factory=dict, json_schema_extra={"consumed_by": ["verify"]})
    action_id: Optional[str] = Field(default=None, json_schema_extra={"consumed_by": ["provenance"]})
    compare_args: Optional[List[str]] = Field(
        default=None,
        description="When set, only these argument keys are compared against the agent's call.",
        json_schema_extra={"consumed_by": ["verify"]},
    )
    expect_error: bool = Field(
        default=False,
        description="This gold call is expected to error (deliberate error-path task); "
        "an errored agent call satisfies a gold action only when this is set.",
        json_schema_extra={"consumed_by": ["verify"]},
    )


class EvaluationCriteria(BaseModel):
    model_config = ConfigDict(extra="forbid")

    actions: List[GoldAction] = Field(default_factory=list, json_schema_extra={"consumed_by": ["verify"]})
    reward_basis: List[Literal["ACTION", "DB", "COMMUNICATE", "NL_ASSERTION"]] = Field(
        json_schema_extra={"consumed_by": ["verify"]}
    )
    communicate_info: List[str] = Field(default_factory=list, json_schema_extra={"consumed_by": ["verify"]})
    nl_assertions: List[str] = Field(default_factory=list, json_schema_extra={"consumed_by": ["verify"]})
    max_tool_calls: Optional[int] = Field(
        default=None,
        description="Deterministic cap on tool calls; exceeding it fails strict (0 = conversational-only).",
        json_schema_extra={"consumed_by": ["verify"]},
    )
    require_transfer: bool = Field(
        default=False,
        description="The episode must end in transfer_to_human_agents for strict to pass.",
        json_schema_extra={"consumed_by": ["verify"]},
    )


class UserScenarioInstructions(BaseModel):
    """Envelope: the stable keys the user simulator prompt renders; extras stay open."""

    model_config = ConfigDict(extra="allow")

    domain: Optional[str] = None
    reason_for_call: Optional[str] = None
    known_info: Optional[str] = None
    unknown_info: Optional[str] = None
    task_instructions: Optional[str] = None


class UserScenario(BaseModel):
    model_config = ConfigDict(extra="forbid")

    persona: Optional[str] = Field(default=None, json_schema_extra={"consumed_by": ["prompt"]})
    instructions: UserScenarioInstructions = Field(json_schema_extra={"consumed_by": ["prompt"]})


class InitialState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    initialization_data: Dict[str, Any] = Field(
        default_factory=dict,
        description="Envelope: agent_data.active_customer selects the episode customer; "
        "user_data feeds the user simulator.",
        json_schema_extra={"consumed_by": ["verify", "prompt"]},
    )
    initialization_actions: Optional[List[GoldAction]] = Field(
        default=None,
        description="Tool calls replayed at reset() to pre-shape the episode DB; "
        "cleared from the trajectory before scoring.",
        json_schema_extra={"consumed_by": ["verify"]},
    )
    message_history: Optional[List[Dict[str, Any]]] = Field(
        default=None, json_schema_extra={"consumed_by": ["prompt"]}
    )


class TaskData(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str = Field(json_schema_extra={"consumed_by": ["metrics", "provenance"]})
    customer: str = Field(json_schema_extra={"consumed_by": ["verify"]})
    user_scenario: UserScenario = Field(json_schema_extra={"consumed_by": ["prompt", "verify"]})
    evaluation_criteria: EvaluationCriteria = Field(json_schema_extra={"consumed_by": ["verify"]})
    initial_state: InitialState = Field(json_schema_extra={"consumed_by": ["verify", "prompt"]})
    opening_message: Optional[str] = Field(
        default=None,
        description="Fixed first customer message so the first turn is identical across rollouts.",
        json_schema_extra={"consumed_by": ["prompt"]},
    )
