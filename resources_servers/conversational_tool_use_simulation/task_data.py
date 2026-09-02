# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-data schema for the conversational_tool_use_simulation server.

Stateful session server: the verify request (``ConversationalToolUseVerifyRequest``) carries no
task fields — the agent forwards the whole row to POST /seed_session, so required-ness here
mirrors ``ConversationalToolUseSeedSessionRequest`` (app.py:300, extra='allow'): ``policy`` is
the only required field; ``domain_name``/``tools``/``customer_scenario``/``source_artifacts``
carry wire defaults. ``ToolSignature`` and ``CustomerScenario`` mirror the app.py models of the
same names (both extra='allow'); ToolSignature dual-sources ``doc``/``description`` and
``params``/``parameters`` — committed rows carry both spellings.

``id`` and ``metadata`` are row-level provenance the seed request swallows via extra='allow';
``initial_user_message`` and ``rollout_id`` are wire-accepted seed fields absent from committed
rows. Two committed files (example.jsonl, example_parallel_tool_calls.jsonl) share this shape,
differing only in the framework-owned ``_ng_*`` bookkeeping keys.
"""

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class ToolSignature(BaseModel):
    """One simulated tool; mirrors ``ToolSignature`` at app.py:163 (extra='allow')."""

    model_config = ConfigDict(extra="allow")

    name: str = Field(description="Tool name exposed to the agent and matched by the environment simulator.")
    description: Optional[str] = Field(
        default=None, description="Tool description; fallback when 'doc' is absent (normalized_doc)."
    )
    parameters: Optional[Dict[str, Any]] = Field(
        default=None, description="JSON-schema parameters dict; fallback when 'params' is absent."
    )
    returns: Optional[Dict[str, Any]] = Field(
        default=None, description="JSON schema of the tool's return value, used by the environment simulator."
    )
    strict: bool = Field(default=True, description="Whether the tool schema is exposed as strict.")
    doc: Optional[str] = Field(default=None, description="Preferred docstring; wins over 'description'.")
    params: Optional[Dict[str, Any]] = Field(
        default=None, description="Preferred JSON-schema parameters dict; wins over 'parameters'."
    )


class CustomerScenario(BaseModel):
    """The simulated customer; mirrors ``CustomerScenario`` at app.py:134 (extra='allow')."""

    model_config = ConfigDict(extra="allow")

    customer_persona: str = ""
    reason_for_contact: str = ""
    customer_details: str = ""
    unknown_info: Optional[str] = Field(
        default=None, description="Details the customer does NOT know; null in some committed rows."
    )
    task_instructions: str = ""
    representative_domain: Optional[str] = None
    outside_policy_scope: Optional[bool] = Field(
        default=None,
        description="Ground truth for the transfer gate in verify() when enforce_transfer_ground_truth is on.",
    )


class TaskData(BaseModel):
    model_config = ConfigDict(extra="allow")

    policy: str = Field(
        description="Customer-service policy text; drives the representative's system prompt and judge prompts.",
        json_schema_extra={"consumed_by": ["prompt", "verify"]},
    )
    domain_name: str = Field(
        default="",
        description="Business domain of the scenario (e.g. 'bookstore chains with events').",
        json_schema_extra={"consumed_by": ["prompt", "verify"]},
    )
    profile: Optional[Literal["general", "proactive"]] = Field(
        default=None,
        description="Conversation profile; selects judge behavior and is echoed into the verify result.",
        json_schema_extra={"consumed_by": ["verify"]},
    )
    tools: List[ToolSignature] = Field(
        default_factory=list,
        description="Simulated tools served to the agent and executed by the LLM environment simulator.",
        json_schema_extra={"consumed_by": ["prompt", "verify"]},
    )
    customer_scenario: CustomerScenario = Field(
        default_factory=CustomerScenario,
        description=(
            "Persona/goal spec driving the simulated user; outside_policy_scope gates the "
            "transfer ground truth in verify()."
        ),
        json_schema_extra={"consumed_by": ["prompt", "verify"]},
    )
    source_artifacts: Dict[str, Any] = Field(
        default_factory=dict,
        description="Data-gen pointers (source_name, scenario_file, ...); echoed verbatim into the verify result.",
        json_schema_extra={"consumed_by": ["verify", "provenance"]},
    )
    initial_user_message: Optional[str] = Field(
        default=None,
        description="Optional canned opening user message; wire-accepted, absent from committed rows.",
        json_schema_extra={"consumed_by": ["prompt"]},
    )
    rollout_id: Optional[str] = Field(
        default=None,
        description="Optional session-seeding identifier; wire-accepted, absent from committed rows.",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    id: Optional[str] = Field(
        default=None,
        description="Stable task identifier. Never read by server code (swallowed by extra='allow').",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description=(
            "Data-gen provenance (~20 keys: dataset_name, generator models, tool_names, "
            "scenario_file, ...). Never read by server code."
        ),
        json_schema_extra={"consumed_by": ["provenance"]},
    )
