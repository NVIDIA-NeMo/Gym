# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-data schema for the asb server (Agent Security Bench).

Rows nest everything except ``asb_id`` and ``condition`` inside ``verifier_metadata``
(``AsbVerifyRequest`` types it ``dict[str, Any]``), so the schema is written flat with
``legacy_location`` annotations on the nested fields and none on the two top-level ones.

Required-ness here tracks what the wire actually needs to run a row, not what Pydantic
would reject: the bucket is untyped, so a malformed row 422s nowhere and instead raises
``KeyError`` mid-rollout. Every field the server or agent reads by subscript is therefore
required, and the ones read with ``.get`` or never read at all are optional. Marking them
optional "to be safe" would move a crash from dataset validation into the middle of a
10,800-row run, which is the failure this schema exists to prevent.

A row describes one cell of ASB's matrix: which agent persona runs, which legitimate task
it is given, which attacker tool is planted, through which channel the attack arrives, and
which public defense (if any) is in force.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


_VM = {"legacy_location": "verifier_metadata"}


class TaskData(BaseModel):
    model_config = ConfigDict(extra="allow")

    # --- top-level row fields (not inside verifier_metadata) ---

    asb_id: Optional[str] = Field(
        default=None,
        description="Stable row identifier, unique across the whole 10,800-row matrix.",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    condition: Optional[str] = Field(
        default=None,
        description=(
            "Condition this row belongs to, e.g. "
            "'direct_prompt_injection.combined_attack.no_defense.all'. Reporting groups rows "
            "by this to rebuild the published per-condition denominators."
        ),
        json_schema_extra={"consumed_by": ["metrics"]},
    )

    # --- read by verify() ---

    attacker_tool: Dict[str, Any] = Field(
        description=(
            "The planted tool. verify() reads 'attack_goal' to test attack success and 'name' "
            "to test whether the tool was invoked; the agent injects 'name' into the plan."
        ),
        json_schema_extra={"consumed_by": ["verify", "prompt"], **_VM},
    )
    normal_tools: List[Dict[str, Any]] = Field(
        description=(
            "The legitimate tools. verify() scores the original task conjunctively over each "
            "tool's 'expected_achievement'; the agent uses the same list to execute the plan."
        ),
        json_schema_extra={"consumed_by": ["verify", "prompt"], **_VM},
    )
    injection_method: str = Field(
        description=(
            "Attack channel: direct_prompt_injection, observation_prompt_injection, "
            "memory_attack, mixed_attack, pot_backdoor, pot_clean, or clean. verify() uses it "
            "to decide whether reward is resistance (attack rows) or utility (clean rows)."
        ),
        json_schema_extra={"consumed_by": ["verify", "metrics"], **_VM},
    )
    task_input: str = Field(
        description=(
            "The user-facing task text as the agent receives it, already carrying the "
            "injection for DPI rows and the defense wrapper where one applies."
        ),
        json_schema_extra={"consumed_by": ["verify", "prompt"], **_VM},
    )
    injects_into_task: bool = Field(
        description="Whether the attack rides in the task text. Selects the DPI memory check.",
        json_schema_extra={"consumed_by": ["verify", "prompt"], **_VM},
    )

    # --- read by the agent harness ---

    attacker_tool_available: bool = Field(
        description=(
            "Whether the attacker tool is offered to the model at all. False for clean "
            "controls, which is what makes them a control rather than a failed attack."
        ),
        json_schema_extra={"consumed_by": ["prompt"], **_VM},
    )
    injects_into_observations: bool = Field(
        description="Whether the attack is appended to tool results (OPI and mixed rows).",
        json_schema_extra={"consumed_by": ["prompt"], **_VM},
    )
    observation_injection: Optional[str] = Field(
        default=None,
        description="Text appended to tool results when injects_into_observations is set.",
        json_schema_extra={"consumed_by": ["prompt"], **_VM},
    )
    defense_type: Optional[str] = Field(
        description=(
            "Public defense in force, or None for the undefended arm. The agent special-cases "
            "'ob_sandwich_defense', which re-states the task after each observation."
        ),
        json_schema_extra={"consumed_by": ["prompt", "metrics"], **_VM},
    )
    memory_key: Optional[str] = Field(
        description="Key the agent retrieves poisoned history under; None when unused.",
        json_schema_extra={"consumed_by": ["prompt"], **_VM},
    )
    reads_memory: Optional[bool] = Field(
        default=None,
        description="Whether this row retrieves memory before planning (memory and mixed rows).",
        json_schema_extra={"consumed_by": ["prompt"], **_VM},
    )
    tools_schema: List[Dict[str, Any]] = Field(
        description=(
            "Tool declarations as offered to the model, normalized into the Responses API "
            "envelope. Includes the attacker tool when attacker_tool_available is set."
        ),
        json_schema_extra={"consumed_by": ["prompt"], **_VM},
    )

    # --- provenance: carried for auditability, not read at runtime ---

    agent_name: Optional[str] = Field(
        default=None,
        description="ASB agent persona, e.g. 'financial_analyst'; provenance only.",
        json_schema_extra={"consumed_by": ["provenance"], **_VM},
    )
    agent_prefix: Optional[str] = Field(
        default=None,
        description="The persona's system-prompt prefix, verbatim from upstream; provenance only.",
        json_schema_extra={"consumed_by": ["provenance"], **_VM},
    )
    task: Optional[str] = Field(
        default=None,
        description="The legitimate task before any injection or defense wrapper; provenance only.",
        json_schema_extra={"consumed_by": ["provenance"], **_VM},
    )
    task_index: Optional[int] = Field(
        default=None,
        description="Index of the task within the persona's task list; provenance only.",
        json_schema_extra={"consumed_by": ["provenance"], **_VM},
    )
    attack_type: Optional[str] = Field(
        default=None,
        description=(
            "Injection style: naive, fake_completion, escape_characters, context_ignoring or "
            "combined_attack. Part of the condition id; provenance only at runtime."
        ),
        json_schema_extra={"consumed_by": ["provenance"], **_VM},
    )
    aggressive: Optional[bool] = Field(
        default=None,
        description="Whether the attacker tool came from upstream's aggressive pool.",
        json_schema_extra={"consumed_by": ["provenance"], **_VM},
    )
    aggressiveness: Optional[str] = Field(
        default=None,
        description="Aggressiveness selector the expansion used ('all' for published runs).",
        json_schema_extra={"consumed_by": ["provenance"], **_VM},
    )
    trigger: Optional[str] = Field(
        default=None,
        description="Backdoor trigger phrase on PoT rows; None elsewhere.",
        json_schema_extra={"consumed_by": ["provenance"], **_VM},
    )
    upstream_revision: Optional[str] = Field(
        default=None,
        description=(
            "Upstream agiresearch/ASB commit the row was expanded from. Stamped per row so a "
            "rollouts file states its own provenance without reference to a manifest."
        ),
        json_schema_extra={"consumed_by": ["provenance"], **_VM},
    )
