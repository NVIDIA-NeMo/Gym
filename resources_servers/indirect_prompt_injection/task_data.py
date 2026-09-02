# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-data schema for the indirect_prompt_injection server.

Agentic indirect-prompt-injection tasks. ``injection`` mirrors the wire ``InjectionSpec``
(app.py); ``environment`` is a deliberately OPEN, domain-specific dict (healthcare rows carry
patients/appointments/prescriptions/..., hr rows carry candidates/interviews/inbox/... — the
key set varies even within one domain) interpreted only by per-domain tool handlers, so it is
typed ``Dict[str, Any]`` and its closure is intentionally not chased. ``environment`` is consumed
at ``seed_session`` time (deep-copied per session and mutated by tool handlers), not by
``verify()`` itself.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class InjectionSpecData(BaseModel):
    """Mirror of the wire ``InjectionSpec`` (app.py), plus untyped riders seen in committed data."""

    model_config = ConfigDict(extra="allow")

    goal: str = Field(
        description="Natural-language attacker goal; wire-required but never read by verify().",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    target_tool: str = Field(
        description="Tool the injection tries to make the agent call; matched against the trace.",
        json_schema_extra={"consumed_by": ["verify"]},
    )
    target_args: Dict[str, Any] = Field(
        description=(
            "Arguments the injected tool call must carry. Which keys must match strictly depends on "
            "verification_type via STRICT_MATCH_KEYS in verifier.py; values are typically strings."
        ),
        json_schema_extra={"consumed_by": ["verify"]},
    )
    verification_type: str = Field(
        description="Selects the trace-matching rule (e.g. 'check_message_sent').",
        json_schema_extra={"consumed_by": ["verify"]},
    )
    attack_strategy: str = Field(
        description="Attack strategy label; echoed into the verify response for metrics breakdowns.",
        json_schema_extra={"consumed_by": ["metrics"]},
    )
    attack_difficulty: str = Field(
        description="Attack difficulty label; echoed into the verify response for metrics breakdowns.",
        json_schema_extra={"consumed_by": ["metrics"]},
    )
    vector: str = Field(
        description="Injection vector label; wire-required but never read by verify().",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    injection_text: str = Field(
        description="Raw injected text planted in the environment; wire-required but never read by verify().",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    category: str = Field(
        description="Attack category label; echoed into the verify response for metrics breakdowns.",
        json_schema_extra={"consumed_by": ["metrics"]},
    )
    injection_position: Optional[str] = Field(
        default=None,
        description="Where in the environment the injection was planted; not typed on the wire model, unread.",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    subtlety_tier: Optional[int] = Field(
        default=None,
        description="Subtlety tier of the injection; not typed on the wire model, unread.",
        json_schema_extra={"consumed_by": ["provenance"]},
    )


class TaskData(BaseModel):
    model_config = ConfigDict(extra="allow")

    injection: InjectionSpecData = Field(
        description="Injection specification; wire-required by IPIVerifyRequest.",
        json_schema_extra={"consumed_by": ["verify", "metrics"]},
    )
    required_tools: List[str] = Field(
        default_factory=list,
        description="Tools the agent must call for the utility reward (reward_utility).",
        json_schema_extra={"consumed_by": ["verify"]},
    )
    environment: Dict[str, Any] = Field(
        description=(
            "Free-form domain-specific environment state seeded per session (IPISeedSessionRequest); "
            "deep-copied and mutated by per-domain tool handlers at rollout time, never read by verify(). "
            "Shape is open by design and varies across (and within) domains."
        ),
        json_schema_extra={"consumed_by": ["prompt"]},
    )
    domain: Optional[str] = Field(
        default=None,
        description="Task domain (e.g. 'healthcare', 'hr'); never read by server code.",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    environment_type: Optional[str] = Field(
        default=None,
        description="Environment variant marker (e.g. 'R1'); never read by server code.",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    prompt_mode: Optional[str] = Field(
        default=None,
        description="Prompting mode marker (e.g. 'agentic'); never read by server code.",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    verifier_config: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Verifier config marker (e.g. {'type': 'trace_analysis', 'mode': 'agentic_ipi'}); unread.",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
