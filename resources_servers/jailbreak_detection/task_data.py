# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-data schema for the jailbreak_detection server.

Adversarial prompts judged for safe handling. Mirrors ``JailbreakDetectionRunRequest`` (app.py,
``extra="allow"``: adversarial_prompt/attack_type/metadata all Optional). ``response_policy_mapped``
is deliberately NOT typed on the wire model — verify() reads it defensively via
``(body.model_extra or {}).get(...)`` and it switches on the policy-aware verification path only
when ``policy_verifier_templates_path`` is configured — but it appears in every committed row and
gates behavior, so it is declared here as Optional.
"""

from typing import Any, Dict, Optional

from pydantic import BaseModel, ConfigDict, Field


class TaskData(BaseModel):
    model_config = ConfigDict(extra="allow")

    adversarial_prompt: Optional[str] = Field(
        default=None,
        description=(
            "The original adversarial/jailbreak prompt shown to the judge. When absent, verify() falls "
            "back to the last user message in responses_create_params.input."
        ),
        json_schema_extra={"consumed_by": ["verify"]},
    )
    attack_type: Optional[str] = Field(
        default=None,
        description="Attack categorization (e.g. 'DAN', 'roleplay'); typed on the wire but never read by verify().",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional additional metadata; typed on the wire but absent from all committed rows.",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    response_policy_mapped: Optional[str] = Field(
        default=None,
        description=(
            "Response-policy key (e.g. 'hard_refusal_no_redirection') that dispatches to the "
            "policy-specific verifier templates when policy_verifier_templates_path is configured. "
            "Reaches server code only via model_extra on today's wire."
        ),
        json_schema_extra={"consumed_by": ["verify"]},
    )
