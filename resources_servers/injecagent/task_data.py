# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-data schema for the injecagent server (InjecAgent base setting, indirect prompt injection).

Fields are top-level row columns written by ``benchmarks/injecagent/prepare.py`` from the pinned
upstream test cases. Required-ness mirrors the wire models in ``app.py``: ``seed_session`` reads the
user-tool fields, ``verify`` reads the attack fields, and ``compute_metrics`` groups on ``attack_kind``
and ``attack_type``. The poisoned tool result and the injected instruction live in
``responses_create_params.input`` (as ``function_call`` and ``function_call_output`` items) and are
mirrored here only so the row is self-describing.
"""

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class TaskData(BaseModel):
    model_config = ConfigDict(extra="allow")

    injecagent_id: str = Field(
        description=(
            "Identifies one upstream test case: `<attack_kind>-<row>`, e.g. `direct_harm-0001` or "
            "`data_stealing-0544`. Row numbers follow the order of the upstream JSON files."
        ),
        json_schema_extra={"consumed_by": ["verify", "provenance"]},
    )
    attack_kind: Literal["direct_harm", "data_stealing"] = Field(
        description=(
            "Which upstream family the case belongs to. Direct harm scores one attacker tool at step 1; "
            "data stealing scores a data-access tool at step 1 and an exfiltration tool at step 2."
        ),
        json_schema_extra={"consumed_by": ["verify", "metrics"]},
    )
    attack_type: str = Field(
        description=(
            "Upstream's attack category: Financial Harm, Physical Harm or Data Security Harm for direct "
            "harm; Financial Data, Physical Data or Others for data stealing. compute_metrics slices on it."
        ),
        json_schema_extra={"consumed_by": ["verify", "metrics"]},
    )
    attacker_tools: List[str] = Field(
        description=(
            "The attacker tool names in scoring order: one for direct harm, two (data access, then "
            "exfiltration) for data stealing. The first tool call of each step is compared against them."
        ),
        json_schema_extra={"consumed_by": ["verify", "seed_session"]},
    )
    user_tool: str = Field(
        description=(
            "The tool the user asked for, whose poisoned result is the seed of the case. A re-run of it "
            "counts as a valid non-attack only when the visible text says 'again' or 're-run'."
        ),
        json_schema_extra={"consumed_by": ["verify", "seed_session"]},
    )
    user_tool_arguments: Dict[str, Any] = Field(
        description=(
            "The arguments of the seed user-tool call, as a JSON object. Upstream sends them as a JSON "
            "string holding a Python dict repr; the values are unchanged. A repeated call with the same "
            "arguments returns the same poisoned result."
        ),
        json_schema_extra={"consumed_by": ["seed_session"]},
    )
    user_tool_response: str = Field(
        description=(
            "The poisoned tool result (upstream `Tool Response`, JSON-encoded) that carries the injected "
            "instruction. Returned verbatim when the model re-runs the user tool."
        ),
        json_schema_extra={"consumed_by": ["seed_session"]},
    )
    setting: str = Field(
        default="base",
        description="Upstream setting: only `base` (no override prefix) is ported.",
        json_schema_extra={"consumed_by": ["verify", "provenance"]},
    )
    attacker_instruction: Optional[str] = Field(
        default=None,
        description="The injected instruction text, for reading rollouts; never read by the verifier.",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    expected_achievements: Optional[str] = Field(
        default=None,
        description="Upstream's description of what the attacker gains; never read by the verifier.",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    protocol: Optional[str] = Field(
        default=None,
        description="Always `upstream_finetuned_function_calling`: the upstream protocol the row reproduces.",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    upstream_revision: Optional[str] = Field(
        default=None,
        description="The upstream git revision the row was built from, so a rollout names the data it scored.",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
