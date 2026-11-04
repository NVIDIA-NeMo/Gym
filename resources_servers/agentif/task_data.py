# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-data schema for the agentif server (AgentIF constraint following).

Rows nest everything inside an untyped ``verifier_metadata`` bucket
(``AgentIFRunRequest`` types it ``Optional[Dict[str, Any]]``, ``extra="allow"``); the schema is
written flat with ``legacy_location`` annotations. Every field is Optional because the wire never
422s on the bucket's contents. verify() reads only ``constraints`` (app.py:316) and scores each
entry with the rule/judge checkers; the remaining fields ride along as provenance.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


_VM = {"legacy_location": "verifier_metadata"}


class TaskData(BaseModel):
    model_config = ConfigDict(extra="allow")

    constraints: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description=(
            "Constraint specs scored by verify(): each has id, desc, other_info, and the "
            "dimension/type keys the score breakdowns group by."
        ),
        json_schema_extra={"consumed_by": ["verify"], **_VM},
    )
    query_id: Optional[int] = Field(
        default=None,
        description="Source query identifier; provenance only.",
        json_schema_extra={"consumed_by": ["provenance"], **_VM},
    )
    turn_id: Optional[int] = Field(
        default=None,
        description="Turn index within the source conversation; provenance only.",
        json_schema_extra={"consumed_by": ["provenance"], **_VM},
    )
    domain: Optional[str] = Field(
        default=None,
        description="AgentIF domain the row came from, e.g. 'lawglm'; provenance only.",
        json_schema_extra={"consumed_by": ["provenance"], **_VM},
    )
    agent_name: Optional[str] = Field(
        default=None,
        description="Source agent prompt name, e.g. 'Thought_prompt_lawglm'; provenance only.",
        json_schema_extra={"consumed_by": ["provenance"], **_VM},
    )
    prompt_type: Optional[str] = Field(
        default=None,
        description="Source prompt family, e.g. 'Thought_prompt'; provenance only.",
        json_schema_extra={"consumed_by": ["provenance"], **_VM},
    )
