# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-data schema for the inverse_if server.

Per-task rubric-graded LLM judging: each row can carry its OWN judge prompt template and system
prompt (falling back to config defaults when absent). All fields are Optional mirroring
``InverseIFRunRequest`` (app.py, ``extra="allow"``); ``metadata`` doubles as a fallback source for
``prompt``/``reference_response``/``rubric`` when the top-level fields are missing.
"""

from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field


class TaskData(BaseModel):
    model_config = ConfigDict(extra="allow")

    uuid: Optional[Union[str, int]] = Field(
        default=None,
        description="Non-standard row identifier (str in committed data, wire also allows int); pass-through.",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    task_id: Optional[int] = Field(
        default=None,
        description="Integer task identifier; pass-through only.",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    prompt: Optional[str] = Field(
        default=None,
        description=(
            "Original instruction prompt; fills the judge template's {prompt} placeholder. "
            "Falls back to metadata['prompt'] then ''."
        ),
        json_schema_extra={"consumed_by": ["verify"]},
    )
    reference_response: Optional[str] = Field(
        default=None,
        description=(
            "Gold/standard response; fills the judge template's {standard_response} placeholder. "
            "Falls back to metadata['reference_response'] then ''."
        ),
        json_schema_extra={"consumed_by": ["verify"]},
    )
    rubric: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description=(
            "Rubric criteria as a list of dicts shaped {'id': str, 'criteria': str}; each entry is read "
            "defensively via .get(default ''). Falls back to metadata['rubric'] then []. Wire types this "
            "List[dict], so entries stay open dicts."
        ),
        json_schema_extra={"consumed_by": ["verify"]},
    )
    judge_prompt_template: Optional[str] = Field(
        default=None,
        description=(
            "Per-task judge prompt template with {prompt}/{model_response}/{standard_response}/{criteria} "
            "placeholders (normalized by dataset_preprocess.py); falls back to "
            "config.default_judge_prompt_template."
        ),
        json_schema_extra={"consumed_by": ["verify"]},
    )
    judge_system_prompt: Optional[str] = Field(
        default=None,
        description="Per-task judge system prompt; falls back to config.default_judge_system_prompt.",
        json_schema_extra={"consumed_by": ["verify"]},
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description=(
            "Provenance dict ({'domain': str, 'l1_taxonomy': str} in committed data). Also serves as a "
            "fallback source for prompt/reference_response/rubric when the top-level fields are missing."
        ),
        json_schema_extra={"consumed_by": ["verify", "provenance"]},
    )
