# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-data schema for the swerl_llm_judge server.

All task fields are top-level (no ``verifier_metadata``) and mirror ``SWEJudgeRunRequest``
(app.py:33), where everything is Optional except ``grading_mode``'s default. ``verify()`` parses
a ``<solution>`` block from the last assistant message and compares the extracted letter(s)
against ``expected_answer``, with the allowed letters taken from the keys of ``options`` — a list
of SINGLE-KEY dicts (e.g. ``[{"A": text}, {"B": text}]``), not one mapping.
``instance_id``/``dataset_name``/``dataset_split``/``metadata`` are pure passthrough.
"""

from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field


class TaskData(BaseModel):
    model_config = ConfigDict(extra="allow")

    instance_id: Optional[str] = Field(
        default=None,
        description="Source SWE instance identifier; passthrough only.",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    dataset_name: Optional[str] = Field(
        default=None,
        description="Source dataset name; passthrough only.",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    dataset_split: Optional[str] = Field(
        default=None,
        description="Source dataset split; passthrough only.",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    expected_answer: Optional[Union[str, List[str]]] = Field(
        default=None,
        description="Correct option letter(s); normalized to uppercase single letters by verify().",
        json_schema_extra={"consumed_by": ["verify"]},
    )
    options: Optional[List[Dict[str, str]]] = Field(
        default=None,
        description="Answer choices as a list of single-key dicts; the keys define the allowed letters.",
        json_schema_extra={"consumed_by": ["verify"]},
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Untyped passthrough ({'repo': str} in committed data); never read by verify().",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    grading_mode: Literal["lenient", "strict"] = Field(
        default="lenient",
        description="'strict' requires a well-formed <solution> block; 'lenient' also accepts bare letters.",
        json_schema_extra={"consumed_by": ["verify"]},
    )
