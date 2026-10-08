# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-data schema for the structured_outputs server.

All task fields are top-level (no ``verifier_metadata``) and mirror
``StructuredOutputsVerifyRequest`` (app.py:49): ``schema_str`` is the only wire-required field.
Datasets are heterogeneous per generation — v3 rows carry only the first eight metadata fields in
text-response mode, while v4 rows add sixteen tool-call-related fields with
``response_mode='tool_call'`` — but there is no usable in-band discriminator (``response_mode`` is
Optional with a ``'text'`` default and absent from v3 rows, so a discriminated union would reject
them). A single model with wire-mirrored Optionals covers both generations. Only ``schema_str``,
``schema_type``, ``response_mode``, ``tool_name`` and ``tool_payload_key`` drive ``verify()``;
most other fields are echoed and read by ``compute_metrics`` via ``r.get(...)`` for per-slice
mean-reward metrics (``schema_type`` and ``response_mode`` are also sliced on), except
``source_format``, ``num_turns``, ``source_schema_type``, ``num_distractors`` and
``source_record_id``, which are provenance only — ``compute_metrics`` never reads them.
"""

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class TaskData(BaseModel):
    model_config = ConfigDict(extra="allow")

    schema_str: str = Field(
        description="JSON Schema as a JSON-encoded string; strictified then validated against the model output.",
        json_schema_extra={"consumed_by": ["verify"]},
    )
    schema_type: Literal["json", "yaml", "xml", "toml", "csv"] = Field(
        default="json",
        description="Serialization format the model output is parsed as before schema validation.",
        json_schema_extra={"consumed_by": ["verify", "metrics"]},
    )
    problem_type: Optional[str] = Field(
        default=None,
        description="Problem-family label (e.g. 'multistep_related', 'direct_tool_call').",
        json_schema_extra={"consumed_by": ["metrics"]},
    )
    schema_repr: Optional[str] = Field(
        default=None,
        description="How the schema was presented in the prompt (e.g. 'json', 'tool').",
        json_schema_extra={"consumed_by": ["metrics"]},
    )
    source_format: Optional[str] = Field(
        default=None,
        description="Source schema serialization format; null in some v3 rows.",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    num_turns: Optional[int] = Field(
        default=None,
        description="Number of conversation turns in the prepared prompt.",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    source_record_id: Optional[str] = Field(
        default=None,
        description="Identifier of the source SDG record (e.g. 'DS1-786FB3AE').",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    response_mode: Optional[str] = Field(
        default="text",
        description="'text' (default) validates output_text; 'tool_call' extracts function_call arguments.",
        json_schema_extra={"consumed_by": ["verify", "metrics"]},
    )
    tool_name: Optional[str] = Field(
        default=None,
        description="Expected function_call name (tool-call mode only).",
        json_schema_extra={"consumed_by": ["verify"]},
    )
    tool_schema_mode: Optional[str] = Field(
        default=None,
        description="How the target schema was embedded in the tool definition (v4 generation knob).",
        json_schema_extra={"consumed_by": ["metrics"]},
    )
    tool_payload_key: Optional[str] = Field(
        default=None,
        description="Key to unwrap from parsed tool arguments before schema validation (tool-call mode).",
        json_schema_extra={"consumed_by": ["verify"]},
    )
    tool_choice: Optional[str] = Field(
        default=None,
        description="tool_choice setting used when generating the prompt (v4 generation knob).",
        json_schema_extra={"consumed_by": ["metrics"]},
    )
    parallel_tool_calls: Optional[bool] = Field(
        default=None,
        description="parallel_tool_calls setting used when generating the prompt (v4 generation knob).",
        json_schema_extra={"consumed_by": ["metrics"]},
    )
    source_schema_type: Optional[str] = Field(
        default=None,
        description="Serialization format of the source schema record (v4 generation knob).",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    num_tools: Optional[int] = Field(
        default=None,
        description="Total number of tools presented, including distractors (v4 generation knob).",
        json_schema_extra={"consumed_by": ["metrics"]},
    )
    num_distractors: Optional[int] = Field(
        default=None,
        description="Number of distractor tools presented (v4 generation knob).",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    has_distractors: Optional[bool] = Field(
        default=None,
        description="Whether distractor tools were presented (v4 generation knob).",
        json_schema_extra={"consumed_by": ["metrics"]},
    )
    instruction_layout: Optional[str] = Field(
        default=None,
        description="Prompt layout variant (v4 generation knob).",
        json_schema_extra={"consumed_by": ["metrics"]},
    )
    instruction_detail_level: Optional[str] = Field(
        default=None,
        description="Instruction verbosity variant (v4 generation knob).",
        json_schema_extra={"consumed_by": ["metrics"]},
    )
    system_instruction_style: Optional[str] = Field(
        default=None,
        description="System-prompt style variant (v4 generation knob).",
        json_schema_extra={"consumed_by": ["metrics"]},
    )
    tool_name_style: Optional[str] = Field(
        default=None,
        description="Tool naming style variant (v4 generation knob).",
        json_schema_extra={"consumed_by": ["metrics"]},
    )
    distractor_style: Optional[str] = Field(
        default=None,
        description="Distractor-tool style variant (v4 generation knob).",
        json_schema_extra={"consumed_by": ["metrics"]},
    )
    tool_union_mode: Optional[str] = Field(
        default=None,
        description="Tool schema union-mode variant (v4 generation knob); null in all committed v4 rows.",
        json_schema_extra={"consumed_by": ["metrics"]},
    )
