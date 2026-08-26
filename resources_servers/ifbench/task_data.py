# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-data schema for the ifbench server.

Ground truth rides as top-level row fields; there is no verifier_metadata (and no agent_ref in
committed rows). Mirrors ``IFBenchRunRequest`` (app.py): ``id``, ``instruction_id_list``,
``prompt`` and ``kwargs`` are wire-required; ``grading_mode`` carries the wire's Literal set and
'fraction' default.

The heterogeneity lives inside ``kwargs`` and is deliberately left open: each element is the
constructor-kwargs dict for the parallel instruction in ``instruction_id_list``, serialized in
committed data as the full union of 42 nullable keys (identical keyset every row) of which only
the per-instruction-family subset is non-null (observed non-null: keyword1..keyword5 str, N
float, percentage float; the other 34 keys — forbidden_words, num_words, end_phrase,
prompt_to_repeat, section_spliter, ... — are always null in committed data, so their non-null
types are unknowable from it). verify() filters nulls defensively ({k: v for k, v in
(kwargs or {}).items() if v is not None}) and tolerates null list entries, so elements stay
``Any`` exactly like the wire's bare ``List`` — do not chase closure here. The awkward field name
``kwargs`` is load-bearing wire format and must be kept verbatim.
"""

from typing import Any, List, Literal

from pydantic import BaseModel, ConfigDict, Field


class TaskData(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: int = Field(
        description=(
            "Numeric task identifier. Required by the wire request model but never read by "
            "verify logic — identity/passthrough only."
        ),
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    instruction_id_list: List[str] = Field(
        description=(
            "Instruction registry ids (e.g. 'count:keywords_multiple') resolved through "
            "INSTRUCTION_DICT; parallel to `kwargs`."
        ),
        json_schema_extra={"consumed_by": ["verify"]},
    )
    prompt: str = Field(
        description=(
            "Original prompt text. Fed back to repeat:* instructions whose get_instruction_args() includes 'prompt'."
        ),
        json_schema_extra={"consumed_by": ["verify", "prompt"]},
    )
    kwargs: List[Any] = Field(
        description=(
            "Per-instruction constructor kwargs, parallel to `instruction_id_list`. Elements are "
            "dicts (in committed data: the full 42-nullable-key union with only the "
            "instruction-family subset non-null) or null; nulls are filtered before "
            "build_description(**kwargs). Deliberately untyped to mirror the wire's bare List."
        ),
        json_schema_extra={"consumed_by": ["verify"]},
    )
    grading_mode: Literal["binary", "fraction"] = Field(
        default="fraction",
        description=(
            "'binary' = reward 1.0 only when all instructions pass; 'fraction' = mean over "
            "per-instruction pass booleans. All committed rows use 'fraction'."
        ),
        json_schema_extra={"consumed_by": ["verify"]},
    )
