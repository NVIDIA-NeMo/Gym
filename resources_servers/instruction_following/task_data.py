# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-data schema for the instruction_following server (IFEval-style).

Two dataset shapes coexist by design: older blends (including the committed example.jsonl) keep
``instruction_id_list``/``prompt``/``kwargs``/``grading_mode`` at the row top level, while the
current format nests them under ``verifier_metadata`` (the server's ``_migrate_legacy_metadata``
before-validator accepts both). This flat schema validates both generations unchanged, because
core validation splices ``verifier_metadata`` contents up before checking. Shared English fields
have no ``legacy_location`` marker because the wire accepts both placements. Indic's ``language``
is read only from ``verifier_metadata`` and therefore carries that marker. Required-ness mirrors
``InstructionFollowingRunRequest``'s after-validator, which rejects rows missing
``instruction_id_list``/``prompt``/``kwargs`` from the merged metadata.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, JsonValue


class TaskData(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: int = Field(
        description="Task identifier; wire-required int, pass-through only.",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    task_id: Optional[str] = Field(
        default=None,
        description="Stable upstream task identifier, when the benchmark provides one (IFEval: the source `key`).",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    instruction_id_list: List[str] = Field(
        description=(
            "Registry keys of the verifiable instructions to check "
            "(e.g. 'length_constraints:nth_paragraph_first_word')."
        ),
        json_schema_extra={"consumed_by": ["verify"]},
    )
    prompt: str = Field(
        description=(
            "Original instruction prompt (duplicates the user turn in responses_create_params.input). "
            "Required by the wire after-validator and passed to the Indic checker."
        ),
        json_schema_extra={"consumed_by": ["verify", "provenance"]},
    )
    kwargs: List[Optional[Dict[str, Any]]] = Field(
        description=(
            "Per-instruction constructor args, aligned with instruction_id_list. Key sets are "
            "heterogeneous per instruction type (e.g. {last_word}, {num_paragraphs, nth_paragraph, "
            "first_word}, {N, relation}); entries may be null or {}, and None values inside a dict "
            "are filtered out by verify()."
        ),
        json_schema_extra={"consumed_by": ["verify"]},
    )
    grading_mode: Optional[str] = Field(
        default=None,
        description="'binary' (all instructions must pass) or 'fraction'; verify() defaults to 'binary'.",
        json_schema_extra={"consumed_by": ["verify"]},
    )
    language: Optional[str] = Field(
        default=None,
        description="Language code required by the Indic backend to select checkers and group accuracy metrics.",
        json_schema_extra={"consumed_by": ["verify", "metrics"], "legacy_location": "verifier_metadata"},
    )
    uuid: Optional[str] = Field(
        default=None,
        description="Stable Indic task identifier combining the source language and source key.",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    subset_for_metrics: Optional[str] = Field(
        default=None,
        description="Language label used to group Indic dataset metrics.",
        json_schema_extra={"consumed_by": ["metrics"]},
    )
    metadata: Optional[Dict[str, JsonValue]] = Field(
        default=None,
        description="Indic source dataset, split, key, quality tags, and grading-harness provenance.",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
