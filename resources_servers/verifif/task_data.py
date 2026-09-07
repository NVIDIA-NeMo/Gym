# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-data schema for the verifif server.

Multi-language verifiable instruction following. All task fields ride at the row top level and
mirror ``TuringVIFRunRequest`` (app.py). ``instructions`` is DELIBERATELY open: each item carries
``instruction_id`` plus per-instruction kwargs (observed across committed rows: end_phrase,
forbidden_words, frequency, keyword, num_words, relation, start_phrase, and envelope keys
uid/source/is_misalignment_check) that verify() validates with a hand-rolled
``validate_instructions_schema`` against ``EXPECTED_ARGUMENTS`` — intentionally outside Pydantic,
so this schema keeps it ``List[Dict[str, Any]]`` and does not chase closure. ``llm_judge`` items
replicate app.py's ``LLMJudgeItem`` wire shape (``pass_criteria`` is wire-defaulted to "YES" and
absent from committed rows). ``language`` codes in data are non-uniform ('fre', 'pt-BR', ...);
verify() falls back to 'en' for unsupported codes.
"""

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class LLMJudgeItem(BaseModel):
    """One custom LLM-judge question; mirrors app.py's ``LLMJudgeItem`` wire model."""

    model_config = ConfigDict(extra="allow")

    uid: int = Field(description="Judge-question identifier, unique within the row.")
    content: str = Field(description="The judge question, written in the row's language.")
    pass_criteria: Literal["YES", "NO"] = Field(
        default="YES",
        description=(
            "Expected judge verdict for the response to pass ('YES': judge must answer YES; 'NO': "
            "judge must answer NO). Wire-defaulted; absent from all committed rows."
        ),
    )
    source: Literal["user", "system"] = Field(description="Which conversation role the judged criterion comes from.")
    is_misalignment_check: bool = Field(description="Whether this check probes misalignment rather than quality.")


class TaskData(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: int = Field(
        default=0,
        description="Row identifier (int in committed data, e.g. 54585); wire-defaulted to 0.",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    instructions: List[Dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "Programmatically-checkable instructions. Each item has instruction_id (e.g. "
            "'startend:start_checker', 'keywords:frequency') plus per-instruction_id kwargs validated at "
            "runtime by validate_instructions_schema/EXPECTED_ARGUMENTS, and envelope keys uid/source/"
            "is_misalignment_check read defensively via .get. Open by design; do not tighten."
        ),
        json_schema_extra={"consumed_by": ["verify"]},
    )
    llm_judge: List[LLMJudgeItem] = Field(
        default_factory=list,
        description="Custom LLM-judge questions scored alongside the programmatic instruction checks.",
        json_schema_extra={"consumed_by": ["verify"]},
    )
    prompt: Optional[str] = Field(
        default=None,
        description="The original user prompt; typed on the wire but absent from all committed rows and unread.",
        json_schema_extra={"consumed_by": ["prompt"]},
    )
    language: str = Field(
        default="en",
        description=(
            "Language code for multi-language validation. Committed values are non-uniform ('fre', 'es', "
            "'de', 'it', 'pt-BR'); verify() falls back to 'en' when the code is not in SUPPORTED_LANGS."
        ),
        json_schema_extra={"consumed_by": ["verify"]},
    )
