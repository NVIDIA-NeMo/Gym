# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-data schema for the mcqa server (parent of the MCQA family).

Ground truth rides as top-level row fields; there is no verifier_metadata. Mirrors
``MCQARunRequest`` (app.py): everything is Optional except ``grading_mode``, which carries the
wire's default and Literal set (``lenient_answer_colon_md`` is reachable only via server config,
so it is deliberately absent from the row-level Literal, exactly like the wire). Two committed
row variants share this one model: plain rows carry ``grading_mode`` explicitly; template rows
drop it (the default applies) and instead carry ``template_metadata`` + ``reward_profiles``.

``template_metadata`` stays an untyped dict on purpose: verify() only probes ``output_regex``
(str OR list[str]) and ``answer_prefix`` defensively, and feeds may add template keys freely.
``reward_profiles`` is undeclared on today's wire (silently dropped by the request model) but is
present in committed data, so it is declared here as optional provenance. Heirs: gpqa_diamond
aliases this model; bunsenbench_chemistry_mcq extends it.
"""

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class TaskData(BaseModel):
    model_config = ConfigDict(extra="allow")

    uuid: Optional[str] = Field(
        default=None,
        description="Stable task identifier; echoed through, never graded on.",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    options: Optional[List[Dict[str, Optional[str]]]] = Field(
        default=None,
        description=(
            "Answer options as a list of single-key dicts letter->text, e.g. "
            '[{"A": "Karyotyping"}, {"B": "PCR"}]. Values may be null. The keys define the '
            "allowed-letter set for answer extraction."
        ),
        json_schema_extra={"consumed_by": ["verify"]},
    )
    expected_answer: Optional[str] = Field(
        default=None,
        description="The correct option letter (e.g. 'B').",
        json_schema_extra={"consumed_by": ["verify"]},
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description=(
            "Arbitrary provenance bucket; explicitly ignored for grading "
            "(_extract_options_and_expected reads only the top-level fields)."
        ),
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    grading_mode: Literal[
        "strict_single_letter_boxed",
        "lenient_boxed",
        "lenient_answer_colon",
    ] = Field(
        default="strict_single_letter_boxed",
        description=(
            "Answer-extraction mode. Server config.grading_mode overrides the row value when set; "
            "'lenient_answer_colon_md' exists only as a config override, never in rows."
        ),
        json_schema_extra={"consumed_by": ["verify"]},
    )
    template_metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description=(
            "Prompt-template descriptor, deliberately untyped. verify() probes 'output_regex' "
            "(str or list[str] of custom extraction regexes) and 'answer_prefix'; committed rows "
            "also carry template_id/template_prompt/weight/prompt_type/format_type."
        ),
        json_schema_extra={"consumed_by": ["verify", "prompt"]},
    )
    reward_profiles: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description=(
            "Baselining pass rates per model: [{model_hf_path, num_generations, pass_rate}]. "
            "Undeclared on today's wire request model (dropped in transit); provenance only."
        ),
        json_schema_extra={"consumed_by": ["provenance"]},
    )
