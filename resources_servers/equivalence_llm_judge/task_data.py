# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-data schema for the equivalence_llm_judge server.

LLM-judge equivalence grading over top-level fields (no ``verifier_metadata`` bucket). Four
committed row shapes share one open envelope, so this is a single permissive model, not a union:
data/example.jsonl {question, expected_answer}; data/example_nl2bash.jsonl {expected_answer};
data/example_openqa.jsonl adds {uuid, reward_profiles, template_metadata};
data/example_prepare.jsonl adds agent_ref. Required-ness mirrors ``LLMJudgeRunRequest``
(app.py:113, ``extra="allow"``): every task field is wire-Optional. ``template_metadata`` is a
known risk field — consumed by ``verify()`` yet UNDECLARED on the wire model, reaching code only
via ``extra="allow"`` + ``hasattr`` (app.py:305/322/368); its subkeys stay an open dict here on
purpose. The judge question comes from ``responses_create_params.input`` (last user message), not
from the row's ``question`` field.
"""

from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field


class TaskData(BaseModel):
    model_config = ConfigDict(extra="allow")

    expected_answer: Optional[str] = Field(
        default=None,
        description=(
            "Ground-truth answer the judge compares the model answer against; the only field grading "
            "needs. When falsy, verify() falls back to metadata.expected_answer (_extract_expected_answer, "
            "app.py:207); if neither is present it silently judges against an empty expected answer "
            '(app.py:426 `or ""`) rather than erroring.'
        ),
        json_schema_extra={"consumed_by": ["verify"]},
    )
    question: Optional[str] = Field(
        default=None,
        description=(
            "Ride-along question text; never read — the judge prompt's question is re-extracted from the "
            "last user message in responses_create_params.input."
        ),
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    uuid: Optional[Union[str, int]] = Field(
        default=None,
        description="Task identifier (openqa rows); declared on the wire but never read.",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    options: Optional[List[Dict[str, str]]] = Field(
        default=None,
        description=(
            "MCQA-compatibility field declared on LLMJudgeRunRequest but never read; absent from all committed rows."
        ),
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description=(
            "Fallback grading bucket: verify() reads metadata['expected_answer'] when the top-level "
            "expected_answer is falsy. Absent from all committed rows."
        ),
        json_schema_extra={"consumed_by": ["verify"]},
    )
    template_metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description=(
            "Prompt-template context on openqa rows: {template_id, template_prompt, output_regex, weight, "
            "prompt_type, format_type}. verify() reads only 'output_regex' (per-record answer-extraction "
            "override when config.use_per_record_regex) — via hasattr + .get because the wire model never "
            "declares this field. Kept an open dict; do not close its subkeys."
        ),
        json_schema_extra={"consumed_by": ["verify"]},
    )
    reward_profiles: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description=(
            "Per-model pass-rate profile entries on openqa rows ({model_hf_path, num_generations, "
            "pass_rate}); never read by the server."
        ),
        json_schema_extra={"consumed_by": ["provenance"]},
    )
