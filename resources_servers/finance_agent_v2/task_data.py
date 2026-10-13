# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-data schema for the finance_agent_v2 server.

Rubric-judged finance research rows with all task fields at the top level (no ``verifier_metadata``).
Required-ness mirrors ``FinanceAgentV2RunRequest``/``FinanceAgentV2VerifyRequest`` (app.py:217/228):
``question`` defaults to '' and ``expected_answer``/``rubric`` are Optional to support unlabeled dry
runs (no rubric -> unscoreable dry run, not an error). Scoring reads only ``rubric``; the judge
question is re-extracted from ``responses_create_params.input`` and the model answer from the last
``submit_final_result`` tool call. ``question_type`` and ``expert_time_mins`` exist in the committed
data but are UNDECLARED on the wire model (which is not ``extra="allow"``), so today Pydantic's
default ``extra="ignore"`` silently drops them at the verify boundary — they are declared here as
Optional provenance so the dataset columns stop being invisible to tooling.
"""

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class TaskData(BaseModel):
    model_config = ConfigDict(extra="allow")

    question: str = Field(
        default="",
        description=(
            "Finance research question; typed on the wire and echoed into the verify response but not used "
            "for scoring — the judge re-extracts the question from the last user message in "
            "responses_create_params.input."
        ),
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    expected_answer: Optional[str] = Field(
        default=None,
        description=(
            "Expert reference answer ('A complete answer must establish...'); echoed into the verify "
            "response only, never fed to the judge (the rubric is the scoring input)."
        ),
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    rubric: Optional[str] = Field(
        default=None,
        description=(
            "The scoring input: a JSON-encoded string (stays str on the wire) of rubric criterion entries, "
            "each {'criteria': str, 'operator': str, optional 'modifiers': {'severity': ..., 'category': "
            "'must_pass'}}. Parsed defensively by _parse_rubric (app.py:670); None/empty/unparsable means an "
            "unscoreable dry run."
        ),
        json_schema_extra={"consumed_by": ["verify"]},
    )
    question_type: Optional[str] = Field(
        default=None,
        description=(
            "Question category (e.g. 'General Qualitative Analysis'). In the committed data but undeclared "
            "on the wire model, so it is silently dropped at request validation today and never reaches "
            "verify() or the rollout echo."
        ),
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    expert_time_mins: Optional[int] = Field(
        default=None,
        description=(
            "Estimated expert completion time in minutes. In the committed data but undeclared on the wire "
            "model, so it is silently dropped at request validation today."
        ),
        json_schema_extra={"consumed_by": ["provenance"]},
    )
