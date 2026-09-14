# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-data schema for the longmemeval server (LLM-judged long-term-memory QA).

All task fields live inside the legacy ``verifier_metadata`` bucket on today's wire
(``LongMemEvalRunRequest`` at app.py:248 types it as an untyped ``Optional[Dict[str, Any]]``,
``extra="allow"``); the schema is written flat with ``legacy_location`` annotations. Every field
is Optional because the wire never 422s on the bucket's contents — verify() reads each subkey
defensively via ``_as_str``. ``is_bad_metadata`` (app.py:220) does define a de-facto required
subset: non-empty ``question_type`` AND ``question``, otherwise the row is classified
bad_metadata and scored 0.0, so treat those two as required-in-practice for dataset prep.
"""

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class TaskData(BaseModel):
    model_config = ConfigDict(extra="allow")

    question_id: Optional[str] = Field(
        default=None,
        description=(
            "Upstream LongMemEval question id. A '_abs' suffix marks an abstention row: verify() ORs "
            "'_abs' in question_id with the abstention flag when choosing the rubric."
        ),
        json_schema_extra={"consumed_by": ["verify", "provenance"], "legacy_location": "verifier_metadata"},
    )
    question_type: Optional[str] = Field(
        default=None,
        description=(
            "Rubric selector: one of single-session-user, single-session-assistant, "
            "single-session-preference, multi-session, temporal-reasoning, knowledge-update. "
            "Required-in-practice (empty -> bad_metadata, reward 0.0); an unknown value yields "
            "ERROR_UNKNOWN_QUESTION_TYPE, so it stays a plain str rather than a Literal."
        ),
        json_schema_extra={"consumed_by": ["verify", "metrics"], "legacy_location": "verifier_metadata"},
    )
    question: Optional[str] = Field(
        default=None,
        description="The question posed over the chat history. Required-in-practice (empty -> bad_metadata).",
        json_schema_extra={"consumed_by": ["verify"], "legacy_location": "verifier_metadata"},
    )
    answer: Optional[str] = Field(
        default=None,
        description=(
            "Gold answer, or the preference rubric for single-session-preference rows. verify() "
            "grades against the empty string when absent."
        ),
        json_schema_extra={"consumed_by": ["verify"], "legacy_location": "verifier_metadata"},
    )
    abstention: Optional[bool] = Field(
        default=None,
        description=(
            "True when the correct behavior is declining to answer; verify() reads it as "
            "bool(meta.get('abstention')) and also infers it from a '_abs' question_id suffix."
        ),
        json_schema_extra={"consumed_by": ["verify"], "legacy_location": "verifier_metadata"},
    )
    question_date: Optional[str] = Field(
        default=None,
        description=(
            "Timestamp of the question, e.g. '2023/05/24 (Wed) 22:03'. Present in every committed "
            "row but never read by verify() or compute_metrics()."
        ),
        json_schema_extra={"consumed_by": ["provenance"], "legacy_location": "verifier_metadata"},
    )
    split: Optional[str] = Field(
        default=None,
        description="Dataset split label (e.g. 'oracle'); never read by verify() or compute_metrics().",
        json_schema_extra={"consumed_by": ["provenance"], "legacy_location": "verifier_metadata"},
    )
    topk_context: Optional[int] = Field(
        default=None,
        description=(
            "Number of retrieved context sessions the prompt was built with (e.g. 50); never read "
            "by verify() or compute_metrics()."
        ),
        json_schema_extra={"consumed_by": ["provenance"], "legacy_location": "verifier_metadata"},
    )
