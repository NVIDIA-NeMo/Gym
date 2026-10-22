# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-data schema for the beam server.

Task fields live inside ``verifier_metadata`` (``BEAMVerifyRequest`` in app.py declares it as a
loosely-typed dict), so they are annotated ``legacy_location: verifier_metadata``. Rows are built by
``benchmarks/beam/prepare.py``: one row per BEAM probing question. ``verify()`` needs ``ability`` and
a non-empty ``rubric``; without them it returns reward 0 with ``judge_error=bad_metadata``.
"""

from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class TaskData(BaseModel):
    model_config = ConfigDict(extra="allow")

    ability: Literal[
        "abstention",
        "contradiction_resolution",
        "event_ordering",
        "information_extraction",
        "instruction_following",
        "knowledge_update",
        "multi_session_reasoning",
        "preference_following",
        "summarization",
        "temporal_reasoning",
    ] = Field(
        description="BEAM memory ability probed; event_ordering adds Kendall-tau scoring. Mirrors ABILITIES in app.py.",
        json_schema_extra={"consumed_by": ["verify", "metrics"], "legacy_location": "verifier_metadata"},
    )
    rubric: List[str] = Field(
        min_length=1,
        description="Upstream per-question rubric items, each graded independently by the LLM judge.",
        json_schema_extra={"consumed_by": ["verify"], "legacy_location": "verifier_metadata"},
    )
    chat_size: Optional[Literal["100K", "500K", "1M"]] = Field(
        default=None,
        description="BEAM split (chat length) the conversation comes from; groups chat_size/* metrics.",
        json_schema_extra={"consumed_by": ["metrics"], "legacy_location": "verifier_metadata"},
    )
    conversation_id: Optional[str] = Field(
        default=None,
        description="Upstream conversation identifier; echoed into the verify response.",
        json_schema_extra={"consumed_by": ["provenance"], "legacy_location": "verifier_metadata"},
    )
    question_index: Optional[int] = Field(
        default=None,
        description="Index of the question within its ability for this conversation.",
        json_schema_extra={"consumed_by": ["provenance"], "legacy_location": "verifier_metadata"},
    )
    question: Optional[str] = Field(
        default=None,
        description="The probing question (already appended to the prompt); not shown to the judge.",
        json_schema_extra={"consumed_by": ["provenance"], "legacy_location": "verifier_metadata"},
    )
    reference_answer: Optional[str] = Field(
        default=None,
        description="Upstream reference answer; not used for scoring (the rubric is).",
        json_schema_extra={"consumed_by": ["provenance"], "legacy_location": "verifier_metadata"},
    )
    difficulty: Optional[str] = Field(
        default=None,
        description="Upstream difficulty label, when present.",
        json_schema_extra={"consumed_by": ["provenance"], "legacy_location": "verifier_metadata"},
    )
    prompt_tokens: Optional[int] = Field(
        default=None,
        description="Prompt length counted at prepare time (top-level row field); used for length filtering only.",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
