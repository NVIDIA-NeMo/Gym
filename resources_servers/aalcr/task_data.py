# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-data schema for the aalcr server.

There is no verifier_metadata: all nine task fields ride as REQUIRED top-level row fields on
``AALCRVerifyRequest`` (app.py), so all nine stay required here even though verify() only reads
``question``, ``answer``, and ``input_tokens_band`` — the other six exist to be echoed into
``AALCRVerifyResponse``. ``input_tokens_band`` is narrowed to the five-band Literal: verify()'s
``match`` statement has no default case, so any other value crashes with an UnboundLocalError,
making the enum the server's real contract even though the wire types it as a bare ``str``.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class TaskData(BaseModel):
    model_config = ConfigDict(extra="allow")

    document_category: str = Field(
        description="Category of the source document set; echoed into the verify response, never graded on.",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    document_set_id: str = Field(
        description="Identifier of the document set this question is drawn from; echo-only.",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    question_id: int = Field(
        description="Numeric question identifier within the document set; echo-only.",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    question: str = Field(
        description="The long-context question; interpolated into the LLM judge prompt for reference.",
        json_schema_extra={"consumed_by": ["verify"]},
    )
    answer: str = Field(
        description="The official answer the judge compares the candidate answer against.",
        json_schema_extra={"consumed_by": ["verify"]},
    )
    data_source_filenames: str = Field(
        description="Source document filenames (single string, not a JSON list); echo-only.",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    data_source_urls: str = Field(
        description="Source document URLs (single string, not a JSON list); echo-only.",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    input_tokens: int = Field(
        description="Prompt length in tokens used to derive input_tokens_band; echo-only.",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    input_tokens_band: Literal["<80k", "80k-100k", "100k-110k", "110k-128k", "128k+"] = Field(
        description=(
            "Context-length band selecting which per-band reward field (reward_lt_80k .. reward_128k_plus) "
            "the row's reward is mirrored into. Wire-typed str, but verify()'s match has no default case, "
            "so these five values are the de-facto enum."
        ),
        json_schema_extra={"consumed_by": ["verify", "metrics"]},
    )
