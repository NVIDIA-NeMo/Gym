# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-data schema for the aalcr server.

There is no verifier_metadata: task fields ride as top-level row fields on ``AALCRVerifyRequest``
(app.py). Version metadata defaults to the legacy protocol so committed examples and historical
rows remain valid, while prepared benchmark rows always record it explicitly.
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
    aa_lcr_version: Literal["1.0.0", "1.1"] = Field(
        default="1.0.0",
        description="AA-LCR benchmark version used to prepare this row.",
        json_schema_extra={"consumed_by": ["verify", "provenance"]},
    )
    aa_lcr_dataset_revision: str = Field(
        default="bdae010bbce259820c0e34c1d7cce210d966fb75",  # pragma: allowlist secret
        pattern=r"^[0-9a-f]{40}$",
        description="Immutable Hugging Face dataset commit used to prepare this row.",
        json_schema_extra={"consumed_by": ["verify", "provenance"]},
    )
    aa_lcr_judge_protocol: Literal["legacy_v1_0", "official_v1_1"] = Field(
        default="legacy_v1_0",
        description="Judge prompt and verdict protocol required for this row.",
        json_schema_extra={"consumed_by": ["verify", "provenance"]},
    )
