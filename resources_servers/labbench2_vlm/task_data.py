# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-data schema for the labbench2_vlm server.

LAB-Bench 2 vision-language QA graded by an LLM judge. All task fields live inside
``verifier_metadata`` on today's wire (``LabbenchVLMVerifyRequest`` types it
``Optional[dict[str, Any]]`` with no subkey typing, ``extra="allow"``); schemas are written flat,
so every field carries ``legacy_location``. Every verify() read is a defensive
``meta.get(..., "")`` with str coercion, so all fields stay Optional. ``media_dir`` is consumed
by the agent harness (labbench2_vlm_agent's ``embed_media_into_row``) at rollout time, not by
verify(); the referenced media files live OUT OF ROW under that relative directory.
"""

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class TaskData(BaseModel):
    model_config = ConfigDict(extra="allow")

    ideal: Optional[str] = Field(
        default=None,
        description="Gold answer, free text; compared to the model answer by the LLM judge.",
        json_schema_extra={"consumed_by": ["verify"], "legacy_location": "verifier_metadata"},
    )
    tag: Optional[str] = Field(
        default=None,
        description=(
            "Task family tag (e.g. 'protocolqa2', 'figqa2-img', 'figqa2-pdf', 'tableqa2-img', "
            "'tableqa2-pdf'). tag.startswith('protocolqa2') selects the protocol judge prompt; "
            "compute_metrics uses it for per-tag breakdowns."
        ),
        json_schema_extra={"consumed_by": ["verify", "metrics"], "legacy_location": "verifier_metadata"},
    )
    id: Optional[str] = Field(
        default=None,
        description="Source item uuid (sometimes suffixed, e.g. '-img'); unused by verify().",
        json_schema_extra={"consumed_by": ["provenance"], "legacy_location": "verifier_metadata"},
    )
    media_dir: Optional[str] = Field(
        default=None,
        description=(
            "Relative directory of the task's media assets (e.g. 'test_media/protocols/<uuid>'); "
            "consumed by the labbench2_vlm_agent harness to embed images/PDFs into the prompt at "
            "rollout time, not by verify()."
        ),
        json_schema_extra={"consumed_by": ["prompt"], "legacy_location": "verifier_metadata"},
    )
    reference_passage: Optional[str] = Field(
        default=None,
        description=(
            "Judge context passage; meaningful for protocolqa2 rows (fed to the protocol judge prompt), "
            "may be a bare figure reference like 'Figure 6B' otherwise."
        ),
        json_schema_extra={"consumed_by": ["verify"], "legacy_location": "verifier_metadata"},
    )
