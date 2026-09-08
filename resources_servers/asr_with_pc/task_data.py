# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-data schema for the asr_with_pc server.

Task fields ride at the top level; there is no verifier_metadata. Mirrors
``ASRWithPCVerifyRequest`` (app.py): one model covers all four task families because the
``task_type`` discriminator is itself Optional (a missing value falls back to the server
config's default), so a discriminated union cannot dispatch on it.

Deliberately OPEN: ``ASR_LEADERBOARD`` rows carry DYNAMIC sibling reference columns named by
``reference_fields`` (e.g. ``text_tn``, ``text_itn``) that verify() fetches via
``body.model_dump()`` and hard-fails on if missing, and compute_metrics() reads the matching
``text_<suffix>`` keys back off rollout dicts. That column set cannot be closed by a schema —
the extra columns flow through ``extra="allow"`` and are consumed by verify/metrics despite
being undeclared here.
"""

from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class TaskData(BaseModel):
    model_config = ConfigDict(extra="allow")

    expected_answer: str = Field(
        default="",
        description="Reference transcript the model transcription is scored against (WER/PER).",
        json_schema_extra={"consumed_by": ["verify"]},
    )
    sample_id: Optional[str] = Field(
        default=None,
        description="Source-dataset sample identifier; never read by verify().",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    split: Optional[str] = Field(
        default=None,
        description="Source-dataset split name (e.g. 'test'); never read by verify().",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    task_type: Optional[Literal["ASR-PC", "ASR", "Hallucination", "ASR_LEADERBOARD"]] = Field(
        default=None,
        description=(
            "Per-row scoring-family override; when absent the server config's task_type (default 'ASR-PC') "
            "applies. 'Hallucination' rows need audio_duration; 'ASR_LEADERBOARD' rows need reference_fields."
        ),
        json_schema_extra={"consumed_by": ["verify"]},
    )
    audio_duration: Optional[float] = Field(
        default=None,
        description=(
            "Audio clip length in seconds, used by 'Hallucination' rows to compute characters-per-minute; "
            "a Hallucination row without it is scored as error='missing_audio_duration'."
        ),
        json_schema_extra={"consumed_by": ["verify"]},
    )
    reference_fields: Optional[List[str]] = Field(
        default=None,
        description=(
            "For 'ASR_LEADERBOARD' rows: names of sibling row columns holding alternate reference "
            "normalizations (e.g. ['text_tn', 'text_itn']). Each named column MUST exist on the row "
            "(verify() raises ValueError if absent) and yields per-suffix wer_<suffix>/is_correct_<suffix> "
            "scores aggregated by compute_metrics()."
        ),
        json_schema_extra={"consumed_by": ["verify", "metrics"]},
    )
