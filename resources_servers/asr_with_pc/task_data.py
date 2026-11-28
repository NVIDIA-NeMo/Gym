# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-data schema for the asr_with_pc server.

Task fields ride at the top level; there is no verifier_metadata. Mirrors
``ASRWithPCVerifyRequest`` (app.py): one model covers all six task types because the
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


# The server's own copies (app._TASK_TYPES, speech_metrics.CerWhitespaceMode) live in modules that need the scoring
# libraries; tests/test_task_data.py keeps these in step with them.
TaskType = Literal["ASR-PC", "ASR", "OpenASR", "Multilingual-ASR", "Hallucination", "ASR_LEADERBOARD"]
CerWhitespaceMode = Literal["keep", "all", "han_adjacent"]


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
    task_type: Optional[TaskType] = Field(
        default=None,
        description=(
            "Per-row scoring-family override; when absent the server config's task_type (default 'ASR-PC') "
            "applies. 'Hallucination' rows need audio_duration; 'ASR_LEADERBOARD' rows need reference_fields; "
            "'Multilingual-ASR' rows need language; 'OpenASR' and 'Multilingual-ASR' report additive edit counts."
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
    normalization: Optional[Literal["numb3rs"]] = Field(
        default=None,
        description="'ASR_LEADERBOARD' rows: 'numb3rs' scores with the category-based Numb3rs normalizer.",
        json_schema_extra={"consumed_by": ["verify", "metrics"]},
    )
    category: Optional[str] = Field(
        default=None,
        description="Numb3rs category (e.g. 'MONEY'); required when normalization is 'numb3rs'.",
        json_schema_extra={"consumed_by": ["verify", "metrics"]},
    )
    benchmark_profile: Optional[str] = Field(
        default=None,
        description="Profile name for the per-profile breakdown (wer::profile::<name>) and the multilingual macro.",
        json_schema_extra={"consumed_by": ["metrics"]},
    )
    subset_for_metrics: Optional[str] = Field(
        default=None,
        description="Subset name for the per-subset breakdown (wer::subset::<name>); chunked sessions group by it.",
        json_schema_extra={"consumed_by": ["metrics"]},
    )
    parent_id: Optional[str] = Field(
        default=None,
        description=(
            "'OpenASR' chunk rows: identifier of the session the chunk belongs to; needs chunk_index. Chunks "
            "of a session are merged into one hypothesis at aggregation."
        ),
        json_schema_extra={"consumed_by": ["verify", "metrics"]},
    )
    chunk_index: Optional[int] = Field(
        default=None,
        description="'OpenASR' chunk rows: zero-based position of the chunk in its session; contiguous within a session.",
        json_schema_extra={"consumed_by": ["verify", "metrics"]},
    )
    language: Optional[str] = Field(
        default=None,
        description=(
            "'Multilingual-ASR' rows: language or locale code (e.g. 'de', 'de_de', 'en-US'); required. "
            "Also the key of the per-language breakdown."
        ),
        json_schema_extra={"consumed_by": ["verify", "metrics"]},
    )
    use_cer: bool = Field(
        default=False,
        description="'Multilingual-ASR' rows: score characters instead of words (wer_* then counts characters).",
        json_schema_extra={"consumed_by": ["verify"]},
    )
    remove_diacritics: Optional[bool] = Field(
        default=None,
        description="'Multilingual-ASR' rows: fold diacritics; unset keeps them (the Open ASR leaderboard default).",
        json_schema_extra={"consumed_by": ["verify"]},
    )
    normalize_compound: Optional[bool] = Field(
        default=None,
        description="'Multilingual-ASR' rows: join word pairs that differ only in spacing; unset is true unless English.",
        json_schema_extra={"consumed_by": ["verify"]},
    )
    merge_compounds: Optional[bool] = Field(
        default=None,
        description="'Multilingual-ASR' rows: let the WER aligner merge compound words; unset is false.",
        json_schema_extra={"consumed_by": ["verify"]},
    )
    cer_whitespace_mode: CerWhitespaceMode = Field(
        default="keep",
        description="'Multilingual-ASR' rows: 'keep' whitespace, drop 'all' of it, or drop it next to Han characters.",
        json_schema_extra={"consumed_by": ["verify"]},
    )
