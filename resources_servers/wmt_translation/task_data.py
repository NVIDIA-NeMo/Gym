# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-data schema for the wmt_translation server.

WMT-style machine translation scored with sentence chrF/spBLEU at verify time and xCOMET-XXL plus
optional language-consistency at aggregation time. All task fields ride at the row top level and
mirror ``WmtTranslationRunRequest`` (app.py:148). verify() itself needs only ``translation`` and
``target_language``, but ``text`` and ``source_language`` stay wire-required because
compute_metrics() consumes them (COMET ``src`` input and per-language-pair aggregation buckets);
the ``*_lang_name`` fields are optional prompt-building conveniences never read after rollout.
"""

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class TaskData(BaseModel):
    model_config = ConfigDict(extra="allow")

    text: str = Field(
        description=(
            "Source-language sentence to translate; used as the COMET `src` input by compute_metrics() "
            "and as prompt material, not read by verify()."
        ),
        json_schema_extra={"consumed_by": ["metrics", "prompt"]},
    )
    translation: str = Field(
        description="Reference translation; ground truth for sentence chrF/spBLEU and COMET `ref`.",
        json_schema_extra={"consumed_by": ["verify", "metrics"]},
    )
    source_language: str = Field(
        description="Source-language code (e.g. 'en'); keys per-pair aggregation in compute_metrics().",
        json_schema_extra={"consumed_by": ["metrics"]},
    )
    target_language: str = Field(
        description=(
            "Target-language/locale code (e.g. 'de_DE', 'ja_JP'); passed to the optional "
            "language-consistency backend by verify() and keys per-pair aggregation."
        ),
        json_schema_extra={"consumed_by": ["verify", "metrics"]},
    )
    source_lang_name: Optional[str] = Field(
        default=None,
        description="Human-readable source-language name (e.g. 'English'); prompt-building convenience only.",
        json_schema_extra={"consumed_by": ["prompt"]},
    )
    target_lang_name: Optional[str] = Field(
        default=None,
        description="Human-readable target-language name (e.g. 'German'); prompt-building convenience only.",
        json_schema_extra={"consumed_by": ["prompt"]},
    )
