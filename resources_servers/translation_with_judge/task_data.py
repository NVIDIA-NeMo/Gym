# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-data schema for the translation_with_judge server.

Mirrors ``TranslationWithJudgeRunRequest`` (app.py:181): fields ride at the row top level, no
``verifier_metadata`` bucket. ``prompt`` and ``solution`` feed both the judge prompt and the
diagnostic sentence-BLEU/chrF computation in verify(); ``src_lang``/``tgt_lang`` (FLORES-200
codes) additionally key compute_metrics()'s per-language-pair aggregation. ``direction``,
``prompt_style``, and ``dataset_type`` ride along for provenance only and are never read.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class TaskData(BaseModel):
    model_config = ConfigDict(extra="allow")

    prompt: List[Dict[str, Any]] = Field(
        description="Instruction-wrapped source segment; its last message's content is the judge's source_text.",
        json_schema_extra={"consumed_by": ["verify", "prompt"]},
    )
    solution: str = Field(
        description="Reference translation; the judge's reference and the sentence-BLEU/chrF ground truth.",
        json_schema_extra={"consumed_by": ["verify"]},
    )
    src_lang: str = Field(
        description="FLORES-200 code (e.g. 'eng_Latn') of `prompt`'s language; names the judge's source language "
        "and keys per-pair aggregation in compute_metrics().",
        json_schema_extra={"consumed_by": ["verify", "metrics"]},
    )
    tgt_lang: str = Field(
        description="FLORES-200 code (e.g. 'kan_Knda') of `solution`'s language; selects the sentence-BLEU "
        "tokenizer, names the judge's target language, and keys per-pair aggregation.",
        json_schema_extra={"consumed_by": ["verify", "metrics"]},
    )
    direction: Optional[str] = Field(
        default=None,
        description="'src2tgt' | 'tgt2src' -- provenance only, not read by verify().",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    prompt_style: Optional[str] = Field(
        default=None,
        description="Which instruction template wrapped the segment -- provenance only.",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    dataset_type: Optional[str] = Field(
        default=None,
        description="Source dataset split label -- provenance only.",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    pass_rate: Optional[float] = Field(
        default=None,
        description="Curriculum-time difficulty signal carried over from data prep -- provenance only.",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    gen_failed: Optional[bool] = Field(
        default=None,
        description="Curriculum-time generation-failure flag carried over from data prep -- provenance only.",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
