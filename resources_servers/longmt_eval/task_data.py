# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-data schema for the longmt_eval server (long-document machine translation, SEGALE-scored).

Task fields ride at the row top level; there is no verifier_metadata. Required-ness mirrors
``LongmtEvalRunRequest`` (app.py:112): ``text``, ``source_language``, ``target_language`` and
``doc_id`` are required, ``target_len`` is Optional (typed on the wire, absent from committed
data). NOTE: today's wire model does not set ``extra="allow"`` (pydantic default ``ignore``), so
the committed provenance columns ``source_lang_name``, ``target_lang_name``, ``seg_id``,
``publication_date`` and ``url`` are silently DROPPED in transit — they are declared here as
optional provenance so dataset tooling keeps them; this schema uses ``extra="allow"`` per
protocol (``extra="ignore"`` is banned).
"""

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class TaskData(BaseModel):
    model_config = ConfigDict(extra="allow")

    text: str = Field(
        description=(
            "The tiktoken-truncated source document (up to ~24k chars in committed rows), written by "
            "prepare.py. IS the prompt source — duplicated in the row alongside "
            "responses_create_params.input — and is passed to the SEGALE actor for alignment scoring."
        ),
        json_schema_extra={"consumed_by": ["verify", "prompt"]},
    )
    source_language: str = Field(
        description=(
            "Source language code (e.g. 'en'). Wire-required but unused by verify() — only "
            "target_language reaches the SEGALE actor; echoed back on the verify response."
        ),
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    target_language: str = Field(
        description=(
            "Target locale code (e.g. 'de_DE'); passed to the SEGALE actor for language-fidelity "
            "checks and used by compute_metrics() as the per-language breakdown key."
        ),
        json_schema_extra={"consumed_by": ["verify", "metrics"]},
    )
    doc_id: str = Field(
        description="Document identifier; wire-required, used by verify() only for error logging.",
        json_schema_extra={"consumed_by": ["verify", "provenance"]},
    )
    target_len: Optional[int] = Field(
        default=None,
        description=(
            "tiktoken token count the source was truncated to. Typed Optional on the wire; absent "
            "from all committed rows."
        ),
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    source_lang_name: Optional[str] = Field(
        default=None,
        description="Human-readable source language name (e.g. 'English'). Dropped by today's wire model.",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    target_lang_name: Optional[str] = Field(
        default=None,
        description="Human-readable target language name (e.g. 'German'). Dropped by today's wire model.",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    seg_id: Optional[int] = Field(
        default=None,
        description="Segment index within the source document. Dropped by today's wire model.",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    publication_date: Optional[int] = Field(
        default=None,
        description="Publication year of the source document (e.g. 1886). Dropped by today's wire model.",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    url: Optional[str] = Field(
        default=None,
        description="Source-document URL (e.g. a gutenberg.org link). Dropped by today's wire model.",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
