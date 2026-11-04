# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-data schema for the over_refusal_detection server (XSTest-style over-refusal judging).

Mirrors ``OverRefusalDetectionRunRequest`` (app.py): all four task fields are Optional with
``None`` defaults and the wire model carries ``extra="allow"``. ``verify()`` reads only
``safe_prompt``, and even that defensively — when it is missing or empty the judge prompt falls
back to the last user message in ``responses_create_params.input``. ``contrast_prompt`` and
``metadata`` are declared on the wire but absent from committed data and unread.
"""

from typing import Any, Dict, Optional

from pydantic import BaseModel, ConfigDict, Field


class TaskData(BaseModel):
    model_config = ConfigDict(extra="allow")

    safe_prompt: Optional[str] = Field(
        default=None,
        description=(
            "The prompt that SHOULD be answered (looks unsafe but is safe); fills the judge prompt. "
            "Falls back to the last user message when missing/empty."
        ),
        json_schema_extra={"consumed_by": ["verify"]},
    )
    category: Optional[str] = Field(
        default=None,
        description="XSTest category (e.g. homonyms, figurative_language); passed through, never read.",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    contrast_prompt: Optional[str] = Field(
        default=None,
        description="Optional unsafe contrast version of the prompt; wire-declared but unread and uncommitted.",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Open provenance dict; wire-declared but unread and absent from committed data.",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
