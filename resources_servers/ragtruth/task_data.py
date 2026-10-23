# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-data schema for the ragtruth server.

Task fields ride at the row top level; there is no verifier_metadata. Mirrors
``RagtruthRunRequest`` (app.py): both fields are wire-optional with defaults, so a row missing
``is_halu`` silently scores against ``False`` — the defaults here reproduce that wire behavior
rather than fix it. ``think_tag`` (reasoning-block stripping) comes from server config, not the
row.
"""

from pydantic import BaseModel, ConfigDict, Field


class TaskData(BaseModel):
    model_config = ConfigDict(extra="allow")

    is_halu: bool = Field(
        default=False,
        description=(
            "Gold binary label — does this case contain a hallucination? Precomputed at prep time "
            "as bool(labels) from the upstream RAGTruth annotations; verify()'s only scoring input "
            "besides the model response, and the basis for corpus-level P/R/F1 in compute_metrics(). "
            "Wire-optional: a row missing it scores against False."
        ),
        json_schema_extra={"consumed_by": ["verify", "metrics"]},
    )
    task_type: str = Field(
        default="",
        description=(
            "RAGTruth task family ('QA', 'Summary', or 'Data2txt'). Never scored against; echoed "
            "through the verify response and consumed by compute_metrics() for per-slice P/R/F1."
        ),
        json_schema_extra={"consumed_by": ["metrics"]},
    )
