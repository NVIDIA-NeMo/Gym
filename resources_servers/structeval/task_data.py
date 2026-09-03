# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-data schema for the structeval server.

All task fields are top-level (no ``verifier_metadata``) and mirror ``StructEvalVerifyRequest``
(app.py:54), where the five identity/spec fields are wire-required. ``verify()`` parses the
assistant text per ``output_type`` (json/yaml/xml/toml/csv, matched case-insensitively) and
checks the key paths in ``raw_output_metric``. ``task_name`` and ``input_type`` are only echoed
by verify() but are read by ``compute_metrics`` for per-slice reward metrics. ``rendering`` is
declared and echoed but never read (the reward formula is hardcoded to the non-renderable path);
it is false in every committed row.
"""

from typing import List

from pydantic import BaseModel, ConfigDict, Field


class TaskData(BaseModel):
    model_config = ConfigDict(extra="allow")

    task_id: str = Field(
        description="StructEval task identifier (e.g. '000500'); wire-required, echoed into the verify response.",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    task_name: str = Field(
        description="Human-readable task name (e.g. 'Text to JSON'); used by compute_metrics for reward slicing.",
        json_schema_extra={"consumed_by": ["metrics"]},
    )
    input_type: str = Field(
        description="Input format label (e.g. 'Text'); used by compute_metrics for reward slicing.",
        json_schema_extra={"consumed_by": ["metrics"]},
    )
    output_type: str = Field(
        description="Target output format; lowercased by verify() to pick the parser (json/yaml/xml/toml/csv); "
        "also used by compute_metrics for reward/render/key-validation slicing.",
        json_schema_extra={"consumed_by": ["verify", "metrics"]},
    )
    raw_output_metric: List[str] = Field(
        description="Dot-notation key paths (or 'csv::' headers) that must be present in the parsed output.",
        json_schema_extra={"consumed_by": ["verify"]},
    )
    rendering: bool = Field(
        default=False,
        description="Declared/echoed but never read by verify(); false in all committed rows.",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
