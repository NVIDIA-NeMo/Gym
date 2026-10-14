# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-data schema for the audiobench_judge server.

Task fields ride at the top level; there is no verifier_metadata. Mirrors
``AudioBenchJudgeVerifyRequest`` (app.py). The audio itself rides in
``responses_create_params.metadata`` and is not task data.
"""

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class TaskData(BaseModel):
    model_config = ConfigDict(extra="allow")

    question: str = Field(
        description="Question or instruction about the audio, shown to the judge alongside the answers.",
        json_schema_extra={"consumed_by": ["verify"]},
    )
    expected_answer: str = Field(
        description="Reference answer the judge rates the model's answer against (0-5).",
        json_schema_extra={"consumed_by": ["verify"]},
    )
    subset_for_metrics: Optional[str] = Field(
        default=None,
        description="AudioBench sub-dataset name; when present, compute_metrics() adds per-sub-dataset metrics.",
        json_schema_extra={"consumed_by": ["metrics"]},
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
