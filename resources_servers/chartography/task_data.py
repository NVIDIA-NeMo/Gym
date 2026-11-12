# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-data schema for the chartography server.

Every field lives inside ``verifier_metadata`` (``ChartographyVerifyRequest`` in app.py declares it as a
loosely typed dict), so each one is annotated ``legacy_location: verifier_metadata``. ``verify()`` reads
``prompt`` and ``golden_answer`` (the judge sees both, plus the response's final answer); ``domain`` keys the
per-domain metrics. The chart itself reaches the policy through ``responses_create_params`` and is never read
by ``verify()``. ``benchmarks/chartography/prepare.py`` writes every field below; the synthetic example rows
carry the four that matter for grading and metrics.
"""

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class TaskData(BaseModel):
    model_config = ConfigDict(extra="allow")

    task_id: str = Field(
        description="Task identifier (the release's task_id for the benchmark rows).",
        json_schema_extra={"consumed_by": ["provenance"], "legacy_location": "verifier_metadata"},
    )
    prompt: str = Field(
        description="The question shown to the policy model, before the chart. verify() passes it to the "
        "judge prompt as the question being answered.",
        json_schema_extra={"consumed_by": ["prompt", "verify"], "legacy_location": "verifier_metadata"},
    )
    golden_answer: str = Field(
        description="The reference answer, one or more parts; a numeric part may carry an acceptable range. "
        "verify() passes it to the judge prompt as the criteria to match.",
        json_schema_extra={"consumed_by": ["verify"], "legacy_location": "verifier_metadata"},
    )
    domain: Optional[str] = Field(
        default=None,
        description="Subject-area slug of domain_combined (for example stem_mechanical_engineering); keys the "
        "per-domain metrics breakdown.",
        json_schema_extra={"consumed_by": ["metrics"], "legacy_location": "verifier_metadata"},
    )
    domain_combined: Optional[str] = Field(
        default=None,
        description="The release's subject-area label, as published. Provenance only.",
        json_schema_extra={"consumed_by": ["provenance"], "legacy_location": "verifier_metadata"},
    )
    source_type: Optional[str] = Field(
        default=None,
        description="Whether the chart was expert-created or sourced online. Provenance only.",
        json_schema_extra={"consumed_by": ["provenance"], "legacy_location": "verifier_metadata"},
    )
    source_url: Optional[str] = Field(
        default=None,
        description="Where a sourced chart was found. Provenance only.",
        json_schema_extra={"consumed_by": ["provenance"], "legacy_location": "verifier_metadata"},
    )
    chart_path: Optional[str] = Field(
        default=None,
        description="The chart's path inside the release. Provenance only: the chart itself is embedded in "
        "responses_create_params.",
        json_schema_extra={"consumed_by": ["provenance"], "legacy_location": "verifier_metadata"},
    )
