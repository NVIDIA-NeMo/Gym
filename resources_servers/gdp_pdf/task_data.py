# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-data schema for the gdp_pdf server.

All fields currently live inside ``verifier_metadata`` (``GdpPdfVerifyRequest`` in app.py declares
it as a loosely-typed dict), so every field below is annotated ``legacy_location:
verifier_metadata``. ``criteria`` is what ``extract_criteria`` in
``benchmarks/gdp_pdf/prepare.py`` derives from the upstream 30-slot wide rubric; ``verify()``
judges each entry independently. ``pdf_relpath`` is resolved and rendered into page images at
rollout time by ``gdp_pdf_agent`` (``responses_api_agents/gdp_pdf_agent/app.py``) -- it is not read
by ``verify()`` itself.

Per Surge AI's own scorer (surge-ai/gdp-pdf, ``src/gdp_pdf/scorer.py``), the judge is shown only
the model's response and one criterion's ``criterion`` text -- never ``prompt``, ``type``,
``severity``, ``implicitness``, or ``subjectiveness``. Those fields are kept as task metadata
(prompt is real data the policy model needs; the rubric metadata fields are useful for provenance
and analysis) but none of them reach the judge prompt.
"""

from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field


class Criterion(BaseModel):
    model_config = ConfigDict(extra="allow")

    index: int = Field(
        description="1-based slot number in the upstream 30-column wide rubric; traces a verdict back to its source column.",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    criterion: str = Field(
        description="The atomic rubric statement graded independently by the judge.",
        json_schema_extra={"consumed_by": ["verify"]},
    )
    type: Optional[str] = Field(
        default=None,
        description="'Primary Intent' or 'Dodged Bullet' -- the latter must NOT be committed, not merely "
        "satisfied. Informational only: the judge is not shown this field, matching upstream.",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    severity: Optional[str] = Field(
        default=None,
        description="Dealbreaker severity if this criterion is missed; informational only, not read by verify().",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    implicitness: Optional[str] = Field(
        default=None,
        description="Whether the document states this explicitly or requires inference.",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    subjectiveness: Optional[str] = Field(
        default=None,
        description="Whether the criterion is objectively checkable or requires judgment.",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    failure_mode: Optional[str] = Field(
        default=None,
        description="'Binary' or 'Scalar' scoring intent upstream; gdp_pdf's judge always grades pass/fail.",
        json_schema_extra={"consumed_by": ["provenance"]},
    )


class TaskData(BaseModel):
    model_config = ConfigDict(extra="allow")

    task_id: str = Field(
        description="Upstream GDP.pdf task identifier.",
        json_schema_extra={"consumed_by": ["provenance"], "legacy_location": "verifier_metadata"},
    )
    task_response_id: Optional[str] = Field(
        default=None,
        description="Upstream worker-response identifier; echo-only.",
        json_schema_extra={"consumed_by": ["provenance"], "legacy_location": "verifier_metadata"},
    )
    domain: Optional[str] = Field(
        default=None,
        description="One of the ten GDP.pdf professional domains; keys the per-domain metrics breakdown.",
        json_schema_extra={"consumed_by": ["metrics"], "legacy_location": "verifier_metadata"},
    )
    prompt: str = Field(
        description="The task prompt shown to the policy model. Not read by verify(): the judge grades "
        "the response against each criterion alone, matching upstream's scorer.",
        json_schema_extra={"consumed_by": ["prompt"], "legacy_location": "verifier_metadata"},
    )
    pdf_relpath: str = Field(
        description="Path to the source PDF relative to the agent's media_base_dir.",
        json_schema_extra={"consumed_by": ["prompt"], "legacy_location": "verifier_metadata"},
    )
    criteria: List[Criterion] = Field(
        description="Atomic rubric criteria for this task (3-30), judged independently. An empty "
        "list scores reward=0.0 with missing_rubric=True rather than a vacuous all-pass.",
        json_schema_extra={"consumed_by": ["verify"], "legacy_location": "verifier_metadata"},
    )
