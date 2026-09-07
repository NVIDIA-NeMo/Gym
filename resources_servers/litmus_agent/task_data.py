# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-data schema for the litmus_agent server (chemistry property QA with a code sandbox).

Ground truth rides as top-level row fields; there is no verifier_metadata. Three row formats
coexist in one file and deliberately share this single open model (there is no in-band
discriminator — the format is detected by which optionals are present):

(a) modern rows: ``answer_type`` + ``answer_format``, no ``match``;
(b) modern rows with a per-row ``match`` reward-rule override;
(c) legacy rows: ``property_type`` + ``use_box_format`` and NO ``answer_type``/``answer_format``.

Required-ness mirrors ``LitmusAgentRunRequest`` (app.py:297, ``extra="allow"``): only
``expected_answer`` is required; everything else is Optional/defaulted so all three generations
keep validating. ``property_type`` is declared here for tooling (it is verify-consumed on legacy
rows) but is deliberately UNdeclared on today's wire model — see its field description before
migrating the request model. Domain-context fields (``method``, ``property``, ``chembl_id``,
``smiles``) are documented pass-through: never required by verify(), but compute_metrics()
groups by ``method`` and ``property``.
"""

from typing import Any, Dict, Optional, Union

from pydantic import BaseModel, ConfigDict, Field


class TaskData(BaseModel):
    model_config = ConfigDict(extra="allow")

    expected_answer: Union[str, float, int] = Field(
        description=(
            "Gold answer, compared after parsing per the resolved answer type. Committed rows carry "
            "it as a string even for numeric properties (e.g. '3', '18.02', 'true')."
        ),
        json_schema_extra={"consumed_by": ["verify"]},
    )
    answer_type: Optional[str] = Field(
        default=None,
        description=(
            "How the answer is parsed: one of 'float', 'bool', 'string'. Optional so legacy rows "
            "carrying only property_type still resolve via _PROPERTY_TYPE_TO_ANSWER_TYPE."
        ),
        json_schema_extra={"consumed_by": ["verify"]},
    )
    output_regex: Optional[str] = Field(
        default=None,
        description=(
            "Preferred extraction path: a regex with exactly one capture group carried directly on "
            "the row. Wins over answer_format when present. Typed on the wire model but absent from "
            "all committed rows."
        ),
        json_schema_extra={"consumed_by": ["verify"]},
    )
    answer_format: Optional[str] = Field(
        default=None,
        description="Extraction-format registry name (fmt_00..fmt_30); used when output_regex is absent.",
        json_schema_extra={"consumed_by": ["verify"]},
    )
    use_box_format: bool = Field(
        default=False,
        description=(
            "Legacy extraction fallback: parse the answer from \\boxed{}. Carried by legacy "
            "property_type rows; wire default False."
        ),
        json_schema_extra={"consumed_by": ["verify"]},
    )
    match: Optional[Dict[str, Any]] = Field(
        default=None,
        description=(
            "Per-row reward-rule override: {'rule': <name>, **params}, e.g. {'rule': 'exact'} or "
            "{'rule': 'abs_window', 'abs_tol': 0.1}. When absent, the default rule for the resolved "
            "answer_type applies."
        ),
        json_schema_extra={"consumed_by": ["verify"]},
    )
    property_type: Optional[str] = Field(
        default=None,
        description=(
            "Legacy answer-kind selector in {'float', 'count', 'fragment', 'bool', 'presence'}. "
            "CAUTION: deliberately UNdeclared on today's wire model — verify() reads it via "
            "body.model_extra.get('property_type'), so declaring it on the request model during "
            "migration would silently break legacy rows unless that read is updated too. Also feeds "
            "the MAE-eligibility set (_MAE_PROPERTY_TYPES)."
        ),
        json_schema_extra={"consumed_by": ["verify", "metrics"]},
    )
    method: Optional[str] = Field(
        default=None,
        description=(
            "Task-generation method (e.g. 'direct'); undeclared wire extra passed through via "
            "extra='allow'. compute_metrics() groups rewards by it (r.get('method', 'unknown'))."
        ),
        json_schema_extra={"consumed_by": ["metrics"]},
    )
    property: Optional[str] = Field(
        default=None,
        description=(
            "Chemical property being asked about (e.g. 'MolWt', 'fr_Al_OH'); undeclared wire extra. "
            "compute_metrics() groups rewards by it (r.get('property', 'unknown'))."
        ),
        json_schema_extra={"consumed_by": ["metrics"]},
    )
    chembl_id: Optional[str] = Field(
        default=None,
        description="ChEMBL molecule identifier; undeclared wire extra, echoed through untouched.",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    smiles: Optional[str] = Field(
        default=None,
        description="SMILES string of the molecule under question; undeclared wire extra, echoed through.",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
