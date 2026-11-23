# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-data schema for the citation_if server.

Per-row reward logic is driven by a top-level ``verifier`` dict (NOT verifier_metadata — no
splicing applies). The wire model ``CitationIfVerifyRequest`` (app.py:115) requires ``verifier``
but keeps it fully untyped (``Dict[str, Any]``); its subkey schema lives in the app.py module
docstring and in scorer.py's defensive ``.get`` defaults, mirrored here as the all-optional
``VerifierSpec`` (every subkey is read via ``.get`` with a default, so none is required).

``_gen_config`` is a data-generation provenance dict (~17-26 keys: date, distractor_mode,
grammar, n_documents, seed, traj_shape, ...) never read by server code; the leading underscore is
not a legal Pydantic field name, so it is declared via alias.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class VerifierSpec(BaseModel):
    """Subkeys of the per-row ``verifier`` dict. All read defensively (scorer.py .get defaults)."""

    model_config = ConfigDict(extra="allow")

    type: Optional[str] = Field(
        default=None,
        description="Row-type tag, 'citation_if' in all committed rows; present in data but never read.",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    mode: str = Field(
        default="cite",
        description="'cite' (citations required) or 'no_cite' (citations forbidden).",
        json_schema_extra={"consumed_by": ["verify"]},
    )
    grammar: str = Field(
        default="",
        description="Citation grammar name compiled per row against id_regex (see GRAMMAR_TABLE in scorer.py).",
        json_schema_extra={"consumed_by": ["verify"]},
    )
    id_kind: str = Field(
        default="full_source",
        description="'full_source' or 'snippet'; selects the DEFAULT_ID_REGEX fallback when id_regex is absent.",
        json_schema_extra={"consumed_by": ["verify"]},
    )
    id_regex: Optional[str] = Field(
        default=None,
        description=(
            "Per-row citation-ID pattern, e.g. 'citation_[a-z0-9]{4}:snippet_[0-9]+'. Authoritative "
            "for grammar parsing; falls back to DEFAULT_ID_REGEX[id_kind] when absent."
        ),
        json_schema_extra={"consumed_by": ["verify"]},
    )
    valid_id_set: Optional[List[str]] = Field(
        default=None,
        description=(
            "Citation IDs visible in the trajectory evidence; coerced to a set for the validity gate. "
            "Null is normalized to the empty set by the scorer (mirrors the expected_ids null treatment)."
        ),
        json_schema_extra={"consumed_by": ["verify"]},
    )
    expected_ids: Optional[List[str]] = Field(
        default=None,
        description=(
            "Ground-truth citation IDs when a correctness signal is available; null means no "
            "correctness gate for the row."
        ),
        json_schema_extra={"consumed_by": ["verify"]},
    )
    expected_slack: int = Field(
        default=1,
        description="Over-citation slack for the correctness gate.",
        json_schema_extra={"consumed_by": ["verify"]},
    )
    min_valid_citations: int = Field(
        default=1,
        description="Minimum count of valid citations required in cite mode.",
        json_schema_extra={"consumed_by": ["verify"]},
    )


class TaskData(BaseModel):
    model_config = ConfigDict(extra="allow")

    verifier: VerifierSpec = Field(
        description=(
            "Per-row verifier spec driving the citation gate sequence. Required on the wire "
            "(CitationIfVerifyRequest declares it with no default) though typed there as an "
            "untyped dict."
        ),
        json_schema_extra={"consumed_by": ["verify"]},
    )
    gen_config: Optional[Dict[str, Any]] = Field(
        default=None,
        alias="_gen_config",
        description=(
            "Data-generation provenance (date, distractor_mode, grammar, n_documents, seed, "
            "traj_shape, ...). Never read by server code; lives at row key '_gen_config'."
        ),
        json_schema_extra={"consumed_by": ["provenance"]},
    )
