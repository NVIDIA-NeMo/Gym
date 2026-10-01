# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-data schema for the combibench server (CombiBench combinatorics in Lean 4).

Fields are top-level row columns, written by ``benchmarks/combibench/prepare.py`` and
templated into the prompt by ``prompt.yaml``. Only ``formal_statement`` is required, and
the rest default to ``None``, which mirrors ``CombibenchRunRequest``: every task field
there is typed ``Any`` on purpose, so a malformed row becomes a ``bad_task`` status from
``verify()`` rather than a validation error that ends the whole run. This schema documents
the shape a *well-formed* row has; it is deliberately not a second gate in front of that
one.

``formal_statement`` is the reference statement with ``sorry`` placeholders -- for
fill-in-the-blank problems an ``abbrev <name>_solution : <type> := sorry`` as well as the
theorem. ``answers`` pairs with those abbreviations positionally, which is upstream's rule
and the reason ``prepare.py`` refuses a row whose answer count does not match the
``_solution`` abbreviations the verifier will find.
"""

from typing import Any, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class TaskData(BaseModel):
    model_config = ConfigDict(extra="allow")

    formal_statement: Any = Field(
        description=(
            "The Lean 4 reference statement with its 'sorry' placeholders: the theorem, plus an "
            "'abbrev <name>_solution : <type> := sorry' for each blank on fill-in-the-blank "
            "problems. Rendered into the prompt, and used by verify() both for the "
            "statement-preservation check and to locate the answer abbreviations."
        ),
        json_schema_extra={"consumed_by": ["verify", "prompt"]},
    )
    answers: Optional[List[str]] = Field(
        default=None,
        description=(
            "Ground-truth answers for a fill-in-the-blank problem, one per '_solution' abbreviation "
            "and in the same order; null for the 55 proof-only problems. verify() zips these "
            "positionally with the abbreviations it finds and appends an equality check per pair."
        ),
        json_schema_extra={"consumed_by": ["verify"]},
    )
    theorem_name: Optional[str] = Field(
        default=None,
        description="Upstream problem identifier, e.g. 'imo_2000_p4'. Identifies the row in a rollout dump.",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    tag: Optional[str] = Field(
        default=None,
        description=(
            "Upstream source family: 'hackmath', 'brualdi', 'imo' or 'math_competitions'. "
            "compute_metrics groups on this for the per-family pass rates; a row without it drops "
            "out of that breakdown."
        ),
        json_schema_extra={"consumed_by": ["metrics"]},
    )
    natural_language: Optional[str] = Field(
        default=None,
        description=(
            "Informal prose statement. Carried for provenance and never shown to the model: "
            "upstream's prompt shows the formal statement only, and this server reproduces that."
        ),
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    source: Optional[str] = Field(
        default=None,
        description="Upstream URL the problem was published at. Provenance only.",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    split: Optional[str] = Field(
        default=None,
        description="'test' (answers withheld) or 'test_with_solution' (published answers substituted).",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    dataset_source: Optional[str] = Field(
        default=None,
        description=(
            "Which upstream copy the row came from: 'github' (the default, whose statements all "
            "compile on Mathlib v4.24.0), 'hf' (6 statements do not), or 'synthetic' for the "
            "committed example. Decides how a score should be read, so it is echoed on every rollout."
        ),
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    dataset_revision: Optional[str] = Field(
        default=None,
        description="Pinned upstream revision the row was prepared from.",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
