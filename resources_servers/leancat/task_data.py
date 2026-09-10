# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-data schema for the leancat server (LeanCat formal category theory in Lean 4).

Every field is a top-level row column on today's rows (``prepare.py``), the same flat shape
``benchmarks/minif2f`` uses -- ``formal_statement`` has to be top-level for the benchmark's
``prompt_config`` to fill ``{formal_statement}`` from it. ``app.py`` also accepts them nested
under ``verifier_metadata`` for hand-written rows. Required-ness mirrors ``LeanCatRunRequest``
(app.py): only ``formal_statement`` is required, because it is the one field ``verify()``
cannot do without -- it is the reference the statement-preservation check compares against.
Everything else is Optional so a hand-written row can exercise the server without carrying the
full upstream record.

``level`` is not read by ``verify()`` but is not provenance either: ``compute_metrics`` groups on
it to produce the Easy/Medium/High breakdown, which is the split the LeanCat paper's argument
rests on. A row without it still scores, but silently drops out of the per-difficulty metrics.
"""

from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field


class TaskData(BaseModel):
    model_config = ConfigDict(extra="allow")

    formal_statement: str = Field(
        description=(
            "The complete reference Lean 4 file -- imports, open/variable preamble, any auxiliary "
            "declarations, and the target theorem whose proof is 'sorry'. Rendered into the prompt, "
            "and used by verify() as the reference for the statement-preservation check."
        ),
        json_schema_extra={"consumed_by": ["verify", "prompt"]},
    )
    problem_id: Optional[str] = Field(
        default=None,
        description="Upstream LeanCat problem id, zero-padded (e.g. '0001'). Identifies the row in a rollout dump.",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    level: Optional[str] = Field(
        default=None,
        description=(
            "Difficulty tier: 'Easy', 'Medium', or 'High'. compute_metrics groups on this to emit the "
            "per-difficulty pass@k breakdown; a row without it drops out of that breakdown."
        ),
        json_schema_extra={"consumed_by": ["metrics"]},
    )
    tag: Optional[List[str]] = Field(
        default=None,
        description=(
            "Topic cluster labels (e.g. ['Limit'], ['Adjunction'], ['Monad']). Carried for slicing "
            "results by area; not read by verify() or today's metrics."
        ),
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    domain: Optional[List[str]] = Field(
        default=None,
        description="Upstream domain labels; 'Category' for every row in Part I (1-categories).",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    natural_language_statement: Optional[str] = Field(
        default=None,
        description=(
            "LaTeX prose statement of the theorem. Unused by the static pass@k protocol, which is "
            "formal-input only; required by the upstream natural-language and LeanBridge variants."
        ),
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    lean_toolchain: Optional[str] = Field(
        default=None,
        description=(
            "Pinned toolchain the row is meaningful under ('leanprover/lean4:v4.19.0'). Recorded per "
            "row so a rollout carries it; the sandbox, not this field, decides what actually compiles."
        ),
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    mathlib_version: Optional[str] = Field(
        default=None,
        description="Pinned Mathlib release the statements are written against ('v4.19.0').",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
