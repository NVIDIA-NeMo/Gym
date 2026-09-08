# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-data schema for the ruler2 server.

Task fields ride at the row top level; there is no verifier_metadata. Rows are heterogeneous
within one file on the documented in-band discriminator ``eval_type``: soft string-match rows
(eval_type="ruler2") carry ``expected_answer`` as a list of reference strings, multichoice rows
(eval_type="multichoice") carry a bare single-letter string — hence the discriminated union; any
single strict type for ``expected_answer`` is wrong.

Deliberately slightly stricter than the wire on the discriminator: ``Ruler2RunRequest`` (app.py)
types ``eval_type`` as an open ``str = "ruler2"`` (verify() routes eval_type=="multichoice" to
exact letter match and EVERY other value — including a missing one — to the default soft-match
route) and ``expected_answer`` as ``Any`` with defensive isinstance coercion. Every committed row
carries an explicit eval_type of one of the two documented values, so the union dispatches on the
documented contract; the per-variant expected_answer typing is the point of the union.
"""

from typing import Annotated, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field


class _Ruler2TaskBase(BaseModel):
    model_config = ConfigDict(extra="allow")

    match_type: str = Field(
        default="all",
        description=(
            "Soft string-match dispatch on the eval_type='ruler2' route: 'all' (avg over refs), "
            "'part' (max over refs, Document-N headers stripped), or '2steps' (avg, last paragraph "
            "only). Any other value raises ValueError in verify(). Ignored on the multichoice route."
        ),
        json_schema_extra={"consumed_by": ["verify"]},
    )
    task: Optional[str] = Field(
        default=None,
        description=(
            "RULER2 subtask name (e.g. 'mk_niah_basic', 'qa_basic'); not scored — consumed by "
            "compute_metrics() as the per-task subset key and the 12-subtask suite mean."
        ),
        json_schema_extra={"consumed_by": ["metrics"]},
    )
    length: Optional[int] = Field(
        default=None,
        description="Context-length bucket of the row; ride-along only, echoed through the verify response.",
        json_schema_extra={"consumed_by": ["provenance"]},
    )


class Ruler2StringMatchTaskData(_Ruler2TaskBase):
    """Soft string-match rows: score = match_type aggregation of max(substring, 1 - WER) per ref."""

    eval_type: Literal["ruler2"] = Field(
        default="ruler2",
        description="Route selector: the default soft string-match route.",
        json_schema_extra={"consumed_by": ["verify"]},
    )
    expected_answer: List[str] = Field(
        description=(
            "Reference strings the soft string match aggregates over; always a list by construction "
            "in prepare.py (single-answer tasks wrap the answer in a one-element list). Wire type is "
            "Any — verify() defensively wraps a non-list as [str(...)]."
        ),
        json_schema_extra={"consumed_by": ["verify"]},
    )


class Ruler2MultichoiceTaskData(_Ruler2TaskBase):
    """Multichoice rows: exact match of the extracted answer against a single uppercase letter."""

    eval_type: Literal["multichoice"] = Field(
        description="Route selector: exact single-letter match.",
        json_schema_extra={"consumed_by": ["verify"]},
    )
    expected_answer: str = Field(
        description=("Gold answer letter (e.g. 'C'); verify() applies str(...).strip().upper(). Wire type is Any."),
        json_schema_extra={"consumed_by": ["verify"]},
    )


TaskData = Annotated[
    Union[Ruler2StringMatchTaskData, Ruler2MultichoiceTaskData],
    Field(discriminator="eval_type"),
]
