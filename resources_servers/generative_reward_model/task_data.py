# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-data schema for the generative_reward_model server.

Rows carry the human ground truth that ``verify()`` scores the policy's judging verdict
against: an overall score/ranking triple and a per-rubric list. Both are read top-level (no
``verifier_metadata`` wrapper), so no field carries a ``legacy_location`` marker.

Required-ness mirrors ``GenerativeRewardModelRunRequest`` in app.py, not what ``verify()``
happens to read: ``id`` is required there, while both ground-truth fields default to ``None``.
Inside them every score and ranking is optional because ``verify()`` reaches them with
``.get()`` and penalizes only the keys actually present — a row carrying just ``ranking``
contributes only the ranking term. ``rubric_id`` is the exception: it is indexed directly
(``r["rubric_id"]``) when matching predicted rubrics to ground truth, so it is required.
"""

from typing import List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field


class OverallGroundTruth(BaseModel):
    """Ground truth for the judge's overall verdict on the response pair."""

    model_config = ConfigDict(extra="allow")

    score_1: Optional[float] = Field(
        default=None,
        description="Individual helpfulness of response 1, 1-5, higher is better.",
        json_schema_extra={"consumed_by": ["verify", "metrics"]},
    )
    score_2: Optional[float] = Field(
        default=None,
        description="Individual helpfulness of response 2, 1-5, higher is better.",
        json_schema_extra={"consumed_by": ["verify", "metrics"]},
    )
    ranking: Optional[float] = Field(
        default=None,
        description=(
            "Relative ranking, 1-6: 1 = response 1 much better, 6 = response 2 much better. "
            "The 3.5 midpoint splits the two directions for ranking binary accuracy."
        ),
        json_schema_extra={"consumed_by": ["verify", "metrics"]},
    )


class RubricGroundTruth(BaseModel):
    """Ground truth for one rubric of the judge's per-rubric breakdown."""

    model_config = ConfigDict(extra="allow")

    rubric_id: int = Field(
        description=(
            "Identifier matching this rubric to the model's predicted rubric. Need not start "
            "at 1 or be contiguous, but the predicted and ground-truth ID sets must match "
            "exactly or the rollout scores as a parse failure."
        ),
        json_schema_extra={"consumed_by": ["verify"]},
    )
    score_1: Optional[float] = Field(
        default=None,
        description="Response 1's score for this rubric, 1-5, higher is better.",
        json_schema_extra={"consumed_by": ["verify", "metrics"]},
    )
    score_2: Optional[float] = Field(
        default=None,
        description="Response 2's score for this rubric, 1-5, higher is better.",
        json_schema_extra={"consumed_by": ["verify", "metrics"]},
    )
    ranking: Optional[float] = Field(
        default=None,
        description="Relative ranking for this rubric, 1-6, same scale as the overall ranking.",
        json_schema_extra={"consumed_by": ["verify", "metrics"]},
    )


class TaskData(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: Union[int, str] = Field(
        description="Stable per-sample identifier. Required by the wire model; unread by verify().",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    ground_truth_overall: Optional[OverallGroundTruth] = Field(
        default=None,
        description=(
            "Overall human verdict. Each present key contributes an L1 penalty against the "
            "model's prediction; absent keys are skipped."
        ),
        json_schema_extra={"consumed_by": ["verify", "metrics"]},
    )
    ground_truth_rubric_scores: Optional[List[RubricGroundTruth]] = Field(
        default=None,
        description=(
            "Per-rubric human verdicts. Their mean penalty is scaled by rubric_weight and added "
            "to the overall penalty."
        ),
        json_schema_extra={"consumed_by": ["verify", "metrics"]},
    )
    dataset: Optional[str] = Field(
        default=None,
        description=(
            "Source-dataset label (e.g. 'rlhf2.4_combined'). Never read by any server code; "
            "carried for provenance only."
        ),
        json_schema_extra={"consumed_by": ["provenance"]},
    )
