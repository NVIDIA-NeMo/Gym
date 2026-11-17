# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-data schema for the agentic_turn_simulation_judge server (the policy writes the ATS judge's verdict).

Task fields ride at the row top level; there is no verifier_metadata. Required-ness mirrors
``AgenticTurnSimulationJudgeRunRequest`` (app.py): only ``gold_ladder`` is required on the wire. ``scripts/validate_rows.py``
adds the checks a built data file must also pass (row_id/group_id present, prompt shape, enum values, duplicates,
group leakage across splits).
"""

from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class GoldFields(BaseModel):
    """The reference verdict's fields (majority over the reference passes); values follow the ATS schema enums."""

    model_config = ConfigDict(extra="allow")

    addresses_concern: Optional[str] = None
    correct: Optional[str] = None
    blocked_by: Optional[str] = None
    request_questionable: Optional[str] = None
    discard: Optional[str] = None
    discard_reason: Optional[str] = None


class LabelInfo(BaseModel):
    """Where the label came from."""

    model_config = ConfigDict(extra="allow")

    policy: Optional[str] = Field(default=None, description="Label policy, e.g. unanimous >= 3 passes.")
    tier: Optional[str] = Field(default=None, description="Label tier (e.g. A unanimous, B 1-2 passes, hand).")
    judge: Optional[str] = Field(default=None, description="Reference judge model.")
    rubric_tag: Optional[str] = None
    rubric_sha256: Optional[str] = None
    set: Optional[str] = Field(default=None, description="Source request set.")
    tag: Optional[str] = Field(default=None, description="Judge-run tag.")


class MetaInfo(BaseModel):
    """Row provenance; compute_metrics slices by source, family, harness and reads hand_ladder."""

    model_config = ConfigDict(extra="allow")

    source: Optional[str] = None
    harness: Optional[str] = None
    family: Optional[str] = None
    domain: Optional[str] = None
    split: Optional[str] = None
    prompt_tokens_est: Optional[int] = None
    hand_ladder: Optional[Literal["yes", "no"]] = Field(
        default=None, description="A human label of the ladder, when one exists (eval rows only)."
    )
    expect: Optional[Any] = None
    stratum: Optional[str] = None


class TaskData(BaseModel):
    model_config = ConfigDict(extra="allow")

    task: Literal["verdict"] = Field(
        default="verdict",
        description="Row kind. Only 'verdict' (write the judge's JSON verdict) exists; 'pairwise' is reserved.",
        json_schema_extra={"consumed_by": ["verify"]},
    )
    row_id: Optional[str] = Field(
        default=None, description="Row identifier.", json_schema_extra={"consumed_by": ["provenance"]}
    )
    group_id: Optional[str] = Field(
        default=None,
        description="Transcript/session key: rows sharing it stay in one split.",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    gold_ladder: Literal["yes", "no"] = Field(
        description="Reference correctness ladder: the reward label.",
        json_schema_extra={"consumed_by": ["verify", "metrics"]},
    )
    gold_p_yes: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Share of usable reference passes with ladder yes (reward_mode soft).",
        json_schema_extra={"consumed_by": ["verify"]},
    )
    gold_n: Optional[int] = Field(
        default=None,
        ge=0,
        description="Number of reference passes behind the label.",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    gold_fields: Optional[GoldFields] = Field(
        default=None,
        description="Reference verdict fields (field matches, optional field_bonus).",
        json_schema_extra={"consumed_by": ["verify", "metrics"]},
    )
    label: Optional[LabelInfo] = Field(default=None, json_schema_extra={"consumed_by": ["metrics", "provenance"]})
    meta: Optional[MetaInfo] = Field(default=None, json_schema_extra={"consumed_by": ["metrics", "provenance"]})
