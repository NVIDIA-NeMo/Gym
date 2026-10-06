# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-data schema for the kidbench server (KIDBench child-facing response quality).

Fields are top-level row columns, written by ``benchmarks/kidbench/prepare.py``. One schema covers
both tracks: a single-turn row carries ``prompt`` and ``prompt_type``, a multi-turn row carries
``scenario`` and ``child_goal`` instead, and everything else is shared. Required-ness mirrors
``KidbenchVerifyRequest``: only ``kidbench_id`` and ``category`` are required on the wire.

Nothing here holds upstream text beyond the prompt itself. The judge rubric, category rules and
country rules are read from the pinned upstream checkout at verify time, keyed by ``track``,
``category`` and ``country``.
"""

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class TaskData(BaseModel):
    model_config = ConfigDict(extra="allow")

    kidbench_id: str = Field(
        description=(
            "Identifies one cell: the track, condition shape, category and example index, e.g. "
            "`single-without_cues-english-true-none-moral_reasoning-004`. Unique within a track."
        ),
        json_schema_extra={"consumed_by": ["verify", "provenance"]},
    )
    category: str = Field(
        description=(
            "One of upstream's ten categories (UNICEF's 4Cs risk taxonomy plus a benign control). "
            "Selects the category rules the judge is given, and compute_metrics groups on it."
        ),
        json_schema_extra={"consumed_by": ["verify", "metrics"]},
    )
    track: Literal["single_turn", "multi_turn"] = Field(
        default="single_turn",
        description=(
            "Which protocol the row belongs to. Routes the row to that track's judge and rubric; the "
            "two tracks use different judges upstream, so their scores are not comparable."
        ),
        json_schema_extra={"consumed_by": ["verify"]},
    )
    condition: str = Field(
        default="",
        description=(
            "The cue condition: `no_cue`, `implicit_cue`, `explicit_age`, `cultural` or `cross_lingual` "
            "for single turn; `without_age` or `with_age` for multi turn. compute_metrics groups on it, "
            "which is what makes the paper's cue gains readable from the aggregate."
        ),
        json_schema_extra={"consumed_by": ["verify", "metrics"]},
    )
    country: Optional[str] = Field(
        default=None,
        description=(
            "Set only in the cultural cells. Selects the country rules and switches on the sixth "
            "rubric dimension, `cultural_alignment`."
        ),
        json_schema_extra={"consumed_by": ["verify"]},
    )
    language: str = Field(
        default="english",
        description="Language of the prompt and of the expected reply: english, urdu, hindi or mandarin.",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    prompt: str = Field(
        default="",
        description=(
            "Single turn only: the child's question as sent, which the judge reads beside the reply. "
            "Empty on multi-turn rows, where the child-actor agent generates the child's turns."
        ),
        json_schema_extra={"consumed_by": ["verify"]},
    )
    prompt_type: Optional[str] = Field(
        default=None,
        description="Single turn only: `without_cues` (plain wording) or `with_cues` (a child's own phrasing).",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    scenario: Optional[str] = Field(
        default=None,
        description=(
            "Multi turn only: the situation the child actor is given. Shown to the actor, never to the "
            "model under test."
        ),
        json_schema_extra={"consumed_by": ["prompt"]},
    )
    child_goal: Optional[str] = Field(
        default=None,
        description="Multi turn only: what the child actor keeps pressing for across the five turns.",
        json_schema_extra={"consumed_by": ["prompt"]},
    )
    age_conditioned: Optional[bool] = Field(
        default=None,
        description="Whether the responder's system prompt states that the user is a child aged 7-11.",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    risk_class: Optional[str] = Field(
        default=None,
        description="The risk class the category belongs to: Content, Contact, Conduct, or Control for the benign category.",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    example_index: Optional[int] = Field(
        default=None,
        description=(
            "Index of the base question within its category. With `category` it identifies the same "
            "question across all of its condition cells."
        ),
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    upstream_file: Optional[str] = Field(
        default=None,
        description="The upstream dataset file the row was expanded from, e.g. `with_cues/english_true_india.json`.",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    upstream_revision: Optional[str] = Field(
        default=None,
        description="The upstream git revision the row was built from, so a rollout names the data it scored.",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
