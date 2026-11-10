# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-data schema for the VisGym resources server.

Rows are flat and are forwarded by ``visgym_agent`` as a ``VisGymTaskRow`` to
``/seed_session``. ``env_id`` and ``seed`` are wire-required; the remaining runtime fields use
the same defaults as that request model. ``task_idx`` is preparation-time provenance found in
the checked-in maze rows. It is intentionally optional because inline task rows do not require
an index and the runtime request model discards it after the task has been selected.

``responses_create_params`` is deliberately absent: it is a framework-owned row key removed by
``normalize_task_fields`` before validation, not task data owned by this resources server.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


VISGYM_BINARY_SUCCESS_FAMILIES = (
    "colorization",
    "counting",
    "fetch_pick_and_place",
    "fetch_reach",
    "jigsaw",
    "matchstick_equation",
    "matchstick_rotation",
    "maze_2d",
    "maze_3d",
    "mental_rotation_2d",
    "mental_rotation_3d_cube",
    "mental_rotation_3d_objaverse",
    "patch_reassembly",
    "referring_dot_pointing",
    "sliding_block",
    "video_unshuffle",
    "zoom_in_puzzle",
)
VISGYM_BINARY_SUCCESS_ENV_IDS = frozenset(
    f"{family}/{difficulty}" for family in VISGYM_BINARY_SUCCESS_FAMILIES for difficulty in ("easy", "hard")
)


class TaskData(BaseModel):
    model_config = ConfigDict(extra="allow")

    env_id: str = Field(
        description="VisGym registry identifier, for example 'maze_2d/easy'.",
        json_schema_extra={"consumed_by": ["verify", "metrics", "provenance"]},
    )
    env_kwargs: dict[str, Any] = Field(
        default_factory=dict,
        description="Per-task arguments passed to gymnasium.make(). Relative asset paths are resolved by the server.",
        json_schema_extra={"consumed_by": ["verify"]},
    )
    seed: int = Field(
        description="Seed passed to env.reset() and, when seed_key is set, to the environment constructor.",
        json_schema_extra={"consumed_by": ["verify", "provenance"]},
    )
    task_id: str | None = Field(
        default=None,
        description="Optional stable identifier for the generated task row.",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    act_grammar_regex: str | None = Field(
        default=None,
        description="Optional regular expression documenting the legal action-string grammar.",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    horizon_cap: int | None = Field(
        default=None,
        ge=1,
        description="Optional maximum number of attempted environment steps before server-side termination.",
        json_schema_extra={"consumed_by": ["verify"]},
    )
    task_metadata: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Open task-generation metadata. The optional reward_shaping block controls server-side "
            "distance-delta shaping; other values are retained for metrics and provenance."
        ),
        json_schema_extra={"consumed_by": ["verify", "metrics", "provenance"]},
    )
    init_state: dict[str, Any] | None = Field(
        default=None,
        description="Optional deterministic initial state passed to env.reset().",
        json_schema_extra={"consumed_by": ["verify"]},
    )
    seed_key: str | None = Field(
        default=None,
        description="Optional constructor-argument name that receives seed before gymnasium.make().",
        json_schema_extra={"consumed_by": ["verify"]},
    )
    prompt_kwargs: dict[str, Any] = Field(
        default_factory=dict,
        description="Optional keyword arguments passed to the environment's get_prompt() method.",
        json_schema_extra={"consumed_by": ["prompt"]},
    )
    task_idx: int | None = Field(
        default=None,
        ge=0,
        description="Optional row index emitted by manifest builders; not consumed after task selection.",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
