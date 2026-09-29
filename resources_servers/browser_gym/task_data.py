# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-data schema for the browser_gym server (Playwright CUA environment).

Rows nest every task-owned field inside a ``verifier_metadata`` bucket, so the schema is written
flat with ``legacy_location`` annotations. ``prepare_data.py`` emits all four fields from the gym's
``/api/v1/get_expected_state`` API.

``verify()`` (app.py) reads ``gym_url`` and ``task_id`` to post the browser's localStorage dump to
``{gym_url}/api/v1/get_actual_state``; when either is missing it logs a warning and returns
``reward=0.0`` rather than 422ing, so both are Optional here. ``start_url`` and ``viewport`` are
consumed by the browser agent when it seeds the Playwright session (``/seed_session``), not by
the verifier.
"""

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


_VM = {"legacy_location": "verifier_metadata"}


class Viewport(BaseModel):
    """Browser viewport the agent seeds the Playwright session with (``prepare_data.py`` defaults to 1280x720)."""

    model_config = ConfigDict(extra="allow")

    width: int = Field(
        description="Viewport width in pixels, e.g. 1280.",
        json_schema_extra={"consumed_by": ["run"]},
    )
    height: int = Field(
        description="Viewport height in pixels, e.g. 720.",
        json_schema_extra={"consumed_by": ["run"]},
    )


class TaskData(BaseModel):
    model_config = ConfigDict(extra="allow")

    task_id: Optional[str] = Field(
        default=None,
        description=(
            "Gym task identifier (the verifier key from get_expected_state, e.g. "
            "'CAL-DELETE-AN-EVENT-005'). verify() sends it to the gym's get_actual_state endpoint and "
            "uses it to map assertion categories for the per-category pass-rate metrics; missing -> "
            "reward 0.0."
        ),
        json_schema_extra={"consumed_by": ["verify", "metrics"], **_VM},
    )
    gym_url: Optional[str] = Field(
        default=None,
        description=(
            "Base URL of the gym deployment (e.g. 'https://nvidia.calendar.rlgym.turing.com'). verify() "
            "posts the localStorage dump to '{gym_url}/api/v1/get_actual_state'; missing -> reward 0.0."
        ),
        json_schema_extra={"consumed_by": ["verify"], **_VM},
    )
    start_url: Optional[str] = Field(
        default=None,
        description=(
            "URL the browser agent navigates to when seeding the Playwright session; prepare_data.py "
            "defaults it to gym_url when the task has no dedicated start page."
        ),
        json_schema_extra={"consumed_by": ["run"], **_VM},
    )
    viewport: Optional[Viewport] = Field(
        default=None,
        description="Browser viewport size the agent seeds the session with.",
        json_schema_extra={"consumed_by": ["run"], **_VM},
    )
