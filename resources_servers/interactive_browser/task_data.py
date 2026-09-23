# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-data schema for the interactive_browser server.

A row says where the episode starts and how it is graded. Both are read at
``/seed_session`` (app.py:181), which stores the grading spec on the session; ``/verify``
takes no task fields of its own.

The grading keys live inside the legacy ``verifier_metadata`` bucket, so they carry
``legacy_location``. ``initial_url`` is top-level on the wire and does not.

Exactly one grading key is expected per row: ``_score`` tries ``answer_equals``,
``final_url``, ``url_contains``, then ``dom_contains``, and raises when a row carries
none of them rather than scoring every rollout zero.
"""

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class TaskData(BaseModel):
    model_config = ConfigDict(extra="allow")

    initial_url: str = Field(
        default="about:blank",
        description=(
            "Where the browser opens. A relative path (e.g. 'site/index.html') resolves against "
            "the server directory, so example rows do not hard-code machine paths."
        ),
        json_schema_extra={"consumed_by": ["prompt"]},
    )
    final_url: Optional[str] = Field(
        default=None,
        description="Scores 1.0 when the graded page's URL equals this exactly.",
        json_schema_extra={"consumed_by": ["verify"], "legacy_location": "verifier_metadata"},
    )
    url_contains: Optional[str] = Field(
        default=None,
        description="Scores 1.0 when the graded page's URL contains this substring.",
        json_schema_extra={"consumed_by": ["verify"], "legacy_location": "verifier_metadata"},
    )
    dom_contains: Optional[str] = Field(
        default=None,
        description=(
            "Scores 1.0 when the graded page's title plus visible text contains this, "
            "case-insensitively. Grades the page, so it cannot check what the model reported."
        ),
        json_schema_extra={"consumed_by": ["verify"], "legacy_location": "verifier_metadata"},
    )
    answer_equals: Optional[str] = Field(
        default=None,
        description=(
            "Scores 1.0 when the answer passed to browser_finish equals this after stripping. "
            "The key to use when the task asks the model to report something."
        ),
        json_schema_extra={"consumed_by": ["verify"], "legacy_location": "verifier_metadata"},
    )
