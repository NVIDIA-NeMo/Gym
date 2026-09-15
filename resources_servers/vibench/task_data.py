# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-data schema for the vibench server.

Rows are flat top-level (no ``verifier_metadata``) and identify one ``(app, artifact)`` pair in
a ViBench checkout: which PRDs form the brief, which test plans grade it, and which asset
directories may be staged. Paths are stored relative to ``vibench_repo_root`` rather than
absolute so a dataset is not tied to one machine; the server rejects any that escape that root.

Required-ness mirrors ``VibenchTaskRequest`` (app.py), the shared request model behind both
``seed_session`` and ``verify``: ``app``, ``prd_files`` and ``test_plans`` are wire-required,
the rest carry defaults.

The asset split is the security-relevant part of this schema. ``asset_dirs`` holds fixtures the
PRD refers to and is staged into the *build* sandbox; ``test_assets_dir`` holds fixtures the
evaluation agent uploads while driving the finished app and is read only at grade time. Feeding
the latter to the builder would hand the model its own test fixtures.
"""

from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field


class TaskData(BaseModel):
    model_config = ConfigDict(extra="allow")

    app: str = Field(
        description="ViBench app directory under prds/, e.g. 'wedding'. Names the task with artifact.",
        json_schema_extra={"consumed_by": ["verify", "metrics", "provenance"]},
    )
    artifact: str = Field(
        default="mvp",
        description=(
            "Which artifact of the app to build: 'mvp', or a feature such as 'feature1' / "
            "'feature1-on_mvp'. Selects the prompt goal and the test-plan directory."
        ),
        json_schema_extra={"consumed_by": ["verify", "metrics", "provenance"]},
    )
    prd_files: List[str] = Field(
        description=(
            "PRD paths relative to vibench_repo_root, in order. A feature artifact prepends the "
            "MVP PRD so the brief carries the prior context. seed_session concatenates these."
        ),
        json_schema_extra={"consumed_by": ["prompt"]},
    )
    test_plans: List[str] = Field(
        description=(
            "Test-plan paths relative to vibench_repo_root. Each is graded in its own compose "
            "project and contributes one entry to reward_components; the reward is their mean."
        ),
        json_schema_extra={"consumed_by": ["verify", "metrics"]},
    )
    asset_dirs: List[str] = Field(
        default_factory=list,
        description=(
            "Static fixture directories the PRD refers to, staged into the build sandbox. "
            "Grader-only fixtures belong in test_assets_dir instead."
        ),
        json_schema_extra={"consumed_by": ["prompt"]},
    )
    test_assets_dir: Optional[str] = Field(
        default=None,
        description=(
            "Fixtures the evaluation agent uploads while driving the app, passed to "
            "run-evaluate-post-seeding.py. Never staged into the build sandbox."
        ),
        json_schema_extra={"consumed_by": ["verify"]},
    )
