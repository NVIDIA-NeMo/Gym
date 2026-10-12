# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pinned task identities consumed by the TB4 resources runner.

The agent adapter supplies rollout/session identity and capture controls at run
time. Those controls are not required in dataset rows.
"""

from pydantic import BaseModel, ConfigDict, Field


class TaskData(BaseModel):
    """Dataset-owned fields; the runner checks identities against its manifest."""

    model_config = ConfigDict(extra="allow")

    task_id: str | None = Field(
        default=None, description="Namespaced task identity, such as terminal-bench/ks-solver-cpp."
    )
    task_name: str = Field(description="Namespaced task name, such as terminal-bench/ks-solver-cpp.")
    task_ref: str = Field(description="Pinned task package digest from the benchmark manifest.")
    dataset_ref: str = Field(description="Pinned dataset digest from the benchmark manifest.")
