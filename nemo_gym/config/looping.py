# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in completed-rollout repetition analysis settings."""

from pydantic import BaseModel, ConfigDict, Field


class LoopingConfig(BaseModel):
    """Resource settings for passive looping analysis; detector thresholds are fixed."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    workers: int = Field(
        default=1, ge=1, description="Processes in a live LoopingExecutor; saved evals share health_check_workers."
    )
    max_pending: int = Field(default=16, ge=1, description="Maximum admitted live scans, including running scans.")
