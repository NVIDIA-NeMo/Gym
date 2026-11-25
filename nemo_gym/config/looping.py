# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in completed-rollout repetition analysis settings."""

from pydantic import BaseModel, ConfigDict


class LoopingConfig(BaseModel):
    """Passive looping analysis settings; detector thresholds are fixed."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
