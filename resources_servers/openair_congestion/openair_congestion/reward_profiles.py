# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The single reward contract used by supported OpenAir replay episodes."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RewardProfile:
    version: str
    prb_pressure_threshold: float


def select_reward_profile(
    tier: str,
) -> RewardProfile:
    """Return the V1 contract for a supported synthetic tier."""

    if tier != "replay":
        raise ValueError(f"tier {tier!r} is not supported by this contribution")
    return RewardProfile(
        version="openair_v1",
        prb_pressure_threshold=0.85,
    )


__all__ = ["RewardProfile", "select_reward_profile"]
