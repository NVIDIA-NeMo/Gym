# SPDX-License-Identifier: Apache-2.0
"""Central reward-profile selection for live and replay OpenAir episodes."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RewardProfile:
    version: str
    prb_pressure_threshold: float


def select_reward_profile(
    tier: str,
    *,
    connected_t1_runner: bool = False,
) -> RewardProfile:
    """Select the versioned reward contract for an episode."""

    normalized = tier.upper()
    if normalized == "T2":
        return RewardProfile(
            version="openair_t2_v3",
            prb_pressure_threshold=0.85,
        )
    if normalized == "T1" and connected_t1_runner:
        return RewardProfile(
            version="openair_v1",
            prb_pressure_threshold=0.08,
        )
    return RewardProfile(
        version="openair_v1",
        prb_pressure_threshold=0.85,
    )


__all__ = ["RewardProfile", "select_reward_profile"]
