# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Approximate VS-Bench gameplay scores from rounded paper references.

Reference: https://arxiv.org/html/2506.02387v3#A6 (Appendix F, Table 6).
These anchors are rounded; results are not exact official leaderboard scores.
"""

import math


# Random and oracle returns, as printed in the paper.
ROUNDED_ANCHORS = {
    "hanabi": (0.0, 24.0),
    "overcooked": (0.2, 40.0),
    "knights_archers_zombies": (1.0, 20.0),
    "breakthrough": (-1.0, 1.0),
    "kuhn_poker": (-0.1, 0.0),
    "simple_push": (-16.3, 0.0),
    "coin_dilemma": (-0.2, 27.9),
    "monster_hunt": (-20.2, 176.2),
    "battle_of_the_colors": (0.3, 59.7),
}


def _finite(value: float) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("Gameplay values must be finite")
    return value


def normalized_gameplay_approx(game: str, raw_return: float) -> float:
    """Normalize a non-Pong raw return; keep negative and above-oracle values."""
    if game == "atari_pong":
        raise ValueError("Pong requires model points and survival steps, not raw return")
    random, oracle = ROUNDED_ANCHORS[game]
    return 100 * (_finite(raw_return) - random) / (oracle - random)


def normalized_pong_approx(model_points: float, survival_steps: float) -> float:
    """Use the paper's separate 90% points and 10% survival components."""
    return 100 * (0.9 * _finite(model_points) / 3 + 0.1 * (_finite(survival_steps) - 147.2) / (398 - 147.2))
