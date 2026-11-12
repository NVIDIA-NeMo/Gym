# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Paper gameplay geometry, versioned separately from source-default suites.

Reference: https://arxiv.org/html/2506.02387v3#A6 (Appendix F.4 and F.5).
Seeds are implementation choices, not specified by the paper. For Poker, the
six seed values correspond to the six deals in the pinned author environment;
this is a balanced sampling scheme, not an author-provided seed list.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class PaperEpisode:
    game: str
    block: int
    index: int
    environment_seed: int
    model_seat: int


def paper_competitive_episodes(game: str) -> tuple[PaperEpisode, ...]:
    if game == "kuhn_poker":
        return tuple(
            PaperEpisode(game, block, seat * 60 + repeat * 6 + deal, deal, seat)
            for block in range(10)
            for seat in range(2)
            for repeat in range(10)
            for deal in range(6)
        )
    if game == "breakthrough":
        return tuple(PaperEpisode(game, 0, seat * 10 + seed, seed, seat) for seat in range(2) for seed in range(10))
    raise ValueError(f"No paper competitive schedule for {game!r}")
