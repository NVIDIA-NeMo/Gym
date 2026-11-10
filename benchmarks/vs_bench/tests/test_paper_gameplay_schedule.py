# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from collections import Counter

import pytest

from benchmarks.vs_bench.paper_gameplay_schedule import paper_competitive_episodes


def test_poker_has_ten_balanced_blocks_with_every_deal_at_each_seat():
    rows = paper_competitive_episodes("kuhn_poker")
    assert len(rows) == 1200
    assert len({(x.block, x.index) for x in rows}) == 1200
    assert Counter((x.block, x.model_seat, x.environment_seed) for x in rows) == {
        (block, seat, deal): 10 for block in range(10) for seat in range(2) for deal in range(6)
    }


def test_breakthrough_has_twenty_balanced_games():
    rows = paper_competitive_episodes("breakthrough")
    assert len(rows) == 20
    assert Counter(x.model_seat for x in rows) == {0: 10, 1: 10}
    assert len({(x.block, x.index) for x in rows}) == 20
    with pytest.raises(ValueError):
        paper_competitive_episodes("hanabi")
