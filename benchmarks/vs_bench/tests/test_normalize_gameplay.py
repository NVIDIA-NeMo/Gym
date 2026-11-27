# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math

import pytest

from benchmarks.vs_bench.normalize_gameplay import ROUNDED_ANCHORS, normalized_gameplay_approx, normalized_pong_approx


def test_printed_anchors_and_unclipped_results():
    for game, (random, oracle) in ROUNDED_ANCHORS.items():
        assert normalized_gameplay_approx(game, random) == 0
        assert normalized_gameplay_approx(game, oracle) == pytest.approx(100)
        assert normalized_gameplay_approx(game, random - (oracle - random)) == pytest.approx(-100)
        assert normalized_gameplay_approx(game, oracle + (oracle - random)) == pytest.approx(200)


def test_pong_needs_both_components():
    assert normalized_pong_approx(0, 147.2) == 0
    assert normalized_pong_approx(3, 398) == 100
    assert normalized_pong_approx(0.1, 172.2) == pytest.approx(3.996810207336522)
    with pytest.raises(ValueError, match="requires model points"):
        normalized_gameplay_approx("atari_pong", 1)
    with pytest.raises(ValueError, match="finite"):
        normalized_pong_approx(math.nan, 100)
