# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Test paper-style aggregate from the authors' returned Pong components."""

import pytest

from benchmarks.vs_bench.prepare import GAMES
from responses_api_agents.vs_bench_gameplay.app import VSBenchGameplayAgent


def result(game):
    return {
        "verifier_metadata": {"game": game},
        "reward": 0.0,
        "terminal_returns": [[0.0, 3.0]],
        "episode_steps": [398],
        "model_seats": [0, 1] if game in ("coin_dilemma", "monster_hunt", "battle_of_the_colors") else [1],
    }


def test_pong_uses_model_seat_and_only_reports_complete_macro():
    metrics = VSBenchGameplayAgent.compute_metrics(None, [[result("atari_pong")]])
    assert metrics["vs_bench/gameplay/atari_pong/model_points"] == 3
    assert metrics["vs_bench/gameplay/atari_pong/survival_steps"] == 398
    assert metrics["vs_bench/gameplay/atari_pong/normalized_approx"] == pytest.approx(100)
    assert "vs_bench/gameplay/normalized_equal_game_mean_approx" not in metrics
    metrics = VSBenchGameplayAgent.compute_metrics(None, [[result(game)] for game in GAMES])
    assert "vs_bench/gameplay/normalized_equal_game_mean_approx" not in metrics


def test_poker_complete_and_partial_coverage_and_duplicate_rejection():
    from dataclasses import asdict

    from benchmarks.vs_bench.paper_gameplay_schedule import paper_competitive_episodes

    rows = []
    for episode in paper_competitive_episodes("kuhn_poker"):
        row = result("kuhn_poker")
        row["verifier_metadata"].update(protocol="paper_competitive_v1", paper_episode=asdict(episode))
        row["model_seats"] = [episode.model_seat]
        row["reward"] = 0.0
        rows.append([row])
    partial = VSBenchGameplayAgent.compute_metrics(None, rows[:-1])
    assert partial["vs_bench/gameplay/kuhn_poker/episodes"] == 1199
    assert "vs_bench/gameplay/kuhn_poker/normalized_approx" not in partial
    full = VSBenchGameplayAgent.compute_metrics(None, rows)
    assert full["vs_bench/gameplay/kuhn_poker/normalized_approx"] == pytest.approx(100)
    with pytest.raises(ValueError, match="Duplicate"):
        VSBenchGameplayAgent.compute_metrics(None, rows + rows[:1])


def test_mixed_motive_retains_each_seat_and_scores_their_mean():
    row = result("coin_dilemma")
    row["terminal_returns"] = [[2.0, 10.0], [4.0, 2.0]]
    row["reward"] = 3.0
    metrics = VSBenchGameplayAgent.compute_metrics(None, [[row]])
    assert metrics["vs_bench/gameplay/coin_dilemma/raw_seat_0_return"] == 3
    assert metrics["vs_bench/gameplay/coin_dilemma/raw_seat_1_return"] == 6
    assert metrics["vs_bench/gameplay/coin_dilemma/raw_two_seat_mean"] == 4.5
    assert metrics["vs_bench/gameplay/coin_dilemma/normalized_approx"] == pytest.approx(100 * 4.7 / 28.1)


def test_complete_paper_aggregate_excludes_legacy_or_incomplete_suites():
    from dataclasses import asdict

    from benchmarks.vs_bench.paper_gameplay_schedule import paper_competitive_episodes

    tasks = []
    for game in GAMES:
        if game in ("kuhn_poker", "breakthrough"):
            for episode in paper_competitive_episodes(game):
                row = result(game)
                row["verifier_metadata"].update(protocol="paper_competitive_v1", paper_episode=asdict(episode))
                row["model_seats"] = [episode.model_seat]
                tasks.append([row])
        else:
            row = result(game)
            row["verifier_metadata"]["protocol"] = "paper_competitive_v1"
            row["terminal_returns"] = row["terminal_returns"] * 10
            row["episode_steps"] = row["episode_steps"] * 10
            tasks.append([row])
    metrics = VSBenchGameplayAgent.compute_metrics(None, tasks)
    key = "vs_bench/gameplay/normalized_equal_game_mean_approx"
    assert metrics[key] == pytest.approx(sum(metrics[f"vs_bench/gameplay/{g}/normalized_approx"] for g in GAMES) / 10)
    assert key not in VSBenchGameplayAgent.compute_metrics(None, tasks[:-1])
    tasks[-1][0]["verifier_metadata"].pop("protocol")
    assert key not in VSBenchGameplayAgent.compute_metrics(None, tasks)


def test_headline_is_exposed_only_when_present():
    key = "vs_bench/gameplay/normalized_equal_game_mean_approx"
    assert VSBenchGameplayAgent.get_key_metrics(None, {"vs_bench/gameplay/games_completed": 9}) == {}
    assert VSBenchGameplayAgent.get_key_metrics(None, {key: 12.5}) == {key: 12.5}
