# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import pytest

from resources_servers.eqbench4.metrics import (
    POLICY,
    dropped_conversation_alerts,
    dropped_conversation_metrics,
    eqbench4_metrics,
    eqbench4_repeat_metrics,
    key_metrics,
    policy_outcomes,
)
from resources_servers.eqbench4.rating import Outcome, fit_ratings


REFERENCE = [Outcome("a", "b", 0.7), Outcome("b", "c", 0.6), Outcome("a", "c", 0.8)]


def _rollout(status: str, turns: int, comparisons: list[tuple[str, float]], traits: dict | None = None) -> dict:
    return {
        "conversation_status": status,
        "policy_turns": turns,
        "comparisons": [{"opponent": opponent, "fraction": fraction} for opponent, fraction in comparisons],
        "trait_scores": traits,
    }


def test_conversation_statistics_and_traits_without_comparisons() -> None:
    tasks = [
        [_rollout("persona_ended", 5, [], {"warmth": 6.0, "challenge": 2.0})],
        [_rollout("completed", 8, [], {"warmth": 8.0})],
    ]
    assert eqbench4_metrics(tasks, REFERENCE, resamples=5) == {
        "eqbench4_judged_conversations": 2,
        "eqbench4_comparisons": 0,
        "eqbench4_persona_ended_rate": 0.5,
        "eqbench4_mean_policy_turns": 6.5,
        "eqbench4_trait/challenge": 2.0,
        "eqbench4_trait/warmth": 7.0,
    }
    assert eqbench4_metrics([], REFERENCE, resamples=5) == {
        "eqbench4_judged_conversations": 0,
        "eqbench4_comparisons": 0,
    }
    assert eqbench4_repeat_metrics(tasks, REFERENCE) == {}


def test_rating_uses_reference_and_policy_comparisons_together() -> None:
    tasks = [
        [_rollout("completed", 8, [("a", 0.3), ("c", 0.6)])],
        [_rollout("persona_ended", 4, [("b", 0.0)])],
    ]
    outcomes = policy_outcomes([rollout for task in tasks for rollout in task])
    assert outcomes == [Outcome(POLICY, "a", 0.3), Outcome(POLICY, "c", 0.6), Outcome(POLICY, "b", 0.0)]
    expected = fit_ratings(["a", "b", "c", POLICY], [*REFERENCE, *outcomes])

    metrics = eqbench4_metrics(tasks, REFERENCE, resamples=10)
    assert metrics["eqbench4_elo"] == pytest.approx(expected[POLICY])
    assert metrics["eqbench4_rank"] == 1 + sum(value > expected[POLICY] for value in expected.values())
    assert metrics["eqbench4_win_rate"] == pytest.approx(0.3)
    assert metrics["eqbench4_persona_ended_rate"] == 0.5
    assert list(name for name in metrics if name.startswith("eqbench4_ladder/")) == [
        f"eqbench4_ladder/{model}" for model in sorted("abc", key=lambda model: -expected[model])
    ]
    assert eqbench4_repeat_metrics(tasks, REFERENCE) == {
        "eqbench4_elo": pytest.approx(expected[POLICY]),
        "eqbench4_win_rate": pytest.approx(0.3),
    }


def test_key_metrics_select_the_headline() -> None:
    assert key_metrics({"eqbench4_elo": 1100.0, "eqbench4_conversations": 3}) == {"eqbench4_elo": 1100.0}


def test_dropped_conversations_and_alerts() -> None:
    rows = [
        {"conversation_status": "completed"},
        {"conversation_status": "policy_failed", "policy_stop_reason": "output_budget"},
        {"conversation_status": "policy_failed"},
        {"conversation_status": "persona_failed"},
    ]
    metrics = dropped_conversation_metrics(rows)
    assert metrics["eqbench4_policy_stopped_conversations"] == 2
    assert metrics["eqbench4_policy_stopped/output_budget"] == 1
    assert metrics["eqbench4_policy_stopped/empty_reply"] == 1
    assert metrics["eqbench4_policy_stopped_rate/context_limit"] == 0.0
    assert dropped_conversation_alerts(metrics) == [
        "EQ-Bench 4: the policy stopped replying in 2 of 4 conversations (50.0%), which are left out of the score: "
        "output_budget 1 (raise max_output_tokens or lower the policy's reasoning effort); "
        "empty_reply 1 (inspect the policy's replies in the rollouts; a server reasoning parser hides the whole reply "
        "as reasoning when the model does not close </think>).",
        "EQ-Bench 4: the persona simulator gave no usable reply in 1 of 4 conversations (25.0%), which are left out "
        "of the score; check the persona model, which should be the leaderboard's for comparable ratings.",
    ]
    empty = dropped_conversation_metrics([])
    assert empty["eqbench4_policy_stopped_rate"] == 0.0 and dropped_conversation_alerts(empty) == []
