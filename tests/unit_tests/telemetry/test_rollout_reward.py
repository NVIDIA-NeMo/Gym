# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``gym.rollout.rewarded_total``: scored attempts by reward range, against a real in-memory reader."""

import pytest

from nemo_gym.rollouts import rollout_collection
from nemo_gym.telemetry import GymSpanGroup, gym_metrics
from tests.unit_tests.telemetry.test_sandbox_active import collected_metrics  # noqa: F401 - fixture


pytest.importorskip("opentelemetry.sdk.metrics")

REWARDED = gym_metrics.ROLLOUT_REWARDED_INSTRUMENT
DISPATCH = gym_metrics.DISPATCH_NAME_ATTRIBUTE
MASKED = gym_metrics.ROLLOUT_MASKED_ATTRIBUTE
RANGE = gym_metrics.REWARD_RANGE_ATTRIBUTE


def _values(collected, name):
    return {tuple(sorted(p.attributes.items())): p.value for p in collected().get(name, [])}


def _key(masked=False, **attrs):
    return tuple(sorted({DISPATCH: "env", MASKED: masked, **attrs}.items()))


def _rollout_group_only(monkeypatch):
    monkeypatch.setattr(rollout_collection, "is_span_group_enabled", lambda group: group == GymSpanGroup.ROLLOUT)


def test_rewards_are_counted_by_range_including_penalties(collected_metrics):  # noqa: F811
    for reward in (-100.0, 0.0, 0.0, 0.3, 0.999, 1.0, 1.0, 2.0):
        gym_metrics.record_rollout_reward(reward, dispatch_name="env", masked=False)
    assert _values(collected_metrics, REWARDED) == {
        _key(**{RANGE: "negative"}): 1,
        _key(**{RANGE: "zero"}): 2,
        _key(**{RANGE: "partial"}): 2,
        _key(**{RANGE: "full"}): 2,
        _key(**{RANGE: "above_one"}): 1,
    }


def test_driver_records_scored_rewards_only(collected_metrics, monkeypatch):  # noqa: F811
    _rollout_group_only(monkeypatch)
    rollout_collection._record_rollout_outcome({"reward": 1.0}, "scored", "env")
    rollout_collection._record_rollout_outcome({"reward": 0}, "scored", "env")
    rollout_collection._record_rollout_outcome(
        {"reward": 0.5, rollout_collection.MASK_SAMPLE_KEY: True}, "scored", "env"
    )
    rollout_collection._record_rollout_outcome(
        {"reward": 0.0, rollout_collection.NG_FAILURE_CLASS_KEY: "agent_run_error"}, "failed", "env"
    )
    rollout_collection._record_rollout_outcome(
        {"reward": 0.0, rollout_collection.NG_FAILURE_CLASS_KEY: "kill_shaped"}, "omitted", "env"
    )
    assert _values(collected_metrics, REWARDED) == {
        _key(**{RANGE: "full"}): 1,
        _key(**{RANGE: "zero"}): 1,
        _key(masked=True, **{RANGE: "partial"}): 1,
    }


@pytest.mark.parametrize("result", [{}, {"reward": None}, {"reward": True}, {"reward": "1"}, {"reward": float("nan")}])
def test_a_result_without_a_numeric_reward_is_not_a_zero(collected_metrics, monkeypatch, result):  # noqa: F811
    _rollout_group_only(monkeypatch)
    rollout_collection._record_rollout_outcome(result, "scored", "env")
    assert _values(collected_metrics, REWARDED) == {}
    # The attempt itself is still counted.
    assert len(collected_metrics()[gym_metrics.ROLLOUT_COMPLETED_INSTRUMENT]) == 1


def test_nothing_is_recorded_when_the_rollout_group_is_off(collected_metrics, monkeypatch):  # noqa: F811
    monkeypatch.setattr(rollout_collection, "is_span_group_enabled", lambda group: False)
    rollout_collection._record_rollout_outcome({"reward": 1.0}, "scored", "env")
    assert _values(collected_metrics, REWARDED) == {}
