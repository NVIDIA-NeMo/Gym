# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Latency accounting behind the dispatch budget."""

from nemo_gym.rollout_collection import NG_ELAPSED_KEY, DispatchLatencyTracker, _observed_elapsed


def _tracker(*durations):
    t = DispatchLatencyTracker()
    for d in durations:
        t.record(d)
    return t


class TestQuantiles:
    def test_median_of_odd_sample(self):
        assert _tracker(10, 20, 30, 40, 50).quantile(0.5) == 30

    def test_no_samples_yields_none(self):
        assert DispatchLatencyTracker().quantile(0.5) is None

    def test_non_positive_durations_are_ignored(self):
        t = _tracker(10, 0, -5, 20)
        assert t.quantile(0.5) == 15


class TestDrainMargin:
    def test_explicit_value_wins(self):
        assert _tracker(10, 20, 30, 40, 50).drain_margin(99.0) == 99.0

    def test_adapts_to_p75_once_enough_samples(self):
        t = _tracker(10, 20, 30, 40, 50)
        assert t.drain_margin(None) == t.quantile(0.75)

    def test_withheld_below_five_samples(self):
        """Four points cannot describe a distribution; guessing would drain wrongly."""
        assert _tracker(10, 20, 30, 40).drain_margin(None) is None


class TestSummary:
    def test_reports_task_hours_and_drained(self):
        t = _tracker(3600, 3600)
        t.record_drained()
        out = t.summary()
        assert "task-hours" in out
        assert "Drained" in out
        assert t.drained == 1

    def test_empty_is_stated_not_crashed(self):
        assert "No completed rollouts" in DispatchLatencyTracker().summary()


class TestObservedElapsed:
    def test_reads_top_level(self):
        assert _observed_elapsed({NG_ELAPSED_KEY: 12.5}) == 12.5

    def test_reads_response_metadata(self):
        assert _observed_elapsed({"response": {"metadata": {NG_ELAPSED_KEY: 7.0}}}) == 7.0

    def test_missing_is_none(self):
        assert _observed_elapsed({}) is None

    def test_non_numeric_is_none(self):
        assert _observed_elapsed({NG_ELAPSED_KEY: "soon"}) is None

    def test_non_positive_is_none(self):
        assert _observed_elapsed({NG_ELAPSED_KEY: 0}) is None
        assert _observed_elapsed({NG_ELAPSED_KEY: -3}) is None
