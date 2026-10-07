# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Latency accounting behind the dispatch budget."""

from nemo_gym.rollout_collection import NG_ELAPSED_KEY, DispatchLatencyTracker, observed_elapsed


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

    def test_recording_order_does_not_matter(self):
        t = _tracker(50, 10, 40, 20, 30)
        assert (t.quantile(0.5), t.quantile(0.75)) == (30, 40)


class TestDrainMargin:
    def test_explicit_value_wins(self):
        assert _tracker(10, 20, 30, 40, 50).drain_margin(99.0) == 99.0

    def test_adapts_to_p75_once_enough_samples(self):
        t = _tracker(10, 20, 30, 40, 50)
        assert t.drain_margin(None) == t.quantile(0.75)

    def test_withheld_below_five_samples(self):
        """Four points cannot describe a distribution; guessing would drain wrongly."""
        assert _tracker(10, 20, 30, 40).drain_margin(None) is None


class TestRecordFailure:
    def test_a_failure_that_ran_long_raises_the_margin(self):
        t = _tracker(10, 20, 30, 40, 50)
        t.record_failure(600)
        assert t.quantile(1.0) == 600
        assert t.drain_margin(None) > 40

    def test_an_instant_failure_cannot_drag_the_margin_down(self):
        t = _tracker(600, 600, 600, 600, 600)
        for _ in range(20):
            t.record_failure(0.2)
        assert t.drain_margin(None) == 600

    def test_failures_before_any_completion_are_ignored(self):
        t = DispatchLatencyTracker()
        t.record_failure(3600)
        assert t.quantile(0.5) is None


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

    def test_all_drained_reports_count_without_completions(self):
        t = DispatchLatencyTracker()
        t.record_drained()
        t.record_drained()

        out = t.summary()

        assert "No completed rollouts" in out
        assert "Drained (not dispatched, no time left in the budget): 2" in out


class TestObservedElapsed:
    def test_reads_top_level(self):
        assert observed_elapsed({NG_ELAPSED_KEY: 12.5}) == 12.5

    def test_reads_response_metadata(self):
        assert observed_elapsed({"response": {"metadata": {NG_ELAPSED_KEY: 7.0}}}) == 7.0

    def test_missing_is_none(self):
        assert observed_elapsed({}) is None

    def test_non_numeric_is_none(self):
        assert observed_elapsed({NG_ELAPSED_KEY: "soon"}) is None

    def test_non_positive_is_none(self):
        assert observed_elapsed({NG_ELAPSED_KEY: 0}) is None
        assert observed_elapsed({NG_ELAPSED_KEY: -3}) is None


class TestTimingSummary:
    def test_empty_without_thresholds(self):
        assert DispatchLatencyTracker().timing_summary() == ""

    def test_start_coverage_and_timed_out_rows(self):
        t = DispatchLatencyTracker(total=4, start_report_within_s=1800, long_rollout_s=9000)
        t0 = t._t0
        for offset in (60, 600, 1500, 2400):  # three in the window, one late
            t.record_start(t0 + offset)
        for result in ({"reward": 1.0}, {"reward": 0.0, "timed_out": 1}, {"reward": 0.0}, {"reward": 1.0}):
            t.record_outcome(result)
        for seconds in (1200, 10800, 3000, 9500):
            t.record(seconds)
        table = t.timing_summary()
        assert "all rollouts started by" in table and "40.0 min" in table and "WARNING" in table
        assert "started within 30 min" in table and "75.0%" in table and "WARNING" in table
        assert "ended by wall-clock limit" in table and "25.0% (1)" in table
        assert "longer than 150 min" in table and "50.0%" in table

    def test_healthy_run_has_no_warnings(self):
        t = DispatchLatencyTracker(total=2, start_report_within_s=1800, long_rollout_s=9000)
        for offset in (10, 20):
            t.record_start(t._t0 + offset)
        for _ in range(2):
            t.record_outcome({"reward": 1.0})
            t.record(100)
        assert "WARNING" not in t.timing_summary()
        assert "[OK] Start coverage" in t.summary()

    def test_live_warning_fires_once_on_first_late_start(self, capsys):
        t = DispatchLatencyTracker(total=3, start_report_within_s=60, start_report_min_fraction=0.99)
        t.record_start(t._t0 + 1)
        t.record_start(t._t0 + 120)
        t.record_start(t._t0 + 130)
        out = capsys.readouterr().out
        assert out.count("Start coverage") == 1 and "[WARNING]" in out and "33.3%" in out

    def test_timeout_failure_class_counts_as_timed_out(self):
        t = DispatchLatencyTracker(total=1, start_report_within_s=60)
        t.record_outcome({"_ng_failure_class": "timeout_exceeded"})
        assert "100.0% (1)" in t.timing_summary()


class TestPartialStart:
    def test_partial_start_reports_count_and_waits_for_the_window(self):
        t = DispatchLatencyTracker(total=100, start_report_within_s=1800)
        for offset in (10, 20, 30):
            t.record_start(t._t0 + offset)
        table = t.timing_summary()
        assert "rollouts started so far" in table and "3/100" in table
        assert "(window open)" in table
        assert "WARNING" not in table

    def test_partial_start_warns_once_the_window_has_passed(self):
        t = DispatchLatencyTracker(total=100, start_report_within_s=1800)
        t._t0 -= 3600  # an hour into the run
        for offset in (10, 20, 30):
            t.record_start(t._t0 + offset)
        table = t.timing_summary()
        assert "3/100" in table and "(window open)" not in table
        assert table.count("WARNING") == 2
