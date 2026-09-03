# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import math

import pytest
from scipy import stats

from nemo_gym.statistical_tests.paired_test import minimum_detectable_effect, paired_difference_stats, paired_test


class TestPairedDifferenceStats:
    def test_empty_deltas(self):
        assert paired_difference_stats([]) == (0, None, None)

    def test_single_delta_has_no_se(self):
        n, mean_diff, se = paired_difference_stats([0.4])
        assert n == 1
        assert mean_diff == pytest.approx(0.4)
        assert se is None

    def test_hand_checkable_example(self):
        # deltas = [1, 2, 3]: mean=2, sample variance (ddof=1) = 1, se = 1/sqrt(3)
        n, mean_diff, se = paired_difference_stats([1.0, 2.0, 3.0])
        assert n == 3
        assert mean_diff == pytest.approx(2.0)
        assert se == pytest.approx(1.0 / math.sqrt(3))


class TestPairedTestTwoSided:
    def test_no_paired_tasks(self):
        result = paired_test([], metric="reward")
        assert result.n_pairs == 0
        assert result.p_value is None
        assert result.significant is None
        assert result.note == "no paired tasks."

    def test_single_paired_task_cannot_estimate_se(self):
        result = paired_test([0.3], metric="reward")
        assert result.n_pairs == 1
        assert result.mean_diff == pytest.approx(0.3)
        assert result.se is None
        assert result.p_value is None
        assert "cannot estimate a standard error" in result.note

    def test_zero_variance_nonzero_mean_is_significant(self):
        # Every task moved by exactly the same amount -- the t-stat is undefined (0/0), but the
        # point estimate unambiguously says "these differ".
        result = paired_test([0.5, 0.5, 0.5], metric="reward")
        assert result.se == 0.0
        assert result.p_value == 0.0
        assert result.significant is True
        assert "zero variance" in result.note

    def test_zero_variance_zero_mean_is_not_significant(self):
        result = paired_test([0.0, 0.0, 0.0], metric="reward")
        assert result.se == 0.0
        assert result.p_value == 1.0
        assert result.significant is False

    def test_identical_runs_give_a_high_p_value(self):
        # No systematic difference: symmetric deltas around 0 should not reject H0.
        result = paired_test([0.1, -0.1, 0.05, -0.05, 0.0], metric="reward", alpha=0.05)
        assert result.significant is False
        assert result.p_value > 0.05

    def test_large_consistent_regression_is_significant(self):
        result = paired_test([-0.5, -0.6, -0.4, -0.55, -0.45, -0.5], metric="reward", alpha=0.05)
        assert result.mean_diff < 0
        assert result.significant is True
        assert result.p_value < 0.05

    def test_p_value_matches_scipy_ttest_1samp_directly(self):
        deltas = [0.2, -0.1, 0.3, 0.05, -0.05, 0.15]
        result = paired_test(deltas, metric="reward")
        t_stat, expected_p = stats.ttest_1samp(deltas, popmean=0.0)
        assert result.p_value == pytest.approx(expected_p)

    def test_mde_present_when_computable(self):
        result = paired_test([0.1, -0.1, 0.3, 0.05, -0.05, 0.15], metric="reward")
        assert result.minimum_detectable_effect is not None
        assert result.minimum_detectable_effect > 0


class TestPairedTestNonInferiority:
    def test_regression_within_margin_is_significant(self):
        # Candidate is worse by ~0.05 on average, comfortably inside a 0.2 margin -> not
        # meaningfully worse -> H0 rejected -> significant=True.
        deltas = [-0.05, -0.06, -0.04, -0.05, -0.05, -0.06]
        result = paired_test(deltas, metric="reward", margin=0.2)
        assert result.significant is True
        assert result.p_value < 0.05

    def test_regression_beyond_margin_is_not_significant(self):
        # Candidate is worse by ~0.5 on average, well past a 0.1 margin -> cannot conclude
        # non-inferiority -> significant=False.
        deltas = [-0.5, -0.55, -0.45, -0.5, -0.52, -0.48]
        result = paired_test(deltas, metric="reward", margin=0.1)
        assert result.significant is False

    def test_p_value_matches_shifted_scipy_ttest(self):
        deltas = [-0.05, -0.06, -0.04, -0.05, -0.05, -0.06]
        margin = 0.2
        result = paired_test(deltas, metric="reward", margin=margin)
        shifted = [d + margin for d in deltas]
        t_stat, _ = stats.ttest_1samp(shifted, popmean=0.0)
        n = len(deltas)
        expected_p = stats.t.sf(t_stat, n - 1)
        assert result.p_value == pytest.approx(expected_p)

    def test_zero_variance_exactly_at_margin_boundary(self):
        # -0.25 is exactly representable in binary floating point, so the variance really is 0.0,
        # not just numerically tiny -- this exercises the zero-variance branch deterministically.
        result = paired_test([-0.25, -0.25, -0.25], metric="reward", margin=0.25)
        assert result.se == 0.0
        # mean_diff == -margin is the H0 boundary itself, not "> -margin" -- not significant.
        assert result.significant is False


class TestMinimumDetectableEffect:
    def test_none_below_two_points(self):
        assert minimum_detectable_effect(0.1, 1, 0.05, margin=None) is None
        assert minimum_detectable_effect(None, 5, 0.05, margin=None) is None

    def test_none_when_se_is_zero(self):
        assert minimum_detectable_effect(0.0, 5, 0.05, margin=None) is None

    def test_larger_alpha_or_more_data_shrinks_mde(self):
        loose = minimum_detectable_effect(0.1, 10, 0.10, margin=None)
        tight = minimum_detectable_effect(0.1, 10, 0.01, margin=None)
        assert loose < tight

        few = minimum_detectable_effect(0.1, 5, 0.05, margin=None)
        many = minimum_detectable_effect(0.1, 50, 0.05, margin=None)
        assert many < few
