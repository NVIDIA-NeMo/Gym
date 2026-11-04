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

from nemo_gym.metrics.calibration import brier_score, expected_calibration_error, log_loss


class TestBrierScore:
    def test_sums_squared_error_against_one_hot(self) -> None:
        assert brier_score({"a": 0.7, "b": 0.2, "c": 0.1}, "a") == pytest.approx(0.3**2 + 0.2**2 + 0.1**2)

    def test_certain_and_right_is_zero(self) -> None:
        assert brier_score({"a": 1.0, "b": 0.0}, "a") == 0.0

    def test_certain_and_wrong_is_two(self) -> None:
        assert brier_score({"a": 0.0, "b": 1.0}, "a") == pytest.approx(2.0)

    def test_gold_must_be_an_option(self) -> None:
        with pytest.raises(ValueError, match="not among"):
            brier_score({"a": 1.0}, "z")


class TestLogLoss:
    def test_is_negative_log_of_gold_probability(self) -> None:
        assert log_loss(0.25) == pytest.approx(math.log(4))

    def test_zero_probability_is_clipped_to_the_floor(self) -> None:
        assert log_loss(0.0) == pytest.approx(-math.log(1e-6))
        assert log_loss(0.0, floor=1e-3) == pytest.approx(-math.log(1e-3))


class TestExpectedCalibrationError:
    def test_matches_hand_computation(self) -> None:
        # Bin [0.9, 1.0): confidence 0.9 twice, one right, gap 0.4. Bin [0.6, 0.7): confidence 0.6, right, gap 0.4.
        assert expected_calibration_error([0.9, 0.9, 0.6], [1.0, 0.0, 1.0]) == pytest.approx(0.4)

    def test_calibrated_answers_score_zero(self) -> None:
        assert expected_calibration_error([0.5] * 4, [1.0, 0.0, 1.0, 0.0]) == pytest.approx(0.0)

    def test_confidence_of_one_falls_in_the_top_bin(self) -> None:
        assert expected_calibration_error([1.0, 1.0], [1.0, 1.0]) == pytest.approx(0.0)

    def test_bin_count_changes_grouping(self) -> None:
        # One bin pools everything: mean confidence 0.7 against accuracy 0.5.
        assert expected_calibration_error([0.9, 0.5], [0.0, 1.0], num_bins=1) == pytest.approx(0.2)

    @pytest.mark.parametrize(("confidences", "correct"), [([0.5], [1.0, 0.0]), ([], [])])
    def test_rejects_mismatched_or_empty_input(self, confidences: list[float], correct: list[float]) -> None:
        with pytest.raises(ValueError):
            expected_calibration_error(confidences, correct)
