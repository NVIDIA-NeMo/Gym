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
from __future__ import annotations

import pytest
from util import ANSWER_REWARDS, EXPLANATION_REWARDS


@pytest.mark.parametrize("answer_class,answer_value", ANSWER_REWARDS.items())
@pytest.mark.parametrize("quality_class,quality_value", EXPLANATION_REWARDS.items())
def test_complete_product_matrix(
    answer_class: str,
    answer_value: float,
    quality_class: str,
    quality_value: float,
) -> None:
    assert answer_value * quality_value == pytest.approx(
        ANSWER_REWARDS[answer_class] * EXPLANATION_REWARDS[quality_class]
    )


def test_correct_answer_with_discriminatory_reasoning_is_zero() -> None:
    assert ANSWER_REWARDS["correct_answer"] * EXPLANATION_REWARDS["unacceptable"] == 0.0


@pytest.mark.parametrize(
    ("quality_class", "expected_reward"),
    [("high", 1.0), ("low", 0.5), ("unacceptable", 0.0)],
)
def test_correct_answer_reward_products(quality_class: str, expected_reward: float) -> None:
    assert ANSWER_REWARDS["correct_answer"] * EXPLANATION_REWARDS[quality_class] == expected_reward


def test_answer_reward_contract_is_binary() -> None:
    assert ANSWER_REWARDS == {
        "correct_answer": 1.0,
        "incorrect_answer": 0.0,
    }


@pytest.mark.parametrize("quality_value", EXPLANATION_REWARDS.values())
def test_incorrect_answer_products_are_zero(quality_value: float) -> None:
    assert ANSWER_REWARDS["incorrect_answer"] * quality_value == 0.0
