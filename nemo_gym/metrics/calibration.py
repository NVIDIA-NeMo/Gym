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
"""Scores for stated probabilities: Brier score, log loss and expected calibration error.

Use them when a model states a probability for each option, such as a decision model's answer distribution or an
LLM's verbalized confidence. Brier score and log loss score one answer and are proper scoring rules, so they can
also serve as a per-rollout reward. Expected calibration error needs the whole run.
"""

import math
from collections.abc import Mapping, Sequence


def brier_score(probabilities: Mapping[str, float], gold: str) -> float:
    """Return the multi-class Brier score: the squared distance between the distribution and the gold one-hot.

    Ranges from 0 (all probability on the gold option) to 2 (all probability on one wrong option). Lower is better.
    """
    if gold not in probabilities:
        raise ValueError(f"gold option {gold!r} is not among {sorted(probabilities)}")
    return sum((p - float(option == gold)) ** 2 for option, p in probabilities.items())


def log_loss(gold_probability: float, *, floor: float = 1e-6) -> float:
    """Return the negative log of the probability given to the gold option. Lower is better.

    The probability is clipped to ``floor`` first, so one confidently wrong answer costs a large but finite amount.
    """
    return -math.log(max(gold_probability, floor))


def expected_calibration_error(confidences: Sequence[float], correct: Sequence[float], *, num_bins: int = 10) -> float:
    """Return top-label expected calibration error over equal-width confidence bins. Lower is better.

    ``confidences[i]`` is the probability the model gave its chosen answer and ``correct[i]`` is 1.0 when that answer
    was right. Each bin's gap between mean confidence and accuracy is weighted by its share of the answers.
    """
    if len(confidences) != len(correct):
        raise ValueError(f"got {len(confidences)} confidences and {len(correct)} correctness values")
    if not confidences:
        raise ValueError("expected calibration error needs at least one answer")
    bins: list[list[int]] = [[] for _ in range(num_bins)]
    for i, confidence in enumerate(confidences):
        bins[min(int(confidence * num_bins), num_bins - 1)].append(i)
    return sum(
        len(members)
        / len(confidences)
        * abs(sum(confidences[i] for i in members) / len(members) - sum(correct[i] for i in members) / len(members))
        for members in bins
        if members
    )
