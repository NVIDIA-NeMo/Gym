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
"""Paired-difference significance test via the Central Limit Theorem.

Pure functions, no I/O. Per the RFC (Evan Miller, "Adding Error Bars to Evals", arXiv:2411.00640,
"Resampling" + "Paired analysis" sections): each task's score is already collapsed to one
question-level mean before it reaches here (see `pairing.py`), so `deltas` is one independent
observation per task. The CLT licenses a t-approximation on their average even though the
individual per-task deltas are not themselves normal.

Two framings, chosen by whether `margin` is given:

- Two-sided (`margin=None`): H0 `mu_d = 0` vs H1 `mu_d != 0` -- "did anything change at all?"
- One-sided non-inferiority (`margin=delta`): H0 `mu_d <= -delta` vs H1 `mu_d > -delta` -- "is the
  candidate not meaningfully worse than the tolerance band?" `significant=True` here means H0 was
  rejected, i.e. the candidate is *not* meaningfully worse -- the opposite reading from the
  two-sided case, where `significant=True` means a real difference was found.
"""

from typing import List, Optional, Tuple

from scipy import stats

from nemo_gym.statistical_tests.schema import PairedTestResult


DEFAULT_POWER = 0.8


def paired_difference_stats(deltas: List[float]) -> Tuple[int, Optional[float], Optional[float]]:
    """`(n, mean_diff, se)`. `se` is `None` below `n=2` -- a standard error needs at least 2 points."""
    n = len(deltas)
    if n == 0:
        return n, None, None
    mean_diff = sum(deltas) / n
    if n < 2:
        return n, mean_diff, None
    variance = sum((delta - mean_diff) ** 2 for delta in deltas) / (n - 1)
    se = variance**0.5 / n**0.5
    return n, mean_diff, se


def minimum_detectable_effect(
    se: Optional[float],
    n: int,
    alpha: float,
    *,
    margin: Optional[float],
    power: float = DEFAULT_POWER,
) -> Optional[float]:
    """Smallest true effect this test could detect at `power`, given the data already collected."""
    if se is None or se == 0 or n < 2:
        return None
    df = n - 1
    tail_alpha = alpha if margin is not None else alpha / 2
    t_alpha = stats.t.ppf(1 - tail_alpha, df)
    t_power = stats.t.ppf(power, df)
    return (t_alpha + t_power) * se


def paired_test(
    deltas: List[float],
    *,
    metric: str,
    alpha: float = 0.05,
    margin: Optional[float] = None,
) -> PairedTestResult:
    """Run the paired CLT/t test described above and package the result."""
    n, mean_diff, se = paired_difference_stats(deltas)

    if n == 0:
        return PairedTestResult(metric=metric, margin=margin, alpha=alpha, n_pairs=0, note="no paired tasks.")
    if n < 2:
        return PairedTestResult(
            metric=metric,
            margin=margin,
            alpha=alpha,
            n_pairs=n,
            mean_diff=mean_diff,
            note="only 1 paired task: cannot estimate a standard error.",
        )

    df = n - 1
    if se < 1e-12:
        # Every paired delta is (numerically) identical: the t-statistic is undefined (0/0), or
        # so close to it that scipy's tail probability is meaningless noise. Read the verdict
        # directly from the point estimate instead of dividing by ~zero.
        threshold = 0.0 if margin is None else -margin
        significant = mean_diff != threshold if margin is None else mean_diff > threshold
        return PairedTestResult(
            metric=metric,
            margin=margin,
            alpha=alpha,
            n_pairs=n,
            mean_diff=mean_diff,
            se=0.0,
            p_value=0.0 if significant else 1.0,
            significant=significant,
            note="every paired delta was identical (zero variance); the t-test is degenerate, so the "
            "verdict is read directly from the point estimate rather than a p-value.",
        )

    if margin is None:
        t_stat = mean_diff / se
        p_value = 2 * stats.t.sf(abs(t_stat), df)
    else:
        t_stat = (mean_diff + margin) / se
        p_value = stats.t.sf(t_stat, df)

    significant = p_value < alpha
    mde = minimum_detectable_effect(se, n, alpha, margin=margin)

    return PairedTestResult(
        metric=metric,
        margin=margin,
        alpha=alpha,
        n_pairs=n,
        mean_diff=mean_diff,
        se=se,
        p_value=float(p_value),
        significant=significant,
        minimum_detectable_effect=mde,
    )
