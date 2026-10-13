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
"""Metric name components and primary-metric filtering."""

# Revisit and simplify this metric name grammar when resolving
# https://github.com/NVIDIA-NeMo/Gym/issues/3471.

from enum import StrEnum
from typing import Tuple

from nemo_gym.global_config import ATTEMPT_INDEX_KEY_NAME, ROLLOUT_INDEX_KEY_NAME, TASK_INDEX_KEY_NAME


STAT_SEPARATOR = "/"
ACROSS_REPEATS_MARKER = f"_across_repeats{STAT_SEPARATOR}"
COMPLETION_TOKEN_METRIC_NAMES: Tuple[str, ...] = ("mean_completion_tokens", "mean_tokens_per_turn")


class Stat(StrEnum):
    MEAN = "mean"
    MAX = "max"
    MIN = "min"
    MEDIAN = "median"
    STD = "std"
    SEM = "sem"
    SE = "se"
    P25 = "p25"
    P75 = "p75"
    CI_LOW_95 = "ci_low_95"
    CI_HIGH_95 = "ci_high_95"
    HISTOGRAM = "histogram"

    @property
    def prefix(self) -> str:
        return f"{self.value}{STAT_SEPARATOR}"

    @property
    def across_repeats_prefix(self) -> str:
        return f"{self.value}{ACROSS_REPEATS_MARKER}"


# Companion statistics `compute_pass_majority_metrics` appends to a pass@k metric name.
class PassMajorityStat(StrEnum):
    STD_DEV_ACROSS_RUNS = "std_dev_across_runs"
    STD_ERR_ACROSS_RUNS = "std_err_across_runs"
    AVG_SAMPLE_STD_DEV = "avg_sample_std_dev"

    @property
    def suffix(self) -> str:
        return f"{STAT_SEPARATOR}{self.value}"


PASS_MAJORITY_STAT_SUFFIXES: Tuple[str, ...] = tuple(stat.suffix for stat in PassMajorityStat)


# Existing statistics, uncertainty estimates, and metadata are not primary metrics.
METRIC_EXCLUDED_PREFIXES = tuple(stat.prefix for stat in Stat if stat != Stat.MEAN)
METRIC_EXCLUDED_SUFFIXES = (
    f"/{Stat.MAX}",
    f"/{Stat.MIN}",
    f"/{Stat.MEDIAN}",
    "/p5",
    f"/{Stat.P25}",
    f"/{Stat.P75}",
    "/p95",
    f"/{Stat.CI_LOW_95}",
    f"/{Stat.CI_HIGH_95}",
    "/ci_lower",
    "/ci_upper",
    "_ci_lower",
    "_ci_upper",
    "_ci95_lower",
    "_ci95_upper",
    f"/{ATTEMPT_INDEX_KEY_NAME}",
) + PASS_MAJORITY_STAT_SUFFIXES
METRIC_EXCLUDED_NAMES = (
    TASK_INDEX_KEY_NAME,
    ROLLOUT_INDEX_KEY_NAME,
    ATTEMPT_INDEX_KEY_NAME,
    "sample_count",
    "missing_count",
    "num_repeats",
    "token_usage_version",
)


def is_primary_metric(name: object) -> bool:
    """Whether a metric name is a point estimate suitable for repeat aggregation and comparison."""
    return (
        isinstance(name, str)
        and name not in METRIC_EXCLUDED_NAMES
        and not name.startswith(METRIC_EXCLUDED_PREFIXES)
        and not name.endswith(METRIC_EXCLUDED_SUFFIXES)
        and ACROSS_REPEATS_MARKER not in name
    )
