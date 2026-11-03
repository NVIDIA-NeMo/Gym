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
"""Rollout metrics: reward profiles, pass@k and majority statistics, and the metric definitions they share."""

# Public API
from typing import TYPE_CHECKING

from nemo_gym._public_api import lazy_exports


if TYPE_CHECKING:
    from nemo_gym.metrics.metrics_config import Stat
    from nemo_gym.metrics.reward_profile import (
        AggregateMetricsMixin,
        RewardProfiler,
        add_avg_sample_std_dev,
        compute_aggregate_metrics,
        compute_pass_majority_metrics,
        compute_subset_metrics,
        highest_k_metrics,
    )


# What code outside the library, such as servers, imports from this package. Each name's submodule loads
# on first use. Code inside nemo_gym imports the submodules directly.
_PUBLIC_API = {
    "AggregateMetricsMixin": "reward_profile",
    "RewardProfiler": "reward_profile",
    "Stat": "metrics_config",
    "add_avg_sample_std_dev": "reward_profile",
    "compute_aggregate_metrics": "reward_profile",
    "compute_pass_majority_metrics": "reward_profile",
    "compute_subset_metrics": "reward_profile",
    "highest_k_metrics": "reward_profile",
}
__getattr__, __dir__ = lazy_exports(__name__, _PUBLIC_API)
__all__ = sorted(_PUBLIC_API)
