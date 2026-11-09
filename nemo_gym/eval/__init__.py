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
"""Evaluation on top of collected rollouts: eval plans, reverifying saved rollouts, and comparing runs."""

from typing import TYPE_CHECKING

from nemo_gym._public_api import lazy_exports


if TYPE_CHECKING:
    from nemo_gym.eval.comparison.planning.schema import RepeatPlanningManifest, RepeatPlanningResult
    from nemo_gym.eval.comparison.statistics.repeat_planning import (
        PlanningCell,
        PlanningParameters,
        PlanningStatistics,
        RepeatAllocation,
        SimulationResult,
        plan_repeats,
        power_for_counts,
        simulate_plan,
    )
    from nemo_gym.eval.comparison.statistics.unpaired_welch_fixed_suite import (
        BenchmarkSamples,
        SampleSummary,
        aggregate_welch,
    )


_PUBLIC_API = {
    "BenchmarkSamples": "comparison.statistics.unpaired_welch_fixed_suite",
    "PlanningCell": "comparison.statistics.repeat_planning",
    "PlanningParameters": "comparison.statistics.repeat_planning",
    "PlanningStatistics": "comparison.statistics.repeat_planning",
    "RepeatAllocation": "comparison.statistics.repeat_planning",
    "RepeatPlanningManifest": "comparison.planning.schema",
    "RepeatPlanningResult": "comparison.planning.schema",
    "SampleSummary": "comparison.statistics.unpaired_welch_fixed_suite",
    "SimulationResult": "comparison.statistics.repeat_planning",
    "aggregate_welch": "comparison.statistics.unpaired_welch_fixed_suite",
    "plan_repeats": "comparison.statistics.repeat_planning",
    "power_for_counts": "comparison.statistics.repeat_planning",
    "simulate_plan": "comparison.statistics.repeat_planning",
}
__getattr__, __dir__ = lazy_exports(__name__, _PUBLIC_API)
__all__ = sorted(_PUBLIC_API)
