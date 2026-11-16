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
"""Resolve catalog reward definitions for evaluation comparisons."""

from nemo_gym.config.config_types import ConfigError
from nemo_gym.environment.manifest import load_manifest
from nemo_gym.environment.registry import resolve_catalog_entry
from nemo_gym.eval.comparison.suite.schema import SuiteMetricResult


def resolve_benchmark_reward_metric(benchmark: str) -> SuiteMetricResult:
    """Read theoretical reward bounds and their provenance from a benchmark manifest."""
    entry = resolve_catalog_entry(benchmark, kind="benchmark")
    if entry.manifest_path is None:
        raise ConfigError(f"Benchmark '{benchmark}' has no reward manifest.")
    manifest = load_manifest(entry.manifest_path)
    low, high = manifest.reward.range
    return SuiteMetricResult(
        name="mean/reward",
        low=low,
        high=high,
        higher_is_better=manifest.reward.higher_is_better,
        definition_source=f"{entry.manifest_path.resolve()}:reward",
    )
