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
"""Load direct planning inputs and assemble completed numerical report records."""

import shlex
import sys
from datetime import datetime, timezone
from math import sqrt
from pathlib import Path

import yaml
from pydantic import ValidationError

from nemo_gym.config.config_types import ConfigError
from nemo_gym.config.secret_utils import hide_secrets_in_overrides
from nemo_gym.eval.comparison.planning.schema import (
    RepeatPlanningCellResult,
    RepeatPlanningCLIConfig,
    RepeatPlanningManifest,
    RepeatPlanningMetricResult,
    RepeatPlanningResult,
    RepeatPlanningTotals,
)
from nemo_gym.eval.comparison.statistics.repeat_planning import (
    PlanningCell,
    PlanningParameters,
    plan_repeats,
    simulate_plan,
)
from nemo_gym.package_info import __version__


def invoked_command() -> str:
    """Record the planning command and mask secret Hydra overrides."""
    return shlex.join(["gym", "eval", "plan-repeats", *hide_secrets_in_overrides(sys.argv[1:])])


def load_repeat_planning_manifest(path: Path) -> RepeatPlanningManifest:
    """Read a strict, versioned planning YAML without loading evaluation artifacts."""
    try:
        return RepeatPlanningManifest.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
    except (OSError, UnicodeError, yaml.YAMLError, ValidationError) as error:
        raise ConfigError(f"Cannot load repeat planning manifest '{path}': {error}") from error


def build_repeat_planning_result(config: RepeatPlanningCLIConfig, command: str) -> RepeatPlanningResult:
    """Plan one aggregate and validate its fixed counts against simulated Welch decisions."""
    path = Path(config.repeat_plan_fpath).resolve()
    manifest = load_repeat_planning_manifest(path)
    cells = []
    resolved_metrics = []
    try:
        parameters = PlanningParameters(
            baseline_mode=manifest.baseline.mode,
            candidate_experiments=manifest.candidate_experiments,
            margin_points=manifest.margin_points,
            target_power=manifest.target_power,
            alpha=manifest.alpha,
            expected_delta_points=manifest.expected_delta_points,
            max_repeats=manifest.max_repeats,
        )
        for cell in manifest.cells:
            definitions = cell.metric_definitions()
            metrics = [
                RepeatPlanningMetricResult(
                    name=metric.name,
                    weight=1 / len(definitions),
                    baseline_std_points=metric.baseline_std_points,
                    candidate_std_points=(
                        metric.baseline_std_points
                        if metric.candidate_std_points is None
                        else metric.candidate_std_points
                    ),
                    candidate_std_defaulted=metric.candidate_std_points is None,
                )
                for metric in definitions
            ]
            resolved_metrics.append(metrics)
            # A multi-metric cell supplies the SD of its per-repeat average.
            # Marginal metric SDs alone do not determine that composite SD.
            baseline_std = cell.baseline_std_points
            if baseline_std is None:
                baseline_std = metrics[0].baseline_std_points
                candidate_std = metrics[0].candidate_std_points
            else:
                candidate_std = baseline_std if cell.candidate_std_points is None else cell.candidate_std_points
            cells.append(
                PlanningCell(
                    model=cell.model,
                    benchmark=cell.benchmark,
                    baseline_variance=baseline_std**2,
                    candidate_variance=candidate_std**2,
                    repeat_step=cell.repeat_step,
                    repeat_cost=cell.repeat_cost,
                    existing_baseline_repeats=cell.existing_baseline_repeats,
                )
            )
        allocation = plan_repeats(cells, parameters=parameters)
        simulation = (
            simulate_plan(
                cells,
                allocation,
                parameters=parameters,
                trials=manifest.validation_trials,
                seed=manifest.seed,
            )
            if manifest.validation_trials
            else None
        )
    except (ValueError, OverflowError) as error:
        raise ConfigError(f"Cannot plan repeats for '{path}': {error}") from error

    results = []
    for index, (cell, metrics) in enumerate(zip(cells, resolved_metrics, strict=True)):
        baseline = allocation.baseline_counts[index]
        candidate = allocation.candidate_counts[index]
        new_baseline = allocation.additional_baseline_counts[index]
        results.append(
            RepeatPlanningCellResult(
                model=cell.model,
                benchmark=cell.benchmark,
                weight=1 / len(cells),
                metrics=metrics,
                baseline_std_points=sqrt(cell.baseline_variance),
                candidate_std_points=sqrt(cell.candidate_variance),
                repeat_step=cell.repeat_step,
                repeat_cost=cell.repeat_cost,
                existing_baseline_repeats=cell.existing_baseline_repeats or 0,
                baseline_repeats=baseline,
                new_baseline_repeats=new_baseline,
                candidate_repeats_per_experiment=candidate,
                candidate_repeats_all_experiments=manifest.candidate_experiments * candidate,
                new_repeat_cost=cell.repeat_cost * (new_baseline + manifest.candidate_experiments * candidate),
            )
        )
    totals = RepeatPlanningTotals(
        existing_baseline_passes=sum(cell.existing_baseline_repeats for cell in results),
        new_baseline_passes=sum(cell.new_baseline_repeats for cell in results),
        candidate_passes_per_experiment=sum(cell.candidate_repeats_per_experiment for cell in results),
        candidate_passes_all_experiments=sum(cell.candidate_repeats_all_experiments for cell in results),
        total_additional_passes=allocation.total_additional_passes,
        total_cost=allocation.total_cost,
    )
    return RepeatPlanningResult(
        generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        nemo_gym_version=__version__,
        command=command,
        plan_fpath=path,
        settings=manifest,
        cells=results,
        allocation=allocation,
        simulation=simulation,
        totals=totals,
    )
