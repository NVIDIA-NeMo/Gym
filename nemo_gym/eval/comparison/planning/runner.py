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
from pathlib import Path

import yaml
from pydantic import ValidationError

from nemo_gym.config.config_types import ConfigError
from nemo_gym.config.secret_utils import hide_secrets_in_overrides
from nemo_gym.eval.comparison.planning.schema import (
    RepeatPlanningCellResult,
    RepeatPlanningCLIConfig,
    RepeatPlanningManifest,
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
    plan_path = Path(config.repeat_plan_fpath).resolve()
    manifest = load_repeat_planning_manifest(plan_path)
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
        planning_cells = [
            PlanningCell(
                model=cell.model,
                benchmark=cell.benchmark,
                baseline_variance=cell.baseline_std_points**2,
                candidate_variance=(
                    cell.baseline_std_points if cell.candidate_std_points is None else cell.candidate_std_points
                )
                ** 2,
                repeat_step=cell.repeat_step,
                repeat_cost=cell.repeat_cost,
                existing_baseline_repeats=cell.existing_baseline_repeats,
            )
            for cell in manifest.cells
        ]
        allocation = plan_repeats(planning_cells, parameters=parameters)
        simulation = (
            simulate_plan(
                planning_cells,
                allocation,
                parameters=parameters,
                trials=manifest.validation_trials,
                seed=manifest.seed,
            )
            if manifest.validation_trials
            else None
        )
    except (ValueError, OverflowError) as error:
        raise ConfigError(f"Cannot plan repeats for '{plan_path}': {error}") from error

    cell_results = []
    for index, cell in enumerate(manifest.cells):
        baseline_count = allocation.baseline_counts[index]
        candidate_count = allocation.candidate_counts[index]
        new_baseline_count = allocation.additional_baseline_counts[index]
        cell_results.append(
            RepeatPlanningCellResult(
                model=cell.model,
                benchmark=cell.benchmark,
                weight=1 / len(planning_cells),
                baseline_std_points=cell.baseline_std_points,
                candidate_std_points=(
                    cell.baseline_std_points if cell.candidate_std_points is None else cell.candidate_std_points
                ),
                repeat_step=cell.repeat_step,
                repeat_cost=cell.repeat_cost,
                existing_baseline_repeats=cell.existing_baseline_repeats or 0,
                baseline_repeats=baseline_count,
                new_baseline_repeats=new_baseline_count,
                candidate_repeats_per_experiment=candidate_count,
                candidate_repeats_all_experiments=manifest.candidate_experiments * candidate_count,
                new_repeat_cost=cell.repeat_cost
                * (new_baseline_count + manifest.candidate_experiments * candidate_count),
            )
        )
    totals = RepeatPlanningTotals(
        existing_baseline_passes=sum(cell.existing_baseline_repeats for cell in cell_results),
        new_baseline_passes=sum(cell.new_baseline_repeats for cell in cell_results),
        candidate_passes_per_experiment=sum(cell.candidate_repeats_per_experiment for cell in cell_results),
        candidate_passes_all_experiments=sum(cell.candidate_repeats_all_experiments for cell in cell_results),
        total_additional_passes=allocation.total_additional_passes,
        total_cost=allocation.total_cost,
    )
    return RepeatPlanningResult(
        generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        nemo_gym_version=__version__,
        command=command,
        plan_fpath=plan_path,
        settings=manifest,
        cells=cell_results,
        allocation=allocation,
        simulation=simulation,
        totals=totals,
    )
