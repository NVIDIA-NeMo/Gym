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
"""Orchestrate suite artifact loading, processing, and result provenance."""

import shlex
import sys
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

from nemo_gym.config.config_types import ConfigError
from nemo_gym.config.secret_utils import hide_secrets_in_overrides
from nemo_gym.eval.comparison.suite.failures import read_saved_failures, saved_failure_warning
from nemo_gym.eval.comparison.suite.loading import load_suite_manifest, load_suite_run, resolve_metric_definitions
from nemo_gym.eval.comparison.suite.processing import aggregate_suite_cells, compare_suite_cell
from nemo_gym.eval.comparison.suite.schema import (
    SuiteComparisonConfig,
    SuiteComparisonResult,
    SuiteFailureSummary,
    SuiteMetricResult,
    summary_sources,
)
from nemo_gym.package_info import __version__


def invoked_command() -> str:
    """Record the suite command with its Hydra overrides, masking secret values."""
    return shlex.join(["gym", "eval", "compare-suite", *hide_secrets_in_overrides(sys.argv[1:])])


def build_suite_comparison(
    config: SuiteComparisonConfig,
    command: str,
    *,
    reward_metric_resolver: Callable[[str], SuiteMetricResult] | None = None,
) -> SuiteComparisonResult:
    """Build typed results, optionally resolving catalog reward scales through the supplied callback."""
    path = Path(config.suite_fpath).resolve()
    manifest = load_suite_manifest(path)
    definitions = {
        benchmark: resolve_metric_definitions(benchmark, definition, reward_metric_resolver=reward_metric_resolver)
        for benchmark, definition in manifest.benchmarks.items()
    }
    cells = []
    warnings = []
    per_model = []
    observed_sources: dict[tuple[str, Path, str], str] = {}
    for model, benchmarks in manifest.models.items():
        model_cells = []
        for benchmark, sources in benchmarks.items():
            baseline = tuple(load_suite_run(source, directory=path.parent) for source in sources.baseline)
            candidate = tuple(load_suite_run(source, directory=path.parent) for source in sources.candidate)
            for role, runs in (("baseline", baseline), ("candidate", candidate)):
                for index, run in enumerate(runs, 1):
                    label = f"{model}/{benchmark} {role}"
                    if len(runs) > 1:
                        label += f" source {index}"
                    for source_kind, source_path in (("aggregate", run.aggregate_metrics), ("rollouts", run.rollouts)):
                        identity = source_kind, source_path.resolve(), run.agent
                        if identity in observed_sources:
                            raise ConfigError(
                                f"{label} reuses the same {source_kind} file and agent as {observed_sources[identity]}. "
                                "Suite Welch comparisons require independent source observations."
                            )
                        observed_sources[identity] = label
            cell = compare_suite_cell(
                model=model,
                benchmark=benchmark,
                baseline=baseline,
                candidate=candidate,
                metrics=[metric.model_copy(deep=True) for metric in definitions[benchmark]],
                confidence_level=config.confidence_level,
            )
            model_cells.append(cell)
            if cell.exclusion_reason:
                warnings.append(f"{model}/{benchmark}: {cell.exclusion_reason}")
            for role, run in (("baseline", cell.baseline), ("candidate", cell.candidate)):
                warnings.extend(f"{model}/{benchmark} {role}: {note}" for note in run.coverage.notes)
        eligible = [cell for cell in model_cells if cell.eligible]
        for cell in eligible:
            cell.model_weight = 1 / len(eligible)
        per_model.append(aggregate_suite_cells(model, model_cells, confidence_level=config.confidence_level))
        cells.extend(model_cells)
    eligible = [cell for cell in cells if cell.eligible]
    if not eligible:
        raise ConfigError("No eligible model-benchmark comparisons remain. " + " ".join(warnings))
    for cell in eligible:
        cell.overall_weight = 1 / len(eligible)
    result = SuiteComparisonResult(
        generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        nemo_gym_version=__version__,
        command=command,
        suite_fpath=path,
        confidence_level=config.confidence_level,
        cells=cells,
        per_model=per_model,
        overall=aggregate_suite_cells("overall", cells, confidence_level=config.confidence_level),
        warnings=warnings,
    )
    failures_by_path: dict[Path, SuiteFailureSummary | None] = {}
    uses_by_path: dict[Path, list[str]] = {}
    for cell in result.cells:
        for role, collection in (("baseline", cell.baseline), ("candidate", cell.candidate)):
            for index, run in enumerate(summary_sources(collection), 1):
                if run.rollouts not in failures_by_path:
                    failures_by_path[run.rollouts] = read_saved_failures(run.rollouts)
                    uses_by_path[run.rollouts] = []
                run.failures = failures_by_path[run.rollouts]
                source_label = f"{role} source {index}" if collection.sources else role
                label = f"{cell.model}/{cell.benchmark} {source_label}"
                if label not in uses_by_path[run.rollouts]:
                    uses_by_path[run.rollouts].append(label)
    for path, summary in failures_by_path.items():
        if summary is not None:
            warning = saved_failure_warning(summary)
            if warning is not None:
                result.warnings.append(f"{', '.join(uses_by_path[path])}: {warning}")
    return result
