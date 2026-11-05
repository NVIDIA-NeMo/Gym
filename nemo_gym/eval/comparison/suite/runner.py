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

import hashlib
import shlex
import sys
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryFile

from nemo_gym.config.config_types import ConfigError
from nemo_gym.config.secret_utils import hide_secrets_in_overrides
from nemo_gym.eval.comparison.suite.loading import load_suite_manifest, load_suite_run, resolve_metric_definitions
from nemo_gym.eval.comparison.suite.processing import aggregate_suite_cells, compare_suite_cell
from nemo_gym.eval.comparison.suite.schema import (
    SuiteComparisonConfig,
    SuiteComparisonResult,
    SuiteHealthSummary,
    SuiteManifest,
    SuiteMetricResult,
)
from nemo_gym.package_info import __version__
from nemo_gym.utils.path_utils import (
    aggregate_metrics_path_for,
    attempt_extension_path_for,
    failures_path_for,
    materialized_path_for,
    rerun_masked_journal_path_for,
    resolved_config_path_for,
    resubmit_offset_path_for,
)


def invoked_command() -> str:
    """Record the suite command with its Hydra overrides, masking secret values."""
    return shlex.join(["gym", "eval", "compare-suite", *hide_secrets_in_overrides(sys.argv[1:])])


def _health_report_directories(
    manifest: SuiteManifest, *, manifest_path: Path, output_root: str | None
) -> dict[Path, Path]:
    """Validate report destinations and prepare a writable output root before scans."""
    if output_root is None:
        return {}
    from nemo_gym.artifacts.run_files import QUALITY_SUMMARY_FILENAME, ROLLOUT_VERDICTS_FILENAME
    from nemo_gym.rollouts.rollout_records import coverage_path_for, journal_path_for, resolve_rollout_path
    from nemo_gym.rollouts.rollout_recovery import manifest_path_for

    sources = {manifest_path}
    rollouts = set()
    for benchmarks in manifest.models.values():
        for pair in benchmarks.values():
            for run in (pair.baseline, pair.candidate):
                path = resolve_rollout_path(manifest_path.parent / run.rollouts, read_only=True)
                rollouts.add(path)
                sources.update(
                    (
                        path.resolve(),
                        (manifest_path.parent / run.aggregate_metrics).resolve()
                        if run.aggregate_metrics is not None
                        else aggregate_metrics_path_for(path).resolve(),
                        (manifest_path.parent / run.materialized_inputs).resolve()
                        if run.materialized_inputs is not None
                        else materialized_path_for(path).resolve(),
                        materialized_path_for(path).resolve(),
                        failures_path_for(path).resolve(),
                        manifest_path_for(path).resolve(),
                        journal_path_for(path).resolve(),
                        coverage_path_for(path).resolve(),
                        attempt_extension_path_for(path).resolve(),
                        rerun_masked_journal_path_for(path).resolve(),
                        resubmit_offset_path_for(path).resolve(),
                        resolved_config_path_for(path).resolve(),
                        (path.parent / QUALITY_SUMMARY_FILENAME).resolve(),
                        (path.parent / ROLLOUT_VERDICTS_FILENAME).resolve(),
                    )
                )
    source_directories = {path.parent for path in rollouts}
    root = Path(output_root).resolve()
    directories: dict[Path, Path] = {}
    planned_targets: set[Path] = set()
    for path in sorted(rollouts):
        name = "health-check-" + hashlib.sha256(str(path).encode()).hexdigest()[:12]
        directory = (root / name).resolve()
        targets = {
            (directory / filename).resolve() for filename in (QUALITY_SUMMARY_FILENAME, ROLLOUT_VERDICTS_FILENAME)
        }
        if directory in source_directories or targets & sources:
            raise ConfigError(
                f"Health reports in '{directory}' would overwrite source artifacts; "
                "choose another --health-check-output-dir."
            )
        if directory in directories.values() or len(targets) != 2 or targets & planned_targets:
            raise ConfigError(
                f"Health report paths in '{directory}' overlap other reports; "
                "choose another --health-check-output-dir."
            )
        directories[path] = directory
        planned_targets.update(targets)
    try:
        root.mkdir(parents=True, exist_ok=True)
        with TemporaryFile(dir=root):
            pass
    except OSError as error:
        raise ConfigError(f"Cannot write to --health-check-output-dir '{root}': {error}") from error
    return directories


def _check_suite_health(
    result: SuiteComparisonResult,
    manifest: SuiteManifest,
    *,
    workers: int | None,
    output_root: str | None,
) -> None:
    """Attach file health only after the complete numerical comparison has passed validation."""
    directories = _health_report_directories(manifest, manifest_path=result.suite_fpath, output_root=output_root)
    from nemo_gym.eval.comparison.suite.health import check_rollout_file_health, rollout_health_warning

    health_by_path: dict[Path, SuiteHealthSummary] = {}
    for cell in result.cells:
        for role, run in (("baseline", cell.baseline), ("candidate", cell.candidate)):
            if run.rollouts not in health_by_path:
                health_by_path[run.rollouts] = check_rollout_file_health(
                    run.rollouts, workers=workers, output_dir=directories.get(run.rollouts)
                )
            run.health = health_by_path[run.rollouts]
            warning = rollout_health_warning(run.health)
            if warning is not None:
                result.warnings.append(f"{cell.model}/{cell.benchmark} {role}: {warning}")


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
            baseline = load_suite_run(sources.baseline, directory=path.parent)
            candidate = load_suite_run(sources.candidate, directory=path.parent)
            for role, run in (("baseline", baseline), ("candidate", candidate)):
                label = f"{model}/{benchmark} {role}"
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
        health_check_enabled=config.check_health,
        cells=cells,
        per_model=per_model,
        overall=aggregate_suite_cells("overall", cells, confidence_level=config.confidence_level),
        warnings=warnings,
    )
    if config.check_health:
        _check_suite_health(
            result,
            manifest,
            workers=config.health_check_workers,
            output_root=config.health_check_output_dirpath,
        )
    return result
