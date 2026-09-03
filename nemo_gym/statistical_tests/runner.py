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
"""End-to-end execution of `gym eval test` -- and what `gym eval compare` calls internally.

`run_paired_test` is the one function both commands share, in-process, no CLI/subprocess involved.
"""

import shlex
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple

from nemo_gym import _resolve_under_cwd_or_install
from nemo_gym.comparison.loading import LoadedRun, build_loaded_run, load_agg_metrics_file, resolve_agent_selections
from nemo_gym.config_types import ConfigError
from nemo_gym.package_info import __version__
from nemo_gym.secret_utils import hide_secrets_in_overrides
from nemo_gym.statistical_tests.paired_test import paired_test
from nemo_gym.statistical_tests.pairing import paired_task_deltas, resolve_metrics
from nemo_gym.statistical_tests.schema import STATS_SUBDIR_NAME, PairedTestConfig, PairedTestReport, PairedTestResult


def invoked_command() -> str:
    """The `gym eval test` invocation, for provenance in the report."""
    return shlex.join(["gym", "eval", "test", *hide_secrets_in_overrides(sys.argv[1:])])


def run_paired_test(
    baseline: LoadedRun,
    candidate: LoadedRun,
    *,
    metric: str,
    margin: Optional[float],
    alpha: float,
) -> PairedTestResult:
    """The single-metric function `gym eval test` and `gym eval compare` both call."""
    deltas = paired_task_deltas(baseline, candidate, metric)
    if not deltas:
        return PairedTestResult(
            metric=metric,
            margin=margin,
            alpha=alpha,
            n_pairs=0,
            note=f"no per-task `mean/{metric}` value on both sides for any common task.",
        )
    return paired_test(deltas, metric=metric, alpha=alpha, margin=margin)


def build_paired_test_report(config: PairedTestConfig, command: str) -> PairedTestReport:
    """Load both sides, pick the agent, resolve which metrics to test, and run them all."""
    baseline_file = load_agg_metrics_file(
        config.baseline_rollouts_jsonl_fpath,
        role="baseline",
        aggregate_metrics_fpath_override=config.baseline_aggregate_metrics_fpath,
    )
    (candidate_rollouts_fpath,) = config.candidate_rollouts_jsonl_fpaths
    candidate_agg_override = (
        config.candidate_aggregate_metrics_fpaths[0] if config.candidate_aggregate_metrics_fpaths else None
    )
    candidate_file = load_agg_metrics_file(
        candidate_rollouts_fpath,
        role="candidate",
        index=0,
        aggregate_metrics_fpath_override=candidate_agg_override,
    )

    selections, warnings, _skipped_agents = resolve_agent_selections(
        baseline_file,
        [candidate_file],
        agent_name=config.agent_name,
        baseline_agent_name=config.baseline_agent_name,
        candidate_agent_names=config.candidate_agent_names,
    )
    if len(selections) != 1:
        raise ConfigError(
            "gym eval test compares exactly one agent pair; narrow the selection with --agent, "
            "--baseline-agent, or --candidate-agents."
        )
    selection = selections[0]
    baseline_run = build_loaded_run(baseline_file, selection.baseline_agent)
    candidate_run = build_loaded_run(candidate_file, selection.candidate_agents[0])

    notes: List[str] = []
    if config.metric:
        metrics = list(dict.fromkeys(config.metric))
        for metric in metrics:
            if not paired_task_deltas(baseline_run, candidate_run, metric):
                raise ConfigError(
                    f"--metric '{metric}' has no per-task `mean/{metric}` value on both sides -- nothing to "
                    "test. Check the metric name, or omit --metric to test every key metric with pairing data."
                )
    else:
        metrics, skipped = resolve_metrics(baseline_run, candidate_run, None)
        if not metrics:
            raise ConfigError(
                "No key metric has per-task pairing data to test. Pass --metric explicitly if you know "
                "which field to test."
            )
        if skipped:
            notes.append(f"Skipped {len(skipped)} key metric(s) with no per-task pairing data: {', '.join(skipped)}.")

    results = [
        run_paired_test(baseline_run, candidate_run, metric=metric, margin=config.margin, alpha=config.alpha)
        for metric in metrics
    ]

    return PairedTestReport(
        generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        nemo_gym_version=__version__,
        command=command,
        baseline_rollouts_jsonl_fpath=str(baseline_file.rollouts_jsonl_fpath),
        baseline_aggregate_metrics_fpath=str(baseline_file.aggregate_metrics_fpath),
        candidate_rollouts_jsonl_fpath=str(candidate_file.rollouts_jsonl_fpath),
        candidate_aggregate_metrics_fpath=str(candidate_file.aggregate_metrics_fpath),
        baseline_agent=selection.baseline_agent,
        candidate_agent=selection.candidate_agents[0],
        baseline_task_count=baseline_run.num_tasks,
        candidate_task_count=candidate_run.num_tasks,
        results=results,
        notes=notes,
        warnings=warnings,
    )


def resolve_output_dir(config: PairedTestConfig) -> Path:
    """Where to write the report.

    An explicit `--output-dir` is used literally -- no nesting. Left unset, this nests under
    `statistical_tests/` inside the candidate run's own directory (the same default directory
    `gym eval compare` uses for `compare_report.*`), so the two artifacts sit side by side rather
    than one being written into the other.
    """
    if config.output_dirpath:
        p = Path(config.output_dirpath)
        return p if p.is_absolute() else Path.cwd() / p
    base = _resolve_under_cwd_or_install(config.candidate_rollouts_jsonl_fpaths[-1]).parent
    return base / STATS_SUBDIR_NAME


def run_and_write_paired_test(config: PairedTestConfig, command: str) -> Tuple[PairedTestReport, List[Path]]:
    """Build the report and write its artifacts, for the standalone `gym eval test` CLI path."""
    from nemo_gym.statistical_tests.report import write_paired_test_reports

    report = build_paired_test_report(config, command)
    output_dir = resolve_output_dir(config)
    written = write_paired_test_reports(report, config, output_dir)
    return report, written
