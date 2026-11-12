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
"""End-to-end execution of `gym eval compare`."""

import shlex
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Tuple

from nemo_gym import _resolve_under_cwd_or_install
from nemo_gym._moved_paths import moved_names_getattr
from nemo_gym.config.secret_utils import hide_secrets_in_overrides
from nemo_gym.eval.comparison.diff import compare_runs
from nemo_gym.eval.comparison.loading import (
    build_loaded_run,
    combine_run_files,
    load_agg_metrics_file,
    resolve_source_agent_selections,
    validate_independent_sources,
)
from nemo_gym.eval.comparison.pooling import pool_loaded_runs
from nemo_gym.eval.comparison.report import write_reports
from nemo_gym.eval.comparison.schema import ComparisonConfig, ComparisonResult
from nemo_gym.package_info import __version__


def invoked_command() -> str:
    """The `gym eval compare` invocation, for provenance in the report."""
    return shlex.join(["gym", "eval", "compare", *hide_secrets_in_overrides(sys.argv[1:])])


def build_comparison_result(config: ComparisonConfig, command: str) -> ComparisonResult:
    """Load both sides, pick the agent(s), and assemble the full comparison."""
    baseline_pooled = config.baseline_rollouts_jsonl_fpaths is not None
    candidate_pooled = config.pooled_candidate_rollouts_jsonl_fpaths is not None
    baseline_paths = config.baseline_rollouts_jsonl_fpaths or [config.baseline_rollouts_jsonl_fpath]
    baseline_overrides = config.baseline_aggregate_metrics_fpaths or [config.baseline_aggregate_metrics_fpath]
    baseline_files = [
        load_agg_metrics_file(
            fpath,
            role="baseline",
            index=index,
            aggregate_metrics_fpath_override=baseline_overrides[index] if index < len(baseline_overrides) else None,
        )
        for index, fpath in enumerate(baseline_paths)
    ]
    candidate_paths = config.pooled_candidate_rollouts_jsonl_fpaths or config.candidate_rollouts_jsonl_fpaths
    candidate_files = [
        load_agg_metrics_file(
            fpath,
            role="candidate",
            index=index,
            aggregate_metrics_fpath_override=(
                config.candidate_aggregate_metrics_fpaths[index] if config.candidate_aggregate_metrics_fpaths else None
            ),
        )
        for index, fpath in enumerate(candidate_paths)
    ]
    selections, warnings, skipped = resolve_source_agent_selections(
        baseline_files=baseline_files,
        candidate_files=candidate_files,
        agent_name=config.agent_name,
        baseline_agent_name=config.baseline_agent_name,
        baseline_agent_names=config.baseline_agent_names,
        candidate_agent_names=config.candidate_agent_names,
    )

    comparisons = []
    for selection in selections:
        if baseline_pooled or candidate_pooled:
            validate_independent_sources(
                baseline_files=baseline_files, candidate_files=candidate_files, selection=selection
            )
        baseline_runs = [
            build_loaded_run(source, agent) for source, agent in zip(baseline_files, selection.baseline_agents)
        ]
        baseline_run = (
            pool_loaded_runs(
                baseline_runs, source_labels=[str(source.aggregate_metrics_fpath) for source in baseline_files]
            )
            if baseline_pooled
            else baseline_runs[0]
        )
        candidate_runs = [
            build_loaded_run(run_file, agent) for run_file, agent in zip(candidate_files, selection.candidate_agents)
        ]
        if candidate_pooled:
            candidate_runs = [
                pool_loaded_runs(
                    candidate_runs, source_labels=[str(source.aggregate_metrics_fpath) for source in candidate_files]
                )
            ]
        comparisons.append(compare_runs(baseline_run, candidate_runs, confidence_level=config.confidence_level))

    baseline_summary = (
        combine_run_files(
            baseline_files,
            agent_names_by_source=[
                [selection.baseline_agents[index] for selection in selections] for index in range(len(baseline_files))
            ],
        )
        if baseline_pooled
        else baseline_files[0]
    )
    candidate_summaries = (
        [
            combine_run_files(
                candidate_files,
                agent_names_by_source=[
                    [selection.candidate_agents[index] for selection in selections]
                    for index in range(len(candidate_files))
                ],
            )
        ]
        if candidate_pooled
        else candidate_files
    )
    return ComparisonResult(
        generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        nemo_gym_version=__version__,
        command=command,
        confidence_level=config.confidence_level,
        baseline=baseline_summary,
        candidates=candidate_summaries,
        comparisons=comparisons,
        skipped_agents=skipped,
        # Run-level warnings only. Per-comparison observations stay on their `AgentComparison.notes`
        # so the report renders each one once, under the agent it applies to.
        warnings=warnings,
    )


def resolve_output_dir(config: ComparisonConfig) -> Path:
    """`--output-dir`, defaulting to the last candidate source's directory.

    The default resolves the candidate path the same way loading does, so the report lands next to
    the metrics file that was actually read rather than at a same-named path under the cwd.
    """
    if config.output_dirpath:
        p = Path(config.output_dirpath)
        return p if p.is_absolute() else Path.cwd() / p
    # if we use the same location as the candidate rollouts, it should be save to write there and we can use _resolve_under_cwd_or_install function
    paths = config.pooled_candidate_rollouts_jsonl_fpaths or config.candidate_rollouts_jsonl_fpaths
    return _resolve_under_cwd_or_install(paths[-1]).parent


def run_comparison(config: ComparisonConfig, command: str) -> Tuple[ComparisonResult, List[Path]]:
    """Build the comparison and write its report artifacts."""
    result = build_comparison_result(config, command)
    return result, write_reports(result, resolve_output_dir(config), config.report_format)


__getattr__ = moved_names_getattr(__name__)
