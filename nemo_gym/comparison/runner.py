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
from nemo_gym.comparison.diff import compare_runs
from nemo_gym.comparison.loading import build_loaded_run, load_agg_metrics_file, resolve_agent_selections
from nemo_gym.comparison.report import write_reports
from nemo_gym.comparison.schema import ComparisonConfig, ComparisonResult
from nemo_gym.package_info import __version__
from nemo_gym.secret_utils import hide_secrets_in_overrides


def invoked_command() -> str:
    """The `gym eval compare` invocation, for provenance in the report."""
    return shlex.join(["gym", "eval", "compare", *hide_secrets_in_overrides(sys.argv[1:])])


def build_comparison_result(config: ComparisonConfig, command: str) -> ComparisonResult:
    """Load both sides, pick the agent(s), and assemble the full comparison."""
    baseline_file = load_agg_metrics_file(
        config.baseline_rollouts_jsonl_fpath,
        role="baseline",
        aggregate_metrics_fpath_override=config.baseline_aggregate_metrics_fpath,
    )
    candidate_files = [
        load_agg_metrics_file(
            fpath,
            role="candidate",
            index=index,
            aggregate_metrics_fpath_override=(
                config.candidate_aggregate_metrics_fpaths[index] if config.candidate_aggregate_metrics_fpaths else None
            ),
        )
        for index, fpath in enumerate(config.candidate_rollouts_jsonl_fpaths)
    ]

    selections, warnings, skipped = resolve_agent_selections(
        baseline_file,
        candidate_files,
        agent_name=config.agent_name,
        baseline_agent_name=config.baseline_agent_name,
        candidate_agent_names=config.candidate_agent_names,
    )

    comparisons = []
    for selection in selections:
        baseline_run = build_loaded_run(baseline_file, selection.baseline_agent)
        candidate_runs = [
            build_loaded_run(run_file, agent) for run_file, agent in zip(candidate_files, selection.candidate_agents)
        ]
        comparisons.append(compare_runs(baseline_run, candidate_runs))

    return ComparisonResult(
        generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        nemo_gym_version=__version__,
        command=command,
        baseline=baseline_file,
        candidates=candidate_files,
        comparisons=comparisons,
        skipped_agents=skipped,
        # Run-level warnings only. Per-comparison observations stay on their `AgentComparison.notes`
        # so the report renders each one once, under the agent it applies to.
        warnings=warnings,
    )


def resolve_output_dir(config: ComparisonConfig) -> Path:
    """`--output-dir`, defaulting to the candidate run's own directory.

    The default resolves the candidate path the same way loading does, so the report lands next to
    the metrics file that was actually read rather than at a same-named path under the cwd.
    """
    if config.output_dirpath:
        p = Path(config.output_dirpath)
        return p if p.is_absolute() else Path.cwd() / p
    # if we use the same location as the candidate rollouts, it should be save to write there and we can use _resolve_under_cwd_or_install function
    return _resolve_under_cwd_or_install(config.candidate_rollouts_jsonl_fpaths[-1]).parent


def run_comparison(config: ComparisonConfig, command: str) -> Tuple[ComparisonResult, List[Path]]:
    """Build the comparison and write its report artifacts."""
    result = build_comparison_result(config, command)
    return result, write_reports(result, resolve_output_dir(config), config.report_format)
