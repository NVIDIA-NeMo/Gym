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
"""`gym compare`: diff a baseline eval run's aggregate metrics against a candidate run."""

import shlex
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Tuple

import orjson

from nemo_gym import _resolve_under_cwd_or_install
from nemo_gym.cli.utils import exit_cleanly_on_config_error, print_rich_table
from nemo_gym.compare.diff import compare_runs
from nemo_gym.compare.loading import (
    build_loaded_run,
    load_run_file,
    resolve_agent_selections,
    run_labels,
)
from nemo_gym.compare.report import render_markdown
from nemo_gym.compare.schema import CompareConfig, ComparisonResult, RunRef
from nemo_gym.config_types import ConfigError
from nemo_gym.global_config import GlobalConfigDictParserConfig, get_global_config_dict
from nemo_gym.package_info import __version__


MARKDOWN_REPORT_NAME = "compare_report.md"
JSON_REPORT_NAME = "compare_report.json"


def build_comparison_result(config: CompareConfig, *, command: str) -> ComparisonResult:
    """Load both sides, pick the agent(s), and assemble the full comparison."""
    baseline_file = load_run_file(
        config.baseline_rollouts_jsonl_fpath,
        role="baseline",
        flag_label="--baseline",
        aggregate_metrics_fpath_override=config.baseline_aggregate_metrics_fpath,
    )
    candidate_files = [
        load_run_file(
            fpath,
            role="candidate",
            index=index,
            flag_label="--candidates",
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

    labels = run_labels([baseline_file.rollouts_jsonl_fpath, *(f.rollouts_jsonl_fpath for f in candidate_files)])
    baseline_label, candidate_labels = labels[0], labels[1:]

    comparisons = []
    for selection in selections:
        baseline_run = build_loaded_run(baseline_file, selection.baseline_agent, label=baseline_label)
        candidate_runs = [
            build_loaded_run(run_file, agent, label=candidate_labels[index])
            for index, (run_file, agent) in enumerate(zip(candidate_files, selection.candidate_agents))
        ]
        comparisons.append(compare_runs(baseline_run, candidate_runs))

    def run_ref(run_file, label: str, repeats) -> RunRef:
        return RunRef(
            role=run_file.role,
            label=label,
            rollouts_jsonl_fpath=str(run_file.rollouts_jsonl_fpath),
            aggregate_metrics_fpath=str(run_file.aggregate_metrics_fpath),
            agents=run_file.agent_names,
            num_repeats=repeats,
        )

    first = comparisons[0]
    return ComparisonResult(
        generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        nemo_gym_version=__version__,
        command=command,
        baseline=run_ref(baseline_file, baseline_label, first.baseline_repeat_count),
        candidates=[
            run_ref(run_file, candidate_labels[index], first.candidate_repeat_counts[index])
            for index, run_file in enumerate(candidate_files)
        ],
        comparisons=comparisons,
        skipped_agents=skipped,
        # Run-level warnings only. Per-comparison observations stay on their `AgentComparison.notes`
        # so the report renders each one once, under the agent it applies to.
        warnings=warnings,
    )


def resolve_output_dir(config: CompareConfig) -> Path:
    """`--output-dir`, defaulting to the candidate run's own directory.

    The default resolves the candidate path the same way loading does, so the report lands next to
    the metrics file that was actually read rather than at a same-named path under the cwd.
    """
    if config.output_dirpath:
        return Path(config.output_dirpath)
    return _resolve_under_cwd_or_install(config.candidate_rollouts_jsonl_fpaths[-1]).parent


def write_reports(result: ComparisonResult, output_dir: Path, report_format: str) -> List[Path]:
    """Write the requested artifacts, returning the paths written."""
    if output_dir.exists() and not output_dir.is_dir():
        raise ConfigError(f"--output-dir '{output_dir}' exists and is not a directory.")
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        raise ConfigError(f"Cannot write to --output-dir '{output_dir}': {e}") from e

    written: List[Path] = []
    if report_format in ("md", "both"):
        markdown_fpath = output_dir / MARKDOWN_REPORT_NAME
        markdown_fpath.write_text(render_markdown(result))
        written.append(markdown_fpath)
    if report_format in ("json", "both"):
        json_fpath = output_dir / JSON_REPORT_NAME
        json_fpath.write_bytes(orjson.dumps(result.model_dump(mode="json"), option=orjson.OPT_INDENT_2))
        written.append(json_fpath)
    return written


def summary_lines(result: ComparisonResult, written: List[Path]) -> Tuple[str, ...]:
    """The short stdout recap printed after the files are written."""
    lines = [
        f"Baseline:  {result.baseline.aggregate_metrics_fpath}",
        *(f"Candidate: {run.aggregate_metrics_fpath}" for run in result.candidates),
        f"Agents compared: {', '.join(sorted({c.baseline_agent for c in result.comparisons}))}",
    ]
    for comparison in result.comparisons:
        for summary in comparison.flips:
            if summary.mode == "binary":
                lines.append(
                    f"Sample flips: {summary.fail_to_pass_count} fail→pass, "
                    f"{summary.pass_to_fail_count} pass→fail over {summary.common_task_count} common tasks"
                )
            elif summary.mode == "continuous":
                lines.append(
                    f"Per-task changes: {len(summary.flips)} of {summary.common_task_count} common tasks moved"
                )
            else:
                lines.append(f"Sample flips unavailable: {summary.reason}")
    lines += [f"Wrote: {path}" for path in written]
    return tuple(lines)


def _print_key_metrics_table(result: ComparisonResult) -> None:  # pragma: no cover
    from rich.markup import escape
    from rich.table import Table

    from nemo_gym.compare.report import _fmt, _fmt_ci, _fmt_delta_cell

    for comparison in result.comparisons:
        table = Table(title=f"Key metrics — {comparison.baseline_agent}")
        table.add_column("Metric")
        table.add_column("Drop (cand - base)", justify="right")
        table.add_column("Baseline", justify="right")
        table.add_column("Baseline 95% CI")
        table.add_column("Candidate", justify="right")
        table.add_column("Candidate 95% CI")
        for row in comparison.metrics:
            if not row.is_key_metric:
                continue
            candidate = row.candidates[0] if row.candidates else None
            # escape() so the `[avg-of-k]` in pass@k metric names isn't parsed as Rich markup.
            table.add_row(
                escape(row.metric),
                _fmt_delta_cell(candidate),
                _fmt(row.baseline.value if row.baseline else None),
                _fmt_ci(row.baseline),
                _fmt(candidate.value if candidate else None),
                _fmt_ci(candidate),
            )
        print_rich_table(table)


@exit_cleanly_on_config_error
def compare() -> None:  # pragma: no cover
    global_config_dict = get_global_config_dict(
        global_config_dict_parser_config=GlobalConfigDictParserConfig(
            initial_global_config_dict=GlobalConfigDictParserConfig.NO_MODEL_GLOBAL_CONFIG_DICT,
        )
    )
    config = CompareConfig.model_validate(global_config_dict)

    result = build_comparison_result(config, command=shlex.join(["gym", "compare", *sys.argv[1:]]))
    written = write_reports(result, resolve_output_dir(config), config.report_format)

    _print_key_metrics_table(result)
    print("\n".join(summary_lines(result, written)))
