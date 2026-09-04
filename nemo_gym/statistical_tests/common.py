# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Everything a statistical test needs that isn't its own statistic: loading the run pair, and writing its report."""

import re
import shlex
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import orjson

from nemo_gym import _resolve_under_cwd_or_install
from nemo_gym.comparison.loading import LoadedRun, build_loaded_run, load_agg_metrics_file, resolve_agent_selections
from nemo_gym.comparison.schema import RunFile
from nemo_gym.config_types import ConfigError
from nemo_gym.package_info import __version__
from nemo_gym.secret_utils import hide_secrets_in_overrides
from nemo_gym.statistical_tests.schema import STATS_SUBDIR_NAME, StatTestConfig


MISSING = "—"


def fmt(value: Optional[float]) -> str:
    return MISSING if value is None else (f"{value:.4f}" if abs(value) < 10 else f"{value:.2f}")


def fmt_p(value: Optional[float]) -> str:
    return MISSING if value is None else (f"{value:.4f}" if value >= 0.0001 else f"{value:.2e}")


def fmt_bool(value: Optional[bool]) -> str:
    return MISSING if value is None else ("yes" if value else "no")


@dataclass(frozen=True)
class RunPair:
    baseline_file: RunFile
    candidate_file: RunFile
    baseline: LoadedRun
    candidate: LoadedRun
    baseline_agent: str
    candidate_agent: str
    warnings: List[str]

    def report_identity(self, config: StatTestConfig, command: str) -> Dict[str, Any]:
        return {
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "nemo_gym_version": __version__,
            "command": command,
            "test": config.test,
            "baseline_rollouts_jsonl_fpath": str(self.baseline_file.rollouts_jsonl_fpath),
            "baseline_aggregate_metrics_fpath": str(self.baseline_file.aggregate_metrics_fpath),
            "candidate_rollouts_jsonl_fpath": str(self.candidate_file.rollouts_jsonl_fpath),
            "candidate_aggregate_metrics_fpath": str(self.candidate_file.aggregate_metrics_fpath),
            "baseline_agent": self.baseline_agent,
            "candidate_agent": self.candidate_agent,
            "baseline_task_count": self.baseline.num_tasks,
            "candidate_task_count": self.candidate.num_tasks,
            "warnings": self.warnings,
        }


def load_run_pair(config: StatTestConfig) -> RunPair:
    baseline_file = load_agg_metrics_file(
        config.baseline_rollouts_jsonl_fpath,
        role="baseline",
        aggregate_metrics_fpath_override=config.baseline_aggregate_metrics_fpath,
    )
    (candidate_fpath,) = config.candidate_rollouts_jsonl_fpaths
    candidate_agg = config.candidate_aggregate_metrics_fpaths[0] if config.candidate_aggregate_metrics_fpaths else None
    candidate_file = load_agg_metrics_file(
        candidate_fpath, role="candidate", index=0, aggregate_metrics_fpath_override=candidate_agg
    )

    selections, warnings, _skipped = resolve_agent_selections(
        baseline_file,
        [candidate_file],
        agent_name=config.agent_name,
        baseline_agent_name=config.baseline_agent_name,
        candidate_agent_names=config.candidate_agent_names,
    )
    if len(selections) != 1:
        raise ConfigError(
            "gym eval stat-test compares exactly one agent pair; narrow the selection with --agent, "
            "--baseline-agent, or --candidate-agents."
        )
    s = selections[0]
    return RunPair(
        baseline_file,
        candidate_file,
        build_loaded_run(baseline_file, s.baseline_agent),
        build_loaded_run(candidate_file, s.candidate_agents[0]),
        s.baseline_agent,
        s.candidate_agents[0],
        warnings,
    )


def invoked_command(subcommand: str = "stat-test") -> str:
    return shlex.join(["gym", "eval", subcommand, *hide_secrets_in_overrides(sys.argv[1:])])


def sanitize_filename_part(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.+-]+", "-", text).strip("-")


def report_stem(config: StatTestConfig) -> str:
    return "__".join([sanitize_filename_part(config.test), *config.filename_parts(), f"alpha-{config.alpha:g}"])


def resolve_output_dir(config: StatTestConfig) -> Path:
    if config.output_dirpath:
        p = Path(config.output_dirpath)
        return p if p.is_absolute() else Path.cwd() / p
    return _resolve_under_cwd_or_install(config.candidate_rollouts_jsonl_fpaths[-1]).parent / STATS_SUBDIR_NAME


def write_reports(output_dir: Path, stem: str, *, report_format: str, markdown: str, payload: dict) -> List[Path]:
    if output_dir.exists() and not output_dir.is_dir():
        raise ConfigError(f"--output-dir '{output_dir}' exists and is not a directory.")
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        written = []
        if report_format in ("md", "both"):
            (output_dir / f"{stem}.md").write_text(markdown, encoding="utf-8")
            written.append(output_dir / f"{stem}.md")
        if report_format in ("json", "both"):
            (output_dir / f"{stem}.json").write_bytes(orjson.dumps(payload, option=orjson.OPT_INDENT_2))
            written.append(output_dir / f"{stem}.json")
        return written
    except OSError as e:
        raise ConfigError(f"Cannot write the report into '{output_dir}': {e}") from e
