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
"""Reading `*_aggregate_metrics.json` for `gym eval compare`, and picking which agent to compare.

All filesystem I/O for the compare feature lives here. The rollouts JSONL a user points at is
never opened: it is the run's identity and the handle its `_aggregate_metrics.json` sibling is
derived from.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

import orjson

from nemo_gym import _resolve_under_cwd_or_install
from nemo_gym.cli.utils import did_you_mean
from nemo_gym.comparison.schema import RunFile
from nemo_gym.config_types import ConfigError, ConfigPathNotFoundError
from nemo_gym.global_config import (
    AGENT_REF_KEY_NAME,
    CI_LOW_95_ACROSS_REPEATS_PREFIX,
    EXPECTED_NUM_ROLLOUTS_KEY_NAME,
)
from nemo_gym.path_utils import aggregate_metrics_path_for


RunRole = Literal["baseline", "candidate"]


@dataclass(frozen=True)
class LoadedRun:
    """One side of the comparison, narrowed to a single agent.

    Only what the diff actually consumes. Run identity (role, paths, label) stays on `RunFile`,
    which the report carries directly, so it is deliberately not duplicated here.
    """

    agent_name: str
    agent_metrics: Dict[str, Any]
    key_metrics: Dict[str, Any]
    group_level_metrics: List[Dict[str, Any]]
    num_tasks: int = 0
    num_repeats: Optional[int] = None
    has_repeat_cis: bool = False


@dataclass(frozen=True)
class AgentSelection:
    """Which agent to read from each side for one comparison block."""

    baseline_agent: str
    candidate_agents: Tuple[str, ...]


def resolve_aggregate_metrics_fpath(rollouts_jsonl_fpath: str, override: Optional[str]) -> Path:
    """The `*_aggregate_metrics.json` to read: the explicit override, else the derived sibling."""
    if override:
        return _resolve_under_cwd_or_install(override)
    return aggregate_metrics_path_for(_resolve_under_cwd_or_install(rollouts_jsonl_fpath))


def _read_agent_entries(metrics_fpath: Path) -> Dict[str, Dict[str, Any]]:
    try:
        raw = metrics_fpath.read_bytes()
    except OSError as e:
        raise ConfigError(f"Cannot read aggregate metrics at '{metrics_fpath}': {e}") from e

    try:
        payload = orjson.loads(raw)
    except orjson.JSONDecodeError as e:
        raise ConfigError(f"'{metrics_fpath}' is not valid JSON: {e}") from e

    if not isinstance(payload, list):
        raise ConfigError(
            f"'{metrics_fpath}' is not a valid aggregate-metrics file (expected a JSON list of per-agent entries)."
        )
    if not payload:
        raise ConfigError(f"'{metrics_fpath}' contains no agent entries.")

    entries: Dict[str, Dict[str, Any]] = {}
    for position, entry in enumerate(payload):
        if not isinstance(entry, dict):
            raise ConfigError(
                f"'{metrics_fpath}' entry {position} is not an object (expected a JSON list of per-agent entries)."
            )
        agent_name = (entry.get(AGENT_REF_KEY_NAME) or {}).get("name")
        if not agent_name:
            raise ConfigError(f"'{metrics_fpath}' entry {position} has no `{AGENT_REF_KEY_NAME}.name`.")
        if agent_name in entries:
            raise ConfigError(f"'{metrics_fpath}' contains two entries for agent '{agent_name}'.")
        entries[agent_name] = entry
    return entries


def load_agg_metrics_file(
    rollouts_jsonl_fpath: str,
    role: RunRole,
    index: int = 0,
    aggregate_metrics_fpath_override: Optional[str] = None,
) -> RunFile:
    """Resolve and parse one run's `*_aggregate_metrics.json`."""
    metrics_path = resolve_aggregate_metrics_fpath(rollouts_jsonl_fpath, aggregate_metrics_fpath_override)

    if not metrics_path.exists():
        raise ConfigPathNotFoundError(
            f"Aggregate metrics not found: '{metrics_path}', derived from '{rollouts_jsonl_fpath}'. "
            "The run may have been collected with --disable-aggregation: run `gym eval aggregate` first, "
            "or point at the metrics file directly with "
            "'baseline_aggregate_metrics_fpath' or 'candidate_aggregate_metrics_fpaths'."
        )

    return RunFile(
        role=role,
        index=index,
        rollouts_jsonl_fpath=_resolve_under_cwd_or_install(rollouts_jsonl_fpath),
        aggregate_metrics_fpath=metrics_path,
        entries_by_agent=_read_agent_entries(metrics_path),
    )


def _sole_agent(run_file: RunFile) -> str:
    """The file's only agent, or an error asking which of several to use."""
    if len(run_file.entries_by_agent) == 1:
        return next(iter(run_file.entries_by_agent))
    raise ConfigError(
        f"The {run_file.role} run's metrics at '{run_file.aggregate_metrics_fpath}' contain "
        f"{len(run_file.entries_by_agent)} agents: {', '.join(run_file.agent_names)}. "
        "Set 'agent_name' to compare one of them on both sides, or 'baseline_agent_name' / "
        "'candidate_agent_names' to choose a different agent per side."
    )


def _require_agent(run_file: RunFile, agent_name: str) -> None:
    if agent_name not in run_file.entries_by_agent:
        raise ConfigError(
            f"Agent '{agent_name}' is not in '{run_file.aggregate_metrics_fpath}'. "
            f"Available: {', '.join(run_file.agent_names)}." + did_you_mean(agent_name, run_file.agent_names)
        )


def resolve_agent_selections(
    baseline_file: RunFile,
    candidate_files: Sequence[RunFile],
    *,
    agent_name: Optional[str] = None,
    baseline_agent_name: Optional[str] = None,
    candidate_agent_names: Optional[Sequence[str]] = None,
) -> Tuple[List[AgentSelection], List[str], Dict[str, List[str]]]:
    """Decide which agent to read from each side.

    Precedence is most-specific-first: a per-side name beats `agent_name`, which beats the default
    full join over agent names common to every run. Returns the selections, any warnings, and the
    agents that were present but not compared (keyed by run label).
    """
    warnings: List[str] = []
    skipped: Dict[str, List[str]] = {}
    per_side_given = baseline_agent_name is not None or candidate_agent_names is not None

    if per_side_given:
        baseline_agent = baseline_agent_name or agent_name or _sole_agent(baseline_file)
        _require_agent(baseline_file, baseline_agent)
        candidates: List[str] = []
        for position, run_file in enumerate(candidate_files):
            explicit = candidate_agent_names[position] if candidate_agent_names else None
            candidate_agent = explicit or agent_name or _sole_agent(run_file)
            _require_agent(run_file, candidate_agent)
            candidates.append(candidate_agent)
        return [AgentSelection(baseline_agent, tuple(candidates))], warnings, skipped

    if agent_name:
        for run_file in (baseline_file, *candidate_files):
            _require_agent(run_file, agent_name)
        return [AgentSelection(agent_name, tuple(agent_name for _ in candidate_files))], warnings, skipped

    matched = set(baseline_file.entries_by_agent)
    for run_file in candidate_files:
        matched &= set(run_file.entries_by_agent)

    if not matched:
        listing = "\n".join(
            f"  {run_file.role} ({run_file.aggregate_metrics_fpath}): {', '.join(run_file.agent_names)}"
            for run_file in (baseline_file, *candidate_files)
        )
        raise ConfigError(
            "No agent name is present in every run, so there is nothing to compare.\n"
            f"{listing}\n"
            "Set 'agent_name' to force one agent on both sides, or 'baseline_agent_name' / "
            "'candidate_agent_names' to pick a different agent per side."
        )

    for run_file in (baseline_file, *candidate_files):
        extras = sorted(set(run_file.entries_by_agent) - matched)
        if extras:
            label = f"{run_file.role}[{run_file.index}]" if run_file.role == "candidate" else run_file.role
            skipped[label] = extras
            warnings.append(f"Agent(s) {', '.join(extras)} are present only in {label} and were not compared.")

    return (
        [AgentSelection(name, tuple(name for _ in candidate_files)) for name in sorted(matched)],
        warnings,
        skipped,
    )


def _derive_num_repeats(
    group_level_metrics: List[Dict[str, Any]],
    repeat_level_metrics: List[Dict[str, Any]],
) -> Optional[int]:
    """How many repeats the run collected: the most any single task has.

    `expected_num_rollouts` is per task, and a partially recovered run leaves some tasks short of
    the rest, so the max is the run's repeat count. `repeat_level_metrics` has exactly one entry
    per repeat but is absent from single-repeat runs and from files written before it existed.
    """
    expected = [
        group[EXPECTED_NUM_ROLLOUTS_KEY_NAME]
        for group in group_level_metrics
        if isinstance(group.get(EXPECTED_NUM_ROLLOUTS_KEY_NAME), int)
    ]
    if expected:
        return max(expected)
    if repeat_level_metrics:
        return len(repeat_level_metrics)
    return None


def build_loaded_run(run_file: RunFile, agent_name: str) -> LoadedRun:
    """Narrow a parsed run file to one agent."""
    entry = run_file.entries_by_agent[agent_name]
    agent_metrics = entry.get("agent_metrics") or {}
    key_metrics = entry.get("key_metrics") or {}
    group_level_metrics = entry.get("group_level_metrics") or []
    repeat_level_metrics = entry.get("repeat_level_metrics") or []

    return LoadedRun(
        agent_name=agent_name,
        agent_metrics=agent_metrics,
        key_metrics=key_metrics,
        group_level_metrics=group_level_metrics,
        num_tasks=len(group_level_metrics),
        num_repeats=_derive_num_repeats(group_level_metrics, repeat_level_metrics),
        has_repeat_cis=any(key.startswith(CI_LOW_95_ACROSS_REPEATS_PREFIX) for key in agent_metrics),
    )
