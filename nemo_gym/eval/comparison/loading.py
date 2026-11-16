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

Input file I/O for the compare feature lives here. The rollouts JSONL a user points at is
never opened: it is the run's identity and the handle from which its `_aggregate_metrics.json`
sibling is derived. Aggregate files written before repeat-level statistics existed are enriched from
their already-recorded per-rollout summaries in memory.
"""

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Mapping, Optional, Sequence, Tuple

import orjson

from nemo_gym import _resolve_under_cwd_or_install
from nemo_gym.config.config_types import ConfigError, ConfigPathNotFoundError
from nemo_gym.config.global_config import (
    AGENT_REF_KEY_NAME,
    EXPECTED_NUM_ROLLOUTS_KEY_NAME,
    ROLLOUT_INDEX_KEY_NAME,
    ROLLOUT_INFOS_KEY_NAME,
)
from nemo_gym.eval.comparison.schema import RunFile, RunSource
from nemo_gym.metrics.metrics_config import COMPLETION_TOKEN_METRIC_NAMES, Stat
from nemo_gym.rollouts.rollout_records import resolve_rollout_path
from nemo_gym.utils.path_utils import aggregate_metrics_path_for, task_selection_path_for
from nemo_gym.utils.task_ids import read_task_selection_record
from nemo_gym.utils.text import did_you_mean


RunRole = Literal["baseline", "candidate"]


def is_token_accounting_only(row: Mapping[str, object]) -> bool:
    """Identify recorded repeat rows that contain accounting without quality estimates."""
    measurements = row.keys() - {AGENT_REF_KEY_NAME, ROLLOUT_INDEX_KEY_NAME}
    return bool(measurements) and measurements <= set(COMPLETION_TOKEN_METRIC_NAMES)


@dataclass(frozen=True)
class LoadedRun:
    """One side of the comparison, narrowed to a single agent."""

    agent_name: str
    agent_metrics: Dict[str, Any]
    key_metrics: Dict[str, Any]
    group_level_metrics: List[Dict[str, Any]]
    repeat_level_metrics: List[Dict[str, Any]]
    num_tasks: int = 0
    num_repeats: Optional[int] = None
    has_repeat_cis: bool = False
    pooled: bool = False
    notes: Tuple[str, ...] = ()
    flip_unavailable_reason: Optional[str] = None


@dataclass(frozen=True)
class AgentSelection:
    """Which agent to read from each side for one comparison block."""

    baseline_agent: str
    candidate_agents: Tuple[str, ...]


@dataclass(frozen=True)
class SourceAgentSelection:
    """Agents selected for every source in the two comparison sides."""

    baseline_agents: Tuple[str, ...]
    candidate_agents: Tuple[str, ...]


def resolve_aggregate_metrics_fpath(rollouts_jsonl_fpath: str, override: Optional[str]) -> Path:
    """The `*_aggregate_metrics.json` to read: the explicit override, else the derived sibling."""
    if override:
        return _resolve_under_cwd_or_install(override)
    output = resolve_rollout_path(_resolve_under_cwd_or_install(rollouts_jsonl_fpath), read_only=True)
    return aggregate_metrics_path_for(output)


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


def _compute_repeat_metrics(entry: Dict[str, Any]) -> Dict[str, Any]:
    """Add repeat metrics to one legacy aggregate entry."""
    from nemo_gym.metrics.reward_profile import compute_aggregate_metrics

    computed = compute_aggregate_metrics(
        [info for group in entry.get("group_level_metrics", []) for info in group.get(ROLLOUT_INFOS_KEY_NAME, [])]
    )
    return entry | {
        "agent_metrics": computed.agent_metrics
        | (entry.get("agent_metrics") or {})
        | {"num_repeats": _derive_num_repeats(entry.get("group_level_metrics", []), computed.repeat_level_metrics)},
        "repeat_level_metrics": computed.repeat_level_metrics,
    }


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
            "--baseline-agg-metrics for --baseline, --baseline-runs-agg-metrics for --baseline-runs, "
            "or --candidates-agg-metrics for either candidate option."
        )
    entries_by_agent = _read_agent_entries(metrics_path)
    if any("repeat_level_metrics" not in entry for entry in entries_by_agent.values()):
        warnings.warn(
            f"Repeat-level metrics are missing in '{metrics_path}'; calculating them from stored rollout summaries.",
            stacklevel=2,
        )
        entries_by_agent = {
            name: entry if "repeat_level_metrics" in entry else _compute_repeat_metrics(entry)
            for name, entry in entries_by_agent.items()
        }
    rollouts = _resolve_under_cwd_or_install(rollouts_jsonl_fpath)
    return RunFile(
        role=role,
        index=index,
        rollouts_jsonl_fpath=rollouts,
        aggregate_metrics_fpath=metrics_path,
        entries_by_agent=entries_by_agent,
        task_selection=_task_selection_summary(rollouts),
    )


def _task_selection_summary(rollouts: Path) -> Optional[Dict[str, Any]]:
    """A subset run's selection without its ids (source, counts, digest), from the record beside its rollouts."""
    record = read_task_selection_record(task_selection_path_for(rollouts))
    if record is None:
        return None
    return {key: record.get(key) for key in ("source", "tasks", "rows", "task_ids_sha256")}


def _sole_agent(run_file: RunFile) -> str:
    """The file's only agent, or an error asking which of several to use."""
    if len(run_file.entries_by_agent) == 1:
        return next(iter(run_file.entries_by_agent))
    raise ConfigError(
        f"The {run_file.role} run's metrics at '{run_file.aggregate_metrics_fpath}' contain "
        f"{len(run_file.entries_by_agent)} agents: {', '.join(run_file.agent_names)}. "
        "Use --agent to select one on both sides, --baseline-agent to select one for all baseline sources, "
        "or --baseline-agents / --candidate-agents to select an agent per pooled source."
    )


def _require_agent(run_file: RunFile, agent_name: str) -> None:
    if agent_name not in run_file.entries_by_agent:
        raise ConfigError(
            f"Agent '{agent_name}' is not in '{run_file.aggregate_metrics_fpath}'. "
            f"Available: {', '.join(run_file.agent_names)}." + did_you_mean(agent_name, run_file.agent_names)
        )


def _common_agents(run_files: Sequence[RunFile]) -> Tuple[List[str], List[str], Dict[str, List[str]]]:
    """Find the default agent join and account for entries outside it."""
    matched = set(run_files[0].entries_by_agent)
    for run_file in run_files[1:]:
        matched &= set(run_file.entries_by_agent)
    if not matched:
        listing = "\n".join(
            f"  {run_file.role} ({run_file.aggregate_metrics_fpath}): {', '.join(run_file.agent_names)}"
            for run_file in run_files
        )
        raise ConfigError(
            "No agent name is present in every run, so there is nothing to compare.\n"
            f"{listing}\n"
            "Use --agent to select one on both sides, --baseline-agent to select one for all baseline sources, "
            "or --baseline-agents / --candidate-agents to select an agent per pooled source."
        )
    warnings: List[str] = []
    skipped: Dict[str, List[str]] = {}
    for run_file in run_files:
        extras = sorted(set(run_file.entries_by_agent) - matched)
        if extras:
            label = (
                f"{run_file.role}[{run_file.index}]"
                if run_file.role == "candidate" or run_file.index
                else run_file.role
            )
            skipped[label] = extras
            warnings.append(f"Agent(s) {', '.join(extras)} are present only in {label} and were not compared.")
    return sorted(matched), warnings, skipped


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

    matched, warnings, skipped = _common_agents((baseline_file, *candidate_files))
    return (
        [AgentSelection(name, tuple(name for _ in candidate_files)) for name in matched],
        warnings,
        skipped,
    )


def resolve_source_agent_selections(
    *,
    baseline_files: Sequence[RunFile],
    candidate_files: Sequence[RunFile],
    agent_name: Optional[str] = None,
    baseline_agent_name: Optional[str] = None,
    baseline_agent_names: Optional[Sequence[str]] = None,
    candidate_agent_names: Optional[Sequence[str]] = None,
) -> Tuple[List[SourceAgentSelection], List[str], Dict[str, List[str]]]:
    """Select common agents across all sources, or honor explicit per-source names."""
    if baseline_agent_name is None and baseline_agent_names is None and candidate_agent_names is None:
        files = (*baseline_files, *candidate_files)
        if agent_name:
            for run_file in files:
                _require_agent(run_file, agent_name)
            matched, notes, skipped = [agent_name], [], {}
        else:
            matched, notes, skipped = _common_agents(files)
        return (
            [
                SourceAgentSelection(
                    tuple(name for _ in baseline_files),
                    tuple(name for _ in candidate_files),
                )
                for name in matched
            ],
            notes,
            skipped,
        )

    baseline_agents = []
    for index, source in enumerate(baseline_files):
        name = (
            baseline_agent_names[index]
            if baseline_agent_names is not None
            else baseline_agent_name or agent_name or _sole_agent(source)
        )
        _require_agent(source, name)
        baseline_agents.append(name)
    candidate_agents = []
    for index, source in enumerate(candidate_files):
        name = candidate_agent_names[index] if candidate_agent_names is not None else agent_name or _sole_agent(source)
        _require_agent(source, name)
        candidate_agents.append(name)
    return [SourceAgentSelection(tuple(baseline_agents), tuple(candidate_agents))], [], {}


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
        repeat_level_metrics=repeat_level_metrics,
        num_tasks=len(group_level_metrics),
        num_repeats=_derive_num_repeats(group_level_metrics, repeat_level_metrics),
        has_repeat_cis=any(key.startswith(Stat.CI_LOW_95.across_repeats_prefix) for key in agent_metrics),
    )


def validate_independent_sources(
    *, baseline_files: Sequence[RunFile], candidate_files: Sequence[RunFile], selection: SourceAgentSelection
) -> None:
    """Reject reused selected source observations, including filesystem aliases."""
    seen: Dict[Tuple[str, Path, str], RunFile] = {}
    files = (*baseline_files, *candidate_files)
    agents = (*selection.baseline_agents, *selection.candidate_agents)
    for source, agent in zip(files, agents, strict=True):
        for kind, path in (
            ("Aggregate metrics", source.aggregate_metrics_fpath),
            ("Rollouts", source.rollouts_jsonl_fpath),
        ):
            identity = kind, path.resolve(), agent
            if identity in seen:
                previous = seen[identity]
                raise ConfigError(
                    f"{kind} '{path}' are reused by {previous.role}[{previous.index}] and "
                    f"{source.role}[{source.index}] for agent '{agent}'. Each source must be an independent run; duplicate or "
                    "shared baseline/candidate artifacts would count the same repeats more than once."
                )
            seen[identity] = source


def combine_run_files(sources: Sequence[RunFile], *, agent_names_by_source: Sequence[Sequence[str]]) -> RunFile:
    """Represent a pooled side while retaining every source path and its recorded counts."""
    provenance = []
    for source, selected in zip(sources, agent_names_by_source, strict=True):
        agents = {name: build_loaded_run(source, name) for name in selected}
        provenance.append(
            RunSource(
                rollouts_jsonl_fpath=source.rollouts_jsonl_fpath,
                aggregate_metrics_fpath=source.aggregate_metrics_fpath,
                selected_agents=list(selected),
                repeat_counts_by_agent={
                    name: len(run.repeat_level_metrics) or run.num_repeats for name, run in agents.items()
                },
                scored_repeat_counts_by_agent={
                    name: sum(not is_token_accounting_only(row) for row in run.repeat_level_metrics)
                    for name, run in agents.items()
                },
                task_counts_by_agent={name: run.num_tasks for name, run in agents.items()},
            )
        )
    return sources[0].model_copy(update={"sources": provenance})
