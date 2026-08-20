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
"""Reading `*_aggregate_metrics.json` for `gym compare`, and picking which agent to compare.

All filesystem I/O for the compare feature lives here. The rollouts JSONL a user points at is
never opened: it is the run's identity and the handle its `_aggregate_metrics.json` sibling is
derived from.
"""

import difflib
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, FrozenSet, List, Literal, Optional, Sequence, Tuple

import orjson

from nemo_gym import _resolve_under_cwd_or_install
from nemo_gym.config_types import ConfigError, ConfigPathNotFoundError
from nemo_gym.global_config import AGENT_REF_KEY_NAME, TASK_INDEX_KEY_NAME
from nemo_gym.path_utils import aggregate_metrics_path_for


RunRole = Literal["baseline", "candidate"]

# Prefix of the cross-repeat confidence-interval keys merged into `agent_metrics`. Only written for
# `mean/*` metrics on runs with >= 2 repeats, and only by runs collected after repeat-level metrics
# landed -- so its absence is normal, not an error.
CI_LOW_PREFIX = "ci_low_95_across_repeats/"
CI_HIGH_PREFIX = "ci_high_95_across_repeats/"
SE_PREFIX = "se_across_repeats/"
MEAN_ACROSS_REPEATS_PREFIX = "mean_across_repeats/"
STD_ERR_ACROSS_RUNS_SUFFIX = "/std_err_across_runs"


@dataclass(frozen=True)
class RunFile:
    """A parsed `*_aggregate_metrics.json`, indexed by agent name."""

    role: RunRole
    index: int
    flag_label: str
    rollouts_jsonl_fpath: Path
    aggregate_metrics_fpath: Path
    entries_by_agent: Dict[str, Dict[str, Any]]

    @property
    def agent_names(self) -> List[str]:
        return sorted(self.entries_by_agent)


@dataclass(frozen=True)
class LoadedRun:
    """One side of the comparison, narrowed to a single agent."""

    role: RunRole
    index: int
    label: str
    rollouts_jsonl_fpath: Path
    aggregate_metrics_fpath: Path
    available_agent_names: Tuple[str, ...]
    agent_name: str
    agent_metrics: Dict[str, Any]
    key_metrics: Dict[str, Any]
    group_level_metrics: List[Dict[str, Any]]
    repeat_level_metrics: List[Dict[str, Any]] = field(default_factory=list)
    num_tasks: int = 0
    num_repeats: Optional[int] = None
    task_indices: FrozenSet[int] = frozenset()
    has_rollout_infos: bool = False
    has_repeat_cis: bool = False


@dataclass(frozen=True)
class AgentSelection:
    """Which agent to read from each side for one comparison block."""

    baseline_agent: str
    candidate_agents: Tuple[str, ...]


def resolve_aggregate_metrics_fpath(rollouts_jsonl_fpath: Path, override: Optional[str]) -> Path:
    """The `*_aggregate_metrics.json` to read: the explicit override, else the derived sibling."""
    if override:
        return _resolve_under_cwd_or_install(override)
    return aggregate_metrics_path_for(rollouts_jsonl_fpath)


def _read_agent_entries(metrics_fpath: Path) -> Dict[str, Dict[str, Any]]:
    try:
        raw = metrics_fpath.read_bytes()
    except OSError as e:
        # `exists()` is true for a directory and for a file the user cannot read, so the guard in
        # load_run_file lets both through to here. Convert rather than surfacing a raw traceback.
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


def load_run_file(
    rollouts_jsonl_fpath: str,
    *,
    role: RunRole,
    index: int = 0,
    flag_label: str,
    aggregate_metrics_fpath_override: Optional[str] = None,
) -> RunFile:
    """Resolve and parse one run's `*_aggregate_metrics.json`."""
    rollouts_path = _resolve_under_cwd_or_install(rollouts_jsonl_fpath)
    metrics_path = resolve_aggregate_metrics_fpath(rollouts_path, aggregate_metrics_fpath_override)

    if not metrics_path.exists():
        if not rollouts_path.exists():
            raise ConfigPathNotFoundError(
                f"Run not found: '{rollouts_path}' ({flag_label}). Check the path is spelled correctly."
            )
        raise ConfigPathNotFoundError(
            f"Aggregate metrics not found: '{metrics_path}', derived from '{rollouts_path}' ({flag_label}). "
            "The run may have been collected with --disable-aggregation: run `gym eval aggregate` first, "
            "or point at the metrics file directly with "
            f"+{'baseline_aggregate_metrics_fpath' if role == 'baseline' else 'candidate_aggregate_metrics_fpaths'}."
        )

    return RunFile(
        role=role,
        index=index,
        flag_label=flag_label,
        rollouts_jsonl_fpath=rollouts_path,
        aggregate_metrics_fpath=metrics_path,
        entries_by_agent=_read_agent_entries(metrics_path),
    )


def _sole_agent(run_file: RunFile, flag_hint: str) -> str:
    if len(run_file.entries_by_agent) == 1:
        return next(iter(run_file.entries_by_agent))
    raise ConfigError(
        f"'{run_file.aggregate_metrics_fpath}' ({run_file.flag_label}) contains "
        f"{len(run_file.entries_by_agent)} agents: {', '.join(run_file.agent_names)}. "
        f"Pass {flag_hint} to choose one."
    )


def _require_agent(run_file: RunFile, agent_name: str, flag_label: str) -> None:
    if agent_name not in run_file.entries_by_agent:
        # Same " Did you mean `X`?" shape the CLI uses for unknown component names, inlined so the
        # compare package stays independent of `nemo_gym.cli`.
        close = difflib.get_close_matches(agent_name, run_file.agent_names, n=1)
        raise ConfigError(
            f"Agent '{agent_name}' ({flag_label}) is not in '{run_file.aggregate_metrics_fpath}'. "
            f"Available: {', '.join(run_file.agent_names)}." + (f" Did you mean `{close[0]}`?" if close else "")
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

    Precedence is most-specific-first: a per-side flag beats `--agent`, which beats the default
    full join over agent names common to every run. Returns the selections, any warnings, and the
    agents that were present but not compared (keyed by run label).
    """
    warnings: List[str] = []
    skipped: Dict[str, List[str]] = {}
    per_side_given = baseline_agent_name is not None or candidate_agent_names is not None

    if per_side_given:
        baseline_agent = baseline_agent_name or agent_name or _sole_agent(baseline_file, "--baseline-agent")
        _require_agent(baseline_file, baseline_agent, "--baseline-agent" if baseline_agent_name else "--agent")
        candidates: List[str] = []
        for position, run_file in enumerate(candidate_files):
            explicit = candidate_agent_names[position] if candidate_agent_names else None
            candidate_agent = explicit or agent_name or _sole_agent(run_file, "--candidate-agents")
            _require_agent(run_file, candidate_agent, "--candidate-agents" if explicit else "--agent")
            candidates.append(candidate_agent)
        return [AgentSelection(baseline_agent, tuple(candidates))], warnings, skipped

    if agent_name:
        for run_file in (baseline_file, *candidate_files):
            _require_agent(run_file, agent_name, "--agent")
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
            "Pass --agent NAME to force one agent on both sides, or --baseline-agent / --candidate-agents "
            "to pick a different agent per side."
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
    num_agents_in_file: int,
) -> Optional[int]:
    """Best available repeat count, most broadly-supported source first.

    `expected_num_rollouts` is present on every generation of the file, so it is tried first.
    `repeat_level_metrics` is only usable when the file holds a single agent: aggregation writes a
    hardcoded placeholder agent name there, so entries cannot be attributed to a real agent.
    """
    expected = Counter(
        group["expected_num_rollouts"]
        for group in group_level_metrics
        if isinstance(group.get("expected_num_rollouts"), int)
    )
    if expected:
        return expected.most_common(1)[0][0]
    if repeat_level_metrics and num_agents_in_file == 1:
        return len(repeat_level_metrics)
    rollout_counts = [
        len(group["rollout_infos"]) for group in group_level_metrics if isinstance(group.get("rollout_infos"), list)
    ]
    return max(rollout_counts) if rollout_counts else None


def build_loaded_run(run_file: RunFile, agent_name: str, *, label: str) -> LoadedRun:
    """Narrow a parsed run file to one agent."""
    entry = run_file.entries_by_agent[agent_name]
    agent_metrics = entry.get("agent_metrics") or {}
    group_level_metrics = entry.get("group_level_metrics") or []
    repeat_level_metrics = entry.get("repeat_level_metrics") or []

    return LoadedRun(
        role=run_file.role,
        index=run_file.index,
        label=label,
        rollouts_jsonl_fpath=run_file.rollouts_jsonl_fpath,
        aggregate_metrics_fpath=run_file.aggregate_metrics_fpath,
        available_agent_names=tuple(run_file.agent_names),
        agent_name=agent_name,
        agent_metrics=agent_metrics,
        key_metrics=entry.get("key_metrics") or {},
        group_level_metrics=group_level_metrics,
        repeat_level_metrics=repeat_level_metrics,
        num_tasks=len(group_level_metrics),
        num_repeats=_derive_num_repeats(group_level_metrics, repeat_level_metrics, len(run_file.entries_by_agent)),
        task_indices=frozenset(
            group[TASK_INDEX_KEY_NAME] for group in group_level_metrics if TASK_INDEX_KEY_NAME in group
        ),
        has_rollout_infos=any(isinstance(group.get("rollout_infos"), list) for group in group_level_metrics),
        has_repeat_cis=any(key.startswith(CI_LOW_PREFIX) for key in agent_metrics),
    )


def run_labels(rollouts_paths: Sequence[Path]) -> List[str]:
    """Short, unique display labels for the runs -- their parent directory names where that suffices."""
    labels = [path.parent.name or path.stem for path in rollouts_paths]
    if len(set(labels)) == len(labels):
        return labels
    return [str(Path(path.parent.parent.name) / (path.parent.name or path.stem)) for path in rollouts_paths]
