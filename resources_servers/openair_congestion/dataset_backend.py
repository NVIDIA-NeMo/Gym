# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Dataset-replay backend: serve recorded KPI snapshots through the Backend
contract.

``ReplayBackend`` synthesizes trajectories from seeds; this backend replays a
dataset file instead. Same reset/step/close contract, so switching is
config-only (``backend: dataset_replay``).

The accepted format is JSONL with one nested ``cells[]`` / ``ues[]`` KPI
snapshot per timestep. Missing optional KPI fields are synthesized with the
same heuristics the live env uses (``openair_congestion/env.py``), so a sparse
dataset still yields the stable observation contract used by diagnostics and
conversion tooling.

Actions are pass-through: the data is pre-recorded, so
``step()`` advances a pointer and does not mutate KPIs. The guardrail still
runs (rejected actions earn the same penalty semantics as ReplayEnv) and the
reward is computed over the served observation pair via the unchanged
``rewards.compute_breakdown(prev_obs, curr_obs, action, rejected=...)``.
"""

from __future__ import annotations

import json
import math
import threading
import uuid
from dataclasses import asdict, dataclass, field, replace
from dataclasses import fields as dataclass_fields
from pathlib import Path
from typing import Any, NoReturn, Optional

# Load the backend contract before the colocated domain imports so an
# incomplete checkout fails with the backend's targeted diagnostic.
from resources_servers.openair_congestion.backends import Backend


# isort: split
from openair_congestion import guardrail as _guardrail
from openair_congestion import rewards as _rewards
from openair_congestion.reward_profiles import select_reward_profile
from openair_congestion.schemas import (
    AgentAux,
    EpisodeMeta,
    LastActionEcho,
    Observation,
    ToolCall,
)
from openair_congestion.tools import MAX_CELLS, MAX_UES


# Stamped into step() info["dynamics_mode"] so trainers can tell recorded-data
# rollouts apart from ReplayEnv's synthetic action-effect model.
DATASET_DYNAMICS_MODE = "provided_data_passthrough_v1"

# --- KPI-snapshot rows -> Observation ----------------------------------------
#
# Field-synthesis heuristics below mirror the live env's observation builder
# (openair_congestion/env.py::_build_observation) so a sparse dataset row
# produces the same derived KPIs the env would produce.


def _jain(values: list[float]) -> float:
    """Jain fairness index (same as env.py::_jain)."""
    if not values or all(v <= 0.0 for v in values):
        return 1.0
    s = sum(values)
    n = len(values)
    sq = sum(v * v for v in values)
    return float((s * s) / max(1e-9, n * sq))


def _num(
    raw: dict[str, Any],
    key: str,
    default: float,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    """Read a finite JSON number without coercing or repairing supplied data."""

    value = raw.get(key)
    if value is None:
        parsed = float(default)
    else:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{key} must be a JSON number, got {value!r}")
        parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"{key} must be finite, got {value!r}")
    if minimum is not None and parsed < minimum:
        raise ValueError(f"{key} must be >= {minimum}, got {parsed}")
    if maximum is not None and parsed > maximum:
        raise ValueError(f"{key} must be <= {maximum}, got {parsed}")
    return parsed


def _integer(
    raw: dict[str, Any],
    key: str,
    default: int,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    """Read an integral finite JSON number without accepting bool or strings."""

    parsed = _num(raw, key, float(default))
    if not parsed.is_integer():
        raise ValueError(f"{key} must be an integer, got {parsed}")
    result = int(parsed)
    if minimum is not None and result < minimum:
        raise ValueError(f"{key} must be >= {minimum}, got {result}")
    if maximum is not None and result > maximum:
        raise ValueError(f"{key} must be <= {maximum}, got {result}")
    return result


def _parse_ue(raw: dict[str, Any], ue_idx: int) -> dict[str, Any]:
    """Parse one UE record; synthesize any missing optional field.

    ``delivered_mbps`` is required; everything else falls back to the env's
    defaults (sinr=10.0, bler=0.0) or derivation heuristics.
    """
    if not isinstance(raw, dict):
        raise TypeError(f"dataset UE record #{ue_idx} must be an object, got {type(raw).__name__}")
    if "delivered_mbps" not in raw:
        raise ValueError(
            f"dataset UE record #{ue_idx} is missing required field 'delivered_mbps'; got keys {sorted(raw)}"
        )
    delivered = _num(raw, "delivered_mbps", 0.0, minimum=0.0)
    offered = _num(raw, "offered_mbps", max(delivered, 1.0), minimum=0.0)
    sinr = _num(raw, "sinr_db", 10.0, minimum=-20.0, maximum=40.0)
    bler = _num(raw, "bler", 0.0, minimum=0.0, maximum=1.0)
    # env.py heuristics: mcs from SINR, backlog from offered - delivered,
    # PDB violation when backlog exceeds 500 kB.
    mcs_mean = _num(
        raw,
        "mcs_mean",
        max(0.0, min(27.0, (max(sinr, -10.0) + 5.0) * 1.2)),
        minimum=0.0,
        maximum=27.0,
    )
    buffer_kb = _num(
        raw,
        "buffer_occupancy_kb",
        max(0.0, (offered - delivered) * 50.0),
        minimum=0.0,
    )
    pdb = _integer(raw, "pdb_violations", 1 if buffer_kb > 500.0 else 0, minimum=0)
    return {
        "ue_id": _integer(raw, "ue_id", ue_idx, minimum=0, maximum=MAX_UES - 1),
        "offered_mbps": offered,
        "delivered_mbps": delivered,
        "bler": bler,
        "mcs_mean": mcs_mean,
        "sinr_db": sinr,
        "buffer_occupancy_kb": buffer_kb,
        "pdb_violations": pdb,
        # Both the JSON alias '5qi' and the field name 'qos_5qi' are accepted.
        "5qi": _integer(
            {"5qi": raw.get("5qi", raw.get("qos_5qi"))},
            "5qi",
            9,
            minimum=1,
            maximum=127,
        ),
    }


def _parse_cell(raw: dict[str, Any], cell_idx: int) -> dict[str, Any]:
    """Parse one cell record; synthesize any missing optional field.

    ``prb_util_dl_p50`` and a non-empty ``ues`` list are required; everything
    else falls back to the env's derivation heuristics.
    """
    if not isinstance(raw, dict):
        raise TypeError(f"dataset cell record #{cell_idx} must be an object, got {type(raw).__name__}")
    if "prb_util_dl_p50" not in raw:
        raise ValueError(
            f"dataset cell record #{cell_idx} is missing required field 'prb_util_dl_p50'; got keys {sorted(raw)}"
        )
    ues_raw = raw.get("ues") or []
    if not ues_raw:
        raise ValueError(f"dataset cell record #{cell_idx} has no 'ues' entries")
    ues = [_parse_ue(ue, i) for i, ue in enumerate(ues_raw)]

    p50 = _num(raw, "prb_util_dl_p50", 0.0, minimum=0.0, maximum=1.0)
    # Heuristics mirror env.py exactly (see schemas.KPI_PROVENANCE_V1 notes).
    p99 = _num(
        raw,
        "prb_util_dl_p99",
        max(p50, min(1.0, p50 * 1.15 + 0.02)),
        minimum=0.0,
        maximum=1.0,
    )
    if p99 < p50:
        raise ValueError(f"prb_util_dl_p99 must be >= prb_util_dl_p50 (got {p99} < {p50})")
    ul_p50 = _num(raw, "prb_util_ul_p50", min(1.0, p50 * 0.4), minimum=0.0, maximum=1.0)
    sched_latency = _num(raw, "sched_latency_ms_p99", 5.0 + 20.0 * p99, minimum=0.0)
    n_ues = _integer(raw, "rrc_connected_ues", len(ues), minimum=0, maximum=MAX_UES)
    prach = _num(
        raw,
        "prach_collision_rate",
        0.0 if n_ues < 8 else min(0.5, 0.01 * (n_ues - 8) ** 2),
        minimum=0.0,
        maximum=1.0,
    )
    fairness = _num(
        raw,
        "fairness_jain",
        _jain([u["delivered_mbps"] for u in ues]),
        minimum=0.0,
        maximum=1.0,
    )
    sla = _integer(
        raw,
        "sla_violations_last_window",
        sum(1 for u in ues if u["pdb_violations"] > 0),
        minimum=0,
    )
    return {
        "cell_id": _integer(raw, "cell_id", cell_idx, minimum=0, maximum=MAX_CELLS - 1),
        "prb_util_dl_p50": p50,
        "prb_util_dl_p99": p99,
        "prb_util_ul_p50": ul_p50,
        "sched_latency_ms_p99": sched_latency,
        "rrc_connected_ues": n_ues,
        "prach_collision_rate": prach,
        "fairness_jain": fairness,
        "sla_violations_last_window": sla,
        "ues": ues,
    }


def row_to_observation(
    row: dict[str, Any],
    *,
    step_idx: int,
    episode_id: str,
) -> Observation:
    """Validate one dataset row into a frozen, Pydantic-valid Observation.

    Raises ``ValueError`` or ``TypeError`` with a field-level message; the
    loader wraps either with the row's line number.
    """
    cells_raw = row.get("cells") or []
    if not cells_raw:
        raise ValueError(f"dataset row has no 'cells' entries; got keys {sorted(row)}")
    cells = [_parse_cell(c, i) for i, c in enumerate(cells_raw)]

    global_raw = row.get("global") or {}
    if not isinstance(global_raw, dict):
        raise TypeError(f"dataset row 'global' must be an object, got {type(global_raw).__name__}")
    n_ues_total = sum(len(c["ues"]) for c in cells)
    payload: dict[str, Any] = {
        "t_s": _num(row, "t_s", float(step_idx), minimum=0.0),
        "episode_id": episode_id,
        "cells": cells,
        "global": {
            "n_cells": _integer(global_raw, "n_cells", len(cells), minimum=1, maximum=MAX_CELLS),
            "n_ues_total": _integer(
                global_raw,
                "n_ues_total",
                n_ues_total,
                minimum=0,
                maximum=MAX_UES,
            ),
            "difficulty": _num(global_raw, "difficulty", 0.5, minimum=0.0, maximum=1.0),
            "regime_mix": global_raw.get("regime_mix") or {},
            "tier": global_raw.get("tier", "replay"),
        },
        # Default 'replay' keeps provenance conservative (synthetic). Recorded
        # or measured rows should label kpi_source_mode explicitly.
        "kpi_source_mode": str(row.get("kpi_source_mode", "replay")),
    }
    return Observation.model_validate(payload)


# --- File loading ------------------------------------------------------------


def _reject_json_constant(value: str) -> NoReturn:
    raise ValueError(f"non-finite JSON constant {value!r}")


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _strict_json_loads(raw: str) -> Any:
    """Decode standards-compliant JSON without lossy ambiguity."""

    return json.loads(
        raw,
        parse_constant=_reject_json_constant,
        object_pairs_hook=_reject_duplicate_json_keys,
    )


def _rows_from_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = _strict_json_loads(line)
            except (json.JSONDecodeError, ValueError) as exc:
                raise ValueError(f"{path}:{lineno}: not valid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{lineno}: row must be a JSON object")
            row["_lineno"] = lineno
            rows.append(row)
    return rows


@dataclass(frozen=True)
class EpisodeSource:
    """One recorded episode of validated KPI snapshots."""

    observations: list[Observation]
    recorded_actions: list[Optional[ToolCall]]


def _order_value(path: Path, row: dict[str, Any], key: str) -> float:
    try:
        return _num(row, key, 0.0)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path}:{row.get('_lineno', '?')}: non-numeric {key!r} value {row[key]!r}") from exc


def _sort_by_explicit_step(path: Path, episode_key: str, rows: list[dict[str, Any]]) -> bool:
    """Validate and apply an optional explicit step ordering."""
    has_step = [row.get("step") is not None for row in rows]
    if not any(has_step):
        return False
    if not all(has_step):
        row = next(row for row in rows if row.get("step") is None)
        raise ValueError(
            f"{path}:{row.get('_lineno', '?')} (episode {episode_key!r}): "
            "'step' must be present on every row when used"
        )

    seen: dict[int, int] = {}
    for row in rows:
        value = row["step"]
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(
                f"{path}:{row.get('_lineno', '?')} (episode {episode_key!r}): 'step' must be an integer, got {value!r}"
            )
        if value in seen:
            raise ValueError(
                f"{path}:{row.get('_lineno', '?')} (episode {episode_key!r}): "
                f"duplicate 'step' value {value} (first seen on line {seen[value]})"
            )
        seen[value] = row.get("_lineno", -1)

    rows.sort(key=lambda row: row["step"])
    return True


def load_provided_dataset(path: str | Path) -> dict[str, EpisodeSource]:
    """Load and validate a dataset into per-episode observation trajectories.

    Returns ``{episode_key: EpisodeSource}`` with observations in timestep
    order. Every nested KPI snapshot row is fully validated here so malformed
    data fails at boot with a line-numbered error, never mid-replay.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"dataset file not found: {path}. Point the config's 'dataset_path' "
            "at the dataset JSONL (see README, Dataset Formats)."
        )
    if path.suffix.lower() != ".jsonl":
        raise ValueError(f"unsupported dataset extension {path.suffix!r}; expected nested snapshot JSONL (.jsonl)")
    rows = _rows_from_jsonl(path)

    # Group rows into episodes by 'episode_id' (or 'episode'); a dataset
    # without one becomes a single episode in file order.
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if not isinstance(row.get("cells"), list) or not row["cells"]:
            raise ValueError(
                f"{path}:{row.get('_lineno', '?')}: expected nested KPI snapshot "
                "row with a non-empty 'cells' array; GRPO trace rows are not supported"
            )
        key = str(row.get("episode_id") or row.get("episode") or "episode_0")
        grouped.setdefault(key, []).append(row)

    episodes: dict[str, EpisodeSource] = {}
    for key, group in grouped.items():
        # Order within an episode: explicit 'step' field wins, then 't_s',
        # then file order (stable sort keeps ties in file order).
        if not _sort_by_explicit_step(path, key, group) and all(r.get("t_s") is not None for r in group):
            group.sort(key=lambda r: _order_value(path, r, "t_s"))
        obs_list: list[Observation] = []
        recorded_actions: list[Optional[ToolCall]] = []
        for step_idx, row in enumerate(group):
            try:
                # Placeholder id, re-stamped at reset() via model_copy.
                # key[:56] keeps 'src_' + key within the schema's episode_id
                # max_length=64 for long run names.
                observation = row_to_observation(row, step_idx=step_idx, episode_id=f"src_{key[:56]}")
                obs_list.append(observation)
                raw_recorded_action = row.get("recorded_action")
                recorded_actions.append(
                    None if raw_recorded_action is None else ToolCall.model_validate(raw_recorded_action)
                )
            # ValueError covers pydantic ValidationError (a subclass) and
            # float('bad'); TypeError covers structurally wrong scalar types
            # like "delivered_mbps": [1, 2] or "t_s": {} hitting float().
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"{path}:{row.get('_lineno', '?')} (episode {key!r}, step {step_idx}): {exc}"
                ) from exc
        if len(obs_list) < 2:
            raise ValueError(
                f"episode {key!r} has only {len(obs_list)} row(s); need >= 2 "
                "observations per episode (each step consumes an obs pair)"
            )
        episodes[key] = EpisodeSource(
            observations=obs_list,
            recorded_actions=recorded_actions,
        )
    if not episodes:
        raise ValueError(f"dataset file {path} contains no rows")
    return episodes


# --- The backend --------------------------------------------------------------


@dataclass
class DatasetEpisode:
    """One live replay over a recorded trajectory (internal bookkeeping)."""

    episode_id: str
    meta: EpisodeMeta
    trajectory: list[Observation]
    recorded_actions: list[Optional[ToolCall]]
    step_idx: int = 0
    closed: bool = False
    history: list[Any] = field(default_factory=list)  # guardrail.HistoryEntry
    lock: threading.RLock = field(default_factory=threading.RLock)


class DatasetReplayBackend(Backend):
    """Replay recorded observations through the Backend contract.

    Offline and deterministic, like ReplayBackend, but the trajectory comes
    from an ingested dataset file instead of seed-driven synthesis. Actions
    never mutate the KPIs (the data is pre-recorded); they still pass the
    guardrail and still earn the standard reward via
    ``rewards.compute_breakdown`` over the served (prev_obs, curr_obs) pair.
    """

    def __init__(
        self,
        *,
        dataset_path: str = "data/fixtures/sample_provided.jsonl",
        pool_size: int = 32,
        max_steps_default: int = 60,
        cell_capacity_mbps: float = 60.0,
        reward_weights: Optional[dict[str, float]] = None,
    ) -> None:
        """
        Args:
            dataset_path: Nested KPI snapshot JSONL file. Loaded and validated
                eagerly so bad data fails at server boot.
            pool_size: Max concurrent live episodes (same semantics as
                ReplayBackend's pool).
            max_steps_default: Step budget for task rows lacking max_steps;
                always clamped to the recorded trajectory's length.
            cell_capacity_mbps: Normalizer for the reward's throughput-delta
                term. ReplayEnv gets this from its scenario fingerprint; a
                recorded dataset has no fingerprint, so it is a config knob
                (compute_breakdown's own default is 60.0).
            reward_weights: Per-field overrides on rewards.DEFAULT_WEIGHTS.
                Must match the profile the dataset was recorded under, or
                recomputed rewards drift from the recorded ones.
        """
        self.dataset_path = Path(dataset_path)
        self.pool_size = self._positive_integer("pool_size", pool_size)
        self.max_steps_default = self._positive_integer("max_steps_default", max_steps_default)
        self.cell_capacity_mbps = self._finite_number("cell_capacity_mbps", cell_capacity_mbps, positive=True)
        self.reward_weights = self._validated_reward_weights(reward_weights)

        # episode_key -> validated source trajectory (shared read-only across
        # episodes; per-episode copies get their own episode_id stamps).
        self._sources: dict[str, EpisodeSource] = load_provided_dataset(self.dataset_path)
        self._keys: list[str] = sorted(self._sources)

        self._lock = threading.Lock()
        self._episodes: dict[str, DatasetEpisode] = {}

    @staticmethod
    def _positive_integer(name: str, value: Any) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be a positive integer")
        if value < 1:
            raise ValueError(f"{name} must be a positive integer")
        return value

    @staticmethod
    def _finite_number(name: str, value: Any, *, positive: bool = False) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be a finite number")
        parsed = float(value)
        if not math.isfinite(parsed):
            raise ValueError(f"{name} must be a finite number")
        if positive and parsed <= 0.0:
            raise ValueError(f"{name} must be greater than zero")
        return parsed

    @classmethod
    def _validated_reward_weights(cls, reward_weights: Optional[dict[str, float]]) -> Any:
        if not reward_weights:
            return _rewards.DEFAULT_WEIGHTS
        known = {item.name for item in dataclass_fields(_rewards.DEFAULT_WEIGHTS)}
        validated: dict[str, float] = {}
        for name, value in reward_weights.items():
            if name not in known:
                raise ValueError(f"reward_weights contains unknown field {name!r}")
            try:
                validated[name] = cls._finite_number(f"reward_weights.{name}", value)
            except (TypeError, ValueError) as exc:
                raise type(exc)(f"reward_weights.{name}: {exc}") from exc
        return replace(_rewards.DEFAULT_WEIGHTS, **validated)

    # --- episode selection ----------------------------------------------------

    def _select_key(self, task_params: dict[str, Any]) -> str:
        """Map task_params onto one recorded episode.

        An explicit 'scenario_id' must match a dataset episode key exactly;
        otherwise the seed picks deterministically: keys[seed % n].
        """
        scenario_id = task_params.get("scenario_id")
        if scenario_id is not None:
            key = str(scenario_id)
            if key not in self._sources:
                raise KeyError(f"scenario_id {key!r} not in dataset; available: {self._keys}")
            return key
        seed = int(task_params.get("seed", 0))
        return self._keys[seed % len(self._keys)]

    # --- Backend contract -------------------------------------------------------

    def reset(
        self, task_params: dict[str, Any], *, live_episode_ids: Optional[set[str]] = None
    ) -> tuple[Observation, EpisodeMeta]:
        # Selection is read-only over immutable source data; safe outside
        # the lock (and lets a bad scenario_id fail before touching the pool).
        key = self._select_key(task_params)
        source = self._sources[key]

        # Never hold the pool lock while waiting for an episode lock. Reaping
        # goes through close(), so an orphan cannot be removed while a step is
        # still committing its reward and observation.
        while True:
            with self._lock:
                if len(self._episodes) < self.pool_size:
                    episode_id = f"ds_{uuid.uuid4().hex[:12]}"

                    # Re-stamp observations with the real episode id (frozen
                    # models: model_copy(update=...), same pattern ReplayEnv
                    # uses at reset).
                    trajectory = [obs.model_copy(update={"episode_id": episode_id}) for obs in source.observations]

                    # A trajectory of N observations supports N-1
                    # (prev, curr) steps.
                    budget = int(task_params.get("max_steps") or self.max_steps_default)
                    max_steps = max(1, min(len(trajectory) - 1, budget))

                    first_obs = trajectory[0]
                    meta = EpisodeMeta(
                        episode_id=episode_id,
                        seed=int(task_params.get("seed", 0)),
                        difficulty=first_obs.global_.difficulty,
                        regime_mix=first_obs.global_.regime_mix,
                        tier=first_obs.global_.tier,
                        scenario_id=key,
                        max_steps=max_steps,
                    )
                    episode = DatasetEpisode(
                        episode_id=episode_id,
                        meta=meta,
                        trajectory=trajectory,
                        recorded_actions=list(source.recorded_actions),
                    )
                    self._episodes[episode_id] = episode
                    return first_obs, meta

                live = live_episode_ids or set()
                leaked = [episode_id for episode_id in self._episodes if episode_id not in live]

            if not leaked:
                raise RuntimeError(
                    f"dataset episode pool exhausted ({self.pool_size} live); close episodes or raise pool_size"
                )
            for episode_id in leaked:
                try:
                    self.close(episode_id)
                except KeyError:
                    pass  # another closer won the race

    def step(self, episode_id: str, tool_call: ToolCall) -> tuple[Observation, float, bool, dict[str, Any]]:
        with self._lock:
            episode = self._episodes.get(episode_id)
        if episode is None:
            raise KeyError(f"unknown episode_id {episode_id!r}")
        with episode.lock:
            if episode.closed:
                raise RuntimeError(f"episode {episode_id!r} is closed")

            prev_obs = episode.trajectory[episode.step_idx]
            logical_now_s = float(episode.step_idx)

            # Same guardrail as ReplayEnv.step, fed from the observation
            # itself (a recorded dataset has no scenario fingerprint).
            gr = _guardrail.check(
                tool_call,
                history=episode.history,
                n_cells=max(1, prev_obs.global_.n_cells),
                n_ues=max(1, prev_obs.global_.n_ues_total),
                ue_ids_by_cell={cell.cell_id: {ue.ue_id for ue in cell.ues} for cell in prev_obs.cells},
                now_s=logical_now_s,
            )
            rejected = not gr.accepted

            # Pass-through dynamics: the next observation is the recorded
            # data, unmodified by the action.
            next_idx = min(episode.step_idx + 1, len(episode.trajectory) - 1)
            new_obs = episode.trajectory[next_idx]
            recorded_action = episode.recorded_actions[episode.step_idx]

            reward_profile = select_reward_profile(episode.meta.tier)
            reward_breakdown = _rewards.compute_breakdown(
                prev_obs=prev_obs,
                curr_obs=new_obs,
                action=tool_call,
                rejected=rejected,
                cell_capacity_mbps=self.cell_capacity_mbps,
                weights=self.reward_weights,
                prb_pressure_threshold=reward_profile.prb_pressure_threshold,
            )
            reward = float(reward_breakdown["total"])

            next_history = list(episode.history)
            if not rejected:
                next_history.append(_guardrail.HistoryEntry(action=tool_call, t_s=logical_now_s))
                if len(next_history) > 64:
                    next_history = next_history[-32:]

            # Stamp agent_aux so diagnostic consumers see the same observation
            # shape as the other backends. Schema compatibility does not make
            # prerecorded transitions causal or training-usable.
            aux = AgentAux(
                last_action=LastActionEcho(name=tool_call.name, arguments=tool_call.arguments),
                last_reward=reward,
                last_rejection=gr.reason,
                step_idx=next_idx,
            )
            new_obs = new_obs.model_copy(update={"agent_aux": aux})

            done = next_idx >= episode.meta.max_steps
            info = {
                "guardrail_accepted": gr.accepted,
                "rejection_reason": gr.reason,
                "step_idx": next_idx,
                "kpi_source": "dataset_replay",
                "reward_measurements": reward_breakdown["measurements"],
                "reward_terms": reward_breakdown["terms"],
                "reward_version": reward_profile.version,
                "recorded_action": (None if recorded_action is None else recorded_action.model_dump()),
                **self.reward_contract(episode.meta.tier),
                **self.capabilities(),
            }
            assert math.isfinite(reward), "reward must be finite"

            # Commit only after reward, auxiliary metadata, and response
            # construction all succeed.
            episode.step_idx = next_idx
            episode.history = next_history
            episode.trajectory[next_idx] = new_obs  # idempotent on re-read
            return new_obs, reward, done, info

    def close(self, episode_id: str) -> dict[str, Any]:
        with self._lock:
            episode = self._episodes.get(episode_id)
        if episode is None:
            raise KeyError(f"unknown episode_id {episode_id!r}")
        with episode.lock:
            if episode.closed:
                raise KeyError(f"unknown episode_id {episode_id!r}")
            episode.closed = True
            summary = {"ok": True, "n_steps": episode.step_idx}
            with self._lock:
                if self._episodes.get(episode_id) is episode:
                    self._episodes.pop(episode_id)
        return summary

    def capabilities(self) -> dict[str, Any]:
        """Recorded transitions are diagnostics, not causal policy training."""

        return {
            "backend": "dataset_replay",
            "dynamics_mode": DATASET_DYNAMICS_MODE,
            "action_affects_observation": False,
            "causal_action_effects": False,
            "training_usable": False,
            "diagnostic_only": True,
        }

    def reward_contract(self, tier: str) -> dict[str, Any]:
        contract = super().reward_contract(tier)
        contract["reward_weights"] = asdict(self.reward_weights)
        return contract


__all__ = [
    "DATASET_DYNAMICS_MODE",
    "DatasetEpisode",
    "DatasetReplayBackend",
    "EpisodeSource",
    "load_provided_dataset",
    "row_to_observation",
]
