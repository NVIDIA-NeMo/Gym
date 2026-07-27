# SPDX-License-Identifier: Apache-2.0
"""Live ``OpenAirCongestionEnv`` against the openair-rl-gym docker stack.

Implements ``docs/PLAN.md`` §4.6 (episode lifecycle) and §4.8 (multi-env pool).

Wire-up:

- ``reset(seed, difficulty, regime_mix, tier)``:
    1. Sample a :class:`congestion_gen.schemas.ScenarioSpec` via the
       ``CongestionScenarioSampler`` (deterministic per ``(seed, difficulty,
       regime_mix, tier)``). The sampler chooses a regime per cell from the
       weighted mix and sets per-UE ``offered_mbps`` + ``qos_5qi``.
    2. Allocate the next free :class:`EpisodeSlot`. Episode-level state
       (history, last observation) lives on the slot.
    3. Wait ``steady_state_s`` (default 1 s; configurable via env var) so the
       kpi-exporter scrape window covers post-reset state.
    4. Scrape the kpi-exporter once and synthesize the first
       :class:`Observation` (see ``_build_observation``).

- ``step(episode_id, action)``:
    1. Validate ``ToolCall`` (Pydantic does this in the request model).
    2. Run :func:`openair_congestion.guardrail.check`. Rejected actions do not
       touch the actuator path, delta-based positive reward terms are
       suppressed, and the next observation carries
       ``agent_aux.last_rejection``.
    3. Apply the action to the actuator path. Default live mode logs accepted
       actions only. ``ENV_SCENARIO_MODE=t1_runner`` starts a congestion-gen
       T1 scenario and gives ``set_admission_policy`` one traffic-side
       actuator: suppress one runner UE traffic stream. This is not FlexRIC RC
       or OAI telnet RAN control.
    4. Sleep ``step_dt_s`` (default 1 s; tests use 0.05). PLAN.md §4.6 calls
       this "advance wall-clock by 1 s".
    5. Scrape kpi-exporter, build observation, compute reward, increment
       ``step_idx``, ``done = step_idx >= meta.max_steps``.

- ``render(episode_id, format)``: returns the latest observation as
  either ``ascii`` (human) or ``json`` (machine).

- ``close(episode_id)``: free the slot.

The default actuator stub is intentional and explicit. Connected T1 mode is
also explicit about provenance: its MVP actuator controls generated traffic,
not RAN scheduler state.
"""

from __future__ import annotations

import logging
import os
import threading
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Optional

from . import SCHEMA_VERSION, kpi_client
from . import guardrail as _guardrail
from . import rewards as _rewards
from .reward_profiles import select_reward_profile
from .schemas import (
    AgentAux,
    CellObservation,
    EpisodeMeta,
    LastActionEcho,
    Observation,
    ToolCall,
    UEObservation,
    kpi_provenance_for_source_mode,
)
from .tools import MAX_CELLS, MAX_UES


LOG = logging.getLogger("openair_congestion.env")

# --- Tunables (also exposed as env vars so /reset and tests can tune them) ---


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError:
        LOG.warning("env var %s=%r not a float; using default %s", name, raw, default)
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError:
        LOG.warning("env var %s=%r not an int; using default %s", name, raw, default)
        return default


# --- Episode state ----------------------------------------------------------


@dataclass
class _ScenarioFingerprint:
    """Slim view of the sampler output we actually need at /step time."""

    n_cells: int = 2
    n_ues_total: int = 4
    tier: str = "T1"
    regime_mix: dict[str, float] = field(default_factory=dict)
    # Immutable configured service demand.  ``offered_mbps`` is retained as
    # the historical, mutable admitted-load proxy because changing it would
    # alter frozen T1 observations and rewards.
    requested_mbps: dict[tuple[int, int], float] = field(default_factory=dict)
    offered_mbps: dict[tuple[int, int], float] = field(default_factory=dict)
    qos_5qi: dict[tuple[int, int], int] = field(default_factory=dict)
    cell_capacity_mbps: float = 60.0


@dataclass
class LiveEpisode:
    episode_id: str
    meta: EpisodeMeta
    fingerprint: _ScenarioFingerprint
    step_idx: int = 0
    last_obs: Optional[Observation] = None
    prev_obs: Optional[Observation] = None
    last_action: Optional[ToolCall] = None
    last_reward: Optional[float] = None
    last_rejection: Optional[str] = None
    history: list[_guardrail.HistoryEntry] = field(default_factory=list)
    forced_terminated_service_mbps: dict[tuple[int, int], float] = field(default_factory=dict)
    forced_termination_events: int = 0
    connected_scenario: Optional[Any] = None
    scenario_close_summary: Optional[dict[str, Any]] = None
    created_at: float = field(default_factory=time.time)
    closed: bool = False
    lock: threading.RLock = field(default_factory=threading.RLock, repr=False)


def _ues_by_cell(fp: _ScenarioFingerprint) -> dict[int, int]:
    out: dict[int, int] = {}
    for cell_id, ue_id in fp.offered_mbps:
        out[cell_id] = max(out.get(cell_id, 0), ue_id + 1)
    return out


def _t2_policy_guardrail(
    episode: LiveEpisode,
    action: ToolCall,
    result: _guardrail.GuardrailResult,
) -> _guardrail.GuardrailResult:
    """Enforce the deliberately narrow, service-safe T2 experiment contract."""

    if not result.accepted or episode.fingerprint.tier.upper() != "T2":
        return result
    if action.name not in {"noop", "set_prb_cap"}:
        return _guardrail.GuardrailResult(
            accepted=False,
            reason=f"T2 policy contract does not enable {action.name}",
        )
    if action.name == "set_prb_cap":
        max_prb = action.arguments.get("max_prb")
        if not isinstance(max_prb, int) or max_prb < 200 or max_prb > 273:
            return _guardrail.GuardrailResult(
                accepted=False,
                reason=(f"T2 service floor requires set_prb_cap.max_prb in [200,273], got {max_prb!r}"),
            )
        if action.arguments.get("target") != "ue":
            return _guardrail.GuardrailResult(
                accepted=False,
                reason="T2 policy contract enables only UE-targeted PRB caps",
            )
    return result


@dataclass
class _EpisodeSlot:
    slot_id: int
    episode_id: Optional[str] = None
    busy: bool = False


# --- Sampler shim ----------------------------------------------------------


def _sample_scenario(
    *,
    seed: int,
    difficulty: float,
    regime_mix: Optional[dict[str, float]],
    tier: str,
) -> _ScenarioFingerprint:
    """Bridge to ``congestion_gen.sampler``; degrades gracefully if the package
    is missing (e.g. in CI without the editable install)."""
    n_cells, n_ues_total = _tier_dims(tier)
    fp = _ScenarioFingerprint(
        n_cells=n_cells,
        n_ues_total=n_ues_total,
        tier=tier,
        regime_mix=regime_mix or {"prb_exhaustion": 0.6, "bursty": 0.4},
    )
    try:
        from congestion_gen.sampler import (  # type: ignore[import-untyped]
            CELL_CAPACITY_MBPS,
            CongestionScenarioSampler,
        )
    except Exception as e:
        LOG.warning(
            "congestion_gen unavailable (%s); using fallback scenario "
            "(n_cells=%d, n_ues=%d). This optional package is not required "
            "for the self-contained replay backend.",
            e,
            n_cells,
            n_ues_total,
        )
        fp.cell_capacity_mbps = 60.0
        normalized_difficulty = max(0.0, min(1.0, float(difficulty)))
        regime = fp.regime_mix
        for cell_id in range(n_cells):
            ues_in_cell = max(1, n_ues_total // max(1, n_cells))
            # A clean NeMo Gym checkout does not ship congestion_gen, so the
            # fallback must be a useful control environment on its own. Keep
            # low difficulty below capacity, but make the checked-in
            # medium/high-difficulty examples genuinely oversubscribed.
            load_ratio = 0.55 + 0.7 * normalized_difficulty
            load_ratio += 0.15 * float(regime.get("prb_exhaustion", 0.0))
            load_ratio += 0.08 * float(regime.get("bursty", 0.0))
            load_ratio += 0.03 * float(regime.get("qos_competition", 0.0))
            offered_per_ue = fp.cell_capacity_mbps * load_ratio / ues_in_cell
            for ue_idx in range(ues_in_cell):
                key = (cell_id, ue_idx)
                # QoS competition needs heterogeneous demand and service
                # classes. Alternating 0.6/1.4 factors preserve the cell total
                # while creating an observable fairness decision.
                qos_weight = float(regime.get("qos_competition", 0.0))
                skew = (0.6 if ue_idx % 2 == 0 else 1.4) if qos_weight > 0.0 else 1.0
                offered = offered_per_ue * (1.0 + qos_weight * (skew - 1.0))
                fp.requested_mbps[key] = offered
                fp.offered_mbps[key] = offered
                fp.qos_5qi[key] = 1 if qos_weight > 0.0 and ue_idx % 2 == 0 else 9
        return fp

    sampler = CongestionScenarioSampler(
        seed=seed,
        difficulty=difficulty,
        regime_weights=regime_mix,
        num_cells=n_cells,
        num_ues_max=max(n_cells, n_ues_total),
    )
    spec = sampler.sample()
    fp.cell_capacity_mbps = float(CELL_CAPACITY_MBPS)
    fp.regime_mix = dict(getattr(spec, "regime_mix", fp.regime_mix))
    # ScenarioSpec.ues is a flat list with .cell_id, .ue_id, and a nested
    # .traffic carrying .bandwidth_mbps and .qos_5qi.
    # Renumber UE ids per cell so they are 0-indexed within each cell, which
    # is what the kpi-exporter labels expose.
    by_cell: dict[int, list] = {}
    for ue_spec in getattr(spec, "ues", []):
        cell_id = int(getattr(ue_spec, "cell_id", 0))
        by_cell.setdefault(cell_id, []).append(ue_spec)
    fp.n_cells = max(fp.n_cells, len(by_cell)) if by_cell else fp.n_cells
    fp.n_ues_total = sum(len(v) for v in by_cell.values()) or fp.n_ues_total
    for cell_id, ues in by_cell.items():
        for local_idx, ue_spec in enumerate(ues):
            traffic = getattr(ue_spec, "traffic", None)
            offered = float(getattr(traffic, "bandwidth_mbps", 1.0)) if traffic else 1.0
            if tier.upper() == "T2" and getattr(traffic, "kind", "") == "iperf3_burst":
                offered *= float(getattr(traffic, "burst_duty", 1.0) or 1.0)
            qos_5qi = int(getattr(traffic, "qos_5qi", 9)) if traffic else 9
            fp.requested_mbps[(cell_id, local_idx)] = offered
            fp.offered_mbps[(cell_id, local_idx)] = offered
            fp.qos_5qi[(cell_id, local_idx)] = qos_5qi
    return fp


def _fingerprint_from_spec(
    spec: Any,
    *,
    tier: str,
    cell_capacity_mbps: float,
) -> _ScenarioFingerprint:
    """Build the env's slim episode fingerprint from a ScenarioSpec-like object."""
    cells = list(getattr(spec, "cells", []) or [])
    ues = list(getattr(spec, "ues", []) or [])
    n_cells = max([int(getattr(c, "cell_id", 0)) + 1 for c in cells] or [0])
    n_cells = max(n_cells, len(cells), 1)
    fp = _ScenarioFingerprint(
        n_cells=n_cells,
        n_ues_total=len(ues),
        tier=tier,
        regime_mix=dict(getattr(spec, "regime_mix", {}) or {}),
        cell_capacity_mbps=float(cell_capacity_mbps),
    )
    by_cell: dict[int, list[Any]] = {}
    for ue_spec in ues:
        cell_id = int(getattr(ue_spec, "cell_id", 0))
        by_cell.setdefault(cell_id, []).append(ue_spec)
    for cell_id, cell_ues in by_cell.items():
        for local_idx, ue_spec in enumerate(cell_ues):
            traffic = getattr(ue_spec, "traffic", None)
            offered = float(getattr(traffic, "bandwidth_mbps", 1.0)) if traffic else 1.0
            if tier.upper() == "T2" and getattr(traffic, "kind", "") == "iperf3_burst":
                offered *= float(getattr(traffic, "burst_duty", 1.0) or 1.0)
            qos_5qi = int(getattr(traffic, "qos_5qi", 9)) if traffic else 9
            fp.requested_mbps[(cell_id, local_idx)] = offered
            fp.offered_mbps[(cell_id, local_idx)] = offered
            fp.qos_5qi[(cell_id, local_idx)] = qos_5qi
    return fp


def _tier_dims(tier: str) -> tuple[int, int]:
    """(n_cells, n_ues_total) per tier.

    T2 is 3 cells × 8 UEs each = 24 UEs total (M8 decision, 2026-06; the live
    ``docker-compose.t2.yaml`` stands up exactly 24 nr-ue containers). The old
    value (3, 8) read the GUIDE "3 × 8" shorthand as a total rather than
    per-cell and is corrected here.
    """
    return {
        "T1": (2, 4),
        "T2": (3, 24),
        "T3": (4, 20),
        "replay": (2, 4),
    }.get(tier, (2, 4))


def _reward_prb_pressure_threshold(episode: LiveEpisode) -> float:
    return select_reward_profile(
        episode.fingerprint.tier,
        connected_t1_runner=episode.connected_scenario is not None,
    ).prb_pressure_threshold


def _reward_cell_capacity_mbps(episode: LiveEpisode) -> float:
    """Return a fixed reward denominator for the episode's capacity contract."""
    raw = os.environ.get("ENV_REWARD_CELL_CAPACITY_MBPS")
    if raw is not None and raw.strip():
        try:
            return max(1.0, float(raw))
        except ValueError:
            pass
    if episode.fingerprint.tier.upper() == "T2":
        # T2 caps mutate admitted/offered demand. Deriving the denominator from
        # that mutable ledger lets the action change its own reward scale.
        return max(1.0, float(episode.fingerprint.cell_capacity_mbps))
    if episode.connected_scenario is not None:
        per_cell: dict[int, float] = defaultdict(float)
        for (cell_id, _ue_id), mbps in episode.fingerprint.offered_mbps.items():
            per_cell[int(cell_id)] += max(0.0, float(mbps))
        if per_cell:
            return max(10.0, max(per_cell.values()) * 1.05)
    return float(episode.fingerprint.cell_capacity_mbps)


def _nonnegative_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return max(0.0, float(default))
    if number != number or number in (float("inf"), float("-inf")):
        return max(0.0, float(default))
    return max(0.0, number)


def _record_active_flow_preemption(
    episode: LiveEpisode,
    actuator_info: dict[str, Any],
) -> None:
    """Record the admission proxy's real service effect without changing T1.

    The runner cannot yet reject a future arrival; an applied
    ``set_admission_policy`` terminates an active generated stream.  Preserve
    the legacy ``traffic_side_admission`` kind and mutable ``offered_mbps``
    behavior, while adding an immutable requested-service ledger and explicit
    forced-termination accounting.
    """
    cell_id = actuator_info.get("cell_id")
    ue_id = actuator_info.get("ue_id")
    if not isinstance(cell_id, int) or not isinstance(ue_id, int):
        return

    key = (cell_id, ue_id)
    admitted_before = _nonnegative_float(
        actuator_info.get("admitted_service_mbps_before"),
        episode.fingerprint.offered_mbps.get(key, 0.0),
    )
    requested = _nonnegative_float(
        actuator_info.get("requested_service_mbps"),
        episode.fingerprint.requested_mbps.get(key, admitted_before),
    )
    forced = _nonnegative_float(
        actuator_info.get("forced_terminated_service_mbps"),
        admitted_before,
    )

    # Older/fake connected controllers may not provide the additive fields.
    # Fill them here so callers always receive honest actuator semantics.
    actuator_info["service_action"] = "active_flow_preemption"
    actuator_info["admission_proxy"] = True
    actuator_info["requested_service_mbps"] = requested
    actuator_info["admitted_service_mbps_before"] = admitted_before
    actuator_info["admitted_service_mbps_after"] = 0.0
    actuator_info["forced_terminated_service_mbps"] = forced
    actuator_info["forced_termination_events"] = 1

    if key not in episode.forced_terminated_service_mbps:
        episode.forced_terminated_service_mbps[key] = forced
        episode.forced_termination_events += 1
    # Frozen behavior: the old observation/reward path zeroed offered demand
    # after suppression.  Keep that path unchanged; requested_mbps remains the
    # immutable counterfactual service request.
    episode.fingerprint.offered_mbps[key] = 0.0


def _record_prb_cap_effect(
    episode: LiveEpisode,
    actuator_info: dict[str, Any],
) -> None:
    """Mirror the runner's absolute cap without compounding T2 observations."""

    cell_id = actuator_info.get("cell_id")
    ue_id = actuator_info.get("target_id")
    scale = actuator_info.get("scale")
    if not (isinstance(cell_id, int) and isinstance(ue_id, int) and isinstance(scale, (int, float))):
        return
    key = (cell_id, ue_id)
    if episode.fingerprint.tier.upper() == "T2":
        base = episode.fingerprint.requested_mbps.get(key, episode.fingerprint.offered_mbps.get(key, 10.0))
    else:
        # Preserve the historical T1 bookkeeping contract.
        base = episode.fingerprint.offered_mbps.get(key, 10.0)
    episode.fingerprint.offered_mbps[key] = float(base) * float(scale)


def _service_accounting(
    episode: LiveEpisode,
    obs: Observation,
    actuator_info: dict[str, Any],
) -> dict[str, Any]:
    runner_accounting: dict[str, Any] = {}
    if episode.connected_scenario is not None and hasattr(episode.connected_scenario, "service_accounting"):
        try:
            candidate = episode.connected_scenario.service_accounting()
            if isinstance(candidate, dict):
                runner_accounting = candidate
        except Exception as exc:  # pragma: no cover - accounting is additive
            LOG.warning("connected service accounting unavailable: %s", exc)

    requested_ledger = (
        episode.fingerprint.requested_mbps if episode.fingerprint.requested_mbps else episode.fingerprint.offered_mbps
    )
    requested_fallback = sum(_nonnegative_float(v) for v in requested_ledger.values())
    admitted_fallback = sum(_nonnegative_float(v) for v in episode.fingerprint.offered_mbps.values())
    delivered_fallback = sum(_nonnegative_float(ue.delivered_mbps) for cell in obs.cells for ue in cell.ues)
    local_cumulative_forced = sum(_nonnegative_float(v) for v in episode.forced_terminated_service_mbps.values())
    cumulative_forced = _nonnegative_float(
        runner_accounting.get("cumulative_forced_terminated_service_mbps"),
        local_cumulative_forced,
    )
    forced_events = max(
        float(episode.forced_termination_events),
        _nonnegative_float(runner_accounting.get("forced_termination_events")),
    )
    requested = _nonnegative_float(runner_accounting.get("requested_service_mbps"), requested_fallback)
    admitted = _nonnegative_float(runner_accounting.get("admitted_service_mbps"), admitted_fallback)
    delivered = _nonnegative_float(runner_accounting.get("delivered_service_mbps"), delivered_fallback)
    forced = _nonnegative_float(
        runner_accounting.get("forced_terminated_service_mbps"),
        cumulative_forced,
    )
    step_forced = 0.0
    step_events = 0.0
    if actuator_info.get("applied") and actuator_info.get("service_action") == "active_flow_preemption":
        step_forced = _nonnegative_float(actuator_info.get("forced_terminated_service_mbps"))
        step_events = _nonnegative_float(actuator_info.get("forced_termination_events"), 1.0)

    return {
        "version": "1.0",
        "requested_service_mbps": requested,
        "admitted_service_mbps": admitted,
        "delivered_service_mbps": delivered,
        "forced_terminated_service_mbps": forced,
        "cumulative_forced_terminated_service_mbps": cumulative_forced,
        "forced_termination_events": forced_events,
        "step_forced_terminated_service_mbps": step_forced,
        "step_forced_termination_events": step_events,
        "unadmitted_service_mbps": max(0.0, requested - admitted),
        "undelivered_admitted_service_mbps": max(0.0, admitted - delivered),
        "per_cell": (
            dict(runner_accounting.get("per_cell", {})) if isinstance(runner_accounting.get("per_cell"), dict) else {}
        ),
    }


# --- Observation builder ---------------------------------------------------


def _jain(values: list[float]) -> float:
    if not values or all(v <= 0.0 for v in values):
        return 1.0
    s = sum(values)
    n = len(values)
    sq = sum(v * v for v in values)
    return float((s * s) / max(1e-9, n * sq))


def _build_observation(
    *,
    snapshot: kpi_client.KpiSnapshot,
    episode: LiveEpisode,
    t_s: float,
) -> Observation:
    fp = episode.fingerprint
    cell_ids = snapshot.cell_ids() or list(range(fp.n_cells))
    cell_ids = [c for c in cell_ids if 0 <= c < MAX_CELLS]
    if not cell_ids:
        cell_ids = list(range(min(MAX_CELLS, max(1, fp.n_cells))))

    cells: list[CellObservation] = []
    for cell_id in cell_ids:
        prb_dl_p50 = float(snapshot.prb_util.get(cell_id, 0.0))
        prb_dl_p50 = max(0.0, min(1.0, prb_dl_p50))
        prb_dl_p99 = max(prb_dl_p50, min(1.0, prb_dl_p50 * 1.15 + 0.02))
        prb_ul_p50 = max(0.0, min(1.0, prb_dl_p50 * 0.4))
        sched_latency_ms_p99 = 5.0 + 20.0 * prb_dl_p99

        snap_n_ues = max(
            0,
            min(MAX_UES, int(snapshot.active_ue_count.get(cell_id, 0))),
        )
        ue_ids_in_cell = snapshot.ues_in_cell(cell_id)
        explicit_zero_ues = cell_id in snapshot.active_ue_count and snap_n_ues == 0
        if explicit_zero_ues:
            ue_ids_in_cell = []
        elif not ue_ids_in_cell:
            # Fallback: derive from fingerprint
            ue_ids_in_cell = sorted(u for (c, u) in fp.offered_mbps.keys() if c == cell_id)
        ue_ids_in_cell = [u for u in ue_ids_in_cell if 0 <= u < MAX_UES]
        if not ue_ids_in_cell and not explicit_zero_ues:
            ue_ids_in_cell = [0]

        prach_collision_rate = 0.0 if snap_n_ues < 8 else min(0.5, 0.01 * (snap_n_ues - 8) ** 2)
        prach_weight = max(
            0.0,
            min(1.0, float(fp.regime_mix.get("prach_storm", 0.0))),
        )
        if prach_weight > 0.0:
            planned_arrivals = int(8 + 24 * float(episode.meta.difficulty))
            planned_pressure = min(0.5, 0.01 * max(0, planned_arrivals - 8) ** 2)
            prach_collision_rate = max(
                prach_collision_rate,
                prach_weight * planned_pressure,
            )

        ues: list[UEObservation] = []
        thru: list[float] = []
        for ue_id in ue_ids_in_cell:
            delivered = float(snapshot.ue_throughput(cell_id, ue_id, default=0.0))
            sinr = float(snapshot.ue_sinr(cell_id, ue_id, default=10.0))
            bler = float(snapshot.ue_bler(cell_id, ue_id, default=0.0))
            offered = float(fp.offered_mbps.get((cell_id, ue_id), max(delivered, 1.0)))
            requested = float(fp.requested_mbps.get((cell_id, ue_id), offered))
            admitted = offered
            active_cap = None
            if fp.tier.upper() == "T2":
                active_cap = (
                    273
                    if requested <= 1e-9
                    else max(
                        0,
                        min(273, int(round(273.0 * admitted / requested))),
                    )
                )
            mcs_mean = max(0.0, min(27.0, (max(sinr, -10.0) + 5.0) * 1.2))
            buffer_occupancy_kb = max(0.0, (offered - delivered) * 50.0)
            pdb_violations = 1 if buffer_occupancy_kb > 500.0 else 0
            qos_5qi = int(fp.qos_5qi.get((cell_id, ue_id), 9))

            ues.append(
                UEObservation.model_validate(
                    {
                        "ue_id": ue_id,
                        "offered_mbps": offered,
                        "requested_mbps": requested,
                        "admitted_mbps": admitted,
                        "prb_cap_max_prb": active_cap,
                        "delivered_mbps": delivered,
                        "bler": max(0.0, min(1.0, bler)),
                        "mcs_mean": mcs_mean,
                        "sinr_db": max(-20.0, min(40.0, sinr)),
                        "buffer_occupancy_kb": buffer_occupancy_kb,
                        "pdb_violations": pdb_violations,
                        "5qi": qos_5qi,
                    }
                )
            )
            thru.append(delivered)

        cells.append(
            CellObservation(
                cell_id=cell_id,
                prb_util_dl_p50=prb_dl_p50,
                prb_util_dl_p99=prb_dl_p99,
                prb_util_ul_p50=prb_ul_p50,
                sched_latency_ms_p99=sched_latency_ms_p99,
                rrc_connected_ues=len(ues),
                prach_collision_rate=prach_collision_rate,
                fairness_jain=_jain(thru),
                sla_violations_last_window=sum(1 for u in ues if u.pdb_violations > 0),
                ues=ues,
            )
        )

    aux = AgentAux(
        last_action=(
            LastActionEcho(name=episode.last_action.name, arguments=episode.last_action.arguments)
            if episode.last_action is not None
            else None
        ),
        last_reward=episode.last_reward,
        last_rejection=episode.last_rejection,
        step_idx=episode.step_idx,
    )

    return Observation.model_validate(
        {
            "schema_version": SCHEMA_VERSION,
            "t_s": float(t_s),
            "episode_id": episode.episode_id,
            "cells": [c.model_dump(by_alias=True) for c in cells],
            "global": {
                "n_cells": len(cells),
                "n_ues_total": sum(len(c.ues) for c in cells),
                "difficulty": episode.meta.difficulty,
                "regime_mix": episode.fingerprint.regime_mix,
                "tier": episode.meta.tier,
            },
            "kpi_source_mode": snapshot.source_mode or "unknown",
            "kpi_provenance": kpi_provenance_for_source_mode(snapshot.source_mode or "unknown"),
            "agent_aux": aux.model_dump(),
        }
    )


# --- LiveEnv ----------------------------------------------------------------


class LiveEnv:
    """Concrete env over the running T1 stack via kpi-exporter scraping.

    Thread-safe via a per-instance lock around the episode registry; /reset
    /step /close all serialise on the lock since the episode dict is mutated.
    The kpi-exporter scrape happens *outside* the lock (it can take 50–200ms
    over the loopback HTTP and must not block other episodes).
    """

    def __init__(
        self,
        *,
        kpi_url: Optional[str] = None,
        pool_size: Optional[int] = None,
        step_dt_s: Optional[float] = None,
        steady_state_s: Optional[float] = None,
        max_steps_default: int = 60,
        scenario_mode: Optional[str] = None,
        scenario_controller_factory: Optional[Any] = None,
    ) -> None:
        self.kpi_url = kpi_url or os.environ.get("KPI_EXPORTER_URL", kpi_client.DEFAULT_URL)
        self.pool_size = pool_size if pool_size is not None else _env_int("ENV_POOL_SIZE", 4)
        self.step_dt_s = step_dt_s if step_dt_s is not None else _env_float("ENV_STEP_DT_S", 1.0)
        self.steady_state_s = steady_state_s if steady_state_s is not None else _env_float("ENV_STEADY_STATE_S", 1.0)
        self.max_steps_default = max_steps_default
        self.scenario_mode = (scenario_mode or os.environ.get("ENV_SCENARIO_MODE", "off")).strip().lower()
        if self.scenario_mode in ("t1_runner", "t2_runner") and self.pool_size != 1:
            LOG.warning(
                "ENV_SCENARIO_MODE=%s is single-episode today; forcing pool_size from %d to 1",
                self.scenario_mode,
                self.pool_size,
            )
            self.pool_size = 1
        self._scenario_controller_factory = scenario_controller_factory

        self._lock = threading.Lock()
        self._slots = [_EpisodeSlot(slot_id=i) for i in range(self.pool_size)]
        self._episodes: dict[str, LiveEpisode] = {}
        self._t0 = time.monotonic()

    # ----- public API -------------------------------------------------------

    def n_episodes_live(self) -> int:
        with self._lock:
            return sum(1 for s in self._slots if s.busy)

    def reset(
        self,
        *,
        seed: int = 0,
        difficulty: float = 0.5,
        regime_mix: Optional[dict[str, float]] = None,
        scenario_id: Optional[str] = None,
        tier: str = "T1",
        max_steps: Optional[int] = None,
    ) -> tuple[Observation, EpisodeMeta]:
        # Allocate slot first so we fail fast if the pool is full.
        with self._lock:
            slot = next((s for s in self._slots if not s.busy), None)
            if slot is None:
                raise RuntimeError(f"env pool exhausted ({self.pool_size} slots all busy)")
            slot.busy = True
            episode_id = f"ep_{uuid.uuid4().hex[:8]}"
            slot.episode_id = episode_id

        connected_scenario: Optional[Any] = None
        try:
            episode_max_steps = int(max_steps if max_steps is not None else self.max_steps_default)
            if self.scenario_mode in ("t1_runner", "t2_runner"):
                controller = self._new_scenario_controller()
                connected_scenario = controller.start(
                    episode_id=episode_id,
                    seed=seed,
                    difficulty=difficulty,
                    regime_mix=regime_mix,
                    max_steps=episode_max_steps,
                    step_dt_s=self.step_dt_s,
                )
                fingerprint = _fingerprint_from_spec(
                    connected_scenario.spec,
                    tier=tier,
                    cell_capacity_mbps=connected_scenario.cell_capacity_mbps,
                )
                if scenario_id is None:
                    scenario_id = f"{self.scenario_mode}:{episode_id}"
            else:
                fingerprint = _sample_scenario(
                    seed=seed,
                    difficulty=difficulty,
                    regime_mix=regime_mix,
                    tier=tier,
                )
            meta = EpisodeMeta(
                episode_id=episode_id,
                seed=seed,
                difficulty=difficulty,
                regime_mix=fingerprint.regime_mix,
                tier=tier,
                scenario_id=scenario_id,
                max_steps=episode_max_steps,
            )
            episode = LiveEpisode(
                episode_id=episode_id,
                meta=meta,
                fingerprint=fingerprint,
                connected_scenario=connected_scenario,
            )

            # Steady-state wait so the next scrape covers post-reset KPIs.
            if self.scenario_mode in ("t1_runner", "t2_runner"):
                snapshot = self._fetch_validated_snapshot(
                    require_active=True,
                    wait_s=max(self.steady_state_s, 8.0),
                    expected_snapshot_id=connected_scenario.snapshot_id,
                    min_snapshot_revision=connected_scenario.snapshot_revision,
                )
            else:
                if self.steady_state_s > 0.0:
                    time.sleep(self.steady_state_s)
                snapshot = kpi_client.fetch(self.kpi_url)
                self._validate_snapshot_mode(snapshot, require_active=True)

            logical_t = connected_scenario.observation_time_s() if connected_scenario is not None else None
            t_s = logical_t if logical_t is not None else time.monotonic() - self._t0
            obs = _build_observation(snapshot=snapshot, episode=episode, t_s=t_s)
            episode.last_obs = obs

            with self._lock:
                self._episodes[episode_id] = episode

            LOG.info(
                "reset episode_id=%s seed=%s diff=%.2f tier=%s mode=%s n_cells=%d n_ues=%d",
                episode_id,
                seed,
                difficulty,
                tier,
                self.scenario_mode,
                obs.global_.n_cells,
                obs.global_.n_ues_total,
            )
            return obs, meta
        except Exception:
            if connected_scenario is not None:
                try:
                    connected_scenario.close()
                except Exception as exc:  # pragma: no cover
                    LOG.warning("connected scenario close during reset failed: %s", exc)
            self._free_slot(episode_id)
            raise

    def step(
        self,
        episode_id: str,
        action: ToolCall,
    ) -> tuple[Observation, float, bool, dict]:
        with self._lock:
            episode = self._episodes.get(episode_id)
        if episode is None:
            raise KeyError(f"unknown episode_id {episode_id!r}")
        with episode.lock:
            if episode.closed:
                raise RuntimeError(f"episode {episode_id!r} is closed")

            # --- guardrail --------------------------------------------------
            gr = _guardrail.check(
                action,
                history=episode.history,
                n_cells=episode.fingerprint.n_cells,
                n_ues=max(1, episode.fingerprint.n_ues_total),
                n_ues_by_cell=_ues_by_cell(episode.fingerprint),
                ue_ids_by_cell=(
                    {cell.cell_id: {ue.ue_id for ue in cell.ues} for cell in episode.last_obs.cells}
                    if episode.last_obs is not None
                    else None
                ),
                now_s=float(episode.step_idx),
            )
            gr = _t2_policy_guardrail(episode, action, gr)
            rejected = not gr.accepted
            actuator_info: dict[str, Any] = {
                "mode": "log_only",
                "tool": action.name,
                "applied": False,
                "reason": "no_connected_scenario",
            }
            history_recorded = False

            if not rejected:
                if episode.connected_scenario is not None:
                    if action.name != "noop":
                        self._fetch_validated_snapshot(
                            require_active=True,
                            wait_s=max(self.steady_state_s, 3.0),
                            expected_snapshot_id=(episode.connected_scenario.snapshot_id),
                            min_snapshot_revision=(episode.connected_scenario.snapshot_revision),
                        )
                    actuator_info = episode.connected_scenario.apply_action(action)
                    if actuator_info.get("applied"):
                        self._record_history(episode, action, t_s=float(episode.step_idx))
                        history_recorded = True
                        kind = actuator_info.get("kind")
                        if kind == "traffic_side_admission":
                            _record_active_flow_preemption(episode, actuator_info)
                        elif kind == "traffic_side_prb_cap":
                            _record_prb_cap_effect(episode, actuator_info)
                LOG.info(
                    "step episode_id=%s step=%d mode=%s action=%s args=%s actuator=%s",
                    episode_id,
                    episode.step_idx,
                    self.scenario_mode,
                    action.name,
                    action.arguments,
                    actuator_info,
                )
            elif episode.connected_scenario is not None:
                actuator_info = {
                    "mode": "traffic_side",
                    "tool": action.name,
                    "applied": False,
                    "reason": "guardrail_rejected",
                }

            # --- pacing ------------------------------------------------------
            if self.step_dt_s > 0.0:
                time.sleep(self.step_dt_s)
            snapshot_revision: int | None = None
            if episode.connected_scenario is not None:
                snapshot_revision = episode.connected_scenario.tick()

            # --- scrape + observation ---------------------------------------
            snapshot = self._fetch_validated_snapshot(
                wait_s=max(self.steady_state_s, 3.0),
                expected_snapshot_id=(
                    episode.connected_scenario.snapshot_id if episode.connected_scenario is not None else None
                ),
                min_snapshot_revision=snapshot_revision,
            )
            logical_t = (
                episode.connected_scenario.observation_time_s() if episode.connected_scenario is not None else None
            )
            t_s = logical_t if logical_t is not None else time.monotonic() - self._t0
            episode.prev_obs = episode.last_obs
            episode.last_action = action
            episode.last_rejection = gr.reason
            new_obs = _build_observation(snapshot=snapshot, episode=episode, t_s=t_s)
            service_accounting = _service_accounting(
                episode,
                new_obs,
                actuator_info,
            )
            reward_profile = select_reward_profile(
                episode.fingerprint.tier,
                connected_t1_runner=episode.connected_scenario is not None,
            )

            # --- reward ------------------------------------------------------
            reward_breakdown = _rewards.compute_breakdown(
                prev_obs=episode.prev_obs,
                curr_obs=new_obs,
                action=action,
                rejected=rejected,
                cell_capacity_mbps=_reward_cell_capacity_mbps(episode),
                prb_pressure_threshold=reward_profile.prb_pressure_threshold,
                service_accounting=service_accounting,
                reward_version=reward_profile.version,
            )
            reward = float(reward_breakdown["total"])

            # --- bookkeeping ------------------------------------------------
            if not rejected and not history_recorded:
                self._record_history(episode, action, t_s=float(episode.step_idx))

            episode.step_idx += 1
            done = episode.step_idx >= episode.meta.max_steps
            aux = AgentAux(
                last_action=LastActionEcho(
                    name=action.name,
                    arguments=action.arguments,
                ),
                last_reward=reward,
                last_rejection=gr.reason,
                step_idx=episode.step_idx,
            )
            new_obs = new_obs.model_copy(update={"agent_aux": aux})
            episode.last_reward = reward
            episode.last_obs = new_obs

            info = {
                "guardrail_accepted": gr.accepted,
                "rejection_reason": gr.reason,
                "step_idx": episode.step_idx,
                "kpi_source": snapshot.source_mode,
                "scenario_mode": self.scenario_mode,
                "actuator": actuator_info,
                "service_accounting": service_accounting,
                "reward_measurements": reward_breakdown["measurements"],
                "reward_terms": reward_breakdown["terms"],
                "reward_version": reward_profile.version,
                # Carry the exact snapshot evidence into evaluation traces.
                # In connected T2 the controller's snapshot id is the active
                # episode id and the revision advances once per logical step.
                # Downstream gates can therefore reject an older env/exporter
                # stack even if it still labels itself ``runner_snapshot``.
                "snapshot_id": snapshot.snapshot_id,
                "snapshot_revision": snapshot.snapshot_revision,
                "snapshot_fresh": snapshot.snapshot_fresh,
            }
            return new_obs, reward, done, info

    def render(self, episode_id: str, *, format: str = "ascii"):
        with self._lock:
            episode = self._episodes.get(episode_id)
        if episode is None or episode.last_obs is None:
            raise KeyError(f"unknown episode_id {episode_id!r}")
        if format == "ascii":
            from . import render as _rd

            return _rd.to_ascii(episode.last_obs)
        if format == "json":
            return episode.last_obs.model_dump(by_alias=True)
        raise ValueError(f"unknown render format {format!r}")

    def close(self, episode_id: str) -> dict:
        episode = self._free_slot(episode_id)
        if episode is None:
            raise KeyError(f"unknown episode_id {episode_id!r}")
        return {
            "ok": True,
            "n_steps": episode.step_idx,
            "scenario": episode.scenario_close_summary,
        }

    def close_all(self) -> list[dict[str, Any]]:
        with self._lock:
            episode_ids = list(self._episodes)
        summaries: list[dict[str, Any]] = []
        for episode_id in episode_ids:
            try:
                summaries.append({"episode_id": episode_id, **self.close(episode_id)})
            except KeyError:
                continue
        return summaries

    # ----- internal --------------------------------------------------------

    def _free_slot(self, episode_id: str) -> Optional[LiveEpisode]:
        slot_to_free: _EpisodeSlot | None = None
        with self._lock:
            ep = self._episodes.pop(episode_id, None)
            for s in self._slots:
                if s.episode_id == episode_id:
                    slot_to_free = s
                    break
        if ep is not None:
            with ep.lock:
                ep.closed = True
                if ep.connected_scenario is not None:
                    try:
                        ep.scenario_close_summary = ep.connected_scenario.close()
                    except Exception as exc:  # pragma: no cover
                        ep.scenario_close_summary = {"ok": False, "err": str(exc)}
                        LOG.warning("connected scenario close failed: %s", exc)
        if slot_to_free is not None:
            with self._lock:
                slot_to_free.busy = False
                slot_to_free.episode_id = None
        return ep

    def _new_scenario_controller(self) -> Any:
        factory = self._scenario_controller_factory
        if factory is not None:
            if hasattr(factory, "start"):
                return factory
            return factory()
        if self.scenario_mode == "t2_runner":
            from .scenario_control import T2RunnerScenarioController

            return T2RunnerScenarioController.from_env()
        from .scenario_control import T1RunnerScenarioController

        return T1RunnerScenarioController.from_env()

    def _record_history(
        self,
        episode: LiveEpisode,
        action: ToolCall,
        *,
        t_s: float,
    ) -> None:
        episode.history.append(_guardrail.HistoryEntry(action=action, t_s=t_s))
        # Keep the history bounded (rate-limit window only needs ~5s).
        if len(episode.history) > 64:
            episode.history = episode.history[-32:]

    def _fetch_validated_snapshot(
        self,
        *,
        require_active: bool = False,
        wait_s: float | None = None,
        expected_snapshot_id: str | None = None,
        min_snapshot_revision: int | None = None,
    ) -> kpi_client.KpiSnapshot:
        """Scrape kpi-exporter; poll in connected modes until validation passes."""
        snapshot = kpi_client.fetch(self.kpi_url)
        if self.scenario_mode not in ("t1_runner", "t2_runner"):
            self._validate_snapshot_mode(snapshot, require_active=require_active)
            return snapshot

        timeout = wait_s if wait_s is not None else max(self.steady_state_s, 3.0)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                self._validate_snapshot_mode(
                    snapshot,
                    require_active=require_active,
                    expected_snapshot_id=expected_snapshot_id,
                    min_snapshot_revision=min_snapshot_revision,
                )
                return snapshot
            except kpi_client.KpiScrapeError:
                # Logical T2 publishes each step synchronously; poll quickly
                # enough that evaluation speed is not dominated by exporter
                # cadence while still avoiding a busy loop.
                time.sleep(0.1)
                snapshot = kpi_client.fetch(self.kpi_url)
        self._validate_snapshot_mode(
            snapshot,
            require_active=require_active,
            expected_snapshot_id=expected_snapshot_id,
            min_snapshot_revision=min_snapshot_revision,
        )
        return snapshot

    def _validate_snapshot_mode(
        self,
        snapshot: kpi_client.KpiSnapshot,
        *,
        require_active: bool = False,
        expected_snapshot_id: str | None = None,
        min_snapshot_revision: int | None = None,
    ) -> None:
        if self.scenario_mode not in ("t1_runner", "t2_runner"):
            return
        if snapshot.source_mode != "runner_snapshot":
            raise RuntimeError(
                f"ENV_SCENARIO_MODE={self.scenario_mode} requires kpi-exporter "
                f"SOURCE_MODE=runner_snapshot; got {snapshot.source_mode!r}"
            )
        if snapshot.snapshot_fresh is not True:
            age = f" age={snapshot.snapshot_age_s:.1f}s" if snapshot.snapshot_age_s is not None else ""
            raise kpi_client.KpiScrapeError(f"runner snapshot is stale or missing{age}")
        if expected_snapshot_id and snapshot.snapshot_id != expected_snapshot_id:
            raise kpi_client.KpiScrapeError(
                "runner snapshot identity has not reached the active episode: "
                f"expected={expected_snapshot_id!r}, actual={snapshot.snapshot_id!r}"
            )
        if min_snapshot_revision is not None and (
            snapshot.snapshot_revision is None or snapshot.snapshot_revision < min_snapshot_revision
        ):
            raise kpi_client.KpiScrapeError(
                "runner snapshot revision has not reached the logical step: "
                f"expected>={min_snapshot_revision}, "
                f"actual={snapshot.snapshot_revision!r}"
            )
        if require_active:
            active_total = sum(max(0, int(v)) for v in snapshot.active_ue_count.values())
            if active_total <= 0:
                raise kpi_client.KpiScrapeError("runner snapshot has no active UEs")


__all__ = [
    "LiveEnv",
    "LiveEpisode",
    "_build_observation",
    "_fingerprint_from_spec",
    "_sample_scenario",
    "_tier_dims",
]
