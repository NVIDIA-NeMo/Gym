# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Connected scenario controller for ``LiveEnv``.

The default env path remains replay/synthetic. This module is imported only
when ``ENV_SCENARIO_MODE=t1_runner`` is enabled, and wraps the existing
``congestion_gen.runner.ScenarioRunner`` lifecycle so /reset starts a real
T1 traffic scenario and /close tears it down.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from .schemas import ToolCall


def _publish_snapshot_router(router: Path, target: Path) -> None:
    """Point the KPI exporter symlink at the active episode snapshot.

    When episodes use per-episode ``work_dir/snapshot.json`` files, the
    long-lived kpi-exporter keeps reading a stable ``RUNNER_SNAPSHOT_PATH``
    while this router is repointed on each /reset.
    """
    router = router.expanduser()
    if not router.is_absolute():
        router = (Path.cwd() / router).resolve()
    else:
        router = router.parent.resolve() / router.name
    target = target.resolve()
    router.parent.mkdir(parents=True, exist_ok=True)
    if router.is_symlink():
        try:
            link_target = Path(os.readlink(router))
            if not link_target.is_absolute():
                link_target = (router.parent / link_target).resolve()
            else:
                link_target = link_target.resolve()
            if link_target == target:
                return
        except OSError:
            pass
    tmp = router.with_name(f".{router.name}.tmp")
    if tmp.exists() or tmp.is_symlink():
        tmp.unlink()
    tmp.symlink_to(str(target))
    if router.exists() or router.is_symlink():
        router.unlink()
    tmp.rename(router)


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_path(name: str) -> Path | None:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return None
    return Path(raw).expanduser()


def _prepare_t2_policy_scenario(
    spec: Any,
    *,
    cell_capacity_mbps: float,
    episode_horizon_s: float,
    traffic_liveness_s: float | None = None,
    max_ues_per_cell: int = 8,
) -> Any:
    """Make T2's crossed load axis real without changing historical T1.

    The generic sampler's regimes use incompatible traffic scales (and some
    encode difficulty in duty cycle, channel events, or QoS mix rather than
    aggregate Mbps).  Normalize the *time-average* configured demand in every
    T2 cell to ``capacity * (0.59 + 0.54*difficulty)``.  Lower-load validation
    cases are therefore below capacity and severe cases are mildly
    oversubscribed. Traffic kinds and fixed-rate GBR/IMS demand remain intact;
    elastic demand gets one explicit 30% hog so the only enabled actuator has
    a reachable fairness/congestion improvement instead of shedding an
    arbitrary equal-rate UE.

    T2 owns eight live UE tunnels per cell, so trim before normalization rather
    than treating dropped phantom UEs as service.  NLOS pulses are rescheduled
    into the policy episode horizon; the sampler otherwise spreads them over
    the long process-liveness duration used by connected launchers.
    """

    difficulty = max(0.0, min(1.0, float(getattr(spec, "difficulty", 0.0))))
    capacity = max(1e-6, float(cell_capacity_mbps))
    target_per_cell = capacity * (0.59 + 0.54 * difficulty)
    horizon = max(8.0, float(episode_horizon_s))
    liveness = max(horizon + 5.0, float(traffic_liveness_s or 0.0))
    by_cell: dict[int, list[Any]] = {}
    for ue in list(getattr(spec, "ues", []) or []):
        by_cell.setdefault(int(getattr(ue, "cell_id", 0)), []).append(ue)
    next_ue_id = 1 + max([int(getattr(ue, "ue_id")) for ue in list(getattr(spec, "ues", []) or [])] or [-1])

    kept_ues: list[Any] = []
    per_cell_meta: dict[str, dict[str, Any]] = {}
    for cell in list(getattr(spec, "cells", []) or []):
        cell_id = int(getattr(cell, "cell_id", 0))
        selected = list(by_cell.get(cell_id, [])[:max_ues_per_cell])
        if not selected:
            raise ValueError(f"T2 cell {cell_id} has no configured UEs")
        original_count = len(selected)
        while len(selected) < max_ues_per_cell:
            template = selected[len(selected) % original_count]
            original_imsi = str(getattr(template, "imsi"))
            suffix_width = min(6, len(original_imsi))
            unique_imsi = original_imsi[:-suffix_width] + f"{next_ue_id:0{suffix_width}d}"[-suffix_width:]
            selected.append(
                template.model_copy(
                    update={
                        "ue_id": next_ue_id,
                        "imsi": unique_imsi,
                        "start_offset_s": float(template.start_offset_s) + 0.001 * len(selected),
                    }
                )
            )
            next_ue_id += 1
        elastic = [ue for ue in selected if int(getattr(ue.traffic, "qos_5qi", 9)) == 9]
        hog_id = int(max(elastic, key=lambda ue: float(ue.traffic.bandwidth_mbps)).ue_id) if elastic else None
        baseline = 0.0
        fixed_qos = 0.0
        original_rates: dict[int, float] = {}
        for ue in selected:
            traffic = getattr(ue, "traffic")
            duty = (
                float(getattr(traffic, "burst_duty", 1.0) or 1.0)
                if getattr(traffic, "kind", "") == "iperf3_burst"
                else 1.0
            )
            service_rate = float(getattr(traffic, "bandwidth_mbps")) * duty
            original_rates[int(getattr(ue, "ue_id"))] = service_rate
            baseline += service_rate
            if int(getattr(traffic, "qos_5qi", 9)) in {1, 5}:
                fixed_qos += service_rate
        if baseline <= 0.0:
            raise ValueError(f"T2 cell {cell_id} has no positive configured demand")
        preserve_fixed_qos = bool(elastic and fixed_qos < target_per_cell)
        desired_rates: dict[int, float] = {}
        if preserve_fixed_qos:
            for ue in selected:
                ue_id = int(getattr(ue, "ue_id"))
                if int(getattr(ue.traffic, "qos_5qi", 9)) in {1, 5}:
                    desired_rates[ue_id] = original_rates[ue_id]
            elastic_budget = target_per_cell - fixed_qos
            if len(elastic) == 1:
                desired_rates[hog_id] = elastic_budget
            else:
                hog_rate = 0.30 * elastic_budget
                desired_rates[hog_id] = hog_rate
                other_rate = (elastic_budget - hog_rate) / (len(elastic) - 1)
                for ue in elastic:
                    ue_id = int(getattr(ue, "ue_id"))
                    if ue_id != hog_id:
                        desired_rates[ue_id] = other_rate
        else:
            scale = target_per_cell / baseline
            desired_rates = {ue_id: rate * scale for ue_id, rate in original_rates.items()}
        for ue in selected:
            traffic = getattr(ue, "traffic")
            ue_id = int(getattr(ue, "ue_id"))
            duty = (
                float(getattr(traffic, "burst_duty", 1.0) or 1.0)
                if getattr(traffic, "kind", "") == "iperf3_burst"
                else 1.0
            )
            normalized_traffic = traffic.model_copy(
                update={
                    "bandwidth_mbps": desired_rates[ue_id] / duty,
                    # Keep the real traffic process alive for the connected
                    # launcher's configured lifetime. Modeled activity advances
                    # on the separate logical policy-step clock.
                    "duration_s": max(float(traffic.duration_s), liveness),
                }
            )
            kept_ues.append(
                ue.model_copy(
                    update={
                        # The PRACH regime is represented by explicit modeled arrival
                        # pressure. Starting every assigned stream at logical t=0
                        # makes the crossed load contract identical across policies.
                        "start_offset_s": 0.0,
                        "traffic": normalized_traffic,
                    }
                )
            )
        per_cell_meta[str(cell_id)] = {
            "n_live_ues": len(selected),
            "n_sampled_ues_before_fill": original_count,
            "baseline_time_average_mbps": baseline,
            "aggregate_normalization_scale": target_per_cell / baseline,
            "fixed_qos_service_mbps": fixed_qos,
            "elastic_hog_ue_id": hog_id,
            "elastic_hog_target_share": 0.30 if len(elastic) > 1 else None,
            "target_time_average_mbps": target_per_cell,
        }

    kept_ids = {int(getattr(ue, "ue_id")) for ue in kept_ues}
    pulse_duration = min(max(3.0, 3.0 + 5.0 * difficulty), horizon / 2.0)
    new_cells: list[Any] = []
    for cell in list(getattr(spec, "cells", []) or []):
        events = [
            event for event in list(getattr(cell, "nlos_events", []) or []) if int(getattr(event, "ue_id")) in kept_ids
        ]
        target_ids = sorted({int(getattr(event, "ue_id")) for event in events})
        schedule = {
            ue_id: min(max(1.0, horizon - pulse_duration - 1.0), 2.0 + 3.0 * idx)
            for idx, ue_id in enumerate(target_ids)
        }
        rescheduled = []
        for event in events:
            start = schedule[int(getattr(event, "ue_id"))]
            event_t = start if getattr(event, "action", "raise") == "raise" else (start + pulse_duration)
            rescheduled.append(event.model_copy(update={"t_s": event_t}))
        new_cells.append(cell.model_copy(update={"nlos_events": rescheduled}))

    manifest_meta = dict(getattr(spec, "manifest_meta", {}) or {})
    manifest_meta["t2_load_contract"] = {
        "version": "t2_time_average_load_v1",
        "target_formula": "cell_capacity_mbps*(0.59+0.54*difficulty)",
        "cell_capacity_mbps": capacity,
        "time_average_burst_load": True,
        "max_ues_per_cell": max_ues_per_cell,
        "episode_horizon_s": horizon,
        "traffic_liveness_s": liveness,
        "clock_mode": "logical_policy_step",
        "per_cell": per_cell_meta,
    }
    return spec.model_copy(
        update={
            "cells": new_cells,
            "ues": kept_ues,
            "manifest_meta": manifest_meta,
        }
    )


@dataclass
class StartedScenario:
    """One started T1 scenario owned by an env episode."""

    runner: Any
    spec: Any
    work_dir: Path
    snapshot_path: Path
    cell_capacity_mbps: float
    snapshot_id: str = ""
    logical_step_s: float | None = None
    close_summary: dict[str, Any] | None = None

    def tick(self) -> int | None:
        if self.logical_step_s is None:
            self.runner.tick()
            return None
        revision = self.runner.advance_logical_time(self.logical_step_s)
        # Publish synchronously. The env subsequently waits until the exporter
        # exposes this exact snapshot id and revision, avoiding a stale writer
        # cycle after a logical step.
        self.runner.publish_snapshot(self.snapshot_path)
        return int(revision)

    @property
    def snapshot_revision(self) -> int | None:
        if self.logical_step_s is None:
            return None
        return int(getattr(self.runner, "_logical_step_revision", 0))

    def observation_time_s(self) -> float | None:
        if self.logical_step_s is None:
            return None
        return float(self.runner.current_scenario_time_s())

    def service_accounting(self) -> dict[str, Any]:
        """Return the runner's current, traffic-window-aware service ledger."""
        snapshot = self.runner.snapshot()
        accounting = snapshot.get("service_accounting", {})
        return dict(accounting) if isinstance(accounting, dict) else {}

    def apply_action(self, action: ToolCall) -> dict[str, Any]:
        notes = "traffic-side simulator control; OAI telnet actuators deferred until --telnetsrv is enabled on gNBs"
        if action.name == "noop":
            return {
                "mode": "traffic_side",
                "tool": action.name,
                "applied": False,
                "reason": "noop",
            }
        if action.name in {"set_qos_weights", "set_scheduler_policy"}:
            return {
                "mode": "log_only",
                "tool": action.name,
                "applied": False,
                "reason": "no_connected_actuator_for_tool",
                "notes": notes,
            }
        if action.name == "set_admission_policy":
            return self._apply_admission_policy(action, notes=notes)
        if action.name == "set_prb_cap":
            return self._apply_prb_cap(action, notes=notes)
        if action.name == "set_mcs_bounds":
            return self._apply_mcs_bounds(action, notes=notes)
        if action.name == "set_ul_power_control":
            return self._apply_ul_power_control(action, notes=notes)
        return {
            "mode": "log_only",
            "tool": action.name,
            "applied": False,
            "reason": "no_connected_actuator_for_tool",
            "notes": notes,
        }

    def _apply_admission_policy(self, action: ToolCall, *, notes: str) -> dict[str, Any]:
        """Apply the connected admission *proxy*.

        The public tool name stays ``set_admission_policy`` for the frozen T1
        contract, but the available runner effect is active-flow preemption,
        not rejection of a new RRC setup.  Surface that distinction in every
        actuator result so downstream metrics cannot mistake the proxy for
        true admission control.
        """
        cell_id = int(action.arguments.get("cell_id", 0))
        threshold = float(action.arguments.get("accept_threshold_pct", 100.0))
        snapshot = self.runner.snapshot()
        load_pct = _cell_load_pct(snapshot, cell_id)
        if load_pct <= threshold:
            return {
                "mode": "traffic_side",
                "tool": action.name,
                "kind": "traffic_side_admission",
                "service_action": "active_flow_preemption",
                "admission_proxy": True,
                "cell_id": cell_id,
                "accept_threshold_pct": threshold,
                "current_load_pct": load_pct,
                "applied": False,
                "reason": "load_below_accept_threshold",
                "notes": notes,
            }
        event = self.runner.suppress_traffic_stream(
            cell_id=cell_id,
            reason=f"set_admission_policy accept_threshold_pct={threshold:.1f}",
        )
        return {
            "mode": "traffic_side",
            "tool": action.name,
            "kind": "traffic_side_admission",
            "service_action": "active_flow_preemption",
            "admission_proxy": True,
            "cell_id": cell_id,
            "accept_threshold_pct": threshold,
            "current_load_pct": load_pct,
            "applied": bool(event.get("applied", False)),
            "ue_id": event.get("ue_id"),
            "scenario_ue_id": event.get("scenario_ue_id"),
            "container": event.get("container"),
            "configured_offered_mbps": event.get("configured_offered_mbps"),
            "qos_5qi": event.get("qos_5qi"),
            "requested_service_mbps": event.get("requested_service_mbps"),
            "admitted_service_mbps_before": event.get("admitted_service_mbps_before"),
            "admitted_service_mbps_after": event.get("admitted_service_mbps_after"),
            "forced_terminated_service_mbps": event.get("forced_terminated_service_mbps"),
            "forced_termination_events": event.get("forced_termination_events", 0),
            "reason": event.get("reject_reason"),
            "runner_event": event,
            "notes": notes,
        }

    def _apply_prb_cap(self, action: ToolCall, *, notes: str) -> dict[str, Any]:
        args = action.arguments or {}
        cell_id = int(args.get("cell_id", 0))
        target = str(args.get("target", "ue"))
        target_id = int(args.get("target_id", 0))
        max_prb = int(args.get("max_prb", 50))
        event = self.runner.apply_prb_cap(
            cell_id=cell_id,
            target=target,
            target_id=target_id,
            max_prb=max_prb,
            reason=f"set_prb_cap max_prb={max_prb}",
        )
        return {
            "mode": "traffic_side",
            "tool": action.name,
            "kind": "traffic_side_prb_cap",
            "cell_id": cell_id,
            "target": target,
            "target_id": target_id,
            "max_prb": max_prb,
            "scale": event.get("scale"),
            "applied": bool(event.get("applied", False)),
            "reason": event.get("reject_reason"),
            "runner_event": event,
            "notes": notes,
        }

    def _apply_mcs_bounds(self, action: ToolCall, *, notes: str) -> dict[str, Any]:
        args = action.arguments or {}
        cell_id = int(args.get("cell_id", 0))
        mcs_min = int(args.get("mcs_min", 0))
        mcs_max = int(args.get("mcs_max", 28))
        target_bler = float(args.get("target_bler", 0.1))
        event = self.runner.apply_mcs_bounds(
            cell_id=cell_id,
            mcs_min=mcs_min,
            mcs_max=mcs_max,
            target_bler=target_bler,
            reason=f"set_mcs_bounds mcs_max={mcs_max}",
        )
        return {
            "mode": "traffic_side",
            "tool": action.name,
            "kind": "traffic_side_mcs_bounds",
            "cell_id": cell_id,
            "mcs_min": mcs_min,
            "mcs_max": mcs_max,
            "target_bler": target_bler,
            "applied": bool(event.get("applied", False)),
            "runner_event": event,
            "notes": notes,
        }

    def _apply_ul_power_control(self, action: ToolCall, *, notes: str) -> dict[str, Any]:
        args = action.arguments or {}
        cell_id = int(args.get("cell_id", 0))
        p0_dbm = float(args.get("p0_dbm", -90.0))
        alpha = float(args.get("alpha", 1.0))
        event = self.runner.apply_ul_power_control(
            cell_id=cell_id,
            p0_dbm=p0_dbm,
            alpha=alpha,
            reason=f"set_ul_power_control p0={p0_dbm}",
        )
        return {
            "mode": "traffic_side",
            "tool": action.name,
            "kind": "traffic_side_ul_power",
            "cell_id": cell_id,
            "p0_dbm": p0_dbm,
            "alpha": alpha,
            "applied": bool(event.get("applied", False)),
            "runner_event": event,
            "notes": notes,
        }

    def close(self) -> dict[str, Any]:
        if self.close_summary is not None:
            return dict(self.close_summary)

        deliveries: list[Any] = []
        per_cell: dict[int, dict[str, float]] = {}
        err = ""
        try:
            try:
                self.runner.wait_for_completion(timeout_s=0.1)
            except Exception as exc:
                err = f"wait_for_completion: {type(exc).__name__}: {exc}"
            try:
                deliveries, per_cell = self.runner.harvest()
            except Exception as exc:
                err = f"{err}; " if err else ""
                err += f"harvest: {type(exc).__name__}: {exc}"
        finally:
            self.runner.teardown()

        self.close_summary = {
            "ok": err == "",
            "work_dir": str(self.work_dir),
            "snapshot_path": str(self.snapshot_path),
            "n_deliveries": len(deliveries),
            "per_cell": {str(k): v for k, v in per_cell.items()},
            "err": err,
        }
        (self.work_dir / "env_close_summary.json").write_text(json.dumps(self.close_summary, indent=2) + "\n")
        return dict(self.close_summary)


def _cell_load_pct(snapshot: dict[str, Any], cell_id: int) -> float:
    raw = snapshot.get("prb_util_est_per_cell", {})
    value: Any = 0.0
    if isinstance(raw, dict):
        value = raw.get(str(cell_id), raw.get(cell_id, 0.0))
    try:
        return max(0.0, min(100.0, 100.0 * float(value)))
    except (TypeError, ValueError):
        return 0.0


@dataclass(frozen=True)
class T1RunnerScenarioController:
    """Config-only factory for ``StartedScenario`` instances."""

    work_root: Path
    snapshot_path: Optional[Path]
    snapshot_period_s: float
    grace_s: float
    cell_capacity_mbps: float
    traffic_target: str
    ext_dn_container: str

    @classmethod
    def from_env(cls) -> "T1RunnerScenarioController":
        return cls(
            work_root=Path(os.environ.get("ENV_SCENARIO_WORK_ROOT", "data/eval/env_connected")).expanduser(),
            snapshot_path=_env_path("ENV_SCENARIO_SNAPSHOT_PATH"),
            snapshot_period_s=_env_float("ENV_SCENARIO_SNAPSHOT_PERIOD_S", 1.0),
            grace_s=_env_float("ENV_SCENARIO_GRACE_S", 10.0),
            # Realized per-cell capacity for the live T1 pool (2 UEs/cell over
            # RFsim), not the sampler's 250 Mbps nominal-population design label.
            # prb_util_est = intended_offered / this; 60 puts d>=0.5 scenarios
            # into genuine congestion. See runner._offered_scale + 2026-06-13
            # capacity-calibration diagnosis.
            cell_capacity_mbps=_env_float("ENV_SCENARIO_CELL_CAPACITY_MBPS", 60.0),
            traffic_target=os.environ.get("ENV_SCENARIO_TRAFFIC_TARGET", "").strip(),
            ext_dn_container=os.environ.get("ENV_SCENARIO_EXT_DN", "openair-ext-dn"),
        )

    def start(
        self,
        *,
        episode_id: str,
        seed: int,
        difficulty: float,
        regime_mix: Optional[dict[str, float]],
        max_steps: int,
        step_dt_s: float,
    ) -> StartedScenario:
        from congestion_gen.cli import T1_CELL_TO_CONTAINERS, T1_SIONNA_ENDPOINTS
        from congestion_gen.materializer import DEFAULT_TRAFFIC_TARGET
        from congestion_gen.runner import RunnerConfig, ScenarioRunner
        from congestion_gen.sampler import CongestionScenarioSampler
        from congestion_gen.validate import validate_scenario

        duration_env = os.environ.get("ENV_SCENARIO_DURATION_S")
        duration_s = (
            float(duration_env)
            if duration_env and duration_env.strip()
            else max(5.0, float(max_steps) * max(1.0, float(step_dt_s)))
        )
        # Sample the FULL designed UE population per cell (PRB-exhaustion peaks
        # at 4 + 0.9*16 ~= 18-20 UEs/cell). The runner's assign_ues later trims
        # this to the live container pool (2 UEs/cell) and runner._offered_scale
        # rescales the kept UEs back up to the cell's *intended* total offered
        # load. Truncating here (the old num_ues_max=4) collapsed intended load
        # to ~=capacity — defeating that rescale, so d=0.9 scenarios never
        # oversubscribed and noop dominated. 2 cells * 20 max keeps both cells'
        # full population. See 2026-06-14 congestion-regime diagnosis.
        sampler = CongestionScenarioSampler(
            seed=seed,
            difficulty=difficulty,
            regime_weights=regime_mix,
            num_cells=2,
            num_ues_max=2 * 20,
            duration_s=duration_s,
        )
        spec = sampler.sample()
        validate_scenario(spec)

        work_dir = (self.work_root / episode_id).resolve()
        work_dir.mkdir(parents=True, exist_ok=True)
        snapshot_path = Path(self.snapshot_path or (work_dir / "snapshot.json")).resolve()
        cfg = RunnerConfig(
            work_dir=work_dir,
            snapshot_path=snapshot_path,
            ext_dn_container=self.ext_dn_container,
            traffic_target=self.traffic_target or DEFAULT_TRAFFIC_TARGET,
            sionna_endpoints=dict(T1_SIONNA_ENDPOINTS),
            cell_capacity_mbps=self.cell_capacity_mbps,
            grace_s=self.grace_s,
            snapshot_period_s=self.snapshot_period_s,
        )

        runner = ScenarioRunner(spec, dict(T1_CELL_TO_CONTAINERS), cfg)
        try:
            runner.materialize()
            runner.prepare_iperf3_servers()
            runner.start_traffic()
            snapshot_path.parent.mkdir(parents=True, exist_ok=True)
            runner.publish_snapshot(snapshot_path)
            runner.start_snapshot_writer()
            router = _env_path("ENV_SCENARIO_SNAPSHOT_ROUTER")
            if router is not None:
                _publish_snapshot_router(router, snapshot_path)
        except Exception:
            runner.teardown()
            raise

        return StartedScenario(
            runner=runner,
            spec=runner.spec,
            work_dir=work_dir,
            snapshot_path=snapshot_path,
            cell_capacity_mbps=self.cell_capacity_mbps,
            snapshot_id=episode_id,
        )


class T2RunnerScenarioController(T1RunnerScenarioController):
    """T2 (M8): 3 cells × 8 UEs. Same lifecycle as the T1 controller but samples
    3 cells and drives traffic via the multi-UE-container path
    (``T2_CELL_TO_CONTAINERS`` + ``T2ScenarioRunner``). The T1 controller is
    untouched; only ``start`` is overridden.
    """

    def start(
        self,
        *,
        episode_id: str,
        seed: int,
        difficulty: float,
        regime_mix: Optional[dict[str, float]],
        max_steps: int,
        step_dt_s: float,
    ) -> StartedScenario:
        from congestion_gen.cli import T2_CELL_TO_CONTAINERS, T2_SIONNA_ENDPOINTS
        from congestion_gen.materializer import DEFAULT_TRAFFIC_TARGET
        from congestion_gen.runner import RunnerConfig, T2ScenarioRunner
        from congestion_gen.sampler import CongestionScenarioSampler
        from congestion_gen.validate import validate_scenario

        duration_env = os.environ.get("ENV_SCENARIO_DURATION_S")
        duration_s = (
            float(duration_env)
            if duration_env and duration_env.strip()
            else max(5.0, float(max_steps) * max(1.0, float(step_dt_s)))
        )
        # Sample enough population for every regime (PRACH peaks at 32/cell),
        # then establish the T2-only, eight-live-UE load contract.
        sampler = CongestionScenarioSampler(
            seed=seed,
            difficulty=difficulty,
            regime_weights=regime_mix,
            num_cells=3,
            num_ues_max=3 * 32,
            duration_s=duration_s,
        )
        spec = _prepare_t2_policy_scenario(
            sampler.sample(),
            cell_capacity_mbps=self.cell_capacity_mbps,
            episode_horizon_s=float(max_steps) * max(1.0, float(step_dt_s)),
            traffic_liveness_s=duration_s,
        )
        validate_scenario(spec)

        work_dir = (self.work_root / episode_id).resolve()
        work_dir.mkdir(parents=True, exist_ok=True)
        snapshot_path = Path(self.snapshot_path or (work_dir / "snapshot.json")).resolve()
        cfg = RunnerConfig(
            work_dir=work_dir,
            snapshot_path=snapshot_path,
            ext_dn_container=self.ext_dn_container,
            traffic_target=self.traffic_target or DEFAULT_TRAFFIC_TARGET,
            sionna_endpoints=dict(T2_SIONNA_ENDPOINTS),
            cell_capacity_mbps=self.cell_capacity_mbps,
            grace_s=self.grace_s,
            snapshot_period_s=self.snapshot_period_s,
            logical_clock=True,
        )

        runner = T2ScenarioRunner(spec, dict(T2_CELL_TO_CONTAINERS), cfg)
        try:
            runner.materialize()
            runner.prepare_iperf3_servers()
            runner.start_traffic()
            snapshot_path.parent.mkdir(parents=True, exist_ok=True)
            runner.publish_snapshot(snapshot_path)
            runner.start_snapshot_writer()
            router = _env_path("ENV_SCENARIO_SNAPSHOT_ROUTER")
            if router is not None:
                _publish_snapshot_router(router, snapshot_path)
        except Exception:
            runner.teardown()
            raise

        return StartedScenario(
            runner=runner,
            spec=runner.spec,
            work_dir=work_dir,
            snapshot_path=snapshot_path,
            cell_capacity_mbps=self.cell_capacity_mbps,
            snapshot_id=episode_id,
            logical_step_s=max(1.0, float(step_dt_s)),
        )


__all__ = [
    "StartedScenario",
    "T1RunnerScenarioController",
    "T2RunnerScenarioController",
]
