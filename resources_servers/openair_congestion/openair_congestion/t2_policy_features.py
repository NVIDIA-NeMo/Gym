# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Observable, deterministic decision features for the connected T2 policy.

The connected T2 expert historically compared full-precision Python floats
while the learned policy received values rounded to two decimal places.  In
particular, every severe cell is normalized to the same 64.560 Mbps demand,
so insignificant binary-summation differences could select a target that was
not recoverable from the rendered observation.

This module is the single source of truth for both the expert and the compact
T2 renderer.  Decision quantities are quantized before comparisons, and the
renderer prints those exact integers.  No action is masked or rewritten: the
helper only describes and selects the rule-based expert's label.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from decimal import ROUND_HALF_EVEN, Decimal, InvalidOperation

from .schemas import CellObservation, Observation, ToolCall, UEObservation
from .tools import MAX_UES


T2_CELL_CAPACITY_MILLI_MBPS = 60_000
T2_EXPERT_CADENCE_STEPS = 3
T2_MAX_PRB = 273
T2_MIN_PRB = 200

_ONE = Decimal("1")


def _quantize_scaled(value: float, scale: int) -> int:
    """Return a deterministic integer representation of a finite scalar."""

    return int(
        (Decimal(str(float(value))) * scale).quantize(
            _ONE,
            rounding=ROUND_HALF_EVEN,
        )
    )


def quantize_milli_mbps(value: float) -> int:
    """Quantize Mbps to the 0.001-Mbps precision exposed to the policy."""

    return _quantize_scaled(value, 1_000)


def quantize_deci_kb(value: float) -> int:
    """Quantize buffer occupancy to the 0.1-kB policy precision."""

    return _quantize_scaled(value, 10)


def format_milli_mbps(value: int) -> str:
    """Render an integer milli-Mbps value without returning to float math."""

    sign = "-" if value < 0 else ""
    magnitude = abs(value)
    return f"{sign}{magnitude // 1_000}.{magnitude % 1_000:03d}"


def format_deci_kb(value: int) -> str:
    """Render integer deci-kB without returning to float arithmetic."""

    sign = "-" if value < 0 else ""
    magnitude = abs(value)
    return f"{sign}{magnitude // 10}.{magnitude % 10}"


def _requested_mbps(ue: UEObservation) -> float:
    return float(ue.requested_mbps if ue.requested_mbps is not None else ue.offered_mbps)


def _admitted_mbps(ue: UEObservation) -> float:
    return float(ue.admitted_mbps if ue.admitted_mbps is not None else ue.offered_mbps)


def _jain(values: list[int]) -> float:
    if not values or sum(values) <= 0:
        return 1.0
    total = sum(values)
    squares = sum(value * value for value in values)
    return (total * total) / max(1, len(values) * squares)


def _fairness_gain_ppm(
    cell: CellObservation,
    target_id: int,
    max_prb: int,
    current_max_prb: int,
) -> int:
    delivered = [max(0, quantize_milli_mbps(float(ue.delivered_mbps))) for ue in cell.ues]
    if not delivered or sum(delivered) <= 0:
        return 0
    after = list(delivered)
    for index, ue in enumerate(cell.ues):
        if ue.ue_id == target_id:
            # Delivered telemetry already includes the current cap.  Project a
            # tightening relative to that setpoint rather than applying the
            # new absolute scale to an already-scaled value a second time.
            after[index] = after[index] * max_prb // max(1, current_max_prb)
            break
    return _quantize_scaled(_jain(after) - _jain(delivered), 1_000_000)


@dataclass(frozen=True)
class T2CellDecisionFeatures:
    """Decision-sufficient, policy-visible features for one T2 cell."""

    cell_id: int
    requested_total_milli_mbps: int
    admitted_total_milli_mbps: int
    overload_milli_mbps: int
    active_caps: tuple[tuple[int, int], ...]
    release_ue_id: int | None
    candidate_ue_id: int | None
    candidate_max_prb: int | None
    candidate_fairness_gain_ppm: int | None

    @property
    def has_candidate(self) -> bool:
        return (
            self.candidate_ue_id is not None
            and self.candidate_max_prb is not None
            and self.candidate_fairness_gain_ppm is not None
        )


@dataclass(frozen=True)
class T2DecisionFeatures:
    """Complete current-observation state used by the connected T2 expert."""

    capacity_milli_mbps: int
    expert_cadence_wait_steps: int
    cells: tuple[T2CellDecisionFeatures, ...]

    @property
    def expert_ready(self) -> bool:
        return self.expert_cadence_wait_steps == 0


@dataclass(frozen=True)
class T2PolicyTextDerivation:
    """Strict T/C/U reconstruction and independently derived policy contract."""

    observation: Observation
    features: T2DecisionFeatures
    policy_row: str
    cell_rows: tuple[str, ...]
    history: "T2PolicyHistory | None"
    action: ToolCall


@dataclass(frozen=True)
class T2PolicyHistory:
    """Canonical model-visible echo of the immediately preceding decision."""

    action: ToolCall
    rejection: str


def format_t2_policy_row(features: T2DecisionFeatures) -> str:
    """Render the global T2 decision-state row consumed by the policy."""

    return (
        f"P|{format_milli_mbps(features.capacity_milli_mbps)}|"
        f"{features.expert_cadence_wait_steps}|{int(features.expert_ready)}"
    )


def format_t2_cell_row(cell: T2CellDecisionFeatures) -> str:
    """Render one exact per-cell T2 decision-feature row."""

    active_caps = ",".join(f"{ue_id}:{max_prb}" for ue_id, max_prb in cell.active_caps) or "none"
    candidate_ue = cell.candidate_ue_id if cell.candidate_ue_id is not None else "-"
    candidate_max_prb = cell.candidate_max_prb if cell.candidate_max_prb is not None else "-"
    fairness_gain = cell.candidate_fairness_gain_ppm if cell.candidate_fairness_gain_ppm is not None else "-"
    release_ue = cell.release_ue_id if cell.release_ue_id is not None else "-"
    return (
        f"D|{cell.cell_id}|"
        f"{format_milli_mbps(cell.requested_total_milli_mbps)}|"
        f"{format_milli_mbps(cell.admitted_total_milli_mbps)}|"
        f"{format_milli_mbps(cell.overload_milli_mbps)}|"
        f"{candidate_ue}|{candidate_max_prb}|{fairness_gain}|"
        f"{active_caps}|{release_ue}"
    )


def _candidate_for_cell(
    cell: CellObservation,
    *,
    admitted_total_milli_mbps: int,
    capacity_milli_mbps: int,
) -> tuple[int, int, int] | None:
    """Return the highest-ranked actionable ``(ue, max_prb, gain_ppm)``."""

    ranked = sorted(
        cell.ues,
        key=lambda ue: (
            quantize_milli_mbps(float(ue.delivered_mbps)),
            quantize_deci_kb(float(ue.buffer_occupancy_kb)),
            -ue.ue_id,
        ),
        reverse=True,
    )
    for target in ranked:
        requested = quantize_milli_mbps(_requested_mbps(target))
        admitted = quantize_milli_mbps(_admitted_mbps(target))
        if requested <= 0:
            continue
        other_admitted = admitted_total_milli_mbps - admitted
        available = max(0, capacity_milli_mbps - other_admitted)
        max_prb = T2_MAX_PRB * available // requested
        max_prb = max(T2_MIN_PRB, min(T2_MAX_PRB, max_prb))
        if max_prb >= T2_MAX_PRB:
            continue
        current = int(target.prb_cap_max_prb) if target.prb_cap_max_prb is not None else T2_MAX_PRB
        if max_prb >= current:
            # Under overload, an existing cap may only be tightened.  Equal
            # values are ineffective and larger values silently relax the
            # intervention; demand recovery uses the explicit release path.
            continue
        return (
            target.ue_id,
            max_prb,
            _fairness_gain_ppm(cell, target.ue_id, max_prb, current),
        )
    return None


def build_t2_decision_features(
    obs: Observation,
    *,
    capacity_milli_mbps: int = T2_CELL_CAPACITY_MILLI_MBPS,
) -> T2DecisionFeatures:
    """Build the exact state shared by T2 rendering and expert labeling."""

    if obs.global_.tier.upper() != "T2":
        raise ValueError("T2 decision features require a T2 observation")
    if not isinstance(capacity_milli_mbps, int) or isinstance(capacity_milli_mbps, bool) or capacity_milli_mbps <= 0:
        raise ValueError("T2 capacity_milli_mbps must be a positive integer")

    cells: list[T2CellDecisionFeatures] = []
    for cell in sorted(obs.cells, key=lambda item: item.cell_id):
        # Aggregate the exact integers shown on U rows.  This makes every D
        # total reproducible from policy-visible values, including half-even
        # rounding boundaries on individual UEs.
        requested_total = sum(quantize_milli_mbps(_requested_mbps(ue)) for ue in cell.ues)
        admitted_total = sum(quantize_milli_mbps(_admitted_mbps(ue)) for ue in cell.ues)
        overload = max(0, admitted_total - capacity_milli_mbps)
        active_caps = tuple(
            sorted(
                (
                    (ue.ue_id, int(ue.prb_cap_max_prb))
                    for ue in cell.ues
                    if (ue.prb_cap_max_prb is not None and ue.prb_cap_max_prb < T2_MAX_PRB)
                ),
                key=lambda item: item[0],
            )
        )
        release_ue = active_caps[0][0] if requested_total <= capacity_milli_mbps and active_caps else None
        candidate = (
            _candidate_for_cell(
                cell,
                admitted_total_milli_mbps=admitted_total,
                capacity_milli_mbps=capacity_milli_mbps,
            )
            if overload > 0
            else None
        )
        cells.append(
            T2CellDecisionFeatures(
                cell_id=cell.cell_id,
                requested_total_milli_mbps=requested_total,
                admitted_total_milli_mbps=admitted_total,
                overload_milli_mbps=overload,
                active_caps=active_caps,
                release_ue_id=release_ue,
                candidate_ue_id=candidate[0] if candidate else None,
                candidate_max_prb=candidate[1] if candidate else None,
                candidate_fairness_gain_ppm=candidate[2] if candidate else None,
            )
        )

    step_idx = int(obs.agent_aux.step_idx)
    wait_steps = (T2_EXPERT_CADENCE_STEPS - step_idx % T2_EXPERT_CADENCE_STEPS) % (T2_EXPERT_CADENCE_STEPS)
    return T2DecisionFeatures(
        capacity_milli_mbps=capacity_milli_mbps,
        expert_cadence_wait_steps=wait_steps,
        cells=tuple(cells),
    )


def select_t2_expert_action(features: T2DecisionFeatures) -> ToolCall:
    """Select the connected T2 expert label from rendered decision features."""

    if not features.expert_ready:
        return ToolCall(name="noop", arguments={})

    release = next(
        (cell for cell in features.cells if cell.release_ue_id is not None),
        None,
    )
    if release is not None:
        return ToolCall(
            name="set_prb_cap",
            arguments={
                "cell_id": release.cell_id,
                "target": "ue",
                "target_id": release.release_ue_id,
                "max_prb": T2_MAX_PRB,
            },
        )

    candidates = [cell for cell in features.cells if cell.has_candidate]
    if not candidates:
        return ToolCall(name="noop", arguments={})
    selected = max(
        candidates,
        key=lambda cell: (
            cell.overload_milli_mbps,
            cell.candidate_fairness_gain_ppm,
            -cell.cell_id,
        ),
    )
    return ToolCall(
        name="set_prb_cap",
        arguments={
            "cell_id": selected.cell_id,
            "target": "ue",
            "target_id": selected.candidate_ue_id,
            "max_prb": selected.candidate_max_prb,
        },
    )


def t2_expert_action(
    obs: Observation,
    *,
    capacity_milli_mbps: int = T2_CELL_CAPACITY_MILLI_MBPS,
) -> ToolCall:
    """Convenience wrapper used by the public expert policy."""

    return select_t2_expert_action(
        build_t2_decision_features(
            obs,
            capacity_milli_mbps=capacity_milli_mbps,
        )
    )


_UNSIGNED_INT_RE = re.compile(r"(?:0|[1-9][0-9]*)\Z")
_SOURCE_RE = re.compile(r"[A-Za-z0-9_.-]+\Z")


def _format_scaled(value: int, decimal_places: int) -> str:
    scale = 10**decimal_places
    sign = "-" if value < 0 else ""
    magnitude = abs(value)
    if decimal_places == 0:
        return f"{sign}{magnitude}"
    return f"{sign}{magnitude // scale}.{magnitude % scale:0{decimal_places}d}"


def _parse_unsigned_int(
    token: str,
    *,
    label: str,
    minimum: int = 0,
    maximum: int | None = None,
) -> int:
    if _UNSIGNED_INT_RE.fullmatch(token) is None:
        raise ValueError(f"{label} is not a canonical unsigned integer: {token!r}")
    value = int(token)
    if value < minimum or (maximum is not None and value > maximum):
        bounds = f"[{minimum},{maximum}]" if maximum is not None else f">={minimum}"
        raise ValueError(f"{label}={value} is outside {bounds}")
    return value


def _parse_fixed(
    token: str,
    *,
    label: str,
    decimal_places: int,
    minimum_scaled: int | None = None,
    maximum_scaled: int | None = None,
) -> int:
    sign = r"-?" if minimum_scaled is None or minimum_scaled < 0 else ""
    pattern = rf"{sign}(?:0|[1-9][0-9]*)\.[0-9]{{{decimal_places}}}\Z"
    if re.fullmatch(pattern, token) is None:
        raise ValueError(f"{label} is not canonical fixed-point({decimal_places}): {token!r}")
    try:
        scaled = int(
            (Decimal(token) * (10**decimal_places)).quantize(
                _ONE,
                rounding=ROUND_HALF_EVEN,
            )
        )
    except (InvalidOperation, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} cannot be parsed safely: {token!r}") from exc
    negative_zero = (
        scaled == 0 and minimum_scaled is not None and minimum_scaled < 0 and token == f"-0.{'0' * decimal_places}"
    )
    if _format_scaled(scaled, decimal_places) != token and not negative_zero:
        raise ValueError(f"{label} has a non-canonical representation: {token!r}")
    if minimum_scaled is not None and scaled < minimum_scaled:
        raise ValueError(f"{label}={token} is below its minimum")
    if maximum_scaled is not None and scaled > maximum_scaled:
        raise ValueError(f"{label}={token} is above its maximum")
    return scaled


def _parse_t_row(line: str) -> tuple[int, int, str]:
    fields = line.split("|")
    if len(fields) != 5 or fields[0] != "T":
        raise ValueError("canonical policy text must start with one five-field T row")
    t_deci = _parse_fixed(
        fields[1],
        label="T.time",
        decimal_places=1,
        minimum_scaled=0,
    )
    step_idx = _parse_unsigned_int(fields[2], label="T.step")
    if fields[3] != "T2":
        raise ValueError(f"T.tier must be 'T2', got {fields[3]!r}")
    if _SOURCE_RE.fullmatch(fields[4]) is None:
        raise ValueError(f"T.source is malformed: {fields[4]!r}")
    return t_deci, step_idx, fields[4]


def _parse_c_row(line: str, *, expected_cell_id: int) -> dict:
    fields = line.split("|")
    if len(fields) != 10 or fields[0] != "C":
        raise ValueError("expected one canonical ten-field C row")
    cell_id = _parse_unsigned_int(fields[1], label="C.cell", maximum=2)
    if cell_id != expected_cell_id:
        raise ValueError(
            "C rows must be unique, contiguous, and ordered from cell 0: "
            f"expected={expected_cell_id}, observed={cell_id}"
        )
    return {
        "cell_id": cell_id,
        "prb_util_dl_p50": _parse_fixed(
            fields[2],
            label=f"C[{cell_id}].prb50",
            decimal_places=3,
            minimum_scaled=0,
            maximum_scaled=1_000,
        )
        / 1_000,
        "prb_util_dl_p99": _parse_fixed(
            fields[3],
            label=f"C[{cell_id}].prb99",
            decimal_places=3,
            minimum_scaled=0,
            maximum_scaled=1_000,
        )
        / 1_000,
        "prb_util_ul_p50": _parse_fixed(
            fields[4],
            label=f"C[{cell_id}].ul_prb",
            decimal_places=3,
            minimum_scaled=0,
            maximum_scaled=1_000,
        )
        / 1_000,
        "sched_latency_ms_p99": _parse_fixed(
            fields[5],
            label=f"C[{cell_id}].latency",
            decimal_places=1,
            minimum_scaled=0,
        )
        / 10,
        "fairness_jain": _parse_fixed(
            fields[6],
            label=f"C[{cell_id}].jain",
            decimal_places=3,
            minimum_scaled=0,
            maximum_scaled=1_000,
        )
        / 1_000,
        "prach_collision_rate": _parse_fixed(
            fields[7],
            label=f"C[{cell_id}].prach",
            decimal_places=3,
            minimum_scaled=0,
            maximum_scaled=1_000,
        )
        / 1_000,
        "rrc_connected_ues": _parse_unsigned_int(
            fields[8],
            label=f"C[{cell_id}].rrc_ues",
            maximum=24,
        ),
        "sla_violations_last_window": _parse_unsigned_int(
            fields[9],
            label=f"C[{cell_id}].sla",
        ),
        "ues": [],
    }


def _parse_u_row(
    line: str,
    *,
    expected_cell_id: int,
) -> dict:
    fields = line.split("|")
    if len(fields) != 13 or fields[0] != "U":
        raise ValueError("expected one canonical thirteen-field U row")
    location = fields[1].split("/")
    if len(location) != 2:
        raise ValueError(f"U location is malformed: {fields[1]!r}")
    cell_id = _parse_unsigned_int(location[0], label="U.cell", maximum=2)
    ue_id = _parse_unsigned_int(
        location[1],
        label="U.ue",
        maximum=MAX_UES - 1,
    )
    if cell_id != expected_cell_id:
        raise ValueError(
            f"U rows must remain nested under their C row: expected_cell={expected_cell_id}, observed={cell_id}"
        )
    qos_5qi = _parse_unsigned_int(
        fields[2],
        label=f"U[{cell_id}/{ue_id}].5qi",
        minimum=1,
        maximum=127,
    )
    requested = _parse_fixed(
        fields[3],
        label=f"U[{cell_id}/{ue_id}].requested",
        decimal_places=3,
        minimum_scaled=0,
    )
    admitted = _parse_fixed(
        fields[4],
        label=f"U[{cell_id}/{ue_id}].admitted",
        decimal_places=3,
        minimum_scaled=0,
    )
    delivered = _parse_fixed(
        fields[5],
        label=f"U[{cell_id}/{ue_id}].delivered",
        decimal_places=3,
        minimum_scaled=0,
    )
    cap_flag = fields[6]
    if cap_flag not in {"on", "off"}:
        raise ValueError(f"U[{cell_id}/{ue_id}].cap must be 'on' or 'off'")
    max_prb = _parse_unsigned_int(
        fields[7],
        label=f"U[{cell_id}/{ue_id}].max_prb",
        minimum=T2_MIN_PRB,
        maximum=T2_MAX_PRB,
    )
    if (cap_flag == "on") != (max_prb < T2_MAX_PRB):
        raise ValueError(f"U[{cell_id}/{ue_id}] cap flag/max_prb are inconsistent: {cap_flag}|{max_prb}")
    sinr_centi = _parse_fixed(
        fields[8],
        label=f"U[{cell_id}/{ue_id}].sinr",
        decimal_places=2,
        minimum_scaled=-2_000,
        maximum_scaled=4_000,
    )
    bler_milli = _parse_fixed(
        fields[9],
        label=f"U[{cell_id}/{ue_id}].bler",
        decimal_places=3,
        minimum_scaled=0,
        maximum_scaled=1_000,
    )
    mcs_deci = _parse_fixed(
        fields[10],
        label=f"U[{cell_id}/{ue_id}].mcs",
        decimal_places=1,
        minimum_scaled=0,
        maximum_scaled=270,
    )
    buffer_deci = _parse_fixed(
        fields[11],
        label=f"U[{cell_id}/{ue_id}].buffer",
        decimal_places=1,
        minimum_scaled=0,
    )
    pdb = _parse_unsigned_int(
        fields[12],
        label=f"U[{cell_id}/{ue_id}].pdb",
    )
    return {
        "ue_id": ue_id,
        "offered_mbps": admitted / 1_000,
        "requested_mbps": requested / 1_000,
        "admitted_mbps": admitted / 1_000,
        "prb_cap_max_prb": max_prb,
        "delivered_mbps": delivered / 1_000,
        "sinr_db": sinr_centi / 100,
        "bler": bler_milli / 1_000,
        "mcs_mean": mcs_deci / 10,
        "buffer_occupancy_kb": buffer_deci / 10,
        "pdb_violations": pdb,
        "5qi": qos_5qi,
    }


def derive_t2_policy_contract_from_text(
    user_text: str,
    *,
    capacity_milli_mbps: int = T2_CELL_CAPACITY_MILLI_MBPS,
) -> T2PolicyTextDerivation:
    """Strictly rederive P/D rows and the expert label from T/C/U rows only.

    Supplied P/D rows are required and compared byte-for-byte with the
    independent derivation, but none of their values are used to reconstruct
    the observation.  Optional L history and the assistant label are likewise
    never policy inputs here.
    """

    if not isinstance(user_text, str) or not user_text:
        raise ValueError("T2 policy text must be a non-empty string")
    lines = user_text.splitlines()
    if not lines or any(not line for line in lines):
        raise ValueError("T2 policy text contains a missing or blank row")
    if "\n".join(lines) != user_text:
        raise ValueError("T2 policy text has non-canonical line endings")

    t_deci, step_idx, source = _parse_t_row(lines[0])
    if len(lines) < 4 or not lines[1].startswith("P|"):
        raise ValueError("canonical T2 policy text requires one P row after T")
    supplied_policy_row = lines[1]
    supplied_cell_rows: list[str] = []
    cells: list[dict] = []
    index = 2
    while index < len(lines) and lines[index].startswith("C|"):
        cell = _parse_c_row(lines[index], expected_cell_id=len(cells))
        index += 1
        while index < len(lines) and lines[index].startswith("U|"):
            cell["ues"].append(
                _parse_u_row(
                    lines[index],
                    expected_cell_id=cell["cell_id"],
                )
            )
            index += 1
        if not cell["ues"]:
            raise ValueError(f"C[{cell['cell_id']}] is missing its U rows")
        if index >= len(lines) or not lines[index].startswith("D|"):
            raise ValueError(f"C[{cell['cell_id']}] is missing its D row")
        supplied_cell_rows.append(lines[index])
        cells.append(cell)
        index += 1
    if not cells:
        raise ValueError("canonical T2 policy text contains no C/U rows")

    history: T2PolicyHistory | None = None
    if index < len(lines) and lines[index].startswith("L|"):
        history_fields = lines[index].split("|")
        if len(history_fields) != 4:
            raise ValueError("optional L row must contain exactly four fields")
        history_name = history_fields[1]
        if history_name not in {"noop", "set_prb_cap"}:
            raise ValueError(f"L.tool is outside the T2 action space: {history_name!r}")
        try:
            history_arguments = json.loads(history_fields[2])
        except json.JSONDecodeError as exc:
            raise ValueError("L.arguments must be canonical JSON") from exc
        if (
            not isinstance(history_arguments, dict)
            or json.dumps(
                history_arguments,
                sort_keys=True,
                separators=(",", ":"),
            )
            != history_fields[2]
        ):
            raise ValueError("L.arguments must be a canonical JSON object")
        if history_name == "noop":
            if history_arguments:
                raise ValueError("L noop must have empty arguments")
        elif (
            set(history_arguments) != {"cell_id", "target", "target_id", "max_prb"}
            or history_arguments.get("target") != "ue"
            or not isinstance(history_arguments.get("cell_id"), int)
            or isinstance(history_arguments.get("cell_id"), bool)
            or not 0 <= history_arguments["cell_id"] <= 2
            or not isinstance(history_arguments.get("target_id"), int)
            or isinstance(history_arguments.get("target_id"), bool)
            or not 0 <= history_arguments["target_id"] < MAX_UES
            or not isinstance(history_arguments.get("max_prb"), int)
            or isinstance(history_arguments.get("max_prb"), bool)
            or not T2_MIN_PRB <= history_arguments["max_prb"] <= T2_MAX_PRB
        ):
            raise ValueError("L set_prb_cap arguments are outside the T2 contract")
        history_rejection = history_fields[3]
        if not history_rejection:
            raise ValueError("L.rejection must be a non-empty token")
        history = T2PolicyHistory(
            action=ToolCall(name=history_name, arguments=history_arguments),
            rejection=history_rejection,
        )
        index += 1
    if index >= len(lines) or lines[index] != "A|one_tool_call_or_noop":
        raise ValueError("canonical T2 policy text must end with the exact A row")
    index += 1
    if index != len(lines):
        raise ValueError(f"unexpected or duplicate row after A: {lines[index]!r}")

    try:
        observation = Observation.model_validate(
            {
                "t_s": t_deci / 10,
                "episode_id": "t2_policy_text_derivation",
                "cells": cells,
                "global": {
                    "n_cells": len(cells),
                    "n_ues_total": sum(len(cell["ues"]) for cell in cells),
                    "difficulty": 0.0,
                    "regime_mix": {},
                    "tier": "T2",
                },
                "kpi_source_mode": source,
                "agent_aux": {"step_idx": step_idx},
            }
        )
    except ValueError as exc:
        raise ValueError(f"canonical T/C/U rows do not form an Observation: {exc}") from exc

    features = build_t2_decision_features(
        observation,
        capacity_milli_mbps=capacity_milli_mbps,
    )
    expected_policy_row = format_t2_policy_row(features)
    expected_cell_rows = tuple(format_t2_cell_row(cell) for cell in features.cells)
    if supplied_policy_row != expected_policy_row:
        raise ValueError(
            "supplied P row differs from its T/C/U derivation: "
            f"supplied={supplied_policy_row!r}, expected={expected_policy_row!r}"
        )
    if tuple(supplied_cell_rows) != expected_cell_rows:
        raise ValueError(
            "supplied D rows differ from their T/C/U derivation: "
            f"supplied={tuple(supplied_cell_rows)!r}, "
            f"expected={expected_cell_rows!r}"
        )
    return T2PolicyTextDerivation(
        observation=observation,
        features=features,
        policy_row=expected_policy_row,
        cell_rows=expected_cell_rows,
        history=history,
        action=select_t2_expert_action(features),
    )


def derive_t2_expert_action_from_text(
    user_text: str,
    *,
    capacity_milli_mbps: int = T2_CELL_CAPACITY_MILLI_MBPS,
) -> ToolCall:
    """Return the exact expert label independently derived from canonical text."""

    return derive_t2_policy_contract_from_text(
        user_text,
        capacity_milli_mbps=capacity_milli_mbps,
    ).action


__all__ = [
    "T2_CELL_CAPACITY_MILLI_MBPS",
    "T2_EXPERT_CADENCE_STEPS",
    "T2_MAX_PRB",
    "T2_MIN_PRB",
    "T2CellDecisionFeatures",
    "T2DecisionFeatures",
    "T2PolicyTextDerivation",
    "T2PolicyHistory",
    "build_t2_decision_features",
    "derive_t2_expert_action_from_text",
    "derive_t2_policy_contract_from_text",
    "format_deci_kb",
    "format_t2_cell_row",
    "format_t2_policy_row",
    "format_milli_mbps",
    "quantize_milli_mbps",
    "quantize_deci_kb",
    "select_t2_expert_action",
    "t2_expert_action",
]
