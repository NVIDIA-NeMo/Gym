# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Deterministic image-history policies for Gym-owned OSWorld agents.

History selection and model-message rendering change on different schedules.
This module owns only the selection decision.  The model adapter remains the
authority for prompts, roles, templates, and response parsing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping

from responses_api_agents.osworld_agent.trajectory import stable_id


HistoryPolicyName = Literal["fixed", "hysteresis"]
HistoryDisposition = Literal["text", "live_image", "drop"]


def _positive_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{field} must be a positive integer")
    return value


@dataclass(frozen=True)
class HistoryPolicySpec:
    """Canonical, semantic configuration for one history policy."""

    name: HistoryPolicyName
    low_water: int
    high_water: int
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.name not in {"fixed", "hysteresis"}:
            raise ValueError(f"Unsupported history policy name: {self.name!r}")
        if self.schema_version != 1:
            raise ValueError(f"Unsupported history policy schema_version: {self.schema_version}")
        _positive_int(self.low_water, field="history_policy.low_water")
        _positive_int(self.high_water, field="history_policy.high_water")
        if self.high_water < self.low_water:
            raise ValueError("history_policy.high_water must be greater than or equal to low_water")
        if self.name == "fixed" and self.high_water != self.low_water:
            raise ValueError("fixed history policy requires one equal low/high window")
        if self.name == "hysteresis" and self.high_water == self.low_water:
            raise ValueError("hysteresis high_water must exceed low_water; use fixed for an equal window")

    @classmethod
    def fixed(cls, keep_images: int) -> HistoryPolicySpec:
        keep = _positive_int(keep_images, field="history_policy.params.keep_images")
        return cls(name="fixed", low_water=keep, high_water=keep)

    @classmethod
    def hysteresis(cls, *, low_water: int, high_water: int) -> HistoryPolicySpec:
        low = _positive_int(low_water, field="history_policy.params.low_water")
        high = _positive_int(high_water, field="history_policy.params.high_water")
        return cls(name="hysteresis", low_water=low, high_water=high)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> HistoryPolicySpec:
        if not isinstance(value, Mapping):
            raise TypeError("history_policy must be a mapping")
        allowed = {"schema_version", "name", "params"}
        extras = sorted(set(value) - allowed)
        if extras:
            raise ValueError("Unsupported history_policy fields: " + ", ".join(extras))
        schema_version = value.get("schema_version", 1)
        if isinstance(schema_version, bool) or not isinstance(schema_version, int):
            raise ValueError("history_policy.schema_version must be an integer")
        name = value.get("name")
        params = value.get("params")
        if not isinstance(params, Mapping):
            raise ValueError("history_policy.params must be a mapping")
        if name == "fixed":
            expected = {"keep_images"}
            extras = sorted(set(params) - expected)
            missing = sorted(expected - set(params))
            if extras or missing:
                raise ValueError(
                    f"fixed history_policy.params must contain only keep_images (missing={missing}, extra={extras})"
                )
            spec = cls.fixed(params["keep_images"])
        elif name == "hysteresis":
            expected = {"low_water", "high_water"}
            extras = sorted(set(params) - expected)
            missing = sorted(expected - set(params))
            if extras or missing:
                raise ValueError(
                    "hysteresis history_policy.params must contain only low_water and high_water"
                    f" (missing={missing}, extra={extras})"
                )
            spec = cls.hysteresis(
                low_water=params["low_water"],
                high_water=params["high_water"],
            )
        else:
            raise ValueError(f"Unsupported history_policy.name: {name!r}")
        if schema_version != spec.schema_version:
            raise ValueError(f"Unsupported history policy schema_version: {schema_version}")
        return spec

    @classmethod
    def from_legacy(
        cls,
        *,
        keep_images: int,
        max_live_images: int | None = None,
    ) -> HistoryPolicySpec:
        low = _positive_int(keep_images, field="max_image_history_length")
        if max_live_images is None:
            return cls.fixed(low)
        high = _positive_int(max_live_images, field="max_live_images")
        if high < low:
            raise ValueError("max_live_images must be greater than or equal to max_image_history_length")
        if high == low:
            return cls.fixed(low)
        return cls.hysteresis(low_water=low, high_water=high)

    def to_config(self) -> dict[str, Any]:
        params: dict[str, int]
        if self.name == "fixed":
            params = {"keep_images": self.low_water}
        else:
            params = {"low_water": self.low_water, "high_water": self.high_water}
        return {
            "schema_version": self.schema_version,
            "name": self.name,
            "params": params,
        }

    @property
    def policy_id(self) -> str:
        return stable_id("osworld-history-policy", self.to_config())

    def to_contract(self) -> dict[str, Any]:
        return {
            **self.to_config(),
            "history_policy_id": self.policy_id,
            # This is a structural expectation only. Exact-trace authority
            # still measures token/media append compatibility at runtime.
            "supports_append_stable_intervals": self.name == "hysteresis",
        }


@dataclass(frozen=True)
class HistoryPolicyState:
    """Explicit per-trajectory state; callers own reset and replay."""

    image_window_start: int = 0
    compaction_epoch: int = 0

    def __post_init__(self) -> None:
        for field, value in (
            ("image_window_start", self.image_window_start),
            ("compaction_epoch", self.compaction_epoch),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"history policy state {field} must be a non-negative integer")


@dataclass(frozen=True)
class HistoryTurnDecision:
    turn_index: int
    disposition: HistoryDisposition


@dataclass(frozen=True)
class HistoryPlan:
    """One deterministic selection decision including the current turn."""

    spec: HistoryPolicySpec
    completed_turns: int
    decisions: tuple[HistoryTurnDecision, ...]
    image_window_start: int
    compaction_triggered: bool
    compaction_epoch: int
    next_state: HistoryPolicyState

    @property
    def image_turns(self) -> tuple[int, ...]:
        return tuple(item.turn_index for item in self.decisions if item.disposition == "live_image")

    @property
    def text_turns(self) -> tuple[int, ...]:
        return tuple(item.turn_index for item in self.decisions if item.disposition == "text")

    @property
    def dropped_turns(self) -> tuple[int, ...]:
        return tuple(item.turn_index for item in self.decisions if item.disposition == "drop")

    def telemetry(self) -> dict[str, Any]:
        return {
            "prompt_snapshot_count": len(self.image_turns),
            "snapshot_window_start": self.image_window_start,
            "snapshot_compaction_triggered": self.compaction_triggered,
            "snapshot_window_min": self.spec.low_water,
            "snapshot_window_max": self.spec.high_water,
            "history_policy_id": self.spec.policy_id,
            "history_policy_name": self.spec.name,
            "history_policy_schema_version": self.spec.schema_version,
            "history_policy_compaction_epoch": self.compaction_epoch,
            "history_selection_append_expected": self.completed_turns > 0 and not self.compaction_triggered,
        }


def plan_history(
    spec: HistoryPolicySpec,
    state: HistoryPolicyState,
    *,
    completed_turns: int,
) -> HistoryPlan:
    """Select text/live-image turns without rendering model messages.

    Turn indices cover completed turns ``[0, completed_turns)`` plus the
    current observation at ``completed_turns``.  Every current strategy folds
    older turns into text; no turn is dropped.
    """

    if isinstance(completed_turns, bool) or not isinstance(completed_turns, int) or completed_turns < 0:
        raise ValueError("completed_turns must be a non-negative integer")
    total_turns = completed_turns + 1
    previous_start = max(0, min(state.image_window_start, completed_turns))

    if spec.name == "fixed":
        image_window_start = max(0, total_turns - spec.low_water)
        compacted = image_window_start != previous_start
    else:
        image_window_start = previous_start
        compacted = total_turns - image_window_start > spec.high_water
        if compacted:
            image_window_start = max(0, total_turns - spec.low_water)

    epoch = state.compaction_epoch + int(compacted)
    decisions = tuple(
        HistoryTurnDecision(
            turn_index=index,
            disposition="text" if index < image_window_start else "live_image",
        )
        for index in range(total_turns)
    )
    next_state = HistoryPolicyState(
        image_window_start=image_window_start,
        compaction_epoch=epoch,
    )
    return HistoryPlan(
        spec=spec,
        completed_turns=completed_turns,
        decisions=decisions,
        image_window_start=image_window_start,
        compaction_triggered=compacted,
        compaction_epoch=epoch,
        next_state=next_state,
    )
