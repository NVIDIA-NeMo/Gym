# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Deterministic image-history policies for Gym-owned OSWorld agents.

History selection and model-message rendering change on different schedules.
This module owns only the selection decision.  The model adapter remains the
authority for prompts, roles, templates, and response parsing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Literal, Mapping

from responses_api_agents.osworld_agent.trajectory import stable_id


HistoryPolicyName = Literal["fixed", "hysteresis", "sink_window"]
HistoryDisposition = Literal["text", "live_image", "drop"]

IntervalTuple = tuple[tuple[int, int], ...]


def _merge_intervals(intervals: Iterable[tuple[int, int]]) -> IntervalTuple:
    """Collapse touching/overlapping half-open ranges into ascending disjoint ones."""

    merged: list[list[int]] = []
    for low, high in sorted(item for item in intervals if item[0] < item[1]):
        if merged and low <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], high)
        else:
            merged.append([low, high])
    return tuple((low, high) for low, high in merged)


def _count_intervals(intervals: IntervalTuple) -> int:
    return sum(high - low for low, high in intervals)


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
    sink: int = 0

    def __post_init__(self) -> None:
        if self.name not in {"fixed", "hysteresis", "sink_window"}:
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
        if self.name == "sink_window":
            _positive_int(self.sink, field="history_policy.params.sink")
            if self.low_water <= self.sink:
                raise ValueError(
                    "sink_window low_water must exceed sink; a compacted prompt still needs one recent image"
                )
        elif self.sink != 0:
            raise ValueError(f"{self.name} history policy does not accept a sink")

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
    def sink_window(cls, *, sink: int, low_water: int, high_water: int) -> HistoryPolicySpec:
        return cls(
            name="sink_window",
            low_water=_positive_int(low_water, field="history_policy.params.low_water"),
            high_water=_positive_int(high_water, field="history_policy.params.high_water"),
            sink=_positive_int(sink, field="history_policy.params.sink"),
        )

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
        elif name == "sink_window":
            expected = {"sink", "low_water", "high_water"}
            extras = sorted(set(params) - expected)
            missing = sorted(expected - set(params))
            if extras or missing:
                raise ValueError(
                    "sink_window history_policy.params must contain only sink, low_water and high_water"
                    f" (missing={missing}, extra={extras})"
                )
            spec = cls.sink_window(
                sink=params["sink"],
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
        elif self.name == "hysteresis":
            params = {"low_water": self.low_water, "high_water": self.high_water}
        else:
            params = {"sink": self.sink, "low_water": self.low_water, "high_water": self.high_water}
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
            #
            # The property belongs to the *recent* window, not to the policy
            # name: only a hysteretic window appends between compactions. A
            # sliding window rewrites the interval on every turn, whether or
            # not it carries a sink.
            "supports_append_stable_intervals": self.high_water > self.low_water,
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
    image_intervals: IntervalTuple
    compaction_triggered: bool
    compaction_epoch: int
    next_state: HistoryPolicyState
    image_budget_clamped: bool = False

    @property
    def image_window_start(self) -> int:
        """First live-image turn -- defined only for a contiguous image set.

        Deliberately fails closed. Any consumer still reading this scalar under
        a policy that emits a sink plus a recent window would otherwise build a
        silently wrong prompt and run a whole benchmark to a plausible-looking
        score. Crashing on the first step is the cheaper failure.
        """

        if len(self.image_intervals) != 1:
            raise ValueError(
                "image_window_start is undefined for a non-contiguous image set; "
                f"read image_intervals instead (got {self.image_intervals!r})"
            )
        return self.image_intervals[0][0]

    @property
    def recent_window_start(self) -> int:
        """Start of the trailing live-image interval; always defined."""

        return self.image_intervals[-1][0] if self.image_intervals else self.completed_turns + 1

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
            "snapshot_window_start": self.recent_window_start,
            "snapshot_image_intervals": [list(item) for item in self.image_intervals],
            "snapshot_sink_size": self.spec.sink,
            "snapshot_image_budget_clamped": self.image_budget_clamped,
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
    max_images: int | None = None,
) -> HistoryPlan:
    """Select text/live-image turns without rendering model messages.

    Turn indices cover completed turns ``[0, completed_turns)`` plus the
    current observation at ``completed_turns``.  Every current strategy folds
    older turns into text; no turn is dropped.

    ``max_images`` is an optional hard clamp on how many live images the plan
    may emit.  It exists so a caller that can measure the rendered prompt can
    re-plan against a token budget without this module having to know anything
    about tokenizers.  The function stays pure and replayable: the same inputs
    always produce the same plan.  Clamping only ever shrinks the trailing
    window; the sink is preserved because dropping it would change what the
    policy means rather than just how much of it fits.
    """

    if isinstance(completed_turns, bool) or not isinstance(completed_turns, int) or completed_turns < 0:
        raise ValueError("completed_turns must be a non-negative integer")
    if max_images is not None:
        if isinstance(max_images, bool) or not isinstance(max_images, int) or max_images < 1:
            raise ValueError("max_images must be a positive integer when provided")
    total_turns = completed_turns + 1
    previous_start = max(0, min(state.image_window_start, completed_turns))

    if spec.name == "fixed":
        image_window_start = max(0, total_turns - spec.low_water)
        compacted = image_window_start != previous_start
        intervals = ((image_window_start, total_turns),)
    elif spec.name == "hysteresis":
        image_window_start = previous_start
        compacted = total_turns - image_window_start > spec.high_water
        if compacted:
            image_window_start = max(0, total_turns - spec.low_water)
        intervals = ((image_window_start, total_turns),)
    else:
        sink = min(spec.sink, total_turns)
        window_start = max(previous_start, sink)
        live = _merge_intervals(((0, sink), (window_start, total_turns)))
        # low/high count total live images including the sink, matching what
        # keep_images means for `fixed` and low/high mean for `hysteresis`.
        compacted = _count_intervals(live) > spec.high_water
        if compacted:
            window_start = max(sink, total_turns - (spec.low_water - sink))
        image_window_start = window_start
        intervals = _merge_intervals(((0, sink), (window_start, total_turns)))

    clamped = False
    if max_images is not None and _count_intervals(intervals) > max_images:
        clamped = True
        sink = min(spec.sink, total_turns) if spec.name == "sink_window" else 0
        # Keep at least one recent image; the current observation is the one
        # turn the model cannot act without.
        recent = max(1, max_images - sink)
        window_start = max(sink, total_turns - recent)
        image_window_start = window_start
        intervals = _merge_intervals(((0, sink), (window_start, total_turns)))
        compacted = True

    epoch = state.compaction_epoch + int(compacted)
    decisions = tuple(
        HistoryTurnDecision(
            turn_index=index,
            disposition=(
                "live_image" if any(low <= index < high for low, high in intervals) else "text"
            ),
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
        image_intervals=intervals,
        compaction_triggered=compacted,
        compaction_epoch=epoch,
        next_state=next_state,
        image_budget_clamped=clamped,
    )
