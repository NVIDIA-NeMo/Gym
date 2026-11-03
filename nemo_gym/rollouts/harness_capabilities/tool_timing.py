# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Independent timing capability checks for a declared harness runtime and mode."""

from dataclasses import dataclass

from pydantic import ValidationError

from nemo_gym.core.operation_timing import OperationTiming

from .checks import SemanticCheck
from .results import Results


@dataclass(frozen=True)
class ToolTimingScope:
    """Exact runtime/mode/tool paths and authoritative sources exercised by a probe.

    An empty source list means this boundary is not claimed by this probe. It
    does not convert missing measurements into successful capability evidence.
    """

    runtime_version: str
    mode: str
    tool_paths: tuple[str, ...]
    lifecycle_sources: tuple[str, ...] = ()
    executor_sources: tuple[str, ...] = ()
    process_attribution: bool = False

    def __post_init__(self) -> None:
        if not self.runtime_version or not self.mode or not self.tool_paths or not all(self.tool_paths):
            raise ValueError("Timing scope requires a pinned runtime, mode and tested tool paths")


def check_tool_timing(tools: list[dict], scope: ToolTimingScope, results: Results, *, available: bool) -> None:
    """Keep lifecycle, executor and process claims independent in existing results."""
    observed = [tool for tool in tools if tool.get("tool_name") in scope.tool_paths]
    for boundary, sources in (("lifecycle", scope.lifecycle_sources), ("executor", scope.executor_sources)):
        valid = set(scope.tool_paths).issubset(tool.get("tool_name") for tool in observed)
        for tool in observed:
            measurements = (
                [
                    value
                    for value in tool.get("timings", [])
                    if isinstance(value, dict) and value.get("boundary") == boundary
                ]
                if isinstance(tool.get("timings"), list)
                else []
            )
            if len(measurements) != 1:
                valid = False
                continue
            try:
                measurement = OperationTiming.model_validate(measurements[0])
            except ValidationError:
                valid = False
                continue
            if measurement.timing_source not in sources:
                valid = False
            if measurement.duration_ms is None:
                valid = False
            if measurement.duration_source is None:
                valid = False
        results.run(
            SemanticCheck(
                id=f"tools.timing.{boundary}",
                tier="P0",
                evidence=("TE-5",),
                location="$.ng_trajectory.tool_calls",
                depends_on=(),
                applies=bool(sources),
                available=available,
                reason=f"expected {boundary} measurements from the declared source for every tested tool attempt",
                predicate=lambda valid=valid: valid,
            )
        )
    # Timing alone cannot prove which OS process or hardware performed the work.
    results.run(
        SemanticCheck(
            id="tools.process_attribution",
            tier="P0",
            evidence=("TE-5",),
            location="$.ng_trajectory.tool_calls",
            depends_on=(),
            applies=scope.process_attribution,
            available=False,
            reason="process attribution is not demonstrated by timing records",
            predicate=lambda: False,
        )
    )
