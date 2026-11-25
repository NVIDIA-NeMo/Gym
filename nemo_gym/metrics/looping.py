# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Count-pooled looping metrics with explicit assessment coverage."""

from collections import Counter
from dataclasses import dataclass, field

from nemo_gym.trajectory.looping import LoopingAssessment


FLAGS = (
    "severe_text",
    "severe_tool_cycle",
    "severe_union",
    "warning_text",
    "warning_tool_cycle",
    "warning_union",
)


@dataclass
class LoopingCounts:
    """Accumulate assessments for one explicitly selected population."""

    counts: Counter[str] = field(default_factory=Counter)

    def observe(self, assessment: LoopingAssessment, *, valid: bool = True) -> None:
        """Add one rollout, retaining invalid rows in coverage only."""
        self.counts["selected_total"] += 1
        if not valid:
            self.counts["invalid_total"] += 1
            return
        for flag_name in FLAGS:
            flag_value = getattr(assessment, flag_name)
            is_assessed = type(flag_value) is bool
            self.counts[f"{flag_name}/assessed_total"] += int(is_assessed)
            self.counts[f"{flag_name}/flagged_total"] += int(flag_value is True)
        for gap in assessment.gaps:
            self.counts[f"gaps/{gap}_total"] += 1

    def metrics(self, *, expected: int | None = None) -> dict[str, float | None]:
        """Return sufficient counts and rates; empty assessment rates remain absent."""
        selected_total = self.counts["selected_total"]
        if expected is not None and expected < selected_total:
            raise ValueError("expected looping population is smaller than selected records")
        if expected is None:
            population_total = selected_total
        else:
            population_total = expected
        result = {f"looping/{key}": float(value) for key, value in self.counts.items()}
        result["looping/selected_total"] = float(selected_total)
        result["looping/expected_total"] = float(population_total) if expected is not None else None
        for flag_name in FLAGS:
            assessed_total = self.counts[f"{flag_name}/assessed_total"]
            flagged_total = self.counts[f"{flag_name}/flagged_total"]
            result[f"looping/{flag_name}/assessed_total"] = float(assessed_total)
            result[f"looping/{flag_name}/flagged_total"] = float(flagged_total)
            result[f"looping/{flag_name}/unknown_total"] = float(population_total - assessed_total)
            if assessed_total:
                result[f"looping/{flag_name}/rate_assessed"] = flagged_total / assessed_total
            else:
                result[f"looping/{flag_name}/rate_assessed"] = None
        return result
